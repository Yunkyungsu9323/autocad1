import streamlit as st
import cv2
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
import plotly.graph_objects as go
import tempfile
import os
import easyocr

st.set_page_config(page_title="Pro Sketch Converter v2", layout="wide")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

def process_image_improved(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, enable_3d):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    h, w = img.shape
    scale = real_width_mm / w if real_width_mm > 0 else 1.0

    # 1. 전처리 (대비 향상 및 이진화)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    # 2. OCR 마스킹
    reader = load_ocr_reader()
    ocr_results = reader.readtext(img)
    for (bbox, text, prob) in ocr_results:
        if prob < 0.3: continue
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(binary, [pts], (0))

    # 3. 선 추출
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add("WALL_2D", color=7)
    if enable_3d:
        doc.layers.add("WALL_3D_VERT", color=2) # 기둥 (노랑)
        doc.layers.add("WALL_3D_CEIL", color=3) # 천장 (초록)

    plot_x, plot_y, plot_z = [], [], []
    v_columns = set()

    def get_snap(pt):
        if snap_size == 0: return pt
        return (round(pt[0]/snap_size)*snap_size, round(pt[1]/snap_size)*snap_size)

    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_adj * peri, True)
        pts = [get_snap((p[0][0]*scale, (h-p[0][1])*scale)) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0]) # 폐곡선 처리
            
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if p1 == p2: continue
                
                # --- 2D 평면선 (항상 생성) ---
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0), dxfattribs={'layer': 'WALL_2D'})
                
                if enable_3d:
                    # --- 3D 벽체 옵션 ---
                    # 1. 기둥 생성 (중복 좌표 방지)
                    for p in [p1, p2]:
                        if p not in v_columns:
                            msp.add_line((p[0], p[1], 0), (p[0], p[1], wall_height_mm), 
                                         dxfattribs={'layer': 'WALL_3D_VERT'})
                            v_columns.add(p)
                    # 2. 천장 수평선
                    msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm), 
                                 dxfattribs={'layer': 'WALL_3D_CEIL'})

                    # Plotly (3D 박스 프레임)
                    plot_x.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                    plot_y.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                    plot_z.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])
                else:
                    # Plotly (2D 평면만)
                    plot_x.extend([p1[0], p2[0], None])
                    plot_y.extend([p1[1], p2[1], None])
                    plot_z.extend([0, 0, None])

    return doc, plot_x, plot_y, plot_z

# --- UI ---
st.title("📐 Pro Sketch Converter (3D 선택 모드)")

with st.sidebar:
    st.header("🏗️ 모드 선택")
    # 여기서 3D 여부를 체크할 수 있습니다!
    enable_3d = st.checkbox("3D 벽체 세우기 (평면도용)", value=True)
    
    st.divider()
    st.header("⚙️ 치수 설정")
    real_w = st.number_input("실제 가로 폭 (mm)", value=10000)
    wall_h = st.number_input("벽 높이 (mm)", value=2400, disabled=not enable_3d)
    
    st.divider()
    st.subheader("정밀도 조정")
    eps = st.slider("직선화 강도 (높을수록 단순해짐)", 0.001, 0.050, 0.015, format="%.3f")
    snap = st.selectbox("그리드 스냅 (mm)", [0, 1, 5, 10, 50], index=2)

uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    c1, c2 = st.columns(2)
    bytes_data = uploaded.read()
    c1.image(bytes_data, caption="원본 스케치")
    
    with st.spinner("이미지 분석 및 벡터화 중..."):
        # 함수 호출 시 enable_3d 값을 전달합니다.
        doc, px, py, pz = process_image_improved(bytes_data, real_w, wall_h, snap, eps, enable_3d)
        
        if doc and px:
            # 시각화 설정
            color = '#00ffcc' if enable_3d else '#ffffff'
            fig = go.Figure(go.Scatter3d(x=px, y=py, z=pz, mode='lines', 
                                         line=dict(color=color, width=2 if enable_3d else 4)))
            
            # 2D 모드일 때는 위에서 보는 시점으로 초기화
            camera = dict(eye=dict(x=0, y=0, z=2)) if not enable_3d else None
            
            fig.update_layout(scene=dict(aspectmode='data', bgcolor='black', camera=camera), 
                              paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
            
            c2.plotly_chart(fig, use_container_width=True)
            
            # 파일 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                doc.saveas(tmp.name)
                st.download_button("📥 DXF 다운로드", open(tmp.name, "rb"), "converted_plan.dxf", use_container_width=True)
            os.unlink(tmp.name)

            if enable_3d:
                st.success(f"✅ 3D 벽체 모드: 높이 {wall_h}mm 적용 완료")
            else:
                st.info("✅ 2D 선 따기 모드: 평면 벡터 데이터만 생성")