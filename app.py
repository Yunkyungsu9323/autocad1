import streamlit as st
import cv2
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
import plotly.graph_objects as go
import tempfile
import os
import easyocr

st.set_page_config(page_title="Pro Sketch to DXF (Smart Filter)", layout="wide")

@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except:
        return None

def process_image_smart_filter(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, enable_3d, filter_strength):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    
    h, w, _ = img_bgr.shape
    scale = real_width_mm / w if real_width_mm > 0 else 1.0

    # --- [개선] 스마트 컬러 필터링 ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. 스케치 영역(어두운 부분) 추출 범위를 넓힘
    # filter_strength가 낮을수록 연한 선도 포함함
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, filter_strength]) 
    binary = cv2.inRange(hsv, lower_dark, upper_dark)

    # 2. 격자무늬 특화 제거 (연한 파란색/녹색 계열의 격자만 타겟팅하여 제거)
    # 배경색이 흰색에 가까운지 확인하여 격자만 날림
    lower_grid = np.array([80, 20, 150]) # 연한 하늘색 계열
    upper_grid = np.array([130, 100, 255])
    grid_mask = cv2.inRange(hsv, lower_grid, upper_grid)
    
    # 스케치 영역에서 격자 영역을 빼버림
    binary = cv2.subtract(binary, grid_mask)

    # 3. 선 복원 및 강화
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1) # 얇은 선 두껍게

    # 4. OCR 및 텍스트 처리
    reader = load_ocr_reader()
    detected_texts = []
    if reader:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray)
        for (bbox, text, prob) in ocr_results:
            if prob < 0.3: continue
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(binary, [pts], (0))
            cx = np.mean(pts[:, 0]) * scale
            cy = (h - np.mean(pts[:, 1])) * scale
            detected_texts.append({'text': text, 'x': cx, 'y': cy, 'h': (pts[2][1]-pts[0][1])*scale})

    # 5. 컨투어 및 DXF 생성
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    plot_x, plot_y, plot_z = [], [], []
    v_columns = set()

    def get_snap(pt):
        if snap_size == 0: return pt
        return (round(pt[0]/snap_size)*snap_size, round(pt[1]/snap_size)*snap_size)

    for cnt in contours:
        # 잡티 제거 기준을 상황에 맞게 조정 (너무 작으면 지움)
        if cv2.contourArea(cnt) < 30: continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_adj * peri, True)
        pts = [get_snap((p[0][0]*scale, (h-p[0][1])*scale)) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if p1 == p2: continue
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0), dxfattribs={'layer': 'WALL_2D'})
                
                if enable_3d:
                    for p in [p1, p2]:
                        if p not in v_columns:
                            msp.add_line((p[0], p[1], 0), (p[0], p[1], wall_height_mm), dxfattribs={'layer': 'VERT'})
                            v_columns.add(p)
                    msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm), dxfattribs={'layer': 'CEIL'})
                    
                    plot_x.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                    plot_y.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                    plot_z.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])
                else:
                    plot_x.extend([p1[0], p2[0], None]); plot_y.extend([p1[1], p2[1], None]); plot_z.extend([0, 0, None])

    return doc, plot_x, plot_y, plot_z

# --- UI ---
st.title("📐 Smart Sketch to DXF")

with st.sidebar:
    st.header("🎨 필터 미세 조정")
    # 이 값을 높이면 연한 선도 살아나지만 격자도 같이 나올 수 있습니다.
    filter_val = st.slider("인식 민감도 (연한 선 복원)", 50, 255, 180, help="중요한 선이 지워진다면 이 값을 높이세요.")
    
    st.divider()
    st.header("🏗️ 치수 및 3D")
    enable_3d = st.checkbox("3D 벽체 세우기", value=True)
    real_w = st.number_input("실제 가로 폭 (mm)", value=10000)
    wall_h = st.number_input("벽 높이 (mm)", value=2400)
    
    st.divider()
    st.header("⚙️ 정밀도")
    eps = st.slider("직선화 강도", 0.001, 0.050, 0.015, format="%.3f")
    snap = st.selectbox("그리드 스냅 (mm)", [0, 1, 5, 10, 50], index=2)

uploaded = st.file_uploader("그림 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    bytes_data = uploaded.read()
    col1, col2 = st.columns(2)
    col1.image(bytes_data, caption="원본 이미지")

    with st.spinner("중요 선 복원 중..."):
        res = process_image_smart_filter(bytes_data, real_w, wall_h, snap, eps, enable_3d, filter_val)
        
        if res:
            doc, px, py, pz = res
            fig = go.Figure(go.Scatter3d(x=px, y=py, z=pz, mode='lines', line=dict(color='#00ffcc', width=2)))
            fig.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
            col2.plotly_chart(fig, use_container_width=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                doc.saveas(tmp.name)
                st.download_button("📥 DXF 다운로드", open(tmp.name, "rb"), "smart_drawing.dxf", use_container_width=True)
            os.unlink(tmp.name)