import streamlit as st
import cv2
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
import plotly.graph_objects as go
import tempfile
import os
import easyocr

# 페이지 설정
st.set_page_config(page_title="Sketch to DXF Pro", layout="wide")

# 1. 메모리 세이프 OCR 로더
@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False, download_enabled=True)
    except Exception as e:
        st.warning(f"OCR 엔진 로딩 지연 중: {e}")
        return None

def process_sketch_pro(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, enable_3d, filter_strength, user_instruction=""):
    # 이미지 로드
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    
    h, w, _ = img_bgr.shape
    
    # [수정 요청 반영: 스케일 조정]
    final_scale = real_width_mm / w if real_width_mm > 0 else 1.0
    if "배 키워" in user_instruction or "배 크게" in user_instruction:
        final_scale *= 1.2  # 기본 1.2배 확대

    # 2. 스마트 컬러 필터 (격자무늬 제거)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, filter_strength]) 
    binary = cv2.inRange(hsv, lower_dark, upper_dark)

    lower_grid = np.array([75, 20, 150]) 
    upper_grid = np.array([135, 120, 255])
    grid_mask = cv2.inRange(hsv, lower_grid, upper_grid)
    binary = cv2.subtract(binary, grid_mask)

    kernel = np.ones((2,2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 3. OCR (텍스트 인식 및 선 제외)
    reader = load_ocr_reader()
    detected_texts = []
    if reader:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray)
            for (bbox, text, prob) in ocr_results:
                if prob < 0.3: continue
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(binary, [pts], (0))
                cx = np.mean(pts[:, 0]) * final_scale
                cy = (h - np.mean(pts[:, 1])) * final_scale
                detected_texts.append({'text': text, 'x': cx, 'y': cy, 'h': (pts[2][1]-pts[0][1])*final_scale})
        except: pass

    # 4. 벡터화 및 DXF 생성
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    doc.layers.add("WALL_2D", color=7)
    if enable_3d:
        doc.layers.add("VERT_COL", color=2)
        doc.layers.add("CEIL_LINE", color=3)

    plot_x, plot_y, plot_z = [], [], []
    v_columns = set()

    # [수정 요청 반영: 직각 보정 모드 체크]
    ortho_mode = any(word in user_instruction for word in ["직각", "수직", "반듯하게", "똑바로"])

    def get_snap(pt):
        if snap_size == 0: return pt
        return (round(pt[0]/snap_size)*snap_size, round(pt[1]/snap_size)*snap_size)

    for cnt in contours:
        if cv2.contourArea(cnt) < 40: continue 
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_adj * peri, True)
        pts = [get_snap((p[0][0]*final_scale, (h-p[0][1])*final_scale)) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                
                if ortho_mode:
                    dx = abs(p1[0] - p2[0])
                    dy = abs(p1[1] - p2[1])
                    if dx > dy: p2 = (p2[0], p1[1]) # 가로선으로 보정
                    else: p2 = (p1[0], p2[1])       # 세로선으로 보정

                if p1 == p2: continue
                
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0), dxfattribs={'layer': 'WALL_2D'})
                
                if enable_3d:
                    for p in [p1, p2]:
                        if p not in v_columns:
                            msp.add_line((p[0], p[1], 0), (p[0], p[1], wall_height_mm), dxfattribs={'layer': 'VERT_COL'})
                            v_columns.add(p)
                    msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm), dxfattribs={'layer': 'CEIL_LINE'})
                    
                    plot_x.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                    plot_y.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                    plot_z.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])
                else:
                    plot_x.extend([p1[0], p2[0], None])
                    plot_y.extend([p1[1], p2[1], None])
                    plot_z.extend([0, 0, None])

    for dt in detected_texts:
        t = msp.add_text(dt['text'], dxfattribs={'height': dt['h']*0.8, 'color': 1})
        t.set_placement((dt['x'], dt['y'], 0), align=TextEntityAlignment.MIDDLE_CENTER)

    return doc, plot_x, plot_y, plot_z

# --- Streamlit UI ---
st.title("📐 Professional Sketch to DXF")

with st.sidebar:
    st.header("1. 기본 설정")
    enable_3d = st.checkbox("🏗️ 3D 벽체 세우기", value=True)
    filter_val = st.slider("🔍 인식 민감도", 50, 255, 160)
    
    st.divider()
    st.header("2. 실제 치수 (mm)")
    real_w = st.number_input("도면 실제 가로 폭", value=10000)
    wall_h = st.number_input("벽 높이", value=2400, disabled=not enable_3d)
    
    st.divider()
    st.header("3. AI 수정 요청")
    user_comment = st.text_input("수정 사항 입력:", placeholder="예: '직각으로 펴줘'")
    
    st.divider()
    st.header("4. 벡터화 옵션")
    eps = st.slider("직선화 강도", 0.001, 0.050, 0.015, format="%.3f")
    snap = st.selectbox("그리드 스냅 (mm)", [0, 1, 5, 10, 50], index=2)

uploaded = st.file_uploader("이미지 파일 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    bytes_data = uploaded.read()
    col1, col2 = st.columns(2)
    
    # st.image 에러 방지를 위한 호환성 처리
    try:
        col1.image(bytes_data, caption="원본 이미지", use_container_width=True)
    except TypeError:
        col1.image(bytes_data, caption="원본 이미지", use_column_width=True)

    with st.spinner("AI 분석 및 수정 반영 중..."):
        res = process_sketch_pro(bytes_data, real_w, wall_h, snap, eps, enable_3d, filter_val, user_comment)
        
        if res:
            doc, px, py, pz = res
            fig = go.Figure(go.Scatter3d(x=px, y=py, z=pz, mode='lines', 
                                         line=dict(color='#00ffcc' if enable_3d else '#ffffff', width=2)))
            fig.update_layout(scene=dict(aspectmode='data', bgcolor='black'), 
                              paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
            
            col2.plotly_chart(fig, use_container_width=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                doc.saveas(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button("📥 DXF 다운로드", f, "pro_plan_final.dxf", use_container_width=True)
            os.unlink(tmp.name)