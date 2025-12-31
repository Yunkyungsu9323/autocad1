import streamlit as st
import cv2
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
import plotly.graph_objects as go
import tempfile
import os
import easyocr
import math

# 페이지 설정
st.set_page_config(page_title="Sketch to DXF Pro", layout="wide")

# 세션 상태 초기화
if "cmd" not in st.session_state:
    st.session_state.cmd = "일반"

# 1. OCR 로더
@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except Exception:
        return None

# 2. 분석 엔진 (척도 파라미터 calib_px, calib_mm 추가)
def process_sketch_pro(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, enable_3d, filter_strength, user_cmd, calib_px=0, calib_mm=0):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    
    h, w, _ = img_bgr.shape
    
    # --- [척도 계산 로직] ---
    if calib_px > 0 and calib_mm > 0:
        scale = calib_mm / calib_px
    else:
        scale = real_width_mm / w if real_width_mm > 0 else 1.0

    # 버튼 명령에 따른 파라미터 보정
    f_val = 200 if user_cmd == "잡티 제거" else filter_strength
    s_val = 50 if user_cmd == "선 연결" else snap_size
    e_val = 0.040 if user_cmd == "직각 보정" else epsilon_adj

    # 전처리
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, f_val]))
    grid_mask = cv2.inRange(hsv, np.array([75, 20, 150]), np.array([135, 120, 255]))
    binary = cv2.subtract(binary, grid_mask)
    binary = cv2.dilate(binary, np.ones((2,2), np.uint8), iterations=1)

    # OCR
    reader = load_ocr_reader()
    detected_texts = []
    if reader:
        try:
            results = reader.readtext(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
            for (bbox, text, prob) in results:
                if prob < 0.3: continue
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(binary, [pts], (0))
                cx = np.mean(pts[:, 0]) * scale
                cy = (h - np.mean(pts[:, 1])) * scale
                detected_texts.append({'text': text, 'x': cx, 'y': cy, 'h': (pts[2][1]-pts[0][1])*scale})
        except: pass

    # 벡터화
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    px, py, pz = [], [], []

    for cnt in contours:
        if cv2.contourArea(cnt) < 40: continue
        approx = cv2.approxPolyDP(cnt, e_val * cv2.arcLength(cnt, True), True)
        pts = [(round(p[0][0]*scale/s_val)*s_val, round((h-p[0][1])*scale/s_val)*s_val) if s_val > 0 
               else (p[0][0]*scale, (h-p[0][1])*scale) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if p1 == p2: continue
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0))
                if enable_3d:
                    msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm))
                    px.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                    py.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                    pz.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])
                else:
                    px.extend([p1[0], p2[0], None]); py.extend([p1[1], p2[1], None]); pz.extend([0, 0, None])

    for dt in detected_texts:
        t = msp.add_text(dt['text'], dxfattribs={'height': dt['h']*0.8})
        t.set_placement((dt['x'], dt['y'], 0), align=TextEntityAlignment.MIDDLE_CENTER)

    return doc, px, py, pz

# --- UI ---
with st.sidebar:
    st.header("⚙️ 1. 도면 척도(Scale) 설정")
    cal_mode = st.radio("척도 설정 방식", ["이미지 전체 폭 기준", "특정 구간 지정 기준"])
    
    if cal_mode == "이미지 전체 폭 기준":
        real_w = st.number_input("도면 실제 가로폭 (mm)", value=10000)
        c_px, c_mm = 0, 0
    else:
        c_px = st.number_input("이미지 상의 거리 (px)", value=100)
        c_mm = st.number_input("해당 구간 실제 길이 (mm)", value=900)
        real_w = 0

    st.divider()
    st.header("⚙️ 2. 기본 설정")
    enable_3d = st.checkbox("🏗️ 3D 모드", value=True)
    wall_h = st.number_input("벽높이(mm)", value=2400)
    
    st.divider()
    st.subheader("🤖 3. AI 수정 버튼")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📏 직각 보정", use_container_width=True): st.session_state.cmd = "직각 보정"
        if st.button("🔗 선 연결", use_container_width=True): st.session_state.cmd = "선 연결"
    with c2:
        if st.button("🧹 잡티 제거", use_container_width=True): st.session_state.cmd = "잡티 제거"
        if st.button("🔄 초기화", use_container_width=True): st.session_state.cmd = "일반"
    
    st.write(f"현재 모드: **{st.session_state.cmd}**")
    st.divider()
    f_val = st.slider("인식 민감도", 50, 255, 160)
    eps = st.slider("직선화 강도", 0.001, 0.050, 0.015)
    snap = st.selectbox("그리드 스냅(mm)", [0, 1, 5, 10, 50], index=3)

st.title("📐 Sketch to DXF Pro")
uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    data = uploaded.read()
    with st.spinner("AI 엔진 가동 중..."):
        # 수정된 파라미터들(c_px, c_mm)을 엔진에 전달
        res = process_sketch_pro(data, real_w, wall_h, snap, eps, enable_3d, f_val, st.session_state.cmd, c_px, c_mm)
        if res:
            doc, px, py, pz = res
            col1, col2 = st.columns(2)
            with col1:
                st.image(data, caption="원본 이미지", use_column_width=True)
            with col2:
                fig = go.Figure(go.Scatter3d(x=px, y=py, z=pz, mode='lines', line=dict(color='#00ffcc', width=2)))
                fig.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig, use_container_width=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    doc.saveas(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 DXF 다운로드", f, "output.dxf", use_container_width=True)
                os.unlink(tmp.name)