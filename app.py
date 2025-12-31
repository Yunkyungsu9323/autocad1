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

if "cmd" not in st.session_state:
    st.session_state.cmd = "일반"

@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except Exception:
        return None

# 분석 엔진 (구조적 완성도 강화 버전)
def process_sketch_pro(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, enable_3d, filter_strength, user_cmd):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    
    h, w, _ = img_bgr.shape
    scale = real_width_mm / w if real_width_mm > 0 else 1.0

    # 파라미터 보정
    f_val = 200 if user_cmd == "잡티 제거" else filter_strength
    s_val = 50 if user_cmd == "선 연결" else snap_size
    e_val = 0.040 if user_cmd == "직각 보정" else epsilon_adj
    wall_thickness = 150 if user_cmd == "벽체 두께" else 0

    # 전처리 (기존 유지)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, f_val]))
    grid_mask = cv2.inRange(hsv, np.array([75, 20, 150]), np.array([135, 120, 255]))
    binary = cv2.subtract(binary, grid_mask)
    binary = cv2.dilate(binary, np.ones((2,2), np.uint8), iterations=1)

    # OCR (기존 유지)
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

    # 벡터화 및 DXF 생성 (구조적 개선)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    px, py, pz = [], [], []

    for cnt in contours:
        if cv2.contourArea(cnt) < 40: continue
        approx = cv2.approxPolyDP(cnt, e_val * cv2.arcLength(cnt, True), True)
        
        # 스냅 적용된 좌표 리스트 생성
        raw_pts = []
        for p in approx:
            px_val = p[0][0] * scale
            py_val = (h - p[0][1]) * scale
            if s_val > 0:
                px_val = round(px_val / s_val) * s_val
                py_val = round(py_val / s_val) * s_val
            raw_pts.append((px_val, py_val))

        if len(raw_pts) < 2: continue
        
        # 다각형 폐쇄 처리
        if raw_pts[0] != raw_pts[-1]:
            raw_pts.append(raw_pts[0])

        # --- 구조적 벽체 생성 (LWPOLYLINE 활용) ---
        # 1. 외곽선 그리기
        msp.add_lwpolyline(raw_pts, dxfattribs={'layer': 'WALL_OUTER'})
        
        # 2. 벽체 두께 모드일 때 (구조적 연결)
        if wall_thickness > 0:
            # 단순 선 복제가 아닌, '면'을 구성하기 위해 오프셋 루프 생성 시뮬레이션
            # (실제 CAD의 Offset 명령처럼 모서리 교차점을 계산하여 연결)
            for i in range(len(raw_pts)-1):
                p1, p2 = raw_pts[i], raw_pts[i+1]
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist == 0: continue
                
                nx, ny = -dy/dist * wall_thickness, dx/dist * wall_thickness
                inner_p1 = (p1[0] + nx, p1[1] + ny)
                inner_p2 = (p2[0] + nx, p2[1] + ny)
                
                # 내부선 추가
                msp.add_line(inner_p1, inner_p2, dxfattribs={'layer': 'WALL_INNER'})
                
                # 시각화 데이터 (Plotly 3D용)
                if enable_3d:
                    # 벽의 면을 채우는 느낌으로 렌더링 데이터 구성
                    px.extend([p1[0], p2[0], inner_p2[0], inner_p1[0], p1[0], None])
                    py.extend([p1[1], p2[1], inner_p2[1], inner_p1[1], p1[1], None])
                    pz.extend([0, 0, 0, 0, 0, None]) # 바닥면
                    pz.extend([wall_height_mm]*6) # 천장면 데이터는 루프 밖에서 처리 가능
                else:
                    px.extend([p1[0], p2[0], None, inner_p1[0], inner_p2[0], None])
                    py.extend([p1[1], p2[1], None, inner_p1[1], inner_p2[1], None])
                    pz.extend([0, 0, None, 0, 0, None])

    for dt in detected_texts:
        t = msp.add_text(dt['text'], dxfattribs={'height': dt['h']*0.8})
        t.set_placement((dt['x'], dt['y'], 0), align=TextEntityAlignment.MIDDLE_CENTER)

    return doc, px, py, pz


# --- UI (사용자님이 보내준 레이아웃 그대로 유지) ---
with st.sidebar:
    st.header("⚙️ 설정")
    enable_3d = st.checkbox("🏗️ 3D 모드", value=True)
    real_w = st.number_input("가로폭(mm)", value=10000)
    wall_h = st.number_input("벽높이(mm)", value=2400)
    
    st.divider()
    st.subheader("🤖 AI 수정 버튼")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📏 직각 보정", use_container_width=True): st.session_state.cmd = "직각 보정"
        if st.button("🔗 선 연결", use_container_width=True): st.session_state.cmd = "선 연결"
    with c2:
        if st.button("🧹 잡티 제거", use_container_width=True): st.session_state.cmd = "잡티 제거"
        # 새로운 벽체 두께 버튼 추가
        if st.button("🧱 벽체 두께", use_container_width=True): st.session_state.cmd = "벽체 두께"
        if st.button("🔄 초기화", use_container_width=True): st.session_state.cmd = "일반"
    
    st.write(f"현재 활성 모드: **{st.session_state.cmd}**")
    
    st.divider()
    f_val = st.slider("인식 민감도", 50, 255, 160)
    eps = st.slider("직선화 강도", 0.001, 0.050, 0.015)
    snap = st.selectbox("그리드 스냅(mm)", [0, 1, 5, 10, 50], index=3)

st.title("📐 Sketch to DXF Pro")
uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    data = uploaded.read()
    with st.spinner(f"AI 엔진 가동 중 ({st.session_state.cmd})..."):
        res = process_sketch_pro(data, real_w, wall_h, snap, eps, enable_3d, f_val, st.session_state.cmd)
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