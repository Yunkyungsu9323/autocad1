import streamlit as st
import cv2
import numpy as np
import ezdxf
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os

# 1. 페이지 설정
st.set_page_config(page_title="AI Sketch to DXF Pro", layout="wide")

# [세션 상태] 버튼 클릭 값 저장
if "cmd" not in st.session_state:
    st.session_state.cmd = ""

# --- [AI 분석 엔진: 기존 로직 그대로 복구] ---
def process_sketch_ai_engine(image_bytes, real_width_mm, wall_height_mm, user_instruction):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    final_scale = real_width_mm / w if real_width_mm > 0 else 1.0
    
    # 버튼 명령에 따른 파라미터 미세 조정
    cleanup_val = 40
    if "깔끔" in user_instruction: cleanup_val = 200
    if "세밀" in user_instruction: cleanup_val = 5
    
    snap_size = 10
    if "연결" in user_instruction: snap_size = 25
    
    ortho_mode = "직각" in user_instruction
    thick_mode = "두께" in user_instruction

    # 전처리 로직 (그리드 제거 및 이진화)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 160]))
    grid_mask = cv2.inRange(hsv, np.array([75, 20, 150]), np.array([135, 120, 255]))
    binary = cv2.subtract(binary, grid_mask)
    binary = cv2.dilate(binary, np.ones((2,2), np.uint8), iterations=1)

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    px_list, py_list, pz_list = [], [], []
    v_cols = set()

    def apply_snap(pt, s):
        if s == 0: return pt
        return (round(pt[0]/s)*s, round(pt[1]/s)*s)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < cleanup_val: continue
        approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
        pts = [apply_snap((p[0][0]*final_scale, (h-p[0][1])*final_scale), snap_size) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if ortho_mode:
                    dx, dy = abs(p1[0]-p2[0]), abs(p1[1]-p2[1])
                    p2 = (p2[0], p1[1]) if dx > dy else (p1[0], p2[1])
                if p1 == p2: continue
                
                # 벽체 생성 로직
                offsets = [0] if not thick_mode else [-100, 100]
                for off in offsets:
                    msp.add_line((p1[0]+off, p1[1]+off, 0), (p2[0]+off, p2[1]+off, 0))
                
                # 시각화용 3D 데이터
                px_list.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                py_list.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                pz_list.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])

    return doc, px_list, py_list, pz_list, img_rgb

# --- [UI 레이아웃] ---

with st.sidebar:
    st.header("⚙️ 설정")
    real_w = st.number_input("가로폭(mm)", value=10000)
    wall_h = st.number_input("벽높이(mm)", value=2400)
    
    st.divider()
    
    # 🔴 [핵심] 입력창 대신 버튼 6개 배치
    st.subheader("🤖 AI 수정 명령")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📏 직각 보정", use_container_width=True): st.session_state.cmd = "직각"
        if st.button("🔗 선 연결", use_container_width=True): st.session_state.cmd = "연결"
        if st.button("🔍 세밀 인식", use_container_width=True): st.session_state.cmd = "세밀"
    with c2:
        if st.button("🧹 잡티 제거", use_container_width=True): st.session_state.cmd = "깔끔"
        if st.button("🧱 두께 생성", use_container_width=True): st.session_state.cmd = "두께"
        if st.button("🔄 초기화", use_container_width=True): st.session_state.cmd = ""
    
    if st.session_state.cmd:
        st.success(f"현재 적용: {st.session_state.cmd}")

# 메인 화면
st.title("📐 Sketch to DXF Pro")
uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    # 업로드된 이미지 처리
    image_data = uploaded.read()
    with st.spinner(f"AI가 {st.session_state.cmd} 모드로 분석 중..."):
        res = process_sketch_ai_engine(image_data, real_w, wall_h, st.session_state.cmd)
        
        if res:
            doc, px_d, py_d, pz_d, img_rgb = res
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("🔍 원본 분석")
                st.plotly_chart(px.imshow(img_rgb), use_container_width=True)
            with col_b:
                st.write("🏗️ 3D 프리뷰")
                fig = go.Figure(go.Scatter3d(x=px_d, y=py_d, z=pz_d, mode='lines', line=dict(color='#00ffcc', width=3)))
                fig.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig, use_container_width=True)
                
            # DXF 다운로드
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                doc.saveas(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button("📥 DXF 다운로드", f, file_name="output.dxf", use_container_width=True)
            os.unlink(tmp.name)