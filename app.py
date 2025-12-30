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

# 세션 상태 초기화 (버튼 클릭 값 유지)
if "cmd" not in st.session_state:
    st.session_state.cmd = ""

# --- [AI 엔진 핵심 로직] ---
def process_sketch_ai_engine(image_bytes, real_width_mm, wall_height_mm, user_instruction):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    final_scale = real_width_mm / w if real_width_mm > 0 else 1.0
    
    # 버튼 클릭 내용(user_instruction)에 따라 파라미터 변경
    cleanup_val = 40
    if "깔끔" in user_instruction: cleanup_val = 200
    if "세밀" in user_instruction: cleanup_val = 5
    
    snap_size = 10
    if "연결" in user_instruction: snap_size = 25
    
    ortho_mode = "직각" in user_instruction
    thick_mode = "두께" in user_instruction

    # 전처리
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 160]))
    binary = cv2.dilate(binary, np.ones((2,2), np.uint8), iterations=1)

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    px_list, py_list, pz_list = [], [], []

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < cleanup_val: continue
        approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
        pts = [(p[0][0]*final_scale, (h-p[0][1])*final_scale) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if ortho_mode:
                    dx, dy = abs(p1[0]-p2[0]), abs(p1[1]-p2[1])
                    p2 = (p2[0], p1[1]) if dx > dy else (p1[0], p2[1])
                
                # 도면 및 시각화 데이터 생성
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0))
                px_list.extend([p1[0], p2[0], None])
                py_list.extend([p1[1], p2[1], None])
                pz_list.extend([0, 0, None])

    return doc, px_list, py_list, pz_list, img_rgb

# --- [UI 레이아웃] ---
st.title("📐 AI Sketch to DXF Pro")

# 1. 수정 명령 섹션 (이미지에서 텍스트 상자가 있던 자리를 버튼으로 교체)
st.write("### 🤖 수정 명령")

# 가로로 6개의 버튼 배치 (입력창 대신 이 버튼들을 클릭)
btn_cols = st.columns(6)

if btn_cols[0].button("📏 직각 보정", use_container_width=True): 
    st.session_state.cmd = "직각으로 반듯하게"
if btn_cols[1].button("🧹 잡티 제거", use_container_width=True): 
    st.session_state.cmd = "깔끔하게 지워줘"
if btn_cols[2].button("🔗 선 연결", use_container_width=True): 
    st.session_state.cmd = "끊어진 선 연결"
if btn_cols[3].button("🧱 벽체 두께", use_container_width=True): 
    st.session_state.cmd = "두께 생성"
if btn_cols[4].button("🔍 세밀 인식", use_container_width=True): 
    st.session_state.cmd = "세밀하게 디테일"
if btn_cols[5].button("🔄 초기화", use_container_width=True): 
    st.session_state.cmd = ""

# 현재 선택된 모드 표시 (입력창이 없으므로 사용자가 뭘 눌렀는지 알려줌)
if st.session_state.cmd:
    st.success(f"현재 적용 중: **{st.session_state.cmd}**")
else:
    st.info("위 버튼을 눌러 보정 명령을 내리세요.")

st.divider()

# 2. 메인 설정 및 업로드
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("⚙️ 도면 설정")
    real_w = st.number_input("도면 실제 가로폭 (mm)", value=10000)
    wall_h = st.number_input("벽체 높이 (mm)", value=2400)
    uploaded = st.file_uploader("스캔 이미지 업로드", type=['png', 'jpg', 'jpeg'])

with c2:
    if uploaded:
        with st.spinner("AI 엔진 가동 중..."):
            # 현재 st.session_state.cmd 값을 엔진에 전달
            res = process_sketch_ai_engine(uploaded.read(), real_w, wall_h, st.session_state.cmd)
            if res:
                doc, px_d, py_d, pz_d, img_rgb = res
                st.write("🏗️ 변환 결과 프리뷰")
                fig_3d = go.Figure(go.Scatter3d(x=px_d, y=py_d, z=pz_d, mode='lines', line=dict(color='#00ffcc', width=4)))
                fig_3d.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    doc.saveas(tmp.name)
                    st.download_button("📥 DXF 다운로드", open(tmp.name, "rb"), "output.dxf", use_container_width=True)
                os.unlink(tmp.name)