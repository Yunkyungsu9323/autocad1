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

# 세션 상태 초기화
if "cmd" not in st.session_state:
    st.session_state.cmd = ""

# --- [AI 엔진 함수] ---
def process_sketch_ai_engine(image_bytes, real_width_mm, wall_height_mm, user_instruction=""):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    final_scale = real_width_mm / w if real_width_mm > 0 else 1.0
    
    # 명령어 기반 파라미터 분기
    cleanup_val = 40
    if "깔끔" in user_instruction: cleanup_val = 200
    if "세밀" in user_instruction: cleanup_val = 5
    
    snap_size = 10
    if "연결" in user_instruction: snap_size = 25
    
    ortho_mode = "직각" in user_instruction
    thick_mode = "두께" in user_instruction

    # 이미지 처리 (이진화)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

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
                
                # 라인 추가
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0))
                px_list.extend([p1[0], p2[0], None])
                py_list.extend([p1[1], p2[1], None])
                pz_list.extend([0, 0, None])

    return doc, px_list, py_list, pz_list, img_rgb

# --- [UI 레이아웃] ---

st.title("📐 AI 스케치 변환기")

# 1. 수정 명령 영역 (사용자님이 말씀하신 '타자 치는 칸' 위치)
st.write("### 🤖 수정 명령") 

# 타자 치는 칸 대신 버튼들을 가로로 배치
cmd_cols = st.columns([1, 1, 1, 1, 1, 1])

with cmd_cols[0]:
    if st.button("📏 직각으로", use_container_width=True):
        st.session_state.cmd = "직각으로 반듯하게"
with cmd_cols[1]:
    if st.button("🧹 깔끔하게", use_container_width=True):
        st.session_state.cmd = "깔끔하게 지워줘"
with cmd_cols[2]:
    if st.button("🔗 선 연결", use_container_width=True):
        st.session_state.cmd = "끊어진 선 연결"
with cmd_cols[3]:
    if st.button("🧱 벽 두께", use_container_width=True):
        st.session_state.cmd = "벽체 두께 생성"
with cmd_cols[4]:
    if st.button("🔍 세밀하게", use_container_width=True):
        st.session_state.cmd = "세밀하게 디테일"
with cmd_cols[5]:
    if st.button("🔄 초기화", use_container_width=True):
        st.session_state.cmd = ""

# 현재 활성화된 명령 표시 (타자 칸 대신 들어간 결과)
if st.session_state.cmd:
    st.info(f"**현재 적용된 명령:** {st.session_state.cmd}")
else:
    st.write("위 버튼을 눌러 명령을 선택하세요.")

st.divider()

# 2. 파일 업로드 및 설정
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ 설정")
    real_w = st.number_input("도면 가로폭 (mm)", value=10000)
    wall_h = st.number_input("벽체 높이 (mm)", value=2400)
    uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

with col_right:
    if uploaded:
        with st.spinner("AI 처리 중..."):
            res = process_sketch_ai_engine(uploaded.read(), real_w, wall_h, st.session_state.cmd)
            if res:
                doc, px_d, py_d, pz_d, img_rgb = res
                
                # 시각화
                fig_3d = go.Figure(go.Scatter3d(x=px_d, y=py_d, z=pz_d, mode='lines', line=dict(color='#00ffcc', width=3)))
                fig_3d.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # 다운로드
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    doc.saveas(tmp.name)
                    st.download_button("📥 DXF 다운로드", open(tmp.name, "rb"), "output.dxf", use_container_width=True)
                os.unlink(tmp.name)