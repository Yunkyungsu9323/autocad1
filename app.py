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

# [중요] 세션 상태 초기화: 버튼 클릭 값을 저장하는 저장소입니다.
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
    
    # 세션에 저장된 버튼 명령에 따라 파라미터를 조절합니다.
    cleanup_val = 40
    if "깔끔" in user_instruction: cleanup_val = 200
    if "세밀" in user_instruction: cleanup_val = 5
    
    snap_size = 10
    if "연결" in user_instruction: snap_size = 25
    
    ortho_mode = "직각" in user_instruction
    thick_mode = "두께" in user_instruction

    # 이미지 이진화 처리
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
                
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0))
                px_list.extend([p1[0], p2[0], None])
                py_list.extend([p1[1], p2[1], None])
                pz_list.extend([0, 0, None])

    return doc, px_list, py_list, pz_list, img_rgb

# --- [UI 레이아웃] ---
st.title("📐 AI Sketch to DXF Pro")

# 2. 수정 명령 섹션 (사용자님이 말씀하신 입력창 자리에 버튼 배치)
st.write("### 🤖 수정 명령")

# [핵심 수정] st.text_input(타자치기)을 아예 삭제하고 버튼 6개를 가로로 배치
cmd_cols = st.columns(6)

if cmd_cols[0].button("📏 직각 보정", use_container_width=True): 
    st.session_state.cmd = "직각으로 반듯하게"
if cmd_cols[1].button("🧹 잡티 제거", use_container_width=True): 
    st.session_state.cmd = "깔끔하게 지워줘"
if cmd_cols[2].button("🔗 선 연결", use_container_width=True): 
    st.session_state.cmd = "끊어진 선 연결"
if cmd_cols[3].button("🧱 벽체 두께", use_container_width=True): 
    st.session_state.cmd = "두께 생성"
if cmd_cols[4].button("🔍 세밀 인식", use_container_width=True): 
    st.session_state.cmd = "세밀하게 디테일"
if cmd_cols[5].button("🔄 초기화", use_container_width=True): 
    st.session_state.cmd = ""

# 현재 활성화된 모드를 텍스트창 대신 초록색 바(st.success)로 보여줍니다.
if st.session_state.cmd:
    st.success(f"**현재 적용 중인 AI 모드:** {st.session_state.cmd}")
else:
    st.info("위 버튼을 클릭하여 AI에게 명령을 내리세요.")

st.divider()

# 3. 메인 작업 영역 (설정과 결과)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ 도면 설정")
    real_w = st.number_input("도면 실제 가로폭 (mm)", value=10000)
    wall_h = st.number_input("벽체 높이 (mm)", value=2400)
    uploaded = st.file_uploader("스캔 이미지 업로드", type=['png', 'jpg', 'jpeg'], key="main_loader")

with col2:
    if uploaded:
        # 업로드된 데이터를 읽어와 엔진 실행 (세션에 저장된 cmd 값 사용)
        image_bytes = uploaded.getvalue()
        with st.spinner("AI가 버튼 명령을 수행 중입니다..."):
            res = process_sketch_ai_engine(image_bytes, real_w, wall_h, st.session_state.cmd)
            
            if res:
                doc, px_d, py_d, pz_d, img_rgb = res
                st.write("🏗️ AI 분석 프리뷰")
                fig_3d = go.Figure(go.Scatter3d(x=px_d, y=py_d, z=pz_d, mode='lines', line=dict(color='#00ffcc', width=4)))
                fig_3d.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # DXF 파일 생성 및 다운로드 버튼
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    doc.saveas(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 DXF 파일 받기", f, file_name="output.dxf", use_container_width=True)
                os.unlink(tmp.name)
    else:
        st.warning("이미지를 업로드하면 결과가 여기에 표시됩니다.")