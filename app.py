import streamlit as st
import cv2
import numpy as np
import ezdxf
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os

# 1. 페이지 설정
st.set_page_config(page_title="Sketch to DXF Pro", layout="wide")

# 세션 상태 초기화 (버튼 클릭 값 저장)
if "cmd" not in st.session_state:
    st.session_state.cmd = ""

# --- [AI 엔진 로직] ---
def process_sketch_ai_engine(image_bytes, real_width_mm, wall_height_mm, user_instruction):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    h, w, _ = img_bgr.shape
    
    final_scale = real_width_mm / w if real_width_mm > 0 else 1.0
    
    # 버튼 명령에 따른 설정값 변경
    cleanup_val = 200 if "깔끔" in user_instruction else 40
    ortho_mode = "직각" in user_instruction
    thick_mode = "두께" in user_instruction

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
                px_list.extend([p1[0], p2[0], None]); py_list.extend([p1[1], p2[1], None]); pz_list.extend([0, 0, None])

    return doc, px_list, py_list, pz_list

# --- [UI 레이아웃] ---

# 왼쪽 사이드바 설정
with st.sidebar:
    st.header("설정")
    real_w = st.number_input("가로폭(mm)", value=10000)
    wall_h = st.number_input("벽높이(mm)", value=2400)
    
    st.divider()
    
    # [핵심 수정] 기존의 '수정 명령' 입력창을 삭제하고 버튼 배치
    st.write("**수정 명령 (버튼 클릭)**")
    
    # 버튼들을 2개씩 한 줄에 배치 (공간 효율)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📏 직각으로", use_container_width=True): st.session_state.cmd = "직각"
        if st.button("🔗 선 연결", use_container_width=True): st.session_state.cmd = "연결"
    with col2:
        if st.button("🧹 깔끔하게", use_container_width=True): st.session_state.cmd = "깔끔"
        if st.button("🧱 두께 생성", use_container_width=True): st.session_state.cmd = "두께"
    
    if st.button("🔄 초기화", use_container_width=True):
        st.session_state.cmd = ""

    # 현재 어떤 버튼이 눌렸는지 표시
    if st.session_state.cmd:
        st.info(f"선택됨: {st.session_state.cmd}")

# 메인 화면
st.title("📐 Sketch to DXF Pro (Button Version)")
uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    with st.spinner("AI 분석 중..."):
        res = process_sketch_ai_engine(uploaded.read(), real_w, wall_h, st.session_state.cmd)
        if res:
            doc, px_d, py_d, pz_d = res
            
            # 결과 시각화
            fig_3d = go.Figure(go.Scatter3d(x=px_d, y=py_d, z=pz_d, mode='lines', line=dict(color='#00ffcc', width=3)))
            fig_3d.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 다운로드
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                doc.saveas(tmp.name)
                st.download_button("📥 DXF 다운로드", open(tmp.name, "rb"), "output.dxf", use_container_width=True)
            os.unlink(tmp.name)