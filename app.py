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

def process_sketch_ai_engine(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, filter_strength, user_instruction=""):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # --- [AI 엔진 핵심 로직] ---
    final_scale = real_width_mm / w if real_width_mm > 0 else 1.0
    if any(word in user_instruction for word in ["크게", "확대", "배 키워"]): final_scale *= 1.5
    
    cleanup_val = 40
    if any(word in user_instruction for word in ["깔끔", "지워", "청소", "노이즈"]): cleanup_val = 200
    if any(word in user_instruction for word in ["세밀", "디테일", "작은"]): cleanup_val = 5
    
    snap_engine = snap_size
    if any(word in user_instruction for word in ["연결", "붙여", "이어줘"]): snap_engine = snap_size * 2.5
    
    ortho_mode = any(word in user_instruction for word in ["직각", "수직", "반듯", "똑바로"])
    thick_mode = any(word in user_instruction for word in ["두께", "두껍게", "벽체"])

    # 전처리
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, filter_strength]))
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
        approx = cv2.approxPolyDP(cnt, epsilon_adj * cv2.arcLength(cnt, True), True)
        pts = [apply_snap((p[0][0]*final_scale, (h-p[0][1])*final_scale), snap_engine) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if ortho_mode:
                    dx, dy = abs(p1[0]-p2[0]), abs(p1[1]-p2[1])
                    p2 = (p2[0], p1[1]) if dx > dy else (p1[0], p2[1])
                if p1 == p2: continue
                
                offsets = [0] if not thick_mode else [-100, 100]
                for off in offsets:
                    msp.add_line((p1[0]+off, p1[1]+off, 0), (p2[0]+off, p2[1]+off, 0))
                
                for pt in [p1, p2]:
                    if pt not in v_cols:
                        msp.add_line((pt[0], pt[1], 0), (pt[0], pt[1], wall_height_mm))
                        v_cols.add(pt)
                msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm))
                px_list.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                py_list.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                pz_list.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])

    return doc, px_list, py_list, pz_list, img_rgb

# --- UI 레이아웃 ---
st.title("📐 AI Sketch to DXF Pro")

# 1. 시각적 명령어 가이드 (사용자가 바로 보고 따라할 수 있게)
st.subheader("🤖 AI 명령어 가이드")
cols = st.columns(3)
with cols[0]:
    st.info("**📏 정교한 도면**\n- '직각으로 반듯하게'\n- '끊어진 선 다 연결해줘'")
with cols[1]:
    st.info("**🧹 깔끔한 결과**\n- '잡티 다 지우고 깔끔하게'\n- '벽체 두께 200mm 줘'")
with cols[2]:
    st.info("**🔍 상세 인식**\n- '작은 디테일까지 세밀하게'\n- '도면 1.5배로 확대해줘'")

with st.sidebar:
    st.header("⚙️ 기본 설정")
    real_w = st.number_input("도면 가로폭 (mm)", value=10000)
    wall_h = st.number_input("벽체 높이 (mm)", value=2400)
    
    st.divider()
    st.header("✍️ AI 수정 명령")
    # 사용자가 명령을 입력하는 곳
    user_comment = st.text_input("위에 있는 가이드를 참고해서 입력하세요:", placeholder="예: 직각으로 깔끔하게")
    
    st.divider()
    st.header("🔧 수동 미세 조정")
    filter_val = st.slider("민감도", 50, 255, 160)
    snap = st.selectbox("스냅(mm)", [1, 5, 10, 50], index=2)

uploaded = st.file_uploader("이미지 파일 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    bytes_data = uploaded.read()
    with st.spinner("AI가 명령을 분석하여 도면을 생성 중입니다..."):
        res = process_sketch_ai_engine(bytes_data, real_w, wall_h, snap, 0.015, filter_val, user_comment)
        if res:
            doc, px_d, py_d, pz_d, img_rgb = res
            c1, c2 = st.columns(2)
            with c1:
                st.write("🔍 원본 분석")
                fig_img = px.imshow(img_rgb)
                fig_img.update_layout(margin=dict(l=0,r=0,b=0,t=0), xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(fig_img, use_container_width=True)
            with c2:
                st.write("🏗️ AI 결과물")
                fig_3d = go.Figure(go.Scatter3d(x=px_d, y=py_d, z=pz_d, mode='lines', line=dict(color='#00ffcc', width=2)))
                fig_3d.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    doc.saveas(tmp.name)
                    st.download_button("📥 DXF 다운로드", open(tmp.name, "rb"), "output.dxf", use_container_width=True)
                os.unlink(tmp.name)