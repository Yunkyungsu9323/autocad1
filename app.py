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

def process_sketch_final(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, filter_strength, user_instruction=""):
    # 이미지 로드
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 스케일 결정
    final_scale = real_width_mm / w if real_width_mm > 0 else 1.0
    if "크게" in user_instruction: final_scale *= 1.2

    # 2. 격자 제거 및 이진화
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, filter_strength]))
    grid_mask = cv2.inRange(hsv, np.array([75, 20, 150]), np.array([135, 120, 255]))
    binary = cv2.subtract(binary, grid_mask)
    binary = cv2.dilate(binary, np.ones((2,2), np.uint8), iterations=1)

    # 3. DXF 및 시각화 데이터 준비
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    px_list, py_list, pz_list = [], [], []
    v_cols = set()
    ortho_mode = any(word in user_instruction for word in ["직각", "수직", "반듯"])

    def get_snap(pt):
        if snap_size == 0: return pt
        return (round(pt[0]/snap_size)*snap_size, round(pt[1]/snap_size)*snap_size)

    # 4. 윤곽선 분석
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 40: continue
        approx = cv2.approxPolyDP(cnt, epsilon_adj * cv2.arcLength(cnt, True), True)
        pts = [get_snap((p[0][0]*final_scale, (h-p[0][1])*final_scale)) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if ortho_mode:
                    dx, dy = abs(p1[0]-p2[0]), abs(p1[1]-p2[1])
                    p2 = (p2[0], p1[1]) if dx > dy else (p1[0], p2[1])
                if p1 == p2: continue
                
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0))
                # 3D 벽체 데이터 생성
                for pt in [p1, p2]:
                    if pt not in v_cols:
                        msp.add_line((pt[0], pt[1], 0), (pt[0], pt[1], wall_height_mm))
                        v_cols.add(pt)
                msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm))
                
                # 시각화용 데이터
                px_list.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                py_list.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                pz_list.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])

    return doc, px_list, py_list, pz_list, img_rgb

# --- UI 레이아웃 ---
st.title("📐 Sketch to DXF Pro (No-Error Version)")

with st.sidebar:
    st.header("설정")
    real_w = st.number_input("가로폭(mm)", value=10000)
    wall_h = st.number_input("벽높이(mm)", value=2400)
    user_comment = st.text_input("수정 명령", placeholder="예: 직각으로")
    filter_val = st.slider("민감도", 50, 255, 160)
    snap = st.selectbox("그리드 스냅", [1, 5, 10, 50], index=2)
    eps = st.slider("직선화", 0.001, 0.050, 0.015)

uploaded = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    bytes_data = uploaded.read()
    
    with st.spinner("이미지 분석 중..."):
        res = process_sketch_final(bytes_data, real_w, wall_h, snap, eps, filter_val, user_comment)
        
        if res:
            doc, px_data, py_data, pz_data, img_rgb = res
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 원본 이미지 (Plotly View)")
                # st.image 대신 Plotly로 이미지 표시 (TypeError 원천 차단)
                fig_img = px.imshow(img_rgb)
                fig_img.update_layout(margin=dict(l=0,r=0,b=0,t=0), xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(fig_img, use_container_width=True)

            with col2:
                st.write("### 3D 벡터 프리뷰")
                fig_3d = go.Figure(go.Scatter3d(x=px_data, y=py_data, z=pz_data, mode='lines', line=dict(color='#00ffcc', width=2)))
                fig_3d.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # DXF 다운로드
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    doc.saveas(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 DXF 다운로드", f, file_name="output.dxf")
                os.unlink(tmp.name)