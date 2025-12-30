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

# 세션 상태 초기화 (가장 상단에 위치해야 합니다)
if "cmd" not in st.session_state:
    st.session_state.cmd = ""

def process_sketch_ai_engine(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, filter_strength, user_instruction=""):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # --- [AI 엔진 핵심 로직] ---
    final_scale = real_width_mm / w if real_width_mm > 0 else 1.0
    
    # 텍스트 명령 기반 파라미터 튜닝
    cleanup_val = 40
    if any(word in user_instruction for word in ["깔끔", "지워"]): cleanup_val = 200
    if any(word in user_instruction for word in ["세밀", "디테일"]): cleanup_val = 5
    
    snap_engine = snap_size
    if any(word in user_instruction for word in ["연결", "붙여"]): snap_engine = snap_size * 2.5
    
    ortho_mode = any(word in user_instruction for word in ["직각", "반듯"])
    thick_mode = "두께" in user_instruction

    # 전처리 (이미지 필터링)
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
                
                # 두께 모드 처리
                offsets = [0] if not thick_mode else [-100, 100]
                for off in offsets:
                    msp.add_line((p1[0]+off, p1[1]+off, 0), (p2[0]+off, p2[1]+off, 0))
                
                # 3D 기둥 및 상단 라인 (시각화용)
                for pt in [p1, p2]:
                    if pt not in v_cols:
                        msp.add_line((pt[0], pt[1], 0), (pt[0], pt[1], wall_height_mm))
                        v_cols.add(pt)
                msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm))
                
                # Plotly 데이터 구성
                px_list.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                py_list.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                pz_list.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])

    return doc, px_list, py_list, pz_list, img_rgb

# --- UI 레이아웃 ---
st.title("📐 AI Sketch to DXF Pro")

# 1. 클릭형 명령어 버튼 섹션
st.subheader("🤖 원클릭 AI 보정 모드")
btn_cols = st.columns(6)

# 각 버튼 클릭 시 세션 상태에 명령 저장 및 페이지 리런(Rerun)
if btn_cols[0].button("📏 직각 보정", use_container_width=True): 
    st.session_state.cmd = "직각으로 반듯하게"
if btn_cols[1].button("🧹 잡티 제거", use_container_width=True): 
    st.session_state.cmd = "깔끔하게 지워줘"
if btn_cols[2].button("🔗 선 연결", use_container_width=True): 
    st.session_state.cmd = "끊어진 선 연결해줘"
if btn_cols[3].button("🧱 벽체 두께", use_container_width=True): 
    st.session_state.cmd = "벽체 두께 생성"
if btn_cols[4].button("🔍 세밀 인식", use_container_width=True): 
    st.session_state.cmd = "세밀하게 디테일 살려줘"
if btn_cols[5].button("🔄 초기화", use_container_width=True): 
    st.session_state.cmd = ""

# 현재 적용된 모드 표시
if st.session_state.cmd:
    st.info(f"**현재 활성화된 AI 모드:** {st.session_state.cmd}")
else:
    st.light_grey("기본 인식 모드입니다. 버튼을 눌러 보정을 시작하세요.")

---

with st.sidebar:
    st.header("⚙️ 기본 설정")
    real_w = st.number_input("도면 실제 가로폭 (mm)", value=10000)
    wall_h = st.number_input("3D 벽체 높이 (mm)", value=2400)
    
    st.divider()
    st.header("✍️ AI 커스텀 명령")
    # 버튼이 아닌 수동 입력도 가능하도록 유지
    user_comment = st.text_input("직접 명령 입력:", value=st.session_state.cmd)
    st.caption("버튼을 누르거나 상세 내용을 직접 타이핑하세요.")

uploaded = st.file_uploader("스캔한 도면 또는 스케치 이미지 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    bytes_data = uploaded.read()
    # 최신 입력값(버튼 혹은 텍스트박스)을 사용하여 엔진 호출
    with st.spinner(f"AI가 '{user_comment}' 설정을 적용 중입니다..."):
        res = process_sketch_ai_engine(bytes_data, real_w, wall_h, 10, 0.015, 160, user_comment)
        
        if res:
            doc, px_d, py_d, pz_d, img_rgb = res
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("🔍 분석된 이미지 영역")
                fig_img = px.imshow(img_rgb)
                fig_img.update_layout(margin=dict(l=0,r=0,b=0,t=0), xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(fig_img, use_container_width=True)
                
            with c2:
                st.write("🏗️ AI 벡터화 프리뷰 (3D)")
                fig_3d = go.Figure(go.Scatter3d(
                    x=px_d, y=py_d, z=pz_d, 
                    mode='lines', 
                    line=dict(color='#00ffcc', width=4)
                ))
                fig_3d.update_layout(
                    scene=dict(aspectmode='data', bgcolor='black'), 
                    paper_bgcolor='black', 
                    margin=dict(l=0,r=0,b=0,t=0)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # DXF 다운로드 처리
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    doc.saveas(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            label="📥 변환된 DXF 파일 다운로드",
                            data=f,
                            file_name="ai_converted_drawing.dxf",
                            mime="application/dxf",
                            use_container_width=True
                        )
                os.unlink(tmp.name)