import streamlit as st
import cv2
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
import plotly.graph_objects as go
import tempfile
import os
import easyocr

# 페이지 설정
st.set_page_config(page_title="Sketch to DXF Pro", layout="wide")

# 세션 상태 초기화 (버튼 클릭 값 저장용)
if "cmd" not in st.session_state:
    st.session_state.cmd = "일반"

# 1. 메모리 세이프 OCR 로더
@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False, download_enabled=True)
    except Exception as e:
        st.warning(f"OCR 엔진 로딩 지연 중: {e}")
        return None

# 2. 메인 분석 엔진 (기존 로직 유지 + 버튼 명령 연동)
def process_sketch_pro(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, enable_3d, filter_strength, user_cmd):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    
    h, w, _ = img_bgr.shape
    scale = real_width_mm / w if real_width_mm > 0 else 1.0

    # [버튼 연동] 명령에 따른 필터/스냅 수치 강제 조정
    current_filter = 200 if user_cmd == "잡티 제거" else filter_strength
    current_snap = 50 if user_cmd == "선 연결" else snap_size
    current_eps = 0.040 if user_cmd == "직각 보정" else epsilon_adj

    # 컬러 필터 (격자 제거 로직)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, current_filter]) 
    binary = cv2.inRange(hsv, lower_dark, upper_dark)

    lower_grid = np.array([75, 20, 150]) 
    upper_grid = np.array([135, 120, 255])
    grid_mask = cv2.inRange(hsv, lower_grid, upper_grid)
    binary = cv2.subtract(binary, grid_mask)

    kernel = np.ones((2,2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # OCR 처리
    reader = load_ocr_reader()
    detected_texts = []
    if reader:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray)
            for (bbox, text, prob) in ocr_results:
                if prob < 0.3: continue
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(binary, [pts], (0)) # 텍스트 자리는 벡터에서 제외
                cx = np.mean(pts[:, 0]) * scale
                cy = (h - np.mean(pts[:, 1])) * scale
                detected_texts.append({'text': text, 'x': cx, 'y': cy, 'h': (pts[2][1]-pts[0][1])*scale})
        except: pass

    # 벡터화 및 DXF 생성
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    plot_x, plot_y, plot_z = [], [], []
    v_columns = set()

    def get_snap(pt):
        if current_snap == 0: return pt
        return (round(pt[0]/current_snap)*current_snap, round(pt[1]/current_snap)*current_snap)

    for cnt in contours:
        if cv2.contourArea(cnt) < 40: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, current_eps * peri, True)
        pts = [get_snap((p[0][0]*scale, (h-p[0][1])*scale)) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if p1 == p2: continue
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0))
                
                if enable_3d:
                    for p in [p1, p2]:
                        if p not in v_columns:
                            msp.add_line((p[0], p[1], 0), (p[0], p[1], wall_height_mm))
                            v_columns.add(p)
                    msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm))
                    plot_x.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                    plot_y.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                    plot_z.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])
                else:
                    plot_x.extend([p1[0], p2[0], None])
                    plot_y.extend([p1[1], p2[1], None])
                    plot_z.extend([0, 0, None])

    for dt in detected_texts:
        t = msp.add_text(dt['text'], dxfattribs={'height': dt['h']*0.8})
        t.set_placement((dt['x'], dt['y'], 0), align=TextEntityAlignment.MIDDLE_CENTER)

    return doc, plot_x, plot_y, plot_z

# --- Streamlit UI ---
st.title("📐 Professional Sketch to DXF")

with st.sidebar:
    st.header("1. 기본 설정")
    enable_3d = st.checkbox("🏗️ 3D 벽체 세우기", value=True)
    real_w = st.number_input("도면 실제 가로 폭(mm)", value=10000)
    wall_h = st.number_input("벽 높이(mm)", value=2400, disabled=not enable_3d)
    
    st.divider()
    
    # 🔴 [여기가 수정된 버튼 영역]
    st.header("2. AI 수정 명령")
    st.write("버튼을 누르면 즉시 보정됩니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📏 직각 보정", use_container_width=True): st.session_state.cmd = "직각 보정"
        if st.button("🔗 선 연결", use_container_width=True): st.session_state.cmd = "선 연결"
    with col2:
        if st.button("🧹 잡티 제거", use_container_width=True): st.session_state.cmd = "잡티 제거"
        if st.button("🔄 초기화", use_container_width=True): st.session_state.cmd = "일반"
    
    if st.session_state.cmd != "일반":
        st.success(f"현재 적용: {st.session_state.cmd}")
    
    st.divider()
    st.header("3. 미세 조정 (수동)")
    filter_val = st.slider("인식 민감도", 50, 255, 160)
    eps = st.slider("직선화 강도", 0.001, 0.050, 0.015)
    snap = st.selectbox("그리드 스냅 (mm)", [0, 1, 5, 10, 50], index=3)

uploaded = st.file_uploader("이미지 파일 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    bytes_data = uploaded.read()
    with st.spinner(f"AI 엔진 분석 중 ({st.session_state.cmd})..."):
        res = process_sketch_pro(bytes_data, real_w, wall_h, snap, eps, enable_3d, filter_val, st.session_state.cmd)
        
        if res:
            doc, px, py, pz = res
            col1, col2 = st.columns(2)
            with col1:
                st.image(bytes_data, caption="원본 이미지", use_container_width=True)
            with col2:
                fig = go.Figure(go.Scatter3d(x=px, y=py, z=pz, mode='lines', line=dict(color='#00ffcc', width=2)))
                fig.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
                st.plotly_chart(fig, use_container_width=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                    doc.saveas(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 DXF 다운로드", f, "pro_plan.dxf", use_container_width=True)
                os.unlink(tmp.name)