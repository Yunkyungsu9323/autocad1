import streamlit as st
import cv2
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
import plotly.graph_objects as go
import tempfile
import os
import easyocr
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Sketch to DXF Pro", layout="wide")

@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except: return None

def process_sketch_pro(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, enable_3d, filter_strength):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    h, w, _ = img_bgr.shape
    scale = real_width_mm / w
    
    # 전처리 및 격자 제거
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, filter_strength]))
    grid_mask = cv2.inRange(hsv, np.array([75, 20, 150]), np.array([135, 120, 255]))
    binary = cv2.subtract(binary, grid_mask)
    binary = cv2.dilate(binary, np.ones((2,2), np.uint8), iterations=1)

    # 선 추출 및 DXF 생성
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    px, py, pz = [], [], []
    v_cols = set()

    for cnt in contours:
        if cv2.contourArea(cnt) < 40: continue
        approx = cv2.approxPolyDP(cnt, epsilon_adj * cv2.arcLength(cnt, True), True)
        pts = [(round(p[0][0]*scale/snap_size)*snap_size, round((h-p[0][1])*scale/snap_size)*snap_size) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if p1 == p2: continue
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0), dxfattribs={'layer': 'AUTO_WALL'})
                if enable_3d:
                    for p in [p1, p2]:
                        if p not in v_cols:
                            msp.add_line((p[0], p[1], 0), (p[0], p[1], wall_height_mm), dxfattribs={'layer': 'VERT'})
                            v_cols.add(p)
                    msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm), dxfattribs={'layer': 'CEIL'})
                    px.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                    py.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                    pz.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])
                else:
                    px.extend([p1[0], p2[0], None]); py.extend([p1[1], p2[1], None]); pz.extend([0, 0, None])

    return doc, px, py, pz, h, w

# --- UI ---
st.sidebar.header("🛠️ 설정 및 도구")
mode = st.sidebar.radio("작업 모드", ["자동 분석 결과", "수동 보정 편집"])
real_w = st.sidebar.number_input("실제 가로폭(mm)", value=10000)
wall_h = st.sidebar.number_input("벽 높이(mm)", value=2400)
filter_val = st.sidebar.slider("인식 민감도", 50, 255, 160)
snap = st.sidebar.selectbox("그리드 스냅(mm)", [1, 5, 10, 50], index=2)

uploaded = st.file_uploader("도면 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    bytes_data = uploaded.read()
    img_pil = Image.open(uploaded)
    
    # 1. 자동 분석 실행 (항상 백그라운드에서 실행)
    doc, px, py, pz, oh, ow = process_sketch_pro(bytes_data, real_w, wall_h, snap, 0.015, True, filter_val)

    col1, col2 = st.columns(2)

    with col1:
        if mode == "수동 보정 편집":
            st.subheader("🖋️ 가이드 선 그리기")
            # 캔버스 호출 (AttributeError 방지를 위해 배경 이미지를 600x600으로 리사이즈 권장)
            canvas_result = st_canvas(
                stroke_width=3, stroke_color="#FF0000",
                background_image=img_pil.resize((600, int(600*oh/ow))),
                update_streamlit=True, height=int(600*oh/ow), width=600,
                drawing_mode="line", key="canvas_pro"
            )
            
            if canvas_result.json_data:
                df = pd.json_normalize(canvas_result.json_data["objects"])
                if not df.empty:
                    msp = doc.modelspace()
                    s = real_w / 600
                    for _, obj in df.iterrows():
                        if obj['type'] == 'line':
                            x1, y1 = (obj['left']+obj['x1'])*s, (int(600*oh/ow)-(obj['top']+obj['y1']))*s
                            x2, y2 = (obj['left']+obj['x2'])*s, (int(600*oh/ow)-(obj['top']+obj['y2']))*s
                            msp.add_line((x1,y1,0), (x2,y2,0), dxfattribs={'layer':'MANUAL', 'color':1})
                            px.extend([x1, x2, None]); py.extend([y1, y2, None]); pz.extend([0, 0, None])
        else:
            st.image(bytes_data, use_container_width=True, caption="자동 분석된 영역")

    with col2:
        st.subheader("📦 3D 미리보기")
        fig = go.Figure(go.Scatter3d(x=px, y=py, z=pz, mode='lines', line=dict(color='#00ffcc', width=2)))
        fig.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            doc.saveas(tmp.name)
            st.download_button("📥 최종 DXF 다운로드", open(tmp.name, "rb"), "result.dxf", use_container_width=True)