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

# 페이지 설정
st.set_page_config(page_title="Sketch to DXF Pro + Editor", layout="wide")

# 1. 메모리 세이프 OCR 로더
@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False, download_enabled=True)
    except Exception as e:
        st.warning(f"OCR 엔진 로딩 지연 중: {e}")
        return None

def process_sketch_pro(image_bytes, real_width_mm, wall_height_mm, snap_size, epsilon_adj, enable_3d, filter_strength):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None: return None
    
    h, w, _ = img_bgr.shape
    scale = real_width_mm / w if real_width_mm > 0 else 1.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, filter_strength]) 
    binary = cv2.inRange(hsv, lower_dark, upper_dark)

    lower_grid = np.array([75, 20, 150]) 
    upper_grid = np.array([135, 120, 255])
    grid_mask = cv2.inRange(hsv, lower_grid, upper_grid)
    binary = cv2.subtract(binary, grid_mask)

    kernel = np.ones((2,2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    reader = load_ocr_reader()
    detected_texts = []
    if reader:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray)
            for (bbox, text, prob) in ocr_results:
                if prob < 0.3: continue
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(binary, [pts], (0))
                cx = np.mean(pts[:, 0]) * scale
                cy = (h - np.mean(pts[:, 1])) * scale
                detected_texts.append({'text': text, 'x': cx, 'y': cy, 'h': (pts[2][1]-pts[0][1])*scale})
        except:
            pass

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add("WALL_2D_AUTO", color=7)
    if enable_3d:
        doc.layers.add("VERT_COL", color=2)
        doc.layers.add("CEIL_LINE", color=3)

    plot_x, plot_y, plot_z = [], [], []
    v_columns = set()

    def get_snap(pt):
        if snap_size == 0: return pt
        return (round(pt[0]/snap_size)*snap_size, round(pt[1]/snap_size)*snap_size)

    for cnt in contours:
        if cv2.contourArea(cnt) < 40: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_adj * peri, True)
        pts = [get_snap((p[0][0]*scale, (h-p[0][1])*scale)) for p in approx]
        
        if len(pts) > 1:
            pts.append(pts[0])
            for i in range(len(pts)-1):
                p1, p2 = pts[i], pts[i+1]
                if p1 == p2: continue
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0), dxfattribs={'layer': 'WALL_2D_AUTO'})
                if enable_3d:
                    for p in [p1, p2]:
                        if p not in v_columns:
                            msp.add_line((p[0], p[1], 0), (p[0], p[1], wall_height_mm), dxfattribs={'layer': 'VERT_COL'})
                            v_columns.add(p)
                    msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm), dxfattribs={'layer': 'CEIL_LINE'})
                    plot_x.extend([p1[0], p2[0], p2[0], p1[0], p1[0], None])
                    plot_y.extend([p1[1], p2[1], p2[1], p1[1], p1[1], None])
                    plot_z.extend([0, 0, wall_height_mm, wall_height_mm, 0, None])
                else:
                    plot_x.extend([p1[0], p2[0], None]); plot_y.extend([p1[1], p2[1], None]); plot_z.extend([0, 0, None])

    for dt in detected_texts:
        t = msp.add_text(dt['text'], dxfattribs={'height': dt['h']*0.8, 'color': 1})
        t.set_placement((dt['x'], dt['y'], 0), align=TextEntityAlignment.MIDDLE_CENTER)

    return doc, plot_x, plot_y, plot_z, h, w

# --- UI ---
st.title("📐 Pro Sketch to DXF + Interactive Editor")

with st.sidebar:
    st.header("1. 편집 및 3D 설정")
    edit_mode = st.radio("작업 모드", ("자동 분석", "수동 편집 (캔버스)"))
    enable_3d = st.checkbox("🏗️ 3D 벽체 세우기", value=True)
    filter_val = st.slider("🔍 인식 민감도", 50, 255, 160)
    
    st.divider()
    st.header("2. 실제 치수 (mm)")
    real_w = st.number_input("도면 실제 가로 폭", value=10000)
    wall_h = st.number_input("벽 높이", value=2400, disabled=not enable_3d)
    
    st.divider()
    st.header("3. 캔버스 도구")
    drawing_mode = st.selectbox("그리기 도구", ("line", "rect", "transform"))
    stroke_color = st.color_picker("선 색상", "#FF0000")

uploaded = st.file_uploader("이미지 파일 업로드", type=['png', 'jpg', 'jpeg'])

if uploaded:
    bytes_data = uploaded.read()
    img_for_canvas = Image.open(uploaded)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        if edit_mode == "수동 편집 (캔버스)":
            st.subheader("🖋️ 캔버스 편집")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color=stroke_color,
                background_image=img_for_canvas,
                update_streamlit=True,
                height=600,
                width=600,
                drawing_mode=drawing_mode,
                key="canvas",
            )
        else:
            st.image(bytes_data, caption="원본 이미지", use_container_width=True)

    with st.spinner("데이터 처리 중..."):
        # 기본 자동 분석 실행
        doc, px, py, pz, orig_h, orig_w = process_sketch_pro(bytes_data, real_w, wall_h, 10, 0.015, enable_3d, filter_val)
        
        # 수동 편집 데이터가 있다면 DXF에 병합
        if edit_mode == "수동 편집 (캔버스)" and canvas_result.json_data is not None:
            msp = doc.modelspace()
            doc.layers.add("WALL_MANUAL", color=1) # 수동 선은 빨간색 레이어
            
            scale_canvas = real_w / 600 # 캔버스 크기(600) 대비 실제 mm 스케일
            
            df = pd.json_normalize(canvas_result.json_data["objects"])
            if not df.empty:
                for _, obj in df.iterrows():
                    if obj['type'] == 'line':
                        # 좌표 계산 (캔버스 600px 기준 -> 실제 mm)
                        x1, y1 = (obj['left'] + obj['x1']) * scale_canvas, (600 - (obj['top'] + obj['y1'])) * scale_canvas
                        x2, y2 = (obj['left'] + obj['x2']) * scale_canvas, (600 - (obj['top'] + obj['y2'])) * scale_canvas
                        
                        msp.add_line((x1, y1, 0), (x2, y2, 0), dxfattribs={'layer': 'WALL_MANUAL'})
                        # 3D 시각화에도 추가
                        px.extend([x1, x2, None]); py.extend([y1, y2, None]); pz.extend([0, 0, None])

        # 결과 시각화
        fig = go.Figure(go.Scatter3d(x=px, y=py, z=pz, mode='lines', line=dict(color='#00ffcc', width=2)))
        fig.update_layout(scene=dict(aspectmode='data', bgcolor='black'), paper_bgcolor='black', margin=dict(l=0,r=0,b=0,t=0))
        
        with col2:
            st.plotly_chart(fig, use_container_width=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                doc.saveas(tmp.name)
                st.download_button("📥 최종 DXF 다운로드 (자동+수동)", open(tmp.name, "rb"), "final_drawing.dxf", use_container_width=True)
            os.unlink(tmp.name)