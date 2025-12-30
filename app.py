import streamlit as st
import cv2
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
import plotly.graph_objects as go
import tempfile
import os
import easyocr

st.set_page_config(page_title="Sketch to DXF (Mode Select)", layout="wide")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False) 

def process_image(image_bytes, real_width_mm, wall_height_mm, epsilon_factor, min_area, enable_extrusion):
    # 1. 이미지 로드
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # 2. 스케일 계산
    scale = 1.0
    if real_width_mm > 0:
        scale = real_width_mm / w

    # 3. OCR
    reader = load_ocr_reader()
    ocr_results = reader.readtext(img, detail=1, paragraph=False)
    detected_texts = []
    img_for_lines = img.copy()

    for (bbox, text, prob) in ocr_results:
        if prob < 0.3: continue
        tl = (int(bbox[0][0]), int(bbox[0][1]))
        br = (int(bbox[2][0]), int(bbox[2][1]))
        cx = (tl[0] + br[0]) / 2
        cy = (tl[1] + br[1]) / 2
        detected_texts.append({
            'text': text,
            'x': cx * scale,
            'y': (h - cy) * scale,
            'height': (br[1] - tl[1]) * scale
        })
        cv2.rectangle(img_for_lines, (tl[0]-5, tl[1]-5), (br[0]+5, br[1]+5), (255), -1)

    # 4. 전처리 및 세선화
    blurred = cv2.GaussianBlur(img_for_lines, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    try:
        # opencv-contrib-python이 설치되어 있어야 함
        thinned = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except:
        thinned = binary # 없으면 그냥 이진화 이미지 사용

    # 5. 선 추출
    contours, _ = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 6. DXF 생성
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    plot_x, plot_y, plot_z = [], [], []
    line_count = 0
    
    for contour in contours:
        if cv2.contourArea(contour) < (min_area * 0.1): continue

        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        
        # 좌표 변환
        curr_points = []
        for p in points:
            x_mm = float(p[0]) * scale
            y_mm = float(h - p[1]) * scale
            curr_points.append((x_mm, y_mm))

        if len(curr_points) < 2: continue

        # --- 모드에 따른 처리 ---
        if enable_extrusion:
            # [모드 1] 3D 벽 세우기 (평면도용)
            for i in range(len(curr_points) - 1):
                p1 = curr_points[i]
                p2 = curr_points[i+1]
                
                # DXF
                msp.add_line((p1[0], p1[1], 0), (p2[0], p2[1], 0), dxfattribs={'layer': 'FLOOR'})
                msp.add_line((p1[0], p1[1], wall_height_mm), (p2[0], p2[1], wall_height_mm), dxfattribs={'layer': 'CEILING', 'color': 3})
                msp.add_line((p1[0], p1[1], 0), (p1[0], p1[1], wall_height_mm), dxfattribs={'layer': 'WALL_VERT', 'color': 2})
                msp.add_line((p2[0], p2[1], 0), (p2[0], p2[1], wall_height_mm), dxfattribs={'layer': 'WALL_VERT', 'color': 2})

                # Plotly (Wireframe Box)
                rect_x = [p1[0], p1[0], p2[0], p2[0], p1[0], None]
                rect_y = [p1[1], p1[1], p2[1], p2[1], p1[1], None]
                rect_z = [0, wall_height_mm, wall_height_mm, 0, 0, None]
                plot_x.extend(rect_x); plot_y.extend(rect_y); plot_z.extend(rect_z)

        else:
            # [모드 2] 단순 선 따기 (복잡한 그림용)
            dxf_pts = [(p[0], p[1], 0) for p in curr_points]
            msp.add_lwpolyline(curr_points, dxfattribs={'layer': 'SKETCH', 'color': 7})
            
            # Plotly (Just Lines on Z=0)
            lx, ly, lz = [], [], []
            for p in curr_points:
                lx.append(p[0]); ly.append(p[1]); lz.append(0)
            lx.append(None); ly.append(None); lz.append(None) # 선 끊기
            
            plot_x.extend(lx); plot_y.extend(ly); plot_z.extend(lz)

        line_count += 1

    # 텍스트
    for dt in detected_texts:
        dxf_text = msp.add_text(dt['text'], dxfattribs={'height': dt['height']*0.8, 'color': 1})
        dxf_text.set_placement((dt['x'], dt['y'], 0), align=TextEntityAlignment.MIDDLE_CENTER)

    return doc, line_count, plot_x, plot_y, plot_z, detected_texts

def main():
    st.title("📐 도면 변환기 (모드 선택)")
    
    st.sidebar.header("1. 변환 모드 (중요!)")
    enable_extrude = st.sidebar.checkbox("🏗️ 3D 벽 세우기 (평면도일 때만 체크!)", value=True, 
                                       help="체크하면 선을 위로 끌어올려 벽을 만듭니다. 입체 그림(투시도)을 넣을 땐 끄세요!")

    st.sidebar.header("2. 설정")
    real_width = st.sidebar.number_input("실제 가로 폭 (mm)", value=10000, step=100)
    
    # [변경됨] 기본값(value)을 2400 -> 1000으로 수정했습니다.
    wall_height = st.sidebar.number_input("벽 높이 (mm)", value=1000, step=100, disabled=not enable_extrude)

    st.sidebar.divider()
    epsilon_val = st.sidebar.slider("선 단순화", 0.001, 0.020, 0.005, format="%.3f")
    min_area_val = st.sidebar.slider("잡티 제거", 0, 50, 5)

    uploaded_file = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="원본", use_container_width=True)

        with st.spinner('변환 중...'):
            doc, l_count, px, py, pz, texts = process_image(
                uploaded_file.read(), real_width, wall_height, epsilon_val, min_area_val, enable_extrude
            )

            fig = go.Figure()
            line_color = '#00ff00' if enable_extrude else '#ffffff' 
            fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines', line=dict(color=line_color, width=2), name='Lines'))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False, backgroundcolor="#222"),
                    yaxis=dict(visible=False, backgroundcolor="#222"),
                    zaxis=dict(visible=False, backgroundcolor="#222"),
                    bgcolor='#222', aspectmode='data'
                ),
                paper_bgcolor='#222', margin=dict(l=0, r=0, t=0, b=0), height=600, showlegend=False
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                doc.saveas(tmp.name)
                tmp_path = tmp.name

        with col2:
            st.plotly_chart(fig, use_container_width=True)
            if enable_extrude:
                st.info(f"ℹ️ 현재 '벽 세우기' 모드입니다. (높이: {wall_height}mm)")
            else:
                st.success("ℹ️ '단순 선 따기' 모드입니다. 그림을 있는 그대로 벡터화했습니다.")

        with open(tmp_path, "rb") as file:
            st.download_button("📥 DXF 다운로드", file, "result.dxf", "image/vnd.dxf")
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()