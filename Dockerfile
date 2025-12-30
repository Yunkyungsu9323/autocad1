FROM python:3.11-slim

# --- 시스템 라이브러리 (OpenCV / OCR 필수) ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# --- 작업 디렉토리 ---
WORKDIR /app

# --- Python 패키지 설치 ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 소스 복사 ---
COPY . .

# --- Streamlit 포트 ---
EXPOSE 8501

# --- 실행 ---
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
