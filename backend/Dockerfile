# =============================================================================
# Urban Intelligence Platform - Backend Dockerfile
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and spatial geometry
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY . .

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Render/CPU inference safety defaults. Keep a single model/pass in memory and
# cap CPU math libraries so concurrent native threads do not multiply RAM use.
# These can still be overridden in the Render service environment.
ENV TORCH_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV ROAD_AI_USE_ONNX=0
ENV ROAD_AI_USE_PRETRAINED_POTHOLE=0
ENV ROAD_AI_TILED=0
ENV ROAD_AI_IMGSZ=512
ENV ROAD_AI_MAX_SIDE=768
ENV ROAD_AI_NMS_IOU=0.60
ENV ROAD_AI_POTHOLE_MIN_CONF=0.12
ENV ANPR_USE_EASYOCR=0
ENV ANPR_IMGSZ=320
ENV ANPR_MAX_IMAGE_SIDE=1280
ENV ANPR_RELEASE_AFTER_REQUEST=1

EXPOSE 8000

# One worker is intentional: every worker would load its own PyTorch/YOLO model.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
