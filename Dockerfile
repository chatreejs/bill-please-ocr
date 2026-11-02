# Keep the Python version in sync with .tool-versions
FROM python:3.13.18-slim AS base
# libgomp1 for Paddle. OpenCV is the headless build (see shims/opencv-contrib-python), so no libgl1.
# The build stage needs it too, because it imports PaddleOCR to download the models.
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
COPY shims ./shims
RUN uv sync --frozen --no-dev --no-install-project
# Delete files we never load: C++ headers, the OpenVINO backend (we run Paddle Inference), test suites
# and OpenCV's Haar cascades. The warm-up below fails the build if anything here turns out to be needed.
RUN cd .venv/lib/python3.13/site-packages \
  && rm -rf paddle/include paddle/libs/libopenvino* pandas/tests cv2/data
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the models into the image so pods start without internet access.
# Loading the engine downloads them from Hugging Face into PADDLE_PDX_CACHE_HOME. The warm-up predict
# fails the build, not the pod, when the Paddle runtime cannot run the models on this CPU.
ARG OCR_DET_MODEL=PP-OCRv5_mobile_det
ARG OCR_REC_MODEL=th_PP-OCRv5_mobile_rec
ENV PADDLE_PDX_CACHE_HOME=/opt/paddlex OCR_DET_MODEL=${OCR_DET_MODEL} OCR_REC_MODEL=${OCR_REC_MODEL}
RUN .venv/bin/python -c "from receipt_ocr.config import load_config; from receipt_ocr.engine import PaddleOcrEngine; import numpy as np; PaddleOcrEngine(load_config()).recognize(np.full((64, 256, 3), 255, dtype=np.uint8))"

FROM base
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
ARG OCR_DET_MODEL=PP-OCRv5_mobile_det
ARG OCR_REC_MODEL=th_PP-OCRv5_mobile_rec
ENV PATH=/app/.venv/bin:$PATH \
  PYTHONUNBUFFERED=1 \
  PADDLE_PDX_CACHE_HOME=/opt/paddlex \
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  OCR_DET_MODEL=${OCR_DET_MODEL} \
  OCR_REC_MODEL=${OCR_REC_MODEL} \
  PORT=8000
# Separate layers, largest and least often changed first, so a code change pushes and pulls only src
COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/paddlex /opt/paddlex
COPY --from=build /app/src /app/src
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1
CMD ["python", "-m", "receipt_ocr"]
