# Keep the Python version in sync with .tool-versions
FROM python:3.13.18-slim AS base
# libgomp1 for Paddle. libgl1 and libglib2.0-0 for opencv-contrib-python, which PaddleX pins (not the headless build).
# The build stage needs them too, because it imports PaddleOCR to download the models.
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
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
COPY --from=build /app /app
COPY --from=build /opt/paddlex /opt/paddlex
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1
CMD ["python", "-m", "receipt_ocr"]
