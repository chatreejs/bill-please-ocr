# Converts PP-OCRv5 to ONNX and runs it on ONNX Runtime with CUDA 12 on an NVIDIA GPU.

# paddle2onnx supports Python up to 3.12, so the conversion runs in its own stage
FROM python:3.12-slim AS convert
# paddle2onnx loads Paddle to read the models, and Paddle needs libgomp1
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir paddlepaddle==3.0.0 paddle2onnx==2.1.0 "pyyaml>=6.0" setuptools packaging
# Pin the Hugging Face revisions, so a rebuild converts the same weights
ARG OCR_DET_MODEL=PP-OCRv5_mobile_det
ARG OCR_DET_REVISION=0d63e78e2b680928f6b1747d76a08db6e645efb7
ARG OCR_REC_MODEL=th_PP-OCRv5_mobile_rec
ARG OCR_REC_REVISION=9e25080455925d3943f5db885fddc79db6a07ca3
COPY scripts/convert_onnx.py /convert_onnx.py
RUN python /convert_onnx.py /opt/onnx \
  "${OCR_DET_MODEL}=${OCR_DET_REVISION}" "${OCR_REC_MODEL}=${OCR_REC_REVISION}"

FROM python:3.13.18-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
# Installs onnxruntime-gpu with the CUDA runtime, cuBLAS, cuFFT and cuDNN wheels
RUN uv sync --frozen --no-dev --no-install-project
# Deletes OpenCV's Haar cascades, which the service never loads
RUN rm -rf .venv/lib/python3.13/site-packages/cv2/data
COPY src ./src
RUN uv sync --frozen --no-dev
COPY --from=convert /opt/onnx /opt/onnx

# The build has no GPU, so the warm-up runs the converted models on ONNX Runtime's CPU provider.
# If a converted model does not run, the build fails.
ARG OCR_DET_MODEL=PP-OCRv5_mobile_det
ARG OCR_REC_MODEL=th_PP-OCRv5_mobile_rec
RUN OCR_DEVICE=cpu OCR_ONNX_MODEL_DIR=/opt/onnx \
  OCR_DET_MODEL=${OCR_DET_MODEL} OCR_REC_MODEL=${OCR_REC_MODEL} \
  .venv/bin/python -c "from receipt_ocr.config import load_config; from receipt_ocr.engine import OnnxOcrEngine; import numpy as np; OnnxOcrEngine(load_config()).recognize(np.full((64, 256, 3), 255, dtype=np.uint8))"

FROM python:3.13.18-slim
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
ARG OCR_DET_MODEL=PP-OCRv5_mobile_det
ARG OCR_REC_MODEL=th_PP-OCRv5_mobile_rec
ENV PATH=/app/.venv/bin:$PATH \
  PYTHONUNBUFFERED=1 \
  OCR_ONNX_MODEL_DIR=/opt/onnx \
  OCR_DEVICE=gpu:0 \
  OCR_DET_MODEL=${OCR_DET_MODEL} \
  OCR_REC_MODEL=${OCR_REC_MODEL} \
  PORT=8000
COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/onnx /opt/onnx
COPY --from=build /app/src /app/src
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1
CMD ["python", "-m", "receipt_ocr"]
