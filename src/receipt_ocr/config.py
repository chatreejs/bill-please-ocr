import os
from dataclasses import dataclass, field

from receipt_ocr.log import parse_log_level

# Same list as SUPPORTED_IMAGE_TYPES in bill-please-mcp src/schemas/receipt.ts
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"})


def _int(name: str, fallback: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return fallback


def _bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return fallback


def _float(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return fallback


@dataclass(frozen=True)
class Config:
    port: int = 8000
    log_level: str = "INFO"
    max_image_bytes: int = 5 * 1024 * 1024
    # The web app resizes photos to at most 1600 px. HEIC originals arrive at full size, and the
    # service scales them down to this.
    max_image_side: int = 1600
    # On the first three bills the defaults beat wider boxes, a 2400 px image and PP-OCRv5_server_det
    det_model: str = "PP-OCRv5_mobile_det"
    rec_model: str = "th_PP-OCRv5_mobile_rec"
    # The service drops words below this recognition score, such as stray marks and stains
    min_confidence: float = 0.5
    # The Docker build converts the models to ONNX here
    onnx_model_dir: str = "/opt/onnx"
    # gpu:N runs on CUDA device N. The image build sets cpu for its warm-up, because it has no GPU.
    device: str = "gpu:0"
    # ONNX Runtime's CUDA arena cap. llm-server shares the 4 GB T400 and leaves about 1.2 GB.
    gpu_mem_limit_mb: int = 768
    # Number of ONNX Runtime threads for the CPU work between the CUDA kernels
    cpu_threads: int = 2
    supported_types: frozenset[str] = field(default=SUPPORTED_IMAGE_TYPES)


def load_config() -> Config:
    return Config(
        port=_int("PORT", 8000),
        log_level=parse_log_level(os.environ.get("LOG_LEVEL")),
        max_image_bytes=_int("MAX_IMAGE_BYTES", 5 * 1024 * 1024),
        max_image_side=_int("MAX_IMAGE_SIDE", 1600),
        det_model=os.environ.get("OCR_DET_MODEL", "PP-OCRv5_mobile_det"),
        rec_model=os.environ.get("OCR_REC_MODEL", "th_PP-OCRv5_mobile_rec"),
        min_confidence=_float("OCR_MIN_CONFIDENCE", 0.5),
        onnx_model_dir=os.environ.get("OCR_ONNX_MODEL_DIR", "/opt/onnx"),
        device=os.environ.get("OCR_DEVICE", "gpu:0"),
        gpu_mem_limit_mb=_int("OCR_GPU_MEM_LIMIT_MB", 768),
        cpu_threads=_int("OCR_CPU_THREADS", 2),
    )
