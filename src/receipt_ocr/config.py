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
    # The web resizes to <= 1600 px. HEIC originals arrive full size and get scaled down to this.
    max_image_side: int = 1600
    # On the synthetic receipt, mobile detection took 3.2 s and PP-OCRv5_server_det 5.1 s on a Mac CPU
    det_model: str = "PP-OCRv5_mobile_det"
    rec_model: str = "th_PP-OCRv5_mobile_rec"
    # Words below this recognition score are dropped (stray marks, stains)
    min_confidence: float = 0.5
    device: str = "cpu"
    cpu_threads: int = 2
    # oneDNN on x86 fails in Paddle 3.3 with "ConvertPirAttribute2RuntimeAttribute not support
    # [pir::ArrayAttribute<pir::DoubleAttribute>]" on the first predict, so it stays off
    enable_mkldnn: bool = False
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
        device=os.environ.get("OCR_DEVICE", "cpu"),
        cpu_threads=_int("OCR_CPU_THREADS", 2),
        enable_mkldnn=_bool("OCR_ENABLE_MKLDNN", False),
    )
