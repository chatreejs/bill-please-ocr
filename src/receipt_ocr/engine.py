import math
import threading
from typing import Protocol

import numpy as np

from receipt_ocr.config import Config
from receipt_ocr.lines import Word
from receipt_ocr.log import adopt_library_loggers, get_logger

log = get_logger("ocr.engine")


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> list[Word]:
        """Detect and recognize text in a BGR image."""
        ...


def _top_edge_angle(poly: np.ndarray) -> float:
    (x0, y0), (x1, y1) = poly[0], poly[1]
    return math.atan2(float(y1 - y0), float(x1 - x0)) if x1 != x0 else 0.0


class PaddleOcrEngine:
    """PaddleOCR 3.x text detection and Thai recognition. The document orientation, unwarping and
    text-line orientation models stay off, because the EXIF rotation in image.py covers phone photos."""

    def __init__(self, config: Config):
        # Imported here so tests and the app module load without Paddle
        from paddleocr import PaddleOCR

        adopt_library_loggers()

        self._min_confidence = config.min_confidence
        self._ocr = PaddleOCR(
            text_detection_model_name=config.det_model,
            text_recognition_model_name=config.rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=config.device,
            cpu_threads=config.cpu_threads,
        )
        # The Paddle predictor is not safe to call from two threads at once
        self._lock = threading.Lock()
        log.info(
            "Loaded OCR models det=%s rec=%s device=%s", config.det_model, config.rec_model, config.device
        )

    def recognize(self, image: np.ndarray) -> list[Word]:
        with self._lock:
            results = self._ocr.predict(image)
        if not results:
            return []
        result = results[0]
        return [
            Word(
                text=text,
                confidence=float(score),
                box=tuple(int(v) for v in box),
                angle=_top_edge_angle(poly),
            )
            for text, score, box, poly in zip(
                result["rec_texts"],
                result["rec_scores"],
                result["rec_boxes"],
                result["rec_polys"],
                strict=True,
            )
            if float(score) >= self._min_confidence
        ]
