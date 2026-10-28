"""Runs the real PaddleOCR models on the synthetic receipt. Slow and needs the models,
so it only runs with `uv run pytest -m paddle`."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from receipt_ocr.config import Config  # noqa: E402
from receipt_ocr.image import decode_image  # noqa: E402
from receipt_ocr.lines import build_lines  # noqa: E402

pytestmark = pytest.mark.paddle

FIXTURES = Path(__file__).parent / "fixtures"


def test_reads_thai_and_english_receipt():
    from receipt_ocr.engine import PaddleOcrEngine

    engine = PaddleOcrEngine(Config())
    pixels = decode_image((FIXTURES / "receipt.jpg").read_bytes(), 1600)
    lines = [" ".join(line.text.split()) for line in build_lines(engine.recognize(pixels))]

    for expected in (
        "ผัดไทยกุ้งสด 2 240.00",
        "Pad Krapow Moo 1 90.00",
        "Singha Beer 2 180.00",
        "ส่วนลดสมาชิก -100.00",
        "รวมทั้งสิ้น 853.33",
    ):
        assert expected in lines
