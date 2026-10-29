"""Runs the real PaddleOCR models on the receipts in fixtures/receipts and scores the lines
against their answers (see receipts.py). Slow and needs the models, so it only runs with
`uv run pytest -m paddle`."""

import os
from pathlib import Path

import pytest
from receipts import MIN_SIMILARITY, load_answer, receipts, score_receipt

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from receipt_ocr.config import Config  # noqa: E402
from receipt_ocr.image import decode_image  # noqa: E402
from receipt_ocr.lines import build_lines  # noqa: E402

pytestmark = pytest.mark.paddle


@pytest.fixture(scope="module")
def engine():
    from receipt_ocr.engine import PaddleOcrEngine

    return PaddleOcrEngine(Config())


@pytest.mark.parametrize("image", receipts(), ids=lambda p: p.name)
def test_reads_receipt(engine, image: Path):
    pixels = decode_image(image.read_bytes(), Config().max_image_side)
    lines = [line.text for line in build_lines(engine.recognize(pixels))]
    score = score_receipt(load_answer(image), lines)

    report = [f"mean similarity {score.mean_similarity:.3f}, items pass at {MIN_SIMILARITY}"]
    report += [
        f"{item.similarity:.2f} {item.name!r} <- {item.line!r}" for item in score.items if not item.passed
    ]
    if not score.total_found:
        report.append("total not found")
    assert score.passed, "\n".join(report) + "\n\nlines:\n" + "\n".join(lines)
