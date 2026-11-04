"""Runs the real ONNX models on the receipts in fixtures/receipts and scores the lines against their
answers, with the scoring in receipts.py. The test is slow and needs the converted models in
OCR_ONNX_MODEL_DIR (default /opt/onnx), so it runs only with `uv run pytest -m models`. It runs on the
CPU, so it works without a GPU."""

import os
from pathlib import Path

import pytest
from receipts import MIN_SIMILARITY, load_answer, receipts, score_receipt

from receipt_ocr.config import Config
from receipt_ocr.image import decode_image
from receipt_ocr.lines import build_lines

pytestmark = pytest.mark.models

MODEL_DIR = os.environ.get("OCR_ONNX_MODEL_DIR", "/opt/onnx")


@pytest.fixture(scope="module")
def engine():
    config = Config(device="cpu", onnx_model_dir=MODEL_DIR)
    for model in (config.det_model, config.rec_model):
        if not (Path(MODEL_DIR) / model / "inference.onnx").is_file():
            pytest.skip(f"no converted {model} in {MODEL_DIR}")
    pytest.importorskip("onnxruntime")
    from receipt_ocr.engine import OnnxOcrEngine

    return OnnxOcrEngine(config)


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
