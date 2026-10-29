"""Runs the real PaddleOCR models on the receipts in fixtures/receipts. Slow and needs the
models, so it only runs with `uv run pytest -m paddle`.

Each receipt is an image (`<name>.jpg`, .png, .webp, .heic or .heif) next to `<name>.json`,
the answer of the bill:

    {
      "items": [{"name": "ผัดไทยกุ้งสด", "qty": 2, "amount": 240.00}],
      "total": 853.33
    }

Each item must come back on one line that holds its name and its amount, and some line must
hold the total. `qty` is optional and checked on the item's line when present. Spaces don't
count when matching names.
"""

import json
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from receipt_ocr.config import Config  # noqa: E402
from receipt_ocr.image import decode_image  # noqa: E402
from receipt_ocr.lines import build_lines  # noqa: E402

pytestmark = pytest.mark.paddle

RECEIPTS = Path(__file__).parent / "fixtures" / "receipts"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def receipts() -> list[Path]:
    return sorted(p for p in RECEIPTS.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def numbers(line: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUMBER.findall(line)]


def squash(text: str) -> str:
    return "".join(text.split())


@pytest.fixture(scope="module")
def engine():
    from receipt_ocr.engine import PaddleOcrEngine

    return PaddleOcrEngine(Config())


@pytest.mark.parametrize("image", receipts(), ids=lambda p: p.name)
def test_reads_receipt(engine, image: Path):
    answer = json.loads(image.with_suffix(".json").read_text())
    pixels = decode_image(image.read_bytes(), Config().max_image_side)
    lines = [line.text for line in build_lines(engine.recognize(pixels))]

    missing = []
    for item in answer.get("items", []):
        name = squash(item["name"])
        wanted = [item["amount"]] + ([item["qty"]] if "qty" in item else [])
        if not any(name in squash(line) and all(w in numbers(line) for w in wanted) for line in lines):
            missing.append(item)
    if "total" in answer and not any(answer["total"] in numbers(line) for line in lines):
        missing.append({"total": answer["total"]})

    assert not missing, "not found:\n" + "\n".join(map(str, missing)) + "\n\nlines:\n" + "\n".join(lines)
