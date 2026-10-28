"""Regenerate the synthetic receipt fixtures.

  1. Render receipt.html with headless Chrome (Pillow without raqm mangles Thai marks):
       "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
         --hide-scrollbars --window-size=584,760 --screenshot=$PWD/receipt.png file://$PWD/receipt.html
  2. uv run python tests/fixtures/make_fixtures.py

Writes receipt.jpg (tilted, tinted, JPEG-compressed like a phone photo) and receipt.words.json
(what PaddleOCR recognizes in it, the input for the line reconstruction tests).
"""

import json
import os
from pathlib import Path

from PIL import Image, ImageFilter

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

HERE = Path(__file__).parent


def make_photo() -> Path:
    image = Image.open(HERE / "receipt.png").convert("RGB")
    # Slightly off-white thermal paper, tilted 2 degrees, a little soft
    tint = Image.new("RGB", image.size, (238, 234, 226))
    image = Image.composite(image, tint, image.convert("L").point(lambda v: 255 - v))
    image = image.rotate(2, expand=True, fillcolor=(90, 80, 70), resample=Image.Resampling.BICUBIC)
    image = image.filter(ImageFilter.GaussianBlur(0.6))
    target = HERE / "receipt.jpg"
    image.save(target, quality=85)
    return target


def record_words(photo: Path) -> None:
    from receipt_ocr.config import Config
    from receipt_ocr.engine import PaddleOcrEngine
    from receipt_ocr.image import decode_image

    engine = PaddleOcrEngine(Config())
    words = engine.recognize(decode_image(photo.read_bytes(), 1600))
    data = [
        {"text": w.text, "confidence": round(w.confidence, 4), "box": list(w.box), "angle": round(w.angle, 5)}
        for w in words
    ]
    (HERE / "receipt.words.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    record_words(make_photo())
