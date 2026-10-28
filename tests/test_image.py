import io

import pytest
from PIL import Image

from receipt_ocr.image import InvalidImageError, decode_image


def encode(image: Image.Image, fmt: str, **params) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **params)
    return buffer.getvalue()


def test_returns_bgr():
    pixels = decode_image(encode(Image.new("RGB", (40, 20), (255, 0, 0)), "PNG"), 1600)
    assert pixels.shape == (20, 40, 3)
    assert pixels[0, 0].tolist() == [0, 0, 255]


def test_applies_exif_rotation():
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, display rotated 90 degrees clockwise
    data = encode(Image.new("RGB", (40, 20)), "JPEG", exif=exif)
    assert decode_image(data, 1600).shape[:2] == (40, 20)


def test_downscales_but_never_upscales():
    big = encode(Image.new("RGB", (3200, 2400)), "JPEG")
    assert decode_image(big, 1600).shape[:2] == (1200, 1600)
    small = encode(Image.new("RGB", (800, 600)), "WEBP")
    assert decode_image(small, 1600).shape[:2] == (600, 800)


def test_rejects_non_images():
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image", 1600)
