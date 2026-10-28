import io

import numpy as np
import pi_heif
from PIL import Image, ImageOps, UnidentifiedImageError

pi_heif.register_heif_opener()

# A 1600 px receipt is ~2.5 MP. This rejects decompression bombs long before memory runs out.
Image.MAX_IMAGE_PIXELS = 60_000_000


class InvalidImageError(ValueError):
    """The bytes are not an image Pillow can decode."""


def decode_image(data: bytes, max_side: int) -> np.ndarray:
    """Decode upload bytes into a BGR array, upright and at most `max_side` px on the long side."""
    try:
        with Image.open(io.BytesIO(data)) as opened:
            # Phone photos are often stored sideways with an EXIF orientation tag
            image = ImageOps.exif_transpose(opened)
            image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImageError(str(error)) from error

    if max(image.size) > max_side:
        # Downscale only. Upscaling a small photo adds no detail for the recognizer.
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    # PaddleOCR expects OpenCV channel order
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
