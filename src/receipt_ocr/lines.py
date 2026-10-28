"""Line reconstruction: turn separate OCR boxes into receipt rows.

OCR returns one box per text run, e.g. "Pad Thai", "2" and "240.00" for one row. The LLM
needs them back on one line with the columns lined up:

  Pad Thai                 2      240.00
  ส่วนลดสมาชิก                     -200.00
"""

import math
import re
import statistics
import unicodedata
from dataclasses import dataclass

Box = tuple[int, int, int, int]  # x0, y0, x1, y1

# Amounts like 240.00, -100.00, 1,250 or 10%. Receipts right-align them.
NUMBER = re.compile(r"^[-+(]?[\d,]*\.?\d+%?\)?$")

# Photos are rarely tilted more than this. A larger estimate comes from a bad box, so ignore it.
MAX_SKEW = math.radians(15)


@dataclass(frozen=True)
class Word:
    text: str
    confidence: float
    box: Box
    # Angle of the box's top edge in radians, from the OCR polygon. 0 for an axis-aligned box.
    angle: float = 0.0

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class Line:
    text: str
    confidence: float
    box: Box


def visible_length(text: str) -> int:
    """Character cells the text takes up. Thai vowel and tone marks sit above or below
    the consonant and take no cell of their own."""
    return sum(
        1
        for char in text
        if not unicodedata.combining(char) and unicodedata.category(char) not in ("Mn", "Me", "Cf")
    )


def estimate_skew(words: list[Word]) -> float:
    """Median tilt of the wide boxes. Short boxes such as "2" give noisy angles."""
    angles = [w.angle for w in words if w.width >= 2 * w.height and abs(w.angle) <= MAX_SKEW]
    return statistics.median(angles) if angles else 0.0


def _rotated_center(word: Word, skew: float) -> tuple[float, float]:
    """Box centre rotated by -skew, so the rows of a tilted photo become horizontal."""
    cx = (word.box[0] + word.box[2]) / 2
    cy = (word.box[1] + word.box[3]) / 2
    cos, sin = math.cos(skew), math.sin(skew)
    return cx * cos + cy * sin, -cx * sin + cy * cos


def group_rows(words: list[Word]) -> list[list[tuple[Word, float]]]:
    """Group words whose vertical centres are within half a text height of each other.

    Returns each row sorted left to right, with each word's left edge after deskewing."""
    skew = estimate_skew(words)
    centers = {id(w): _rotated_center(w, skew) for w in words}
    rows: list[list[Word]] = []
    row_y = row_height = 0.0

    for word in sorted(words, key=lambda w: centers[id(w)][1]):
        y = centers[id(word)][1]
        height = max(word.height, 1)
        if rows and abs(y - row_y) <= 0.5 * min(height, row_height):
            row = rows[-1]
            row.append(word)
            row_y = statistics.fmean(centers[id(w)][1] for w in row)
            row_height = statistics.median(max(w.height, 1) for w in row)
        else:
            rows.append([word])
            row_y, row_height = y, height

    return [
        sorted(((w, centers[id(w)][0] - w.width / 2) for w in row), key=lambda item: item[1]) for row in rows
    ]


def char_width(words: list[Word]) -> float:
    widths = [w.width / visible_length(w.text) for w in words if visible_length(w.text) > 0]
    return max(statistics.median(widths), 1.0) if widths else 1.0


def build_lines(words: list[Word]) -> list[Line]:
    """Rows from top to bottom. Each word goes to the text column matching its x position, so
    columns line up across rows. Numbers are placed by their right edge, like on the receipt."""
    words = [w for w in words if w.text.strip()]
    if not words:
        return []

    cell = char_width(words)
    rows = group_rows(words)
    left = min(x for row in rows for _, x in row)
    lines = []
    for row in rows:
        text = ""
        for word, x in row:
            length = visible_length(word.text.strip())
            if NUMBER.match(word.text.strip()):
                column = round((x + word.width - left) / cell) - length
            else:
                column = round((x - left) / cell)
            gap = column - visible_length(text)
            text += " " * max(gap, 1 if text else 0) + word.text.strip()
        row_words = [w for w, _ in row]
        lines.append(
            Line(
                text=text,
                confidence=round(statistics.fmean(w.confidence for w in row_words), 4),
                box=(
                    min(w.box[0] for w in row_words),
                    min(w.box[1] for w in row_words),
                    max(w.box[2] for w in row_words),
                    max(w.box[3] for w in row_words),
                ),
            )
        )
    return lines
