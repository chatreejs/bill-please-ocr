"""Scores OCR lines against the answers in fixtures/receipts. Used by test_paddle.py.

Each receipt is an image (`<name>.jpg`, .png, .webp, .heic or .heif) next to `<name>.json`,
the answer of the bill:

    {
      "items": [{"name": "ผัดไทยกุ้งสด", "qty": 2, "amount": 240.00}],
      "total": 853.33
    }

An item's candidates are the lines that hold its amount, and its qty when the answer has one.
Its score is how close the name comes to the best stretch of a candidate line, from 0 to 1, so
a dropped Thai tone mark costs a little instead of failing the item.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

RECEIPTS = Path(__file__).parent / "fixtures" / "receipts"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# An item passes at this score: about one wrong mark in every five characters
MIN_SIMILARITY = 0.8


@dataclass(frozen=True)
class ItemScore:
    name: str
    similarity: float
    line: str | None  # the best candidate line, None when no line holds the amount

    @property
    def passed(self) -> bool:
        return self.similarity >= MIN_SIMILARITY


@dataclass(frozen=True)
class ReceiptScore:
    items: list[ItemScore]
    total_found: bool

    @property
    def passed(self) -> bool:
        return self.total_found and all(item.passed for item in self.items)

    @property
    def mean_similarity(self) -> float:
        return sum(item.similarity for item in self.items) / len(self.items) if self.items else 1.0


def receipts() -> list[Path]:
    return sorted(p for p in RECEIPTS.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def load_answer(image: Path) -> dict:
    return json.loads(image.with_suffix(".json").read_text())


def numbers(line: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUMBER.findall(line)]


def squash(text: str) -> str:
    return "".join(text.split())


def name_similarity(name: str, line: str) -> float:
    """1 minus the edit distance from `name` to its closest substring of `line`, per character
    of `name`. Spaces don't count."""
    name, line = squash(name), squash(line)
    if not name:
        return 1.0
    # Edit distance where the match may start and end anywhere in the line
    previous = [0] * (len(line) + 1)
    for i, a in enumerate(name, 1):
        current = [i]
        for j, b in enumerate(line, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return max(0.0, 1 - min(previous) / len(name))


def score_item(item: dict, lines: list[str]) -> ItemScore:
    wanted = [item["amount"]] + ([item["qty"]] if "qty" in item else [])
    candidates = [line for line in lines if all(w in numbers(line) for w in wanted)]
    if not candidates:
        return ItemScore(item["name"], 0.0, None)
    line = max(candidates, key=lambda line: name_similarity(item["name"], line))
    return ItemScore(item["name"], name_similarity(item["name"], line), line)


def score_receipt(answer: dict, lines: list[str]) -> ReceiptScore:
    items = [score_item(item, lines) for item in answer.get("items", [])]
    total_found = "total" not in answer or any(answer["total"] in numbers(line) for line in lines)
    return ReceiptScore(items, total_found)
