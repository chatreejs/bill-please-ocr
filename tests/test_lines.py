import math

from receipt_ocr.lines import Word, build_lines, estimate_skew, group_rows, visible_length


def word(text: str, x0: int, y0: int, x1: int, y1: int, angle: float = 0.0) -> Word:
    return Word(text=text, confidence=0.9, box=(x0, y0, x1, y1), angle=angle)


def end_column(line: str, token: str) -> int:
    """Visible cells up to the end of `token` in `line`."""
    return visible_length(line[: line.index(token) + len(token)])


def test_visible_length_skips_thai_marks():
    # ก ุ ้ ง: the vowel below and the tone mark above take no cell
    assert visible_length("กุ้ง") == 2
    assert visible_length("ผัดไทยกุ้งสด") == 9
    assert visible_length("Pad Thai") == 8


def test_groups_boxes_on_the_same_row_left_to_right():
    words = [
        word("240.00", 400, 102, 480, 128),
        word("Pad Thai", 20, 100, 140, 130),
        word("2", 300, 101, 312, 129),
        word("Total", 20, 160, 90, 190),
    ]
    rows = group_rows(words)
    assert [[w.text for w, _ in row] for row in rows] == [["Pad Thai", "2", "240.00"], ["Total"]]


def test_separates_close_rows():
    words = [word("A", 20, 100, 60, 124), word("B", 20, 128, 60, 152)]
    assert len(group_rows(words)) == 2


def test_tilted_photo_keeps_rows_together():
    # 4 degrees of tilt: the right end of a row sits ~30 px lower than the left end
    tilt = math.radians(4)
    words = []
    for row, y in enumerate((100, 160, 220)):
        for x, text in ((20, f"item{row}"), (420, "1"), (520, f"{row}0.00")):
            dy = round(x * math.tan(tilt))
            words.append(word(text, x, y + dy, x + 80, y + dy + 26, angle=tilt))
    assert abs(estimate_skew(words) - tilt) < 1e-6
    rows = group_rows(words)
    assert [[w.text for w, _ in row] for row in rows] == [
        ["item0", "1", "00.00"],
        ["item1", "1", "10.00"],
        ["item2", "1", "20.00"],
    ]


def test_skew_ignores_short_and_extreme_boxes():
    words = [word("2", 0, 0, 10, 20, angle=0.5), word("Pad Thai", 0, 0, 100, 20, angle=0.02)]
    assert estimate_skew(words) == 0.02
    assert estimate_skew([word("Pad Thai", 0, 0, 100, 20, angle=math.radians(30))]) == 0.0


def test_right_aligns_numbers():
    # 10 px per character; both amounts end at x=500
    lines = build_lines(
        [
            word("Pad Thai", 0, 0, 80, 20),
            word("240.00", 440, 0, 500, 20),
            word("Discount", 0, 40, 80, 60),
            word("-1,200.00", 410, 40, 500, 60),
        ]
    )
    assert [line.text.split() for line in lines] == [["Pad", "Thai", "240.00"], ["Discount", "-1,200.00"]]
    assert end_column(lines[0].text, "240.00") == end_column(lines[1].text, "-1,200.00")


def test_line_box_and_confidence():
    lines = build_lines(
        [
            Word("Pad Thai", 0.8, (10, 100, 90, 120)),
            Word("120.00", 1.0, (300, 98, 360, 122)),
        ]
    )
    assert lines[0].box == (10, 98, 360, 122)
    assert lines[0].confidence == 0.9


def test_empty_input():
    assert build_lines([]) == []
    assert build_lines([word("  ", 0, 0, 10, 10)]) == []
