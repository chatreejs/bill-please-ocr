from receipts import ItemScore, name_similarity, numbers, score_receipt


def test_numbers_drop_thousands_separators():
    assert numbers("2.00 I11.ปลาหมึก 1,590.00") == [2.0, 11.0, 1590.0]
    assert numbers("ส่วนลด -100.00") == [-100.0]


def test_name_similarity_finds_the_name_inside_the_line():
    assert name_similarity("Pasta with cheese", "Pasta with cheese   $71.00") == 1.0
    assert name_similarity("ผัดไทย กุ้งสด", "1.00 M07.ผัดไทยกุ้งสด 265.00") == 1.0


def test_name_similarity_charges_each_wrong_mark():
    # O read as 0: one edit in 17 characters
    assert round(name_similarity("O06.ข้าวไข่ข้นหมู", "1.00 006.ข้าวไข่ข้นหมู 275.00"), 3) == round(1 - 1 / 17, 3)
    assert name_similarity("ขนม", "Pad Thai") == 0.0


def test_score_receipt_needs_the_amount_on_the_line():
    lines = ["Pad Thai 120.00", "Beer 90.00", "TOTAL 210.00"]
    answer = {"items": [{"name": "Pad Thai", "amount": 120}, {"name": "Beer", "amount": 95}], "total": 210}
    score = score_receipt(answer, lines)

    assert score.items == [ItemScore("Pad Thai", 1.0, "Pad Thai 120.00"), ItemScore("Beer", 0.0, None)]
    assert score.total_found
    assert not score.passed
    assert score.mean_similarity == 0.5


def test_score_receipt_checks_qty_when_given():
    score = score_receipt({"items": [{"name": "Beer", "qty": 2, "amount": 90}]}, ["1 Beer 90.00"])
    assert score.items[0].line is None
