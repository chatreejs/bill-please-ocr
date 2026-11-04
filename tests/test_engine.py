import numpy as np
import pytest

from receipt_ocr.engine import (
    DetSettings,
    OnnxPipeline,
    RecSettings,
    crop,
    ctc_decode,
    det_boxes,
    det_normalize,
    det_resize,
    parse_gpu_device,
    rec_characters,
    rec_normalize,
    sort_boxes,
    to_batch,
)

CHARACTERS = ["blank", "a", "b", " "]


def test_det_resize_scales_a_small_side_up_to_64_and_rounds_to_32():
    resized, ratio_h, ratio_w = det_resize(np.zeros((50, 100, 3), np.uint8), DetSettings())
    assert resized.shape == (64, 128, 3)
    assert (ratio_h, ratio_w) == (64 / 50, 128 / 100)


def test_det_resize_keeps_large_images_and_rounds_half_to_even():
    resized, _, _ = det_resize(np.zeros((1200, 1600, 3), np.uint8), DetSettings())
    # 1200 is 37.5 times 32, which rounds up to 38 times 32
    assert resized.shape == (1216, 1600, 3)


def test_det_resize_caps_the_long_side():
    resized, _, _ = det_resize(np.zeros((100, 9000, 3), np.uint8), DetSettings())
    assert max(resized.shape[:2]) <= 4000


def test_det_resize_pads_tiny_images():
    resized, _, _ = det_resize(np.zeros((10, 20, 3), np.uint8), DetSettings())
    assert resized.shape[0] % 32 == 0 and resized.shape[1] % 32 == 0


def test_det_normalize_is_nchw_with_imagenet_stats():
    out = det_normalize(np.full((2, 3, 3), 255, np.uint8))
    assert out.shape == (1, 3, 2, 3) and out.dtype == np.float32
    assert out[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, rel=1e-5)


def test_det_boxes_expands_a_text_region_into_a_quad():
    pred = np.zeros((160, 320), np.float32)
    pred[30:50, 20:120] = 0.9
    [box] = det_boxes(pred, 320, 640, DetSettings())
    assert box.dtype == np.int16 and box.shape == (4, 2)
    # det_boxes scales the box 2x back to the source and grows it past the region by the unclip ratio
    assert box[:, 0].min() < 40 and box[:, 0].max() > 238
    assert box[:, 1].min() < 60 and box[:, 1].max() > 98


def test_det_boxes_drops_weak_and_tiny_regions():
    pred = np.zeros((160, 320), np.float32)
    pred[30:50, 20:120] = 0.4  # above thresh, below box_thresh
    pred[100:102, 20:120] = 0.9  # thinner than min_size
    assert det_boxes(pred, 160, 320, DetSettings()) == []


def quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int16)


def test_sort_boxes_reads_rows_top_down_then_left_to_right():
    right, left, below = quad(200, 22, 300, 40), quad(20, 25, 150, 42), quad(20, 80, 90, 100)
    assert [b[0][0] for b in sort_boxes([below, right, left])] == [20, 200, 20]


def test_crop_straightens_and_turns_tall_boxes():
    image = np.zeros((100, 100, 3), np.uint8)
    assert crop(image, quad(10, 10, 60, 20)).shape[:2] == (10, 50)
    assert crop(image, quad(10, 10, 20, 60)).shape[:2] == (10, 50)


def test_rec_normalize_pads_to_the_wider_of_320_and_the_crop():
    settings = RecSettings()
    short = rec_normalize(np.full((24, 40, 3), 255, np.uint8), settings)
    assert short.shape == (3, 48, 320)
    assert short[0, 0, 0] == pytest.approx(1.0) and short[0, 0, 100] == 0
    long = rec_normalize(np.zeros((10, 200, 3), np.uint8), settings)
    assert long.shape == (3, 48, 960)
    assert rec_normalize(np.zeros((10, 1000, 3), np.uint8), settings).shape == (3, 48, 3200)


def test_to_batch_pads_to_the_widest():
    batch = to_batch([np.ones((3, 48, 320), np.float32), np.ones((3, 48, 400), np.float32)])
    assert batch.shape == (2, 3, 48, 400) and batch[0, 0, 0, 399] == 0


def one_hot(indices):
    out = np.full((len(indices), len(CHARACTERS)), 0.01, np.float32)
    out[np.arange(len(indices)), indices] = 0.9
    return out


def test_ctc_decode_merges_repeats_and_drops_blanks():
    pred = np.stack([one_hot([1, 1, 0, 1, 2, 3, 2]), one_hot([0] * 7)])
    assert ctc_decode(pred, CHARACTERS) == [("aab b", pytest.approx(0.9)), ("", 0.0)]


def test_rec_characters_add_blank_and_space():
    config = {"PostProcess": {"character_dict": ["x", "y"]}}
    assert rec_characters(config) == ["blank", "x", "y", " "]


@pytest.mark.parametrize(("device", "expected"), [("cpu", None), ("gpu", 0), ("gpu:1", 1), ("cuda:0", 0)])
def test_parse_gpu_device(device, expected):
    assert parse_gpu_device(device) == expected


def test_pipeline_crops_each_box_and_keeps_reading_order():
    image = np.zeros((160, 320, 3), np.uint8)
    image[20:40, 200:300] = 200
    image[22:42, 20:150] = 50
    pred = np.zeros((1, 1, 160, 320), np.float32)
    pred[0, 0, 20:40, 200:300] = 0.9
    pred[0, 0, 22:42, 20:150] = 0.9
    seen = []

    def run_det(batch):
        assert batch.shape == (1, 3, 160, 320)
        return pred

    def run_rec(batch):
        seen.append(batch.shape)
        # A bright crop reads "b" and a dark one "a"
        return np.stack([one_hot([2 if item[0, 24].max() > 0.3 else 1]) for item in batch])

    results = OnnxPipeline(run_det, run_rec, CHARACTERS).predict(image)
    assert [text for text, _, _ in results] == ["a", "b"]
    assert [int(poly[0][0]) < 20 for _, _, poly in results] == [True, False]
    assert seen == [(2, 3, 48, 320)]


def test_pipeline_returns_nothing_for_a_blank_image():
    def run_det(batch):
        return np.zeros((1, 1, *batch.shape[2:]), np.float32)

    def run_rec(batch):
        raise AssertionError("no crops to read")

    assert OnnxPipeline(run_det, run_rec, CHARACTERS).predict(np.zeros((64, 256, 3), np.uint8)) == []
