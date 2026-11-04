"""PP-OCRv5 detection and Thai recognition on ONNX Runtime with CUDA.

The pre- and post-processing follow PaddleX 3.7.2 step by step, so the converted models read the same
text as PaddleOCR did. On the dev receipts the item scores matched and about 1 character in 300
differed. The PaddleX source for each step:

- detection resize and normalize: models/text_detection (DetResizeForTest type 0, NormalizeImage)
- detection boxes: DBPostProcess.boxes_from_bitmap with box_type quad and score_mode fast
- box order and crops: pipelines/components/common (SortQuadBoxes, CropByPolys get_minarea_rect_crop)
- recognition: models/text_recognition (OCRReisizeNormImg, ToBatch, CTCLabelDecode)
- settings: configs/pipelines/OCR.yaml (limit_side_len 64, limit_type min, rec batch_size 6)

scripts/convert_onnx.py writes <model dir>/<model>/inference.onnx and config.json, the model's
inference.yml as JSON.
"""

import json
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pyclipper

from receipt_ocr.config import Config
from receipt_ocr.lines import Word
from receipt_ocr.log import get_logger

log = get_logger("ocr.engine")


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> list[Word]:
        """Detect and recognize text in a BGR image."""
        ...


def _top_edge_angle(poly: np.ndarray) -> float:
    (x0, y0), (x1, y1) = poly[0], poly[1]
    return math.atan2(float(y1 - y0), float(x1 - x0)) if x1 != x0 else 0.0


# Runs one model on an input batch and returns its first output
Run = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DetSettings:
    limit_side_len: int = 64
    limit_type: str = "min"
    max_side_limit: int = 4000
    thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5
    max_candidates: int = 1000
    min_size: int = 3


@dataclass(frozen=True)
class RecSettings:
    image_shape: tuple[int, int, int] = (3, 48, 320)
    batch_size: int = 6
    max_width: int = 3200


DET_MEAN = (0.485, 0.456, 0.406)
DET_STD = (0.229, 0.224, 0.225)


# Detection


def det_resize(image: np.ndarray, settings: DetSettings) -> tuple[np.ndarray, float, float]:
    """Resize to sides that are multiples of 32. Returns the image and the height and width ratios."""
    h, w = image.shape[:2]
    if h + w < 64:
        padded = np.zeros((max(32, h), max(32, w), image.shape[2]), np.uint8)
        padded[:h, :w] = image
        image = padded
        h, w = image.shape[:2]
    limit = settings.limit_side_len
    if settings.limit_type == "max":
        ratio = limit / max(h, w) if max(h, w) > limit else 1.0
    else:
        ratio = limit / min(h, w) if min(h, w) < limit else 1.0
    resize_h, resize_w = int(h * ratio), int(w * ratio)
    if max(resize_h, resize_w) > settings.max_side_limit:
        ratio = settings.max_side_limit / max(resize_h, resize_w)
        resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    if (resize_h, resize_w) == (h, w):
        return image, 1.0, 1.0
    return cv2.resize(image, (resize_w, resize_h)), resize_h / h, resize_w / w


def det_normalize(image: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC to a float32 NCHW batch of one. PaddleX applies the RGB mean to BGR channels."""
    alpha = np.array([(1 / 255) / s for s in DET_STD], np.float32)
    beta = np.array([-m / s for m, s in zip(DET_MEAN, DET_STD, strict=True)], np.float32)
    normalized = image.astype(np.float32) * alpha + beta
    return normalized.transpose(2, 0, 1)[np.newaxis]


def _mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour)
    points = sorted(cv2.boxPoints(rect), key=lambda p: p[0])
    top_left, bottom_left = (points[0], points[1]) if points[1][1] > points[0][1] else (points[1], points[0])
    top_right, bottom_right = (
        (points[2], points[3]) if points[3][1] > points[2][1] else (points[3], points[2])
    )
    return np.array([top_left, top_right, bottom_right, bottom_left]), min(rect[1])


def _box_score(pred: np.ndarray, box: np.ndarray) -> float:
    h, w = pred.shape
    box = box.copy()
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    box[:, 0] -= xmin
    box[:, 1] -= ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def _unclip(box: np.ndarray, ratio: float) -> np.ndarray:
    distance = cv2.contourArea(box) * ratio / cv2.arcLength(box, True)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    paths = offset.Execute(distance)
    try:
        return np.array(paths)
    except ValueError:
        # The paths have different lengths, so keep the first one, as PaddleX does
        return np.array(paths[0])


def det_boxes(pred: np.ndarray, src_h: int, src_w: int, settings: DetSettings) -> list[np.ndarray]:
    """Quad boxes, as int16 [4, 2] arrays in source pixels, from the DB probability map [H, W]."""
    bitmap = pred > settings.thresh
    height, width = bitmap.shape
    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours[: settings.max_candidates]:
        points, short_side = _mini_box(contour)
        if short_side < settings.min_size:
            continue
        if _box_score(pred, points.reshape(-1, 2)) < settings.box_thresh:
            continue
        box, short_side = _mini_box(_unclip(points, settings.unclip_ratio).reshape(-1, 1, 2))
        if short_side < settings.min_size + 2:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] * (src_w / width)), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] * (src_h / height)), 0, src_h)
        boxes.append(box.astype(np.int16))
    return boxes


def sort_boxes(boxes: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Top to bottom, then left to right for boxes whose tops are within 10 px."""
    ordered = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    for i in range(len(ordered) - 1):
        for j in range(i, -1, -1):
            same_row = abs(int(ordered[j + 1][0][1]) - int(ordered[j][0][1])) < 10
            if same_row and ordered[j + 1][0][0] < ordered[j][0][0]:
                ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
            else:
                break
    return ordered


def crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Straighten the box's minimum-area rectangle; turn tall crops on their side."""
    points, _ = _mini_box(np.asarray(box).astype(np.int32))
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), target)
    cropped = cv2.warpPerspective(
        image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    if cropped.size and cropped.shape[0] / cropped.shape[1] >= 1.5:
        cropped = np.rot90(cropped)
    return cropped


# Recognition


def rec_normalize(image: np.ndarray, settings: RecSettings) -> np.ndarray:
    """RGB uint8 crop to float32 CHW at the model height, padded with zeros on the right."""
    channels, target_h, target_w = settings.image_shape
    h, w = image.shape[:2]
    max_wh_ratio = max(target_w / target_h, w / h)
    width = int(target_h * max_wh_ratio)
    if width > settings.max_width:
        resized = cv2.resize(image, (settings.max_width, target_h))
        resized_w = width = settings.max_width
    else:
        resized_w = min(width, int(math.ceil(target_h * w / h)))
        resized = cv2.resize(image, (resized_w, target_h))
    resized = resized.astype(np.float32).transpose(2, 0, 1) / 255
    resized = (resized - 0.5) / 0.5
    padded = np.zeros((channels, target_h, width), np.float32)
    padded[:, :, :resized_w] = resized
    return padded


def to_batch(images: Sequence[np.ndarray]) -> np.ndarray:
    width = max(image.shape[2] for image in images)
    return np.stack([np.pad(image, ((0, 0), (0, 0), (0, width - image.shape[2]))) for image in images])


def ctc_decode(pred: np.ndarray, characters: Sequence[str]) -> list[tuple[str, float]]:
    """Greedy CTC over [N, T, C]. characters[0] is the blank."""
    indices = pred.argmax(axis=-1)
    probs = pred.max(axis=-1)
    results = []
    for index, prob in zip(indices, probs, strict=True):
        keep = np.ones(len(index), dtype=bool)
        keep[1:] = index[1:] != index[:-1]
        keep &= index != 0
        text = "".join(characters[i] for i in index[keep])
        kept = prob[keep]
        results.append((text, float(kept.mean()) if len(kept) else 0.0))
    return results


def rec_characters(rec_config: dict) -> list[str]:
    """The CTC output classes: blank, the model's dictionary, then a space."""
    return ["blank", *rec_config["PostProcess"]["character_dict"], " "]


class OnnxPipeline:
    """Detection, crops and recognition, with each model behind a plain function so tests can fake it."""

    def __init__(
        self,
        run_det: Run,
        run_rec: Run,
        characters: Sequence[str],
        det: DetSettings | None = None,
        rec: RecSettings | None = None,
    ):
        self._run_det = run_det
        self._run_rec = run_rec
        self._characters = list(characters)
        self._det = det or DetSettings()
        self._rec = rec or RecSettings()

    def predict(self, image: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        """Text, score and quad polygon for each box, in reading order. image is BGR."""
        src_h, src_w = image.shape[:2]
        resized, _, _ = det_resize(image, self._det)
        pred = self._run_det(det_normalize(resized))[0, 0]
        boxes = sort_boxes(det_boxes(pred, src_h, src_w, self._det))

        crops, polys = [], []
        for box in boxes:
            cropped = crop(image, box)
            if cropped.size and cropped.shape[0] and cropped.shape[1]:
                crops.append(cv2.cvtColor(np.ascontiguousarray(cropped), cv2.COLOR_BGR2RGB))
                polys.append(box)
        if not crops:
            return []

        # PaddleX sorts the crops by aspect ratio, so each batch pads to a similar width
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
        results: list[tuple[str, float] | None] = [None] * len(crops)
        size = self._rec.batch_size
        for start in range(0, len(order), size):
            chunk = order[start : start + size]
            batch = to_batch([rec_normalize(crops[i], self._rec) for i in chunk])
            for i, result in zip(chunk, ctc_decode(self._run_rec(batch), self._characters), strict=True):
                results[i] = result
        return [(text, score, poly) for (text, score), poly in zip(results, polys, strict=True)]


def parse_gpu_device(device: str) -> int | None:
    """'gpu' or 'gpu:1' to a CUDA device id; anything else runs on the CPU."""
    name, _, index = device.partition(":")
    if name.lower() not in {"gpu", "cuda"}:
        return None
    return int(index) if index else 0


class OnnxOcrEngine:
    def __init__(self, config: Config):
        # Import ONNX Runtime here, so the tests load without it
        import onnxruntime as ort

        model_dir = Path(config.onnx_model_dir)
        det_dir, rec_dir = model_dir / config.det_model, model_dir / config.rec_model
        rec_config = json.loads((rec_dir / "config.json").read_text(encoding="utf-8"))

        device_id = parse_gpu_device(config.device)
        providers: list = ["CPUExecutionProvider"]
        if device_id is not None:
            # Load the CUDA and cuDNN libraries from the nvidia-* wheels
            ort.preload_dlls()
            cuda = {
                "device_id": device_id,
                # llm-server shares the card, so cap the arena and grow it only on demand
                "gpu_mem_limit": config.gpu_mem_limit_mb * 1024 * 1024,
                "arena_extend_strategy": "kSameAsRequested",
                # On the T400 with cuDNN 9.27, HEURISTIC and EXHAUSTIVE find no algorithm for some
                # detection convolutions, and DEFAULT runs every size
                "cudnn_conv_algo_search": "DEFAULT",
            }
            providers.insert(0, ("CUDAExecutionProvider", cuda))

        options = ort.SessionOptions()
        options.intra_op_num_threads = config.cpu_threads
        options.log_severity_level = 3
        self._det = ort.InferenceSession(str(det_dir / "inference.onnx"), options, providers=providers)
        self._rec = ort.InferenceSession(str(rec_dir / "inference.onnx"), options, providers=providers)
        # Without this, a CUDA error moves the session to the CPU for good and only prints a warning
        self._det.disable_fallback()
        self._rec.disable_fallback()
        run_options = ort.RunOptions()
        if device_id is not None:
            # Every receipt has a new input size, and the arena keeps the blocks of each one. Freeing
            # them after each run keeps a 1600 px square image inside OCR_GPU_MEM_LIMIT_MB.
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{device_id}")
        active = self._det.get_providers()
        if device_id is not None and "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"OCR_DEVICE={config.device} but ONNX Runtime has only {active}")

        rec_shape = rec_config["PreProcess"]["transform_ops"]
        image_shape = next(op["RecResizeImg"]["image_shape"] for op in rec_shape if "RecResizeImg" in op)
        self._pipeline = OnnxPipeline(
            self._run(self._det, run_options),
            self._run(self._rec, run_options),
            rec_characters(rec_config),
            rec=RecSettings(image_shape=tuple(image_shape)),
        )
        self._min_confidence = config.min_confidence
        self._lock = threading.Lock()
        log.info(
            "Loaded OCR models det=%s rec=%s device=%s providers=%s",
            config.det_model,
            config.rec_model,
            config.device,
            ",".join(active),
            extra={
                "fields": {
                    "ocr": {
                        "det": config.det_model,
                        "rec": config.rec_model,
                        "device": config.device,
                        "providers": active,
                    }
                }
            },
        )

    @staticmethod
    def _run(session, run_options) -> Run:
        name = session.get_inputs()[0].name
        return lambda batch: session.run(None, {name: batch}, run_options)[0]

    def recognize(self, image: np.ndarray) -> list[Word]:
        with self._lock:
            results = self._pipeline.predict(image)
        return [
            Word(
                text=text,
                confidence=score,
                box=(
                    int(poly[:, 0].min()),
                    int(poly[:, 1].min()),
                    int(poly[:, 0].max()),
                    int(poly[:, 1].max()),
                ),
                angle=_top_edge_angle(poly),
            )
            for text, score, poly in results
            if score >= self._min_confidence
        ]
