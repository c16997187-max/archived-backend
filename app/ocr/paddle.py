"""PP-OCRv6 detection (DB) and recognition (CTC) on ONNX Runtime (Phase 7).

Loaded from `models/ppocrv6_<tier>_{det,rec}`. The pre- and post-processing follow each
model's `inference.yml`, written out here so the host needs no Paddle packages.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.ocr.text_reader import Reading, Region, TextBox

# Detection. Pages are 1700 x 2200; a longer side is scaled down to this.
DET_MAX_SIDE = 1600
DET_STRIDE = 32
DET_MEAN = np.float32([0.485, 0.456, 0.406])
DET_STD = np.float32([0.229, 0.224, 0.225])
# Used when inference.yml leaves them out.
DET_DEFAULTS = {"thresh": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5, "max_candidates": 1000}
# Shorter than this, in model-map pixels, is a speck, not text.
MIN_BOX_SIDE = 3

# Recognition.
REC_HEIGHT = 48
REC_MIN_WIDTH = 320
REC_MAX_WIDTH = 3200
REC_BATCH = 32
BLANK = ""


def _session_input(session: Any) -> str:
    return session.get_inputs()[0].name


def _bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- detection


def _det_scale(height: int, width: int) -> float:
    return min(1.0, DET_MAX_SIDE / max(height, width))


def det_inputs(images: Sequence[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    """One batch tensor for the detector, and each image's scale. Images are scaled so the
    longer side fits DET_MAX_SIDE, then padded with white paper to a common size that is a
    multiple of 32, so differently sized crops share one model call."""
    scaled = []
    scales = []
    for image in images:
        bgr = _bgr(image)
        height, width = bgr.shape[:2]
        scale = _det_scale(height, width)
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        scaled.append(bgr)
        scales.append(scale)
    pad_h = max(math.ceil(img.shape[0] / DET_STRIDE) * DET_STRIDE for img in scaled)
    pad_w = max(math.ceil(img.shape[1] / DET_STRIDE) * DET_STRIDE for img in scaled)
    batch = np.full((len(scaled), pad_h, pad_w, 3), 255, dtype=np.uint8)
    for i, img in enumerate(scaled):
        batch[i, : img.shape[0], : img.shape[1]] = img
    normalized = (batch.astype(np.float32) / 255.0 - DET_MEAN) / DET_STD
    return np.ascontiguousarray(normalized.transpose(0, 3, 1, 2)), scales


def regions_from_map(
    prob: np.ndarray,
    scale: float,
    image_size: tuple[int, int],
    thresh: float,
    box_thresh: float,
    unclip_ratio: float,
    max_candidates: int,
) -> list[Region]:
    """DB post-processing: threshold the text probability map, take each blob's rotated
    rectangle, grow it by area * ratio / perimeter (the DB "unclip"), and return its
    upright bounding box in the original image's pixels."""
    width, height = image_size
    mask = (prob > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours[:max_candidates]:
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        if min(rw, rh) < MIN_BOX_SIDE:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blob = np.zeros((h, w), np.uint8)
        cv2.fillPoly(blob, [contour - (x, y)], 1)
        score = float(prob[y : y + h, x : x + w][blob.astype(bool)].mean())
        if score < box_thresh:
            continue
        grow = rw * rh * unclip_ratio / (2 * (rw + rh))
        corners = cv2.boxPoints(((cx, cy), (rw + 2 * grow, rh + 2 * grow), angle)) / scale
        x0, y0 = np.floor(corners.min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        if x1 - x0 >= 1 and y1 - y0 >= 1:
            regions.append(Region(int(x0), int(y0), int(x1 - x0), int(y1 - y0), score))
    return sorted(regions, key=lambda r: (r.y, r.x))


class PaddleDetector:
    def __init__(self, session: Any, postprocess: dict[str, Any], model_version: str) -> None:
        self.session = session
        self.model_version = model_version
        settings = {**DET_DEFAULTS, **postprocess}
        self.thresh = float(settings["thresh"])
        self.box_thresh = float(settings["box_thresh"])
        self.unclip_ratio = float(settings["unclip_ratio"])
        self.max_candidates = int(settings["max_candidates"])

    @classmethod
    def from_dir(cls, directory: Path, session: Any, model_version: str) -> "PaddleDetector":
        config = _read_yaml(directory / "inference.yml")
        return cls(session, config.get("PostProcess") or {}, model_version)

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Region]]:
        if not images:
            return []
        batch, scales = det_inputs(images)
        maps = self.session.run(None, {_session_input(self.session): batch})[0]
        results = []
        for image, prob, scale in zip(images, maps, scales, strict=True):
            height, width = image.shape[:2]
            results.append(
                regions_from_map(
                    prob[0],
                    scale,
                    (width, height),
                    self.thresh,
                    self.box_thresh,
                    self.unclip_ratio,
                    self.max_candidates,
                )
            )
        return results

    def detect(self, image: np.ndarray) -> list[Region]:
        return self.detect_batch([image])[0]


# -------------------------------------------------------------- recognition


def charset_for(dictionary: list[str], classes: int) -> list[str]:
    """Index 0 is the CTC blank; PaddleOCR appends a space when the model has room for it."""
    charset = [BLANK, *(str(c) for c in dictionary)]
    if classes == len(charset) + 1:
        charset.append(" ")
    if classes != len(charset):
        raise ValueError(f"Model has {classes} classes but the dictionary gives {len(charset)}")
    return charset


def rec_inputs(crops: Sequence[np.ndarray]) -> np.ndarray:
    """Height 48 with the aspect ratio kept, padded on the right to the widest crop."""
    ratios = [c.shape[1] / max(c.shape[0], 1) for c in crops]
    width = min(max(REC_MIN_WIDTH, math.ceil(REC_HEIGHT * max(ratios))), REC_MAX_WIDTH)
    batch = np.zeros((len(crops), 3, REC_HEIGHT, width), dtype=np.float32)
    for i, (crop, ratio) in enumerate(zip(crops, ratios, strict=True)):
        resized_w = max(1, min(width, math.ceil(REC_HEIGHT * ratio)))
        resized = cv2.resize(_bgr(crop), (resized_w, REC_HEIGHT), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        batch[i, :, :, :resized_w] = normalized.transpose(2, 0, 1)
    return batch


def ctc_decode(probs: np.ndarray, charset: list[str]) -> list[Reading]:
    """Greedy CTC: best class per time step, repeats collapsed, blanks dropped. A character's
    probability is its time step's; the reading's confidence is the lowest of them."""
    best = probs.argmax(axis=-1)
    best_prob = probs.max(axis=-1)
    readings = []
    for indices, scores in zip(best, best_prob, strict=True):
        keep = indices != 0
        keep[1:] &= indices[1:] != indices[:-1]
        text = "".join(charset[i] for i in indices[keep])
        kept = scores[keep]
        stripped = text.strip()
        confidence = float(kept.min()) if stripped else None
        readings.append(Reading(stripped, confidence))
    return readings


class PaddleRecognizer:
    """PP-OCRv6 text-line recognizer. Strong on print; usable on neat handwriting."""

    def __init__(self, session: Any, charset: list[str], model_version: str) -> None:
        self.session = session
        self.charset = charset
        self.model_version = model_version

    @classmethod
    def from_dir(cls, directory: Path, session: Any, model_version: str) -> "PaddleRecognizer":
        dictionary = _read_yaml(directory / "inference.yml")["PostProcess"]["character_dict"]
        classes = session.get_outputs()[0].shape[-1]
        return cls(session, charset_for(dictionary, int(classes)), model_version)

    def recognize_batch(
        self, crops: Sequence[np.ndarray], max_tokens: Sequence[int] | None = None
    ) -> list[Reading]:
        """`max_tokens` is ignored: CTC reads the whole crop at once and cannot run long."""
        results: list[Reading | None] = [None] * len(crops)
        # Similar widths together, so little of each batch is padding.
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
        for start in range(0, len(order), REC_BATCH):
            chunk = order[start : start + REC_BATCH]
            batch = rec_inputs([crops[i] for i in chunk])
            probs = self.session.run(None, {_session_input(self.session): batch})[0]
            for i, reading in zip(chunk, ctc_decode(probs, self.charset), strict=True):
                results[i] = reading
        return [r for r in results if r is not None]


# ------------------------------------------------------------ full page


class PaddleTextReader:
    """Detection plus recognition, for the generic path's full page."""

    def __init__(self, detector: PaddleDetector, recognizer: PaddleRecognizer) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.model_version = f"{detector.model_version}+{recognizer.model_version}"

    def detect(self, page: np.ndarray) -> list[Region]:
        return self.detector.detect(page)

    def recognize(self, page: np.ndarray, regions: list[Region]) -> list[TextBox]:
        crops = [page[r.y : r.y + r.h, r.x : r.x + r.w] for r in regions]
        readings = self.recognizer.recognize_batch(crops)
        return [
            TextBox(reading.text, r.x, r.y, r.w, r.h, reading.confidence)
            for r, reading in zip(regions, readings, strict=True)
            if reading.text and reading.confidence is not None
        ]

    def read(self, page: np.ndarray) -> list[TextBox]:
        return self.recognize(page, self.detect(page))
