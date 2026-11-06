"""Steps 6 to 9 of the OCR pipeline: read each field crop, score it, normalize it and decide
whether a person must check it (Phase 7).

No database. Field values never reach the log; `problem` codes are safe to log.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from app.ocr.normalize import GUARDIAN_FIELDS, base_name, is_required, normalize_field
from app.ocr.preprocess import FieldCrop
from app.ocr.templates import FieldBox
from app.ocr.text_reader import Reading, Region

if TYPE_CHECKING:
    from app.ocr.models import OcrEngine

CHECKBOX_VERSION = "checkbox-darkpixel-v1"
CHECKED = "checked"
UNCHECKED = "unchecked"
# Share of dark pixels inside the box, away from its printed border, that makes it ticked.
# Measured on generated ticks and crosses; check against real forms.
CHECKED_RATIO = 0.06
CHECKBOX_MARGIN = 0.2
# Darker than this share of the paper's brightness counts as ink.
INK_LEVEL = 0.6
# Two checkboxes that together answer one question.
CHECKBOX_GROUPS = {"sex": {"sex_male": "M", "sex_female": "F"}}

# TrOCR decodes token by token; a long address needs room. Hitting the limit flags the field.
MAX_TOKENS = 64
LONG_MAX_TOKENS = 96
LONG_FIELDS = frozenset(
    {"address", "house_street", "place_of_birth", "father_name", "mother_name",
     "guardian_name", "school_name"}
)  # fmt: skip
# Detected lines this close (a share of the shorter one's height) are the same line.
SAME_LINE_OVERLAP = 0.5
LINE_PAD = 4


@dataclass(frozen=True)
class FieldResult:
    name: str
    bbox: dict[str, int] | None
    raw: str | None
    normalized: str | None
    confidence: float | None
    needs_check: bool
    model_version: str
    problem: str | None = None  # an error code, never the value


def _bbox(box: FieldBox) -> dict[str, int]:
    return {"x": box.x, "y": box.y, "w": box.w, "h": box.h}


def max_tokens_for(field_name: str) -> int:
    return LONG_MAX_TOKENS if base_name(field_name) in LONG_FIELDS else MAX_TOKENS


# ---------------------------------------------------------------- checkboxes


def checkbox_state(clean: np.ndarray) -> Reading:
    """Count dark pixels inside the box, leaving out a margin where its border is printed.
    Confidence is 0.5 at the threshold and 1.0 at zero or twice the threshold."""
    height, width = clean.shape[:2]
    dy, dx = int(height * CHECKBOX_MARGIN), int(width * CHECKBOX_MARGIN)
    inner = clean[dy : height - dy, dx : width - dx]
    if inner.size == 0:
        return Reading(UNCHECKED, 0.0)
    paper = float(np.percentile(clean, 90)) or 255.0
    ratio = float((inner < paper * INK_LEVEL).mean())
    distance = min(1.0, abs(ratio - CHECKED_RATIO) / CHECKED_RATIO)
    return Reading(CHECKED if ratio >= CHECKED_RATIO else UNCHECKED, 0.5 + 0.5 * distance)


def _group_results(checks: dict[str, tuple[FieldBox, Reading]], threshold: float):
    """One answer per group, such as `sex` from `sex_male` and `sex_female`."""
    results = []
    for group, options in CHECKBOX_GROUPS.items():
        present = {name: checks[name] for name in options if name in checks}
        if len(present) != len(options):
            continue
        ticked = [options[n] for n, (_, r) in present.items() if r.text == CHECKED]
        confidence = min(r.confidence for _, r in present.values())
        boxes = [box for box, _ in present.values()]
        left, top = min(b.x for b in boxes), min(b.y for b in boxes)
        bbox = {
            "x": left,
            "y": top,
            "w": max(b.x + b.w for b in boxes) - left,
            "h": max(b.y + b.h for b in boxes) - top,
        }
        value = ticked[0] if len(ticked) == 1 else None
        problem = None if value else ("pick_one" if ticked else "required")
        results.append(
            FieldResult(
                group,
                bbox,
                value,
                value,
                confidence,
                problem is not None or confidence < threshold,
                CHECKBOX_VERSION,
                problem,
            )
        )
    return results


# -------------------------------------------------------------------- lines


def line_boxes(regions: Sequence[Region]) -> list[tuple[int, int, int, int]]:
    """(left, top, right, bottom) of each line of detected text, top to bottom. Regions
    that share most of their height are one line."""
    lines: list[list[int]] = []
    for region in sorted(regions, key=lambda r: r.y):
        left, top = region.x, region.y
        right, bottom = region.x + region.w, region.y + region.h
        if lines:
            last = lines[-1]
            shared = min(last[3], bottom) - max(last[1], top)
            if shared >= SAME_LINE_OVERLAP * min(last[3] - last[1], bottom - top):
                last[:] = [min(last[0], left), min(last[1], top),
                           max(last[2], right), max(last[3], bottom)]  # fmt: skip
                continue
        lines.append([left, top, right, bottom])
    return [(left, top, right, bottom) for left, top, right, bottom in lines]


def split_lines(crop: np.ndarray, regions: Sequence[Region]) -> list[np.ndarray]:
    """TrOCR reads one line, and invents words to fill blank space beside the writing. So
    each line of detected writing is cut out on its own, close around the text. A crop
    where nothing was detected stays whole."""
    lines = line_boxes(regions)
    if not lines:
        return [crop]
    height, width = crop.shape[:2]
    return [
        crop[
            max(top - LINE_PAD, 0) : min(bottom + LINE_PAD, height),
            max(left - 2 * LINE_PAD, 0) : min(right + 2 * LINE_PAD, width),
        ]
        for left, top, right, bottom in lines
    ]


def _join(readings: Sequence[Reading]) -> Reading:
    text = " ".join(r.text for r in readings if r.text)
    scores = [r.confidence for r in readings if r.confidence is not None]
    return Reading(
        text, min(scores) if scores and text else None, any(r.truncated for r in readings)
    )


def read_handwritten(crops: Sequence[FieldCrop], engine: "OcrEngine") -> list[Reading]:
    """Step 6 for handwriting: find the lines in every crop (one detector call), then read
    every line of every crop in one recognizer call, and join each crop's lines.

    A crop where the detector finds no writing is read as empty without the recognizer:
    TrOCR given a blank box invents text, and skipping it saves the time. An empty
    required field is still flagged in step 9."""
    if not crops:
        return []
    regions = engine.detector.detect_batch([c.clean for c in crops])
    lines: list[np.ndarray] = []
    owners: list[int] = []
    limits: list[int] = []
    for i, (crop, found) in enumerate(zip(crops, regions, strict=True)):
        if not found:
            continue
        for line in split_lines(crop.clean, found):
            lines.append(line)
            owners.append(i)
            limits.append(max_tokens_for(crop.name))
    readings = engine.handwriting.recognize_batch(lines, limits) if lines else []
    per_crop: list[list[Reading]] = [[] for _ in crops]
    for owner, reading in zip(owners, readings, strict=True):
        per_crop[owner].append(reading)
    return [_join(found) for found in per_crop]


# ------------------------------------------------------------------- fields


def finish_field(
    name: str,
    bbox: dict[str, int] | None,
    reading: Reading,
    model_version: str,
    threshold: float,
) -> FieldResult:
    """Steps 7 to 9: keep the lowest character score, normalize, and flag."""
    normalized = normalize_field(name, reading.text)
    problem = normalized.problem
    if reading.truncated:
        problem = "truncated"
    elif problem is None and normalized.value is None and is_required(name):
        problem = "required"
    unsure = reading.confidence is not None and reading.confidence < threshold
    return FieldResult(
        name,
        bbox,
        reading.text or None,
        normalized.value,
        reading.confidence,
        problem is not None or unsure,
        model_version,
        problem,
    )


def _missing_guardian(results: list[FieldResult]) -> list[FieldResult]:
    """Phase 8 needs a guardian or a parent; flag the guardian field when all are empty."""
    present = {r.name: r for r in results if r.name in GUARDIAN_FIELDS}
    if not present or any(r.normalized for r in present.values()):
        return results
    target = next(name for name in GUARDIAN_FIELDS if name in present)
    return [
        replace(r, needs_check=True, problem=r.problem or "required") if r.name == target else r
        for r in results
    ]


def recognize_crops(
    crops: Sequence[FieldCrop], engine: "OcrEngine", threshold: float
) -> list[FieldResult]:
    """Every field of one form, in template order, plus one answer per checkbox group.
    Each model is called once per form with all of its crops."""
    handwritten = [c for c in crops if c.box.kind == "handwritten"]
    printed = [c for c in crops if c.box.kind == "printed"]
    readings: dict[str, tuple[Reading, str]] = {}
    for crop, reading in zip(handwritten, read_handwritten(handwritten, engine), strict=True):
        readings[crop.name] = (reading, engine.handwriting.model_version)
    printed_readings = engine.printed.recognize_batch([c.clean for c in printed])
    for crop, reading in zip(printed, printed_readings, strict=True):
        readings[crop.name] = (reading, engine.printed.model_version)

    results = []
    checks: dict[str, tuple[FieldBox, Reading]] = {}
    for crop in crops:
        if crop.box.kind == "checkbox":
            reading = checkbox_state(crop.clean)
            checks[crop.name] = (crop.box, reading)
            results.append(
                FieldResult(
                    crop.name,
                    _bbox(crop.box),
                    reading.text,
                    reading.text,
                    reading.confidence,
                    reading.confidence < threshold,
                    CHECKBOX_VERSION,
                )
            )
        else:
            reading, version = readings[crop.name]
            results.append(finish_field(crop.name, _bbox(crop.box), reading, version, threshold))
    results.extend(_group_results(checks, threshold))
    return _missing_guardian(results)
