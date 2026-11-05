"""Generated forms for the OCR tests. Every name, number and image here is made up."""

from pathlib import Path

import cv2
import numpy as np

from app.ocr.preprocess import TemplateCandidate
from app.ocr.templates import FormLayout, parse_layout
from app.ocr.text_reader import TextBox
from scripts.make_placeholder_form import render_form

PAGE = {"width": 1700, "height": 2200}
# A small shift, tilt and perspective, like a photo the phone crop did not quite square up.
SMALL_TILT = np.float64([[1, 0.02, 25], [-0.015, 1, -18], [1e-5, -1e-5, 1]])


def _box(x: int, y: int, w: int = 700, h: int = 80, kind: str = "handwritten") -> dict:
    return {"x": x, "y": y, "w": w, "h": h, "kind": kind}


def enrollment_layout(name: str = "enrollment") -> FormLayout:
    fields = {}
    for i, field in enumerate(
        ("last_name", "first_name", "middle_name", "birth_date", "barangay", "province",
         "lrn", "religion", "father_name", "mother_name", "guardian_name", "contact_number")
    ):  # fmt: skip
        fields[field] = _box(100 + (i % 2) * 800, 300 + (i // 2) * 180)
    fields["sex_male"] = _box(100, 1500, 40, 40, "checkbox")
    fields["sex_female"] = _box(400, 1500, 40, 40, "checkbox")
    return parse_layout(
        {
            "format_version": 1,
            "name": name,
            "version": 1,
            "image": f"{name}.png",
            "page": PAGE,
            "fields": fields,
        }
    )


def sf10_layout() -> FormLayout:
    """An SF10-style form with two school-year blocks, and the school before the year."""
    return parse_layout(
        {
            "format_version": 1,
            "name": "sf10_test",
            "version": 1,
            "image": "sf10_test.png",
            "page": PAGE,
            "fields": {"last_name": _box(100, 260), "first_name": _box(900, 260)},
            "repeating_sections": [
                {
                    "name": "school_year",
                    "instances": [{"x": 100, "y": 500}, {"x": 100, "y": 1200}],
                    "fields": {
                        "school_name": _box(0, 40, 1400),
                        "school_year": _box(0, 180, 600),
                        "grade_level": _box(800, 180, 600),
                        "general_average": _box(0, 320, 600),
                        "final_status": _box(800, 320, 600),
                    },
                }
            ],
        }
    )


def unrelated_layout() -> FormLayout:
    """A different form: three narrow columns where the enrollment form has two."""
    fields = {
        f"item_{i}": _box(100 + (i % 3) * 500, 300 + (i // 3) * 170, 420, 70) for i in range(30)
    }
    return parse_layout(
        {
            "format_version": 1,
            "name": "unrelated",
            "version": 1,
            "image": "unrelated.png",
            "page": PAGE,
            "fields": fields,
        }
    )


def save_png(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def candidate(layout: FormLayout, directory: Path, template_id: int | None = None):
    """A template whose blank image is written to `directory`."""
    save_png(directory / layout.image, render_form(layout))
    return TemplateCandidate(template_id, layout, directory / layout.image)


def tilted(image: np.ndarray, matrix: np.ndarray = SMALL_TILT) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, matrix, (width, height), borderValue=255)


def noise_page() -> np.ndarray:
    return np.random.default_rng(1).integers(0, 255, (2200, 1700), dtype=np.uint8)


def white_page() -> np.ndarray:
    return np.full((2200, 1700), 255, dtype=np.uint8)


def box(text: str, x: int, y: int, w: int = 200, h: int = 40, confidence: float = 0.9) -> TextBox:
    return TextBox(text, x, y, w, h, confidence)


class StubReader:
    """Returns fixed boxes, as a real detector might have read them. It reads nothing, so
    tests using it say nothing about real recognition accuracy."""

    model_version = "stub-reader-1"

    def __init__(self, boxes: list[TextBox]) -> None:
        self.boxes = boxes

    def read(self, page: np.ndarray) -> list[TextBox]:
        assert page.ndim == 2  # the cleaned grayscale page, never the colour photo
        return self.boxes
