"""What the OCR steps need from the models (Phase 7).

`PaddleTextReader` (app/ocr/paddle.py) is the real `TextReader`; `TrOcrRecognizer` and
`PaddleRecognizer` are the real `LineRecognizer`s. Tests use small fakes of these.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Reading:
    """One recognized line or crop.

    `confidence` is the lowest per-character probability, so one unsure character is enough
    to flag the field; None when nothing was read. `truncated` means decoding stopped at its
    length limit, so the end of the text may be missing.
    """

    text: str
    confidence: float | None
    truncated: bool = False


@dataclass(frozen=True)
class Region:
    """Where the detector found text, in image pixels. `score` is its mean text probability."""

    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass(frozen=True)
class TextBox:
    """One piece of detected text, in page pixels."""

    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


class LineRecognizer(Protocol):
    model_version: str

    def recognize_batch(
        self, crops: Sequence[np.ndarray], max_tokens: Sequence[int] | None = None
    ) -> list[Reading]:
        """Read one line of text from each crop, all in as few model calls as possible.
        `max_tokens` is a per-crop length limit, for models that decode token by token."""
        ...


class TextDetector(Protocol):
    model_version: str

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Region]]:
        """Find the text in each image, in one model call."""
        ...


class TextReader(Protocol):
    model_version: str

    def detect(self, page: np.ndarray) -> list[Region]:
        """Find the text on a cleaned grayscale page, without reading it."""
        ...

    def recognize(self, page: np.ndarray, regions: list[Region]) -> list[TextBox]:
        """Read the text in each region. Regions where nothing was read are left out."""
        ...

    def read(self, page: np.ndarray) -> list[TextBox]:
        """Detect and recognize all text on a cleaned grayscale page."""
        ...
