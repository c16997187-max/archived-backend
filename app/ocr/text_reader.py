"""What the template-free path needs from a text detector and recognizer.

No implementation ships yet: PP-OCRv6 plugs in here in Part 2, loaded from `models/`.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TextBox:
    """One piece of detected text, in page pixels."""

    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


class TextReader(Protocol):
    model_version: str

    def read(self, page: np.ndarray) -> list[TextBox]:
        """Detect and recognize all text on a cleaned grayscale page."""
        ...
