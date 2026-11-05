"""Draw a stand-in blank form from a template JSON, until the real form is scanned.

    python -m scripts.make_placeholder_form docs/forms/enrollment_v1.json

The image holds printed labels only. Replace it with a flat, sharp photo of the real blank
form, fix the boxes in the JSON to match, and remove `"provisional": true`.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from app.ocr.templates import FormLayout, TemplateError, read_layout_file

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INK = 30
_SERIAL_MARKS = 40


def _label_of(name: str) -> str:
    """ "school_year[1].school_name" becomes "School Name"."""
    return name.rsplit(".", 1)[-1].replace("_", " ").title()


def render_form(layout: FormLayout, title: str | None = None) -> np.ndarray:
    """A grayscale blank form: heading, printed labels and a ruling line under each box."""
    width, height = layout.page.width, layout.page.height
    page = np.full((height, width), 255, dtype=np.uint8)

    heading = (title or layout.name).upper()
    cv2.putText(page, heading, (100, 110), _FONT, 1.6, _INK, 3, cv2.LINE_AA)
    cv2.putText(
        page, f"FORM VERSION {layout.version}", (100, 165), _FONT, 0.8, _INK, 2, cv2.LINE_AA
    )
    cv2.rectangle(page, (50, 50), (width - 50, height - 50), _INK, 3)
    cv2.line(page, (50, 195), (width - 50, 195), _INK, 2)
    # An irregular strip, unique to the form's name, so each form has its own features.
    rng = np.random.default_rng(sum(heading.encode()))
    for i in range(_SERIAL_MARKS):
        x = 100 + i * (width - 200) // _SERIAL_MARKS
        cv2.rectangle(
            page,
            (x, height - 130),
            (x + int(rng.integers(6, 24)), height - 130 + int(rng.integers(20, 60))),
            _INK,
            -1,
        )

    for name, box in layout.boxes().items():
        label_y = max(box.y - 8, 20)
        if box.kind == "checkbox":
            cv2.rectangle(page, (box.x, box.y), (box.x + box.w, box.y + box.h), _INK, 2)
            cv2.putText(
                page,
                _label_of(name),
                (box.x + box.w + 12, box.y + box.h - 8),
                _FONT,
                0.7,
                _INK,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(page, _label_of(name), (box.x, label_y), _FONT, 0.7, _INK, 2, cv2.LINE_AA)
            bottom = box.y + box.h - 4
            cv2.line(page, (box.x, bottom), (box.x + box.w, bottom), _INK, 2)
    return page


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw a placeholder blank form.")
    parser.add_argument("template", type=Path, help="Path to the template JSON")
    args = parser.parse_args(argv)

    try:
        layout = read_layout_file(args.template)
    except TemplateError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not layout.provisional:
        print("Refusing: this template is not provisional.", file=sys.stderr)
        return 1

    ok, encoded = cv2.imencode(".png", render_form(layout))
    if not ok:
        print("Could not encode the image.", file=sys.stderr)
        return 1
    target = args.template.parent / layout.image
    target.write_bytes(encoded.tobytes())
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
