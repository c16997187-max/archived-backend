from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.ocr import preprocess as pp
from app.ocr.preprocess import DebugSink, PreprocessFailure, preprocess
from scripts.make_placeholder_form import render_form
from tests.ocr_images import (
    candidate,
    enrollment_layout,
    noise_page,
    save_png,
    sf10_layout,
    tilted,
    unrelated_layout,
    white_page,
)

_EXIF_ORIENTATION = 0x0112


def _marked_form(layout, field: str) -> np.ndarray:
    """The blank form with a black square in the middle of one field's box."""
    page = render_form(layout)
    box = layout.boxes()[field]
    cx, cy = box.x + box.w // 2, box.y + box.h // 2
    page[cy - 15 : cy + 15, cx - 15 : cx + 15] = 0
    return page


# --- Load and orient ---------------------------------------------------------------


def test_load_applies_exif_rotation(tmp_path: Path):
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)
    pixels[:, :100] = 255  # white left half, black right half
    exif = Image.Exif()
    exif[_EXIF_ORIENTATION] = 6  # shown rotated 90 degrees clockwise
    path = tmp_path / "photo.jpg"
    Image.fromarray(pixels).save(path, exif=exif, quality=95)

    loaded = pp.load_oriented(path)

    assert loaded.shape[:2] == (200, 100)
    assert loaded[20, 50].mean() > 200  # the white half is now on top
    assert loaded[180, 50].mean() < 50


def test_corners_apply_to_the_oriented_photo(tmp_path: Path):
    # Stored sideways, with a white patch that shows at the top once it is upright.
    pixels = np.zeros((400, 800, 3), dtype=np.uint8)
    pixels[:, :400] = 255
    exif = Image.Exif()
    exif[_EXIF_ORIENTATION] = 6
    path = tmp_path / "photo.png"
    Image.fromarray(pixels).save(path, exif=exif)
    top_half = {
        "top_left": {"x": 0, "y": 0},
        "top_right": {"x": 1, "y": 0},
        "bottom_right": {"x": 1, "y": 0.5},
        "bottom_left": {"x": 0, "y": 0.5},
    }

    page = pp.flatten(pp.load_oriented(path), top_half, (170, 220))

    assert page.mean() > 240


def test_missing_file_fails_with_a_reason(tmp_path: Path):
    with pytest.raises(PreprocessFailure) as failure:
        preprocess(tmp_path / "gone.jpg", None, [candidate(enrollment_layout(), tmp_path)])
    assert failure.value.reason == pp.REASON_MISSING


def test_unreadable_file_fails_with_a_reason(tmp_path: Path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    with pytest.raises(PreprocessFailure) as failure:
        preprocess(path, None, [candidate(enrollment_layout(), tmp_path)])
    assert failure.value.reason == pp.REASON_UNREADABLE


def test_truncated_image_fails_with_a_reason(tmp_path: Path):
    whole = save_png(tmp_path / "whole.png", noise_page())
    path = tmp_path / "cut.png"
    path.write_bytes(whole.read_bytes()[:5000])
    with pytest.raises(PreprocessFailure) as failure:
        preprocess(path, None, [candidate(enrollment_layout(), tmp_path)])
    assert failure.value.reason == pp.REASON_UNREADABLE


# --- Flatten -----------------------------------------------------------------------


def test_flatten_recovers_a_known_quadrilateral():
    photo = np.zeros((1000, 800, 3), dtype=np.uint8)
    quad = np.int32([(100, 200), (700, 150), (750, 900), (50, 850)])
    cv2.fillConvexPoly(photo, quad, (255, 255, 255))
    corners = {
        name: {"x": float(x) / 800, "y": float(y) / 1000}
        for name, (x, y) in zip(
            ("top_left", "top_right", "bottom_right", "bottom_left"), quad, strict=True
        )
    }

    page = pp.flatten(photo, corners, (170, 220))

    assert page.shape[:2] == (220, 170)
    assert page[5:-5, 5:-5].min() > 200  # the white shape fills the whole page


def test_flatten_with_no_corners_uses_the_whole_photo():
    assert pp.flatten(np.zeros((50, 60, 3), np.uint8), None, (170, 220)).shape[:2] == (220, 170)


# --- Align -------------------------------------------------------------------------


def test_tilted_form_aligns_with_field_in_its_box(tmp_path: Path):
    layout = enrollment_layout()
    photo = save_png(tmp_path / "photo.png", tilted(_marked_form(layout, "birth_date")))

    result = preprocess(photo, None, [candidate(layout, tmp_path, 7)])

    assert result.ocr_path == pp.TEMPLATE_PATH
    assert result.template.id == 7
    assert result.alignment_score >= pp.MIN_ALIGNMENT_SCORE
    assert result.clean_page is None
    crop = next(c for c in result.crops if c.name == "birth_date")
    dark = np.argwhere(cv2.cvtColor(crop.raw, cv2.COLOR_BGR2GRAY)[:60] < 60)
    # The square landed centred in the box, within a few pixels.
    assert abs(dark[:, 1].mean() - crop.box.w / 2) < 5
    assert abs(dark[:, 0].mean() - crop.box.h / 2) < 5


def test_alignment_survives_a_filled_in_blurred_photo(tmp_path: Path):
    layout = enrollment_layout()
    page = tilted(render_form(layout))
    rng = np.random.default_rng(0)
    for box in layout.boxes().values():
        for _ in range(6):
            stroke = rng.integers([box.x, box.y], [box.x + box.w, box.y + box.h], size=(5, 2))
            cv2.polylines(page, [stroke.astype(np.int32)], False, 40, 3)
    blurred = cv2.GaussianBlur(page, (5, 5), 0) + rng.normal(0, 12, page.shape)
    photo = save_png(tmp_path / "photo.png", np.clip(blurred, 0, 255).astype(np.uint8))

    assert preprocess(photo, None, [candidate(layout, tmp_path)]).ocr_path == pp.TEMPLATE_PATH


def test_best_of_several_templates_wins(tmp_path: Path):
    layout = sf10_layout()
    others = [enrollment_layout(), unrelated_layout()]
    photo = save_png(tmp_path / "photo.png", tilted(render_form(layout)))

    result = preprocess(photo, None, [candidate(t, tmp_path) for t in [*others, layout]])

    assert result.template.layout.name == "sf10_test"


def test_repeating_sections_are_cut_for_every_instance(tmp_path: Path):
    layout = sf10_layout()
    photo = save_png(
        tmp_path / "photo.png", tilted(_marked_form(layout, "school_year[1].general_average"))
    )

    result = preprocess(photo, None, [candidate(layout, tmp_path)])

    crops = {c.name: c for c in result.crops}
    assert "school_year[0].school_name" in crops
    assert "school_year[1].final_status" in crops
    marked = crops["school_year[1].general_average"].raw
    empty = crops["school_year[0].general_average"].raw
    assert marked.min() < 60
    # Below the printed label's reach, the other year's box has no square.
    assert empty[20:60, 280:320].min() > 60


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(lambda: render_form(unrelated_layout(), "unrelated memo"), id="different"),
        pytest.param(
            lambda: render_form(unrelated_layout(), "enrollment"), id="different-same-heading"
        ),
        pytest.param(noise_page, id="noise"),
        pytest.param(white_page, id="blank"),
    ],
)
def test_no_matching_template_goes_to_the_generic_path(tmp_path: Path, image):
    photo = save_png(tmp_path / "photo.png", image())

    result = preprocess(photo, None, [candidate(enrollment_layout(), tmp_path)])

    assert result.ocr_path == pp.GENERIC_PATH
    assert result.template is None
    assert result.crops == []
    assert result.alignment_score < pp.MIN_ALIGNMENT_SCORE
    assert result.clean_page.shape == (2200, 1700)  # flattened and grayscale


def test_generic_page_is_flattened_with_the_stored_corners(tmp_path: Path):
    image = np.zeros((2200, 1700), np.uint8)
    image[:1100] = 255
    photo = save_png(tmp_path / "photo.png", image)
    top_half = {
        "top_left": {"x": 0, "y": 0},
        "top_right": {"x": 1, "y": 0},
        "bottom_right": {"x": 1, "y": 0.5},
        "bottom_left": {"x": 0, "y": 0.5},
    }

    result = preprocess(photo, top_half, [candidate(enrollment_layout(), tmp_path)])

    assert result.clean_page.mean() > 240


def test_no_templates_fails(tmp_path: Path):
    photo = save_png(tmp_path / "photo.png", white_page())
    with pytest.raises(PreprocessFailure) as failure:
        preprocess(photo, None, [])
    assert failure.value.reason == pp.REASON_NO_TEMPLATES


def test_template_with_missing_image_is_skipped(tmp_path: Path):
    layout = enrollment_layout()
    good = candidate(layout, tmp_path)
    broken = pp.TemplateCandidate(None, unrelated_layout(), tmp_path / "gone.png")
    photo = save_png(tmp_path / "photo.png", render_form(layout))

    assert preprocess(photo, None, [broken, good]).template is good
    with pytest.raises(PreprocessFailure):
        preprocess(photo, None, [broken])


# --- Clean -------------------------------------------------------------------------


def _lined_crop() -> np.ndarray:
    """A pale crop with a ruling line and a pen stroke crossing it."""
    crop = np.full((80, 500, 3), 235, dtype=np.uint8)
    cv2.line(crop, (0, 55), (499, 55), (60, 60, 60), 2)
    cv2.line(crop, (250, 15), (250, 75), (20, 20, 20), 5)
    return crop


def test_ruling_lines_go_and_crossing_strokes_stay():
    clean = pp.clean_crop(_lined_crop(), "handwritten")

    assert clean[50:60, 50:200].min() > 120  # the line is gone
    assert clean[25:45, 248:253].max() < 100  # the stroke above the line stays
    assert clean[65:72, 248:253].max() < 100  # and below it


def test_vertical_dividers_go():
    crop = np.full((80, 500, 3), 235, dtype=np.uint8)
    cv2.line(crop, (120, 0), (120, 79), (60, 60, 60), 2)

    assert pp.clean_crop(crop, "handwritten")[:, 116:125].min() > 120


def test_checkbox_keeps_its_outline():
    crop = np.full((40, 40, 3), 235, dtype=np.uint8)
    cv2.rectangle(crop, (2, 2), (37, 37), (40, 40, 40), 2)

    clean = pp.clean_crop(crop, "checkbox")

    assert clean[2:4, 5:35].max() < 150
    assert clean.shape == (40, 40)


# --- Debug -------------------------------------------------------------------------


def test_debug_saves_every_stage_on_the_template_path(tmp_path: Path):
    layout = sf10_layout()
    photo = save_png(tmp_path / "in" / "photo.png", tilted(render_form(layout)))
    out = tmp_path / "out"

    result = preprocess(photo, None, [candidate(layout, tmp_path / "in")], DebugSink(out))

    for name in (
        "01_oriented",
        "02_flattened",
        "03_matches_sf10_test_v1",
        "04_aligned",
        "05_overlay",
    ):
        assert (out / f"{name}.png").stat().st_size > 0
    assert len(list((out / "fields").glob("*_raw.png"))) == len(result.crops)
    assert len(list((out / "fields").glob("*_clean.png"))) == len(result.crops)
    assert (out / "fields" / "school_year[1].school_name_clean.png").exists()
    assert "chosen path: template (sf10_test v1)" in (out / "scores.txt").read_text()


def test_debug_saves_the_generic_path(tmp_path: Path):
    photo = save_png(tmp_path / "in" / "photo.png", white_page())
    out = tmp_path / "out"

    preprocess(photo, None, [candidate(enrollment_layout(), tmp_path / "in")], DebugSink(out))

    assert (out / "04_clean_page.png").exists()
    assert not (out / "05_overlay.png").exists()
    assert "chosen path: generic" in (out / "scores.txt").read_text()


def test_debug_writes_the_failure_reason(tmp_path: Path):
    out = tmp_path / "out"
    with pytest.raises(PreprocessFailure):
        preprocess(tmp_path / "gone.jpg", None, [], DebugSink(out))
    assert pp.REASON_MISSING in (out / "failure.txt").read_text()
