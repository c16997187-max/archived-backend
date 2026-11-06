"""Steps 6 to 9 with fake models: routing, batching, lines, checkboxes and flags.
Every image is drawn here and every value is made up."""

import cv2
import numpy as np
import pytest

from app.ocr.preprocess import FieldCrop, clean_crop
from app.ocr.recognize import (
    CHECKED,
    LONG_MAX_TOKENS,
    MAX_TOKENS,
    UNCHECKED,
    checkbox_state,
    finish_field,
    line_boxes,
    recognize_crops,
    split_lines,
)
from app.ocr.templates import FieldBox
from app.ocr.text_reader import Reading, Region
from tests.ocr_images import FakeDetector, FakeRecognizer, fake_engine

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _blank(w: int = 700, h: int = 80) -> np.ndarray:
    return np.full((h, w), 255, np.uint8)


def _written(*lines: str, w: int = 700, h: int = 80) -> np.ndarray:
    crop = _blank(w, h)
    for i, text in enumerate(lines):
        cv2.putText(crop, text, (15, 30 + i * 35), FONT, 0.8, 0, 2)
    return crop


def _crop(name: str, image: np.ndarray, kind: str = "handwritten", x: int = 100) -> FieldCrop:
    h, w = image.shape[:2]
    return FieldCrop(name, FieldBox(x=x, y=300, w=w, h=h, kind=kind), image, image)


def _checkbox(mark: str) -> np.ndarray:
    """A 40 px printed box: empty, ticked, crossed or a faint stray dot."""
    box = np.full((40, 40), 255, np.uint8)
    cv2.rectangle(box, (1, 1), (38, 38), 30, 2)
    if mark == "tick":
        cv2.line(box, (9, 20), (17, 30), 20, 4)
        cv2.line(box, (17, 30), (32, 8), 20, 4)
    elif mark == "cross":
        cv2.line(box, (8, 8), (32, 32), 20, 4)
        cv2.line(box, (32, 8), (8, 32), 20, 4)
    elif mark == "dot":
        cv2.circle(box, (20, 20), 1, 60, -1)
    return box


# ---------------------------------------------------------------- checkboxes


@pytest.mark.parametrize(("mark", "state"), [("tick", CHECKED), ("cross", CHECKED),
                                             ("empty", UNCHECKED), ("dot", UNCHECKED)])  # fmt: skip
def test_checkbox_counts_dark_pixels_inside_the_border(mark, state):
    reading = checkbox_state(
        clean_crop(cv2.cvtColor(_checkbox(mark), cv2.COLOR_GRAY2BGR), "checkbox")
    )
    assert reading.text == state
    assert 0.5 <= reading.confidence <= 1.0


def test_clear_checkboxes_are_confident():
    assert checkbox_state(_checkbox("empty")).confidence == 1.0
    assert checkbox_state(_checkbox("cross")).confidence == 1.0


def _sex_crops(male: str, female: str) -> list[FieldCrop]:
    return [
        _crop("sex_male", _checkbox(male), "checkbox", x=100),
        _crop("sex_female", _checkbox(female), "checkbox", x=400),
    ]


@pytest.mark.parametrize(
    ("male", "female", "value", "problem"),
    [("tick", "empty", "M", None), ("empty", "cross", "F", None),
     ("tick", "tick", None, "pick_one"), ("empty", "empty", None, "required")],
)  # fmt: skip
def test_sex_comes_from_its_two_boxes(male, female, value, problem):
    results = {r.name: r for r in recognize_crops(_sex_crops(male, female), fake_engine(), 0.8)}
    sex = results["sex"]
    assert (sex.normalized, sex.problem, sex.needs_check) == (value, problem, problem is not None)
    assert sex.bbox == {"x": 100, "y": 300, "w": 340, "h": 40}


# --------------------------------------------------------------------- lines


def test_regions_on_one_line_merge_and_lines_stay_in_order():
    regions = [
        Region(300, 42, 100, 30, 0.9),
        Region(10, 5, 200, 30, 0.9),
        Region(250, 8, 80, 28, 0.9),
    ]
    assert line_boxes(regions) == [(10, 5, 330, 36), (300, 42, 400, 72)]


def test_two_line_crop_is_split_and_each_line_cut_close():
    crop = _written("Purok 7", "Baguio City", h=90)
    lines = split_lines(crop, FakeDetector().detect_batch([crop])[0])
    assert len(lines) == 2
    assert all(line.shape[0] < 45 and line.shape[1] < 300 for line in lines)


def test_crop_with_nothing_detected_stays_whole():
    crop = _blank()
    assert split_lines(crop, []) == [crop]


def test_lines_are_joined_with_a_space_and_the_least_sure_line_decides():
    crop = _written("Purok 7", "Baguio City", h=90)
    answers = iter([Reading("Purok 7", 0.9), Reading("Baguio City", 0.6)])
    hand = FakeRecognizer("fake-hand-1", read=lambda _: next(answers))
    [result] = recognize_crops([_crop("address", crop)], fake_engine(hand), 0.8)
    assert result.raw == "Purok 7 Baguio City"
    assert result.confidence == 0.6
    assert result.needs_check


def test_every_line_of_every_handwritten_crop_goes_in_one_call():
    detector = FakeDetector()
    hand = FakeRecognizer("fake-hand-1")
    printed = FakeRecognizer("fake-printed-1")
    crops = [
        _crop("address", _written("Purok 7", "Baguio City", h=90)),
        _crop("last_name", _written("Madeupname")),
        _crop("religion", _blank()),
        _crop("school_name", _written("Test School"), kind="printed"),
    ]
    recognize_crops(crops, fake_engine(hand, printed, detector=detector), 0.8)
    assert detector.batches == [3]
    assert len(hand.calls) == 1
    lines, limits = hand.calls[0]
    assert len(lines) == 3  # two lines and one line; the blank crop is not sent
    assert limits == [LONG_MAX_TOKENS, LONG_MAX_TOKENS, MAX_TOKENS]
    assert len(printed.calls) == 1 and len(printed.calls[0][0]) == 1


def test_crop_with_no_writing_detected_is_empty_without_the_recognizer():
    hand = FakeRecognizer("fake-hand-1", Reading("invented", 0.3))
    [result] = recognize_crops([_crop("religion", _blank())], fake_engine(hand), 0.8)
    assert hand.calls == []
    assert (result.raw, result.confidence, result.needs_check) == (None, None, False)


def test_no_handwritten_fields_means_no_detector_call():
    detector = FakeDetector()
    recognize_crops(_sex_crops("tick", "empty"), fake_engine(detector=detector), 0.8)
    assert detector.batches == []


# --------------------------------------------------------------------- flags


def test_confident_valid_field_is_not_flagged():
    result = finish_field("lrn", None, Reading("1234 5678 9012", 0.97), "m", 0.8)
    assert (result.raw, result.normalized) == ("1234 5678 9012", "123456789012")
    assert not result.needs_check


def test_low_confidence_is_flagged():
    result = finish_field("religion", None, Reading("Testfaith", 0.79), "m", 0.8)
    assert result.needs_check
    assert result.problem is None


def test_failing_validation_is_flagged_with_its_code():
    result = finish_field("contact_number", None, Reading("0917123", 0.99), "m", 0.8)
    assert (result.needs_check, result.problem) == (True, "invalid_contact_number")


def test_misread_required_field_keeps_its_own_problem():
    result = finish_field("birth_date", None, Reading("0910112018", 0.9), "m", 0.8)
    assert (result.needs_check, result.problem) == (True, "invalid_date")


def test_empty_required_field_is_flagged():
    result = finish_field("last_name", None, Reading("", None), "m", 0.8)
    assert (result.raw, result.needs_check, result.problem) == (None, True, "required")


def test_empty_optional_field_is_not_flagged():
    assert not finish_field("religion", None, Reading("", None), "m", 0.8).needs_check


def test_truncated_reading_is_flagged():
    result = finish_field(
        "address", None, Reading("Purok 7 Baguio", 0.99, truncated=True), "m", 0.8
    )
    assert (result.needs_check, result.problem) == (True, "truncated")


def test_guardian_is_flagged_only_when_no_parent_or_guardian_was_read():
    def run(father: str) -> dict:
        answers = {"guardian_name": "", "father_name": father}
        crops = [_crop(n, _written(v) if v else _blank()) for n, v in answers.items()]
        read = lambda c: Reading("Juan Testname", 0.95) if (c < 128).any() else Reading("", None)  # noqa: E731
        hand = FakeRecognizer("fake-hand-1", read=read)
        return {r.name: r for r in recognize_crops(crops, fake_engine(hand), 0.8)}

    assert run("Juan Testname")["guardian_name"].needs_check is False
    missing = run("")["guardian_name"]
    assert (missing.needs_check, missing.problem) == (True, "required")
