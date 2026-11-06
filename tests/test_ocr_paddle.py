"""PP-OCRv6 pre- and post-processing with fake ONNX sessions."""

import numpy as np
import pytest

from app.ocr.paddle import (
    DET_STRIDE,
    REC_HEIGHT,
    REC_MIN_WIDTH,
    PaddleDetector,
    PaddleRecognizer,
    PaddleTextReader,
    charset_for,
    ctc_decode,
    det_inputs,
    rec_inputs,
    regions_from_map,
)
from app.ocr.text_reader import Region
from tests.onnx_fakes import Arg, FakeSession

CHARSET = ["", "a", "b", "c", " "]


def _probs(steps: list[tuple[int, float]], classes: int = len(CHARSET)) -> np.ndarray:
    """One row: the given class wins each time step with the given probability."""
    probs = np.zeros((len(steps), classes), np.float32)
    for t, (index, p) in enumerate(steps):
        probs[t] = (1 - p) / (classes - 1)
        probs[t, index] = p
    return probs


def test_ctc_collapses_repeats_and_drops_blanks():
    row = _probs([(1, 0.9), (1, 0.9), (0, 0.99), (1, 0.8), (2, 0.95), (0, 0.9), (3, 0.7)])
    [reading] = ctc_decode(row[None], CHARSET)
    assert reading.text == "aabc"
    assert reading.confidence == pytest.approx(0.7)  # the least sure character


def test_ctc_blank_steps_do_not_lower_confidence():
    row = _probs([(0, 0.3), (1, 0.9), (0, 0.2)])
    assert ctc_decode(row[None], CHARSET)[0].confidence == pytest.approx(0.9)


def test_ctc_nothing_read_has_no_confidence():
    [reading] = ctc_decode(_probs([(0, 0.99), (4, 0.9)])[None], CHARSET)
    assert (reading.text, reading.confidence) == ("", None)


def test_charset_adds_the_blank_and_the_space_the_model_has_room_for():
    assert charset_for(["a", "b"], 4) == ["", "a", "b", " "]
    assert charset_for(["a", "b"], 3) == ["", "a", "b"]
    with pytest.raises(ValueError):
        charset_for(["a", "b"], 7)


def test_detector_input_pads_to_a_shared_multiple_of_32_with_white():
    small = np.zeros((50, 70), np.uint8)
    wide = np.zeros((40, 130), np.uint8)
    batch, scales = det_inputs([small, wide])
    assert batch.shape == (2, 3, 64, 160)
    assert batch.shape[2] % DET_STRIDE == 0 and batch.shape[3] % DET_STRIDE == 0
    assert scales == [1.0, 1.0]
    white = batch[0, :, 60, 100]  # padding, outside the small image
    black = batch[0, :, 10, 10]
    assert (white > black).all()


def test_large_pages_are_scaled_down():
    _, [scale] = det_inputs([np.zeros((3200, 1600), np.uint8)])
    assert scale == pytest.approx(0.5)


def test_db_postprocess_boxes_each_blob_and_scales_back():
    prob = np.zeros((100, 200), np.float32)
    prob[10:20, 20:80] = 0.9  # a word
    prob[50:60, 100:180] = 0.8  # another
    prob[90:92, 5:7] = 0.9  # a speck
    regions = regions_from_map(prob, 0.5, (400, 200), 0.3, 0.6, 1.5, 100)
    assert len(regions) == 2
    first, second = regions
    assert first.y < second.y
    # Scaled back to the original (2x) and grown a little around the blob.
    assert 20 < first.x < 40 and 158 < first.x + first.w < 180
    assert first.y < 20 and first.y + first.h > 38
    assert first.score == pytest.approx(0.9)


def test_db_postprocess_drops_faint_blobs():
    prob = np.zeros((50, 50), np.float32)
    prob[10:30, 10:30] = 0.4
    assert regions_from_map(prob, 1.0, (50, 50), 0.3, 0.6, 1.5, 100) == []


def test_recognizer_input_keeps_aspect_and_pads_to_the_widest():
    batch = rec_inputs([np.zeros((24, 48), np.uint8), np.zeros((48, 960), np.uint8)])
    assert batch.shape == (2, 3, REC_HEIGHT, 960)
    assert (batch[0, :, :, 96:] == 0).all()  # the short crop's padding
    assert rec_inputs([np.zeros((48, 10), np.uint8)]).shape[3] == REC_MIN_WIDTH


def _rec_session(answers: dict[int, list[tuple[int, float]]]) -> FakeSession:
    """Reads each crop by its width, so the test can tell whether order survives sorting."""

    def respond(feed):
        batch = feed["x"]
        rows = []
        for image in batch:
            used = int((np.abs(image).sum(axis=(0, 1)) > 0).sum())
            rows.append(_probs(answers[used]))
        return [np.stack(rows)]

    return FakeSession([Arg("x", ["b", 3, 48, "w"])], [Arg("out", ["b", "t", 5])], respond)


def test_recognizer_returns_readings_in_the_order_given():
    answers = {400: [(1, 0.9)], 100: [(2, 0.9)], 200: [(3, 0.9)]}
    recognizer = PaddleRecognizer(_rec_session(answers), CHARSET, "rec-1")
    crops = [np.full((48, w), 200, np.uint8) for w in (400, 100, 200)]
    assert [r.text for r in recognizer.recognize_batch(crops)] == ["a", "b", "c"]


def test_text_reader_crops_each_region_and_skips_empty_reads():
    detector = PaddleDetector(None, {}, "det-1")
    answers = {120: [(1, 0.9)], 80: [(0, 0.9)]}
    recognizer = PaddleRecognizer(_rec_session(answers), CHARSET, "rec-1")
    reader = PaddleTextReader(detector, recognizer)
    page = np.full((200, 400), 200, np.uint8)
    regions = [Region(10, 10, 120, 48, 0.9), Region(10, 100, 80, 48, 0.9)]
    boxes = reader.recognize(page, regions)
    assert [(b.text, b.x, b.y) for b in boxes] == [("a", 10, 10)]
    assert reader.model_version == "det-1+rec-1"


def test_detector_runs_one_batch_for_many_images():
    def respond(feed):
        batch = feed["x"]
        maps = np.zeros((len(batch), 1, *batch.shape[2:]), np.float32)
        maps[:, 0, 5:15, 5:25] = 0.9
        return [maps]

    session = FakeSession([Arg("x", ["b", 3, "h", "w"])], [Arg("m", ["b", 1, "h", "w"])], respond)
    detector = PaddleDetector(session, {"thresh": 0.3, "box_thresh": 0.5}, "det-1")
    results = detector.detect_batch([np.zeros((40, 100), np.uint8), np.zeros((60, 50), np.uint8)])
    assert len(session.feeds) == 1
    assert [len(r) for r in results] == [1, 1]
