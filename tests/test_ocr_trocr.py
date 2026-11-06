"""TrOCR decoding with fake ONNX sessions: batching, stopping, confidence and truncation."""

import json
import math

import numpy as np
import pytest

from app.ocr.trocr import TrOcrRecognizer
from tests.onnx_fakes import EOS, PAD, START, VOCAB, FakeTokenizer, tokens, trocr_sessions


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"size": 384, "image_mean": [0.5] * 3, "image_std": [0.5] * 3})
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"decoder_start_token_id": START, "eos_token_id": EOS, "pad_token_id": PAD})
    )
    return tmp_path


def _recognizer(model_dir, scripts, strength=None):
    encoder, decoder = trocr_sessions(scripts, strength)
    recognizer = TrOcrRecognizer(
        model_dir, encoder=encoder, decoder=decoder, tokenizer=FakeTokenizer(), model_version="t"
    )
    return recognizer, encoder, decoder


def _crops(n: int) -> list[np.ndarray]:
    return [np.full((40, 200), 255, np.uint8) for _ in range(n)]


def test_rows_of_different_lengths_decode_in_one_batch(model_dir):
    recognizer, encoder, decoder = _recognizer(model_dir, [tokens("ana"), tokens("jo"), []])

    readings = recognizer.recognize_batch(_crops(3))

    assert [r.text for r in readings] == ["ana", "jo", ""]
    assert not any(r.truncated for r in readings)
    assert len(encoder.feeds) == 1  # every crop through the encoder at once
    assert len(decoder.feeds) == 4  # the longest row's three tokens, then end-of-text
    assert readings[2].confidence is None  # nothing read


def test_encoder_runs_in_small_chunks_and_decoding_in_one_batch(model_dir):
    scripts = [tokens(c) for c in "abcdef"]
    recognizer, encoder, decoder = _recognizer(model_dir, scripts)

    readings = recognizer.recognize_batch(_crops(6))

    assert [len(feed["pixel_values"]) for feed in encoder.feeds] == [4, 2]
    assert len(decoder.feeds[0]["input_ids"]) == 6
    assert [r.text for r in readings] == list("abcdef")


def test_finished_rows_are_fed_padding(model_dir):
    recognizer, _, decoder = _recognizer(model_dir, [tokens("ana"), tokens("j")])
    recognizer.recognize_batch(_crops(2))
    assert decoder.feeds[2]["input_ids"][1, 0] == PAD


def test_confidence_is_the_least_sure_token_end_of_text_included(model_dir):
    sure, unsure = 20.0, 2.0
    recognizer, _, _ = _recognizer(model_dir, [tokens("ab")], [[sure, unsure, sure]])

    [reading] = recognizer.recognize_batch(_crops(1))

    expected = math.exp(unsure) / (math.exp(unsure) + VOCAB - 1)
    assert reading.text == "ab"
    assert reading.confidence == pytest.approx(expected, rel=1e-4)


def test_unsure_ending_lowers_confidence(model_dir):
    recognizer, _, _ = _recognizer(model_dir, [tokens("ab")], [[20.0, 20.0, 1.0]])
    [reading] = recognizer.recognize_batch(_crops(1))
    assert reading.confidence < 0.2


def test_hitting_the_token_limit_marks_the_reading_truncated(model_dir):
    recognizer, _, _ = _recognizer(model_dir, [tokens("abcdefgh"), tokens("ab")])

    long, short = recognizer.recognize_batch(_crops(2), max_tokens=[3, 3])

    assert (long.text, long.truncated) == ("abc", True)
    assert (short.text, short.truncated) == ("ab", False)


def test_each_row_has_its_own_limit(model_dir):
    recognizer, _, _ = _recognizer(model_dir, [tokens("abcdef"), tokens("abcdef")])
    first, second = recognizer.recognize_batch(_crops(2), max_tokens=[2, 10])
    assert (first.text, first.truncated) == ("ab", True)
    assert (second.text, second.truncated) == ("abcdef", False)


def test_pixel_values_are_square_rgb_in_minus_one_to_one(model_dir):
    recognizer, encoder, _ = _recognizer(model_dir, [[]])
    crop = np.zeros((40, 300), np.uint8)
    crop[:, 150:] = 255
    recognizer.recognize_batch([crop])
    pixels = encoder.feeds[0]["pixel_values"]
    assert pixels.shape == (1, 3, 384, 384)
    assert pixels.min() == pytest.approx(-1.0) and pixels.max() == pytest.approx(1.0)


def test_no_crops_runs_nothing(model_dir):
    recognizer, encoder, _ = _recognizer(model_dir, [])
    assert recognizer.recognize_batch([]) == []
    assert encoder.feeds == []
