"""Small stand-ins for ONNX Runtime sessions, so tests never need the real models."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Arg:
    name: str
    shape: list[Any]


class FakeSession:
    """Answers `run` with `respond(feed)`; records every feed."""

    def __init__(
        self,
        inputs: list[Arg],
        outputs: list[Arg],
        respond: Callable[[dict[str, np.ndarray]], list[np.ndarray]],
    ) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self.respond = respond
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[Arg]:
        return self._inputs

    def get_outputs(self) -> list[Arg]:
        return self._outputs

    def run(self, _names, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        return self.respond(feed)


# ------------------------------------------------------------------- TrOCR

EOS = 2
START = 2
PAD = 1
VOCAB = 40
HEADS, HEAD_DIM, ENCODER_LENGTH = 2, 4, 5
FIRST_LETTER_ID = 10  # token 10 is "a", 11 is "b", ...


class FakeTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(ord("a") + i - FIRST_LETTER_ID) for i in ids if i >= FIRST_LETTER_ID)


def tokens(text: str) -> list[int]:
    return [FIRST_LETTER_ID + ord(c) - ord("a") for c in text]


def trocr_sessions(scripts: list[list[int]], strength: list[list[float]] | None = None):
    """An encoder and a merged decoder that emit `scripts[row]`, then end-of-text, one token
    per step. `strength[row][step]` sets how sure each step is (default: very sure).

    Like the real merged graph, the cached branch returns placeholder encoder key/values;
    the decoder checks it is always given the first step's.
    """
    layer = ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]
    past_names = [f"past_key_values.0.{part}" for part in layer]
    present_names = [f"present.0.{part}" for part in layer]
    encoder_marker = np.full((len(scripts), HEADS, ENCODER_LENGTH, HEAD_DIM), 7.0, np.float32)

    def encode(feed):
        return [np.zeros((len(feed["pixel_values"]), ENCODER_LENGTH, 8), np.float32)]

    def decode(feed):
        rows = len(feed["input_ids"])
        cached = bool(feed["use_cache_branch"][0])
        past = feed["past_key_values.0.decoder.key"]
        step = past.shape[2] if cached else 0
        if cached:
            assert np.array_equal(feed["past_key_values.0.encoder.key"], encoder_marker[:rows])
        logits = np.zeros((rows, 1, VOCAB), np.float32)
        for row in range(rows):
            script = scripts[row] + [EOS]
            target = script[min(step, len(script) - 1)]
            sure = strength[row][step] if strength and step < len(strength[row]) else 20.0
            logits[row, 0, target] = sure
        decoder_kv = np.zeros((rows, HEADS, step + 1, HEAD_DIM), np.float32)
        encoder_kv = (
            np.zeros((rows, HEADS, 1, HEAD_DIM), np.float32) if cached else (encoder_marker[:rows])
        )
        return [logits, decoder_kv, decoder_kv, encoder_kv, encoder_kv]

    kv_shape = ["batch", HEADS, "length", HEAD_DIM]
    encoder = FakeSession(
        [Arg("pixel_values", ["batch", 3, 384, 384])],
        [Arg("last_hidden_state", ["batch", ENCODER_LENGTH, 8])],
        encode,
    )
    decoder = FakeSession(
        [
            Arg("input_ids", ["batch", "length"]),
            Arg("encoder_hidden_states", ["batch", ENCODER_LENGTH, 8]),
            *(Arg(name, kv_shape) for name in past_names),
            Arg("use_cache_branch", [1]),
        ],
        [Arg("logits", ["batch", "length", VOCAB]), *(Arg(n, kv_shape) for n in present_names)],
        decode,
    )
    return encoder, decoder
