"""TrOCR base-handwritten on ONNX Runtime: one line of handwriting per crop (Phase 7).

`scripts/download_models.py` exports the encoder (kept FP32) and a merged INT8 decoder (one
graph for the first step and the cached steps). Decoding is greedy, batched over every line
of a form, with the decoder's key/value cache; the encoder runs in small chunks to bound
memory.

Confidence: TrOCR emits BPE tokens, not characters, so each character takes the probability
of the token it came from, and the reading's confidence is the lowest token probability
(the end-of-text token included: an unsure ending is an unsure reading).
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tokenizers import Tokenizer

from app.ocr.text_reader import Reading

ENCODER_FILE = "encoder_model.onnx"
DECODER_FILE = "decoder_model_merged.onnx"
DEFAULT_MAX_TOKENS = 64
# Lines per encoder call. The encoder's memory grows with the batch (about 0.15 GB a line
# at FP32) while its speed per line barely changes, so it runs in small chunks; decoding
# then runs over all lines at once.
ENCODER_BATCH = 4
# Past-key-value inputs on the first step, where the merged graph ignores them.
_DUMMY_PAST_LENGTH = 1


def _softmax_max(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(best token, its probability) for each row of a (batch, vocab) array."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    best = probs.argmax(axis=-1)
    return best, probs[np.arange(len(best)), best]


class TrOcrRecognizer:
    def __init__(
        self,
        directory: Path,
        options: Any = None,
        model_version: str = "trocr",
        encoder: Any = None,
        decoder: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """Sessions and tokenizer can be passed in (tests use small fakes); otherwise they
        are opened from `directory`."""
        if encoder is None or decoder is None:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            encoder = encoder or ort.InferenceSession(
                str(directory / ENCODER_FILE), options, providers=providers
            )
            decoder = decoder or ort.InferenceSession(
                str(directory / DECODER_FILE), options, providers=providers
            )
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer or Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.model_version = model_version

        preprocessor = json.loads((directory / "preprocessor_config.json").read_text("utf-8"))
        generation = json.loads((directory / "generation_config.json").read_text("utf-8"))
        size = preprocessor.get("size", 384)
        self.size = size if isinstance(size, int) else int(size.get("height", 384))
        self.mean = np.float32(preprocessor.get("image_mean", [0.5, 0.5, 0.5]))
        self.std = np.float32(preprocessor.get("image_std", [0.5, 0.5, 0.5]))
        self.start_id = int(generation["decoder_start_token_id"])
        self.eos_id = int(generation["eos_token_id"])
        self.pad_id = int(generation.get("pad_token_id", 1))

        names = [i.name for i in decoder.get_inputs()]
        self.past_names = [n for n in names if n.startswith("past_key_values.")]
        self.has_branch_flag = "use_cache_branch" in names
        self.present_names = [o.name for o in decoder.get_outputs()][1:]
        heads_input = next(i for i in decoder.get_inputs() if i.name in self.past_names)
        self.heads = int(heads_input.shape[1])
        self.head_dim = int(heads_input.shape[3])

    def pixel_values(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """RGB, square, scaled to [-1, 1]: what the ViT image processor does."""
        batch = np.empty((len(crops), 3, self.size, self.size), dtype=np.float32)
        for i, crop in enumerate(crops):
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB if crop.ndim == 2 else cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
            normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
            batch[i] = normalized.transpose(2, 0, 1)
        return batch

    def _step(
        self,
        tokens: np.ndarray,
        hidden: np.ndarray,
        past: dict[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        feed: dict[str, Any] = {"input_ids": tokens, "encoder_hidden_states": hidden}
        if past is None:
            dummy = np.zeros(
                (len(tokens), self.heads, _DUMMY_PAST_LENGTH, self.head_dim), dtype=np.float32
            )
            feed.update({name: dummy for name in self.past_names})
        else:
            feed.update(past)
        if self.has_branch_flag:
            feed["use_cache_branch"] = np.array([past is not None])
        outputs = self.decoder.run(None, feed)
        presents = dict(zip(self.present_names, outputs[1:], strict=True))
        new_past = {}
        for name in self.past_names:
            # The encoder's keys and values never change, and the cached branch of the merged
            # graph returns placeholders for them: keep the first step's.
            if past is not None and ".encoder." in name:
                new_past[name] = past[name]
            else:
                new_past[name] = presents[name.replace("past_key_values.", "present.")]
        return outputs[0][:, -1, :], new_past

    def recognize_batch(
        self, crops: Sequence[np.ndarray], max_tokens: Sequence[int] | None = None
    ) -> list[Reading]:
        """Greedy decoding of every crop at once. A row stops at end-of-text or at its own
        `max_tokens`; a row that reaches the limit is marked truncated."""
        if not crops:
            return []
        limits = np.array(
            list(max_tokens) if max_tokens is not None else [DEFAULT_MAX_TOKENS] * len(crops)
        )
        hidden = np.concatenate(
            [
                self.encoder.run(
                    None, {"pixel_values": self.pixel_values(crops[i : i + ENCODER_BATCH])}
                )[0]
                for i in range(0, len(crops), ENCODER_BATCH)
            ]
        )
        rows = len(crops)
        tokens = np.full((rows, 1), self.start_id, dtype=np.int64)
        generated: list[list[int]] = [[] for _ in range(rows)]
        scores: list[list[float]] = [[] for _ in range(rows)]
        done = np.zeros(rows, dtype=bool)
        truncated = np.zeros(rows, dtype=bool)
        past = None
        for step in range(int(limits.max())):
            logits, past = self._step(tokens, hidden, past)
            best, prob = _softmax_max(logits)
            for row in np.flatnonzero(~done):
                scores[row].append(float(prob[row]))
                if best[row] == self.eos_id:
                    done[row] = True
                else:
                    generated[row].append(int(best[row]))
                    if step + 1 >= limits[row]:
                        done[row] = truncated[row] = True
            if done.all():
                break
            tokens = np.where(done, self.pad_id, best).astype(np.int64).reshape(rows, 1)
        readings = []
        for row in range(rows):
            text = self.tokenizer.decode(generated[row], skip_special_tokens=True).strip()
            confidence = min(scores[row]) if text and scores[row] else None
            readings.append(Reading(text, confidence, bool(truncated[row])))
        return readings
