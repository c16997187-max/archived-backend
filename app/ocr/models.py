"""Load the OCR models once per worker process (Phase 7).

Everything comes from `models/`, checked against `models/manifest.json`; nothing is ever
downloaded here. ONNX Runtime gets one thread per physical core, so the API process keeps
the hyper-threads and stays responsive while a form is being read.
"""

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort
import psutil

from app.config import Settings
from app.ocr.model_files import TROCR_DIR, model_entry, paddle_dir
from app.ocr.paddle import PaddleDetector, PaddleRecognizer, PaddleTextReader
from app.ocr.text_reader import LineRecognizer

logger = logging.getLogger(__name__)

HANDWRITING_MODELS = ("trocr", "ppocr")


@dataclass(frozen=True)
class OcrEngine:
    detector: PaddleDetector
    printed: PaddleRecognizer
    handwriting: LineRecognizer
    text_reader: PaddleTextReader
    threads: int

    @property
    def versions(self) -> dict[str, str]:
        return {
            "detector": self.detector.model_version,
            "printed": self.printed.model_version,
            "handwriting": self.handwriting.model_version,
        }


def physical_cores() -> int:
    cores = psutil.cpu_count(logical=False)
    if cores:
        return cores
    return max(1, (os.cpu_count() or 2) // 2)


def session_options(threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def open_session(path: Path, options: ort.SessionOptions) -> Any:
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def load_engine(
    models_dir: Path,
    tier: str = "small",
    handwriting: str = "ppocr",
    threads: int | None = None,
    verify: bool = True,
) -> OcrEngine:
    """Raises `ModelsMissing` when a file is absent or fails its checksum. With
    `handwriting="ppocr"` TrOCR is not loaded at all."""
    if handwriting not in HANDWRITING_MODELS:
        raise ValueError(f"handwriting must be one of {HANDWRITING_MODELS}")
    threads = threads or physical_cores()
    options = session_options(threads)

    det_name, rec_name = paddle_dir("det", tier), paddle_dir("rec", tier)
    det_entry = model_entry(models_dir, det_name, verify)
    rec_entry = model_entry(models_dir, rec_name, verify)
    detector = PaddleDetector.from_dir(
        models_dir / det_name,
        open_session(models_dir / det_name / "inference.onnx", options),
        det_entry["version"],
    )
    printed = PaddleRecognizer.from_dir(
        models_dir / rec_name,
        open_session(models_dir / rec_name / "inference.onnx", options),
        rec_entry["version"],
    )
    reader: LineRecognizer = printed
    if handwriting == "trocr":
        from app.ocr.trocr import TrOcrRecognizer

        entry = model_entry(models_dir, TROCR_DIR, verify)
        reader = TrOcrRecognizer(models_dir / TROCR_DIR, options, entry["version"])
    logger.info(
        "OCR models loaded: %s, %s, handwriting %s, %d threads",
        detector.model_version,
        printed.model_version,
        reader.model_version,
        threads,
    )
    return OcrEngine(detector, printed, reader, PaddleTextReader(detector, printed), threads)


_engine: OcrEngine | None = None
_engine_lock = threading.Lock()


def get_engine(settings: Settings) -> OcrEngine:
    """The process's one engine, loaded on first use. The worker calls this at start-up."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = load_engine(
                settings.models_dir,
                settings.ocr_det_tier,
                settings.ocr_handwriting_model,
                settings.ocr_threads,
            )
        return _engine
