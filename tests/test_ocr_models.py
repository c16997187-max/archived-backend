"""Loading models from models/: checksums, threads, the handwriting setting, and once only."""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.ocr import model_files, models
from app.ocr.model_files import ModelsMissing, model_entry, sha256_file, write_manifest
from app.ocr.paddle import PaddleRecognizer
from tests.onnx_fakes import Arg, FakeSession

REC_YML = "PostProcess:\n  name: CTCLabelDecode\n  character_dict: ['a', 'b']\n"
DET_YML = "PostProcess:\n  thresh: 0.2\n  box_thresh: 0.45\n  unclip_ratio: 1.4\n"
TROCR_FILES = {
    "encoder_model.onnx": "fake",
    "decoder_model_merged.onnx": "fake",
    "tokenizer.json": "{}",
    "preprocessor_config.json": "{}",
    "generation_config.json": "{}",
}


def _install(models_dir: Path, name: str, files: dict[str, str]) -> None:
    directory = models_dir / name
    directory.mkdir(parents=True)
    for filename, content in files.items():
        (directory / filename).write_text(content, encoding="utf-8")
    try:
        entries = model_files.read_manifest(models_dir)["models"]
    except ModelsMissing:
        entries = {}
    entries[name] = {
        "version": f"{name}@test",
        "files": {f: sha256_file(directory / f) for f in files},
    }
    write_manifest(models_dir, entries)


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    _install(tmp_path, "ppocrv6_small_det", {"inference.onnx": "det", "inference.yml": DET_YML})
    _install(tmp_path, "ppocrv6_small_rec", {"inference.onnx": "rec", "inference.yml": REC_YML})
    return tmp_path


@pytest.fixture
def fake_sessions(monkeypatch):
    opened: list[Path] = []

    def open_session(path: Path, _options):
        opened.append(path)
        return FakeSession([Arg("x", [])], [Arg("out", ["b", "t", 4])], lambda _: [])

    monkeypatch.setattr(models, "open_session", open_session)
    return opened


class FakeTrOcr:
    def __init__(self, directory, options, model_version):
        self.directory = directory
        self.model_version = model_version


def test_missing_manifest_says_how_to_get_the_models(tmp_path):
    with pytest.raises(ModelsMissing) as error:
        model_entry(tmp_path, "ppocrv6_small_det")
    assert "scripts.download_models" in str(error.value)


def test_model_not_in_the_manifest_is_missing(models_dir):
    with pytest.raises(ModelsMissing, match="trocr"):
        model_entry(models_dir, "trocr_base_handwritten_onnx")


def test_missing_file_is_refused(models_dir):
    (models_dir / "ppocrv6_small_det" / "inference.onnx").unlink()
    with pytest.raises(ModelsMissing, match="missing"):
        model_entry(models_dir, "ppocrv6_small_det")


def test_changed_file_fails_its_checksum(models_dir):
    (models_dir / "ppocrv6_small_det" / "inference.onnx").write_text("tampered")
    with pytest.raises(ModelsMissing, match="checksum"):
        model_entry(models_dir, "ppocrv6_small_det")
    # Without verification, only presence is checked.
    assert model_entry(models_dir, "ppocrv6_small_det", verify=False)["version"]


def test_one_thread_per_physical_core(monkeypatch):
    monkeypatch.setattr(models.psutil, "cpu_count", lambda logical=True: 6)
    assert models.physical_cores() == 6
    options = models.session_options(models.physical_cores())
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (6, 1)


def test_threads_fall_back_to_half_the_logical_cores(monkeypatch):
    monkeypatch.setattr(models.psutil, "cpu_count", lambda logical=True: None)
    monkeypatch.setattr(models.os, "cpu_count", lambda: 8)
    assert models.physical_cores() == 4


def test_ppocr_handwriting_never_loads_trocr(models_dir, fake_sessions, monkeypatch):
    import app.ocr.trocr

    monkeypatch.setattr(app.ocr.trocr, "TrOcrRecognizer", FakeTrOcr)
    engine = models.load_engine(models_dir, "small", "ppocr", threads=2)
    assert isinstance(engine.handwriting, PaddleRecognizer)
    assert engine.handwriting is engine.printed
    assert engine.threads == 2
    assert all("trocr" not in str(path) for path in fake_sessions)


def test_trocr_handwriting_loads_from_its_folder(models_dir, fake_sessions, monkeypatch):
    import app.ocr.trocr

    _install(models_dir, model_files.TROCR_DIR, TROCR_FILES)
    monkeypatch.setattr(app.ocr.trocr, "TrOcrRecognizer", FakeTrOcr)
    engine = models.load_engine(models_dir, "small", "trocr", threads=2)
    assert engine.handwriting.directory == models_dir / model_files.TROCR_DIR
    assert engine.versions == {
        "detector": "ppocrv6_small_det@test",
        "printed": "ppocrv6_small_rec@test",
        "handwriting": "trocr_base_handwritten_onnx@test",
    }


def test_trocr_without_its_files_is_missing(models_dir, fake_sessions):
    with pytest.raises(ModelsMissing):
        models.load_engine(models_dir, "small", "trocr", threads=1)


def test_engine_loads_once_per_process(monkeypatch):
    loads = []
    monkeypatch.setattr(models, "_engine", None)
    monkeypatch.setattr(models, "load_engine", lambda *args: loads.append(args) or object())
    settings = Settings(ocr_handwriting_model="ppocr", ocr_threads=3)
    first = models.get_engine(settings)
    assert models.get_engine(settings) is first
    assert loads == [(settings.models_dir, "small", "ppocr", 3)]


def test_handwriting_setting_defaults_to_ppocr_and_rejects_others():
    assert Settings.model_fields["ocr_handwriting_model"].default == "ppocr"
    assert Settings(ocr_handwriting_model="trocr").ocr_handwriting_model == "trocr"
    with pytest.raises(ValidationError):
        Settings(ocr_handwriting_model="tesseract")
    with pytest.raises(ValidationError):
        Settings(ocr_confidence_threshold=1.5)


def test_runtime_code_never_imports_download_or_training_libraries():
    """The host is offline: only scripts/download_models.py may use these."""
    forbidden = {"huggingface_hub", "transformers", "optimum", "torch"}
    for path in Path("app").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not {n.split(".")[0] for n in names} & forbidden, path
