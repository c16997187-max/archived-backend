"""The model download script, with a fake downloader: nothing leaves this machine."""

import hashlib
import re
from pathlib import Path

import pytest

from app.ocr.model_files import ModelsMissing, model_entry, read_manifest
from scripts import download_models
from scripts.download_models import PADDLE, TROCR, ChecksumError, Source, fetch, main

CONTENT = {"inference.onnx": b"fake onnx", "inference.yml": b"PostProcess: {}\n"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeHub:
    """Serves `files` from memory and records what was asked for."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[tuple[str, str, str]] = []

    def __call__(self, repo: str, filename: str, revision: str, cache_dir: Path) -> Path:
        self.requests.append((repo, filename, revision))
        path = cache_dir / repo.replace("/", "--") / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename])
        return path


@pytest.fixture
def small_tier(monkeypatch):
    """The small tier, pinned to hashes of the fake files."""
    sources = {
        (kind, "small"): Source(
            f"Test/PP-OCRv6_small_{kind}_onnx",
            "0" * 40,
            "Apache-2.0",
            {name: _sha(data) for name, data in CONTENT.items()},
        )
        for kind in ("det", "rec")
    }
    monkeypatch.setattr(download_models, "PADDLE", sources)
    return sources


def test_fetch_accepts_a_matching_file(tmp_path, small_tier):
    path = fetch(small_tier[("det", "small")], "inference.onnx", tmp_path, FakeHub(CONTENT))
    assert path.read_bytes() == CONTENT["inference.onnx"]


def test_fetch_refuses_a_changed_file(tmp_path, small_tier):
    hub = FakeHub({**CONTENT, "inference.onnx": b"tampered"})
    with pytest.raises(ChecksumError, match="not installed"):
        fetch(small_tier[("det", "small")], "inference.onnx", tmp_path, hub)


def test_install_writes_a_manifest_the_runtime_accepts(tmp_path, small_tier):
    hub = FakeHub(CONTENT)
    assert main(["--models-dir", str(tmp_path), "--skip-trocr"], download=hub) == 0

    for name in ("ppocrv6_small_det", "ppocrv6_small_rec"):
        entry = model_entry(tmp_path, name)  # verifies every checksum
        assert set(entry["files"]) == set(CONTENT)
        assert entry["source"].endswith("@" + "0" * 40)
        assert entry["license"] == "Apache-2.0"
    assert {revision for _, _, revision in hub.requests} == {"0" * 40}


def test_second_run_downloads_nothing(tmp_path, small_tier, capsys):
    main(["--models-dir", str(tmp_path), "--skip-trocr"], download=FakeHub(CONTENT))
    hub = FakeHub(CONTENT)
    assert main(["--models-dir", str(tmp_path), "--skip-trocr"], download=hub) == 0
    assert hub.requests == []
    assert "already installed" in capsys.readouterr().out


def test_damaged_install_is_downloaded_again(tmp_path, small_tier):
    main(["--models-dir", str(tmp_path), "--skip-trocr"], download=FakeHub(CONTENT))
    (tmp_path / "ppocrv6_small_det" / "inference.onnx").write_bytes(b"corrupted")
    hub = FakeHub(CONTENT)
    main(["--models-dir", str(tmp_path), "--skip-trocr"], download=hub)
    assert {repo for repo, _, _ in hub.requests} == {"Test/PP-OCRv6_small_det_onnx"}
    model_entry(tmp_path, "ppocrv6_small_det")


def test_checksum_failure_installs_nothing(tmp_path, small_tier, capsys):
    hub = FakeHub({**CONTENT, "inference.yml": b"changed upstream"})
    assert main(["--models-dir", str(tmp_path), "--skip-trocr"], download=hub) == 1
    assert "SHA-256" in capsys.readouterr().err
    with pytest.raises(ModelsMissing):
        model_entry(tmp_path, "ppocrv6_small_det")


def test_unknown_tier_is_refused(tmp_path):
    assert main(["--models-dir", str(tmp_path), "--tiers", "huge"], download=FakeHub({})) == 2
    assert not (tmp_path / "manifest.json").exists()


def test_every_real_source_is_pinned_to_a_commit_and_hashes():
    for source in [*PADDLE.values(), TROCR]:
        assert re.fullmatch(r"[0-9a-f]{40}", source.revision), source.repo
        assert source.files
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in source.files.values()), source.repo
    assert "pytorch_model.bin" not in TROCR.files  # the same weights, pickled


def test_manifest_format_is_checked(tmp_path):
    (tmp_path / "manifest.json").write_text('{"format": 99, "models": {}}')
    with pytest.raises(ModelsMissing, match="manifest"):
        read_manifest(tmp_path)
