"""The model files under `models/` and their checksums (Phase 7).

`scripts/download_models.py` writes `models/manifest.json`; the worker reads it and checks
every file before loading. Standard library only, so the download script can share it.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = 1
TIERS = ("tiny", "small", "medium")

TROCR_DIR = "trocr_base_handwritten_onnx"
DOWNLOAD_HINT = (
    "Run `python -m scripts.download_models` on a machine with internet access, "
    "then copy models/ here."
)


class ModelsMissing(RuntimeError):
    """A model file is missing or does not match its checksum. The message says what to do."""


def paddle_dir(kind: str, tier: str) -> str:
    """`kind` is "det" or "rec"."""
    return f"ppocrv6_{tier}_{kind}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(models_dir: Path) -> dict[str, Any]:
    path = models_dir / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ModelsMissing(f"No {path}. {DOWNLOAD_HINT}") from None
    except (OSError, ValueError) as exc:
        raise ModelsMissing(f"Cannot read {path}: {exc}. {DOWNLOAD_HINT}") from None
    if data.get("format") != MANIFEST_FORMAT or not isinstance(data.get("models"), dict):
        raise ModelsMissing(f"{path} is not a manifest this version understands. {DOWNLOAD_HINT}")
    return data


def write_manifest(models_dir: Path, models: dict[str, Any]) -> None:
    path = models_dir / MANIFEST_NAME
    data = {"format": MANIFEST_FORMAT, "models": dict(sorted(models.items()))}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def model_entry(models_dir: Path, name: str, verify: bool = True) -> dict[str, Any]:
    """The manifest entry of one model directory, after checking its files are there and,
    with `verify`, that they match their checksums."""
    entry = read_manifest(models_dir)["models"].get(name)
    if entry is None:
        raise ModelsMissing(f"models/{name} is not installed. {DOWNLOAD_HINT}")
    for filename, expected in entry["files"].items():
        path = models_dir / name / filename
        if not path.is_file():
            raise ModelsMissing(f"{path} is missing. {DOWNLOAD_HINT}")
        if verify and sha256_file(path) != expected:
            raise ModelsMissing(f"{path} does not match its checksum. {DOWNLOAD_HINT}")
    return entry
