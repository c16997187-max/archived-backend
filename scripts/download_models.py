r"""Download the OCR models once into models/, check them, and export TrOCR to ONNX INT8.

Run on a development machine with internet access, in its own virtual environment (the
export tools are large and pin an older transformers):

    py -V:3.12 -m venv .venv-models
    .venv-models\Scripts\pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-models.txt
    .venv-models\Scripts\python -m scripts.download_models
    .venv-models\Scripts\python -m scripts.download_models --tiers tiny,small,medium

Then copy models/ to the host PC with the install. The host never downloads anything: the
worker loads only from models/ and checks every file against models/manifest.json.

Every download is pinned to a commit and checked against a SHA-256 written below, so a
changed or tampered file is refused. Running it again skips models that are already
installed and intact.
"""

import argparse
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ocr.model_files import (
    TIERS,
    TROCR_DIR,
    ModelsMissing,
    model_entry,
    paddle_dir,
    read_manifest,
    sha256_file,
    write_manifest,
)

DOWNLOADS_DIR = "_downloads"


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    license: str
    files: dict[str, str]  # file name -> SHA-256


def _paddle(kind: str, tier: str, revision: str, onnx: str, yml: str) -> Source:
    return Source(
        f"PaddlePaddle/PP-OCRv6_{tier}_{kind}_onnx",
        revision,
        "Apache-2.0",
        {"inference.onnx": onnx, "inference.yml": yml},
    )


# Hashes from the Hugging Face file metadata at each pinned commit (checked 2026-11-07).
PADDLE: dict[tuple[str, str], Source] = {
    ("det", "tiny"): _paddle(
        "det", "tiny", "2ba1506c0380b8f0b03dd142459aac66d4421f6c",
        "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
        "3ac018be6f97499a08faa3bbdeb33640968d9307f6736d152902747a9f259593",
    ),
    ("rec", "tiny"): _paddle(
        "rec", "tiny", "2612ab37152ae0a677521bae4e1e3d4fb4cf7c30",
        "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6",
        "66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1",
    ),
    ("det", "small"): _paddle(
        "det", "small", "28fe5895c24fd108c19eb3e8479f4ab385fbfc62",
        "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
        "193f435274bf9f0b5f71a929bbfbcf148282df7e633b34e7c373e8f44741b516",
    ),
    ("rec", "small"): _paddle(
        "rec", "small", "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
        "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
        "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1",
    ),
    ("det", "medium"): _paddle(
        "det", "medium", "61323801669c338b7891481ec7bac61ce31b576a",
        "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
        "7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871",
    ),
    ("rec", "medium"): _paddle(
        "rec", "medium", "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
        "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba",
        "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
    ),
}  # fmt: skip

# The safetensors weights and configs only; pytorch_model.bin is the same weights, pickled.
TROCR = Source(
    "microsoft/trocr-base-handwritten",
    "eaacaf452b06415df8f10bb6fad3a4c11e609406",
    "MIT (fine-tuned on IAM: see README)",
    {
        "model.safetensors": "25a40cddc7e6120140a3d5b9e3dd3878a92ada7b4f312953ab22edc19c2a5acc",
        "config.json": "4c779f24e063c437c3dafd5b2e6c9f59f2fa2bd1dbb4ae6a30153bbbbf19e647",
        "generation_config.json": "41149cdcffec4d657f32dfcddd9b208037f01286c9e07945c724908c58ed0193",  # noqa: E501
        "preprocessor_config.json": "2fcc0da9466ee00be0403b26027373039e032820ebac409e207b32e52e52119d",  # noqa: E501
        "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        "vocab.json": "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
        "special_tokens_map.json": "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",  # noqa: E501
        "tokenizer_config.json": "5a1356884c6ae736a621841535264ba7c5bebd52f169258add2c48fcbb32d50a",
    },
)  # fmt: skip

# Copied next to the ONNX files: the runtime reads them, never the Hugging Face cache.
TROCR_RUNTIME_CONFIGS = ("config.json", "generation_config.json", "preprocessor_config.json")
TROCR_ONNX = ("encoder_model.onnx", "decoder_model_merged.onnx")
# What the exporter writes for the decoder: the first step, and the steps with a cache.
TROCR_DECODER_PARTS = ("decoder_model.onnx", "decoder_with_past_model.onnx")

Downloader = Callable[[str, str, str, Path], Path]


class ChecksumError(RuntimeError):
    pass


def hf_download(repo: str, filename: str, revision: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache_dir))


def fetch(source: Source, filename: str, cache_dir: Path, download: Downloader) -> Path:
    """Download one pinned file and refuse it unless it matches its SHA-256."""
    path = download(source.repo, filename, source.revision, cache_dir)
    actual = sha256_file(path)
    if actual != source.files[filename]:
        raise ChecksumError(
            f"{source.repo}/{filename} has SHA-256 {actual}, expected {source.files[filename]}. "
            "The file changed upstream or was corrupted; it was not installed."
        )
    return path


def _entry(models_dir: Path, name: str, source: Source, version: str) -> dict[str, Any]:
    directory = models_dir / name
    return {
        "version": version,
        "source": f"{source.repo}@{source.revision}",
        "license": source.license,
        "files": {p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()},
    }


def _installed(models_dir: Path, name: str) -> bool:
    try:
        model_entry(models_dir, name, verify=True)
    except ModelsMissing:
        return False
    return True


def install_paddle(
    kind: str, tier: str, models_dir: Path, cache_dir: Path, download: Downloader
) -> dict[str, Any]:
    source = PADDLE[(kind, tier)]
    name = paddle_dir(kind, tier)
    target = models_dir / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for filename in source.files:
        shutil.copyfile(fetch(source, filename, cache_dir, download), target / filename)
    return _entry(models_dir, name, source, f"PP-OCRv6_{tier}_{kind}@{source.revision[:7]}")


def export_trocr(models_dir: Path, cache_dir: Path, download: Downloader) -> dict[str, Any]:
    """Export the encoder and decoder to ONNX, quantize the decoder's weights to INT8, and
    merge its first-step and cached-step graphs into one. Needs requirements-models.txt.

    Quantizing comes before merging because the quantizer skips the merged graph's If
    branches, which would leave the decoder at full size. The encoder stays FP32: on
    generated forms every INT8 encoder tried (all ops or MatMul only, per-channel or not,
    reduced range) read fewer fields right than FP32 (best 11 of 34 against 15 of 34),
    while the INT8 decoder gave the same text as FP32. The decoder runs once per token,
    the encoder once per line, so the decoder is where INT8 saves the time."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.exporters.onnx import main_export
    from optimum.onnx import merge_decoders
    from transformers import AutoTokenizer

    snapshot = None
    for filename in TROCR.files:
        snapshot = fetch(TROCR, filename, cache_dir, download).parent
    assert snapshot is not None

    target = models_dir / TROCR_DIR
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=models_dir) as work:
        fp32, int8 = Path(work) / "fp32", Path(work) / "int8"
        int8.mkdir()
        print("Exporting TrOCR to ONNX (a few minutes)...")
        main_export(str(snapshot), output=fp32, task="image-to-text-with-past", device="cpu")
        shutil.copyfile(fp32 / TROCR_ONNX[0], target / TROCR_ONNX[0])
        for filename in TROCR_DECODER_PARTS:
            print(f"Quantizing {filename} to INT8...")
            quantize_dynamic(
                fp32 / filename, int8 / filename, weight_type=QuantType.QInt8, per_channel=True
            )
        merge_decoders(
            int8 / TROCR_DECODER_PARTS[0],
            int8 / TROCR_DECODER_PARTS[1],
            save_path=target / TROCR_ONNX[1],
            strict=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot).backend_tokenizer
        for directory in (target, fp32):
            tokenizer.save(str(directory / "tokenizer.json"))
            for filename in TROCR_RUNTIME_CONFIGS:
                shutil.copyfile(snapshot / filename, directory / filename)
        smoke_check(fp32, target)
    version = f"trocr-base-handwritten@{TROCR.revision[:7]}-int8dec"
    return _entry(models_dir, TROCR_DIR, TROCR, version)


def smoke_check(fp32_dir: Path, int8_dir: Path) -> None:
    """Read a few generated lines with the full-FP32 export and the installed one, and
    print how far quantization drifts."""
    import cv2
    import numpy as np

    from app.ocr.trocr import TrOcrRecognizer
    from scripts.eval_ocr import edit_distance

    lines = ["Maria Santos", "2015-03-14", "Purok 7, Baguio City"]  # made up
    crops = []
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    for text in lines:
        (width, height), _ = cv2.getTextSize(text, font, 1.5, 2)
        crop = np.full((height + 30, width + 20), 255, np.uint8)
        cv2.putText(crop, text, (10, height + 10), font, 1.5, 20, 2)
        crops.append(crop)
    fp32 = TrOcrRecognizer(fp32_dir).recognize_batch(crops)
    int8 = TrOcrRecognizer(int8_dir).recognize_batch(crops)
    changed = sum(edit_distance(a.text, b.text) for a, b in zip(fp32, int8, strict=True))
    total = max(sum(len(a.text) for a in fp32), 1)
    print(f"INT8 vs FP32 on {len(lines)} generated lines: {changed / total:.1%} characters differ")


def main(argv: list[str] | None = None, download: Downloader = hf_download) -> int:
    parser = argparse.ArgumentParser(description="Download and prepare the OCR models.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--tiers", default="small", help=f"PP-OCRv6 tiers, comma-separated: {','.join(TIERS)}"
    )
    parser.add_argument("--skip-trocr", action="store_true", help="PP-OCRv6 only")
    parser.add_argument("--force", action="store_true", help="Reinstall models already present")
    args = parser.parse_args(argv)

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    if unknown := [t for t in tiers if t not in TIERS]:
        print(f"Unknown tier(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    models_dir: Path = args.models_dir
    cache_dir = models_dir / DOWNLOADS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        models = read_manifest(models_dir)["models"]
    except ModelsMissing:
        models = {}

    try:
        for tier in tiers:
            for kind in ("det", "rec"):
                name = paddle_dir(kind, tier)
                if not args.force and name in models and _installed(models_dir, name):
                    print(f"{name}: already installed")
                    continue
                print(f"{name}: downloading")
                models[name] = install_paddle(kind, tier, models_dir, cache_dir, download)
                write_manifest(models_dir, models)
        if not args.skip_trocr:
            if not args.force and TROCR_DIR in models and _installed(models_dir, TROCR_DIR):
                print(f"{TROCR_DIR}: already installed")
            else:
                print(f"{TROCR_DIR}: downloading about 1.3 GB")
                models[TROCR_DIR] = export_trocr(models_dir, cache_dir, download)
                write_manifest(models_dir, models)
    except ChecksumError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Done. Models are in {models_dir}; {cache_dir} can be deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
