"""Save every intermediate image of the OCR pipeline, to check them by eye.

    python -m scripts.ocr_debug --image photo.jpg
    python -m scripts.ocr_debug --image photo.jpg --corners corners.json
    python -m scripts.ocr_debug --image photo.jpg --template docs/forms/enrollment_v1.json
    python -m scripts.ocr_debug --document-id 12
    python -m scripts.ocr_debug --image photo.jpg --recognize
    python -m scripts.ocr_debug --images-dir private_data/forms --recognize --handwriting ppocr \\
        --predictions private_data/ppocr.csv

--image and --images-dir use no database, and try every template JSON in FORM_TEMPLATES_DIR
unless told otherwise. --document-id runs the real service, and does update the extraction.

--recognize also runs steps 6 to 9 with the models in MODELS_DIR and writes what was read to
06_fields.txt. --handwriting picks TrOCR or PP-OCRv6 for handwritten fields, overriding
OCR_HANDWRITING_MODEL. --predictions writes one CSV row per photo (the file name without its
extension is the `document`), ready for `python -m scripts.eval_ocr`, so both models can be
scored on the Step 0 test set. Keep that CSV in private_data/: it holds what was read.

Run the blank form itself through --image to see your boxes on 05_overlay.png.

Output goes under storage/ocr_debug/, which git ignores, because the crops show minors'
data. Delete it when done, and leave it out of the Phase 11 backups.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal
from app.errors import ApiError
from app.ocr import preprocess
from app.ocr.model_files import ModelsMissing
from app.ocr.models import OcrEngine, load_engine
from app.ocr.preprocess import DebugSink, PreprocessFailure, TemplateCandidate
from app.ocr.recognize import FieldResult, recognize_crops
from app.ocr.templates import load_layout_file
from app.services import ocr

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def _candidates(paths: list[Path], templates_dir: Path) -> list[TemplateCandidate]:
    candidates = []
    for path in paths or sorted(templates_dir.glob("*.json")):
        layout = load_layout_file(path)
        candidates.append(TemplateCandidate(None, layout, path.parent / layout.image))
    return candidates


def read_image(
    path: Path,
    corners: dict | None,
    candidates: list[TemplateCandidate],
    engine: OcrEngine | None,
    threshold: float,
    debug: DebugSink | None = None,
) -> tuple[preprocess.PreprocessResult, list[FieldResult]]:
    """Steps 1 to 9 for one photo, with no database. No fields without an engine, or on
    the generic path, which needs the service."""
    result = preprocess.preprocess(path, corners, candidates, debug)
    if engine is None or result.ocr_path != preprocess.TEMPLATE_PATH:
        return result, []
    fields = recognize_crops(result.crops, engine, threshold)
    if debug:
        ocr.debug_fields(debug, fields)
    return result, fields


def write_predictions(path: Path, rows: dict[str, dict[str, str]]) -> None:
    columns = sorted({name for row in rows.values() for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, ["document", *columns])
        writer.writeheader()
        for document, row in rows.items():
            writer.writerow({"document": document, **row})


def _run_images(
    args: argparse.Namespace, images: list[Path], engine: OcrEngine | None, out: Path
) -> int:
    settings = get_settings()
    corners = None
    try:
        if args.corners:
            corners = json.loads(args.corners.read_text(encoding="utf-8"))
        candidates = _candidates(args.template, settings.form_templates_dir)
    except (OSError, ValueError) as exc:  # TemplateError is a ValueError
        print(exc, file=sys.stderr)
        return 1
    predictions: dict[str, dict[str, str]] = {}
    failures = 0
    for image in images:
        target = out / image.stem if len(images) > 1 else out
        try:
            result, fields = read_image(
                image,
                corners,
                candidates,
                engine,
                settings.ocr_confidence_threshold,
                DebugSink(target),
            )
        except PreprocessFailure as failure:
            print(f"{image.name}: failed: {failure.reason}")
            failures += 1
            predictions[image.stem] = {}  # scored as all empty
            continue
        summary = f"{image.name}: path {result.ocr_path}, best score {result.alignment_score:.3f}"
        if fields:
            summary += f", {len(fields)} fields, {sum(f.needs_check for f in fields)} flagged"
        print(summary)
        predictions[image.stem] = {
            f.name: (f.normalized if args.value == "normalized" else f.raw) or "" for f in fields
        }
    if args.predictions:
        write_predictions(args.predictions, predictions)
        print(f"Predictions: {args.predictions}")
    print(f"Debug files: {out}")
    return 1 if failures else 0


def _run_document(document_id: int, engine: OcrEngine | None, out: Path) -> int:
    with SessionLocal() as session:
        try:
            run = ocr.process_document(session, document_id, get_settings(), engine, out)
        except ApiError:
            print(f"No document with ID {document_id}.", file=sys.stderr)
            return 1
        extraction = run.extraction
        print(f"Extraction {extraction.id}: {extraction.status}, path {extraction.ocr_path}")
        if extraction.error:
            print(f"Reason: {extraction.error}")
        if run.result is None and extraction.status != "failed":
            print("Already finished, so nothing was run.")
        print(f"Debug files: {out}")
        return 1 if extraction.status == "failed" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save the OCR pipeline's intermediate images.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="A photo file; uses no database")
    source.add_argument("--images-dir", type=Path, help="Every photo in a folder; no database")
    source.add_argument("--document-id", type=int, help="A stored document; updates its extraction")
    parser.add_argument("--corners", type=Path, help="JSON file of crop corners, as 0..1 fractions")
    parser.add_argument(
        "--template", type=Path, action="append", default=[], help="Template JSON (repeatable)"
    )
    parser.add_argument("--out", type=Path, help="Output folder (default: storage/ocr_debug/<run>)")
    parser.add_argument("--recognize", action="store_true", help="Also read the fields")
    parser.add_argument(
        "--handwriting", choices=("trocr", "ppocr"), help="Override OCR_HANDWRITING_MODEL"
    )
    parser.add_argument("--predictions", type=Path, help="Write what was read as an eval CSV")
    parser.add_argument(
        "--value", choices=("normalized", "raw"), default="normalized", help="For --predictions"
    )
    args = parser.parse_args(argv)
    if args.predictions and (not args.recognize or args.document_id is not None):
        parser.error("--predictions needs --recognize with --image or --images-dir")

    settings = get_settings()
    out = args.out or settings.storage_dir / "ocr_debug" / f"{datetime.now():%Y%m%d-%H%M%S}"
    engine = None
    if args.recognize:
        try:
            engine = load_engine(
                settings.models_dir,
                settings.ocr_det_tier,
                args.handwriting or settings.ocr_handwriting_model,
                settings.ocr_threads,
            )
        except ModelsMissing as exc:
            print(exc, file=sys.stderr)
            return 1
    if args.document_id is not None:
        return _run_document(args.document_id, engine, out)
    if args.images_dir is not None:
        images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            print(f"No photos in {args.images_dir}", file=sys.stderr)
            return 1
        return _run_images(args, images, engine, out)
    return _run_images(args, [args.image], engine, out)


if __name__ == "__main__":
    raise SystemExit(main())
