"""Save every intermediate image of the OCR pipeline, to check them by eye.

    python -m scripts.ocr_debug --image photo.jpg
    python -m scripts.ocr_debug --image photo.jpg --corners corners.json
    python -m scripts.ocr_debug --image photo.jpg --template docs/forms/enrollment_v1.json
    python -m scripts.ocr_debug --document-id 12

--image uses no database, and tries every template JSON in FORM_TEMPLATES_DIR unless told
otherwise. --document-id runs the real service, and does update the extraction.

Run the blank form itself through --image to see your boxes on 05_overlay.png.

Output goes under storage/ocr_debug/, which git ignores, because the crops show minors'
data. Delete it when done, and leave it out of the Phase 11 backups.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal
from app.errors import ApiError
from app.ocr import preprocess
from app.ocr.preprocess import DebugSink, PreprocessFailure, TemplateCandidate
from app.ocr.templates import load_layout_file
from app.services import ocr


def _candidates(paths: list[Path], templates_dir: Path) -> list[TemplateCandidate]:
    candidates = []
    for path in paths or sorted(templates_dir.glob("*.json")):
        layout = load_layout_file(path)
        candidates.append(TemplateCandidate(None, layout, path.parent / layout.image))
    return candidates


def _run_image(args: argparse.Namespace, templates_dir: Path, out: Path) -> int:
    corners = None
    try:
        if args.corners:
            corners = json.loads(args.corners.read_text(encoding="utf-8"))
        candidates = _candidates(args.template, templates_dir)
    except (OSError, ValueError) as exc:  # TemplateError is a ValueError
        print(exc, file=sys.stderr)
        return 1
    try:
        result = preprocess.preprocess(args.image, corners, candidates, DebugSink(out))
    except PreprocessFailure as failure:
        print(f"Failed: {failure.reason}")
        print(f"Debug files: {out}")
        return 1
    print(f"Path: {result.ocr_path}, best score {result.alignment_score:.3f}")
    print(f"Debug files: {out}")
    return 0


def _run_document(document_id: int, out: Path) -> int:
    with SessionLocal() as session:
        try:
            run = ocr.process_document(session, document_id, get_settings(), debug_dir=out)
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
    source.add_argument("--document-id", type=int, help="A stored document; updates its extraction")
    parser.add_argument("--corners", type=Path, help="JSON file of crop corners, as 0..1 fractions")
    parser.add_argument(
        "--template", type=Path, action="append", default=[], help="Template JSON (repeatable)"
    )
    parser.add_argument("--out", type=Path, help="Output folder (default: storage/ocr_debug/<run>)")
    args = parser.parse_args(argv)

    settings = get_settings()
    out = args.out or settings.storage_dir / "ocr_debug" / f"{datetime.now():%Y%m%d-%H%M%S}"
    if args.document_id is not None:
        return _run_document(args.document_id, out)
    return _run_image(args, settings.form_templates_dir, out)


if __name__ == "__main__":
    raise SystemExit(main())
