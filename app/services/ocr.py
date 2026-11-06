"""Run the OCR pipeline for one stored document and record the outcome (Phase 7).

The worker calls this in Phase 9. Image work happens between commits, so a slow form never
holds a transaction open. Audit rows carry status, path and score only: what was read off
the page is a minor's data and stays in `extractions.full_text` and `ocr_results`.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.errors import ApiError
from app.models import Document, Extraction, OcrResult
from app.ocr import labels, preprocess
from app.ocr.preprocess import DebugSink, PreprocessFailure, PreprocessResult
from app.ocr.text_reader import TextReader
from app.services import audit, form_templates

_FINISHED_STATUSES = ("needs_review", "done")
_RUNNING_STATUSES = ("queued", "processing")


@dataclass(frozen=True)
class OcrRun:
    extraction: Extraction
    # None when the document had already finished, or the run failed.
    result: PreprocessResult | None


def _now() -> datetime:
    return datetime.now(UTC)


def _record_status(session: Session, extraction: Extraction, action: str, before: str) -> None:
    audit.record(
        session,
        action,
        "extraction",
        extraction.id,
        None,
        before={"status": before},
        after={
            "status": extraction.status,
            "document_id": extraction.document_id,
            "ocr_path": extraction.ocr_path,
            "template_id": extraction.template_id,
            "alignment_score": extraction.alignment_score,
        },
    )
    session.commit()


def _begin(session: Session, document_id: int) -> tuple[Extraction, bool]:
    """(extraction to work on, may it run). Finished work is left alone, so a rerun cannot
    duplicate its `ocr_results`. After a failure, a new row keeps the failed run on record."""
    latest = session.scalar(
        select(Extraction)
        .where(Extraction.document_id == document_id)
        .order_by(Extraction.id.desc())
        .limit(1)
        .with_for_update()
    )
    if latest is not None and latest.status in _FINISHED_STATUSES:
        return latest, False
    extraction = latest if latest is not None and latest.status in _RUNNING_STATUSES else None
    before = extraction.status if extraction else "queued"
    if extraction is None:
        extraction = Extraction(document_id=document_id)
        session.add(extraction)
    extraction.status = "processing"
    extraction.started_at = _now()
    session.flush()
    _record_status(session, extraction, "start", before)
    return extraction, True


def _fail(session: Session, extraction: Extraction, reason: str) -> None:
    before = extraction.status
    extraction.status = "failed"
    extraction.finished_at = _now()
    extraction.error = reason
    _record_status(session, extraction, "fail", before)


def _save_generic_page(session: Session, extraction: Extraction, page: labels.GenericPage) -> None:
    """Every field from this path is flagged: nothing here has a template behind it."""
    for field in page.fields:
        session.add(
            OcrResult(
                document_id=extraction.document_id,
                template_id=None,
                field_name=field.name,
                raw_text=field.value or None,
                confidence=field.confidence,
                bbox=field.bbox,
                model_version=page.model_version,
                needs_check=True,
            )
        )
    before = extraction.status
    extraction.full_text = page.full_text
    extraction.status = "needs_review"
    extraction.finished_at = _now()
    _record_status(session, extraction, "finish", before)


def process_document(
    session: Session,
    document_id: int,
    settings: Settings,
    reader: TextReader | None = None,
    debug_dir: Path | None = None,
) -> OcrRun:
    """Steps 1 to 5, then the generic path when no template matched.

    With no `reader` a generic page waits in `processing`, like a template match does until
    recognition exists. Only expected failures mark the extraction `failed`; anything else
    propagates for the worker's retries.
    """
    document = session.get(Document, document_id)
    if document is None:
        raise ApiError(404, "not_found")
    extraction, may_run = _begin(session, document_id)
    if not may_run:
        return OcrRun(extraction, None)

    debug = DebugSink(debug_dir) if debug_dir else None
    candidates = form_templates.active_candidates(session, settings.form_templates_dir)
    photo_path = settings.storage_dir / document.file_path
    corners = document.crop_corners
    # The image work below can take seconds; do not sit in the read transaction above.
    session.commit()
    try:
        result = preprocess.preprocess(photo_path, corners, candidates, debug)
    except PreprocessFailure as failure:
        _fail(session, extraction, failure.reason)
        return OcrRun(extraction, None)

    before = extraction.status
    extraction.ocr_path = result.ocr_path
    extraction.template_id = result.template.id if result.template else None
    extraction.alignment_score = result.alignment_score
    _record_status(session, extraction, "route", before)

    if result.ocr_path == preprocess.GENERIC_PATH and reader is not None:
        try:
            page = labels.read_generic_page(result.clean_page, reader)
        except PreprocessFailure as failure:
            if debug:
                debug.text("failure", failure.reason)
            _fail(session, extraction, failure.reason)
            return OcrRun(extraction, None)
        _save_generic_page(session, extraction, page)
    return OcrRun(extraction, result)
