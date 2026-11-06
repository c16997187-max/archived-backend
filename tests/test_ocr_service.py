import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.errors import ApiError
from app.models import AuditLog, Document, Extraction, FormTemplate, OcrResult, User
from app.ocr import labels, preprocess
from app.ocr.text_reader import Reading
from app.services import form_templates, ocr
from scripts.make_placeholder_form import render_form
from tests.ocr_images import (
    FakeDetector,
    FakeRecognizer,
    StubReader,
    box,
    enrollment_layout,
    fake_engine,
    save_png,
    tilted,
    unrelated_layout,
    white_page,
)

# Made up. Asserted absent from the audit log.
SECRET_LRN = "123456789012"
READABLE_PAGE = [
    box("LRN", 100, 300),
    box(SECRET_LRN, 400, 300),
    box("Pangalan", 100, 400),
    box("Madeupname", 400, 400, confidence=0.5),
    box("Tirahan", 100, 500),
]


@pytest.fixture
def ocr_settings(settings: Settings, tmp_path: Path) -> Settings:
    return settings.model_copy(update={"form_templates_dir": tmp_path / "forms"})


@pytest.fixture
def active_template(db: Session, ocr_settings: Settings) -> FormTemplate:
    layout = enrollment_layout()
    save_png(ocr_settings.form_templates_dir / layout.image, render_form(layout))
    row, _ = form_templates.save_template(db, layout, "development")
    return row


@pytest.fixture
def make_document(db: Session, ocr_settings: Settings, make_user: Callable[..., User]):
    def _make(image, corners: dict | None = None) -> Document:
        digest = f"{db.scalar(select(Document.id).order_by(Document.id.desc()).limit(1)) or 0:064x}"
        relative = f"images/{digest}.png"
        save_png(ocr_settings.storage_dir / relative, image)
        document = Document(
            file_path=relative,
            sha256=digest,
            mime_type="image/png",
            source="upload",
            uploaded_by=make_user().id,
            crop_corners=corners,
        )
        db.add(document)
        db.flush()
        return document

    return _make


def _audit_rows(db: Session, extraction: Extraction) -> list[AuditLog]:
    return list(
        db.scalars(
            select(AuditLog)
            .where(AuditLog.entity == "extraction", AuditLog.entity_id == extraction.id)
            .order_by(AuditLog.id)
        )
    )


def _results(db: Session, document: Document) -> dict[str, OcrResult]:
    rows = db.scalars(select(OcrResult).where(OcrResult.document_id == document.id))
    return {row.field_name: row for row in rows}


def test_template_path_without_models_records_the_match_and_waits(
    db, ocr_settings, active_template, make_document
):
    document = make_document(tilted(render_form(enrollment_layout())))

    run = ocr.process_document(db, document.id, ocr_settings)

    extraction = run.extraction
    assert extraction.status == "processing"
    assert extraction.ocr_path == "template"
    assert extraction.template_id == active_template.id
    assert extraction.alignment_score >= preprocess.MIN_ALIGNMENT_SCORE
    assert extraction.started_at is not None
    assert extraction.finished_at is None
    assert run.result.crops
    assert [r.action for r in _audit_rows(db, extraction)] == ["start", "route"]


def test_no_match_without_models_is_generic_and_waits(
    db, ocr_settings, active_template, make_document
):
    document = make_document(render_form(unrelated_layout(), "unrelated memo"))

    run = ocr.process_document(db, document.id, ocr_settings)

    extraction = run.extraction
    assert extraction.status == "processing"
    assert extraction.error is None
    assert extraction.ocr_path == "generic"
    assert extraction.template_id is None
    assert extraction.alignment_score < preprocess.MIN_ALIGNMENT_SCORE
    assert run.result.clean_page is not None
    assert _results(db, document) == {}


def test_no_match_with_a_reader_flags_every_field_and_stores_the_text(
    db, ocr_settings, active_template, make_document
):
    document = make_document(render_form(unrelated_layout(), "unrelated memo"))

    run = ocr.process_document(
        db, document.id, ocr_settings, fake_engine(text_reader=StubReader(READABLE_PAGE))
    )

    extraction = run.extraction
    assert extraction.status == "needs_review"
    assert extraction.finished_at is not None
    assert extraction.ocr_path == "generic"
    assert f"LRN {SECRET_LRN}" in extraction.full_text

    results = _results(db, document)
    assert set(results) == {"lrn", "full_name", "address"}
    assert all(r.needs_check for r in results.values())
    assert all(r.template_id is None for r in results.values())
    assert all(r.normalized_text is None for r in results.values())
    assert all(r.model_version == StubReader.model_version for r in results.values())
    assert results["lrn"].raw_text == SECRET_LRN
    assert results["full_name"].confidence == 0.5
    assert results["full_name"].bbox["x"] == 400
    assert results["address"].raw_text is None  # the label was there; its value was not


def _filled_enrollment() -> np.ndarray:
    """The test form with a made-up surname and first name written in, and Male ticked."""
    layout = enrollment_layout()
    page = render_form(layout)
    boxes = layout.boxes()
    for name in ("last_name", "first_name"):
        b = boxes[name]
        cv2.putText(page, "Madeupname", (b.x + 15, b.y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    male = boxes["sex_male"]
    cv2.line(page, (male.x + 8, male.y + 8), (male.x + 32, male.y + 32), 0, 4)
    cv2.line(page, (male.x + 32, male.y + 8), (male.x + 8, male.y + 32), 0, 4)
    return page


def _reads_ink(confidence: float = 0.95):
    """A recognizer that reads "Madeupname" wherever there is writing, and nothing elsewhere."""

    def read(crop: np.ndarray) -> Reading:
        return Reading("Madeupname", confidence) if (crop < 100).any() else Reading("", None)

    return read


def test_template_path_reads_every_field_and_needs_review(
    db, ocr_settings, active_template, make_document
):
    document = make_document(tilted(_filled_enrollment()))
    reader = StubReader(READABLE_PAGE)
    engine = fake_engine(FakeRecognizer("fake-hand-1", read=_reads_ink()), text_reader=reader)

    run = ocr.process_document(db, document.id, ocr_settings, engine)

    extraction = run.extraction
    assert extraction.status == "needs_review"
    assert extraction.finished_at is not None
    assert "last_name: Madeupname" in extraction.full_text
    assert not reader.recognized  # the full-page reader is for the generic path

    results = _results(db, document)
    layout = enrollment_layout()
    assert set(results) == {*layout.boxes(), "sex"}
    assert all(r.template_id == active_template.id for r in results.values())
    last = results["last_name"]
    assert (last.raw_text, last.normalized_text) == ("Madeupname", "Madeupname")
    assert last.confidence == pytest.approx(0.95)
    assert not last.needs_check
    assert last.model_version == "fake-hand-1"
    box_ = layout.boxes()["last_name"]
    assert last.bbox == {"x": box_.x, "y": box_.y, "w": box_.w, "h": box_.h}
    # Empty and optional: nothing to check.
    assert results["religion"].raw_text is None
    assert not results["religion"].needs_check
    # Empty and required, or no guardian or parent at all: flagged.
    assert results["birth_date"].needs_check
    assert results["guardian_name"].needs_check
    assert results["sex_male"].raw_text == "checked"
    assert results["sex_female"].raw_text == "unchecked"
    assert (results["sex"].normalized_text, results["sex"].needs_check) == ("M", False)


def test_template_path_flags_unsure_fields(db, ocr_settings, active_template, make_document):
    document = make_document(_filled_enrollment())
    engine = fake_engine(FakeRecognizer("fake-hand-1", read=_reads_ink(0.5)))

    ocr.process_document(db, document.id, ocr_settings, engine)

    last = _results(db, document)["last_name"]
    assert last.normalized_text == "Madeupname"
    assert last.needs_check


def test_template_path_calls_each_model_once_per_form(
    db, ocr_settings, active_template, make_document
):
    document = make_document(_filled_enrollment())
    detector = FakeDetector()
    handwriting = FakeRecognizer("fake-hand-1", read=_reads_ink())
    engine = fake_engine(handwriting, detector=detector)

    ocr.process_document(db, document.id, ocr_settings, engine)

    handwritten = [b for b in enrollment_layout().boxes().values() if b.kind == "handwritten"]
    assert detector.batches == [len(handwritten)]
    assert len(handwriting.calls) == 1


def test_template_audit_never_holds_what_was_read(db, ocr_settings, active_template, make_document):
    document = make_document(_filled_enrollment())
    engine = fake_engine(FakeRecognizer("fake-hand-1", read=_reads_ink()))

    run = ocr.process_document(db, document.id, ocr_settings, engine)

    rows = _audit_rows(db, run.extraction)
    assert [r.action for r in rows] == ["start", "route", "finish"]
    assert "Madeupname" not in json.dumps([[r.before, r.after] for r in rows])


def test_template_path_writes_fields_to_the_debug_folder(
    db, ocr_settings, active_template, make_document, tmp_path
):
    document = make_document(_filled_enrollment())
    engine = fake_engine(FakeRecognizer("fake-hand-1", read=_reads_ink()))

    ocr.process_document(db, document.id, ocr_settings, engine, debug_dir=tmp_path / "debug")

    table = (tmp_path / "debug" / "06_fields.txt").read_text(encoding="utf-8")
    assert "last_name\tMadeupname\tMadeupname\t0.950\t\t\tfake-hand-1" in table


def test_page_with_almost_no_text_fails_as_not_a_form(
    db, ocr_settings, active_template, make_document, tmp_path
):
    document = make_document(white_page())
    out = tmp_path / "debug"

    run = ocr.process_document(db, document.id, ocr_settings, fake_engine(), debug_dir=out)

    extraction = run.extraction
    assert extraction.status == "failed"
    assert extraction.error == labels.REASON_NOT_A_FORM
    assert extraction.ocr_path == "generic"
    assert extraction.finished_at is not None
    assert run.result is None
    assert labels.REASON_NOT_A_FORM in (out / "failure.txt").read_text()
    assert [r.action for r in _audit_rows(db, extraction)] == ["start", "route", "fail"]


def test_no_active_templates_fails(db, ocr_settings, make_document):
    document = make_document(white_page())

    run = ocr.process_document(db, document.id, ocr_settings)

    assert run.extraction.status == "failed"
    assert run.extraction.error == preprocess.REASON_NO_TEMPLATES
    assert run.extraction.ocr_path is None


def test_inactive_template_is_not_tried(db, ocr_settings, active_template, make_document):
    active_template.is_active = False
    db.flush()
    document = make_document(render_form(enrollment_layout()))

    assert ocr.process_document(db, document.id, ocr_settings).extraction.status == "failed"


def test_missing_file_fails(db, ocr_settings, active_template, make_document):
    document = make_document(white_page())
    (ocr_settings.storage_dir / document.file_path).unlink()

    run = ocr.process_document(db, document.id, ocr_settings)

    assert run.extraction.status == "failed"
    assert run.extraction.error == preprocess.REASON_MISSING


def test_unknown_document_is_not_found(db, ocr_settings):
    with pytest.raises(ApiError) as error:
        ocr.process_document(db, 999_999, ocr_settings)
    assert error.value.status_code == 404


def test_audit_never_holds_what_was_read(db, ocr_settings, active_template, make_document):
    document = make_document(render_form(unrelated_layout(), "unrelated memo"))

    run = ocr.process_document(
        db, document.id, ocr_settings, fake_engine(text_reader=StubReader(READABLE_PAGE))
    )

    rows = _audit_rows(db, run.extraction)
    assert [r.action for r in rows] == ["start", "route", "finish"]
    assert rows[-1].after["ocr_path"] == "generic"
    assert all(r.user_id is None for r in rows)
    logged = json.dumps([[r.before, r.after] for r in rows])
    assert SECRET_LRN not in logged
    assert "Madeupname" not in logged


def test_rerun_reuses_a_running_extraction(db, ocr_settings, active_template, make_document):
    document = make_document(render_form(enrollment_layout()))

    first = ocr.process_document(db, document.id, ocr_settings).extraction
    second = ocr.process_document(db, document.id, ocr_settings).extraction

    assert second.id == first.id


def test_rerun_uses_a_queued_extraction(db, ocr_settings, active_template, make_document):
    document = make_document(render_form(enrollment_layout()))
    queued = Extraction(document_id=document.id, status="queued")
    db.add(queued)
    db.flush()

    assert ocr.process_document(db, document.id, ocr_settings).extraction.id == queued.id


def test_rerun_after_failure_keeps_the_failed_run(db, ocr_settings, make_document):
    document = make_document(white_page())
    failed = ocr.process_document(db, document.id, ocr_settings).extraction
    failed_id = failed.id

    retry = ocr.process_document(db, document.id, ocr_settings).extraction

    assert retry.id != failed_id
    assert db.get(Extraction, failed_id).status == "failed"


def test_rerun_of_finished_work_changes_nothing(db, ocr_settings, active_template, make_document):
    document = make_document(render_form(unrelated_layout(), "unrelated memo"))
    engine = fake_engine(text_reader=StubReader(READABLE_PAGE))
    first = ocr.process_document(db, document.id, ocr_settings, engine).extraction
    count = len(_results(db, document))

    again = ocr.process_document(db, document.id, ocr_settings, engine)

    assert again.extraction.id == first.id
    assert again.result is None
    assert (
        db.scalars(select(OcrResult).where(OcrResult.document_id == document.id)).all().__len__()
        == count
    )


def test_unexpected_errors_propagate_for_the_worker_to_retry(
    db, ocr_settings, active_template, make_document, monkeypatch
):
    document = make_document(white_page())

    def boom(*_args):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(preprocess, "preprocess", boom)
    with pytest.raises(RuntimeError):
        ocr.process_document(db, document.id, ocr_settings)

    extraction = db.scalar(select(Extraction).where(Extraction.document_id == document.id))
    assert extraction.status == "processing"


def test_ocr_path_column_rejects_unknown_values(db, make_document):
    document = make_document(white_page())
    db.add(Extraction(document_id=document.id, ocr_path="guesswork"))
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()
