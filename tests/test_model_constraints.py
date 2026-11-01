"""The database itself must reject bad data, even if a service or the UI has a bug."""

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Document, Enrollment, Learner, OcrResult, SchoolYear, Section, User


def assert_rejected(db: Session, constraint: str, obj: object) -> None:
    """Insert `obj` and assert the named constraint rejects it. Runs in a
    savepoint so rows set up earlier in the test survive."""
    with pytest.raises(IntegrityError) as exc, db.begin_nested():
        db.add(obj)
        db.flush()
    assert exc.value.orig.diag.constraint_name == constraint


def add(db: Session, *objs: object) -> None:
    db.add_all(objs)
    db.flush()


def make_year(label: str = "2026-2027", is_current: bool = False) -> SchoolYear:
    start_year = int(label[:4])
    return SchoolYear(
        label=label,
        start_date=date(start_year, 6, 1),
        end_date=date(start_year + 1, 3, 31),
        is_current=is_current,
    )


def make_learner(lrn: str | None = None, last_name: str = "Testlast") -> Learner:
    return Learner(lrn=lrn, last_name=last_name, first_name="Testfirst")


def make_user(id_number: str = "T-0001", role: str = "registrar") -> User:
    return User(id_number=id_number, full_name="Test User", password_hash="x", role=role)


# --- learners -----------------------------------------------------------------


@pytest.mark.parametrize("lrn", ["123456789012", None])
def test_lrn_accepts_12_digits_or_empty(db, lrn):
    add(db, make_learner(lrn))


@pytest.mark.parametrize(
    "lrn", ["12345678901", "1234567890123", "12345678901A", "", " 23456789012"]
)
def test_lrn_rejects_anything_but_12_digits(db, lrn):
    assert_rejected(db, "ck_learners_lrn_12_digits", make_learner(lrn))


def test_lrn_must_be_unique(db):
    add(db, make_learner("123456789012"))

    assert_rejected(db, "uq_learners_lrn_active", make_learner("123456789012"))


def test_many_learners_may_have_no_lrn(db):
    add(db, make_learner(None), make_learner(None))


def test_lrn_of_soft_deleted_learner_can_be_reused(db):
    add(
        db, Learner(lrn="123456789012", last_name="A", first_name="A", deleted_at=datetime.now(UTC))
    )

    add(db, make_learner("123456789012"))


def test_learner_sex_must_be_m_or_f(db):
    assert_rejected(db, "ck_learners_sex", Learner(last_name="A", first_name="A", sex="X"))


def test_learner_name_is_required(db):
    with pytest.raises(IntegrityError) as exc:
        add(db, Learner(last_name="A", first_name=None))
    assert exc.value.orig.diag.column_name == "first_name"


# --- enrollments --------------------------------------------------------------


def test_learner_cannot_be_enrolled_twice_in_one_year(db):
    year, learner = make_year(), make_learner()
    add(db, Enrollment(learner=learner, school_year=year))

    assert_rejected(
        db,
        "uq_enrollments_learner_id_school_year_id_active",
        Enrollment(learner=learner, school_year=year),
    )


def test_learner_can_enroll_in_different_years(db):
    learner = make_learner()
    add(
        db,
        Enrollment(learner=learner, school_year=make_year("2025-2026")),
        Enrollment(learner=learner, school_year=make_year("2026-2027")),
    )


def test_soft_deleted_enrollment_does_not_block_reenrolling(db):
    year, learner = make_year(), make_learner()
    add(db, Enrollment(learner=learner, school_year=year, deleted_at=datetime.now(UTC)))

    add(db, Enrollment(learner=learner, school_year=year))


def test_enrollment_section_must_be_in_the_same_school_year(db):
    last_year, this_year, learner = make_year("2025-2026"), make_year("2026-2027"), make_learner()
    old_section = Section(school_year=last_year, grade_level=7, name="Narra")
    add(db, old_section, this_year, learner)

    assert_rejected(
        db,
        "fk_enrollments_section_in_school_year",
        Enrollment(learner_id=learner.id, school_year_id=this_year.id, section_id=old_section.id),
    )


def test_enrollment_status_must_be_known(db):
    assert_rejected(
        db,
        "ck_enrollments_status",
        Enrollment(learner=make_learner(), school_year=make_year(), status="enrolled"),
    )


# --- school years and sections ------------------------------------------------


def test_only_one_school_year_can_be_current(db):
    add(db, make_year("2025-2026", is_current=True))

    assert_rejected(db, "uq_school_years_one_current", make_year("2026-2027", is_current=True))


def test_school_year_label_must_be_unique(db):
    add(db, make_year("2026-2027"))

    assert_rejected(db, "uq_school_years_label", make_year("2026-2027"))


def test_school_year_must_end_after_it_starts(db):
    year = SchoolYear(label="2026-2027", start_date=date(2026, 6, 1), end_date=date(2026, 5, 1))

    assert_rejected(db, "ck_school_years_dates_in_order", year)


@pytest.mark.parametrize("grade", [-1, 13])
def test_grade_level_must_be_k_to_12(db, grade):
    section = Section(school_year=make_year(), grade_level=grade, name="Narra")

    assert_rejected(db, "ck_sections_grade_level_k_to_12", section)


def test_section_name_is_unique_per_grade_and_year(db):
    year = make_year()
    add(db, Section(school_year=year, grade_level=7, name="Narra"))

    assert_rejected(
        db,
        "uq_sections_school_year_id_grade_level_name",
        Section(school_year_id=year.id, grade_level=7, name="Narra"),
    )


# --- users, documents, OCR ----------------------------------------------------


def test_user_id_number_must_be_unique(db):
    add(db, make_user("T-0001"))

    assert_rejected(db, "uq_users_id_number", make_user("T-0001"))


def test_user_role_must_be_known(db):
    assert_rejected(db, "ck_users_role", make_user(role="superuser"))


def test_same_image_cannot_be_stored_twice(db):
    user = make_user()
    add(db, user)

    def document() -> Document:
        return Document(
            file_path="x.jpg", sha256="a" * 64, mime_type="image/jpeg", source="phone",
            uploaded_by=user.id,
        )  # fmt: skip

    add(db, document())
    assert_rejected(db, "uq_documents_sha256", document())


def test_ocr_confidence_must_be_between_0_and_1(db):
    user = make_user()
    add(db, user)
    doc = Document(
        file_path="x.jpg", sha256="b" * 64, mime_type="image/jpeg", source="upload",
        uploaded_by=user.id,
    )  # fmt: skip
    add(db, doc)

    result = OcrResult(
        document_id=doc.id, field_name="last_name", confidence=1.5, model_version="t"
    )
    assert_rejected(db, "ck_ocr_results_confidence_0_to_1", result)


# --- fuzzy search -------------------------------------------------------------


def test_trigram_search_finds_misspelled_name(db):
    add(db, make_learner(last_name="Santos"), make_learner(last_name="Bautista"))

    found = db.scalars(select(Learner.last_name).where(Learner.last_name.op("%")("Santoz"))).all()

    assert found == ["Santos"]


def test_name_search_indexes_exist(db):
    indexes = set(db.scalars(text("SELECT indexname FROM pg_indexes WHERE tablename = 'learners'")))

    assert {
        "ix_learners_lrn",
        "ix_learners_last_name_first_name",
        "ix_learners_last_name_trgm",
        "ix_learners_first_name_trgm",
    } <= indexes
