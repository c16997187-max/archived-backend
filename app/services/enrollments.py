"""New Enrollment: a draft enrollment for a new or existing learner."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import ApiError
from app.models import Enrollment, Learner, SchoolYear, Section, User
from app.schemas.enrollments import EnrollmentCreate, EnrollmentCreated
from app.schemas.learners import EnrollmentOut, LearnerOut
from app.services import audit
from app.services import learners as learners_service
from app.services.db_errors import conflicts_as_api_errors

ONE_PER_YEAR_INDEX = "uq_enrollments_learner_id_school_year_id_active"


def _invalid(field: str, message: str) -> ApiError:
    return ApiError(422, "validation_error", {field: message})


def _already_enrolled() -> ApiError:
    return ApiError(
        409, "already_enrolled", {"school_year_id": "Already enrolled in this school year"}
    )


def _placement(
    session: Session, body: EnrollmentCreate
) -> tuple[SchoolYear, Section | None, int | None]:
    year = session.get(SchoolYear, body.school_year_id)
    if year is None:
        raise _invalid("school_year_id", "Unknown school year")
    if body.section_id is None:
        return year, None, body.grade_level

    section = session.get(Section, body.section_id)
    if section is None or section.school_year_id != year.id:
        raise _invalid("section_id", "Not a section of this school year")
    if body.grade_level is not None and body.grade_level != section.grade_level:
        raise _invalid("grade_level", "Does not match the section's grade")
    return year, section, section.grade_level


def _linked_learner(session: Session, learner_id: int, year: SchoolYear) -> Learner:
    learner = session.get(Learner, learner_id)
    if learner is None or learner.deleted_at is not None:
        raise _invalid("learner_id", "Unknown learner")
    enrolled = session.scalar(
        select(Enrollment.id).where(
            Enrollment.learner_id == learner.id,
            Enrollment.school_year_id == year.id,
            Enrollment.deleted_at.is_(None),
        )
    )
    if enrolled is not None:
        raise _already_enrolled()
    return learner


def create_enrollment(
    session: Session, actor: User, body: EnrollmentCreate, ip: str | None
) -> EnrollmentCreated:
    year, section, grade_level = _placement(session, body)
    if body.learner is not None:
        learner = learners_service.create_learner(
            session, actor, body.learner, body.confirm_new_learner, ip
        )
    else:
        learner = _linked_learner(session, body.learner_id, year)

    # IDs, not relationships: section=None would also clear the school_year_id it
    # shares with the section foreign key.
    enrollment = Enrollment(
        learner_id=learner.id,
        school_year_id=year.id,
        section_id=section.id if section else None,
        grade_level=grade_level,
        status="draft",
        remarks=body.remarks,
    )
    with conflicts_as_api_errors(session, {ONE_PER_YEAR_INDEX: _already_enrolled}):
        session.add(enrollment)
    audit.record(
        session,
        "create",
        "enrollment",
        enrollment.id,
        actor.id,
        after={
            "learner_id": learner.id,
            "school_year_id": year.id,
            "section_id": section.id if section else None,
            "grade_level": grade_level,
            "status": "draft",
            "remarks": body.remarks,
            "source": learners_service.SOURCE_MANUAL,
        },
        ip_address=ip,
    )
    session.commit()
    return EnrollmentCreated(
        learner=LearnerOut.model_validate(learner),
        enrollment=EnrollmentOut.from_model(enrollment),
    )
