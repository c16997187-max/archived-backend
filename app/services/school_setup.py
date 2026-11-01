"""Admin setup of school years and sections. Every role may read them, because the
New Enrollment and Search Records dropdowns need the lists."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.errors import ApiError
from app.models import SchoolYear, Section, User
from app.schemas.common import Page, PageRequest
from app.schemas.school_setup import SchoolYearCreate, SchoolYearOut, SectionCreate, SectionOut
from app.services import audit
from app.services.db_errors import conflicts_as_api_errors


def _label_taken() -> ApiError:
    return ApiError(409, "duplicate", {"label": "This school year already exists"})


def list_school_years(session: Session, paging: PageRequest) -> Page[SchoolYearOut]:
    total = session.scalar(select(func.count()).select_from(SchoolYear)) or 0
    rows = session.scalars(
        select(SchoolYear)
        .order_by(SchoolYear.start_date.desc(), SchoolYear.id.desc())
        .limit(paging.page_size)
        .offset(paging.offset)
    ).all()
    return Page(
        items=[SchoolYearOut.model_validate(row) for row in rows],
        page=paging.page,
        page_size=paging.page_size,
        total=total,
    )


def create_school_year(
    session: Session, actor: User, body: SchoolYearCreate, ip: str | None
) -> SchoolYear:
    # Checked first so a refused request has changed nothing, not even the current year.
    if session.scalar(select(SchoolYear.id).where(SchoolYear.label == body.label)) is not None:
        raise _label_taken()

    if body.is_current:
        previous = session.scalar(select(SchoolYear).where(SchoolYear.is_current).with_for_update())
        if previous is not None:
            previous.is_current = False
            # Flushed now: the database allows only one current year at any moment.
            session.flush()
            audit.record(
                session,
                "update",
                "school_year",
                previous.id,
                actor.id,
                before={"is_current": True},
                after={"is_current": False},
                ip_address=ip,
            )

    year = SchoolYear(**body.model_dump())
    with conflicts_as_api_errors(
        session,
        {
            "uq_school_years_label": _label_taken,
            "uq_school_years_one_current": lambda: ApiError(
                409,
                "duplicate",
                {"is_current": "Another school year was just made current. Try again."},
            ),
        },
    ):
        session.add(year)
    audit.record(
        session,
        "create",
        "school_year",
        year.id,
        actor.id,
        after=body.model_dump(mode="json"),
        ip_address=ip,
    )
    session.commit()
    return year


def list_sections(
    session: Session,
    paging: PageRequest,
    school_year_id: int | None = None,
    grade: int | None = None,
) -> Page[SectionOut]:
    conditions = []
    if school_year_id is not None:
        conditions.append(Section.school_year_id == school_year_id)
    if grade is not None:
        conditions.append(Section.grade_level == grade)

    total = session.scalar(select(func.count()).select_from(Section).where(*conditions)) or 0
    rows = session.scalars(
        select(Section)
        .where(*conditions)
        .order_by(Section.school_year_id.desc(), Section.grade_level, Section.name, Section.id)
        .limit(paging.page_size)
        .offset(paging.offset)
    ).all()
    return Page(
        items=[SectionOut.model_validate(row) for row in rows],
        page=paging.page,
        page_size=paging.page_size,
        total=total,
    )


def create_section(session: Session, actor: User, body: SectionCreate, ip: str | None) -> Section:
    if session.get(SchoolYear, body.school_year_id) is None:
        raise ApiError(422, "validation_error", {"school_year_id": "Unknown school year"})

    section = Section(**body.model_dump())
    with conflicts_as_api_errors(
        session,
        {
            "uq_sections_school_year_id_grade_level_name": lambda: ApiError(
                409, "duplicate", {"name": "This grade already has a section with this name"}
            )
        },
    ):
        session.add(section)
    audit.record(
        session,
        "create",
        "section",
        section.id,
        actor.id,
        after=body.model_dump(),
        ip_address=ip,
    )
    session.commit()
    return section
