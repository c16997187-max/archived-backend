"""Search, view, create and edit learners (Search Records and Record Details)."""

import re
from datetime import date
from typing import Any

from sqlalchemy import ColumnElement, and_, exists, false, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.errors import ApiError
from app.models import Enrollment, Learner, SchoolYear, User
from app.schemas.common import Page, PageRequest
from app.schemas.learners import (
    EnrollmentOut,
    LearnerCreate,
    LearnerDetail,
    LearnerMatch,
    LearnerOut,
    LearnerSummary,
    LearnerUpdate,
)
from app.services import audit
from app.services.db_errors import conflicts_as_api_errors

LRN_UNIQUE_INDEX = "uq_learners_lrn_active"
LRN_TAKEN = {"lrn": "Already used by another learner"}
# Phase 10 tracks where a value came from: ocr, manual or import. These are typed by staff.
SOURCE_MANUAL = "manual"


def _audit_values(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v.isoformat() if isinstance(v, date) else v for k, v in values.items()}


def _enrollments_by_learner(
    session: Session, learner_ids: list[int], *conditions: ColumnElement[bool]
) -> dict[int, list[Enrollment]]:
    """Each learner's enrollments, newest school year first."""
    rows = session.scalars(
        select(Enrollment)
        .join(Enrollment.school_year)
        .options(joinedload(Enrollment.school_year), joinedload(Enrollment.section))
        .where(Enrollment.learner_id.in_(learner_ids), Enrollment.deleted_at.is_(None), *conditions)
        .order_by(SchoolYear.start_date.desc(), Enrollment.id.desc())
    ).all()
    grouped: dict[int, list[Enrollment]] = {}
    for row in rows:
        grouped.setdefault(row.learner_id, []).append(row)
    return grouped


def _newest(enrollments: dict[int, list[Enrollment]], learner_id: int) -> Enrollment | None:
    return next(iter(enrollments.get(learner_id, [])), None)


def _name_token_matches(token: str) -> ColumnElement[bool]:
    # `%` finds misspellings ("Santoz" finds "Santos"); the substring match covers
    # tokens too short to share trigrams. The trigram indexes serve both.
    return or_(
        Learner.last_name.op("%")(token),
        Learner.first_name.op("%")(token),
        Learner.last_name.icontains(token, autoescape=True),
        Learner.first_name.icontains(token, autoescape=True),
    )


def search_learners(
    session: Session,
    paging: PageRequest,
    q: str | None = None,
    grade: int | None = None,
    section_id: int | None = None,
    school_year_id: int | None = None,
) -> Page[LearnerSummary]:
    conditions: list[ColumnElement[bool]] = [Learner.deleted_at.is_(None)]
    order_by: list[Any] = [Learner.last_name, Learner.first_name, Learner.id]

    tokens = [t for t in re.split(r"[\s,]+", q or "") if t]
    if tokens and all(t.isascii() and t.isdigit() for t in tokens):
        conditions.append(Learner.lrn.startswith("".join(tokens), autoescape=True))
    elif tokens:
        conditions.extend(_name_token_matches(t) for t in tokens)
        full_name = Learner.last_name + " " + Learner.first_name
        order_by.insert(0, func.similarity(full_name, " ".join(tokens)).desc())

    # One EXISTS, so every filter must hold for the same enrollment: a learner in
    # Grade 8 last year and Grade 9 now is not found by this year + Grade 8.
    filters: list[ColumnElement[bool]] = []
    if grade is not None:
        filters.append(Enrollment.grade_level == grade)
    if section_id is not None:
        filters.append(Enrollment.section_id == section_id)
    if school_year_id is not None:
        filters.append(Enrollment.school_year_id == school_year_id)
    if filters:
        conditions.append(
            exists().where(
                Enrollment.learner_id == Learner.id, Enrollment.deleted_at.is_(None), *filters
            )
        )

    total = session.scalar(select(func.count()).select_from(Learner).where(*conditions)) or 0
    learners = session.scalars(
        select(Learner)
        .where(*conditions)
        .order_by(*order_by)
        .limit(paging.page_size)
        .offset(paging.offset)
    ).all()
    enrollments = _enrollments_by_learner(session, [row.id for row in learners], *filters)
    items = [LearnerSummary.from_model(row, _newest(enrollments, row.id)) for row in learners]
    return Page(items=items, page=paging.page, page_size=paging.page_size, total=total)


def _get_active(session: Session, learner_id: int, lock: bool = False) -> Learner:
    learner = session.get(Learner, learner_id, with_for_update=lock)
    if learner is None or learner.deleted_at is not None:
        raise ApiError(404, "not_found")
    return learner


def get_learner(session: Session, learner_id: int) -> LearnerDetail:
    learner = _get_active(session, learner_id)
    enrollments = _enrollments_by_learner(session, [learner.id]).get(learner.id, [])
    return LearnerDetail(
        **LearnerOut.model_validate(learner).model_dump(),
        enrollments=[EnrollmentOut.from_model(e) for e in enrollments],
    )


def update_learner(
    session: Session, actor: User, learner_id: int, body: LearnerUpdate, ip: str | None
) -> Learner:
    """Apply the fields sent, with an audit row of only what actually changed."""
    learner = _get_active(session, learner_id, lock=True)
    sent = body.model_dump(exclude_unset=True)
    changes = {k: v for k, v in sent.items() if getattr(learner, k) != v}
    if not changes:
        return learner

    before = {k: getattr(learner, k) for k in changes}
    if changes.get("lrn") is not None:
        taken = session.scalar(
            select(Learner.id).where(
                Learner.lrn == changes["lrn"],
                Learner.deleted_at.is_(None),
                Learner.id != learner.id,
            )
        )
        if taken is not None:
            raise ApiError(409, "duplicate", LRN_TAKEN)

    with conflicts_as_api_errors(
        session, {LRN_UNIQUE_INDEX: lambda: ApiError(409, "duplicate", LRN_TAKEN)}
    ):
        for field, value in changes.items():
            setattr(learner, field, value)
    audit.record(
        session,
        "update",
        "learner",
        learner.id,
        actor.id,
        before=_audit_values(before),
        after={**_audit_values(changes), "source": SOURCE_MANUAL},
        ip_address=ip,
    )
    session.commit()
    return learner


def _normalized_name(value: Any) -> ColumnElement[str]:
    return func.lower(func.regexp_replace(func.trim(value), r"\s+", " ", "g"))


def find_duplicates(session: Session, data: LearnerCreate) -> list[LearnerMatch]:
    """Active learners with the same LRN, or the same name and birth date."""
    same_lrn = Learner.lrn == data.lrn if data.lrn else false()
    same_name = (
        and_(
            _normalized_name(Learner.last_name) == _normalized_name(data.last_name),
            _normalized_name(Learner.first_name) == _normalized_name(data.first_name),
            Learner.birth_date == data.birth_date,
        )
        if data.birth_date
        else false()
    )
    rows = session.execute(
        select(Learner, same_lrn.label("same_lrn"), same_name.label("same_name"))
        .where(Learner.deleted_at.is_(None), or_(same_lrn, same_name))
        .order_by(Learner.last_name, Learner.first_name, Learner.id)
    ).all()
    enrollments = _enrollments_by_learner(session, [row.Learner.id for row in rows])

    matches = []
    for row in rows:
        reasons = ["lrn"] if row.same_lrn else []
        if row.same_name:
            reasons.append("name_birth_date")
        summary = LearnerSummary.from_model(row.Learner, _newest(enrollments, row.Learner.id))
        matches.append(LearnerMatch(**summary.model_dump(), reasons=reasons))
    return matches


def _duplicate_error(matches: list[LearnerMatch]) -> ApiError:
    # An LRN cannot be overridden, so it is the message the UI should show first.
    if not matches or any("lrn" in m.reasons for m in matches):
        fields = LRN_TAKEN
    else:
        fields = {"learner": "A learner with this name and birth date already exists"}
    return ApiError(
        409,
        "possible_duplicate",
        fields,
        extra={"matches": [m.model_dump(mode="json") for m in matches]},
    )


def create_learner(
    session: Session, actor: User, data: LearnerCreate, confirm_new: bool, ip: str | None
) -> Learner:
    """Add a learner and its audit row to the caller's transaction, unless it looks
    like one we already have. `confirm_new` overrides a same-name match, never an LRN."""
    matches = find_duplicates(session, data)
    lrn_taken = any("lrn" in m.reasons for m in matches)
    if lrn_taken or (matches and not confirm_new):
        raise _duplicate_error(matches)

    values = data.model_dump()
    learner = Learner(**values)
    with conflicts_as_api_errors(
        session,
        {LRN_UNIQUE_INDEX: lambda: _duplicate_error(find_duplicates(session, data))},
    ):
        session.add(learner)

    after = _audit_values({k: v for k, v in values.items() if v is not None})
    after["source"] = SOURCE_MANUAL
    if matches:
        after["duplicate_override"] = True
        after["duplicate_override_matches"] = [m.id for m in matches]
    audit.record(session, "create", "learner", learner.id, actor.id, after=after, ip_address=ip)
    return learner
