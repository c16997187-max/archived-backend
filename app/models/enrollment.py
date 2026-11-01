from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, ForeignKeyConstraint, Index, SmallInteger, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models._common import created_at, one_of, pk, updated_at
from app.models.learner import Learner
from app.models.school import KINDERGARTEN, MAX_GRADE, SchoolYear, Section

ENROLLMENT_STATUSES = ("draft", "pending_review", "approved", "rejected")


class Enrollment(Base):
    """A learner in a section for one school year."""

    __tablename__ = "enrollments"
    __table_args__ = (
        # One learner cannot be enrolled twice in a year. Soft-deleted rows
        # are ignored so a mistaken enrollment can be deleted and redone.
        Index(
            "uq_enrollments_learner_id_school_year_id_active",
            "learner_id",
            "school_year_id",
            unique=True,
            postgresql_where="deleted_at IS NULL",
        ),
        Index("ix_enrollments_school_year_id_section_id", "school_year_id", "section_id"),
        # The section must belong to the same school year. section_id may be
        # NULL while a draft has no section yet.
        ForeignKeyConstraint(
            ["section_id", "school_year_id"],
            ["sections.id", "sections.school_year_id"],
            name="fk_enrollments_section_in_school_year",
        ),
        one_of("status", ENROLLMENT_STATUSES, "status"),
        CheckConstraint(
            f"grade_level BETWEEN {KINDERGARTEN} AND {MAX_GRADE}", name="grade_level_k_to_12"
        ),
    )

    id: Mapped[int] = pk()
    learner_id: Mapped[int] = mapped_column(ForeignKey("learners.id"))
    school_year_id: Mapped[int] = mapped_column(ForeignKey("school_years.id"))
    section_id: Mapped[int | None]
    # Kept here because a draft may have a grade before it has a section. Matches
    # the section when one is set (enforced in app/services/enrollments.py).
    grade_level: Mapped[int | None] = mapped_column(SmallInteger)
    status: Mapped[str] = mapped_column(server_default=text("'draft'"))
    enrolled_at: Mapped[datetime | None]
    remarks: Mapped[str | None]
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()
    deleted_at: Mapped[datetime | None]

    learner: Mapped[Learner] = relationship()
    school_year: Mapped[SchoolYear] = relationship()
    section: Mapped[Section | None] = relationship(overlaps="school_year")
