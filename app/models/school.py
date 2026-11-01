from datetime import date

from sqlalchemy import CheckConstraint, ForeignKey, Index, SmallInteger, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models._common import pk

# Kindergarten is stored as 0, so grade_level runs 0 (K) to 12.
KINDERGARTEN = 0
MAX_GRADE = 12


class SchoolYear(Base):
    __tablename__ = "school_years"
    __table_args__ = (
        UniqueConstraint("label"),
        CheckConstraint("end_date > start_date", name="dates_in_order"),
        # At most one school year can be current.
        Index(
            "uq_school_years_one_current", "is_current", unique=True, postgresql_where="is_current"
        ),
    )

    id: Mapped[int] = pk()
    label: Mapped[str]  # e.g. "2026-2027"
    start_date: Mapped[date]
    end_date: Mapped[date]
    is_current: Mapped[bool] = mapped_column(server_default=text("false"))

    sections: Mapped[list["Section"]] = relationship(back_populates="school_year")


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (
        UniqueConstraint("school_year_id", "grade_level", "name"),
        # Target of the enrollments composite FK, which keeps an enrollment's
        # section in the same school year as the enrollment.
        UniqueConstraint("id", "school_year_id"),
        CheckConstraint(
            f"grade_level BETWEEN {KINDERGARTEN} AND {MAX_GRADE}", name="grade_level_k_to_12"
        ),
    )

    id: Mapped[int] = pk()
    school_year_id: Mapped[int] = mapped_column(ForeignKey("school_years.id"))
    grade_level: Mapped[int] = mapped_column(SmallInteger)
    name: Mapped[str]
    adviser: Mapped[str | None]

    school_year: Mapped[SchoolYear] = relationship(back_populates="sections")
