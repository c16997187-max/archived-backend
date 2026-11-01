from datetime import date, datetime

from sqlalchemy import CheckConstraint, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models._common import created_at, one_of, pk, updated_at

SEXES = ("M", "F")


class Learner(Base):
    """One row per student, kept across school years. Fields follow DepEd SF1.

    Only last and first name are required here: drafts may be saved incomplete,
    and the service layer enforces the rest before a record can be approved.
    """

    __tablename__ = "learners"
    __table_args__ = (
        # Old archived records may have no LRN, so NULL is allowed.
        CheckConstraint("lrn ~ '^[0-9]{12}$'", name="lrn_12_digits"),
        one_of("sex", SEXES, "sex"),
        # Unique among learners that are not soft-deleted, so a mistaken
        # duplicate can be deleted and its LRN given to the right record.
        Index("uq_learners_lrn_active", "lrn", unique=True, postgresql_where="deleted_at IS NULL"),
        Index("ix_learners_lrn", "lrn"),
        Index("ix_learners_last_name_first_name", "last_name", "first_name"),
        # Trigram indexes for fuzzy name search ("Santoz" finds "Santos").
        Index(
            "ix_learners_last_name_trgm",
            "last_name",
            postgresql_using="gin",
            postgresql_ops={"last_name": "gin_trgm_ops"},
        ),
        Index(
            "ix_learners_first_name_trgm",
            "first_name",
            postgresql_using="gin",
            postgresql_ops={"first_name": "gin_trgm_ops"},
        ),
    )

    id: Mapped[int] = pk()
    # No length limit: the lrn_12_digits check is the one place that rejects bad LRNs.
    lrn: Mapped[str | None]
    last_name: Mapped[str]
    first_name: Mapped[str]
    middle_name: Mapped[str | None]
    extension: Mapped[str | None]  # Jr., III
    sex: Mapped[str | None] = mapped_column(String(1))
    birth_date: Mapped[date | None]
    place_of_birth: Mapped[str | None]
    mother_tongue: Mapped[str | None]
    religion: Mapped[str | None]
    ip_group: Mapped[str | None]
    house_street: Mapped[str | None]
    barangay: Mapped[str | None]
    municipality: Mapped[str | None]
    province: Mapped[str | None]
    father_name: Mapped[str | None]
    mother_name: Mapped[str | None]
    guardian_name: Mapped[str | None]
    guardian_relationship: Mapped[str | None]
    contact_number: Mapped[str | None]
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()
    deleted_at: Mapped[datetime | None]
