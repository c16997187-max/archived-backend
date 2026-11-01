from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, Index, SmallInteger, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models._common import created_at, one_of, pk

JOB_STATUSES = ("queued", "running", "done", "failed")


class Job(Base):
    """Background work queue, claimed with SELECT ... FOR UPDATE SKIP LOCKED (Phase 9)."""

    __tablename__ = "jobs"
    __table_args__ = (
        one_of("status", JOB_STATUSES, "status"),
        CheckConstraint("attempts >= 0", name="attempts_not_negative"),
        Index("ix_jobs_status_run_after", "status", "run_after"),
    )

    id: Mapped[int] = pk()
    type: Mapped[str]
    payload: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    status: Mapped[str] = mapped_column(server_default=text("'queued'"))
    attempts: Mapped[int] = mapped_column(SmallInteger, server_default=text("0"))
    run_after: Mapped[datetime] = mapped_column(server_default=func.now())
    locked_by: Mapped[str | None]
    # When the job was claimed; used to requeue jobs stuck after a power cut.
    locked_at: Mapped[datetime | None]
    error: Mapped[str | None]
    created_at: Mapped[datetime] = created_at()
