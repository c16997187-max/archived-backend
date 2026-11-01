from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Index
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models._common import created_at, pk


class AuditLog(Base):
    """Append-only change history. Never store passwords, tokens or images here."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_log_entity_entity_id", "entity", "entity_id"),)

    id: Mapped[int] = pk()
    # NULL for system actions and failed logins with an unknown ID number.
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str]
    entity: Mapped[str]
    entity_id: Mapped[int | None]
    before: Mapped[dict[str, Any] | None]
    after: Mapped[dict[str, Any] | None]
    ip_address: Mapped[str | None] = mapped_column(INET)
    created_at: Mapped[datetime] = created_at()
