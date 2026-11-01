from datetime import datetime

from sqlalchemy import CheckConstraint, SmallInteger, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models._common import created_at, one_of, pk, updated_at

ROLES = ("admin", "registrar", "reviewer")


class User(Base):
    """Staff account. Logs in with the school ID number (Figure 1)."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("id_number"),
        one_of("role", ROLES, "role"),
        CheckConstraint("failed_login_count >= 0", name="failed_login_count_not_negative"),
    )

    id: Mapped[int] = pk()
    id_number: Mapped[str]
    full_name: Mapped[str]
    password_hash: Mapped[str]
    role: Mapped[str]
    is_active: Mapped[bool] = mapped_column(server_default=text("true"))
    last_login_at: Mapped[datetime | None]
    # Consecutive failures since the last successful login; 5 locks the account.
    failed_login_count: Mapped[int] = mapped_column(SmallInteger, server_default=text("0"))
    locked_until: Mapped[datetime | None]
    # Set by an admin password reset; the user must pick a new password first.
    must_change_password: Mapped[bool] = mapped_column(server_default=text("false"))
    # Copied into access tokens; bumping it invalidates every token issued before.
    token_version: Mapped[int] = mapped_column(server_default=text("0"))
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()
