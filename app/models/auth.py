from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models._common import created_at, pk


class RefreshToken(Base):
    """One login session. The raw token lives only in the browser's HTTP-only cookie.

    Each /api/auth/refresh revokes this row and points `replaced_by_id` at the new one,
    so a rotated token presented again can be told apart from a logged-out one.
    """

    __tablename__ = "refresh_tokens"
    __table_args__ = (
        UniqueConstraint("token_hash"),
        CheckConstraint("token_hash ~ '^[0-9a-f]{64}$'", name="token_hash_hex"),
    )

    id: Mapped[int] = pk()
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str]
    # "Remember Me": a long-lived cookie instead of a browser-session one.
    persistent: Mapped[bool] = mapped_column(server_default=text("false"))
    expires_at: Mapped[datetime]
    revoked_at: Mapped[datetime | None]
    replaced_by_id: Mapped[int | None] = mapped_column(ForeignKey("refresh_tokens.id"))
    created_at: Mapped[datetime] = created_at()
