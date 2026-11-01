from datetime import datetime

from sqlalchemy import UniqueConstraint, text
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
    )

    id: Mapped[int] = pk()
    id_number: Mapped[str]
    full_name: Mapped[str]
    password_hash: Mapped[str]
    role: Mapped[str]
    is_active: Mapped[bool] = mapped_column(server_default=text("true"))
    last_login_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()
