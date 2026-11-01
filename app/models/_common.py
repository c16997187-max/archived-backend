from datetime import datetime

from sqlalchemy import CheckConstraint, Identity, func
from sqlalchemy.orm import Mapped, mapped_column


def pk() -> Mapped[int]:
    """Bigint identity primary key. Never key on names: two learners can share one."""
    return mapped_column(Identity(), primary_key=True)


def created_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


def updated_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now(), onupdate=func.now())


def one_of(column: str, values: tuple[str, ...], name: str) -> CheckConstraint:
    """CHECK that `column` holds one of `values`. Used instead of PG ENUM types,
    which are awkward to change in migrations."""
    allowed = ", ".join(f"'{v}'" for v in values)
    return CheckConstraint(f"{column} IN ({allowed})", name=name)
