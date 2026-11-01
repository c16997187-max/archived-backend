from typing import Any

from sqlalchemy.orm import Session

from app.models import AuditLog


def record(
    session: Session,
    action: str,
    entity: str,
    entity_id: int | None,
    user_id: int | None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> AuditLog:
    """Add an audit row to the caller's transaction, so it commits with the change.

    Never pass passwords, tokens or images in `before`/`after`.
    """
    row = AuditLog(
        user_id=user_id,
        action=action,
        entity=entity,
        entity_id=entity_id,
        before=before,
        after=after,
        ip_address=ip_address,
    )
    session.add(row)
    return row
