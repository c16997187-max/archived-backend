from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import ApiError
from app.models import User
from app.models.user import ROLES
from app.services import audit
from app.services.auth import revoke_all_refresh_tokens
from app.services.passwords import check_new_password, hash_password


def create_user(
    session: Session,
    id_number: str,
    full_name: str,
    role: str,
    password: str,
    actor_id: int | None = None,
    ip: str | None = None,
) -> User:
    """Create a staff account and commit it with its audit row.

    `actor_id` is None when the account comes from a command-line script.
    """
    id_number, full_name = id_number.strip(), full_name.strip()
    if not id_number:
        raise ApiError(422, "validation_error", {"id_number": "Required"})
    if not full_name:
        raise ApiError(422, "validation_error", {"full_name": "Required"})
    if role not in ROLES:
        raise ApiError(422, "validation_error", {"role": f"Must be one of {', '.join(ROLES)}"})
    check_new_password(password, "password")
    if session.scalar(select(User.id).where(User.id_number == id_number)) is not None:
        raise ApiError(409, "duplicate", {"id_number": "Already used by another account"})

    user = User(
        id_number=id_number,
        full_name=full_name,
        role=role,
        password_hash=hash_password(password),
    )
    session.add(user)
    session.flush()
    audit.record(
        session,
        "create",
        "user",
        user.id,
        actor_id,
        after={"id_number": id_number, "full_name": full_name, "role": role},
        ip_address=ip,
    )
    session.commit()
    return user


def reset_password(
    session: Session, admin: User, user_id: int, temporary_password: str, ip: str | None
) -> User:
    """Admin "Forgot Password": set a temporary password the user must change at
    next login. Also lifts any lockout and ends the user's sessions."""
    check_new_password(temporary_password, "temporary_password")
    user = session.get(User, user_id, with_for_update=True)
    if user is None:
        raise ApiError(404, "not_found")

    before = {"must_change_password": user.must_change_password}
    user.password_hash = hash_password(temporary_password)
    user.must_change_password = True
    user.failed_login_count = 0
    user.locked_until = None
    user.token_version += 1
    revoked = revoke_all_refresh_tokens(session, user.id, datetime.now(UTC))
    audit.record(
        session,
        "password_reset",
        "user",
        user.id,
        admin.id,
        before=before,
        after={"must_change_password": True, "revoked_sessions": revoked},
        ip_address=ip,
    )
    session.commit()
    return user
