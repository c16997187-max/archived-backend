"""Login, refresh, logout and password change (Phase 4).

Every function writes its audit row in the same transaction as the change and
commits before returning or raising, so failed logins are recorded too.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import Settings
from app.errors import ApiError
from app.models import RefreshToken, User
from app.services import audit
from app.services.passwords import check_new_password, hash_password, verify_password
from app.services.tokens import create_access_token, hash_token, new_refresh_token

MAX_FAILED_LOGINS = 5
LOCKOUT = timedelta(minutes=15)
# Two browser tabs may refresh with the same cookie at once; the loser presents a
# token rotated moments ago. Within this window that is a race, not theft.
REUSE_GRACE_SECONDS = 30

# Verified against when the ID number is unknown, so the response takes as long
# as a wrong password and does not reveal which ID numbers exist.
_DUMMY_HASH = hash_password("not-a-real-password")


@dataclass
class IssuedSession:
    user: User
    access_token: str
    refresh_token: str  # raw value, for the cookie only
    persistent: bool


def _issue_session(
    session: Session, settings: Settings, user: User, persistent: bool, now: datetime
) -> tuple[IssuedSession, RefreshToken]:
    raw, token_hash = new_refresh_token()
    lifetime = (
        timedelta(days=settings.remember_me_days)
        if persistent
        else timedelta(hours=settings.refresh_token_hours)
    )
    row = RefreshToken(
        user_id=user.id, token_hash=token_hash, persistent=persistent, expires_at=now + lifetime
    )
    session.add(row)
    session.flush()
    access = create_access_token(
        user, settings.jwt_secret, timedelta(minutes=settings.access_token_minutes)
    )
    return IssuedSession(user, access, raw, persistent), row


def revoke_all_refresh_tokens(session: Session, user_id: int, now: datetime) -> int:
    """Log the user out everywhere. Returns how many sessions were still active."""
    result = session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now)
    )
    return result.rowcount


def _fail(
    session: Session,
    user: User | None,
    reason: str,
    ip: str | None,
    status_code: int,
    error: str,
    fields: dict[str, str] | None = None,
) -> ApiError:
    after: dict = {"reason": reason}
    if user is not None:
        after["failed_attempts"] = user.failed_login_count
        if user.locked_until is not None:
            after["locked_until"] = user.locked_until.isoformat()
    # Never store the attempted ID number: people sometimes type a password into it.
    audit.record(
        session,
        "login_failed",
        "user",
        user.id if user else None,
        user.id if user else None,
        after=after,
        ip_address=ip,
    )
    session.commit()
    return ApiError(status_code, error, fields)


def _locked_error(user: User) -> dict[str, str]:
    return {"locked_until": user.locked_until.isoformat()}


def login(
    session: Session,
    settings: Settings,
    id_number: str,
    password: str,
    remember_me: bool,
    ip: str | None,
) -> IssuedSession:
    now = datetime.now(UTC)
    user = session.scalar(select(User).where(User.id_number == id_number).with_for_update())

    if user is None:
        verify_password(password, _DUMMY_HASH)
        raise _fail(session, None, "unknown_id_number", ip, 401, "invalid_credentials")

    if user.locked_until is not None and user.locked_until > now:
        raise _fail(session, user, "locked", ip, 423, "account_locked", _locked_error(user))

    if not verify_password(password, user.password_hash):
        user.failed_login_count += 1
        if user.failed_login_count >= MAX_FAILED_LOGINS:
            user.failed_login_count = 0
            user.locked_until = now + LOCKOUT
            raise _fail(
                session, user, "wrong_password", ip, 423, "account_locked", _locked_error(user)
            )
        raise _fail(session, user, "wrong_password", ip, 401, "invalid_credentials")

    if not user.is_active:
        raise _fail(session, user, "inactive", ip, 403, "account_disabled")

    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = now
    issued, row = _issue_session(session, settings, user, remember_me, now)
    audit.record(
        session,
        "login",
        "user",
        user.id,
        user.id,
        after={"remember_me": remember_me, "session_id": row.id},
        ip_address=ip,
    )
    session.commit()
    return issued


def refresh(
    session: Session, settings: Settings, raw_token: str | None, ip: str | None
) -> IssuedSession:
    """Swap a refresh token for a new one and a fresh access token."""
    if not raw_token:
        raise ApiError(401, "not_authenticated")
    now = datetime.now(UTC)
    row = session.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_token(raw_token))
        .with_for_update()
    )
    if row is None:
        raise ApiError(401, "invalid_token")

    if row.revoked_at is not None:
        rotated = row.replaced_by_id is not None
        if rotated and now - row.revoked_at > timedelta(seconds=REUSE_GRACE_SECONDS):
            # An old token came back long after rotation: someone else may hold a copy.
            revoked = revoke_all_refresh_tokens(session, row.user_id, now)
            audit.record(
                session,
                "refresh_token_reuse",
                "user",
                row.user_id,
                row.user_id,
                after={"session_id": row.id, "revoked_sessions": revoked},
                ip_address=ip,
            )
            session.commit()
        raise ApiError(401, "invalid_token")

    user = session.get(User, row.user_id)
    if row.expires_at <= now or user is None or not user.is_active:
        raise ApiError(401, "invalid_token")

    issued, new_row = _issue_session(session, settings, user, row.persistent, now)
    # Keep the original expiry: refreshing does not stretch a session forever.
    new_row.expires_at = row.expires_at
    row.revoked_at = now
    row.replaced_by_id = new_row.id
    session.commit()
    return issued


def logout(session: Session, raw_token: str | None, ip: str | None) -> None:
    """Revoke the cookie's session. Unknown or already revoked tokens are ignored."""
    if not raw_token:
        return
    row = session.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_token(raw_token))
        .with_for_update()
    )
    if row is None or row.revoked_at is not None:
        return
    row.revoked_at = datetime.now(UTC)
    audit.record(
        session,
        "logout",
        "user",
        row.user_id,
        row.user_id,
        after={"session_id": row.id},
        ip_address=ip,
    )
    session.commit()


def change_password(
    session: Session,
    settings: Settings,
    user: User,
    current_password: str,
    new_password: str,
    raw_token: str | None,
    ip: str | None,
) -> IssuedSession:
    """Set a new password, end every other session and start a fresh one here."""
    if not verify_password(current_password, user.password_hash):
        raise ApiError(422, "validation_error", {"current_password": "Incorrect password"})
    check_new_password(new_password, "new_password")
    if new_password == current_password:
        raise ApiError(
            422, "validation_error", {"new_password": "Must differ from the current password"}
        )

    now = datetime.now(UTC)
    # Keep "Remember Me" if this browser's current session had it.
    current = (
        session.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(raw_token)))
        if raw_token
        else None
    )
    persistent = current is not None and current.user_id == user.id and current.persistent
    before = {"must_change_password": user.must_change_password}
    user.password_hash = hash_password(new_password)
    user.must_change_password = False
    user.token_version += 1
    revoked = revoke_all_refresh_tokens(session, user.id, now)
    issued, _ = _issue_session(session, settings, user, persistent, now)
    audit.record(
        session,
        "password_change",
        "user",
        user.id,
        user.id,
        before=before,
        after={"must_change_password": False, "revoked_sessions": revoked},
        ip_address=ip,
    )
    session.commit()
    return issued
