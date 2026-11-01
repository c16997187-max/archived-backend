import ipaddress
from collections.abc import Callable
from typing import Annotated

import jwt
from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.session import get_db
from app.errors import ApiError
from app.models import User
from app.models.user import ROLES
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE, MAX_PAGE_SIZE, PageRequest
from app.services.tokens import decode_access_token

_bearer = HTTPBearer(auto_error=False)
_CHALLENGE = {"WWW-Authenticate": "Bearer"}


def get_app_settings(request: Request) -> Settings:
    """The settings the app was created with (tests create apps with their own)."""
    return request.app.state.settings


def client_ip(request: Request) -> str | None:
    """The caller's IP for audit_log, or None if it is not an IP (e.g. TestClient)."""
    host = request.client.host if request.client else None
    try:
        return str(ipaddress.ip_address(host)) if host else None
    except ValueError:
        return None


DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]


def get_paging(
    page: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 1,
    page_size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
) -> PageRequest:
    return PageRequest(page=page, page_size=page_size)


Paging = Annotated[PageRequest, Depends(get_paging)]


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: DbSession,
    settings: AppSettings,
) -> User:
    if credentials is None:
        raise ApiError(401, "not_authenticated", headers=_CHALLENGE)
    try:
        claims = decode_access_token(credentials.credentials, settings.jwt_secret)
        user_id = int(claims["sub"])
    except (jwt.InvalidTokenError, ValueError):
        raise ApiError(401, "invalid_token", headers=_CHALLENGE) from None

    # Loaded fresh on every request, so deactivation, role changes and password
    # resets (token_version) take effect before the token expires.
    user = db.get(User, user_id)
    if user is None or not user.is_active or claims["ver"] != user.token_version:
        raise ApiError(401, "invalid_token", headers=_CHALLENGE)
    return user


def require_role(*roles: str, allow_password_change: bool = False) -> Callable[..., User]:
    """Dependency that lets in only logged-in users with one of `roles`.

    Attach to every endpoint, e.g. `Depends(require_role("registrar", "admin"))`.
    Users with a pending forced password change are refused everywhere except
    routes that pass `allow_password_change=True`.
    """
    unknown = set(roles) - set(ROLES)
    if not roles or unknown:
        raise ValueError(f"require_role needs roles from {ROLES}, got {roles}")

    def dependency(user: Annotated[User, Depends(get_current_user)]) -> User:
        if user.must_change_password and not allow_password_change:
            raise ApiError(403, "password_change_required")
        if user.role not in roles:
            raise ApiError(403, "forbidden")
        return user

    # Read by tests/test_role_matrix.py to prove every route has a role check.
    dependency.allowed_roles = roles  # type: ignore[attr-defined]
    return dependency
