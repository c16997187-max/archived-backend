import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.models import User

ALGORITHM = "HS256"


def create_access_token(user: User, secret: str, lifetime: timedelta) -> str:
    """Short-lived JWT. `ver` ties it to the user's token_version, so a password
    reset or change invalidates it before it expires."""
    now = datetime.now(UTC)
    claims = {
        "sub": str(user.id),
        "ver": user.token_version,
        "type": "access",
        "iat": now,
        "exp": now + lifetime,
    }
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    """Return the claims, or raise jwt.InvalidTokenError (expired, tampered, wrong type)."""
    claims = jwt.decode(
        token, secret, algorithms=[ALGORITHM], options={"require": ["sub", "ver", "exp"]}
    )
    if claims.get("type") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return claims


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_refresh_token() -> tuple[str, str]:
    """Return (raw token for the cookie, SHA-256 hex for the database)."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_token(raw)
