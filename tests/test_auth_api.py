from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.models import AuditLog, RefreshToken
from app.services.auth import REUSE_GRACE_SECONDS
from app.services.tokens import create_access_token, hash_token
from tests.conftest import TEST_JWT_SECRET, TEST_PASSWORD


def _login(client, user, password=TEST_PASSWORD, **extra):
    return client.post(
        "/api/auth/login", json={"id_number": user.id_number, "password": password, **extra}
    )


def _audit(db, action):
    return db.scalars(select(AuditLog).where(AuditLog.action == action).order_by(AuditLog.id)).all()


def _refresh_with(client, raw_token):
    """Present a specific refresh cookie, like a second browser holding a copy."""
    return client.post("/api/auth/refresh", headers={"Cookie": f"refresh_token={raw_token}"})


# --- login ---------------------------------------------------------------------


def test_login_returns_token_and_http_only_cookie(client, db, make_user):
    user = make_user("registrar")

    response = _login(client, user)

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 30 * 60
    assert body["must_change_password"] is False
    assert body["user"]["id_number"] == user.id_number
    cookie = response.headers["set-cookie"]
    assert "refresh_token=" in cookie
    assert "HttpOnly" in cookie
    assert "Path=/api/auth" in cookie
    assert "samesite=strict" in cookie.lower()
    # Without "Remember Me" it is a browser-session cookie.
    assert "Max-Age" not in cookie

    me = client.get("/api/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.json()["id"] == user.id

    db.refresh(user)
    assert user.last_login_at is not None
    [row] = _audit(db, "login")
    assert (row.user_id, row.entity, row.entity_id) == (user.id, "user", user.id)


def test_remember_me_sets_long_lived_cookie(client, make_user):
    response = _login(client, make_user(), remember_me=True)

    assert f"Max-Age={30 * 86400}" in response.headers["set-cookie"]


def test_wrong_password_is_rejected_and_logged(client, db, make_user):
    user = make_user()

    response = _login(client, user, password="wrong-password")

    assert response.status_code == 401
    assert response.json() == {"error": "invalid_credentials", "fields": {}}
    [row] = _audit(db, "login_failed")
    assert row.user_id == user.id
    assert row.after == {"reason": "wrong_password", "failed_attempts": 1}


def test_unknown_id_number_looks_like_wrong_password(client, db):
    response = client.post(
        "/api/auth/login", json={"id_number": "NO-SUCH-ID", "password": "whatever"}
    )

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_credentials"
    [row] = _audit(db, "login_failed")
    assert row.user_id is None
    # The attempted ID number is not stored (it might be a mistyped password).
    assert row.after == {"reason": "unknown_id_number"}


def test_inactive_user_cannot_log_in(client, make_user):
    response = _login(client, make_user(is_active=False))

    assert response.status_code == 403
    assert response.json()["error"] == "account_disabled"


def test_missing_fields_use_the_one_error_format(client):
    response = client.post("/api/auth/login", json={"id_number": ""})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert set(body["fields"]) == {"id_number", "password"}


# --- lockout -------------------------------------------------------------------


def test_five_failures_lock_the_account_for_15_minutes(client, db, make_user):
    user = make_user()

    statuses = [_login(client, user, password="wrong").status_code for _ in range(5)]
    assert statuses == [401, 401, 401, 401, 423]

    # Even the right password is refused while locked.
    response = _login(client, user)
    assert response.status_code == 423
    assert response.json()["error"] == "account_locked"
    assert "locked_until" in response.json()["fields"]

    db.refresh(user)
    remaining = user.locked_until - datetime.now(UTC)
    assert timedelta(minutes=14) < remaining <= timedelta(minutes=15)
    reasons = [row.after["reason"] for row in _audit(db, "login_failed")]
    assert reasons == ["wrong_password"] * 5 + ["locked"]


def test_login_works_again_after_lock_expires(client, db, make_user):
    user = make_user(failed_login_count=0, locked_until=datetime.now(UTC) - timedelta(seconds=1))

    assert _login(client, user).status_code == 200

    db.refresh(user)
    assert user.locked_until is None
    assert user.failed_login_count == 0


def test_successful_login_resets_failure_count(client, db, make_user):
    user = make_user()
    for _ in range(4):
        _login(client, user, password="wrong")

    assert _login(client, user).status_code == 200
    db.refresh(user)
    assert user.failed_login_count == 0


# --- refresh -------------------------------------------------------------------


def test_refresh_rotates_token_and_rejects_the_old_one(client, make_user):
    _login(client, make_user())
    old = client.cookies.get("refresh_token")

    response = client.post("/api/auth/refresh")

    assert response.status_code == 200
    assert response.json()["access_token"]
    new = client.cookies.get("refresh_token")
    assert new and new != old
    assert _refresh_with(client, old).status_code == 401


def test_refresh_without_cookie_is_unauthenticated(client):
    response = client.post("/api/auth/refresh")

    assert response.status_code == 401
    assert response.json()["error"] == "not_authenticated"


def test_reuse_within_grace_window_is_rejected_without_revoking(client, db, make_user):
    _login(client, make_user())
    old = client.cookies.get("refresh_token")
    client.post("/api/auth/refresh")
    newest = client.cookies.get("refresh_token")

    # A second tab presents the token that was rotated a moment ago.
    response = _refresh_with(client, old)

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"
    assert _refresh_with(client, newest).status_code == 200
    assert _audit(db, "refresh_token_reuse") == []


def test_reuse_after_grace_window_revokes_every_session(client, db, make_user):
    user = make_user()
    _login(client, user)
    old = client.cookies.get("refresh_token")
    client.post("/api/auth/refresh")
    newest = client.cookies.get("refresh_token")
    rotated = db.scalars(
        select(RefreshToken).where(RefreshToken.token_hash == hash_token(old))
    ).one()
    rotated.revoked_at -= timedelta(seconds=REUSE_GRACE_SECONDS + 1)
    db.flush()

    response = _refresh_with(client, old)

    assert response.status_code == 401
    assert _refresh_with(client, newest).status_code == 401
    [row] = _audit(db, "refresh_token_reuse")
    assert row.user_id == user.id
    assert row.after["revoked_sessions"] == 1


def test_expired_refresh_token_is_rejected(client, db, make_user):
    _login(client, make_user())
    row = db.scalars(select(RefreshToken)).one()
    row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.flush()

    assert client.post("/api/auth/refresh").status_code == 401


# --- logout --------------------------------------------------------------------


def test_logout_revokes_session_clears_cookie_and_is_logged(client, db, make_user):
    user = make_user()
    _login(client, user)
    raw = client.cookies.get("refresh_token")

    response = client.post("/api/auth/logout")

    assert response.status_code == 204
    assert 'refresh_token=""' in response.headers["set-cookie"]
    assert _refresh_with(client, raw).status_code == 401
    [row] = _audit(db, "logout")
    assert row.user_id == user.id
    # A logged-out token presented later is not treated as theft.
    assert _audit(db, "refresh_token_reuse") == []


def test_logout_without_cookie_still_succeeds(client):
    assert client.post("/api/auth/logout").status_code == 204


# --- access tokens -------------------------------------------------------------


def test_expired_access_token_is_rejected(client, make_user):
    user = make_user()
    token = create_access_token(user, TEST_JWT_SECRET, timedelta(seconds=-1))

    response = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"


@pytest.mark.parametrize("secret", [TEST_JWT_SECRET + "-other", "another-secret-" + "y" * 32])
def test_token_signed_with_another_secret_is_rejected(client, make_user, secret):
    token = create_access_token(make_user(), secret, timedelta(minutes=5))

    assert client.get("/api/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_tampered_access_token_is_rejected(client, make_user, login):
    header = login(make_user())
    token = header["Authorization"]
    tampered = token[:-2] + ("AA" if not token.endswith("AA") else "BB")

    assert client.get("/api/me", headers={"Authorization": tampered}).status_code == 401


def test_token_version_bump_invalidates_access_token(client, db, make_user, login):
    user = make_user()
    header = login(user)
    user.token_version += 1
    db.flush()

    assert client.get("/api/me", headers=header).status_code == 401


def test_deactivated_user_loses_access_immediately(client, db, make_user, login):
    user = make_user()
    header = login(user)
    user.is_active = False
    db.flush()

    assert client.get("/api/me", headers=header).status_code == 401
