import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.models import AuditLog
from app.services.passwords import verify_password
from tests.conftest import TEST_PASSWORD

TEMP_PASSWORD = "temporary-pass-1"
NEW_PASSWORD = "brand-new-pass-1"


def _reset(client, admin_header, user_id, password=TEMP_PASSWORD):
    return client.post(
        f"/api/users/{user_id}/reset-password",
        json={"temporary_password": password},
        headers=admin_header,
    )


def test_admin_reset_forces_change_and_ends_sessions(client, db, make_user, login):
    admin_header = login(make_user("admin"))
    user = make_user("registrar")
    user_header = login(user)
    user.failed_login_count = 3
    db.flush()
    old_cookie = client.cookies.get("refresh_token")

    response = _reset(client, admin_header, user.id)

    assert response.status_code == 204
    db.refresh(user)
    assert user.must_change_password is True
    assert user.failed_login_count == 0
    assert verify_password(TEMP_PASSWORD, user.password_hash)
    # The user's old access token and refresh cookie no longer work.
    assert client.get("/api/me", headers=user_header).status_code == 401
    refresh = client.post("/api/auth/refresh", headers={"Cookie": f"refresh_token={old_cookie}"})
    assert refresh.status_code == 401


def test_reset_lifts_an_active_lockout(client, db, make_user, login):
    admin_header = login(make_user("admin"))
    user = make_user(locked_until=datetime.now(UTC) + timedelta(minutes=10))

    _reset(client, admin_header, user.id)

    db.refresh(user)
    assert user.locked_until is None


def test_temporary_password_only_allows_me_and_change_password(client, make_user, login):
    admin_header = login(make_user("admin"))
    user = make_user("registrar")
    _reset(client, admin_header, user.id)

    response = client.post(
        "/api/auth/login", json={"id_number": user.id_number, "password": TEMP_PASSWORD}
    )
    assert response.status_code == 200
    assert response.json()["must_change_password"] is True
    header = {"Authorization": f"Bearer {response.json()['access_token']}"}

    assert client.get("/api/me", headers=header).status_code == 200
    blocked = _reset(client, header, user.id)  # any other protected route
    assert blocked.status_code == 403
    assert blocked.json()["error"] == "password_change_required"


def test_change_password_clears_the_flag_and_issues_new_tokens(client, db, make_user, login):
    admin_header = login(make_user("admin"))
    user = make_user("admin")
    _reset(client, admin_header, user.id)
    header = login(user, password=TEMP_PASSWORD)

    response = client.post(
        "/api/auth/change-password",
        json={"current_password": TEMP_PASSWORD, "new_password": NEW_PASSWORD},
        headers=header,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["must_change_password"] is False
    new_header = {"Authorization": f"Bearer {body['access_token']}"}
    # The new token reaches routes that were blocked; the old one is dead.
    other = make_user()
    assert _reset(client, new_header, other.id).status_code == 204
    assert client.get("/api/me", headers=header).status_code == 401
    assert client.post("/api/auth/refresh").status_code == 200
    db.refresh(user)
    assert verify_password(NEW_PASSWORD, user.password_hash)
    [row] = db.scalars(select(AuditLog).where(AuditLog.action == "password_change")).all()
    assert row.before == {"must_change_password": True}


@pytest.mark.parametrize(
    ("current", "new", "field"),
    [
        ("wrong-current", NEW_PASSWORD, "current_password"),
        (TEST_PASSWORD, TEST_PASSWORD, "new_password"),
        (TEST_PASSWORD, "short", "new_password"),
    ],
)
def test_change_password_validation(client, make_user, login, current, new, field):
    header = login(make_user())

    response = client.post(
        "/api/auth/change-password",
        json={"current_password": current, "new_password": new},
        headers=header,
    )

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert field in response.json()["fields"]


@pytest.mark.parametrize("role", ["registrar", "reviewer"])
def test_non_admins_cannot_reset_passwords(client, make_user, login, role):
    header = login(make_user(role))
    target = make_user()

    response = _reset(client, header, target.id)

    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


def test_reset_unknown_user_is_404(client, make_user, login):
    response = _reset(client, login(make_user("admin")), 999_999_999)

    assert response.status_code == 404
    assert response.json() == {"error": "not_found", "fields": {}}


def test_reset_rejects_short_temporary_password(client, make_user, login):
    response = _reset(client, login(make_user("admin")), make_user().id, password="short")

    assert response.status_code == 422
    assert "temporary_password" in response.json()["fields"]


def test_audit_log_never_contains_passwords_or_tokens(client, db, make_user, login):
    admin_header = login(make_user("admin"))
    user = make_user()
    client.post("/api/auth/login", json={"id_number": user.id_number, "password": "wrong-pw-1"})
    _reset(client, admin_header, user.id)
    header = login(user, password=TEMP_PASSWORD)
    client.post(
        "/api/auth/change-password",
        json={"current_password": TEMP_PASSWORD, "new_password": NEW_PASSWORD},
        headers=header,
    )
    refresh_cookie = client.cookies.get("refresh_token")
    client.post("/api/auth/logout")

    rows = db.scalars(select(AuditLog)).all()
    actions = {row.action for row in rows}
    assert {"login", "login_failed", "password_reset", "password_change", "logout"} <= actions
    dumped = json.dumps([[row.before, row.after] for row in rows])
    secrets = [TEST_PASSWORD, "wrong-pw-1", TEMP_PASSWORD, NEW_PASSWORD, refresh_cookie]
    for secret in secrets:
        assert secret not in dumped
    assert header["Authorization"].split()[1] not in dumped


def test_reset_password_refuses_an_id_too_large_for_the_database(client, make_user, login):
    headers = login(make_user("admin"))

    response = client.post(
        f"/api/users/{2**63}/reset-password",
        json={"temporary_password": "temp-test-only-pw"},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert "user_id" in response.json()["fields"]
