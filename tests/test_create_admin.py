import pytest
from sqlalchemy import select

from app.errors import ApiError
from app.models import AuditLog
from app.services.passwords import verify_password
from scripts import create_admin as script

PASSWORD = "test-only-admin-pw"


def test_create_admin_creates_hashed_account_and_audit_row(db):
    user = script.create_admin(db, " A-0001 ", "Test Admin", PASSWORD)

    assert (user.id_number, user.role, user.must_change_password) == ("A-0001", "admin", False)
    assert verify_password(PASSWORD, user.password_hash)
    [row] = db.scalars(select(AuditLog)).all()
    assert (row.action, row.entity_id, row.user_id) == ("create", user.id, None)
    assert row.after == {"id_number": "A-0001", "full_name": "Test Admin", "role": "admin"}


def test_create_admin_refuses_duplicate_id_number(db, make_user):
    existing = make_user("registrar")

    with pytest.raises(ApiError) as exc:
        script.create_admin(db, existing.id_number, "Someone Else", PASSWORD)

    assert exc.value.status_code == 409
    assert "id_number" in exc.value.fields


def test_create_admin_refuses_short_password(db):
    with pytest.raises(ApiError) as exc:
        script.create_admin(db, "A-0002", "Test Admin", "short")

    assert "password" in exc.value.fields


def test_new_admin_can_log_in(client, db):
    user = script.create_admin(db, "A-0003", "Test Admin", PASSWORD)

    response = client.post("/api/auth/login", json={"id_number": "A-0003", "password": PASSWORD})

    assert response.status_code == 200
    assert response.json()["user"]["id"] == user.id


def test_main_stops_when_passwords_do_not_match(monkeypatch, capsys):
    answers = iter(["first-password", "second-password"])
    monkeypatch.setattr(script.getpass, "getpass", lambda prompt: next(answers))

    assert script.main(["--id-number", "A-0004", "--full-name", "Test Admin"]) == 1
    assert "do not match" in capsys.readouterr().err
