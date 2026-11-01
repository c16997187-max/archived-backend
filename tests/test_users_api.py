"""GET and POST /api/users. Roles are covered by tests/test_role_matrix.py."""

import json

import pytest
from sqlalchemy import select

from app.models import AuditLog, User
from app.models.user import ROLES
from app.schemas.users import UserCreate
from app.services import users as users_service
from app.services.passwords import verify_password

TEMP_PASSWORD = "temp-test-only-pw"


@pytest.fixture
def admin(make_user):
    return make_user("admin")


@pytest.fixture
def headers(admin, login):
    return login(admin)


def user_body(**fields):
    return {
        "id_number": "N-0001",
        "full_name": "New Testuser",
        "role": "registrar",
        "temporary_password": TEMP_PASSWORD,
        **fields,
    }


def create(client, headers, **fields):
    return client.post("/api/users", json=user_body(**fields), headers=headers)


def user_audit(db):
    return db.scalars(
        select(AuditLog).where(AuditLog.entity == "user", AuditLog.action == "create")
    ).all()


# --- creating ------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_creates_a_user_who_must_change_password(client, db, headers, role):
    response = create(client, headers, role=role)

    assert response.status_code == 201, response.text
    body = response.json()
    assert (body["id_number"], body["role"]) == ("N-0001", role)
    assert body["must_change_password"] is True
    assert body["is_active"] is True
    assert body["last_login_at"] is None
    user = db.get(User, body["id"])
    assert user.must_change_password is True
    assert verify_password(TEMP_PASSWORD, user.password_hash)


def test_writes_an_audit_row_with_no_password(client, db, admin, headers):
    response = create(client, headers)

    [row] = user_audit(db)
    assert (row.entity_id, row.user_id) == (response.json()["id"], admin.id)
    assert row.after == {
        "id_number": "N-0001",
        "full_name": "New Testuser",
        "role": "registrar",
        "must_change_password": True,
    }
    assert TEMP_PASSWORD not in json.dumps(row.after)
    assert TEMP_PASSWORD not in response.text
    assert "password_hash" not in response.json()


def test_new_user_must_change_password_before_using_the_api(client, headers, login, db):
    user_id = create(client, headers, role="reviewer").json()["id"]
    user_headers = login(db.get(User, user_id), password=TEMP_PASSWORD)

    blocked = client.get("/api/learners", headers=user_headers)
    assert blocked.status_code == 403
    assert blocked.json()["error"] == "password_change_required"

    changed = client.post(
        "/api/auth/change-password",
        json={"current_password": TEMP_PASSWORD, "new_password": "new-test-only-password"},
        headers=user_headers,
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["must_change_password"] is False

    fresh_headers = {"Authorization": f"Bearer {changed.json()['access_token']}"}
    assert client.get("/api/learners", headers=fresh_headers).status_code == 200


def test_text_is_trimmed(client, headers):
    response = create(client, headers, id_number="  N-0002 ", full_name=" New  Testuser ")

    assert response.status_code == 201, response.text
    assert response.json()["id_number"] == "N-0002"
    assert response.json()["full_name"] == "New Testuser"


def test_a_64_character_id_number_can_log_in(client, headers):
    id_number = "N" * 64

    assert create(client, headers, id_number=id_number).status_code == 201
    response = client.post(
        "/api/auth/login", json={"id_number": id_number, "password": TEMP_PASSWORD}
    )
    assert response.status_code == 200, response.text


# --- refused -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"role": "principal"}, "role"),
        ({"role": "ADMIN"}, "role"),
        ({"role": None}, "role"),
        ({"id_number": "  "}, "id_number"),
        ({"id_number": "N" * 65}, "id_number"),
        ({"id_number": "bad\x00id"}, "id_number"),
        ({"full_name": ""}, "full_name"),
        ({"full_name": "x" * 201}, "full_name"),
        ({"temporary_password": "short"}, "temporary_password"),
        ({"temporary_password": "x" * 257}, "temporary_password"),
        ({"must_change_password": False}, "must_change_password"),
        ({"is_active": False}, "is_active"),
    ],
)
def test_bad_fields_are_named_in_the_error(client, db, headers, fields, field):
    response = create(client, headers, **fields)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert field in response.json()["fields"]
    assert user_audit(db) == []


def test_missing_fields_are_named_in_the_error(client, headers):
    response = client.post("/api/users", json={}, headers=headers)

    assert response.status_code == 422
    assert set(response.json()["fields"]) == {
        "id_number",
        "full_name",
        "role",
        "temporary_password",
    }


def test_duplicate_id_number_is_refused(client, db, make_user, headers):
    existing = make_user("reviewer")

    response = create(client, headers, id_number=existing.id_number)

    assert response.status_code == 409
    assert response.json() == {
        "error": "duplicate",
        "fields": {"id_number": "Already used by another account"},
    }
    assert user_audit(db) == []


def test_two_admins_saving_the_same_id_number_at_once_gets_a_409(
    client, db, make_user, headers, monkeypatch
):
    existing = make_user("reviewer")
    # Simulates the race: the check ran before the other admin's account was saved.
    monkeypatch.setattr(db, "scalar", lambda *args, **kwargs: None)

    response = create(client, headers, id_number=existing.id_number)

    assert response.status_code == 409
    assert response.json()["fields"] == {"id_number": "Already used by another account"}


def test_cli_accounts_are_not_forced_to_change_password(db):
    user = users_service.create_user(db, "A-0001", "Test Admin", "admin", TEMP_PASSWORD)

    assert user.must_change_password is False


def test_user_create_roles_match_the_model():
    role_field = UserCreate.model_json_schema()["properties"]["role"]

    assert tuple(role_field["enum"]) == ROLES


# --- listing -------------------------------------------------------------------------


def test_lists_users_by_name_without_secrets(client, make_user, admin, headers):
    zed = make_user("reviewer")
    zed.full_name = "Zed Testuser"
    abe = make_user("registrar")
    abe.full_name = "Abe Testuser"

    response = client.get("/api/users", headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["page"], body["page_size"], body["total"]) == (1, 25, 3)
    names = [row["full_name"] for row in body["items"]]
    assert names == sorted(names)
    assert names[0] == "Abe Testuser"
    assert set(body["items"][0]) == {
        "id",
        "id_number",
        "full_name",
        "role",
        "must_change_password",
        "is_active",
        "last_login_at",
        "created_at",
    }


def test_list_pages(client, make_user, headers):
    for _ in range(3):
        make_user("reviewer")

    first = client.get("/api/users", params={"page_size": 2}, headers=headers).json()
    second = client.get("/api/users", params={"page_size": 2, "page": 2}, headers=headers).json()

    assert first["total"] == 4
    assert (len(first["items"]), len(second["items"])) == (2, 2)
    assert not {r["id"] for r in first["items"]} & {r["id"] for r in second["items"]}


@pytest.mark.parametrize(
    ("params", "field"), [({"page": 0}, "page"), ({"page_size": 101}, "page_size")]
)
def test_bad_list_parameters_are_refused(client, headers, params, field):
    response = client.get("/api/users", params=params, headers=headers)

    assert response.status_code == 422
    assert field in response.json()["fields"]
