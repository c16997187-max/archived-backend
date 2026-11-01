"""School year and section endpoints. Roles are covered by tests/test_role_matrix.py."""

import pytest
from sqlalchemy import select

from app.models import AuditLog, SchoolYear


@pytest.fixture
def headers(make_user, login):
    return login(make_user("admin"))


def year_body(**fields):
    return {"label": "2090-2091", "start_date": "2090-06-01", "end_date": "2091-03-31", **fields}


def audit_rows(db, entity):
    return db.scalars(select(AuditLog).where(AuditLog.entity == entity).order_by(AuditLog.id)).all()


# --- school years --------------------------------------------------------------------


def test_creates_a_school_year_with_an_audit_row(client, db, headers):
    response = client.post("/api/school-years", json=year_body(), headers=headers)

    assert response.status_code == 201, response.text
    assert response.json()["label"] == "2090-2091"
    assert response.json()["is_current"] is False
    row = audit_rows(db, "school_year")[0]
    assert (row.action, row.entity_id) == ("create", response.json()["id"])
    assert row.after["start_date"] == "2090-06-01"


def test_a_new_current_year_replaces_the_old_one_and_audits_both(client, db, headers, make_year):
    old = make_year(is_current=True)

    response = client.post("/api/school-years", json=year_body(is_current=True), headers=headers)

    assert response.status_code == 201, response.text
    assert response.json()["is_current"] is True
    db.refresh(old)
    assert old.is_current is False
    update, create = audit_rows(db, "school_year")
    assert (update.action, update.entity_id) == ("update", old.id)
    assert (update.before, update.after) == ({"is_current": True}, {"is_current": False})
    assert (create.action, create.entity_id) == ("create", response.json()["id"])


def test_duplicate_label_is_refused_and_changes_nothing(client, db, headers, make_year):
    current = make_year(is_current=True, start_year=2090)

    response = client.post("/api/school-years", json=year_body(is_current=True), headers=headers)

    assert response.status_code == 409
    assert response.json()["error"] == "duplicate"
    assert "label" in response.json()["fields"]
    db.refresh(current)
    assert current.is_current is True
    assert audit_rows(db, "school_year") == []


def test_two_admins_saving_the_same_label_at_once_gets_a_409(
    client, db, headers, make_year, monkeypatch
):
    make_year(start_year=2090)
    # Simulates the race: the check ran before the other admin's school year was saved.
    monkeypatch.setattr(db, "scalar", lambda *args, **kwargs: None)

    response = client.post("/api/school-years", json=year_body(), headers=headers)

    assert response.status_code == 409
    assert "label" in response.json()["fields"]


@pytest.mark.parametrize("label", ["2090", "2090-2092", "2090–2091", "abcd-efgh", ""])
def test_bad_labels_are_refused(client, headers, label):
    response = client.post("/api/school-years", json=year_body(label=label), headers=headers)

    assert response.status_code == 422
    assert "label" in response.json()["fields"]


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"end_date": "2090-06-01"}, "end_date"),
        ({"end_date": "2090-01-01"}, "end_date"),
        ({"start_date": "2090-13-01"}, "start_date"),
        ({"end_date": 1}, "end_date"),
        ({"extra": "field"}, "extra"),
    ],
)
def test_bad_school_year_fields_are_refused(client, db, headers, fields, field):
    response = client.post("/api/school-years", json=year_body(**fields), headers=headers)

    assert response.status_code == 422
    assert field in response.json()["fields"]
    assert db.scalars(select(SchoolYear)).all() == []


def test_lists_school_years_newest_first_with_paging(client, headers, make_year):
    old, new = make_year(start_year=2020), make_year(start_year=2030)

    body = client.get("/api/school-years", headers=headers).json()
    one_row = client.get(
        "/api/school-years", params={"page_size": 1, "page": 2}, headers=headers
    ).json()

    assert [row["id"] for row in body["items"]] == [new.id, old.id]
    assert (body["page"], body["page_size"], body["total"]) == (1, 25, 2)
    assert [row["id"] for row in one_row["items"]] == [old.id]


# --- sections ------------------------------------------------------------------------


def test_creates_a_section_with_an_audit_row(client, db, headers, make_year):
    year = make_year()

    response = client.post(
        "/api/sections",
        json={"school_year_id": year.id, "grade_level": "K", "name": " Rosal ", "adviser": "Fake"},
        headers=headers,
    )

    assert response.status_code == 201, response.text
    assert response.json()["grade_level"] == 0
    assert response.json()["name"] == "Rosal"
    row = audit_rows(db, "section")[0]
    assert (row.action, row.entity_id) == ("create", response.json()["id"])
    assert row.after["name"] == "Rosal"


def test_duplicate_section_is_refused(client, db, headers, make_year, make_section):
    year = make_year()
    make_section(year, grade_level=8, name="Rosal")

    response = client.post(
        "/api/sections",
        json={"school_year_id": year.id, "grade_level": 8, "name": "Rosal"},
        headers=headers,
    )

    assert response.status_code == 409
    assert response.json()["error"] == "duplicate"
    assert "name" in response.json()["fields"]
    assert audit_rows(db, "section") == []


def test_same_section_name_in_another_grade_is_fine(client, headers, make_year, make_section):
    year = make_year()
    make_section(year, grade_level=8, name="Rosal")

    response = client.post(
        "/api/sections",
        json={"school_year_id": year.id, "grade_level": 9, "name": "Rosal"},
        headers=headers,
    )

    assert response.status_code == 201


def test_section_needs_a_known_school_year(client, headers):
    response = client.post(
        "/api/sections",
        json={"school_year_id": 999999, "grade_level": 8, "name": "Rosal"},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["fields"] == {"school_year_id": "Unknown school year"}


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"grade_level": 13}, "grade_level"),
        ({"grade_level": "Grade 8"}, "grade_level"),
        ({"name": "  "}, "name"),
        ({"school_year_id": 2**63}, "school_year_id"),
    ],
)
def test_bad_section_fields_are_refused(client, headers, make_year, fields, field):
    body = {"school_year_id": make_year().id, "grade_level": 8, "name": "Rosal", **fields}

    response = client.post("/api/sections", json=body, headers=headers)

    assert response.status_code == 422
    assert field in response.json()["fields"]


def test_lists_sections_with_filters_and_paging(client, headers, make_year, make_section):
    year, other = make_year(), make_year()
    grade_8 = make_section(year, grade_level=8)
    kinder = make_section(year, grade_level=0)
    in_other = make_section(other, grade_level=8)

    def ids(**params):
        response = client.get("/api/sections", params=params, headers=headers)
        assert response.status_code == 200, response.text
        return [row["id"] for row in response.json()["items"]]

    assert ids(school_year=year.id) == [kinder.id, grade_8.id]
    assert ids(grade=8, school_year=other.id) == [in_other.id]
    assert ids(grade="K") == [kinder.id]
    assert set(ids(school_year="", grade="")) == {grade_8.id, kinder.id, in_other.id}
    assert len(ids(page_size=2)) == 2
