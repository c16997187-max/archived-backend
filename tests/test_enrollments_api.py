"""POST /api/enrollments. Roles are covered by tests/test_role_matrix.py."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.models import AuditLog, Enrollment, Learner
from app.schemas.errors import DuplicateLearnerError
from app.services import enrollments as enrollments_service
from app.services import learners as learners_service

LRN = "000000000017"
BIRTH_DATE = date(2012, 5, 20)


@pytest.fixture
def headers(make_user, login):
    return login(make_user("registrar"))


@pytest.fixture
def year(make_year):
    return make_year()


def new_learner(**fields):
    return {"last_name": "Testlast", "first_name": "Testfirst", **fields}


def audit_rows(db, entity):
    return db.scalars(select(AuditLog).where(AuditLog.entity == entity)).all()


def count(db, model):
    return db.scalar(select(func.count()).select_from(model))


def post(client, headers, year, **body):
    return client.post(
        "/api/enrollments", json={"school_year_id": year.id, **body}, headers=headers
    )


# --- creating -----------------------------------------------------------------


def test_creates_learner_and_draft_enrollment_with_audit_rows(client, db, headers, year):
    response = post(
        client,
        headers,
        year,
        remarks="Transferee",
        learner=new_learner(lrn=LRN, birth_date="2012-05-20", contact_number="09000000017"),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["learner"]["lrn"] == LRN
    assert body["enrollment"]["status"] == "draft"
    assert body["enrollment"]["school_year_label"] == year.label
    assert body["enrollment"]["remarks"] == "Transferee"

    learner_audit = audit_rows(db, "learner")[0]
    assert learner_audit.action == "create"
    assert learner_audit.entity_id == body["learner"]["id"]
    assert learner_audit.after["birth_date"] == "2012-05-20"
    assert learner_audit.after["source"] == "manual"
    assert "middle_name" not in learner_audit.after
    assert "duplicate_override" not in learner_audit.after

    enrollment_audit = audit_rows(db, "enrollment")[0]
    assert enrollment_audit.entity_id == body["enrollment"]["id"]
    assert enrollment_audit.after["status"] == "draft"
    assert enrollment_audit.after["source"] == "manual"


def test_a_failed_audit_write_saves_nothing(client, db, make_user, login, year, monkeypatch):
    headers = login(make_user("registrar"))
    db.commit()  # keeps the fixtures through the rollback below
    real_record = learners_service.audit.record

    def fail_for_enrollment(session, action, entity, *args, **kwargs):
        if entity == "enrollment":
            raise RuntimeError("audit failed")
        return real_record(session, action, entity, *args, **kwargs)

    monkeypatch.setattr(learners_service.audit, "record", fail_for_enrollment)

    with pytest.raises(RuntimeError):
        post(client, headers, year, learner=new_learner())
    db.rollback()  # what closing the request's session does in production

    assert count(db, Learner) == 0
    assert count(db, Enrollment) == 0
    assert audit_rows(db, "learner") == []


# --- school year, section and grade -------------------------------------------


def test_grade_is_kept_on_a_draft_with_no_section(client, db, headers, year):
    response = post(client, headers, year, grade_level=8, learner=new_learner())

    assert response.status_code == 201, response.text
    assert response.json()["enrollment"]["grade_level"] == 8
    assert response.json()["enrollment"]["section_id"] is None
    assert db.scalars(select(Enrollment)).one().grade_level == 8


def test_grade_is_filled_in_from_the_section(client, headers, year, make_section):
    section = make_section(year, grade_level=3)

    response = post(client, headers, year, section_id=section.id, learner=new_learner())

    assert response.status_code == 201, response.text
    assert response.json()["enrollment"]["grade_level"] == 3
    assert response.json()["enrollment"]["section_name"] == section.name


@pytest.mark.parametrize("grade", ["K", "k", 0])
def test_kindergarten_is_stored_as_zero(client, headers, year, grade):
    response = post(client, headers, year, grade_level=grade, learner=new_learner())

    assert response.status_code == 201, response.text
    assert response.json()["enrollment"]["grade_level"] == 0


def test_grade_must_match_the_section(client, headers, year, make_section):
    section = make_section(year, grade_level=3)

    response = post(
        client, headers, year, section_id=section.id, grade_level=4, learner=new_learner()
    )

    assert response.status_code == 422
    assert response.json()["fields"] == {"grade_level": "Does not match the section's grade"}


def test_section_must_belong_to_the_school_year(client, db, headers, year, make_year, make_section):
    other_section = make_section(make_year())

    response = post(client, headers, year, section_id=other_section.id, learner=new_learner())

    assert response.status_code == 422
    assert "section_id" in response.json()["fields"]
    assert count(db, Learner) == 0


def test_unknown_section_is_refused(client, headers, year):
    response = post(client, headers, year, section_id=999999, learner=new_learner())

    assert response.status_code == 422
    assert "section_id" in response.json()["fields"]


def test_unknown_school_year_is_refused(client, headers):
    response = client.post(
        "/api/enrollments",
        json={"school_year_id": 999999, "learner": new_learner()},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["fields"] == {"school_year_id": "Unknown school year"}


# --- linking an existing learner ------------------------------------------------


def test_links_an_existing_learner_without_creating_one(client, db, headers, year, make_learner):
    learner = make_learner()

    response = post(client, headers, year, learner_id=learner.id)

    assert response.status_code == 201, response.text
    assert response.json()["learner"]["id"] == learner.id
    assert count(db, Learner) == 1
    assert audit_rows(db, "learner") == []
    assert len(audit_rows(db, "enrollment")) == 1


def test_cannot_link_a_soft_deleted_learner(client, headers, year, make_learner):
    learner = make_learner(deleted_at=datetime.now(UTC))

    response = post(client, headers, year, learner_id=learner.id)

    assert response.status_code == 422
    assert response.json()["fields"] == {"learner_id": "Unknown learner"}


def test_cannot_enroll_twice_in_a_year(client, headers, year, make_learner, make_enrollment):
    learner = make_learner()
    make_enrollment(learner, year)

    response = post(client, headers, year, learner_id=learner.id)

    assert response.status_code == 409
    assert response.json()["error"] == "already_enrolled"
    assert "school_year_id" in response.json()["fields"]


def test_can_reenroll_after_a_soft_deleted_enrollment(
    client, headers, year, make_learner, make_enrollment
):
    learner = make_learner()
    make_enrollment(learner, year, deleted_at=datetime.now(UTC))

    assert post(client, headers, year, learner_id=learner.id).status_code == 201


def test_two_people_enrolling_the_same_learner_at_once_gets_a_409(
    client, headers, year, make_learner, make_enrollment, db, monkeypatch
):
    learner = make_learner()
    make_enrollment(learner, year)
    # Simulates the race: the check ran before the other person's enrollment was saved.
    monkeypatch.setattr(enrollments_service, "_linked_learner", lambda *args: learner)

    response = post(client, headers, year, learner_id=learner.id)

    assert response.status_code == 409
    assert response.json()["error"] == "already_enrolled"


# --- duplicates -----------------------------------------------------------------


def test_same_lrn_returns_the_match_instead_of_creating(
    client, db, headers, year, make_learner, make_section, make_enrollment
):
    existing = make_learner(last_name="Otherlast", lrn=LRN, religion="Fake religion")
    make_enrollment(existing, year, make_section(year, grade_level=5))

    response = post(client, headers, year, learner=new_learner(lrn=LRN))

    assert response.status_code == 409
    body = response.json()
    DuplicateLearnerError.model_validate(body)
    assert body["error"] == "possible_duplicate"
    assert body["fields"] == {"lrn": "Already used by another learner"}
    assert [m["id"] for m in body["matches"]] == [existing.id]
    assert body["matches"][0]["reasons"] == ["lrn"]
    assert body["matches"][0]["enrollment"]["grade_level"] == 5
    # Enough to recognise the student, and no more.
    assert "religion" not in body["matches"][0]
    assert count(db, Learner) == 1
    assert count(db, Enrollment) == 1


def test_same_lrn_cannot_be_overridden(client, db, headers, year, make_learner):
    make_learner(last_name="Otherlast", lrn=LRN)

    response = post(client, headers, year, confirm_new_learner=True, learner=new_learner(lrn=LRN))

    assert response.status_code == 409
    assert count(db, Learner) == 1


@pytest.mark.parametrize(
    ("last_name", "first_name"),
    [
        ("Testlast", "Testfirst"),
        ("TESTLAST", "testfirst"),
        ("  Testlast ", "Testfirst"),
    ],
)
def test_same_name_and_birth_date_returns_the_match(
    client, db, headers, year, make_learner, last_name, first_name
):
    existing = make_learner(birth_date=BIRTH_DATE)

    response = post(
        client,
        headers,
        year,
        learner={"last_name": last_name, "first_name": first_name, "birth_date": "2012-05-20"},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "possible_duplicate"
    assert "learner" in body["fields"]
    assert [m["id"] for m in body["matches"]] == [existing.id]
    assert body["matches"][0]["reasons"] == ["name_birth_date"]
    assert count(db, Learner) == 1


def test_stored_names_with_extra_spaces_still_match(client, headers, year, make_learner):
    make_learner(last_name="Dela  Cruz", birth_date=BIRTH_DATE)

    response = post(
        client,
        headers,
        year,
        learner=new_learner(last_name="dela cruz", birth_date="2012-05-20"),
    )

    assert response.status_code == 409


def test_ntilde_names_match_case_insensitively(client, headers, year, make_learner):
    make_learner(last_name="Peña", birth_date=BIRTH_DATE)

    response = post(
        client, headers, year, learner=new_learner(last_name="PEÑA", birth_date="2012-05-20")
    )

    assert response.status_code == 409


def test_a_learner_matching_both_ways_reports_both_reasons(client, headers, year, make_learner):
    make_learner(lrn=LRN, birth_date=BIRTH_DATE)

    response = post(client, headers, year, learner=new_learner(lrn=LRN, birth_date="2012-05-20"))

    assert response.status_code == 409
    assert response.json()["matches"][0]["reasons"] == ["lrn", "name_birth_date"]
    assert response.json()["fields"] == {"lrn": "Already used by another learner"}


def test_confirming_a_same_name_match_creates_and_records_the_override(
    client, db, headers, year, make_learner
):
    first = make_learner(birth_date=BIRTH_DATE)
    second = make_learner(birth_date=BIRTH_DATE)

    response = post(
        client,
        headers,
        year,
        confirm_new_learner=True,
        learner=new_learner(birth_date="2012-05-20"),
    )

    assert response.status_code == 201, response.text
    assert count(db, Learner) == 3
    after = audit_rows(db, "learner")[0].after
    assert after["duplicate_override"] is True
    assert sorted(after["duplicate_override_matches"]) == sorted([first.id, second.id])


def test_confirming_with_no_match_records_no_override(client, db, headers, year):
    response = post(client, headers, year, confirm_new_learner=True, learner=new_learner())

    assert response.status_code == 201
    after = audit_rows(db, "learner")[0].after
    assert "duplicate_override" not in after
    assert "duplicate_override_matches" not in after


def test_soft_deleted_learners_are_not_duplicates(client, headers, year, make_learner):
    make_learner(lrn=LRN, birth_date=BIRTH_DATE, deleted_at=datetime.now(UTC))

    response = post(client, headers, year, learner=new_learner(lrn=LRN, birth_date="2012-05-20"))

    assert response.status_code == 201, response.text


def test_same_name_with_another_birth_date_is_not_a_duplicate(client, headers, year, make_learner):
    make_learner(birth_date=BIRTH_DATE)

    response = post(client, headers, year, learner=new_learner(birth_date="2012-05-21"))

    assert response.status_code == 201, response.text


def test_same_name_with_no_birth_date_is_not_a_duplicate(client, headers, year, make_learner):
    make_learner()

    assert post(client, headers, year, learner=new_learner()).status_code == 201


def test_two_people_saving_the_same_lrn_at_once_gets_the_match(
    client, db, headers, year, make_learner, monkeypatch
):
    existing = make_learner(last_name="Otherlast", lrn=LRN)
    real_find = learners_service.find_duplicates
    calls = []

    def miss_the_first_time(session, data):
        calls.append(data)
        return [] if len(calls) == 1 else real_find(session, data)

    # Simulates the race: the check ran before the other person's learner was saved.
    monkeypatch.setattr(learners_service, "find_duplicates", miss_the_first_time)

    response = post(client, headers, year, learner=new_learner(lrn=LRN))

    assert response.status_code == 409
    assert response.json()["error"] == "possible_duplicate"
    assert [m["id"] for m in response.json()["matches"]] == [existing.id]


# --- validation -----------------------------------------------------------------


def test_needs_learner_or_learner_id_but_not_both(client, headers, year, make_learner):
    neither = post(client, headers, year)
    both = post(client, headers, year, learner_id=make_learner().id, learner=new_learner())

    for response in (neither, both):
        assert response.status_code == 422
        assert response.json()["fields"] == {"request": "Send either learner or learner_id"}


@pytest.mark.parametrize(
    ("fields", "field", "message"),
    [
        ({"lrn": "12345"}, "lrn", "Must be exactly 12 digits"),
        ({"contact_number": "0917123"}, "contact_number", "Must be 11 digits starting with 09"),
        ({"contact_number": "+639171234567"}, "contact_number", None),
        ({"birth_date": "2012-02-30"}, "birth_date", "Must be a real date in YYYY-MM-DD format"),
        ({"birth_date": 1336000000}, "birth_date", "Must be a real date in YYYY-MM-DD format"),
        ({"sex": "X"}, "sex", None),
        ({"last_name": "   "}, "last_name", "Required"),
        ({"last_name": "Bad\x00name"}, "last_name", "Contains an invalid character"),
        ({"middle_name": "x" * 201}, "middle_name", None),
        ({"nickname": "Unknown field"}, "nickname", None),
    ],
)
def test_bad_learner_fields_are_named_in_the_error(client, headers, year, fields, field, message):
    response = post(client, headers, year, learner=new_learner(**fields))

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert f"learner.{field}" in body["fields"]
    if message:
        assert body["fields"][f"learner.{field}"] == message


def test_birth_date_must_be_in_the_past(client, headers, year):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    response = post(client, headers, year, learner=new_learner(birth_date=tomorrow))

    assert response.status_code == 422
    assert response.json()["fields"] == {"learner.birth_date": "Must be in the past"}


def test_blank_optional_fields_are_saved_as_empty(client, headers, year):
    response = post(client, headers, year, learner=new_learner(lrn="", contact_number=" "))

    assert response.status_code == 201, response.text
    assert response.json()["learner"]["lrn"] is None
    assert response.json()["learner"]["contact_number"] is None


@pytest.mark.parametrize("grade", [13, -1, "Grade 8", True, 8.5])
def test_bad_grade_levels_are_refused(client, headers, year, grade):
    response = post(client, headers, year, grade_level=grade, learner=new_learner())

    assert response.status_code == 422
    assert response.json()["fields"] == {"grade_level": "Must be K or a grade from 0 to 12"}


@pytest.mark.parametrize("field", ["school_year_id", "section_id", "learner_id"])
def test_ids_too_large_for_the_database_are_refused(client, headers, year, field):
    body = {"school_year_id": year.id, "learner": new_learner(), field: 2**63}

    response = client.post("/api/enrollments", json=body, headers=headers)

    assert response.status_code == 422
    assert field in response.json()["fields"]


# --- duplicates: accents ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "typed"),
    [("Peña", "Pena"), ("Muñoz", "Munoz"), ("Pena", "Peña"), ("Peña", "PENA")],
)
def test_same_name_ignoring_accents_returns_the_match(
    client, db, headers, year, make_learner, stored, typed
):
    existing = make_learner(last_name=stored, birth_date=BIRTH_DATE)

    response = post(
        client, headers, year, learner=new_learner(last_name=typed, birth_date="2012-05-20")
    )

    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "possible_duplicate"
    assert [m["id"] for m in body["matches"]] == [existing.id]
    assert body["matches"][0]["last_name"] == stored
    assert body["matches"][0]["reasons"] == ["name_birth_date"]
    assert count(db, Learner) == 1
    assert count(db, Enrollment) == 0


def test_accents_in_first_names_are_ignored_too(client, headers, year, make_learner):
    make_learner(first_name="Niño", birth_date=BIRTH_DATE)

    response = post(
        client, headers, year, learner=new_learner(first_name="Nino", birth_date="2012-05-20")
    )

    assert response.status_code == 409


def test_same_name_ignoring_accents_with_another_birth_date_is_not_a_duplicate(
    client, headers, year, make_learner
):
    make_learner(last_name="Peña", birth_date=BIRTH_DATE)

    response = post(
        client, headers, year, learner=new_learner(last_name="Pena", birth_date="2012-05-21")
    )

    assert response.status_code == 201, response.text


def test_confirming_an_accent_match_creates_and_records_the_override(
    client, db, headers, year, make_learner
):
    existing = make_learner(last_name="Peña", birth_date=BIRTH_DATE)

    response = post(
        client,
        headers,
        year,
        confirm_new_learner=True,
        learner=new_learner(last_name="Pena", birth_date="2012-05-20"),
    )

    assert response.status_code == 201, response.text
    assert response.json()["learner"]["last_name"] == "Pena"
    assert count(db, Learner) == 2
    after = audit_rows(db, "learner")[0].after
    assert after["duplicate_override"] is True
    assert after["duplicate_override_matches"] == [existing.id]


def test_an_accent_match_cannot_override_a_matching_lrn(client, db, headers, year, make_learner):
    make_learner(last_name="Peña", lrn=LRN, birth_date=BIRTH_DATE)

    response = post(
        client,
        headers,
        year,
        confirm_new_learner=True,
        learner=new_learner(last_name="Pena", lrn=LRN, birth_date="2012-05-20"),
    )

    assert response.status_code == 409
    assert response.json()["matches"][0]["reasons"] == ["lrn", "name_birth_date"]
    assert count(db, Learner) == 1
