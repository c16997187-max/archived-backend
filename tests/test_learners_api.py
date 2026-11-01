"""GET /api/learners, GET and PATCH /api/learners/{id}. Roles are in tests/test_role_matrix.py."""

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from app.models import AuditLog, Learner

LRN = "000000000017"
OTHER_LRN = "000000000018"


@pytest.fixture
def headers(make_user, login):
    return login(make_user("registrar"))


def search(client, headers, **params):
    response = client.get("/api/learners", params=params, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def found_ids(client, headers, **params):
    return [row["id"] for row in search(client, headers, **params)["items"]]


def learner_audit(db):
    return db.scalars(select(AuditLog).where(AuditLog.entity == "learner")).all()


# --- search: names ---------------------------------------------------------------


@pytest.mark.parametrize("q", ["Santos", "santos", "Santoz", "Sant", "San"])
def test_finds_a_name_exactly_partly_or_misspelled(client, headers, make_learner, q):
    santos = make_learner(last_name="Santos", first_name="Testa")
    make_learner(last_name="Bautista", first_name="Testb")

    assert found_ids(client, headers, q=q) == [santos.id]


def test_finds_by_first_name(client, headers, make_learner):
    miguel = make_learner(first_name="Miguel")
    make_learner(first_name="Angela")

    assert found_ids(client, headers, q="Migel") == [miguel.id]


@pytest.mark.parametrize("name", ["Peña", "Muñoz", "Ñora"])
@pytest.mark.parametrize("transform", [str, str.lower, str.upper])
def test_finds_names_with_ntilde(client, headers, make_learner, name, transform):
    match = make_learner(last_name=name, first_name="Testa")
    make_learner(last_name="Bautista", first_name="Testb")

    assert found_ids(client, headers, q=transform(name)) == [match.id]


def test_ntilde_name_is_found_alongside_a_first_name(client, headers, make_learner):
    match = make_learner(last_name="Muñoz", first_name="Maria")
    make_learner(last_name="Muñoz", first_name="Jose")

    assert found_ids(client, headers, q="Maria Muñoz") == [match.id]


def test_ntilde_names_come_back_unchanged(client, headers, make_learner):
    make_learner(last_name="Peña", first_name="Niño")

    row = search(client, headers, q="Peña")["items"][0]

    assert (row["last_name"], row["first_name"]) == ("Peña", "Niño")


def test_a_two_word_last_name_finds_everyone_with_it(client, headers, make_learner):
    juan = make_learner(last_name="Dela Cruz", first_name="Juan")
    angela = make_learner(last_name="Dela Cruz", first_name="Angela")
    make_learner(last_name="Bautista", first_name="Juan")

    assert set(found_ids(client, headers, q="Dela Cruz")) == {juan.id, angela.id}


@pytest.mark.parametrize("q", ["Juan Cruz", "Cruz, Juan", "Cruz,Juan", "juan dela cruz"])
def test_several_words_must_all_match(client, headers, make_learner, q):
    match = make_learner(last_name="Dela Cruz", first_name="Juan")
    make_learner(last_name="Dela Cruz", first_name="Angela")
    make_learner(last_name="Bautista", first_name="Juan")

    assert found_ids(client, headers, q=q) == [match.id]


@pytest.mark.parametrize("q", ["%", "_", "S%s", "\\"])
def test_like_wildcards_are_literal(client, headers, make_learner, q):
    make_learner(last_name="Santos")

    assert found_ids(client, headers, q=q) == []


def test_the_closest_name_comes_first(client, headers, make_learner):
    near = make_learner(last_name="Santosa", first_name="Testa")
    exact = make_learner(last_name="Santos", first_name="Testa")

    assert found_ids(client, headers, q="Santos Testa") == [exact.id, near.id]


# --- search: LRN -----------------------------------------------------------------


@pytest.mark.parametrize("q", [LRN, LRN[:6]])
def test_finds_by_lrn_or_its_start(client, headers, make_learner, q):
    match = make_learner(lrn=LRN)
    make_learner(lrn="999999999999")
    make_learner()

    assert found_ids(client, headers, q=q) == [match.id]


def test_digits_are_not_matched_in_the_middle_of_an_lrn(client, headers, make_learner):
    make_learner(lrn=LRN)

    assert found_ids(client, headers, q="17") == []


# --- search: filters ---------------------------------------------------------------


def test_filters_by_grade_section_and_school_year(
    client, headers, make_year, make_section, make_learner, make_enrollment
):
    year, other_year = make_year(), make_year()
    grade_8 = make_section(year, grade_level=8)
    grade_9 = make_section(year, grade_level=9)
    in_8 = make_learner()
    in_9 = make_learner()
    other = make_learner()
    make_enrollment(in_8, year, grade_8)
    make_enrollment(in_9, year, grade_9)
    make_enrollment(other, other_year)

    assert found_ids(client, headers, grade=8) == [in_8.id]
    assert found_ids(client, headers, section=grade_9.id) == [in_9.id]
    assert found_ids(client, headers, school_year=other_year.id) == [other.id]


def test_grade_filter_finds_a_draft_with_no_section(
    client, headers, make_year, make_learner, make_enrollment
):
    learner = make_learner()
    make_enrollment(learner, make_year(), grade_level=8)

    assert found_ids(client, headers, grade=8) == [learner.id]


def test_kindergarten_filter(client, headers, make_year, make_learner, make_enrollment):
    learner = make_learner()
    make_enrollment(learner, make_year(), grade_level=0)
    make_enrollment(make_learner(), make_year(), grade_level=1)

    assert found_ids(client, headers, grade="K") == [learner.id]
    assert found_ids(client, headers, grade=0) == [learner.id]


def test_filters_must_hold_for_the_same_enrollment(
    client, headers, make_year, make_section, make_learner, make_enrollment
):
    last_year, this_year = make_year(), make_year()
    learner = make_learner()
    make_enrollment(learner, last_year, make_section(last_year, grade_level=8))
    make_enrollment(learner, this_year, make_section(this_year, grade_level=9))

    assert found_ids(client, headers, school_year=this_year.id, grade=8) == []
    assert found_ids(client, headers, school_year=this_year.id, grade=9) == [learner.id]


def test_row_shows_the_enrollment_that_matched_the_filter(
    client, headers, make_year, make_section, make_learner, make_enrollment
):
    last_year, this_year = make_year(), make_year()
    learner = make_learner()
    make_enrollment(learner, last_year, make_section(last_year, grade_level=8))
    make_enrollment(learner, this_year, make_section(this_year, grade_level=9))

    newest = search(client, headers)["items"][0]["enrollment"]
    filtered = search(client, headers, school_year=last_year.id)["items"][0]["enrollment"]

    assert (newest["school_year_id"], newest["grade_level"]) == (this_year.id, 9)
    assert (filtered["school_year_id"], filtered["grade_level"]) == (last_year.id, 8)


def test_learner_with_several_enrollments_appears_once(
    client, headers, make_year, make_learner, make_enrollment
):
    learner = make_learner()
    make_enrollment(learner, make_year(), grade_level=8)
    make_enrollment(learner, make_year(), grade_level=8)

    body = search(client, headers, grade=8)

    assert [row["id"] for row in body["items"]] == [learner.id]
    assert body["total"] == 1


def test_learners_with_no_enrollment_appear_when_nothing_is_filtered(
    client, headers, make_year, make_learner
):
    learner = make_learner()

    assert found_ids(client, headers) == [learner.id]
    assert search(client, headers)["items"][0]["enrollment"] is None
    assert found_ids(client, headers, school_year=make_year().id) == []


def test_unknown_filter_ids_return_an_empty_page(client, headers, make_learner):
    make_learner()

    assert search(client, headers, section=999999)["total"] == 0


def test_blank_parameters_mean_no_filter(client, headers, make_learner):
    learner = make_learner()

    assert found_ids(client, headers, q="", grade="", section="", school_year="") == [learner.id]


def test_soft_deleted_rows_are_hidden(client, headers, make_year, make_learner, make_enrollment):
    make_learner(last_name="Deleted", deleted_at=datetime.now(UTC))
    learner = make_learner()
    make_enrollment(learner, make_year(), grade_level=8, deleted_at=datetime.now(UTC))

    assert found_ids(client, headers) == [learner.id]
    assert search(client, headers)["items"][0]["enrollment"] is None
    assert found_ids(client, headers, grade=8) == []


# --- search: paging and shape --------------------------------------------------------


def test_pages_default_to_25_rows(client, headers, make_learner):
    for n in range(27):
        make_learner(last_name=f"Testlast{n:02d}")

    first = search(client, headers)
    second = search(client, headers, page=2)

    assert (first["page"], first["page_size"], first["total"]) == (1, 25, 27)
    assert len(first["items"]) == 25
    assert (second["page"], len(second["items"])) == (2, 2)
    assert not {r["id"] for r in first["items"]} & {r["id"] for r in second["items"]}


def test_page_size_can_be_chosen(client, headers, make_learner):
    for _ in range(3):
        make_learner()

    body = search(client, headers, page_size=2)

    assert (len(body["items"]), body["total"]) == (2, 3)


def test_a_page_past_the_end_is_empty(client, headers, make_learner):
    make_learner()

    body = search(client, headers, page=5)

    assert (body["items"], body["total"]) == ([], 1)


def test_rows_leave_out_sensitive_details(client, headers, make_learner):
    make_learner(religion="Fake", ip_group="Fake", barangay="Fake", guardian_name="Fake")

    row = search(client, headers)["items"][0]

    assert not {"religion", "ip_group", "barangay", "guardian_name"} & set(row)


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"grade": 13}, "grade"),
        ({"grade": "Grade 8"}, "grade"),
        ({"section": "abc"}, "section"),
        ({"section": 2**63}, "section"),
        ({"school_year": 0}, "school_year"),
        ({"page": 0}, "page"),
        ({"page_size": 101}, "page_size"),
        ({"page": 10**7}, "page"),
        ({"q": "x" * 101}, "q"),
        ({"q": "bad\x00text"}, "q"),
    ],
)
def test_bad_parameters_are_named_in_the_error(client, headers, params, field):
    response = client.get("/api/learners", params=params, headers=headers)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert field in response.json()["fields"]


# --- detail --------------------------------------------------------------------------


def test_detail_has_all_fields_and_enrollments_newest_first(
    client, headers, make_year, make_section, make_learner, make_enrollment
):
    old_year, new_year = make_year(), make_year()
    learner = make_learner(lrn=LRN, religion="Fake", birth_date=date(2012, 5, 20))
    make_enrollment(learner, old_year, make_section(old_year, grade_level=8), status="approved")
    make_enrollment(learner, new_year, make_section(new_year, grade_level=9))
    make_enrollment(
        learner, make_year(start_year=2050), grade_level=10, deleted_at=datetime.now(UTC)
    )

    response = client.get(f"/api/learners/{learner.id}", headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["lrn"], body["religion"], body["birth_date"]) == (LRN, "Fake", "2012-05-20")
    assert [e["school_year_id"] for e in body["enrollments"]] == [new_year.id, old_year.id]
    assert body["enrollments"][1]["status"] == "approved"


@pytest.mark.parametrize("method", ["GET", "PATCH"])
def test_unknown_and_deleted_learners_are_404(client, headers, make_learner, method):
    deleted = make_learner(deleted_at=datetime.now(UTC))

    for learner_id in (deleted.id, 999999):
        response = client.request(
            method,
            f"/api/learners/{learner_id}",
            json={} if method == "PATCH" else None,
            headers=headers,
        )
        assert response.status_code == 404
        assert response.json() == {"error": "not_found", "fields": {}}


@pytest.mark.parametrize("learner_id", [2**63, 0, "abc"])
def test_bad_learner_ids_are_422(client, headers, learner_id):
    response = client.get(f"/api/learners/{learner_id}", headers=headers)

    assert response.status_code == 422
    assert "learner_id" in response.json()["fields"]


# --- edit ----------------------------------------------------------------------------


def patch(client, headers, learner, **fields):
    return client.patch(f"/api/learners/{learner.id}", json=fields, headers=headers)


def test_edit_changes_only_the_fields_sent_and_audits_them(client, db, headers, make_learner):
    learner = make_learner(contact_number="09000000017", barangay="Fake Old")

    response = patch(
        client,
        headers,
        learner,
        contact_number="09000000018",
        barangay="Fake Old",
        birth_date="2012-05-20",
    )

    assert response.status_code == 200, response.text
    assert response.json()["contact_number"] == "09000000018"
    assert response.json()["barangay"] == "Fake Old"

    row = learner_audit(db)[0]
    assert (row.action, row.entity_id) == ("update", learner.id)
    assert row.before == {"contact_number": "09000000017", "birth_date": None}
    assert row.after == {
        "contact_number": "09000000018",
        "birth_date": "2012-05-20",
        "source": "manual",
    }


def test_edit_with_no_changes_writes_no_audit_row(client, db, headers, make_learner):
    learner = make_learner(barangay="Fake")

    assert patch(client, headers, learner).status_code == 200
    assert patch(client, headers, learner, barangay="Fake").status_code == 200
    assert learner_audit(db) == []


def test_edit_works_on_approved_records_for_corrections(
    client, db, headers, make_year, make_learner, make_enrollment
):
    learner = make_learner()
    make_enrollment(learner, make_year(), status="approved")

    assert patch(client, headers, learner, barangay="Fake").status_code == 200
    assert len(learner_audit(db)) == 1


def test_an_optional_field_can_be_cleared(client, headers, make_learner):
    learner = make_learner(lrn=LRN, barangay="Fake")

    response = patch(client, headers, learner, lrn=None, barangay="")

    assert response.status_code == 200, response.text
    assert response.json()["lrn"] is None
    assert response.json()["barangay"] is None


@pytest.mark.parametrize("field", ["last_name", "first_name"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_a_name_cannot_be_cleared(client, db, headers, make_learner, field, value):
    learner = make_learner()

    response = patch(client, headers, learner, **{field: value})

    assert response.status_code == 422
    assert response.json()["fields"] == {field: "Required"}
    assert learner_audit(db) == []


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"lrn": "12345"}, "lrn"),
        ({"contact_number": "12345"}, "contact_number"),
        ({"birth_date": "not a date"}, "birth_date"),
        ({"birth_date": "2999-01-01"}, "birth_date"),
        ({"sex": "X"}, "sex"),
        ({"deleted_at": "2026-01-01"}, "deleted_at"),
        ({"id": 5}, "id"),
    ],
)
def test_bad_edits_are_refused(client, db, headers, make_learner, fields, field):
    learner = make_learner()

    response = patch(client, headers, learner, **fields)

    assert response.status_code == 422
    assert field in response.json()["fields"]
    assert learner_audit(db) == []


def test_edit_refuses_an_lrn_held_by_another_learner(client, db, headers, make_learner):
    make_learner(lrn=LRN)
    learner = make_learner(lrn=OTHER_LRN)

    response = patch(client, headers, learner, lrn=LRN)

    assert response.status_code == 409
    assert response.json() == {
        "error": "duplicate",
        "fields": {"lrn": "Already used by another learner"},
    }
    assert learner_audit(db) == []


def test_edit_can_take_an_lrn_from_a_soft_deleted_learner(client, headers, make_learner):
    make_learner(lrn=LRN, deleted_at=datetime.now(UTC))
    learner = make_learner()

    assert patch(client, headers, learner, lrn=LRN).status_code == 200


def test_two_people_saving_the_same_lrn_at_once_gets_a_409(
    client, db, headers, make_learner, monkeypatch
):
    make_learner(lrn=LRN)
    learner = make_learner()
    # Simulates the race: the check ran before the other person's LRN was saved.
    monkeypatch.setattr(db, "scalar", lambda *args, **kwargs: None)

    response = patch(client, headers, learner, lrn=LRN)

    assert response.status_code == 409
    assert response.json()["error"] == "duplicate"
    monkeypatch.undo()
    assert db.get(Learner, learner.id).lrn is None


# --- search: accents -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "q"),
    [
        ("Peña", "Pena"),
        ("Peña", "pena"),
        ("Peña", "PENA"),
        ("Muñoz", "Munoz"),
        ("Muñoz", "munoz"),
        ("Muñoz", "MUNOZ"),
        ("Pena", "Peña"),
        ("Muñoz", "Mun"),
        ("Muñoz", "Munos"),
        ("José", "Jose"),
        ("Jose", "José"),
    ],
)
def test_search_ignores_accents(client, headers, make_learner, stored, q):
    match = make_learner(last_name=stored, first_name="Testa")
    make_learner(last_name="Bautista", first_name="Testb")

    assert found_ids(client, headers, q=q) == [match.id]


def test_search_ignores_accents_in_first_names(client, headers, make_learner):
    match = make_learner(last_name="Testa", first_name="Niño")
    make_learner(last_name="Testb", first_name="Angela")

    assert found_ids(client, headers, q="Nino") == [match.id]


def test_accented_names_come_back_unchanged(client, headers, make_learner):
    make_learner(last_name="Peña", first_name="Niño")

    row = search(client, headers, q="Pena Nino")["items"][0]

    assert (row["last_name"], row["first_name"]) == ("Peña", "Niño")


def test_the_closest_name_comes_first_without_the_accent(client, headers, make_learner):
    near = make_learner(last_name="Peñaflor", first_name="Testa")
    exact = make_learner(last_name="Peña", first_name="Testa")

    assert found_ids(client, headers, q="Pena Testa") == [exact.id, near.id]


@pytest.mark.parametrize("q", ["％", "＿", "⁄"])
def test_characters_unaccent_turns_into_wildcards_stay_literal(client, headers, make_learner, q):
    make_learner(last_name="Santos")

    assert found_ids(client, headers, q=q) == []
