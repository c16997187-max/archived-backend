"""Every shared field rule, with passing and failing cases."""

from datetime import date, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.common import (
    MAX_BIGINT,
    DbId,
    GradeLevel,
    IsoDate,
    OptionalContactNumber,
    OptionalDbId,
    OptionalGradeLevel,
    OptionalLrn,
    OptionalPastDate,
    OptionalText,
    RequiredText,
)


def valid(field_type, value):
    return TypeAdapter(field_type).validate_python(value)


def error_message(field_type, value):
    with pytest.raises(ValidationError) as exc:
        valid(field_type, value)
    return exc.value.errors()[0]["msg"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("000000000017", "000000000017"),
        (" 000000000017 ", "000000000017"),
        ("", None),
        ("  ", None),
        (None, None),
    ],
)
def test_lrn_accepts(value, expected):
    assert valid(OptionalLrn, value) == expected


@pytest.mark.parametrize(
    "value",
    ["12345", "0000000000170", "00000000001a", "0000 0000 0017", 123456789012, "١٢٣٤٥٦٧٨٩٠١٢"],
)
def test_lrn_refuses(value):
    assert error_message(OptionalLrn, value) == "Must be exactly 12 digits"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("09000000017", "09000000017"), (" 09000000017 ", "09000000017"), ("", None), (None, None)],
)
def test_contact_number_accepts(value, expected):
    assert valid(OptionalContactNumber, value) == expected


@pytest.mark.parametrize(
    "value",
    ["0900000001", "090000000170", "19000000017", "+639000000017", "0900-000-0017", 9000000017],
)
def test_contact_number_refuses(value):
    assert error_message(OptionalContactNumber, value) == "Must be 11 digits starting with 09"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (12, 12), ("8", 8), (" 8 ", 8), ("K", 0), ("k", 0)],
)
def test_grade_level_accepts(value, expected):
    assert valid(GradeLevel, value) == expected


@pytest.mark.parametrize(
    "value", [-1, 13, "13", "-1", "Grade 8", "", None, True, False, 8.5, "9" * 5000]
)
def test_grade_level_refuses(value):
    assert error_message(GradeLevel, value) == "Must be K or a grade from 0 to 12"


def test_optional_grade_level_treats_blank_as_none():
    assert valid(OptionalGradeLevel, "") is None


@pytest.mark.parametrize("value", ["2012-05-20", " 2012-05-20 ", date(2012, 5, 20)])
def test_date_accepts(value):
    assert valid(IsoDate, value) == date(2012, 5, 20)


@pytest.mark.parametrize(
    "value",
    ["2012-02-30", "2012-13-01", "20120520", "05/20/2012", "2012-W01-1", "tomorrow", 1336000000],
)
def test_date_refuses(value):
    assert error_message(IsoDate, value) == "Must be a real date in YYYY-MM-DD format"


def test_past_date_accepts_yesterday():
    yesterday = date.today() - timedelta(days=1)

    assert valid(OptionalPastDate, yesterday.isoformat()) == yesterday


@pytest.mark.parametrize("days", [0, 1])
def test_past_date_refuses_today_and_later(days):
    value = (date.today() + timedelta(days=days)).isoformat()

    assert error_message(OptionalPastDate, value) == "Must be in the past"


def test_text_is_trimmed_and_spaces_collapsed():
    assert valid(RequiredText, "  Dela   Cruz ") == "Dela Cruz"
    assert valid(OptionalText, "  Dela   Cruz ") == "Dela Cruz"
    assert valid(RequiredText, "Peña") == "Peña"


@pytest.mark.parametrize("value", ["", "  ", None])
def test_required_text_refuses_blank(value):
    assert error_message(RequiredText, value) == "Required"


@pytest.mark.parametrize("value", ["", "  ", None])
def test_optional_text_treats_blank_as_none(value):
    assert valid(OptionalText, value) is None


@pytest.mark.parametrize("field_type", [RequiredText, OptionalText])
def test_text_refuses_nul_characters(field_type):
    assert error_message(field_type, "bad\x00text") == "Contains an invalid character"


@pytest.mark.parametrize("field_type", [RequiredText, OptionalText])
def test_text_refuses_too_long_values(field_type):
    valid(field_type, "x" * 200)
    with pytest.raises(ValidationError):
        valid(field_type, "x" * 201)


@pytest.mark.parametrize("value", [1, MAX_BIGINT])
def test_db_id_accepts(value):
    assert valid(DbId, value) == value


@pytest.mark.parametrize("value", [0, -1, MAX_BIGINT + 1])
def test_db_id_refuses(value):
    with pytest.raises(ValidationError):
        valid(DbId, value)


def test_optional_db_id_treats_blank_as_none():
    assert valid(OptionalDbId, "") is None
