"""Step 8: every normalizer and rule. All values are made up."""

from datetime import date

import pytest

from app.ocr.normalize import (
    Normalized,
    is_required,
    normalize_field,
    parse_contact_number,
    parse_date,
    parse_grade,
    parse_lrn,
    parse_name,
    parse_school_year,
)

TODAY = date(2026, 11, 7)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2015-03-14", Normalized("2015-03-14")),
        ("2015/3/4", Normalized("2015-03-04")),
        ("03/14/2015", Normalized("2015-03-14")),  # month first, as Philippine forms write it
        ("3-14-2015", Normalized("2015-03-14")),
        ("14/03/2015", Normalized("2015-03-14")),  # 14 cannot be a month
        ("03/04/2015", Normalized("2015-03-04", "ambiguous_date")),
        ("03/03/2015", Normalized("2015-03-03")),  # same either way
        ("03/14/15", Normalized("2015-03-14")),
        ("03/14/98", Normalized("1998-03-14")),
        ("March 14, 2015", Normalized("2015-03-14")),
        ("Mar. 14 2015", Normalized("2015-03-14")),
        ("14 March 2015", Normalized("2015-03-14")),
        ("Marso 14, 2015", Normalized("2015-03-14")),
        ("Setyembre 1, 2016", Normalized("2016-09-01")),
        ("  march   14 ,2015 ", Normalized("2015-03-14")),
    ],
)
def test_dates_become_iso(text, expected):
    assert parse_date(text, TODAY) == expected


@pytest.mark.parametrize("text", ["02/30/2015", "Smarch 14, 2015", "14", "soon", "2015-13-01"])
def test_impossible_dates_are_invalid(text):
    assert parse_date(text, TODAY).problem == "invalid_date"


def test_birth_date_must_be_in_the_past():
    assert parse_date("2099-01-01", TODAY) == Normalized("2099-01-01", "date_not_past")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("09171234567", Normalized("09171234567")),
        ("0917 123 4567", Normalized("09171234567")),
        ("0917-123-4567", Normalized("09171234567")),
        ("+63 917 123 4567", Normalized("09171234567")),
        ("63-917-123-4567", Normalized("09171234567")),
        ("(0917) 123-4567", Normalized("09171234567")),
        ("9171234567", Normalized("09171234567")),
        ("O917123456l", Normalized("09171234561", "digit_guessed")),
        ("0917123456", Normalized("0917123456", "invalid_contact_number")),
        ("hello", Normalized(None, "invalid_contact_number")),
    ],
)
def test_contact_numbers(text, expected):
    assert parse_contact_number(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("123456789012", Normalized("123456789012")),
        ("1234 5678 9012", Normalized("123456789012")),
        ("123-456-789-012", Normalized("123456789012")),
        ("12345678901O", Normalized("123456789010", "digit_guessed")),
        ("12345", Normalized("12345", "invalid_lrn")),
        ("none", Normalized(None, "invalid_lrn")),
    ],
)
def test_lrn_keeps_digits_only(text, expected):
    assert parse_lrn(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Gr. 8", "8"),
        ("Grade 8", "8"),
        ("grade8", "8"),
        ("G8", "8"),
        ("8", "8"),
        ("Grade VIII", "8"),
        ("Grade xii", "12"),
        ("Baitang 4", "4"),
        ("Kinder", "K"),
        ("K", "K"),
        ("Kindergarten", "K"),
    ],
)
def test_grade_levels(text, expected):
    assert parse_grade(text) == Normalized(expected)


@pytest.mark.parametrize("text", ["Grade 13", "Gr. eight", "college"])
def test_unknown_grades_are_invalid(text):
    assert parse_grade(text).problem == "invalid_grade"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-2027", Normalized("2026-2027")),
        ("SY 2026-27", Normalized("2026-2027")),
        ("S.Y. 2026 - 2027", Normalized("2026-2027")),
        ("2026/2027", Normalized("2026-2027")),
        ("2026-2028", Normalized("2026-2028", "invalid_label")),
        ("next year", Normalized(None, "invalid_label")),
    ],
)
def test_school_years(text, expected):
    assert parse_school_year(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  dela   CRUZ ", "Dela Cruz"),
        ("madeupname,", "Madeupname"),
        ("ana-marie", "Ana-Marie"),
    ],
)
def test_names_are_trimmed_and_title_cased(text, expected):
    assert parse_name(text) == Normalized(expected)


def test_fields_are_normalized_by_their_base_name():
    assert normalize_field("school_year[1].grade_level", "Gr. 7") == Normalized("7")
    assert normalize_field("father_name", " juan  testname ") == Normalized("Juan Testname")
    assert normalize_field("extension", "jr") == Normalized("Jr.")
    assert normalize_field("religion", "  Seventh-day   Adventist ") == Normalized(
        "Seventh-day Adventist"
    )


def test_a_lone_trailing_full_stop_is_dropped():
    assert normalize_field("municipality", "Sablan .") == Normalized("Sablan")
    assert normalize_field("middle_name", "torres .") == Normalized("Torres")
    assert normalize_field("house_street", "Km. 4") == Normalized("Km. 4")


def test_empty_reads_have_no_value_and_no_problem():
    assert normalize_field("lrn", "   ") == Normalized(None)


def test_overlong_values_are_flagged():
    assert normalize_field("religion", "x" * 300).problem == "too_long"


def test_required_fields_are_top_level_only():
    assert is_required("last_name")
    assert is_required("school_year")
    assert not is_required("school_year[0].school_year")
    assert not is_required("religion")
