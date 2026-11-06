"""Step 8 of the OCR pipeline: turn what was read into the form the database expects, and
say when it fails the same rules the API applies (Phase 7, rules from Phase 8).

Pure functions on strings. A problem is an error code, never the value, so it is safe to
log and to copy into audit rows.
"""

import re
from dataclasses import dataclass
from datetime import date

from pydantic import TypeAdapter, ValidationError

from app.schemas.common import (
    MAX_TEXT_LENGTH,
    GradeLevel,
    OptionalContactNumber,
    OptionalLrn,
    PastDate,
)

# Phase 8. Checked only where the template has the field, and only outside repeating
# sections. "Guardian or parent" is checked across fields in `missing_guardian`.
REQUIRED_FIELDS = frozenset(
    {"last_name", "first_name", "birth_date", "sex", "grade_level", "school_year"}
)
GUARDIAN_FIELDS = ("guardian_name", "father_name", "mother_name")

NAME_FIELDS = frozenset(
    {"last_name", "first_name", "middle_name", "full_name", "father_name", "mother_name",
     "guardian_name"}
)  # fmt: skip
DATE_FIELDS = frozenset({"birth_date"})

# Misreads of digits in fields that hold only digits. A guess still flags the field.
_DIGIT_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1"})

MONTHS = {
    "january": 1, "jan": 1, "enero": 1,
    "february": 2, "feb": 2, "pebrero": 2, "peb": 2,
    "march": 3, "mar": 3, "marso": 3,
    "april": 4, "apr": 4, "abril": 4, "abr": 4,
    "may": 5, "mayo": 5,
    "june": 6, "jun": 6, "hunyo": 6, "hun": 6,
    "july": 7, "jul": 7, "hulyo": 7, "hul": 7,
    "august": 8, "aug": 8, "agosto": 8, "ago": 8,
    "september": 9, "sep": 9, "sept": 9, "setyembre": 9, "set": 9,
    "october": 10, "oct": 10, "oktubre": 10, "okt": 10,
    "november": 11, "nov": 11, "nobyembre": 11, "nob": 11,
    "december": 12, "dec": 12, "disyembre": 12, "dis": 12,
}  # fmt: skip

ROMAN_GRADES = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
    "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
}  # fmt: skip
NAME_EXTENSIONS = {"jr": "Jr.", "sr": "Sr.", **{k: k.upper() for k in ("ii", "iii", "iv", "v")}}

_ISO = re.compile(r"([0-9]{4})[-/. ]+([0-9]{1,2})[-/. ]+([0-9]{1,2})")
_NUMERIC = re.compile(r"([0-9]{1,2})[-/. ]+([0-9]{1,2})[-/. ]+([0-9]{2}|[0-9]{4})")
_MONTH_FIRST = re.compile(r"([a-z]+)\.? *([0-9]{1,2}) *,? *([0-9]{4})")
_DAY_FIRST = re.compile(r"([0-9]{1,2}) *([a-z]+)\.? *,? *([0-9]{4})")
_SCHOOL_YEAR = re.compile(r"(?:s\.?y\.?)?\s*([0-9]{4})\s*[-/–]\s*([0-9]{2}|[0-9]{4})")
_GRADE = re.compile(
    r"(?:grade|gr|g|baitang)?\.?\s*([0-9]{1,2}|xii|xi|ix|x|viii|vii|vi|iv|v|iii|ii|i)"
)

_TRAILING_MARK = re.compile(r"\s+[.,;:]+\s*$")

_PAST_DATE = TypeAdapter(PastDate)
_LRN = TypeAdapter(OptionalLrn)
_CONTACT = TypeAdapter(OptionalContactNumber)
_GRADE_LEVEL = TypeAdapter(GradeLevel)


@dataclass(frozen=True)
class Normalized:
    """`value` is None when nothing usable was read. `problem` is an error code."""

    value: str | None
    problem: str | None = None


def base_name(field_name: str) -> str:
    """ "school_year[1].grade_level" is a grade_level."""
    return field_name.rsplit(".", 1)[-1]


def is_required(field_name: str) -> bool:
    return "." not in field_name and field_name in REQUIRED_FIELDS


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _code(error: ValidationError) -> str:
    return error.errors()[0]["type"]


def _two_digit_year(year: int, today: date) -> int:
    """A birth year written "15": the latest century that does not put it in the future."""
    century = today.year // 100 * 100
    return century + year if century + year <= today.year else century - 100 + year


def parse_date(text: str, today: date | None = None) -> Normalized:
    """Philippine forms write dates month first (03/14/2015). When both numbers could be
    the month, the date is read month first and flagged as ambiguous."""
    today = today or date.today()
    cleaned = _collapse(text.casefold().replace(",", ", "))
    problem = None
    if match := _ISO.fullmatch(cleaned):
        year, month, day = (int(g) for g in match.groups())
    elif match := _NUMERIC.fullmatch(cleaned):
        first, second, year = (int(g) for g in match.groups())
        if len(match[3]) == 2:
            year = _two_digit_year(year, today)
        if first > 12 >= second:
            day, month = first, second
        else:
            month, day = first, second
            if second <= 12 and first != second:
                problem = "ambiguous_date"
    elif (match := _MONTH_FIRST.fullmatch(cleaned)) and match[1] in MONTHS:
        month, day, year = MONTHS[match[1]], int(match[2]), int(match[3])
    elif (match := _DAY_FIRST.fullmatch(cleaned)) and match[2] in MONTHS:
        day, month, year = int(match[1]), MONTHS[match[2]], int(match[3])
    else:
        return Normalized(None, "invalid_date")
    try:
        value = date(year, month, day)
    except ValueError:
        return Normalized(None, "invalid_date")
    iso = value.isoformat()
    try:
        _PAST_DATE.validate_python(iso)
    except ValidationError as exc:
        return Normalized(iso, _code(exc))
    return Normalized(iso, problem)


def _digits(text: str) -> tuple[str, bool]:
    """Digits only, and whether a letter had to be read as a digit. Letters are only taken
    for digits in text that is mostly digits already, so a word stays a word."""
    plain = re.sub(r"[^0-9]", "", text)
    letters = sum(c.isalpha() for c in text)
    if len(plain) < 2 * letters:
        return plain, False
    guessed = re.sub(r"[^0-9]", "", text.translate(_DIGIT_LOOKALIKES))
    return guessed, guessed != plain


def parse_lrn(text: str) -> Normalized:
    digits, guessed = _digits(text)
    if not digits:
        return Normalized(None, "invalid_lrn")
    try:
        _LRN.validate_python(digits)
    except ValidationError as exc:
        return Normalized(digits, _code(exc))
    return Normalized(digits, "digit_guessed" if guessed else None)


def parse_contact_number(text: str) -> Normalized:
    """+63 917 000 0000, 63-917-000-0000 and 9170000000 all become 09170000000."""
    plus = text.strip().startswith("+")
    digits, guessed = _digits(text)
    if digits.startswith("63") and (plus or len(digits) == 12):
        digits = "0" + digits[2:]
    elif len(digits) == 10 and digits.startswith("9"):
        digits = "0" + digits
    if not digits:
        return Normalized(None, "invalid_contact_number")
    try:
        _CONTACT.validate_python(digits)
    except ValidationError as exc:
        return Normalized(digits, _code(exc))
    return Normalized(digits, "digit_guessed" if guessed else None)


def parse_grade(text: str) -> Normalized:
    """ "Gr. 8", "Grade 8", "G8", "Grade VIII" become "8"; "Kinder" becomes "K"."""
    cleaned = _collapse(text.casefold())
    if re.fullmatch(r"k|kinder|kindergarten|kinder garten", cleaned):
        return Normalized("K")
    match = _GRADE.fullmatch(cleaned)
    if match is None:
        return Normalized(None, "invalid_grade")
    number = match[1]
    grade = int(number) if number.isdigit() else ROMAN_GRADES[number]
    try:
        _GRADE_LEVEL.validate_python(grade)
    except ValidationError as exc:
        return Normalized(str(grade), _code(exc))
    return Normalized("K" if grade == 0 else str(grade))


def parse_school_year(text: str) -> Normalized:
    """ "SY 2026-27" becomes "2026-2027"."""
    match = _SCHOOL_YEAR.fullmatch(_collapse(text.casefold()))
    if match is None:
        return Normalized(None, "invalid_label")
    start = int(match[1])
    end = int(match[2]) if len(match[2]) == 4 else start // 100 * 100 + int(match[2])
    if end != start + 1:
        return Normalized(f"{start}-{end}", "invalid_label")
    return Normalized(f"{start}-{end}")


def parse_name(text: str) -> Normalized:
    """Trimmed and title-cased: "dela CRUZ" becomes "Dela Cruz"."""
    cleaned = _collapse(text).strip(" ,.;:")
    return Normalized(cleaned.title() or None)


def parse_extension(text: str) -> Normalized:
    cleaned = _collapse(text).strip(" ,.").casefold()
    return Normalized(NAME_EXTENSIONS.get(cleaned, cleaned.title()) or None)


def normalize_field(field_name: str, text: str) -> Normalized:
    """The normalized value and any problem. An empty read is None with no problem here:
    whether empty is allowed is the caller's call, through `is_required`."""
    # A lone full stop or comma at the end is a recognizer habit (TrOCR learned from
    # sentences), not part of a form value.
    text = _TRAILING_MARK.sub("", text)
    if not text.strip():
        return Normalized(None)
    name = base_name(field_name)
    if name in DATE_FIELDS:
        result = parse_date(text)
    elif name == "lrn":
        result = parse_lrn(text)
    elif name == "contact_number":
        result = parse_contact_number(text)
    elif name == "grade_level":
        result = parse_grade(text)
    elif name == "school_year":
        result = parse_school_year(text)
    elif name == "extension":
        result = parse_extension(text)
    elif name in NAME_FIELDS:
        result = parse_name(text)
    else:
        result = Normalized(_collapse(text) or None)
    if result.value is not None and len(result.value) > MAX_TEXT_LENGTH:
        return Normalized(result.value[:MAX_TEXT_LENGTH], "too_long")
    return result
