"""Validated field types and pagination, shared by request models and query parameters.

Validators raise PydanticCustomError because a plain ValueError would reach the
UI as "Value error, Must be ...".
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

from annotated_types import Ge, Le, MaxLen
from pydantic import AfterValidator, BaseModel, BeforeValidator, WithJsonSchema
from pydantic_core import PydanticCustomError

from app.models.school import KINDERGARTEN, MAX_GRADE

MAX_TEXT_LENGTH = 200
MAX_SEARCH_LENGTH = 100
MAX_BIGINT = 2**63 - 1

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100
MAX_PAGE = 1_000_000

# [0-9], not \d: \d also matches digits from other scripts, which the database rejects.
_LRN = re.compile(r"[0-9]{12}")
_CONTACT_NUMBER = re.compile(r"09[0-9]{9}")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _blank_to_none(value: Any) -> Any:
    # A form's empty input arrives as "", and `?grade=` as an empty query value.
    return None if isinstance(value, str) and not value.strip() else value


def _clean_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if "\x00" in value:  # psycopg cannot send NUL to PostgreSQL
        raise PydanticCustomError("invalid_text", "Contains an invalid character")
    return " ".join(value.split())


def _required_text(value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PydanticCustomError("required", "Required")
    return _clean_text(value)


def _check_lrn(value: Any) -> str:
    if isinstance(value, str) and _LRN.fullmatch(value.strip()):
        return value.strip()
    raise PydanticCustomError("invalid_lrn", "Must be exactly 12 digits")


def _check_contact_number(value: Any) -> str:
    if isinstance(value, str) and _CONTACT_NUMBER.fullmatch(value.strip()):
        return value.strip()
    raise PydanticCustomError("invalid_contact_number", "Must be 11 digits starting with 09")


def _parse_grade(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip()
        if value.upper() == "K":
            return KINDERGARTEN
        if value.isascii() and value.isdigit() and len(value) <= 2:
            value = int(value)
    # bool is an int in Python, and True must not mean Grade 1.
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_GRADE:
        return value
    raise PydanticCustomError("invalid_grade", "Must be K or a grade from 0 to 12")


def _parse_date(value: Any) -> date:
    # Parsed here because pydantic would also read numbers as timestamps.
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_DATE.fullmatch(value.strip()):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            pass
    raise PydanticCustomError("invalid_date", "Must be a real date in YYYY-MM-DD format")


def _check_past(value: date) -> date:
    if value >= date.today():
        raise PydanticCustomError("date_not_past", "Must be in the past")
    return value


_Text = Annotated[str, BeforeValidator(_clean_text), MaxLen(MAX_TEXT_LENGTH)]

RequiredText = Annotated[str, BeforeValidator(_required_text), MaxLen(MAX_TEXT_LENGTH)]
OptionalText = Annotated[_Text | None, BeforeValidator(_blank_to_none)]

# Set here rather than with Query(max_length=...), which fails on a blank `?q=`.
SearchText = Annotated[
    Annotated[str, BeforeValidator(_clean_text), MaxLen(MAX_SEARCH_LENGTH)] | None,
    BeforeValidator(_blank_to_none),
]

_Lrn = Annotated[
    str,
    BeforeValidator(_check_lrn),
    WithJsonSchema({"type": "string", "pattern": "^[0-9]{12}$", "examples": ["000000000000"]}),
]
OptionalLrn = Annotated[_Lrn | None, BeforeValidator(_blank_to_none)]

_ContactNumber = Annotated[
    str,
    BeforeValidator(_check_contact_number),
    WithJsonSchema({"type": "string", "pattern": "^09[0-9]{9}$", "examples": ["09000000000"]}),
]
OptionalContactNumber = Annotated[_ContactNumber | None, BeforeValidator(_blank_to_none)]

# Stored as an int; Kindergarten is 0, and "K" is accepted as input for it.
GradeLevel = Annotated[
    int,
    BeforeValidator(_parse_grade),
    WithJsonSchema(
        {
            "type": "integer",
            "minimum": KINDERGARTEN,
            "maximum": MAX_GRADE,
            "description": '0 is Kindergarten. "K" is also accepted as input.',
        }
    ),
]
OptionalGradeLevel = Annotated[GradeLevel | None, BeforeValidator(_blank_to_none)]

IsoDate = Annotated[date, BeforeValidator(_parse_date)]
PastDate = Annotated[IsoDate, AfterValidator(_check_past)]
OptionalPastDate = Annotated[PastDate | None, BeforeValidator(_blank_to_none)]

# Keeps out IDs too large for a bigint column, which would otherwise cause a 500.
DbId = Annotated[int, Ge(1), Le(MAX_BIGINT)]
OptionalDbId = Annotated[DbId | None, BeforeValidator(_blank_to_none)]


@dataclass(frozen=True)
class PageRequest:
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class Page[T](BaseModel):
    items: list[T]
    page: int
    page_size: int
    total: int
