from typing import Any

from pydantic import BaseModel

from app.schemas.learners import LearnerMatch


class ErrorResponse(BaseModel):
    """The one error format. `fields` maps a form field to its message."""

    error: str
    fields: dict[str, str]


class DuplicateLearnerError(ErrorResponse):
    """`POST /api/enrollments` when the learner may already exist."""

    matches: list[LearnerMatch] = []  # absent on `already_enrolled`


# status -> (description, example). Examples hold no student data.
_ERROR_DOCS: dict[int, tuple[str, dict[str, Any]]] = {
    401: (
        "Not logged in, or the token or credentials are invalid",
        {"error": "not_authenticated", "fields": {}},
    ),
    403: (
        "Role not allowed, account disabled, or a password change is required first",
        {"error": "forbidden", "fields": {}},
    ),
    404: ("Record not found", {"error": "not_found", "fields": {}}),
    409: (
        "Conflicts with an existing record",
        {"error": "duplicate", "fields": {"lrn": "Already used by another learner"}},
    ),
    422: (
        "Invalid input; `fields` has one message per form field",
        {
            "error": "validation_error",
            "fields": {"contact_number": "Must be 11 digits starting with 09"},
        },
    ),
    423: (
        "Account locked after too many failed logins",
        {"error": "account_locked", "fields": {"locked_until": "2026-11-01T08:15:00+00:00"}},
    ),
}


def error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI `responses` for the given statuses, in our error format.

    Declaring 422 also stops FastAPI documenting its default validation error.
    """
    return {
        code: {
            "model": ErrorResponse,
            "description": _ERROR_DOCS[code][0],
            "content": {"application/json": {"example": _ERROR_DOCS[code][1]}},
        }
        for code in codes
    }


DUPLICATE_LEARNER_RESPONSE: dict[int | str, dict[str, Any]] = {
    409: {
        "model": DuplicateLearnerError,
        "description": (
            "`possible_duplicate`: the learner may already exist, see `matches`. "
            "`already_enrolled`: the learner has an enrollment for this school year "
            "(`matches` is absent)."
        ),
    }
}
