from typing import Any


class ApiError(Exception):
    """An error the client should see, rendered as `{"error": code, "fields": {...}}`.

    Raised by services and dependencies; `app/api/errors.py` turns it into a response.
    """

    def __init__(
        self,
        status_code: int,
        error: str,
        fields: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(error)
        self.status_code = status_code
        self.error = error
        self.fields = fields or {}
        self.headers = headers
        # Added next to `error` and `fields`, e.g. the `matches` of a likely duplicate.
        self.extra = extra
