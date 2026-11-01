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
    ) -> None:
        super().__init__(error)
        self.status_code = status_code
        self.error = error
        self.fields = fields or {}
        self.headers = headers
