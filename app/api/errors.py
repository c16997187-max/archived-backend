from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.errors import ApiError

# Codes for errors Starlette raises itself (unknown path, wrong method, ...).
HTTP_ERROR_CODES = {404: "not_found", 405: "method_not_allowed"}


def error_response(
    status_code: int, error: str, fields: dict[str, str] | None = None, headers=None
) -> JSONResponse:
    return JSONResponse(
        {"error": error, "fields": fields or {}}, status_code=status_code, headers=headers
    )


def _field_name(loc: tuple) -> str:
    # ("body", "id_number") -> "id_number"; the UI matches these to form fields.
    parts = [str(p) for p in loc if p not in ("body", "query", "path", "cookie", "header")]
    return ".".join(parts) or "request"


def register_error_handlers(app: FastAPI) -> None:
    """Give every error the one format the UI understands (CLAUDE.md conventions)."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.error, exc.fields, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields: dict[str, str] = {}
        for err in exc.errors():
            fields.setdefault(_field_name(tuple(err["loc"])), err["msg"])
        return error_response(422, "validation_error", fields)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        code = HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, headers=exc.headers)
