import logging
import traceback

import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError, StatementError
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import ApiError

logger = logging.getLogger(__name__)

# Codes for errors Starlette raises itself (unknown path, wrong method, ...).
HTTP_ERROR_CODES = {404: "not_found", 405: "method_not_allowed"}

# The messages of these quote the values they failed on, which can be student data.
_MESSAGE_MAY_HOLD_STUDENT_DATA = (
    SQLAlchemyError,
    psycopg.Error,
    ValidationError,
    ResponseValidationError,
)


def error_response(
    status_code: int, error: str, fields: dict[str, str] | None = None, headers=None, extra=None
) -> JSONResponse:
    return JSONResponse(
        {"error": error, "fields": fields or {}, **(extra or {})},
        status_code=status_code,
        headers=headers,
    )


def _field_name(loc: tuple) -> str:
    # ("body", "id_number") -> "id_number"; the UI matches these to form fields.
    parts = [str(p) for p in loc if p not in ("body", "query", "path", "cookie", "header")]
    return ".".join(parts) or "request"


def _safe_message(exc: BaseException) -> str:
    if not isinstance(exc, _MESSAGE_MAY_HOLD_STUDENT_DATA):
        return str(exc)
    # What still locates the failure: SQL with placeholders, the code and the constraint.
    if isinstance(exc, StatementError):
        return f"[message hidden] SQL: {exc.statement}"
    if isinstance(exc, psycopg.Error):
        return f"[message hidden] sqlstate={exc.sqlstate} constraint={exc.diag.constraint_name}"
    return "[message hidden]"


def _safe_traceback(exc: BaseException) -> str:
    """Every traceback in the exception chain, without any values that may be student data."""
    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        name = f"{type(current).__module__}.{type(current).__qualname__}"
        parts.append(
            "".join(traceback.format_tb(current.__traceback__))
            + f"{name}: {_safe_message(current)}"
        )
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return "\n\nCaused by:\n".join(parts)


class CatchAllErrorsMiddleware:
    """Turn an unexpected exception into a 500 in our format, with no details exposed.

    Not `@app.exception_handler(Exception)`: Starlette runs that outside CORSMiddleware,
    so the Vite dev UI could not read the response, and then re-raises for Uvicorn to log
    the raw exception, whose message can hold student data.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            response_started = response_started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:
            # The route's template, not the URL: paths hold capture tokens, queries hold names.
            route = getattr(scope.get("route"), "path", "<no route>")
            logger.error(
                "Unhandled error on %s %s\n%s", scope["method"], route, _safe_traceback(exc)
            )
            # Once started, the status is already sent; returning makes Uvicorn drop the
            # connection, so the client cannot take the cut-off response as complete.
            if not response_started:
                await error_response(500, "internal_error")(scope, receive, send)


def register_error_handlers(app: FastAPI) -> None:
    """Give every error the one format the UI understands (CLAUDE.md conventions)."""
    app.add_middleware(CatchAllErrorsMiddleware)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.error, exc.fields, exc.headers, exc.extra)

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
