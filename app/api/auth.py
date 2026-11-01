from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Request, Response

from app.api.deps import AppSettings, DbSession, client_ip, require_role
from app.config import Settings
from app.models import User
from app.models.user import ROLES
from app.schemas.auth import ChangePasswordRequest, LoginRequest, TokenResponse
from app.schemas.errors import error_responses
from app.schemas.users import UserOut
from app.services import auth as auth_service
from app.services.auth import IssuedSession

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"
# Sent only to the auth endpoints, never to the rest of the API.
REFRESH_COOKIE_PATH = "/api/auth"
RefreshCookie = Annotated[str | None, Cookie(alias=REFRESH_COOKIE)]

# login, refresh and logout are public on purpose: they are how a user gets or
# drops a token. Logout is identified by the refresh cookie, so it still works
# after the access token has expired.


def _set_refresh_cookie(response: Response, issued: IssuedSession, settings: Settings) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        issued.refresh_token,
        # Without "Remember Me" the cookie dies with the browser.
        max_age=settings.remember_me_days * 86400 if issued.persistent else None,
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
    )


def _token_response(issued: IssuedSession, settings: Settings) -> TokenResponse:
    return TokenResponse(
        access_token=issued.access_token,
        expires_in=settings.access_token_minutes * 60,
        must_change_password=issued.user.must_change_password,
        user=UserOut.model_validate(issued.user),
    )


@router.post("/login", response_model=TokenResponse, responses=error_responses(401, 403, 422, 423))
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: DbSession,
    settings: AppSettings,
) -> TokenResponse:
    issued = auth_service.login(
        db, settings, body.id_number.strip(), body.password, body.remember_me, client_ip(request)
    )
    _set_refresh_cookie(response, issued, settings)
    return _token_response(issued, settings)


@router.post("/refresh", response_model=TokenResponse, responses=error_responses(401, 422))
def refresh(
    request: Request,
    response: Response,
    db: DbSession,
    settings: AppSettings,
    refresh_token: RefreshCookie = None,
) -> TokenResponse:
    issued = auth_service.refresh(db, settings, refresh_token, client_ip(request))
    _set_refresh_cookie(response, issued, settings)
    return _token_response(issued, settings)


@router.post("/logout", status_code=204, responses=error_responses(422))
def logout(
    request: Request,
    response: Response,
    db: DbSession,
    settings: AppSettings,
    refresh_token: RefreshCookie = None,
) -> None:
    auth_service.logout(db, refresh_token, client_ip(request))
    response.delete_cookie(
        REFRESH_COOKIE,
        path=REFRESH_COOKIE_PATH,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
    )


@router.post(
    "/change-password", response_model=TokenResponse, responses=error_responses(401, 403, 422)
)
def change_password(
    body: ChangePasswordRequest,
    request: Request,
    response: Response,
    user: Annotated[User, Depends(require_role(*ROLES, allow_password_change=True))],
    db: DbSession,
    settings: AppSettings,
    refresh_token: RefreshCookie = None,
) -> TokenResponse:
    issued = auth_service.change_password(
        db,
        settings,
        user,
        body.current_password,
        body.new_password,
        refresh_token,
        client_ip(request),
    )
    _set_refresh_cookie(response, issued, settings)
    return _token_response(issued, settings)
