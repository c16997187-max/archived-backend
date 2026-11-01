from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.deps import DbSession, client_ip, require_role
from app.models import User
from app.models.user import ROLES
from app.schemas.users import ResetPasswordRequest, UserOut
from app.services import users as users_service

router = APIRouter(tags=["users"])


# Allowed during a forced password change, so the UI can still show who is logged in.
@router.get("/me", response_model=UserOut)
def get_me(
    user: Annotated[User, Depends(require_role(*ROLES, allow_password_change=True))],
) -> User:
    return user


@router.post("/users/{user_id}/reset-password", status_code=204)
def reset_password(
    user_id: int,
    body: ResetPasswordRequest,
    request: Request,
    admin: Annotated[User, Depends(require_role("admin"))],
    db: DbSession,
) -> None:
    users_service.reset_password(db, admin, user_id, body.temporary_password, client_ip(request))
