from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.deps import DbSession, Paging, client_ip, require_role
from app.models import User
from app.models.user import ROLES
from app.schemas.common import DbId, Page
from app.schemas.errors import error_responses
from app.schemas.users import ResetPasswordRequest, UserAdminOut, UserCreate, UserOut
from app.services import users as users_service

router = APIRouter(tags=["users"], responses=error_responses(401, 403))


# Allowed during a forced password change, so the UI can still show who is logged in.
@router.get("/me", response_model=UserOut)
def get_me(
    user: Annotated[User, Depends(require_role(*ROLES, allow_password_change=True))],
) -> User:
    return user


@router.get("/users", response_model=Page[UserAdminOut], responses=error_responses(422))
def list_users(
    _: Annotated[User, Depends(require_role("admin"))], db: DbSession, paging: Paging
) -> Page[UserAdminOut]:
    return users_service.list_users(db, paging)


@router.post(
    "/users", response_model=UserAdminOut, status_code=201, responses=error_responses(409, 422)
)
def create_user(
    body: UserCreate,
    request: Request,
    admin: Annotated[User, Depends(require_role("admin"))],
    db: DbSession,
) -> User:
    return users_service.create_user(
        db,
        body.id_number,
        body.full_name,
        body.role,
        body.temporary_password,
        actor_id=admin.id,
        ip=client_ip(request),
        must_change_password=True,
    )


@router.post(
    "/users/{user_id}/reset-password", status_code=204, responses=error_responses(404, 422)
)
def reset_password(
    user_id: DbId,
    body: ResetPasswordRequest,
    request: Request,
    admin: Annotated[User, Depends(require_role("admin"))],
    db: DbSession,
) -> None:
    users_service.reset_password(db, admin, user_id, body.temporary_password, client_ip(request))
