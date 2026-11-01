from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.user import ROLES
from app.schemas.common import IdNumber, RequiredText
from app.services.passwords import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    id_number: str
    full_name: str
    role: str
    must_change_password: bool


class UserAdminOut(UserOut):
    """A row on the admin's user screens."""

    is_active: bool
    last_login_at: datetime | None
    created_at: datetime


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_number: IdNumber
    full_name: RequiredText
    role: Literal[*ROLES]
    # The user must replace this at first login.
    temporary_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)


class ResetPasswordRequest(BaseModel):
    temporary_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
