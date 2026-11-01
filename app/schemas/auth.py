from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.users import UserOut
from app.services.passwords import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH


class LoginRequest(BaseModel):
    id_number: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    # Extends the refresh cookie to REMEMBER_ME_DAYS; the access token stays short.
    remember_me: bool = False


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int  # seconds until access_token expires
    # When true the UI must show the change-password screen; other routes return 403.
    must_change_password: bool
    user: UserOut


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    new_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
