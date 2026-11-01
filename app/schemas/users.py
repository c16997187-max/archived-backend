from pydantic import BaseModel, ConfigDict, Field

from app.services.passwords import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    id_number: str
    full_name: str
    role: str
    must_change_password: bool


class ResetPasswordRequest(BaseModel):
    temporary_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
