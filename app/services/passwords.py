from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.errors import ApiError

MIN_PASSWORD_LENGTH = 8
# Upper bound so a huge request body cannot tie up the CPU in Argon2.
MAX_PASSWORD_LENGTH = 256

_hasher = PasswordHash((Argon2Hasher(),))


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _hasher.verify(password, password_hash)


def check_new_password(password: str, field: str) -> None:
    """Raise a validation error naming `field` if the password is too short or long."""
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise ApiError(
            422,
            "validation_error",
            {field: f"Must be {MIN_PASSWORD_LENGTH} to {MAX_PASSWORD_LENGTH} characters"},
        )
