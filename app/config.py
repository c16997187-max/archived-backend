from functools import lru_cache
from pathlib import Path
from typing import Annotated, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Used only outside production, so a fresh dev checkout runs without a .env.
DEV_JWT_SECRET = "dev-insecure-change-me"
# Values that may be copied verbatim from .env.example or old defaults.
PLACEHOLDER_JWT_SECRETS = {DEV_JWT_SECRET, "change-me"}
MIN_JWT_SECRET_LENGTH = 32
# Vite's dev server, where Fermin runs the UI during development.
DEV_CORS_ORIGINS = ["http://localhost:5173"]


class Settings(BaseSettings):
    """Machine settings, read from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    # 127.0.0.1, not localhost: on Windows localhost tries ::1 first and can hang.
    database_url: str = "postgresql+psycopg://archived:archived@127.0.0.1:5432/archived"
    storage_dir: Path = Path("storage")
    models_dir: Path = Path("models")
    # Required in production (see _check_production_secrets); dev falls back to DEV_JWT_SECRET.
    jwt_secret: str | None = None
    access_token_minutes: int = 30
    # Refresh token lifetime without "Remember Me": about one school day.
    refresh_token_hours: int = 12
    remember_me_days: int = 30
    # Only true once the host PC serves HTTPS; browsers drop Secure cookies over HTTP.
    cookie_secure: bool = False
    # Comma-separated. None means the default for app_env: the Vite dev server in
    # development, nothing in production (the API serves the frontend itself).
    cors_origins: Annotated[list[str] | None, NoDecode] = None
    # Dev-only: admin password for scripts/seed.py. Unset means generate one.
    seed_admin_password: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _split_cors_origins(cls, data: dict) -> dict:
        value = data.get("cors_origins") if isinstance(data, dict) else None
        if isinstance(value, str):
            data["cors_origins"] = [o.strip() for o in value.split(",") if o.strip()]
        return data

    @model_validator(mode="after")
    def _check_production_secrets(self) -> Self:
        if self.app_env == "production":
            secret = self.jwt_secret or ""
            if secret in PLACEHOLDER_JWT_SECRETS or len(secret) < MIN_JWT_SECRET_LENGTH:
                raise ValueError(
                    "APP_ENV=production needs a JWT_SECRET of at least "
                    f"{MIN_JWT_SECRET_LENGTH} characters. Generate one with: "
                    'python -c "import secrets; print(secrets.token_urlsafe(48))"'
                )
        elif not self.jwt_secret:
            self.jwt_secret = DEV_JWT_SECRET
        if self.cors_origins is None:
            self.cors_origins = [] if self.app_env == "production" else list(DEV_CORS_ORIGINS)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
