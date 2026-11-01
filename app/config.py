from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Machine settings, read from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    # 127.0.0.1, not localhost: on Windows localhost tries ::1 first and can hang.
    database_url: str = "postgresql+psycopg://archived:archived@127.0.0.1:5432/archived"
    storage_dir: Path = Path("storage")
    models_dir: Path = Path("models")
    # Dev-only default; production must set JWT_SECRET in .env.
    jwt_secret: str = "dev-insecure-change-me"
    # Dev-only: admin password for scripts/seed.py. Unset means generate one.
    seed_admin_password: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
