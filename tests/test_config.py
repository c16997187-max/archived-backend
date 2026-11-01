import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import DEV_JWT_SECRET, Settings
from app.main import create_app

STRONG_SECRET = "s" * 48
DEV_ORIGIN = "http://localhost:5173"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("APP_ENV", "JWT_SECRET", "CORS_ORIGINS"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("secret", [None, "", "change-me", DEV_JWT_SECRET, "too-short"])
def test_production_refuses_missing_or_weak_jwt_secret(secret):
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None, app_env="production", jwt_secret=secret)


def test_production_refuses_to_start_when_jwt_secret_unset_in_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")

    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None)


def test_production_accepts_strong_jwt_secret():
    settings = Settings(_env_file=None, app_env="production", jwt_secret=STRONG_SECRET)

    assert settings.jwt_secret == STRONG_SECRET


def test_development_falls_back_to_dev_secret():
    assert Settings(_env_file=None).jwt_secret == DEV_JWT_SECRET


def test_cors_defaults_to_vite_in_development_and_nothing_in_production():
    assert Settings(_env_file=None).cors_origins == [DEV_ORIGIN]
    prod = Settings(_env_file=None, app_env="production", jwt_secret=STRONG_SECRET)
    assert prod.cors_origins == []


def test_cors_origins_read_as_comma_separated_list(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test:5173, http://b.test")

    assert Settings(_env_file=None).cors_origins == ["http://a.test:5173", "http://b.test"]


def test_dev_preflight_allows_credentials_from_vite_origin():
    client = TestClient(create_app(Settings(_env_file=None)))

    response = client.options(
        "/api/auth/login",
        headers={"Origin": DEV_ORIGIN, "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_production_sends_no_cors_headers():
    settings = Settings(_env_file=None, app_env="production", jwt_secret=STRONG_SECRET)
    client = TestClient(create_app(settings))

    response = client.get("/api/health", headers={"Origin": DEV_ORIGIN})

    assert "access-control-allow-origin" not in response.headers
