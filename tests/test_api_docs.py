import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

DOC_PATHS = ["/docs", "/redoc", "/openapi.json"]


@pytest.mark.parametrize("path", DOC_PATHS)
def test_docs_enabled_in_development(path):
    client = TestClient(create_app(Settings(app_env="development")))

    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", DOC_PATHS)
def test_docs_disabled_in_production(path):
    client = TestClient(create_app(Settings(app_env="production")))

    assert client.get(path).status_code == 404


def test_health_still_served_in_production():
    client = TestClient(create_app(Settings(app_env="production")))

    assert client.get("/api/health").status_code == 200
