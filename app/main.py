from fastapi import FastAPI

from app.api import health
from app.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # /docs loads Swagger UI from a CDN, so it only works on a dev machine
    # with internet. Production (the offline host PC) exposes no API docs.
    docs_enabled = settings.app_env != "production"

    app = FastAPI(
        title="ArchivED API",
        version="0.1.0",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.include_router(health.router, prefix="/api")
    return app


app = create_app()
