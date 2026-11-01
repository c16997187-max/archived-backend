from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, health, users
from app.api.errors import register_error_handlers
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
    app.state.settings = settings
    register_error_handlers(app)
    # Only for a frontend on another origin (the Vite dev server). In production
    # the API serves the frontend itself, so the list is empty and no CORS is sent.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,  # the refresh cookie
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(health.router, prefix="/api")
    app.include_router(auth.router, prefix="/api")
    app.include_router(users.router, prefix="/api")
    return app


app = create_app()
