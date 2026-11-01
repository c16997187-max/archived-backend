from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


# Public on purpose: used by service monitors before anyone logs in.
# Phase 14 extends this with database, disk, worker and last-backup checks.
@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(status="ok")
