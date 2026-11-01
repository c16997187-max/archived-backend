from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import DbSession, require_role
from app.models import User
from app.models.user import ROLES
from app.schemas.dashboard import DashboardSummary
from app.schemas.errors import error_responses
from app.services import dashboard as dashboard_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"], responses=error_responses(401, 403))


@router.get("/summary", response_model=DashboardSummary)
def get_summary(
    _: Annotated[User, Depends(require_role(*ROLES))], db: DbSession
) -> DashboardSummary:
    return dashboard_service.get_summary(db)
