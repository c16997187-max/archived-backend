from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.deps import DbSession, client_ip, require_role
from app.models import User
from app.schemas.enrollments import EnrollmentCreate, EnrollmentCreated
from app.schemas.errors import DUPLICATE_LEARNER_RESPONSE, error_responses
from app.services import enrollments as enrollments_service

router = APIRouter(prefix="/enrollments", tags=["enrollments"], responses=error_responses(401, 403))


@router.post(
    "",
    response_model=EnrollmentCreated,
    status_code=201,
    responses={**error_responses(422), **DUPLICATE_LEARNER_RESPONSE},
)
def create_enrollment(
    body: EnrollmentCreate,
    request: Request,
    user: Annotated[User, Depends(require_role("registrar", "admin"))],
    db: DbSession,
) -> EnrollmentCreated:
    return enrollments_service.create_enrollment(db, user, body, client_ip(request))
