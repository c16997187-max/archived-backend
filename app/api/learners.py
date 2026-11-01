from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import DbSession, Paging, client_ip, require_role
from app.models import Learner, User
from app.models.user import ROLES
from app.schemas.common import DbId, OptionalDbId, OptionalGradeLevel, Page, SearchText
from app.schemas.errors import error_responses
from app.schemas.learners import LearnerDetail, LearnerOut, LearnerSummary, LearnerUpdate
from app.services import learners as learners_service

router = APIRouter(prefix="/learners", tags=["learners"], responses=error_responses(401, 403))

AnyRole = Annotated[User, Depends(require_role(*ROLES))]


@router.get("", response_model=Page[LearnerSummary], responses=error_responses(422))
def search_learners(
    _: AnyRole,
    db: DbSession,
    paging: Paging,
    q: Annotated[SearchText, Query(description="Name (fuzzy), or the start of an LRN.")] = None,
    grade: Annotated[OptionalGradeLevel, Query()] = None,
    section: Annotated[OptionalDbId, Query(description="Section ID.")] = None,
    school_year: Annotated[OptionalDbId, Query(description="School year ID.")] = None,
) -> Page[LearnerSummary]:
    return learners_service.search_learners(db, paging, q, grade, section, school_year)


@router.get("/{learner_id}", response_model=LearnerDetail, responses=error_responses(404, 422))
def get_learner(learner_id: DbId, _: AnyRole, db: DbSession) -> LearnerDetail:
    return learners_service.get_learner(db, learner_id)


@router.patch("/{learner_id}", response_model=LearnerOut, responses=error_responses(404, 409, 422))
def update_learner(
    learner_id: DbId,
    body: LearnerUpdate,
    request: Request,
    user: Annotated[User, Depends(require_role("registrar", "admin"))],
    db: DbSession,
) -> Learner:
    return learners_service.update_learner(db, user, learner_id, body, client_ip(request))
