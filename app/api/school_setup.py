from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import DbSession, Paging, client_ip, require_role
from app.models import SchoolYear, Section, User
from app.models.user import ROLES
from app.schemas.common import OptionalDbId, OptionalGradeLevel, Page
from app.schemas.errors import error_responses
from app.schemas.school_setup import SchoolYearCreate, SchoolYearOut, SectionCreate, SectionOut
from app.services import school_setup as school_setup_service

router = APIRouter(tags=["school setup"], responses=error_responses(401, 403))

AnyRole = Annotated[User, Depends(require_role(*ROLES))]
Admin = Annotated[User, Depends(require_role("admin"))]


@router.get("/school-years", response_model=Page[SchoolYearOut], responses=error_responses(422))
def list_school_years(_: AnyRole, db: DbSession, paging: Paging) -> Page[SchoolYearOut]:
    return school_setup_service.list_school_years(db, paging)


@router.post(
    "/school-years",
    response_model=SchoolYearOut,
    status_code=201,
    responses=error_responses(409, 422),
)
def create_school_year(
    body: SchoolYearCreate, request: Request, admin: Admin, db: DbSession
) -> SchoolYear:
    return school_setup_service.create_school_year(db, admin, body, client_ip(request))


@router.get("/sections", response_model=Page[SectionOut], responses=error_responses(422))
def list_sections(
    _: AnyRole,
    db: DbSession,
    paging: Paging,
    school_year: Annotated[OptionalDbId, Query(description="School year ID.")] = None,
    grade: Annotated[OptionalGradeLevel, Query()] = None,
) -> Page[SectionOut]:
    return school_setup_service.list_sections(db, paging, school_year, grade)


@router.post(
    "/sections", response_model=SectionOut, status_code=201, responses=error_responses(409, 422)
)
def create_section(body: SectionCreate, request: Request, admin: Admin, db: DbSession) -> Section:
    return school_setup_service.create_section(db, admin, body, client_ip(request))
