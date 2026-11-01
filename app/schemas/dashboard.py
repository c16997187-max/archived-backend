from pydantic import BaseModel, ConfigDict


class DashboardSchoolYear(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str


class DashboardSummary(BaseModel):
    # Null until an admin marks a school year as current; the counts are then 0.
    school_year: DashboardSchoolYear | None
    # Enrollments in the current school year, by status.
    drafts: int
    pending_review: int
    approved: int
    rejected: int
    total_learners: int  # across all school years
