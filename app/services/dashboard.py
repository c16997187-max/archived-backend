from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Enrollment, Learner, SchoolYear
from app.schemas.dashboard import DashboardSchoolYear, DashboardSummary


def get_summary(session: Session) -> DashboardSummary:
    year = session.scalar(select(SchoolYear).where(SchoolYear.is_current))
    counts: dict[str, int] = {}
    if year is not None:
        counts = dict(
            session.execute(
                select(Enrollment.status, func.count())
                .join(Enrollment.learner)
                .where(
                    Enrollment.school_year_id == year.id,
                    Enrollment.deleted_at.is_(None),
                    Learner.deleted_at.is_(None),
                )
                .group_by(Enrollment.status)
            ).all()
        )
    total_learners = session.scalar(
        select(func.count()).select_from(Learner).where(Learner.deleted_at.is_(None))
    )
    return DashboardSummary(
        school_year=DashboardSchoolYear.model_validate(year) if year else None,
        drafts=counts.get("draft", 0),
        pending_review=counts.get("pending_review", 0),
        approved=counts.get("approved", 0),
        rejected=counts.get("rejected", 0),
        total_learners=total_learners or 0,
    )
