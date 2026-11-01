from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import Enrollment, Learner
from app.schemas.common import (
    OptionalContactNumber,
    OptionalLrn,
    OptionalPastDate,
    OptionalText,
    RequiredText,
)


class _LearnerFields(BaseModel):
    # A mistyped field is refused, so an edit is never silently dropped.
    model_config = ConfigDict(extra="forbid")

    lrn: OptionalLrn = None
    middle_name: OptionalText = None
    extension: OptionalText = None
    sex: Literal["M", "F"] | None = None
    birth_date: OptionalPastDate = None
    place_of_birth: OptionalText = None
    mother_tongue: OptionalText = None
    religion: OptionalText = None
    ip_group: OptionalText = None
    house_street: OptionalText = None
    barangay: OptionalText = None
    municipality: OptionalText = None
    province: OptionalText = None
    father_name: OptionalText = None
    mother_name: OptionalText = None
    guardian_name: OptionalText = None
    guardian_relationship: OptionalText = None
    contact_number: OptionalContactNumber = None


class LearnerCreate(_LearnerFields):
    last_name: RequiredText
    first_name: RequiredText


class LearnerUpdate(_LearnerFields):
    """Only the fields sent are changed. The default is not validated, so omitting
    a name is fine but sending it as null is refused."""

    last_name: RequiredText = Field(default=None)
    first_name: RequiredText = Field(default=None)


class EnrollmentOut(BaseModel):
    id: int
    learner_id: int
    school_year_id: int
    school_year_label: str
    section_id: int | None
    section_name: str | None
    grade_level: int | None  # 0 is Kindergarten
    status: str
    enrolled_at: datetime | None
    remarks: str | None

    @classmethod
    def from_model(cls, enrollment: Enrollment) -> "EnrollmentOut":
        return cls(
            id=enrollment.id,
            learner_id=enrollment.learner_id,
            school_year_id=enrollment.school_year_id,
            school_year_label=enrollment.school_year.label,
            section_id=enrollment.section_id,
            section_name=enrollment.section.name if enrollment.section else None,
            grade_level=enrollment.grade_level,
            status=enrollment.status,
            enrolled_at=enrollment.enrolled_at,
            remarks=enrollment.remarks,
        )


class LearnerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    lrn: str | None
    last_name: str
    first_name: str
    middle_name: str | None
    extension: str | None
    sex: str | None
    birth_date: date | None
    place_of_birth: str | None
    mother_tongue: str | None
    religion: str | None
    ip_group: str | None
    house_street: str | None
    barangay: str | None
    municipality: str | None
    province: str | None
    father_name: str | None
    mother_name: str | None
    guardian_name: str | None
    guardian_relationship: str | None
    contact_number: str | None
    created_at: datetime
    updated_at: datetime


class LearnerDetail(LearnerOut):
    enrollments: list[EnrollmentOut]  # newest school year first


class LearnerSummary(BaseModel):
    """A search row. Leaves out address, religion and family details on purpose."""

    id: int
    lrn: str | None
    last_name: str
    first_name: str
    middle_name: str | None
    extension: str | None
    sex: str | None
    birth_date: date | None
    # The newest enrollment matching the search filters, if any.
    enrollment: EnrollmentOut | None

    @classmethod
    def from_model(cls, learner: Learner, enrollment: Enrollment | None) -> "LearnerSummary":
        return cls(
            id=learner.id,
            lrn=learner.lrn,
            last_name=learner.last_name,
            first_name=learner.first_name,
            middle_name=learner.middle_name,
            extension=learner.extension,
            sex=learner.sex,
            birth_date=learner.birth_date,
            enrollment=EnrollmentOut.from_model(enrollment) if enrollment else None,
        )


class LearnerMatch(LearnerSummary):
    """A likely duplicate, shown so the UI can ask "Is this the same student?"."""

    reasons: list[Literal["lrn", "name_birth_date"]]
