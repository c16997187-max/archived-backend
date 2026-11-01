from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.common import DbId, OptionalDbId, OptionalGradeLevel, OptionalText
from app.schemas.learners import EnrollmentOut, LearnerCreate, LearnerOut


class EnrollmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    school_year_id: DbId
    section_id: OptionalDbId = None
    grade_level: OptionalGradeLevel = Field(
        default=None,
        description="Filled in from the section when omitted; must match it when both are sent.",
    )
    remarks: OptionalText = None
    learner: LearnerCreate | None = Field(default=None, description="Details of a new learner.")
    learner_id: OptionalDbId = Field(default=None, description="Enroll an existing learner.")
    confirm_new_learner: bool = Field(
        default=False,
        description=(
            "Create the learner even though one has the same name and birth date. "
            "Has no effect on a matching LRN, which must be linked with learner_id instead."
        ),
    )

    @model_validator(mode="after")
    def _one_learner_source(self) -> "EnrollmentCreate":
        if (self.learner is None) == (self.learner_id is None):
            raise PydanticCustomError("learner_source", "Send either learner or learner_id")
        return self


class EnrollmentCreated(BaseModel):
    learner: LearnerOut
    enrollment: EnrollmentOut
