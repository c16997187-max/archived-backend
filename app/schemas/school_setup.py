import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic_core import PydanticCustomError

from app.schemas.common import DbId, GradeLevel, IsoDate, OptionalText, RequiredText

_LABEL = re.compile(r"([0-9]{4})-([0-9]{4})")


class SchoolYearCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: RequiredText = Field(examples=["2026-2027"])
    start_date: IsoDate
    end_date: IsoDate
    is_current: bool = Field(
        default=False, description="When true, the previous current school year stops being so."
    )

    @field_validator("label")
    @classmethod
    def _label_is_consecutive_years(cls, value: str) -> str:
        match = _LABEL.fullmatch(value)
        if match is None or int(match[2]) != int(match[1]) + 1:
            raise PydanticCustomError("invalid_label", "Must look like 2026-2027")
        return value

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, value: date, info: ValidationInfo) -> date:
        start = info.data.get("start_date")
        if start is not None and value <= start:
            raise PydanticCustomError("invalid_end_date", "Must be after the start date")
        return value


class SchoolYearOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str
    start_date: date
    end_date: date
    is_current: bool


class SectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    school_year_id: DbId
    grade_level: GradeLevel
    name: RequiredText
    adviser: OptionalText = None


class SectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    school_year_id: int
    grade_level: int  # 0 is Kindergarten
    name: str
    adviser: str | None
