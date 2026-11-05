"""The layout of one paper form, as stored in `form_templates.field_boxes` (Phase 7).

Edit the JSON in `docs/forms/`, then run `python -m scripts.load_form_template`. Table
regions for full grades can be added later as a new optional key.
"""

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Names become debug file names and CSV columns, so no path characters.
Name = Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]
Kind = Literal["handwritten", "printed", "checkbox"]


class TemplateError(ValueError):
    """The template JSON is wrong. The message says what to fix."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldBox(_Strict):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    kind: Kind

    def moved(self, dx: int, dy: int) -> "FieldBox":
        return self.model_copy(update={"x": self.x + dx, "y": self.y + dy})


class Page(_Strict):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SectionInstance(_Strict):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class RepeatingSection(_Strict):
    """A block that appears several times, such as one school year on an SF10.
    `fields` are relative to each instance's top-left corner."""

    name: Name
    instances: list[SectionInstance] = Field(min_length=1)
    fields: dict[Name, FieldBox] = Field(min_length=1)


class FormLayout(_Strict):
    format_version: Literal[1]
    name: Name
    version: int = Field(gt=0)
    # A stand-in layout that must not reach the host PC.
    provisional: bool = False
    image: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")  # file name inside form_templates_dir
    page: Page
    fields: dict[Name, FieldBox]
    repeating_sections: list[RepeatingSection] = []

    @model_validator(mode="after")
    def _check_boxes(self) -> Self:
        boxes = self.boxes()
        if not boxes:
            raise ValueError("The template has no fields")
        for name, box in boxes.items():
            if box.x + box.w > self.page.width or box.y + box.h > self.page.height:
                raise ValueError(
                    f"Field '{name}' is outside the {self.page.width}x{self.page.height} page"
                )
        return self

    def boxes(self) -> dict[str, FieldBox]:
        """Every field with page coordinates; sections expand to `name[0].field`."""
        boxes = dict(self.fields)
        for section in self.repeating_sections:
            for i, instance in enumerate(section.instances):
                for name, box in section.fields.items():
                    boxes[f"{section.name}[{i}].{name}"] = box.moved(instance.x, instance.y)
        return boxes


def parse_layout(data: dict[str, Any]) -> FormLayout:
    try:
        return FormLayout.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'template'}: {error['msg']}"
            for error in exc.errors()
        )
        raise TemplateError(problems) from None


def read_layout_file(path: Path) -> FormLayout:
    """Read and validate a template JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TemplateError(f"Cannot read {path}: {exc.strerror}") from None
    except ValueError as exc:
        raise TemplateError(f"{path} is not valid JSON: {exc}") from None
    return parse_layout(data)


def load_layout_file(path: Path) -> FormLayout:
    """Read a template JSON and check that its blank form image sits next to it."""
    layout = read_layout_file(path)
    if not (path.parent / layout.image).is_file():
        raise TemplateError(f"Blank form image '{layout.image}' is not next to {path.name}")
    return layout
