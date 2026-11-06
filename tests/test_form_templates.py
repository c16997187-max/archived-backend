import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import select

from app.errors import ApiError
from app.models import AuditLog, FormTemplate
from app.ocr.templates import TemplateError, load_layout_file, parse_layout
from app.services import form_templates
from scripts import load_form_template, make_placeholder_form
from tests.ocr_images import enrollment_layout, sf10_layout

SHIPPED_TEMPLATE = Path("docs/forms/enrollment_v1.json")
VALID = {
    "format_version": 1,
    "name": "test_form",
    "version": 1,
    "image": "test_form.png",
    "page": {"width": 1000, "height": 1000},
    "fields": {"last_name": {"x": 10, "y": 10, "w": 100, "h": 50, "kind": "handwritten"}},
}


def _with(**changes) -> dict:
    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


@pytest.mark.parametrize(
    ("data", "named"),
    [
        (_with(fields={"lrn": {"x": 950, "y": 0, "w": 100, "h": 50, "kind": "printed"}}), "lrn"),
        (_with(fields={"lrn": {"x": 0, "y": 0, "w": 100, "h": 50, "kind": "typed"}}), "kind"),
        (_with(fields={"lrn": {"x": 0, "y": 0, "w": 0, "h": 50, "kind": "printed"}}), "w"),
        (_with(fields={"../evil": {"x": 0, "y": 0, "w": 9, "h": 9, "kind": "printed"}}), "evil"),
        (_with(fields={}), "no fields"),
        (_with(image="../secrets.png"), "image"),
        (_with(format_version=2), "format_version"),
        (_with(unknown_key=1), "unknown_key"),
    ],
)
def test_bad_templates_are_refused_with_a_useful_message(data: dict, named: str):
    with pytest.raises(TemplateError) as error:
        parse_layout(data)
    assert named in str(error.value)


def test_section_box_outside_the_page_names_the_instance():
    data = _with(
        repeating_sections=[
            {
                "name": "school_year",
                "instances": [{"x": 0, "y": 0}, {"x": 0, "y": 960}],
                "fields": {"grade": {"x": 0, "y": 0, "w": 9, "h": 50, "kind": "handwritten"}},
            }
        ]
    )
    with pytest.raises(TemplateError, match=r"school_year\[1\]\.grade"):
        parse_layout(data)


def test_sections_expand_to_page_coordinates():
    boxes = sf10_layout().boxes()
    assert boxes["school_year[1].grade_level"].x == 900
    assert boxes["school_year[1].grade_level"].y == 1380
    assert boxes["school_year[0].grade_level"].y == 680
    assert "last_name" in boxes


def test_missing_image_is_refused(tmp_path: Path):
    path = tmp_path / "form.json"
    path.write_text(json.dumps(VALID), encoding="utf-8")
    with pytest.raises(TemplateError, match="test_form.png"):
        load_layout_file(path)


def test_broken_json_is_refused(tmp_path: Path):
    path = tmp_path / "form.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TemplateError, match="not valid JSON"):
        load_layout_file(path)


def test_shipped_template_is_valid_and_marked_provisional():
    layout = load_layout_file(SHIPPED_TEMPLATE)
    assert layout.provisional
    assert "last_name" in layout.boxes()


def test_shipped_placeholder_image_is_up_to_date(tmp_path: Path):
    copied = tmp_path / SHIPPED_TEMPLATE.name
    copied.write_text(SHIPPED_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

    assert make_placeholder_form.main([str(copied)]) == 0

    name = load_layout_file(copied).image
    fresh = cv2.imread(str(tmp_path / name), cv2.IMREAD_GRAYSCALE)
    shipped = cv2.imread(str(SHIPPED_TEMPLATE.parent / name), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(fresh, shipped)  # pixels: PNG bytes may differ between platforms


def test_save_creates_then_updates(db):
    layout = enrollment_layout("save_test")

    row, created = form_templates.save_template(db, layout, "development")
    assert created
    assert row.is_active
    assert row.field_boxes["fields"]["lrn"]["kind"] == "handwritten"

    data = layout.model_dump()
    data["fields"]["lrn"]["x"] = 111
    again, created_again = form_templates.save_template(db, parse_layout(data), "development")

    assert not created_again
    assert again.id == row.id
    assert again.field_boxes["fields"]["lrn"]["x"] == 111
    actions = db.scalars(
        select(AuditLog.action)
        .where(AuditLog.entity == "form_template", AuditLog.entity_id == row.id)
        .order_by(AuditLog.id)
    ).all()
    assert actions == ["create", "update"]


def test_update_does_not_reactivate_a_retired_template(db):
    layout = enrollment_layout("retired_test")
    row, _ = form_templates.save_template(db, layout, "development")
    row.is_active = False
    db.flush()

    updated, _ = form_templates.save_template(db, layout, "development")

    assert updated.is_active is False


def test_provisional_template_is_refused_in_production(db):
    data = enrollment_layout("prod_test").model_dump()
    data["provisional"] = True
    with pytest.raises(ApiError):
        form_templates.save_template(db, parse_layout(data), "production")
    assert db.scalar(select(FormTemplate).where(FormTemplate.name == "prod_test")) is None


def test_invalid_stored_template_is_skipped(db, tmp_path: Path):
    good, _ = form_templates.save_template(db, enrollment_layout("good_test"), "development")
    db.add(FormTemplate(name="bad_test", version=1, field_boxes={"nonsense": True}))
    db.flush()

    candidates = form_templates.active_candidates(db, tmp_path)

    assert good.id in [c.id for c in candidates]
    assert all(c.layout.name != "bad_test" for c in candidates)


def test_loader_script_reports_a_bad_file(tmp_path: Path, capsys):
    path = tmp_path / "form.json"
    path.write_text(json.dumps(_with(fields={})), encoding="utf-8")

    assert load_form_template.main([str(path)]) == 1
    assert "no fields" in capsys.readouterr().err
