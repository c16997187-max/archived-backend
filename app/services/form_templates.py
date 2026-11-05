"""Load form layouts into `form_templates` and read back the active ones (Phase 7)."""

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import ApiError
from app.models import FormTemplate
from app.ocr.preprocess import TemplateCandidate
from app.ocr.templates import FormLayout, TemplateError, parse_layout
from app.services import audit

logger = logging.getLogger(__name__)


def _summary(row: FormTemplate, layout: FormLayout) -> dict[str, object]:
    return {
        "name": row.name,
        "version": row.version,
        "field_count": len(layout.boxes()),
        "is_active": row.is_active,
    }


def save_template(
    session: Session, layout: FormLayout, app_env: str, is_active: bool = True
) -> tuple[FormTemplate, bool]:
    """Create or update the row for (name, version). Returns (row, was it created). An
    update keeps `is_active`, so editing boxes never reactivates a retired template."""
    if layout.provisional and app_env == "production":
        raise ApiError(
            422,
            "validation_error",
            {"template": "This is a provisional layout, and cannot be used in production."},
        )
    row = session.scalar(
        select(FormTemplate)
        .where(FormTemplate.name == layout.name, FormTemplate.version == layout.version)
        .with_for_update()
    )
    created = row is None
    before = None
    if created:
        row = FormTemplate(name=layout.name, version=layout.version, is_active=is_active)
        session.add(row)
    else:
        before = _summary(row, parse_layout(row.field_boxes))
    row.field_boxes = layout.model_dump()
    session.flush()
    audit.record(
        session,
        "create" if created else "update",
        "form_template",
        row.id,
        None,
        before=before,
        after=_summary(row, layout),
    )
    session.commit()
    return row, created


def active_candidates(session: Session, templates_dir: Path) -> list[TemplateCandidate]:
    """Every active template. A row that no longer validates is skipped, not fatal: one bad
    template must not stop photos from matching the others."""
    rows = session.scalars(
        select(FormTemplate).where(FormTemplate.is_active).order_by(FormTemplate.id)
    )
    candidates = []
    for row in rows:
        try:
            layout = parse_layout(row.field_boxes)
        except TemplateError:
            logger.warning("Skipping form template %s: its layout is not valid", row.id)
            continue
        candidates.append(TemplateCandidate(row.id, layout, templates_dir / layout.image))
    return candidates
