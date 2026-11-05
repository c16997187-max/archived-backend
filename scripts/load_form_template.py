"""Load a template JSON into `form_templates`. Rerun it after editing the boxes.

    python -m scripts.load_form_template docs/forms/enrollment_v1.json

The same (name, version) updates in place. Check the boxes first by running the blank form
through `python -m scripts.ocr_debug` and opening 05_overlay.png. Provisional templates are
refused when APP_ENV=production.
"""

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal
from app.errors import ApiError
from app.ocr.templates import TemplateError, load_layout_file
from app.services.form_templates import save_template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a form template into the database.")
    parser.add_argument("template", type=Path, help="Path to the template JSON")
    args = parser.parse_args(argv)

    try:
        layout = load_layout_file(args.template)
    except TemplateError as exc:
        print(exc, file=sys.stderr)
        return 1

    with SessionLocal() as session:
        try:
            row, created = save_template(session, layout, get_settings().app_env)
        except ApiError as exc:
            for field, message in exc.fields.items():
                print(f"{field}: {message}", file=sys.stderr)
            return 1
        verb = "Created" if created else "Updated"
        print(f"{verb} template '{row.name}' v{row.version} ({len(layout.boxes())} fields).")
    if layout.provisional:
        print("This layout is provisional: it will not match real photos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
