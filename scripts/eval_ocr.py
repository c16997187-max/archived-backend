"""Score OCR output against typed ground truth (Phase 7).

    python -m scripts.eval_ocr --labels private_data/labels.csv --predictions private_data/run1.csv

Both files have a `document` column and one column per field. Reports, per field and overall,
the character error rate (edit distance / reference characters, summed over documents, so
short fields do not skew it) and the share of values read exactly right.

The CSVs hold real students' data: keep them in private_data/, which git ignores. The report
holds numbers only, never values, so it is safe to paste into the paper.
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

DOCUMENT_COLUMN = "document"
OVERALL = "OVERALL"


class EvalError(ValueError):
    """The input files cannot be compared. The message says why."""


@dataclass(frozen=True)
class FieldScore:
    field: str
    documents: int
    cer: float | None  # None when the labels hold no characters to measure against
    exact_match_rate: float


@dataclass(frozen=True)
class Report:
    fields: list[FieldScore]
    overall: FieldScore
    missing_documents: int
    extra_documents: int


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def normalize(text: str) -> str:
    """Differences that are not reading errors: Unicode form and stray whitespace. Case stays,
    since a wrongly cased name is a real difference."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def read_table(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """(field columns, rows by document). utf-8-sig reads Excel's "CSV UTF-8" and plain UTF-8."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = list(reader.fieldnames or [])
            if DOCUMENT_COLUMN not in columns:
                raise EvalError(f"{path} has no '{DOCUMENT_COLUMN}' column")
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                document = row[DOCUMENT_COLUMN]
                if document in rows:
                    # The row's position, not its content: IDs may be built from names.
                    raise EvalError(f"{path} has a repeated document on line {reader.line_num}")
                rows[document] = row
    except UnicodeDecodeError:
        raise EvalError(f'{path} is not UTF-8. In Excel, save it as "CSV UTF-8".') from None
    except OSError as exc:
        raise EvalError(f"Cannot read {path}: {exc.strerror}") from None
    return [c for c in columns if c != DOCUMENT_COLUMN], rows


def _score(field: str, pairs: list[tuple[str, str]]) -> FieldScore:
    reference_chars = sum(len(label) for label, _ in pairs)
    errors = sum(edit_distance(label, prediction) for label, prediction in pairs)
    exact = sum(label == prediction for label, prediction in pairs)
    return FieldScore(
        field,
        len(pairs),
        errors / reference_chars if reference_chars else None,
        exact / len(pairs) if pairs else 0.0,
    )


def evaluate(labels_path: Path, predictions_path: Path) -> Report:
    fields, labels = read_table(labels_path)
    predicted_fields, predictions = read_table(predictions_path)
    missing_columns = [f for f in fields if f not in predicted_fields]
    if missing_columns:
        raise EvalError(f"{predictions_path} has no column for: {', '.join(missing_columns)}")
    if not labels:
        raise EvalError(f"{labels_path} has no rows")

    by_field: dict[str, list[tuple[str, str]]] = {field: [] for field in fields}
    for document, label_row in labels.items():
        # A form OCR never produced counts as all empty, not as left out of the score.
        predicted_row = predictions.get(document, {})
        for field in fields:
            by_field[field].append(
                (normalize(label_row[field] or ""), normalize(predicted_row.get(field) or ""))
            )

    return Report(
        [_score(field, pairs) for field, pairs in by_field.items()],
        _score(OVERALL, [pair for pairs in by_field.values() for pair in pairs]),
        sum(document not in predictions for document in labels),
        sum(document not in labels for document in predictions),
    )


def format_report(report: Report) -> str:
    width = max(len(score.field) for score in [*report.fields, report.overall])
    lines = [f"{'field':<{width}}  {'docs':>5}  {'CER':>7}  {'exact':>7}"]
    for score in [*report.fields, report.overall]:
        cer = "n/a" if score.cer is None else f"{score.cer:.2%}"
        exact = f"{score.exact_match_rate:.2%}"
        lines.append(f"{score.field:<{width}}  {score.documents:>5}  {cer:>7}  {exact:>7}")
    if report.missing_documents:
        lines.append(f"{report.missing_documents} document(s) had no predictions: scored as empty.")
    if report.extra_documents:
        lines.append(f"Warning: ignored {report.extra_documents} document(s) with no labels.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score OCR output against a labels CSV.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="Also write the report to this file")
    args = parser.parse_args(argv)

    try:
        report = evaluate(args.labels, args.predictions)
    except EvalError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
