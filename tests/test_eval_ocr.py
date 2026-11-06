import json
from pathlib import Path

import pytest

from scripts import eval_ocr
from scripts.eval_ocr import EvalError, edit_distance, evaluate, format_report, normalize


def _csv(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


@pytest.fixture
def labels(tmp_path: Path) -> Path:
    return _csv(tmp_path / "labels.csv", "document,last_name,lrn\nd1,Santos,1234\nd2,Reyes,5678\n")


def _score(report, field):
    return next(s for s in report.fields if s.field == field)


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("", "", 0), ("abc", "abc", 0), ("abc", "", 3), ("", "abc", 3), ("Santos", "Santoz", 1),
     ("kitten", "sitting", 3)],
)  # fmt: skip
def test_edit_distance(a: str, b: str, distance: int):
    assert edit_distance(a, b) == distance


def test_cer_and_exact_match_are_hand_computed(tmp_path: Path, labels: Path):
    predictions = _csv(
        tmp_path / "run.csv", "document,last_name,lrn\nd1,Santoz,1234\nd2,Reyes,56\n"
    )

    report = evaluate(labels, predictions)

    # 1 error in "Santos" + "Reyes" (11 characters); one of two exactly right.
    assert _score(report, "last_name").cer == pytest.approx(1 / 11)
    assert _score(report, "last_name").exact_match_rate == 0.5
    # 2 errors in 8 characters.
    assert _score(report, "lrn").cer == pytest.approx(2 / 8)
    assert _score(report, "lrn").exact_match_rate == 0.5
    # Summed, not an average of the two rates above.
    assert report.overall.cer == pytest.approx(3 / 19)
    assert report.overall.exact_match_rate == 0.5
    assert report.overall.documents == 4


def test_cer_can_pass_one_when_ocr_adds_text(tmp_path: Path):
    labels = _csv(tmp_path / "l.csv", "document,sex\nd1,M\n")
    predictions = _csv(tmp_path / "p.csv", "document,sex\nd1,Male\n")
    assert evaluate(labels, predictions).overall.cer == pytest.approx(3)


def test_excel_bom_and_filipino_letters(tmp_path: Path):
    labels = _csv(tmp_path / "l.csv", "document,last_name\nd1,Peña\n", "utf-8-sig")
    # The same letter, as n + combining tilde.
    predictions = _csv(tmp_path / "p.csv", "document,last_name\nd1,Peña\n")

    report = evaluate(labels, predictions)

    assert report.overall.cer == 0
    assert report.overall.exact_match_rate == 1


def test_non_utf8_file_gets_a_clear_error(tmp_path: Path, labels: Path):
    bad = _csv(tmp_path / "p.csv", "document,last_name,lrn\nd1,Peña,1\n", "cp1252")
    with pytest.raises(EvalError, match="CSV UTF-8"):
        evaluate(labels, bad)


def test_whitespace_is_not_an_error_but_case_is():
    assert normalize("  Dela   Cruz\t") == "Dela Cruz"
    assert normalize("dela cruz") != normalize("Dela Cruz")


def test_empty_labels_report_no_cer_and_empty_predictions_match(tmp_path: Path):
    labels = _csv(tmp_path / "l.csv", "document,extension\nd1,\nd2,\n")
    predictions = _csv(tmp_path / "p.csv", "document,extension\nd1,\nd2,Jr\n")

    report = evaluate(labels, predictions)

    assert report.overall.cer is None
    assert report.overall.exact_match_rate == 0.5
    assert "n/a" in format_report(report)


def test_missing_document_counts_as_empty(tmp_path: Path, labels: Path):
    predictions = _csv(tmp_path / "p.csv", "document,last_name,lrn\nd1,Santos,1234\n")

    report = evaluate(labels, predictions)

    assert report.missing_documents == 1
    assert _score(report, "lrn").cer == pytest.approx(4 / 8)
    assert _score(report, "lrn").exact_match_rate == 0.5
    assert "1 document(s) had no predictions" in format_report(report)


def test_extra_document_is_ignored_with_a_warning(tmp_path: Path, labels: Path):
    predictions = _csv(
        tmp_path / "p.csv",
        "document,last_name,lrn\nd1,Santos,1234\nd2,Reyes,5678\nd3,Cruz,9999\n",
    )

    report = evaluate(labels, predictions)

    assert report.extra_documents == 1
    assert report.overall.cer == 0
    assert report.overall.documents == 4
    assert "Warning" in format_report(report)


def test_label_field_missing_from_predictions_is_an_error(tmp_path: Path, labels: Path):
    predictions = _csv(tmp_path / "p.csv", "document,last_name\nd1,Santos\n")
    with pytest.raises(EvalError, match="lrn"):
        evaluate(labels, predictions)


def test_section_field_names_work_as_columns(tmp_path: Path):
    text = "document,school_year[1].general_average\nd1,90\n"
    report = evaluate(_csv(tmp_path / "l.csv", text), _csv(tmp_path / "p.csv", text))
    assert report.fields[0].field == "school_year[1].general_average"


@pytest.mark.parametrize(
    ("text", "message"),
    [("last_name\nSantos\n", "document"), ("document,lrn\n", "no rows"),
     ("document,lrn\nd1,1\nd1,2\n", "repeated")],
)  # fmt: skip
def test_bad_labels_files_are_refused(tmp_path: Path, text: str, message: str):
    with pytest.raises(EvalError, match=message):
        evaluate(_csv(tmp_path / "l.csv", text), _csv(tmp_path / "p.csv", "document,lrn\nd1,1\n"))


def test_output_never_holds_values(tmp_path: Path, labels: Path, capsys):
    predictions = _csv(tmp_path / "p.csv", "document,last_name,lrn\nd1,Santoz,1234\nd2,Reyez,56\n")
    out = tmp_path / "report.json"

    code = eval_ocr.main(
        ["--labels", str(labels), "--predictions", str(predictions), "--json", str(out)]
    )

    assert code == 0
    captured = capsys.readouterr()
    written = out.read_text(encoding="utf-8")
    assert json.loads(written)["overall"]["documents"] == 4
    for value in ("Santos", "Santoz", "Reyes", "Reyez", "1234", "5678"):
        assert value not in captured.out
        assert value not in written


def test_main_reports_errors_on_stderr(tmp_path: Path, labels: Path, capsys):
    code = eval_ocr.main(["--labels", str(labels), "--predictions", str(tmp_path / "none.csv")])
    assert code == 1
    assert "Cannot read" in capsys.readouterr().err
