"""The real models on generated forms. Skipped when models/ is not installed, as in CI;
run `python -m scripts.download_models` once to enable. All values are made up."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import get_settings
from app.ocr import labels
from app.ocr.model_files import ModelsMissing, model_entry
from app.ocr.normalize import normalize_field
from app.ocr.preprocess import TEMPLATE_PATH, TemplateCandidate, clean_page, preprocess
from app.ocr.recognize import recognize_crops
from app.ocr.templates import load_layout_file
from scripts.bench_ocr import TEMPLATE, make_form
from scripts.make_placeholder_form import render_form
from tests.ocr_images import unrelated_layout

MODELS_DIR = get_settings().models_dir


def _installed(name: str) -> bool:
    try:
        model_entry(MODELS_DIR, name, verify=False)
    except ModelsMissing:
        return False
    return True


pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        not (_installed("ppocrv6_small_det") and _installed("ppocrv6_small_rec")),
        reason="OCR models not installed in models/ (python -m scripts.download_models)",
    ),
]


@pytest.fixture(scope="module", params=["ppocr", "trocr"])
def engine(request):
    from app.ocr.models import load_engine

    if request.param == "trocr" and not _installed("trocr_base_handwritten_onnx"):
        pytest.skip("TrOCR not installed in models/")
    return load_engine(MODELS_DIR, "small", request.param)


def test_generated_form_is_read_end_to_end(engine, tmp_path: Path):
    layout = load_layout_file(TEMPLATE)
    candidate = TemplateCandidate(None, layout, TEMPLATE.parent / layout.image)
    image, written = make_form(layout, np.random.default_rng(3))
    path = tmp_path / "form.png"
    cv2.imwrite(str(path), image)

    prepared = preprocess(path, None, [candidate])
    results = {r.name: r for r in recognize_crops(prepared.crops, engine, 0.8)}

    assert prepared.ocr_path == TEMPLATE_PATH
    assert all(r.confidence is None or 0 <= r.confidence <= 1 for r in results.values())
    ticked = [n for n in ("sex_male", "sex_female") if results[n].raw == "checked"]
    assert len(ticked) == 1
    assert results["sex"].normalized == {"sex_male": "M", "sex_female": "F"}[ticked[0]]
    filled = [name for name, text in written.items() if text]
    exact = sum(
        results[name].normalized == normalize_field(name, written[name]).value for name in filled
    )
    # A sanity floor that wiring mistakes (wrong cache, wrong charset, garbled crops) fall
    # far below. Accuracy itself is measured with scripts/bench_ocr.py and the Step 0 set.
    assert exact >= len(filled) // 4, (engine.handwriting.model_version, exact, len(filled))
    # Empty optional fields stay empty and unflagged.
    assert results["ip_group"].raw is None


def test_generic_page_labels_are_found(engine):
    page = clean_page(cv2.cvtColor(render_form(unrelated_layout(), "unrelated memo"),
                                   cv2.COLOR_GRAY2BGR))  # fmt: skip
    read = labels.read_generic_page(page, engine.text_reader)
    assert "UNRELATED MEMO" in read.full_text.upper()


def test_blank_page_is_not_a_form(engine):
    page = np.full((2200, 1700), 235, np.uint8)
    with pytest.raises(labels.PreprocessFailure):
        labels.read_generic_page(page, engine.text_reader)
