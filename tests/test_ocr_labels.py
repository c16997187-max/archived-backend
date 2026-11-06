import numpy as np
import pytest

from app.ocr.labels import (
    MIN_TEXT_REGIONS,
    REASON_NOT_A_FORM,
    extract_fields,
    full_text,
    read_generic_page,
)
from app.ocr.preprocess import PreprocessFailure
from tests.ocr_images import StubReader, box

PAGE = (1700, 2200)


def _values(boxes) -> dict[str, str]:
    return {f.name: f.value for f in extract_fields(boxes, PAGE)}


@pytest.mark.parametrize(
    ("label", "field"),
    [
        ("Name", "full_name"),
        ("Pangalan", "full_name"),
        ("Name/Pangalan", "full_name"),
        ("Birthdate", "birth_date"),
        ("Petsa ng Kapanganakan", "birth_date"),
        ("Birthdate/Petsa ng Kapanganakan", "birth_date"),
        ("LRN", "lrn"),
        ("LRN:", "lrn"),
        ("Kasarian", "sex"),
        ("Pangalan ng Ama", "father_name"),
        ("Pangalan ng Ina", "mother_name"),
        ("Tirahan", "address"),
        # Misread by OCR, or in other forms of the word.
        ("Petsa ng Kapanganakn", "birth_date"),
        ("Birthdaie", "birth_date"),
        ("PANGALÁN", "full_name"),
        ("Father's Name", "father_name"),
    ],
)
def test_labels_match_in_both_languages(label: str, field: str):
    boxes = [box(label, 100, 300), box("Madeup Value", 400, 300)]
    assert _values(boxes) == {field: "Madeup Value"}


def test_pangalan_ng_ama_is_never_taken_for_pangalan():
    boxes = [box("Pangalan ng Ama", 100, 300), box("Testfather", 400, 300)]
    assert "full_name" not in _values(boxes)


@pytest.mark.parametrize("near_miss", ["LRM", "Sexy", "Se", "1RN"])
def test_short_labels_need_an_exact_match(near_miss: str):
    assert _values([box(near_miss, 100, 300), box("Value", 400, 300)]) == {}


def test_label_and_value_in_one_box():
    assert _values([box("LRN: 123456789012", 100, 300)]) == {"lrn": "123456789012"}
    assert _values([box("LRN 123456789012", 100, 300)]) == {"lrn": "123456789012"}
    assert _values([box("Birthdate: 01/02/2010", 100, 300)]) == {"birth_date": "01/02/2010"}


def test_a_value_starting_like_a_label_stays_a_value():
    boxes = [box("Address", 100, 300), box("School of Testing", 400, 300)]
    assert _values(boxes) == {"address": "School of Testing"}


def test_value_to_the_right_joins_boxes_and_stops_at_the_next_label():
    boxes = [
        box("Name", 100, 300, 100),
        box("Testfirst", 300, 300, 150),
        box("Testlast", 470, 300, 150),
        box("Sex", 800, 300, 80),
        box("F", 950, 300, 30),
    ]
    assert _values(boxes) == {"full_name": "Testfirst Testlast", "sex": "F"}


def test_value_below_when_nothing_is_to_the_right():
    boxes = [box("Address", 100, 300), box("12 Fake Street", 100, 350)]
    assert _values(boxes) == {"address": "12 Fake Street"}


def test_right_is_preferred_over_below():
    boxes = [box("Sex", 100, 300, 80), box("M", 250, 300, 30), box("Below", 100, 350)]
    assert _values(boxes)["sex"] == "M"


def test_box_used_on_the_right_is_not_also_used_below():
    boxes = [
        box("Religion", 100, 300, 300),
        box("Address", 100, 345, 150),
        box("12 Fake Street", 300, 345, 300),
    ]
    assert _values(boxes) == {"religion": "", "address": "12 Fake Street"}


def test_label_with_nothing_beside_it_is_saved_empty():
    fields = extract_fields([box("LRN", 100, 300)], PAGE)
    assert [(f.name, f.value, f.confidence, f.bbox) for f in fields] == [("lrn", "", None, None)]


def test_next_label_directly_below_is_not_taken_as_the_value():
    assert _values([box("LRN", 100, 300), box("Sex", 100, 350)]) == {"lrn": "", "sex": ""}


def test_far_away_text_is_not_a_value():
    assert _values([box("LRN", 100, 300), box("Far", 1500, 300)]) == {"lrn": ""}
    assert _values([box("LRN", 100, 300), box("Far", 100, 800)]) == {"lrn": ""}


def test_label_printed_twice_keeps_the_copy_with_a_value():
    boxes = [box("Name", 100, 300), box("Pangalan", 400, 300), box("Testname", 700, 300)]
    assert _values(boxes) == {"full_name": "Testname"}


def test_confidence_is_the_lowest_of_the_value_boxes_and_bbox_covers_them():
    boxes = [
        box("Name", 100, 300, 100),
        box("Test", 300, 300, 100, confidence=0.9),
        box("Value", 420, 305, 100, confidence=0.4),
    ]
    field = extract_fields(boxes, PAGE)[0]
    assert field.confidence == 0.4
    assert field.bbox == {"x": 300, "y": 300, "w": 220, "h": 45}


def _year_block(top: int, school: str, year: str, average: str) -> list:
    """An SF10-style block, with the school printed before the school year."""
    return [
        box("School", 100, top, 120),
        box(school, 300, top, 300),
        box("School Year", 100, top + 60, 200),
        box(year, 350, top + 60, 200),
        box("General Average", 100, top + 120, 250),
        box(average, 400, top + 120, 100),
    ]


def test_school_before_school_year_groups_into_the_right_blocks():
    boxes = [*_year_block(500, "Test School A", "2020-2021", "85"),
             *_year_block(900, "Test School B", "2021-2022", "90")]  # fmt: skip
    assert _values(boxes) == {
        "school_year[0].school_name": "Test School A",
        "school_year[0].school_year": "2020-2021",
        "school_year[0].general_average": "85",
        "school_year[1].school_name": "Test School B",
        "school_year[1].school_year": "2021-2022",
        "school_year[1].general_average": "90",
    }


def test_label_order_inside_a_block_does_not_change_the_grouping():
    # The second block prints its rows in a different order from the first.
    first = _year_block(500, "Test School A", "2020-2021", "85")
    second = [
        box("General Average", 100, 900, 250),
        box("90", 400, 900, 100),
        box("School Year", 100, 960, 200),
        box("2021-2022", 350, 960, 200),
        box("School", 100, 1020, 120),
        box("Test School B", 300, 1020, 300),
    ]
    values = _values([*first, *second])
    assert values["school_year[1].school_name"] == "Test School B"
    assert values["school_year[1].general_average"] == "90"
    assert values["school_year[0].general_average"] == "85"


def test_tightly_stacked_blocks_split_where_a_field_repeats():
    boxes = [*_year_block(500, "Test School A", "2020-2021", "85"),
             *_year_block(690, "Test School B", "2021-2022", "90")]  # fmt: skip
    values = _values(boxes)
    assert values["school_year[0].school_name"] == "Test School A"
    assert values["school_year[1].school_name"] == "Test School B"
    assert values["school_year[1].general_average"] == "90"


def test_blocks_side_by_side_share_a_band():
    boxes = [
        box("School", 100, 500, 120),
        box("Test School A", 300, 500, 300),
        box("School", 900, 500, 120),
        box("Test School B", 1100, 500, 300),
        box("School Year", 100, 560, 200),
        box("2020-2021", 350, 560, 200),
        box("School Year", 900, 560, 200),
        box("2021-2022", 1150, 560, 200),
    ]
    assert _values(boxes) == {
        "school_year[0].school_name": "Test School A",
        "school_year[0].school_year": "2020-2021",
        "school_year[1].school_name": "Test School B",
        "school_year[1].school_year": "2021-2022",
    }


def test_identity_and_per_year_fields_are_kept_apart():
    boxes = [box("LRN", 100, 300), box("123456789012", 400, 300),
             *_year_block(500, "Test School A", "2020-2021", "85")]  # fmt: skip
    values = _values(boxes)
    assert values["lrn"] == "123456789012"
    assert values["school_year[0].school_year"] == "2020-2021"


def test_full_text_follows_reading_order():
    boxes = [box("second", 400, 303), box("first", 100, 300), box("third", 100, 400)]
    assert full_text(boxes) == "first second\nthird"


# Enough labelled text for a form: five lines of labels and values.
FORM_TEXT = [
    box("LRN", 100, 300),
    box("123456789012345678", 400, 300),
    box("Name", 100, 400),
    box("Madeupname", 400, 400),
    box("Address", 100, 500),
]


def test_page_with_too_few_text_regions_is_not_a_form():
    page = np.zeros((2200, 1700), dtype=np.uint8)
    reader = StubReader(FORM_TEXT[: MIN_TEXT_REGIONS - 1])
    with pytest.raises(PreprocessFailure) as failure:
        read_generic_page(page, reader)
    assert failure.value.reason == REASON_NOT_A_FORM
    assert not reader.recognized  # decided from the detector alone


def test_page_with_tiny_specks_of_text_is_not_a_form():
    specks = [box("x", 100 * i, 300, w=4, h=4) for i in range(1, 12)]
    with pytest.raises(PreprocessFailure):
        read_generic_page(np.zeros((2200, 1700), dtype=np.uint8), StubReader(specks))


def test_empty_page_is_not_a_form():
    with pytest.raises(PreprocessFailure):
        read_generic_page(np.zeros((2200, 1700), dtype=np.uint8), StubReader([]))


def test_generic_page_holds_text_fields_and_model_version():
    reader = StubReader(FORM_TEXT)
    page = read_generic_page(np.zeros((2200, 1700), dtype=np.uint8), reader)
    assert page.full_text.startswith("LRN 123456789012345678\n")
    assert page.fields[0].name == "lrn"
    assert page.model_version == reader.model_version
