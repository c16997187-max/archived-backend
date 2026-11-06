"""The template-free path: find printed labels on a page and take the text beside them.

Works on text boxes from a `TextReader`, so it needs no model of its own. Every value from
here is flagged for review. Nothing here logs what was read.
"""

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from app.ocr.preprocess import PreprocessFailure
from app.ocr.text_reader import Region, TextBox, TextReader

# A page where the detector finds fewer text regions than this, or text covering less of the
# page than this share, is not a form. Checked before recognition, so a photo of a wall costs
# one detector call. A generated blank form has 26 regions covering about 1.8%.
MIN_TEXT_REGIONS = 5
MIN_TEXT_AREA_RATIO = 0.002
REASON_NOT_A_FORM = "This does not look like a form: almost no text was found on the page."

SECTION_NAME = "school_year"

# English and Filipino wording of each label, written in plain lowercase ASCII. Check these
# against real forms. Add new wordings here; the matching needs no other change.
IDENTITY_LABELS: dict[str, tuple[str, ...]] = {
    "full_name": ("name", "full name", "learner name", "name of learner", "pangalan",
                  "buong pangalan", "pangalan ng mag aaral"),
    "last_name": ("last name", "surname", "family name", "apelyido"),
    "first_name": ("first name", "given name", "unang pangalan"),
    "middle_name": ("middle name", "gitnang pangalan"),
    "lrn": ("lrn", "learner reference number", "learner reference no"),
    "sex": ("sex", "gender", "kasarian"),
    "birth_date": ("birthdate", "birth date", "date of birth", "petsa ng kapanganakan",
                   "kapanganakan"),
    "place_of_birth": ("place of birth", "birthplace", "lugar ng kapanganakan"),
    "mother_tongue": ("mother tongue", "katutubong wika", "unang wika"),
    "religion": ("religion", "relihiyon"),
    "address": ("address", "home address", "tirahan"),
    "father_name": ("father", "father s name", "name of father", "pangalan ng ama", "ama"),
    "mother_name": ("mother", "mother s name", "name of mother", "pangalan ng ina", "ina"),
    "guardian_name": ("guardian", "guardian s name", "name of guardian",
                      "pangalan ng tagapag alaga", "tagapag alaga"),
    "contact_number": ("contact number", "contact no", "numero ng telepono", "telepono"),
}  # fmt: skip

# Each school year on an SF10 or SF9 repeats these.
PER_YEAR_LABELS: dict[str, tuple[str, ...]] = {
    "school_year": ("school year", "sy", "taon ng pag aaral", "taong panuruan"),
    "grade_level": ("grade level", "grade", "baitang", "antas"),
    "school_name": ("school", "school name", "name of school", "paaralan",
                    "pangalan ng paaralan"),
    "general_average": ("general average", "pangkalahatang marka", "pangkalahatang average"),
    "final_status": ("final status", "action taken", "remarks", "katayuan"),
}  # fmt: skip

# Below this length a one-letter misread makes a different word, so only exact matches count.
EXACT_ONLY_LENGTH = 4
FUZZY_THRESHOLD = 0.82
# Same line: the boxes share at least this much of the shorter one's height.
SAME_LINE_OVERLAP = 0.5
# In a column: the boxes share at least this much of the narrower one's width.
SAME_COLUMN_OVERLAP = 0.3
# Further than this from the label, as a share of the page, the text belongs to something else.
MAX_RIGHT_GAP_RATIO = 0.25
MAX_BELOW_GAP_RATIO = 0.05
# Per-year labels further apart than this, as a share of page height, are separate blocks.
BLOCK_GAP_RATIO = 0.04

_ALIASES = [
    (alias, field, is_per_year)
    for table, is_per_year in ((IDENTITY_LABELS, False), (PER_YEAR_LABELS, True))
    for field, aliases in table.items()
    for alias in aliases
]


@dataclass(frozen=True)
class GenericField:
    """One value found on the page. `value` is empty when the label had nothing beside it."""

    name: str
    value: str
    confidence: float | None
    bbox: dict[str, int] | None


@dataclass(frozen=True)
class GenericPage:
    full_text: str
    fields: list[GenericField]
    model_version: str


@dataclass(frozen=True)
class _Label:
    box: TextBox
    field: str
    is_per_year: bool
    inline_value: str | None


def _normalize(text: str) -> str:
    """Lowercase ASCII words, so "Pangalán:" and "pangalan" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", no_accents.casefold()).strip()


def _match_alias(text: str) -> tuple[str, bool] | None:
    """(field, is per-year) for the alias closest to `text`. The whole text is compared,
    so "Pangalan ng Ama" can never be taken for "Pangalan"."""
    normalized = _normalize(text)
    if not normalized:
        return None
    best = None
    best_rank = (FUZZY_THRESHOLD, 0)
    for alias, field, is_per_year in _ALIASES:
        if len(alias) <= EXACT_ONLY_LENGTH or len(normalized) <= EXACT_ONLY_LENGTH:
            ratio = 1.0 if normalized == alias else 0.0
        else:
            ratio = SequenceMatcher(None, normalized, alias).ratio()
        rank = (ratio, len(alias))
        if ratio >= FUZZY_THRESHOLD and rank > best_rank:
            best, best_rank = (field, is_per_year), rank
    return best


def _match_label_part(text: str) -> tuple[str, bool] | None:
    """Labels are often printed in both languages: "Name/Pangalan"."""
    for part in text.split("/"):
        if found := _match_alias(part):
            return found
    return None


def _parse_label(box: TextBox) -> _Label | None:
    """A box that is a label, or a label and its value run together ("LRN: 1234")."""
    text = box.text.strip()
    # Split off any value first: fuzzy matching would accept "Birthdate 01" as a label alone,
    # and the "/" in a date would look like a bilingual label.
    head, colon, tail = text.partition(":")
    if colon and (found := _match_label_part(head)):
        return _Label(box, found[0], found[1], tail.strip() or None)
    # With no colon, only digits can follow: "School of Baguio" is a value, not a label.
    if match := re.match(r"^(\D+?)\s+(\d.*)$", text):
        head, tail = match.groups()
        if found := _match_label_part(head):
            return _Label(box, found[0], found[1], tail.strip())
    if found := _match_label_part(text):
        return _Label(box, found[0], found[1], None)
    return None


def _overlap(start_a: int, size_a: int, start_b: int, size_b: int) -> float:
    shared = min(start_a + size_a, start_b + size_b) - max(start_a, start_b)
    return max(shared, 0) / min(size_a, size_b)


def _same_line(a: TextBox, b: TextBox) -> bool:
    return _overlap(a.y, a.h, b.y, b.h) >= SAME_LINE_OVERLAP


def _reading_lines(boxes: list[TextBox]) -> list[list[TextBox]]:
    lines: list[list[TextBox]] = []
    for box in sorted(boxes, key=lambda b: (b.y, b.x)):
        if lines and _same_line(lines[-1][0], box):
            lines[-1].append(box)
        else:
            lines.append([box])
    return [sorted(line, key=lambda b: b.x) for line in lines]


def full_text(boxes: list[TextBox]) -> str:
    """All the text in reading order, one line per printed line."""
    return "\n".join(" ".join(b.text for b in line) for line in _reading_lines(boxes))


def _values_right(label: TextBox, boxes: list[TextBox], stops: list[TextBox], width: int):
    """The boxes to the right on the same line, up to the next label."""
    line = sorted(
        (b for b in boxes if b.x >= label.x + label.w // 2 and _same_line(label, b)),
        key=lambda b: b.x,
    )
    stop_at = min(
        (s.x for s in stops if s is not label and s.x > label.x and _same_line(label, s)),
        default=None,
    )
    found = []
    edge = label.x + label.w
    for box in line:
        if (stop_at is not None and box.x >= stop_at) or box.x - edge > MAX_RIGHT_GAP_RATIO * width:
            break
        found.append(box)
        edge = box.x + box.w
    return found


def _value_below(label: TextBox, boxes: list[TextBox], stops: list[TextBox], height: int):
    """The nearest box under the label, unless another label is nearer."""
    under = [
        b
        for b in (*boxes, *stops)
        if b is not label
        and b.y >= label.y + label.h // 2
        and _overlap(label.x, label.w, b.x, b.w) >= SAME_COLUMN_OVERLAP
        and b.y - (label.y + label.h) <= MAX_BELOW_GAP_RATIO * height
    ]
    nearest = min(under, key=lambda b: b.y, default=None)
    return [] if nearest is None or any(nearest is s for s in stops) else [nearest]


def _as_field(name: str, label: _Label, values: list[TextBox]) -> GenericField:
    if label.inline_value:
        box = label.box
        return GenericField(
            name,
            label.inline_value,
            box.confidence,
            {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
        )
    if not values:
        return GenericField(name, "", None, None)
    left = min(b.x for b in values)
    top = min(b.y for b in values)
    return GenericField(
        name,
        " ".join(b.text.strip() for b in values),
        # The least sure piece decides, as with the per-character score in step 7.
        min(b.confidence for b in values),
        {
            "x": left,
            "y": top,
            "w": max(b.x + b.w for b in values) - left,
            "h": max(b.y + b.h for b in values) - top,
        },
    )


def _group_per_year(labels: list[_Label], page_height: int) -> list[tuple[int, _Label]]:
    """Give each per-year label its block number, using position, not label order: on an
    SF10 the school name usually comes before the school year.

    A new band starts at a wide vertical gap, or when a field repeats directly below its
    earlier copy. A field repeating sideways means blocks side by side in one band, which
    are told apart by the left-to-right order of each field's copies.
    """
    bands: list[list[_Label]] = []
    for label in sorted(labels, key=lambda item: (item.box.y, item.box.x)):
        band = bands[-1] if bands else None
        if band is not None:
            lowest = max(item.box.y + item.box.h for item in band)
            wide_gap = label.box.y - lowest > BLOCK_GAP_RATIO * page_height
            repeats_below = any(
                item.field == label.field
                and _overlap(item.box.x, item.box.w, label.box.x, label.box.w) > 0
                for item in band
            )
            if wide_gap or repeats_below:
                band = None
        if band is None:
            band = []
            bands.append(band)
        band.append(label)

    grouped = []
    first_block = 0
    for band in bands:
        copies: dict[str, int] = {}
        for label in sorted(band, key=lambda item: item.box.x):
            copy = copies.get(label.field, 0)
            copies[label.field] = copy + 1
            grouped.append((first_block + copy, label))
        first_block += max(copies.values())
    return grouped


def extract_fields(boxes: list[TextBox], page_size: tuple[int, int]) -> list[GenericField]:
    """Find every label and take the text to its right or, failing that, below it."""
    width, height = page_size
    labels = []
    values = []
    for box in boxes:
        if label := _parse_label(box):
            labels.append(label)
        else:
            values.append(box)
    stops = [label.box for label in labels]

    # Text to the right is the stronger signal, so it claims its boxes first. A box under
    # one label may be the right-hand value of another.
    found: dict[int, list[TextBox]] = {}
    used: set[int] = set()
    for i, label in enumerate(labels):
        if not label.inline_value:
            found[i] = _values_right(label.box, values, stops, width)
            used.update(id(b) for b in found[i])
    for i, label in enumerate(labels):
        if not label.inline_value and not found[i]:
            free = [b for b in values if id(b) not in used]
            found[i] = _value_below(label.box, free, stops, height)
            used.update(id(b) for b in found[i])

    index_of = {id(label): i for i, label in enumerate(labels)}
    fields: dict[str, GenericField] = {}

    def add(name: str, label: _Label) -> None:
        # Printed twice ("Name" and "Pangalan" apart): keep the copy that has a value.
        candidate = _as_field(name, label, found.get(index_of[id(label)], []))
        if name not in fields or (candidate.value and not fields[name].value):
            fields[name] = candidate

    for label in sorted(labels, key=lambda item: (item.box.y, item.box.x)):
        if not label.is_per_year:
            add(label.field, label)
    per_year = [label for label in labels if label.is_per_year]
    for block, label in _group_per_year(per_year, height):
        add(f"{SECTION_NAME}[{block}].{label.field}", label)
    return list(fields.values())


def looks_like_a_form(regions: list[Region], page_size: tuple[int, int]) -> bool:
    """Judged from how much text the detector found, before anything is read."""
    width, height = page_size
    area = sum(r.w * r.h for r in regions)
    return len(regions) >= MIN_TEXT_REGIONS and area >= MIN_TEXT_AREA_RATIO * width * height


def read_generic_page(page: np.ndarray, reader: TextReader) -> GenericPage:
    """The generic path's whole job: read the page, or say it is not a form."""
    height, width = page.shape[:2]
    regions = reader.detect(page)
    if not looks_like_a_form(regions, (width, height)):
        raise PreprocessFailure(REASON_NOT_A_FORM)
    boxes = reader.recognize(page, regions)
    return GenericPage(
        full_text(boxes), extract_fields(boxes, (width, height)), reader.model_version
    )
