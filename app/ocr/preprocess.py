"""Steps 1 to 5 of the OCR pipeline: from a stored photo to clean field crops (Phase 7).

Pure image work: no database. A photo that matches no template is not an error; it comes
back on the generic path as a cleaned full page. Nothing here logs field contents.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.ocr.templates import FieldBox, FormLayout
from app.services.images import MAX_PIXELS

logger = logging.getLogger(__name__)

TEMPLATE_PATH = "template"
GENERIC_PATH = "generic"
# Used on the generic path, where no template says otherwise.
DEFAULT_PAGE_SIZE = (1700, 2200)

# Alignment. Check these against real phone photos with scripts/ocr_debug.py.
ORB_FEATURES = 5000
RATIO_TEST = 0.75
RANSAC_PIXELS = 5.0
# Fewer inliers than this is chance, whatever the score says.
MIN_INLIERS = 40
# Score = share of the template's page, in grid cells that have features, where features
# matched in the right place. Counting matches instead would let a different form pass on a
# shared letterhead alone. Measured on generated forms: 0.46+ when filled in, blurred or noisy;
# 0.29 or less for a different layout with the same heading.
MIN_ALIGNMENT_SCORE = 0.40
COVERAGE_GRID = (8, 10)
# How much bigger or smaller the photo's page may become when mapped onto the template.
MIN_AREA_RATIO = 0.5
MAX_AREA_RATIO = 2.0

REASON_MISSING = "The image file is missing from storage. Upload the photo again."
REASON_UNREADABLE = "The image file could not be read. Retake the photo and upload it again."
REASON_TOO_LARGE = "The image is too large to process. Retake the photo at a lower resolution."
REASON_NO_TEMPLATES = "No active form templates are set up. Ask an administrator to load one."

_CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
# What Pillow raises on damaged files; it has no single base class for them.
_UNREADABLE = (OSError, SyntaxError, ValueError, EOFError, IndexError, KeyError)


class PreprocessFailure(Exception):
    """An expected failure. `reason` is written for staff and holds no field contents."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class TemplateCandidate:
    """An active template. `id` is None when run from a JSON file with no database."""

    id: int | None
    layout: FormLayout
    image_path: Path


@dataclass(frozen=True)
class FieldCrop:
    name: str
    box: FieldBox
    raw: np.ndarray
    clean: np.ndarray


@dataclass
class PreprocessResult:
    ocr_path: str
    template: TemplateCandidate | None
    alignment_score: float | None
    crops: list[FieldCrop] = field(default_factory=list)
    clean_page: np.ndarray | None = None  # generic path only


@dataclass(frozen=True)
class _Alignment:
    score: float
    inliers: int
    homography: np.ndarray | None
    preview: np.ndarray | None  # matched features, drawn for the debug folder


class DebugSink:
    """Saves intermediate images. These show minors' data: keep them under storage/."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def image(self, name: str, image: np.ndarray) -> None:
        # imencode, not imwrite: OpenCV cannot open non-ASCII paths on Windows.
        ok, encoded = cv2.imencode(".png", image)
        if ok:
            target = self.directory / f"{name}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded.tobytes())

    def text(self, name: str, text: str) -> None:
        (self.directory / f"{name}.txt").write_text(text + "\n", encoding="utf-8")


def load_oriented(path: Path) -> np.ndarray:
    """Step 1. The photo as a BGR array, upright according to its EXIF."""
    try:
        with Image.open(path, formats=("JPEG", "PNG", "HEIF")) as image:
            if image.width * image.height > MAX_PIXELS:
                raise PreprocessFailure(REASON_TOO_LARGE)
            upright = ImageOps.exif_transpose(image).convert("RGB")
    except FileNotFoundError:
        raise PreprocessFailure(REASON_MISSING) from None
    except Image.DecompressionBombError:
        raise PreprocessFailure(REASON_TOO_LARGE) from None
    except _UNREADABLE:
        raise PreprocessFailure(REASON_UNREADABLE) from None
    return cv2.cvtColor(np.asarray(upright), cv2.COLOR_RGB2BGR)


def flatten(photo: np.ndarray, corners: dict[str, Any] | None, size: tuple[int, int]) -> np.ndarray:
    """Step 2. Warp the phone's four crop corners to a `size` page. Corners are fractions
    of the EXIF-oriented photo; batch uploads have none, so the whole photo is the page."""
    width, height = size
    if not corners:
        return cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
    photo_height, photo_width = photo.shape[:2]
    source = np.float32(
        [
            (corners[name]["x"] * photo_width, corners[name]["y"] * photo_height)
            for name in _CORNER_ORDER
        ]
    )
    target = np.float32([(0, 0), (width, 0), (width, height), (0, height)])
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(photo, matrix, size)


def _features(gray: np.ndarray) -> tuple[Any, Any]:
    return cv2.ORB_create(nfeatures=ORB_FEATURES).detectAndCompute(gray, None)


def _cells(points: list[tuple[float, float]], width: int, height: int) -> set[tuple[int, int]]:
    columns, rows = COVERAGE_GRID
    return {(int(x * columns / width), int(y * rows / height)) for x, y in points}


def _sensible_geometry(homography: np.ndarray, width: int, height: int) -> bool:
    """A real page maps to a convex shape of about the same size; a chance match does not."""
    page = np.float32([[(0, 0), (width, 0), (width, height), (0, height)]])
    mapped = cv2.perspectiveTransform(page, homography)
    if not cv2.isContourConvex(mapped):
        return False
    ratio = cv2.contourArea(mapped) / (width * height)
    return MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO


def _align(page: np.ndarray, blank: np.ndarray) -> _Alignment:
    """Step 3, for one template. The homography maps the flattened photo onto the blank."""
    nothing = _Alignment(0.0, 0, None, None)
    photo_gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    photo_points, photo_features = _features(photo_gray)
    blank_points, blank_features = _features(blank)
    if photo_features is None or blank_features is None or len(photo_points) < 2:
        return nothing

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(photo_features, blank_features, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
    if len(good) < MIN_INLIERS:
        return nothing

    source = np.float32([photo_points[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([blank_points[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_PIXELS)
    if homography is None:
        return nothing
    height, width = blank.shape[:2]
    if not _sensible_geometry(homography, width, height):
        return nothing

    inliers = int(mask.sum())
    inlier_matches = [m for m, keep in zip(good, mask.ravel(), strict=True) if keep]
    preview = cv2.drawMatches(
        photo_gray, photo_points, blank, blank_points, inlier_matches[:200], None
    )
    feature_cells = _cells([p.pt for p in blank_points], width, height)
    matched = [blank_points[m.trainIdx].pt for m in inlier_matches]
    score = len(_cells(matched, width, height) & feature_cells) / len(feature_cells)
    return _Alignment(score, inliers, homography, preview)


def _usable_templates(
    candidates: list[TemplateCandidate],
) -> list[tuple[TemplateCandidate, np.ndarray]]:
    usable = []
    for candidate in candidates:
        page = candidate.layout.page
        try:
            # imdecode, not imread: OpenCV cannot open non-ASCII paths on Windows.
            raw = np.frombuffer(candidate.image_path.read_bytes(), dtype=np.uint8)
            blank = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        except OSError:
            blank = None
        if blank is None:
            logger.warning("Skipping template %s: image not readable", candidate.layout.name)
            continue
        usable.append((candidate, cv2.resize(blank, (page.width, page.height))))
    return usable


def _remove_lines(gray: np.ndarray) -> np.ndarray:
    """Find long straight ruling lines and inpaint them, which repairs the handwriting
    strokes that cross them better than painting the line white would."""
    height, width = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    # Long enough that handwriting survives: the bar of a capital T, or a tall "l" that nearly
    # fills a tight box. A real vertical rule, such as an LRN digit divider, crosses all of it.
    across = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 3, 25), 1))
    down = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height * 19 // 20, 15)))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, across) | cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, down
    )
    if not lines.any():
        return gray
    mask = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


def clean_crop(crop: np.ndarray, kind: str) -> np.ndarray:
    """Step 5. Checkboxes keep their lines: step 6 counts dark pixels against the box."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    boosted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(boosted, h=7)
    return denoised if kind == "checkbox" else _remove_lines(denoised)


def clean_page(page: np.ndarray) -> np.ndarray:
    """The generic path's page. Lines stay: the text detector may use the table structure,
    and non-local-means denoising would take seconds at this size."""
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    boosted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.medianBlur(boosted, 3)


def _cut_fields(aligned: np.ndarray, layout: FormLayout) -> list[FieldCrop]:
    """Steps 4 and 5."""
    crops = []
    for name, box in layout.boxes().items():
        raw = aligned[box.y : box.y + box.h, box.x : box.x + box.w].copy()
        crops.append(FieldCrop(name, box, raw, clean_crop(raw, box.kind)))
    return crops


def _overlay(aligned: np.ndarray, crops: list[FieldCrop]) -> np.ndarray:
    """The field boxes drawn on the page: the way to check edits to the template JSON."""
    overlay = aligned.copy()
    for crop in crops:
        box = crop.box
        cv2.rectangle(overlay, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 0, 255), 2)
        cv2.putText(
            overlay, crop.name, (box.x, box.y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return overlay


def _save_failure(debug: DebugSink | None, failure: PreprocessFailure) -> None:
    if debug:
        debug.text("failure", failure.reason)


def preprocess(
    path: Path,
    corners: dict[str, Any] | None,
    candidates: list[TemplateCandidate],
    debug: DebugSink | None = None,
) -> PreprocessResult:
    """Run steps 1 to 5, trying every template and keeping the best score above the bar."""
    try:
        photo = load_oriented(path)
        templates = _usable_templates(candidates)
        if not templates:
            raise PreprocessFailure(REASON_NO_TEMPLATES)
    except PreprocessFailure as failure:
        _save_failure(debug, failure)
        raise
    if debug:
        debug.image("01_oriented", photo)
        debug.image("02_flattened", flatten(photo, corners, DEFAULT_PAGE_SIZE))

    best: tuple[TemplateCandidate, _Alignment, np.ndarray] | None = None
    best_score = 0.0
    scores = []
    for candidate, blank in templates:
        page = candidate.layout.page
        flattened = flatten(photo, corners, (page.width, page.height))
        alignment = _align(flattened, blank)
        layout = candidate.layout
        scores.append(
            f"{layout.name} v{layout.version}: score {alignment.score:.3f}, "
            f"{alignment.inliers} inliers"
        )
        if debug and alignment.preview is not None:
            debug.image(f"03_matches_{layout.name}_v{layout.version}", alignment.preview)
        best_score = max(best_score, alignment.score)
        if (
            alignment.homography is not None
            and alignment.score >= MIN_ALIGNMENT_SCORE
            and alignment.inliers >= MIN_INLIERS
            # Close variants of a form can tie on coverage; more matches then decides.
            and (
                best is None
                or (alignment.score, alignment.inliers) > (best[1].score, best[1].inliers)
            )
        ):
            best = (candidate, alignment, flattened)

    if best is None:
        if debug:
            debug.text("scores", "\n".join([*scores, f"chosen path: {GENERIC_PATH}"]))
        page = clean_page(flatten(photo, corners, DEFAULT_PAGE_SIZE))
        if debug:
            debug.image("04_clean_page", page)
        return PreprocessResult(GENERIC_PATH, None, best_score, clean_page=page)

    candidate, alignment, flattened = best
    layout = candidate.layout
    size = (layout.page.width, layout.page.height)
    # Off-page areas take the paper's tone: stretching the edge pixels would smear dark ones
    # into anything close to the edge, where they would read as ink.
    paper = [float(v) for v in np.median(flattened, axis=(0, 1))]
    aligned = cv2.warpPerspective(flattened, alignment.homography, size, borderValue=paper)
    crops = _cut_fields(aligned, layout)
    if debug:
        chosen = f"chosen path: {TEMPLATE_PATH} ({layout.name} v{layout.version})"
        debug.text("scores", "\n".join([*scores, chosen]))
        debug.image("04_aligned", aligned)
        debug.image("05_overlay", _overlay(aligned, crops))
        for crop in crops:
            debug.image(f"fields/{crop.name}_raw", crop.raw)
            debug.image(f"fields/{crop.name}_clean", crop.clean)
    return PreprocessResult(TEMPLATE_PATH, candidate, alignment.score, crops=crops)
