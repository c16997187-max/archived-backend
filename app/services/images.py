"""Check, clean and store uploaded form photos. No OCR here: that is the worker's job.

Metadata is removed by rewriting the file's segments and chunks, never by decoding
and re-encoding, so the pixels stay exactly as the phone took them.
"""

import errno
import hashlib
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from app.errors import ApiError

register_heif_opener()

MAX_FILE_BYTES = 15 * 1024 * 1024
# Stops decompression bombs: a small file that expands to gigabytes when decoded.
MAX_PIXELS = 120_000_000
HEIC_JPEG_QUALITY = 95

_EXIF_ORIENTATION = 0x0112
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}
# What Pillow raises on damaged files; it has no single base class for them.
_UNREADABLE = (OSError, SyntaxError, ValueError, EOFError, IndexError, KeyError, struct.error)

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = 0xD9
_JPEG_SOS = 0xDA
_JPEG_COM = 0xFE
_JPEG_STANDALONE = {0x01, *range(0xD0, 0xD8)}  # TEM and RST0-7 have no length field
# Frame headers (not C4 DHT, C8 reserved, CC DAC, which share the range).
_JPEG_SOF = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# Everything needed to decode the image: SOFn, DHT, DAC, DQT, DNL, DRI, DHP, EXP.
_JPEG_IMAGE_MARKERS = _JPEG_SOF | {0xC4, 0xCC, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF}
_JPEG_KEPT_APP_PREFIX = {
    0xE0: b"JFIF\x00",
    0xE2: b"ICC_PROFILE\x00",  # colours
    0xEE: b"Adobe",  # needed to decode the colours correctly
}

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_KEPT_CHUNKS = {
    b"IHDR",
    b"PLTE",
    b"IDAT",
    b"IEND",
    b"tRNS",
    b"gAMA",
    b"cHRM",
    b"sRGB",
    b"iCCP",
    b"sBIT",
    b"pHYs",
    b"bKGD",
}


class _MalformedImage(Exception):
    """The file's structure could not be walked, so its metadata cannot be removed."""


@dataclass(frozen=True)
class StoredImage:
    data: bytes
    sha256: str
    mime_type: str
    extension: str


def _unsupported() -> ApiError:
    return ApiError(415, "unsupported_image_type", {"file": "Must be a JPEG or PNG photo"})


def _heic_failed() -> ApiError:
    return ApiError(
        415,
        "heic_conversion_failed",
        {"file": 'Could not read this HEIC photo. Set the camera to "Most Compatible" and retake.'},
    )


def _too_large() -> ApiError:
    return ApiError(413, "file_too_large", {"file": "Must be 15 MB or smaller"})


def _orientation_tiff(orientation: int | None) -> bytes | None:
    """A new TIFF block holding only the orientation. Nothing is copied from the original."""
    if orientation is None or not 2 <= orientation <= 8:
        return None
    entry = struct.pack("<HHIHH", _EXIF_ORIENTATION, 3, 1, orientation, 0)
    return b"II*\x00" + struct.pack("<IH", 8, 1) + entry + struct.pack("<I", 0)


def _strip_jpeg(raw: bytes, orientation: int | None) -> bytes:
    if not raw.startswith(_JPEG_SOI):
        raise _MalformedImage
    out = [_JPEG_SOI]
    exif = None
    if tiff := _orientation_tiff(orientation):
        payload = b"Exif\x00\x00" + tiff
        exif = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    seen_frame = seen_scan = False
    pos = 2

    while True:
        if pos + 2 > len(raw) or raw[pos] != 0xFF:
            raise _MalformedImage
        marker = raw[pos + 1]
        if marker == 0xFF:  # fill byte before a marker
            pos += 1
            continue
        pos += 2
        if marker == _JPEG_EOI:
            if not (seen_frame and seen_scan):
                raise _MalformedImage
            # Whatever follows EOI is dropped: phones append extra images there,
            # each with its own EXIF.
            out.append(b"\xff\xd9")
            return b"".join(out)
        if marker in _JPEG_STANDALONE:
            out.append(bytes((0xFF, marker)))
            continue
        if pos + 2 > len(raw):
            raise _MalformedImage
        length = struct.unpack(">H", raw[pos : pos + 2])[0]
        if length < 2 or pos + length > len(raw):
            raise _MalformedImage
        segment = raw[pos - 2 : pos + length]
        body = raw[pos + 2 : pos + length]
        pos += length

        if 0xE0 <= marker <= 0xEF or marker == _JPEG_COM:
            prefix = _JPEG_KEPT_APP_PREFIX.get(marker)
            if prefix and body.startswith(prefix):
                out.append(segment)
            continue

        if marker != _JPEG_SOS and marker not in _JPEG_IMAGE_MARKERS:
            raise _MalformedImage
        if exif:  # after JFIF, before the image data
            out.append(exif)
            exif = None
        out.append(segment)
        seen_frame = seen_frame or marker in _JPEG_SOF
        if marker == _JPEG_SOS:
            seen_scan = True
            end = _jpeg_scan_end(raw, pos)
            out.append(raw[pos:end])
            pos = end


def _jpeg_scan_end(raw: bytes, start: int) -> int:
    """Where the compressed data after SOS stops: the next real marker."""
    pos = start
    while True:
        pos = raw.find(b"\xff", pos)
        if pos < 0 or pos + 1 >= len(raw):
            raise _MalformedImage
        following = raw[pos + 1]
        if following == 0x00 or 0xD0 <= following <= 0xD7:  # stuffed byte or restart marker
            pos += 2
        elif following == 0xFF:
            pos += 1
        else:
            return pos


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _strip_png(raw: bytes, orientation: int | None) -> bytes:
    if not raw.startswith(_PNG_SIGNATURE):
        raise _MalformedImage
    out = [_PNG_SIGNATURE]
    tiff = _orientation_tiff(orientation)
    pos = len(_PNG_SIGNATURE)

    while True:
        if pos + 12 > len(raw):
            raise _MalformedImage
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        kind = raw[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > len(raw) or (len(out) == 1 and kind != b"IHDR"):
            raise _MalformedImage
        if kind in _PNG_KEPT_CHUNKS:
            out.append(raw[pos:end])
        if kind == b"IHDR" and tiff:
            out.append(_png_chunk(b"eXIf", tiff))
        if kind == b"IEND":
            return b"".join(out)  # anything after IEND is dropped
        pos = end


def _check_pixels(image: Image.Image) -> None:
    if image.width * image.height > MAX_PIXELS:
        raise _too_large()


def _looks_like_heif(raw: bytes) -> bool:
    return raw[4:8] == b"ftyp" and raw[8:12] in _HEIF_BRANDS


def _convert_heif(raw: bytes) -> bytes:
    """HEIC to an upright JPEG with no metadata, so the corners still match what was shown."""
    try:
        with Image.open(BytesIO(raw), formats=("HEIF",)) as image:
            _check_pixels(image)
            upright = ImageOps.exif_transpose(image).convert("RGB")
    except Image.DecompressionBombError:
        raise _too_large() from None
    except _UNREADABLE:
        raise _heic_failed() from None
    upright.info = {}
    buffer = BytesIO()
    upright.save(buffer, format="JPEG", quality=HEIC_JPEG_QUALITY)
    return buffer.getvalue()


def _inspect(raw: bytes) -> tuple[str, int | None]:
    """(format, EXIF orientation) from the file's content; its name and type are ignored."""
    try:
        with Image.open(BytesIO(raw), formats=("JPEG", "PNG")) as image:
            _check_pixels(image)
            image_format = image.format
            image.verify()
        # verify() leaves the image unusable, and reading a PNG's EXIF needs a full load.
        with Image.open(BytesIO(raw), formats=("JPEG", "PNG")) as image:
            return image_format, image.getexif().get(_EXIF_ORIENTATION)
    except Image.DecompressionBombError:
        raise _too_large() from None
    except _UNREADABLE:
        raise _unsupported() from None


def prepare_image(raw: bytes) -> StoredImage:
    """Turn an upload into the bytes to store, or raise the ApiError to show the user."""
    if len(raw) > MAX_FILE_BYTES:
        raise _too_large()
    if not raw:
        raise ApiError(422, "validation_error", {"file": "The file is empty"})

    if _looks_like_heif(raw):
        data = _convert_heif(raw)
        mime_type, extension = "image/jpeg", ".jpg"
    else:
        image_format, orientation = _inspect(raw)
        try:
            if image_format == "JPEG":
                data = _strip_jpeg(raw, orientation)
                mime_type, extension = "image/jpeg", ".jpg"
            else:
                data = _strip_png(raw, orientation)
                mime_type, extension = "image/png", ".png"
        except _MalformedImage:
            # Refused rather than stored with metadata we could not remove.
            raise _unsupported() from None

    return StoredImage(data, hashlib.sha256(data).hexdigest(), mime_type, extension)


def store_image(storage_dir: Path, image: StoredImage) -> str:
    """Write the file durably and return its path relative to `storage_dir`.

    The name is the hash, and files appear only by atomic rename, so an existing
    file is already complete and correct.
    """
    relative = PurePosixPath("images", image.sha256[:2], f"{image.sha256}{image.extension}")
    target = storage_dir / relative
    if target.exists():
        return str(relative)

    temp_name = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".tmp") as temp:
            temp_name = temp.name
            temp.write(image.data)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, target)
        temp_name = None
        _fsync_dir(target.parent)
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            raise ApiError(507, "storage_full") from None
        raise
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
    return str(relative)


def _fsync_dir(path: Path) -> None:
    # Makes the rename itself survive a power cut. Windows cannot open a directory.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
