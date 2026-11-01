"""QR pairing, phone uploads and the desktop poll. Every image is generated here."""

import hashlib
import io
import json
import logging
import re
import struct
import zlib
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.routing import iter_route_contexts
from fastapi.testclient import TestClient
from PIL import Image, ImageOps, PngImagePlugin
from sqlalchemy import func, select, update
from starlette.requests import Request

from app.api.capture import CAPTURE_COOKIE
from app.db.session import get_db
from app.errors import ApiError
from app.log_filters import RedactCaptureToken
from app.main import create_app
from app.models import AuditLog, CaptureSession, Document, Job, User
from app.schemas.capture import CropCorners
from app.services import capture as capture_service
from app.services import images as images_service
from app.services.tokens import hash_token
from tests.test_role_matrix import PUBLIC_ROUTES

CORNERS = {
    "top_left": {"x": 0.1, "y": 0.1},
    "top_right": {"x": 0.9, "y": 0.1},
    "bottom_right": {"x": 0.9, "y": 0.9},
    "bottom_left": {"x": 0.1, "y": 0.9},
}
ORIENTATION = 0x0112
MAKE = 0x010F
MODEL = 0x0110
GPS_IFD = 0x8825
FAKE_MAKE = b"FakeCamMake"
FAKE_MODEL = b"FakeCamModel"
FAKE_GPS_DATE = b"2099:01:02"


def noise_image(seed: int = 0, size: tuple[int, int] = (48, 32)) -> Image.Image:
    # A different picture per seed, so each gets its own hash.
    pixels = bytes((i * 7 + seed * 31) % 256 for i in range(size[0] * size[1] * 3))
    return Image.frombytes("RGB", size, pixels)


def jpeg_bytes(seed: int = 0, **save_args) -> bytes:
    buffer = io.BytesIO()
    noise_image(seed).save(buffer, format="JPEG", **save_args)
    return buffer.getvalue()


def png_bytes(seed: int = 0, **save_args) -> bytes:
    buffer = io.BytesIO()
    noise_image(seed).save(buffer, format="PNG", **save_args)
    return buffer.getvalue()


def heic_bytes(seed: int = 0) -> bytes:
    buffer = io.BytesIO()
    noise_image(seed, (64, 64)).save(buffer, format="HEIF")
    return buffer.getvalue()


def fake_exif(orientation: int | None = 6) -> Image.Exif:
    exif = Image.Exif()
    exif[MAKE] = FAKE_MAKE.decode()
    exif[MODEL] = FAKE_MODEL.decode()
    if orientation:
        exif[ORIENTATION] = orientation
    gps = exif.get_ifd(GPS_IFD)
    gps[1] = "N"
    gps[2] = (16.0, 24.0, 0.0)
    gps[29] = FAKE_GPS_DATE.decode()
    return exif


def pixels(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB").tobytes()


def read_exif(data: bytes) -> Image.Exif:
    with Image.open(io.BytesIO(data)) as image:
        return image.getexif()


@pytest.fixture
def phone(app):
    """A second browser: its own cookie jar, separate from the desktop's client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def staff_headers(make_user, login):
    return login(make_user("registrar"))


@pytest.fixture
def new_session(client, staff_headers):
    def _new() -> dict:
        response = client.post("/api/capture-sessions", headers=staff_headers)
        assert response.status_code == 201, response.text
        return response.json()

    return _new


@pytest.fixture
def claimed(new_session, phone):
    """A session paired with the `phone` client."""
    created = new_session()
    response = phone.post(f"/api/capture-sessions/{created['token']}/claim")
    assert response.status_code == 200, response.text
    return created


def post_image(browser, token, data=None, filename="photo.jpg", mime="image/jpeg", corners=None):
    return browser.post(
        f"/api/capture-sessions/{token}/images",
        data={"corners": json.dumps(corners or CORNERS)},
        files={"file": (filename, data if data is not None else jpeg_bytes(), mime)},
    )


def get_row(db, session_id) -> CaptureSession:
    db.expire_all()
    return db.get(CaptureSession, session_id)


def count(db, model) -> int:
    return db.scalar(select(func.count()).select_from(model))


def audit_actions(db, entity, entity_id) -> list[str]:
    return list(
        db.scalars(
            select(AuditLog.action)
            .where(AuditLog.entity == entity, AuditLog.entity_id == entity_id)
            .order_by(AuditLog.id)
        )
    )


def assert_error(response, status, error):
    assert response.status_code == status, response.text
    assert response.json()["error"] == error
    assert set(response.json()) == {"error", "fields"}


# --- create and read ---------------------------------------------------------------


def test_create_returns_token_once_and_stores_only_its_hash(client, staff_headers, db, new_session):
    created = new_session()

    row = get_row(db, created["id"])
    assert len(created["token"]) == 43
    assert row.token_hash == hash_token(created["token"])
    assert row.token_hash != created["token"]
    assert row.status == "open"
    poll = client.get(f"/api/capture-sessions/{created['id']}", headers=staff_headers)
    assert created["token"] not in poll.text
    assert "token" not in poll.json()


def test_create_sets_reference_number_and_ten_minute_expiry(new_session):
    before = datetime.now(UTC)
    created = new_session()

    assert re.fullmatch(r"CS-\d{6}-[2-9A-HJ-NP-Z]{5}", created["reference_no"])
    expires_at = datetime.fromisoformat(created["expires_at"])
    assert timedelta(minutes=9) < expires_at - before <= timedelta(minutes=10, seconds=5)


def test_create_is_audited_without_the_token(db, new_session):
    created = new_session()

    row = db.scalar(
        select(AuditLog).where(
            AuditLog.entity == "capture_session", AuditLog.entity_id == created["id"]
        )
    )
    assert row.action == "create"
    assert created["token"] not in json.dumps(row.after)


def test_capture_url_uses_public_base_url(settings, app, staff_headers):
    settings.public_base_url = "http://192.0.2.10:8000"
    with TestClient(app) as client:
        created = client.post("/api/capture-sessions", headers=staff_headers).json()

    assert created["capture_url"] == f"http://192.0.2.10:8000/m/{created['token']}"


def test_capture_url_falls_back_to_the_request(new_session):
    created = new_session()

    assert created["capture_url"] == f"http://testserver/m/{created['token']}"


def test_public_base_url_has_its_trailing_slash_removed(settings, app, staff_headers):
    settings.public_base_url = "http://192.0.2.10:8000/"
    with TestClient(app) as client:
        created = client.post("/api/capture-sessions", headers=staff_headers).json()

    assert created["capture_url"].startswith("http://192.0.2.10:8000/m/")


def test_poll_unknown_session_is_404(client, staff_headers):
    response = client.get("/api/capture-sessions/999999", headers=staff_headers)

    assert_error(response, 404, "not_found")


def test_poll_invalid_id_is_422(client, staff_headers):
    response = client.get("/api/capture-sessions/abc", headers=staff_headers)

    assert_error(response, 422, "validation_error")


def test_poll_lists_uploads(client, staff_headers, claimed, phone):
    document_id = post_image(phone, claimed["token"]).json()["document_id"]

    body = client.get(f"/api/capture-sessions/{claimed['id']}", headers=staff_headers).json()

    assert body["status"] == "claimed"
    assert body["upload_count"] == 1
    assert body["upload_limit"] == 50
    assert body["duplicate_count"] == 0
    assert [d["id"] for d in body["documents"]] == [document_id]
    assert "file_path" not in body["documents"][0]


# --- claim -------------------------------------------------------------------------


def test_claim_sets_a_narrow_cookie(new_session, phone, db):
    created = new_session()

    response = phone.post(f"/api/capture-sessions/{created['token']}/claim")

    assert response.status_code == 200
    assert response.json()["reference_no"] == created["reference_no"]
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(f"{CAPTURE_COOKIE}=")
    assert f"Path=/api/capture-sessions/{created['token']}" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert "Secure" not in cookie
    row = get_row(db, created["id"])
    assert row.status == "claimed"
    assert row.phone_token_hash == hash_token(phone.cookies[CAPTURE_COOKIE])
    assert audit_actions(db, "capture_session", created["id"]) == ["create", "claim"]


def test_claim_cookie_is_secure_when_configured(settings, new_session, phone):
    settings.cookie_secure = True
    created = new_session()

    response = phone.post(f"/api/capture-sessions/{created['token']}/claim")

    assert "Secure" in response.headers["set-cookie"]


def test_claim_again_from_the_same_phone_resumes(claimed, phone, db):
    # Phones often reload the page when the camera app closes.
    before = get_row(db, claimed["id"]).phone_token_hash

    response = phone.post(f"/api/capture-sessions/{claimed['token']}/claim")

    assert response.status_code == 200
    assert "set-cookie" not in response.headers
    assert get_row(db, claimed["id"]).phone_token_hash == before


def test_second_phone_cannot_claim(app, claimed, db):
    with TestClient(app) as second_phone:
        response = second_phone.post(f"/api/capture-sessions/{claimed['token']}/claim")

    assert_error(response, 409, "already_claimed")
    assert "set-cookie" not in response.headers
    assert audit_actions(db, "capture_session", claimed["id"])[-1] == "claim_rejected"


def test_second_phone_with_a_made_up_cookie_cannot_claim(app, claimed):
    with TestClient(app) as second_phone:
        response = second_phone.post(
            f"/api/capture-sessions/{claimed['token']}/claim",
            headers={"Cookie": f"{CAPTURE_COOKIE}=made-up"},
        )

    assert_error(response, 409, "already_claimed")


def test_claim_unknown_token_is_404(phone):
    assert_error(phone.post(f"/api/capture-sessions/{'a' * 43}/claim"), 404, "not_found")


def test_claim_malformed_token_is_422(phone):
    assert_error(phone.post("/api/capture-sessions/short/claim"), 422, "validation_error")


def test_claim_expired_is_410(new_session, phone, db):
    created = new_session()
    get_row(db, created["id"]).expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.flush()

    response = phone.post(f"/api/capture-sessions/{created['token']}/claim")

    assert_error(response, 410, "session_expired")
    assert get_row(db, created["id"]).status == "expired"
    assert audit_actions(db, "capture_session", created["id"])[-1] == "expire"


def test_claim_when_owner_is_deactivated_is_410(new_session, phone, db):
    created = new_session()
    row = get_row(db, created["id"])
    db.get(User, row.user_id).is_active = False
    db.flush()

    response = phone.post(f"/api/capture-sessions/{created['token']}/claim")

    assert_error(response, 410, "session_closed")


# --- upload ------------------------------------------------------------------------


def test_upload_saves_file_document_job_and_audit(claimed, phone, db, settings):
    response = post_image(phone, claimed["token"])

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["duplicate"] is False
    assert body["reference_no"] == claimed["reference_no"]
    assert body["upload_count"] == 1
    assert body["uploads_remaining"] == 49

    document = db.get(Document, body["document_id"])
    row = get_row(db, claimed["id"])
    assert document.source == "phone"
    assert document.mime_type == "image/jpeg"
    assert document.uploaded_by == row.user_id
    assert document.capture_session_id == claimed["id"]
    assert document.crop_corners == CORNERS

    stored = (settings.storage_dir / document.file_path).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == document.sha256
    assert document.file_path == f"images/{document.sha256[:2]}/{document.sha256}.jpg"

    job = db.scalar(select(Job).where(Job.payload["document_id"].as_integer() == document.id))
    assert job.type == "ocr"
    assert job.status == "queued"

    log = db.scalar(
        select(AuditLog).where(AuditLog.entity == "document", AuditLog.entity_id == document.id)
    )
    assert log.action == "create"
    assert log.user_id == row.user_id
    assert log.after["capture_session_id"] == claimed["id"]


def test_upload_leaves_no_temp_files(claimed, phone, settings):
    post_image(phone, claimed["token"])

    names = [p.name for p in settings.storage_dir.rglob("*") if p.is_file()]
    assert len(names) == 1
    assert names[0].endswith(".jpg")


def test_upload_png(claimed, phone, db):
    response = post_image(phone, claimed["token"], png_bytes(), "photo.png", "image/png")

    assert response.status_code == 201
    document = db.get(Document, response.json()["document_id"])
    assert document.mime_type == "image/png"
    assert document.file_path.endswith(".png")


def test_upload_type_comes_from_the_content_not_the_label(claimed, phone, db):
    response = post_image(phone, claimed["token"], png_bytes(), "photo.jpg", "image/jpeg")

    assert response.status_code == 201
    assert db.get(Document, response.json()["document_id"]).mime_type == "image/png"


def test_upload_heic_is_converted_to_jpeg(claimed, phone, db, settings):
    response = post_image(phone, claimed["token"], heic_bytes(), "photo.heic", "image/heic")

    assert response.status_code == 201, response.text
    document = db.get(Document, response.json()["document_id"])
    assert document.mime_type == "image/jpeg"
    stored = (settings.storage_dir / document.file_path).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == document.sha256
    with Image.open(io.BytesIO(stored)) as image:
        assert image.format == "JPEG"
        assert image.size == (64, 64)
        assert not image.getexif()


def test_same_heic_twice_is_a_duplicate(claimed, phone):
    data = heic_bytes()

    assert post_image(phone, claimed["token"], data, "a.heic", "image/heic").status_code == 201
    second = post_image(phone, claimed["token"], data, "a.heic", "image/heic")

    assert second.status_code == 200
    assert second.json()["duplicate"] is True


def test_upload_broken_heic_is_415(claimed, phone, db):
    data = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"not really a photo"

    response = post_image(phone, claimed["token"], data, "a.heic", "image/heic")

    assert_error(response, 415, "heic_conversion_failed")
    assert "file" in response.json()["fields"]
    assert count(db, Document) == 0


@pytest.mark.parametrize(
    "data",
    [b"this is not an image", b"%PDF-1.7 fake", b"<svg xmlns='http://www.w3.org/2000/svg'/>"],
)
def test_upload_non_image_labelled_as_jpeg_is_415(claimed, phone, db, data):
    response = post_image(phone, claimed["token"], data)

    assert_error(response, 415, "unsupported_image_type")
    assert count(db, Document) == 0


def test_upload_other_image_format_is_415(claimed, phone):
    buffer = io.BytesIO()
    noise_image().save(buffer, format="GIF")

    response = post_image(phone, claimed["token"], buffer.getvalue())

    assert_error(response, 415, "unsupported_image_type")


def test_upload_empty_file_is_422(claimed, phone):
    response = post_image(phone, claimed["token"], b"")

    assert_error(response, 422, "validation_error")
    assert "file" in response.json()["fields"]


def test_upload_without_a_file_is_422(claimed, phone):
    response = phone.post(
        f"/api/capture-sessions/{claimed['token']}/images",
        data={"corners": json.dumps(CORNERS)},
    )

    assert_error(response, 422, "validation_error")
    assert "file" in response.json()["fields"]


def test_upload_over_the_size_limit_is_413(claimed, phone, db):
    data = jpeg_bytes() + b"\x00" * images_service.MAX_FILE_BYTES

    response = post_image(phone, claimed["token"], data)

    assert_error(response, 413, "file_too_large")
    assert count(db, Document) == 0


def test_upload_at_the_size_limit_is_accepted(claimed, phone):
    # Bytes after the image end are ignored by decoders and dropped when stored.
    data = jpeg_bytes()
    data += b"\x00" * (images_service.MAX_FILE_BYTES - len(data))

    assert post_image(phone, claimed["token"], data).status_code == 201


def test_oversized_request_is_refused_without_parsing_it(claimed, phone, monkeypatch):
    def fail_if_parsed(*args, **kwargs):
        raise AssertionError("the body was parsed")

    monkeypatch.setattr(Request, "form", fail_if_parsed)

    response = post_image(phone, claimed["token"], b"\x00" * (16 * 1024 * 1024))

    assert_error(response, 413, "file_too_large")


def test_upload_without_a_content_length_is_411(claimed, phone):
    def chunks():
        yield b"--x\r\n"

    response = phone.post(
        f"/api/capture-sessions/{claimed['token']}/images",
        content=chunks(),
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )

    assert_error(response, 411, "length_required")


def test_upload_malformed_multipart_is_422(claimed, phone):
    response = phone.post(
        f"/api/capture-sessions/{claimed['token']}/images",
        content=b"garbage",
        headers={"Content-Type": "multipart/form-data"},
    )

    assert_error(response, 422, "validation_error")


def test_too_many_pixels_is_413(claimed, phone, monkeypatch):
    monkeypatch.setattr(images_service, "MAX_PIXELS", 100)

    assert_error(post_image(phone, claimed["token"]), 413, "file_too_large")


def test_full_disk_is_507_and_saves_nothing(claimed, phone, db, monkeypatch):
    def disk_full(*args, **kwargs):
        raise OSError(images_service.errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(images_service.os, "fsync", disk_full)

    response = post_image(phone, claimed["token"])

    assert_error(response, 507, "storage_full")
    assert count(db, Document) == 0
    assert count(db, Job) == 0


def test_full_disk_leaves_no_partial_file(claimed, phone, settings, monkeypatch):
    def disk_full(*args, **kwargs):
        raise OSError(images_service.errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(images_service.os, "fsync", disk_full)

    post_image(phone, claimed["token"])

    assert [p for p in settings.storage_dir.rglob("*") if p.is_file()] == []


def test_document_job_and_audit_roll_back_together(claimed, phone, db, monkeypatch):
    def audit_fails(*args, **kwargs):
        raise RuntimeError("audit failed")

    monkeypatch.setattr(capture_service.audit, "record", audit_fails)

    with pytest.raises(RuntimeError):
        post_image(phone, claimed["token"])

    db.rollback()
    assert count(db, Document) == 0
    assert count(db, Job) == 0
    assert get_row(db, claimed["id"]).upload_count == 0


@pytest.mark.parametrize(
    "corners",
    [
        "not json",
        "{}",
        json.dumps({**CORNERS, "top_left": {"x": 1.5, "y": 0.1}}),
        json.dumps({**CORNERS, "top_left": {"x": -0.1, "y": 0.1}}),
        # Crossed: top and bottom swapped on the right side.
        json.dumps(
            {
                **CORNERS,
                "top_right": CORNERS["bottom_right"],
                "bottom_right": CORNERS["top_right"],
            }
        ),
        # Not in clockwise order.
        json.dumps(
            {
                "top_left": CORNERS["top_left"],
                "top_right": CORNERS["bottom_left"],
                "bottom_right": CORNERS["bottom_right"],
                "bottom_left": CORNERS["top_right"],
            }
        ),
        # All on one line.
        json.dumps(
            {
                "top_left": {"x": 0.1, "y": 0.5},
                "top_right": {"x": 0.3, "y": 0.5},
                "bottom_right": {"x": 0.6, "y": 0.5},
                "bottom_left": {"x": 0.9, "y": 0.5},
            }
        ),
        # Far too small.
        json.dumps(
            {
                "top_left": {"x": 0.5, "y": 0.5},
                "top_right": {"x": 0.51, "y": 0.5},
                "bottom_right": {"x": 0.51, "y": 0.51},
                "bottom_left": {"x": 0.5, "y": 0.51},
            }
        ),
        '{"top_left":{"x":NaN,"y":0.1}}',
    ],
)
def test_bad_corners_are_422(claimed, phone, db, corners):
    response = phone.post(
        f"/api/capture-sessions/{claimed['token']}/images",
        data={"corners": corners},
        files={"file": ("photo.jpg", jpeg_bytes(), "image/jpeg")},
    )

    assert_error(response, 422, "validation_error")
    assert "corners" in response.json()["fields"]
    assert count(db, Document) == 0


def test_missing_corners_are_422(claimed, phone):
    response = phone.post(
        f"/api/capture-sessions/{claimed['token']}/images",
        files={"file": ("photo.jpg", jpeg_bytes(), "image/jpeg")},
    )

    assert_error(response, 422, "validation_error")
    assert "corners" in response.json()["fields"]


def test_whole_photo_is_a_valid_crop():
    # What the phone page sends when it cannot preview a photo.
    CropCorners.model_validate(
        {
            "top_left": {"x": 0, "y": 0},
            "top_right": {"x": 1, "y": 0},
            "bottom_right": {"x": 1, "y": 1},
            "bottom_left": {"x": 0, "y": 1},
        }
    )


# --- duplicates --------------------------------------------------------------------


def test_same_photo_twice_is_one_document_and_one_job(client, staff_headers, claimed, phone, db):
    first = post_image(phone, claimed["token"])
    second = post_image(phone, claimed["token"])

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["document_id"] == first.json()["document_id"]
    assert count(db, Document) == 1
    assert count(db, Job) == 1
    poll = client.get(f"/api/capture-sessions/{claimed['id']}", headers=staff_headers).json()
    assert poll["upload_count"] == 2
    assert poll["duplicate_count"] == 1
    assert len(poll["documents"]) == 1
    assert audit_actions(db, "document", first.json()["document_id"]) == [
        "create",
        "duplicate_upload",
    ]


def test_duplicate_keeps_the_first_corners(claimed, phone, db):
    document_id = post_image(phone, claimed["token"]).json()["document_id"]
    other = {**CORNERS, "top_left": {"x": 0.2, "y": 0.2}}

    post_image(phone, claimed["token"], corners=other)

    db.expire_all()
    assert db.get(Document, document_id).crop_corners == CORNERS


def test_duplicate_across_sessions_is_detected(app, new_session, phone, db):
    first_session = new_session()
    phone.post(f"/api/capture-sessions/{first_session['token']}/claim")
    first = post_image(phone, first_session["token"])

    second_session = new_session()
    with TestClient(app) as second_phone:
        second_phone.post(f"/api/capture-sessions/{second_session['token']}/claim")
        second = post_image(second_phone, second_session["token"])

    assert second.status_code == 200
    assert second.json()["document_id"] == first.json()["document_id"]
    assert count(db, Job) == 1


def test_same_hash_saved_first_by_another_phone_is_a_duplicate(claimed, phone, db, monkeypatch):
    # The moment between checking for the hash and inserting it: only the database
    # constraint can catch this one.
    document_id = post_image(phone, claimed["token"]).json()["document_id"]
    real_find = capture_service._find_document
    answers = iter([None])
    monkeypatch.setattr(
        capture_service,
        "_find_document",
        lambda session, sha256: next(answers, real_find(session, sha256)),
    )

    response = post_image(phone, claimed["token"])

    assert response.status_code == 200
    assert response.json()["document_id"] == document_id
    assert count(db, Job) == 1


# --- limits and expiry -------------------------------------------------------------


def test_fifty_first_upload_is_refused(claimed, phone, db):
    get_row(db, claimed["id"]).upload_count = 49
    db.flush()

    last = post_image(phone, claimed["token"], jpeg_bytes(1))
    refused = post_image(phone, claimed["token"], jpeg_bytes(2))

    assert last.status_code == 201
    assert last.json()["uploads_remaining"] == 0
    assert_error(refused, 409, "upload_limit_reached")
    assert count(db, Document) == 1


def test_duplicates_count_toward_the_limit(claimed, phone, db):
    post_image(phone, claimed["token"])
    get_row(db, claimed["id"]).upload_count = 49
    db.flush()

    assert post_image(phone, claimed["token"]).status_code == 200
    assert_error(post_image(phone, claimed["token"]), 409, "upload_limit_reached")


def test_limit_is_checked_again_after_reading_the_upload(claimed, db, tmp_path):
    # A second upload from the same phone can fill the session during this one.
    get_row(db, claimed["id"]).upload_count = 50
    db.flush()

    with pytest.raises(ApiError) as error:
        capture_service.save_upload(
            db, claimed["id"], jpeg_bytes(), CropCorners(**CORNERS), tmp_path, None
        )

    assert error.value.error == "upload_limit_reached"


def test_upload_extends_the_idle_expiry(claimed, phone, db):
    row = get_row(db, claimed["id"])
    row.expires_at = datetime.now(UTC) + timedelta(minutes=1)
    db.flush()

    post_image(phone, claimed["token"])

    remaining = get_row(db, claimed["id"]).expires_at - datetime.now(UTC)
    assert remaining > timedelta(minutes=9)


def test_claim_extends_the_idle_expiry(new_session, phone, db):
    created = new_session()
    get_row(db, created["id"]).expires_at = datetime.now(UTC) + timedelta(minutes=1)
    db.flush()

    phone.post(f"/api/capture-sessions/{created['token']}/claim")

    assert get_row(db, created["id"]).expires_at - datetime.now(UTC) > timedelta(minutes=9)


def test_rejected_upload_does_not_extend_the_expiry(claimed, phone, db):
    expires_at = datetime.now(UTC) + timedelta(minutes=1)
    get_row(db, claimed["id"]).expires_at = expires_at
    db.flush()

    post_image(phone, claimed["token"], b"not an image")

    assert get_row(db, claimed["id"]).expires_at == expires_at


def test_polling_does_not_extend_the_expiry(client, staff_headers, claimed, db):
    expires_at = datetime.now(UTC) + timedelta(minutes=1)
    get_row(db, claimed["id"]).expires_at = expires_at
    db.flush()

    client.get(f"/api/capture-sessions/{claimed['id']}", headers=staff_headers)

    assert get_row(db, claimed["id"]).expires_at == expires_at


def test_poll_marks_an_idle_session_expired(client, staff_headers, claimed, db):
    get_row(db, claimed["id"]).expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.flush()

    body = client.get(f"/api/capture-sessions/{claimed['id']}", headers=staff_headers).json()

    row = get_row(db, claimed["id"])
    assert body["status"] == "expired"
    assert row.status == "expired"
    assert row.phone_token_hash is None
    assert audit_actions(db, "capture_session", claimed["id"])[-1] == "expire"


def test_upload_after_idle_expiry_is_410(claimed, phone, db):
    get_row(db, claimed["id"]).expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.flush()

    assert_error(post_image(phone, claimed["token"]), 410, "session_expired")
    assert get_row(db, claimed["id"]).status == "expired"
    assert count(db, Document) == 0


def test_upload_when_owner_is_deactivated_is_410(claimed, phone, db):
    db.get(User, get_row(db, claimed["id"]).user_id).is_active = False
    db.flush()

    assert_error(post_image(phone, claimed["token"]), 410, "session_closed")


# --- close -------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["registrar", "admin"])
def test_close_revokes_the_cookie_and_is_audited(client, make_user, login, claimed, db, role):
    closer = make_user(role)

    response = client.post(f"/api/capture-sessions/{claimed['id']}/close", headers=login(closer))

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "closed"
    row = get_row(db, claimed["id"])
    assert row.status == "closed"
    assert row.phone_token_hash is None
    log = db.scalar(
        select(AuditLog).where(
            AuditLog.entity == "capture_session",
            AuditLog.entity_id == claimed["id"],
            AuditLog.action == "close",
        )
    )
    assert log.user_id == closer.id
    assert log.before == {"status": "claimed"}
    assert log.after == {"status": "closed"}


def test_close_an_unclaimed_session(client, staff_headers, new_session, phone):
    created = new_session()

    response = client.post(f"/api/capture-sessions/{created['id']}/close", headers=staff_headers)

    assert response.json()["status"] == "closed"
    assert_error(
        phone.post(f"/api/capture-sessions/{created['token']}/claim"), 410, "session_closed"
    )


def test_upload_after_close_is_410_session_closed(client, staff_headers, claimed, phone, db):
    client.post(f"/api/capture-sessions/{claimed['id']}/close", headers=staff_headers)

    assert CAPTURE_COOKIE in phone.cookies  # the phone still holds the revoked cookie
    assert_error(post_image(phone, claimed["token"]), 410, "session_closed")
    assert count(db, Document) == 0


def test_claim_after_close_is_410_session_closed(client, staff_headers, claimed, phone):
    client.post(f"/api/capture-sessions/{claimed['id']}/close", headers=staff_headers)

    assert_error(
        phone.post(f"/api/capture-sessions/{claimed['token']}/claim"), 410, "session_closed"
    )


def test_close_during_an_upload_refuses_the_upload(claimed, db, tmp_path):
    # Closed by another request after this one passed its checks and read the photo.
    row = get_row(db, claimed["id"])
    assert row.status == "claimed"
    db.execute(
        update(CaptureSession)
        .where(CaptureSession.id == claimed["id"])
        .values(status="closed", phone_token_hash=None)
        .execution_options(synchronize_session=False)
    )
    assert row.status == "claimed"  # this request's copy of the row is now out of date

    with pytest.raises(ApiError) as error:
        capture_service.save_upload(
            db, claimed["id"], jpeg_bytes(), CropCorners(**CORNERS), tmp_path, None
        )

    assert error.value.error == "session_closed"
    assert count(db, Document) == 0


def test_close_twice_changes_nothing(client, staff_headers, claimed, db):
    url = f"/api/capture-sessions/{claimed['id']}/close"
    client.post(url, headers=staff_headers)

    response = client.post(url, headers=staff_headers)

    assert response.status_code == 200
    assert response.json()["status"] == "closed"
    assert audit_actions(db, "capture_session", claimed["id"]).count("close") == 1


def test_close_an_expired_session_stays_expired(client, staff_headers, claimed, db):
    get_row(db, claimed["id"]).expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.flush()

    response = client.post(f"/api/capture-sessions/{claimed['id']}/close", headers=staff_headers)

    assert response.status_code == 200
    assert response.json()["status"] == "expired"
    assert "close" not in audit_actions(db, "capture_session", claimed["id"])


def test_close_keeps_the_uploaded_documents(client, staff_headers, claimed, phone, db):
    document_id = post_image(phone, claimed["token"]).json()["document_id"]

    client.post(f"/api/capture-sessions/{claimed['id']}/close", headers=staff_headers)

    body = client.get(f"/api/capture-sessions/{claimed['id']}", headers=staff_headers).json()
    assert body["status"] == "closed"
    assert [d["id"] for d in body["documents"]] == [document_id]
    assert count(db, Job) == 1


def test_close_unknown_session_is_404(client, staff_headers):
    response = client.post("/api/capture-sessions/999999/close", headers=staff_headers)

    assert_error(response, 404, "not_found")


def test_close_invalid_id_is_422(client, staff_headers):
    response = client.post("/api/capture-sessions/abc/close", headers=staff_headers)

    assert_error(response, 422, "validation_error")


# --- the phone cookie cannot read records ------------------------------------------


def _staff_routes():
    for route in iter_route_contexts(create_app().routes):
        if hasattr(route, "dependant"):
            for method in route.methods:
                if (method, route.path) not in PUBLIC_ROUTES:
                    yield method, route.path


@pytest.mark.parametrize(("method", "path"), sorted(set(_staff_routes())))
def test_phone_cookie_is_refused_on_every_staff_route(
    app, claimed, phone, make_learner, method, path
):
    make_learner()
    cookie = phone.cookies[CAPTURE_COOKIE]
    url = re.sub(r"\{\w+\}", "1", path)

    # Sent by hand: a browser would not even send it here, because of the cookie's Path.
    with TestClient(app) as attacker:
        response = attacker.request(
            method, url, headers={"Cookie": f"{CAPTURE_COOKIE}={cookie}"}, json={}
        )

    assert_error(response, 401, "not_authenticated")


def test_the_staff_route_walk_finds_the_records():
    routes = set(_staff_routes())

    assert ("GET", "/api/learners") in routes
    assert ("GET", "/api/learners/{learner_id}") in routes


@pytest.mark.parametrize("path", ["/api/learners", "/api/me", "/api/dashboard/summary"])
def test_phone_cookie_is_not_a_bearer_token(phone, claimed, path):
    cookie = phone.cookies[CAPTURE_COOKIE]

    response = phone.get(path, headers={"Authorization": f"Bearer {cookie}"})

    assert_error(response, 401, "invalid_token")


def test_capture_token_is_not_a_bearer_token(phone, claimed):
    response = phone.get("/api/learners", headers={"Authorization": f"Bearer {claimed['token']}"})

    assert_error(response, 401, "invalid_token")


def test_phone_cookie_is_not_a_refresh_token(app, phone, claimed):
    cookie = phone.cookies[CAPTURE_COOKIE]

    with TestClient(app) as attacker:
        response = attacker.post("/api/auth/refresh", headers={"Cookie": f"refresh_token={cookie}"})

    assert response.status_code == 401
    assert "access_token" not in response.text


def test_phone_responses_hold_no_personal_data(new_session, phone, db):
    created = new_session()
    owner_name = db.get(User, get_row(db, created["id"]).user_id).full_name

    claim = phone.post(f"/api/capture-sessions/{created['token']}/claim")
    upload = post_image(phone, created["token"])

    assert set(claim.json()) == {
        "reference_no",
        "status",
        "expires_at",
        "upload_count",
        "upload_limit",
        "max_file_bytes",
    }
    assert set(upload.json()) == {
        "document_id",
        "duplicate",
        "reference_no",
        "upload_count",
        "uploads_remaining",
    }
    assert owner_name not in claim.text + upload.text


def test_upload_without_a_cookie_is_401(app, claimed, db):
    with TestClient(app) as stranger:
        response = post_image(stranger, claimed["token"])

    assert_error(response, 401, "not_authenticated")
    assert count(db, Document) == 0


def test_upload_with_a_wrong_cookie_is_403(app, claimed, db):
    with TestClient(app) as stranger:
        stranger.cookies.set(CAPTURE_COOKIE, "made-up")
        response = post_image(stranger, claimed["token"])

    assert_error(response, 403, "forbidden")
    assert count(db, Document) == 0


def test_cookie_from_another_session_is_403(app, new_session, claimed, db):
    other = new_session()
    with TestClient(app) as other_phone:
        other_phone.post(f"/api/capture-sessions/{other['token']}/claim")
        cookie = other_phone.cookies[CAPTURE_COOKIE]

    with TestClient(app) as attacker:
        response = attacker.post(
            f"/api/capture-sessions/{claimed['token']}/images",
            headers={"Cookie": f"{CAPTURE_COOKIE}={cookie}"},
            data={"corners": json.dumps(CORNERS)},
            files={"file": ("photo.jpg", jpeg_bytes(), "image/jpeg")},
        )

    assert_error(response, 403, "forbidden")
    assert count(db, Document) == 0


def test_upload_to_unclaimed_session_is_401(new_session, phone):
    assert_error(post_image(phone, new_session()["token"]), 401, "not_authenticated")


def test_upload_unknown_token_is_404(phone):
    assert_error(post_image(phone, "a" * 43), 404, "not_found")


def test_staff_login_alone_cannot_upload(client, staff_headers, claimed):
    response = client.post(
        f"/api/capture-sessions/{claimed['token']}/images",
        headers=staff_headers,
        data={"corners": json.dumps(CORNERS)},
        files={"file": ("photo.jpg", jpeg_bytes(), "image/jpeg")},
    )

    assert_error(response, 401, "not_authenticated")


# --- metadata stripping ------------------------------------------------------------


def stored_bytes(phone, token, data, db, settings, mime="image/jpeg") -> bytes:
    response = post_image(phone, token, data, mime=mime)
    assert response.status_code == 201, response.text
    document = db.get(Document, response.json()["document_id"])
    stored = (settings.storage_dir / document.file_path).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == document.sha256
    return stored


@pytest.mark.parametrize("make_bytes", [jpeg_bytes, png_bytes], ids=["jpeg", "png"])
def test_gps_and_device_data_are_removed_but_orientation_stays(
    claimed, phone, db, settings, make_bytes
):
    original = make_bytes(exif=fake_exif(orientation=6))
    # Guards the test: the upload really does carry what must be removed.
    assert read_exif(original).get_ifd(GPS_IFD)
    assert read_exif(original)[MAKE] == FAKE_MAKE.decode()
    assert FAKE_MAKE in original

    stored = stored_bytes(phone, claimed["token"], original, db, settings)

    exif = read_exif(stored)
    assert exif[ORIENTATION] == 6
    assert dict(exif) == {ORIENTATION: 6}
    assert not exif.get_ifd(GPS_IFD)
    for fake_value in (FAKE_MAKE, FAKE_MODEL, FAKE_GPS_DATE):
        assert fake_value not in stored


@pytest.mark.parametrize("make_bytes", [jpeg_bytes, png_bytes], ids=["jpeg", "png"])
def test_pixels_are_not_touched(claimed, phone, db, settings, make_bytes):
    original = make_bytes(exif=fake_exif())

    stored = stored_bytes(phone, claimed["token"], original, db, settings)

    assert stored != original
    assert pixels(stored) == pixels(original)


def test_jpeg_compressed_data_is_copied_byte_for_byte():
    original = jpeg_bytes(exif=fake_exif())

    stored = images_service.prepare_image(original).data

    scan = original[original.index(b"\xff\xda") : original.rindex(b"\xff\xd9")]
    assert scan in stored


def test_png_compressed_data_is_copied_byte_for_byte():
    original = png_bytes(exif=fake_exif())

    stored = images_service.prepare_image(original).data

    idat = original[original.index(b"IDAT") : original.index(b"IEND")]
    assert idat in stored


def test_exif_thumbnail_and_other_exif_fields_are_gone():
    exif = fake_exif()
    exif[0x9003] = "2099:01:02 03:04:05"
    exif[0x0131] = "FakeSoftware"
    original = jpeg_bytes(exif=exif)

    stored = images_service.prepare_image(original).data

    assert b"FakeSoftware" not in stored
    assert b"2099:01:02 03:04:05" not in stored


@pytest.mark.parametrize("make_bytes", [jpeg_bytes, png_bytes], ids=["jpeg", "png"])
def test_no_orientation_means_no_exif_block(make_bytes):
    stored = images_service.prepare_image(make_bytes(exif=fake_exif(orientation=None))).data

    assert not read_exif(stored)
    assert FAKE_MAKE not in stored


def test_orientation_one_needs_no_exif_block():
    stored = images_service.prepare_image(jpeg_bytes(exif=fake_exif(orientation=1))).data

    assert not read_exif(stored)


def test_jpeg_xmp_comment_and_trailing_data_are_removed():
    xmp = b"<x:xmpmeta>FakeXmpLocation</x:xmpmeta>"
    original = jpeg_bytes(comment=b"FakeComment", xmp=xmp) + b"FakeTrailerImage"
    assert b"FakeXmpLocation" in original
    assert b"FakeComment" in original

    stored = images_service.prepare_image(original).data

    assert b"FakeXmpLocation" not in stored
    assert b"FakeComment" not in stored
    assert b"FakeTrailerImage" not in stored
    assert stored.endswith(b"\xff\xd9")
    assert pixels(stored) == pixels(original)


def _app_segment(marker: int, body: bytes) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", len(body) + 2) + body


def test_jpeg_unknown_app_segments_are_removed():
    original = jpeg_bytes()
    original = original[:2] + _app_segment(0xE7, b"FakeMakerNote") + original[2:]

    stored = images_service.prepare_image(original).data

    assert b"FakeMakerNote" not in stored
    assert pixels(stored) == pixels(original)


def test_jpeg_keeps_the_segments_colours_depend_on():
    icc = b"ICC_PROFILE\x00\x01\x01" + b"fake-profile"
    adobe = b"Adobe\x00\x64\x00\x00\x00\x00\x01"
    original = jpeg_bytes()
    original = original[:2] + _app_segment(0xE2, icc) + _app_segment(0xEE, adobe) + original[2:]

    stored = images_service.prepare_image(original).data

    assert _app_segment(0xE2, icc) in stored
    assert _app_segment(0xEE, adobe) in stored
    assert pixels(stored) == pixels(original)


def test_jpeg_app_segment_with_a_kept_marker_but_other_content_is_removed():
    original = jpeg_bytes()
    original = original[:2] + _app_segment(0xE2, b"MPF\x00FakeMultiPicture") + original[2:]

    assert b"FakeMultiPicture" not in images_service.prepare_image(original).data


@pytest.mark.parametrize(
    "save_args",
    [{"progressive": True}, {"subsampling": 0}, {"restart_marker_blocks": 1}],
    ids=["progressive", "no-subsampling", "restart-markers"],
)
def test_jpeg_variants_keep_their_pixels(save_args):
    original = jpeg_bytes(exif=fake_exif(), **save_args)

    stored = images_service.prepare_image(original).data

    assert FAKE_MAKE not in stored
    assert read_exif(stored)[ORIENTATION] == 6
    assert pixels(stored) == pixels(original)


def test_cmyk_jpeg_keeps_its_pixels():
    # Pillow writes CMYK JPEGs with an Adobe segment, needed to decode them.
    buffer = io.BytesIO()
    noise_image().convert("CMYK").save(buffer, format="JPEG")
    original = buffer.getvalue()

    stored = images_service.prepare_image(original).data

    assert pixels(stored) == pixels(original)


def test_png_text_and_trailing_data_are_removed():
    info = PngImagePlugin.PngInfo()
    info.add_text("Location", "FakePngLocation")
    info.add_itxt("Comment", "FakePngComment")
    info.add_text("Compressed", "FakeZipped", zip=True)
    original = png_bytes(pnginfo=info) + b"FakeTrailer"

    stored = images_service.prepare_image(original).data

    for fake_value in (b"FakePngLocation", b"FakePngComment", b"Compressed", b"FakeTrailer"):
        assert fake_value not in stored
    assert stored.endswith(b"IEND\xaeB`\x82")
    assert pixels(stored) == pixels(original)


def test_png_rebuilt_exif_chunk_has_a_valid_crc():
    stored = images_service.prepare_image(png_bytes(exif=fake_exif())).data

    start = stored.index(b"eXIf")
    length = struct.unpack(">I", stored[start - 4 : start])[0]
    data_end = start + 4 + length
    assert struct.unpack(">I", stored[data_end : data_end + 4])[0] == zlib.crc32(
        stored[start:data_end]
    )


def test_png_with_transparency_keeps_its_pixels():
    buffer = io.BytesIO()
    noise_image().convert("RGBA").save(buffer, format="PNG")
    original = buffer.getvalue()

    stored = images_service.prepare_image(original).data

    with Image.open(io.BytesIO(stored)) as after, Image.open(io.BytesIO(original)) as before:
        assert after.convert("RGBA").tobytes() == before.convert("RGBA").tobytes()


def test_same_upload_twice_is_a_duplicate_after_stripping(claimed, phone, db):
    original = jpeg_bytes(exif=fake_exif())

    assert post_image(phone, claimed["token"], original).status_code == 201
    assert post_image(phone, claimed["token"], original).status_code == 200
    assert count(db, Document) == 1


def test_same_pixels_with_different_metadata_share_one_document(claimed, phone, db):
    # Stripping makes these the same stored file, so the hash treats them as one photo.
    first = jpeg_bytes(exif=fake_exif())
    exif = fake_exif()
    exif[MODEL] = "AnotherFakeModel"
    second = jpeg_bytes(exif=exif)

    assert post_image(phone, claimed["token"], first).status_code == 201
    assert post_image(phone, claimed["token"], second).status_code == 200
    assert count(db, Document) == 1


def test_stored_exif_is_readable_by_the_ocr_steps_exif_rotation():
    stored = images_service.prepare_image(jpeg_bytes(exif=fake_exif(orientation=6))).data

    with Image.open(io.BytesIO(stored)) as image:
        assert ImageOps.exif_transpose(image).size == (32, 48)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda data: data[: data.rindex(b"\xff\xd9")],  # no end marker
        lambda data: data.replace(b"\xff\xda", b"\xff\x02", 1),  # unknown marker, no scan
    ],
    ids=["no-eoi", "unknown-marker"],
)
def test_jpeg_that_cannot_be_walked_is_refused_not_stored(claimed, phone, db, corrupt):
    response = post_image(phone, claimed["token"], corrupt(jpeg_bytes(exif=fake_exif())))

    assert_error(response, 415, "unsupported_image_type")
    assert count(db, Document) == 0


def test_truncated_png_is_refused(claimed, phone, db):
    data = png_bytes()

    response = post_image(phone, claimed["token"], data[: len(data) // 2])

    assert_error(response, 415, "unsupported_image_type")
    assert count(db, Document) == 0


# --- phone page --------------------------------------------------------------------


def test_phone_page_is_served_with_its_headers(client):
    response = client.get(f"/m/{'a' * 43}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<input id="file-input" type="file" accept="image/*" capture="environment">' in (
        response.text
    )
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "connect-src 'self'" in csp


def test_phone_page_works_offline(client):
    # No CDN, font or analytics: the host PC has no internet.
    page = client.get(f"/m/{'a' * 43}").text

    assert "http://" not in page
    assert "https://" not in page
    assert "<link" not in page
    assert "src=" not in page.replace(".src = photoUrl", "")


def test_phone_page_needs_no_database(settings):
    app = create_app(settings)

    def no_database():
        raise AssertionError("the page used the database")

    app.dependency_overrides[get_db] = no_database
    with TestClient(app) as client:
        assert client.get(f"/m/{'a' * 43}").status_code == 200


def test_phone_page_does_not_contain_the_token(client):
    token = "b" * 43

    assert token not in client.get(f"/m/{token}").text


def test_phone_page_explains_every_upload_error(client):
    page = client.get(f"/m/{'a' * 43}").text

    for code in (
        "already_claimed",
        "session_expired",
        "session_closed",
        "upload_limit_reached",
        "file_too_large",
        "unsupported_image_type",
        "heic_conversion_failed",
        "storage_full",
    ):
        assert code in page


# --- logs and audit ----------------------------------------------------------------


def access_record(path):
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "POST", path, "1.1", 200),
        None,
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (f"/m/{'t' * 43}", "/m/<token>"),
        (f"/api/capture-sessions/{'t' * 43}/claim", "/api/capture-sessions/<token>/claim"),
        (f"/api/capture-sessions/{'t' * 43}/images", "/api/capture-sessions/<token>/images"),
        ("/api/capture-sessions/12", "/api/capture-sessions/12"),
        ("/api/capture-sessions/12/close", "/api/capture-sessions/12/close"),
        ("/api/capture-sessions", "/api/capture-sessions"),
        ("/api/learners/12", "/api/learners/12"),
    ],
)
def test_access_log_redacts_capture_tokens(path, expected):
    record = access_record(path)

    assert RedactCaptureToken().filter(record) is True
    assert record.getMessage() == f'127.0.0.1:5000 - "POST {expected} HTTP/1.1" 200'


def test_create_app_installs_the_token_filter_once():
    create_app()
    create_app()

    filters = [
        f for f in logging.getLogger("uvicorn.access").filters if isinstance(f, RedactCaptureToken)
    ]
    assert len(filters) == 1


def test_audit_log_never_holds_tokens(client, staff_headers, claimed, phone, db):
    cookie = phone.cookies[CAPTURE_COOKIE]
    post_image(phone, claimed["token"])
    post_image(phone, claimed["token"])
    client.post(f"/api/capture-sessions/{claimed['id']}/close", headers=staff_headers)

    rows = db.scalars(select(AuditLog)).all()
    text = json.dumps([[r.action, r.entity, r.before, r.after] for r in rows])

    assert len(rows) >= 5
    for secret in (
        claimed["token"],
        hash_token(claimed["token"]),
        cookie,
        hash_token(cookie),
    ):
        assert secret not in text
