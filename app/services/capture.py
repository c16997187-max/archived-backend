"""QR pairing between a desktop session and one phone, and the phone's uploads (Phase 6)."""

import hmac
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import ColumnElement, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.errors import ApiError
from app.models import CaptureSession, Document, Job, User
from app.schemas.capture import (
    CapturedDocument,
    CaptureSessionCreated,
    CaptureSessionStatus,
    CropCorners,
)
from app.services import audit, images
from app.services.db_errors import constraint_name
from app.services.tokens import hash_token

UPLOAD_LIMIT = 50
# Set on creation and pushed forward by phone activity, so it is both the lifetime
# of an unused QR code and the idle limit. Desktop polling does not extend it.
IDLE_LIFETIME = timedelta(minutes=10)
TOKEN_BYTES = 32
# No 0/O or 1/I: staff read this out and type it.
_REFERENCE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
_SHA256_CONSTRAINT = "uq_documents_sha256"
_ACTIVE_STATUSES = ("open", "claimed")


def _now() -> datetime:
    return datetime.now(UTC)


def _new_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(TOKEN_BYTES)
    return raw, hash_token(raw)


def _new_reference_no(now: datetime) -> str:
    suffix = "".join(secrets.choice(_REFERENCE_ALPHABET) for _ in range(5))
    return f"CS-{now:%y%m%d}-{suffix}"


def _locked(session: Session, condition: ColumnElement[bool]) -> CaptureSession:
    # populate_existing: this request may have loaded the row before it read the
    # upload, and the decision must use what is true now that the row is locked.
    capture = session.scalar(
        select(CaptureSession)
        .where(condition)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if capture is None:
        raise ApiError(404, "not_found")
    return capture


def _expire_if_due(session: Session, capture: CaptureSession, ip: str | None) -> None:
    """Sessions expire on the first access after their time is up; nothing runs on a timer."""
    if capture.status not in _ACTIVE_STATUSES or capture.expires_at > _now():
        return
    before = capture.status
    capture.status = "expired"
    capture.phone_token_hash = None
    audit.record(
        session,
        "expire",
        "capture_session",
        capture.id,
        None,
        before={"status": before},
        after={"status": "expired"},
        ip_address=ip,
    )
    session.commit()


def _ensure_active(session: Session, capture: CaptureSession, ip: str | None) -> None:
    _expire_if_due(session, capture, ip)
    if capture.status == "closed":
        raise ApiError(410, "session_closed")
    if capture.status == "expired":
        raise ApiError(410, "session_expired")
    owner = session.get(User, capture.user_id)
    if owner is None or not owner.is_active:
        raise ApiError(410, "session_closed")


def _ensure_room(capture: CaptureSession) -> None:
    if capture.upload_count >= UPLOAD_LIMIT:
        raise ApiError(
            409,
            "upload_limit_reached",
            {"file": f"This session is full ({UPLOAD_LIMIT} photos). Start a new one."},
        )


def _cookie_matches(capture: CaptureSession, cookie: str | None) -> bool:
    return bool(
        cookie
        and capture.phone_token_hash
        and hmac.compare_digest(hash_token(cookie), capture.phone_token_hash)
    )


def _status_of(session: Session, capture: CaptureSession) -> CaptureSessionStatus:
    documents = session.scalars(
        select(Document).where(Document.capture_session_id == capture.id).order_by(Document.id)
    )
    return CaptureSessionStatus(
        id=capture.id,
        reference_no=capture.reference_no,
        status=capture.status,
        expires_at=capture.expires_at,
        upload_count=capture.upload_count,
        upload_limit=UPLOAD_LIMIT,
        duplicate_count=capture.duplicate_count,
        documents=[CapturedDocument.model_validate(d) for d in documents],
    )


def create_session(
    session: Session, actor: User, public_base_url: str, ip: str | None
) -> CaptureSessionCreated:
    now = _now()
    token, token_hash = _new_token()
    capture = CaptureSession(
        token_hash=token_hash,
        user_id=actor.id,
        reference_no=_new_reference_no(now),
        status="open",
        expires_at=now + IDLE_LIFETIME,
    )
    session.add(capture)
    session.flush()
    audit.record(
        session,
        "create",
        "capture_session",
        capture.id,
        actor.id,
        after={"reference_no": capture.reference_no, "status": "open"},
        ip_address=ip,
    )
    session.commit()
    return CaptureSessionCreated(
        id=capture.id,
        token=token,
        capture_url=f"{public_base_url.rstrip('/')}/m/{token}",
        reference_no=capture.reference_no,
        status=capture.status,
        expires_at=capture.expires_at,
    )


def get_status(session: Session, session_id: int, ip: str | None) -> CaptureSessionStatus:
    capture = session.get(CaptureSession, session_id)
    if capture is None:
        raise ApiError(404, "not_found")
    if capture.status in _ACTIVE_STATUSES and capture.expires_at <= _now():
        capture = _locked(session, CaptureSession.id == session_id)
        _expire_if_due(session, capture, ip)
    return _status_of(session, capture)


def close_session(
    session: Session, actor: User, session_id: int, ip: str | None
) -> CaptureSessionStatus:
    capture = _locked(session, CaptureSession.id == session_id)
    _expire_if_due(session, capture, ip)
    if capture.status in _ACTIVE_STATUSES:
        before = capture.status
        capture.status = "closed"
        # The desktop cannot delete a cookie on the phone, so it stops matching instead.
        capture.phone_token_hash = None
        audit.record(
            session,
            "close",
            "capture_session",
            capture.id,
            actor.id,
            before={"status": before},
            after={"status": "closed"},
            ip_address=ip,
        )
        session.commit()
    return _status_of(session, capture)


def claim(
    session: Session, token: str, cookie: str | None, ip: str | None
) -> tuple[CaptureSession, str | None]:
    """Pair a phone. Returns the session and the new cookie value (None when the same
    phone comes back: phones often reload the page after the camera app closes)."""
    # The row lock is what stops two phones claiming at once.
    capture = _locked(session, CaptureSession.token_hash == hash_token(token))
    _ensure_active(session, capture, ip)

    if capture.status == "claimed":
        if not _cookie_matches(capture, cookie):
            audit.record(
                session,
                "claim_rejected",
                "capture_session",
                capture.id,
                capture.user_id,
                ip_address=ip,
            )
            session.commit()
            raise ApiError(409, "already_claimed")
        capture.expires_at = _now() + IDLE_LIFETIME
        session.commit()
        return capture, None

    raw_cookie, cookie_hash = _new_token()
    capture.phone_token_hash = cookie_hash
    capture.status = "claimed"
    capture.expires_at = _now() + IDLE_LIFETIME
    audit.record(
        session,
        "claim",
        "capture_session",
        capture.id,
        capture.user_id,
        before={"status": "open"},
        after={"status": "claimed"},
        ip_address=ip,
    )
    session.commit()
    return capture, raw_cookie


def authorize_upload(
    session: Session, token: str, cookie: str | None, ip: str | None
) -> CaptureSession:
    """Everything that can be refused before reading the photo off the network.

    Status comes before the cookie, so a closed session says so instead of 403.
    """
    capture = _locked(session, CaptureSession.token_hash == hash_token(token))
    _ensure_active(session, capture, ip)
    if not cookie:
        raise ApiError(401, "not_authenticated")
    if not _cookie_matches(capture, cookie):
        raise ApiError(403, "forbidden")
    _ensure_room(capture)
    # Ends the transaction: a slow phone must not hold the lock or a connection.
    session.commit()
    return capture


def _find_document(session: Session, sha256: str) -> int | None:
    return session.scalar(select(Document.id).where(Document.sha256 == sha256))


def _insert_document(
    session: Session,
    capture: CaptureSession,
    image: images.StoredImage,
    file_path: str,
    corners: CropCorners,
) -> int | None:
    """The new document's ID, or None if the same hash was saved first by someone else."""
    document = Document(
        file_path=file_path,
        sha256=image.sha256,
        mime_type=image.mime_type,
        source="phone",
        uploaded_by=capture.user_id,
        capture_session_id=capture.id,
        crop_corners=corners.model_dump(),
    )
    try:
        with session.begin_nested():
            session.add(document)
            session.flush()
    except IntegrityError as exc:
        if constraint_name(exc) != _SHA256_CONSTRAINT:
            raise
        return None
    return document.id


def save_upload(
    session: Session,
    session_id: int,
    raw: bytes,
    corners: CropCorners,
    storage_dir: Path,
    ip: str | None,
) -> tuple[CaptureSession, int, bool]:
    """Store one photo. Returns (session, document ID, was it a duplicate). The commit
    happens only after the file is on disk, so "saved" is never said too early."""
    image = images.prepare_image(raw)

    capture = _locked(session, CaptureSession.id == session_id)
    # Checked again, now the row is locked: the desktop may have closed it meanwhile.
    _ensure_active(session, capture, ip)
    _ensure_room(capture)

    file_path = images.store_image(storage_dir, image)
    document_id = _find_document(session, image.sha256)
    duplicate = document_id is not None
    if not duplicate:
        document_id = _insert_document(session, capture, image, file_path, corners)
        if document_id is None:  # another phone saved the same photo a moment ago
            document_id = _find_document(session, image.sha256)
            duplicate = True

    if duplicate:
        capture.duplicate_count += 1
    else:
        session.add(Job(type="ocr", payload={"document_id": document_id}))
    audit.record(
        session,
        "duplicate_upload" if duplicate else "create",
        "document",
        document_id,
        capture.user_id,
        after={
            "capture_session_id": capture.id,
            "sha256": image.sha256,
            "mime_type": image.mime_type,
            "source": "phone",
        },
        ip_address=ip,
    )
    capture.upload_count += 1
    capture.expires_at = _now() + IDLE_LIFETIME
    session.commit()
    return capture, document_id, duplicate
