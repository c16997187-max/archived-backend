from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models._common import created_at, one_of, pk

DOCUMENT_SOURCES = ("phone", "upload", "batch")
EXTRACTION_STATUSES = ("queued", "processing", "needs_review", "done", "failed")
CAPTURE_SESSION_STATUSES = ("open", "claimed", "closed", "expired")


class CaptureSession(Base):
    """QR pairing between a desktop session and one phone (Phase 6)."""

    __tablename__ = "capture_sessions"
    __table_args__ = (
        UniqueConstraint("token_hash"),
        UniqueConstraint("reference_no"),
        one_of("status", CAPTURE_SESSION_STATUSES, "status"),
    )

    id: Mapped[int] = pk()
    token_hash: Mapped[str]  # only the hash; the raw token lives in the QR code
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    reference_no: Mapped[str]
    status: Mapped[str] = mapped_column(server_default=text("'open'"))
    expires_at: Mapped[datetime]
    created_at: Mapped[datetime] = created_at()


class Document(Base):
    """Every uploaded image. Files are stored by SHA-256, so a re-sent photo is a duplicate."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("sha256"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_hex"),
        one_of("source", DOCUMENT_SOURCES, "source"),
    )

    id: Mapped[int] = pk()
    file_path: Mapped[str]
    sha256: Mapped[str] = mapped_column(String(64))
    mime_type: Mapped[str]
    source: Mapped[str]
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    capture_session_id: Mapped[int | None] = mapped_column(ForeignKey("capture_sessions.id"))
    created_at: Mapped[datetime] = created_at()


class FormTemplate(Base):
    """Layout of one paper form version: where each field sits on the page."""

    __tablename__ = "form_templates"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = pk()
    name: Mapped[str]
    version: Mapped[int]
    field_boxes: Mapped[dict[str, Any]]
    is_active: Mapped[bool] = mapped_column(server_default=text("true"))


class Extraction(Base):
    """One OCR run on one document."""

    __tablename__ = "extractions"
    __table_args__ = (one_of("status", EXTRACTION_STATUSES, "status"),)

    id: Mapped[int] = pk()
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    enrollment_id: Mapped[int | None] = mapped_column(ForeignKey("enrollments.id"), index=True)
    status: Mapped[str] = mapped_column(server_default=text("'queued'"))
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    error: Mapped[str | None]


class OcrResult(Base):
    """One recognized field on one document."""

    __tablename__ = "ocr_results"
    __table_args__ = (CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence_0_to_1"),)

    id: Mapped[int] = pk()
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    template_id: Mapped[int | None] = mapped_column(ForeignKey("form_templates.id"))
    field_name: Mapped[str]
    raw_text: Mapped[str | None]
    normalized_text: Mapped[str | None]
    confidence: Mapped[float | None]
    bbox: Mapped[dict[str, Any] | None]  # {"x", "y", "w", "h"} in template pixels
    model_version: Mapped[str]
