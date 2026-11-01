"""Import every model here so Alembic sees all tables on Base.metadata."""

from app.models.audit import AuditLog
from app.models.auth import RefreshToken
from app.models.capture import CaptureSession, Document, Extraction, FormTemplate, OcrResult
from app.models.enrollment import Enrollment
from app.models.job import Job
from app.models.learner import Learner
from app.models.school import SchoolYear, Section
from app.models.user import User

__all__ = [
    "AuditLog",
    "CaptureSession",
    "Document",
    "Enrollment",
    "Extraction",
    "FormTemplate",
    "Job",
    "Learner",
    "OcrResult",
    "RefreshToken",
    "SchoolYear",
    "Section",
    "User",
]
