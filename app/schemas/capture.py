from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Anything smaller than this cannot be a photographed page (1% of the photo).
MIN_CROP_AREA = 0.01

Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class CropPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: Fraction
    y: Fraction


class CropCorners(BaseModel):
    """The four corners of the form, as fractions of the EXIF-oriented photo, which
    is what the browser shows and what the OCR pipeline sees after its EXIF step."""

    model_config = ConfigDict(extra="forbid")

    top_left: CropPoint
    top_right: CropPoint
    bottom_right: CropPoint
    bottom_left: CropPoint

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        points = [self.top_left, self.top_right, self.bottom_right, self.bottom_left]
        count = len(points)
        turns = []
        area = 0.0
        for i, a in enumerate(points):
            b, c = points[(i + 1) % count], points[(i + 2) % count]
            turns.append((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x))
            area += a.x * b.y - b.x * a.y
        # With y pointing down, clockwise order turns the same way at every corner.
        if not all(turn > 0 for turn in turns):
            raise ValueError("Corners must form a four-sided shape, in clockwise order")
        if abs(area) / 2 < MIN_CROP_AREA:
            raise ValueError("The cropped area is too small")
        return self


class CaptureSessionCreated(BaseModel):
    id: int
    token: str = Field(description="Shown once, inside the QR code. Only its hash is stored.")
    capture_url: str = Field(description="Put this in the QR code.")
    reference_no: str
    status: str
    expires_at: datetime


class CapturedDocument(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    mime_type: str
    created_at: datetime


class CaptureSessionStatus(BaseModel):
    id: int
    reference_no: str
    status: str = Field(description="open, claimed, closed or expired")
    expires_at: datetime
    upload_count: int = Field(description="Accepted uploads, duplicates included")
    upload_limit: int
    duplicate_count: int
    documents: list[CapturedDocument]


class CaptureClaimed(BaseModel):
    reference_no: str
    status: str
    expires_at: datetime
    upload_count: int
    upload_limit: int
    max_file_bytes: int


class CaptureUploaded(BaseModel):
    document_id: int
    duplicate: bool = Field(description="True if this photo was already saved")
    reference_no: str
    upload_count: int
    uploads_remaining: int
