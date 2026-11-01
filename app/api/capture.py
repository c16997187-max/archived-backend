from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Path, Request, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

from app.api.deps import AppSettings, DbSession, client_ip, require_role
from app.config import Settings
from app.errors import ApiError
from app.models import CaptureSession, User
from app.schemas.capture import (
    CaptureClaimed,
    CaptureSessionCreated,
    CaptureSessionStatus,
    CaptureUploaded,
    CropCorners,
)
from app.schemas.common import DbId
from app.schemas.errors import error_responses
from app.services import capture as capture_service
from app.services.images import MAX_FILE_BYTES

router = APIRouter(prefix="/capture-sessions", tags=["capture"])

CAPTURE_COOKIE = "capture_upload"
# Room for the multipart boundaries and the corners field around the photo.
MAX_FORM_OVERHEAD_BYTES = 64 * 1024
MAX_FORM_FIELD_BYTES = 4 * 1024

# secrets.token_urlsafe(32) is always 43 characters.
CaptureToken = Annotated[str, Path(min_length=43, max_length=43)]
CaptureCookie = Annotated[str | None, Cookie(alias=CAPTURE_COOKIE)]
Staff = Annotated[User, Depends(require_role("registrar", "admin"))]

_STAFF_ERRORS = error_responses(401, 403)

# claim and images take no require_role: a phone has no account. The unguessable
# token and, for images, the cookie from claim stand in for it. Neither returns
# student data, and the cookie is not accepted anywhere else.


@router.post("", response_model=CaptureSessionCreated, status_code=201, responses=_STAFF_ERRORS)
def create_capture_session(
    request: Request, user: Staff, db: DbSession, settings: AppSettings
) -> CaptureSessionCreated:
    base_url = settings.public_base_url or str(request.base_url)
    return capture_service.create_session(db, user, base_url, client_ip(request))


@router.get(
    "/{session_id}",
    response_model=CaptureSessionStatus,
    responses={**_STAFF_ERRORS, **error_responses(404, 422)},
)
def get_capture_session(
    session_id: DbId, request: Request, _: Staff, db: DbSession
) -> CaptureSessionStatus:
    return capture_service.get_status(db, session_id, client_ip(request))


@router.post(
    "/{session_id}/close",
    response_model=CaptureSessionStatus,
    responses={**_STAFF_ERRORS, **error_responses(404, 422)},
)
def close_capture_session(
    session_id: DbId, request: Request, user: Staff, db: DbSession
) -> CaptureSessionStatus:
    return capture_service.close_session(db, user, session_id, client_ip(request))


def _set_capture_cookie(response: Response, value: str, token: str, settings: Settings) -> None:
    response.set_cookie(
        CAPTURE_COOKIE,
        value,
        # Browsers send it only to this session's phone endpoints.
        path=f"/api/capture-sessions/{token}",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
    )


@router.post(
    "/{token}/claim",
    response_model=CaptureClaimed,
    responses=error_responses(404, 409, 410, 422),
)
def claim_capture_session(
    token: CaptureToken,
    request: Request,
    response: Response,
    db: DbSession,
    settings: AppSettings,
    capture_upload: CaptureCookie = None,
) -> CaptureClaimed:
    capture, cookie = capture_service.claim(db, token, capture_upload, client_ip(request))
    if cookie:
        _set_capture_cookie(response, cookie, token, settings)
    return CaptureClaimed(
        reference_no=capture.reference_no,
        status=capture.status,
        expires_at=capture.expires_at,
        upload_count=capture.upload_count,
        upload_limit=capture_service.UPLOAD_LIMIT,
        max_file_bytes=MAX_FILE_BYTES,
    )


def _authorized_phone(
    token: CaptureToken,
    request: Request,
    db: DbSession,
    capture_upload: CaptureCookie = None,
) -> CaptureSession:
    return capture_service.authorize_upload(db, token, capture_upload, client_ip(request))


def _check_content_length(request: Request) -> None:
    # HTTP framing means the server reads no more than this many bytes of body.
    declared = request.headers.get("content-length")
    if declared is None or not declared.isdigit():
        raise ApiError(411, "length_required")
    if int(declared) > MAX_FILE_BYTES + MAX_FORM_OVERHEAD_BYTES:
        raise ApiError(413, "file_too_large", {"file": "Must be 15 MB or smaller"})


async def _read_upload(request: Request) -> tuple[bytes, CropCorners]:
    try:
        # Closing the form deletes the temp file holding the photo.
        async with request.form(
            max_files=1, max_fields=2, max_part_size=MAX_FORM_FIELD_BYTES
        ) as form:
            upload = form.get("file")
            corners = form.get("corners")
            if not isinstance(upload, UploadFile):
                raise ApiError(422, "validation_error", {"file": "Attach a photo"})
            raw = await upload.read(MAX_FILE_BYTES + 1)
    except HTTPException:  # Starlette's error for a malformed multipart body
        raise ApiError(422, "validation_error", {"file": "Could not read the upload"}) from None

    if not isinstance(corners, str):
        raise ApiError(422, "validation_error", {"corners": "Send the four crop corners"})
    try:
        return raw, CropCorners.model_validate_json(corners)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"].removeprefix("Value error, ")
        raise ApiError(422, "validation_error", {"corners": message}) from None


@router.post(
    "/{token}/images",
    response_model=CaptureUploaded,
    status_code=201,
    responses={
        200: {"model": CaptureUploaded, "description": "Already saved; nothing queued again"},
        **error_responses(401, 403, 404, 409, 410, 411, 413, 415, 422, 507),
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file", "corners"],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "corners": {
                                "type": "string",
                                "description": "JSON of the four corners: top_left, top_right, "
                                "bottom_right, bottom_left, each {x, y} from 0 to 1.",
                            },
                        },
                    }
                }
            },
        }
    },
)
async def upload_capture_image(
    capture: Annotated[CaptureSession, Depends(_authorized_phone)],
    request: Request,
    response: Response,
    db: DbSession,
    settings: AppSettings,
) -> CaptureUploaded:
    # The form is read here, not with File(): FastAPI parses File() bodies before it
    # runs dependencies, so anyone could fill the disk before the cookie was checked.
    _check_content_length(request)
    raw, corners = await _read_upload(request)
    capture, document_id, duplicate = await run_in_threadpool(
        capture_service.save_upload,
        db,
        capture.id,
        raw,
        corners,
        settings.storage_dir,
        client_ip(request),
    )
    if duplicate:
        response.status_code = 200
    return CaptureUploaded(
        document_id=document_id,
        duplicate=duplicate,
        reference_no=capture.reference_no,
        upload_count=capture.upload_count,
        uploads_remaining=capture_service.UPLOAD_LIMIT - capture.upload_count,
    )
