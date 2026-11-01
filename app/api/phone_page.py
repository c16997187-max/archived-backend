from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["capture"])

PAGE = Path(__file__).resolve().parent.parent / "static" / "phone_capture.html"
# The token is in the page's URL: never send it to another site or leave it in a cache.
PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}


# Public on purpose: the same static page for any token, with no database lookup and
# no data in it. The page reads the token from its own URL and claims the session.
@router.get("/m/{token}", include_in_schema=False)
def phone_capture_page(token: str) -> FileResponse:
    return FileResponse(PAGE, media_type="text/html", headers=PAGE_HEADERS)
