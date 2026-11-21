"""Serve the standalone AI load test page."""

from fastapi.responses import FileResponse, HTMLResponse

from .._paths import PAGES_DIR
from .._router import ui_router


@ui_router.get("/tests/load", response_class=HTMLResponse)
async def get_load_test_page() -> FileResponse:
    """Return the packaged load test page.

    :returns: The load test document served as HTML.
    """
    return FileResponse(PAGES_DIR / "index.html", media_type="text/html")
