"""Serve the AI dashboard page."""

from fastapi.responses import FileResponse, HTMLResponse

from .._paths import PAGES_DIR
from .._router import ui_router


@ui_router.get("/", response_class=HTMLResponse)
async def get_dashboard() -> FileResponse:
    """Return the dashboard's packaged HTML.

    :returns: The dashboard document served as HTML.
    """
    return FileResponse(PAGES_DIR / "dashboard.html", media_type="text/html")
