"""Serve the shared browser client for the model API."""

from fastapi.responses import FileResponse

from .._paths import SCRIPTS_DIR
from .._router import ui_router


@ui_router.get("/assets/model-api.js", response_class=FileResponse)
async def get_model_api_asset() -> FileResponse:
    """Return the packaged browser API client.

    :returns: The client served as a JavaScript module.
    """
    return FileResponse(
        SCRIPTS_DIR / "fetch" / "model-api.js", media_type="text/javascript"
    )
