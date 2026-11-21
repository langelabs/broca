"""Serve the browser modules packaged with the load test page."""

from typing import get_args

from fastapi import HTTPException
from fastapi.responses import FileResponse

from ....ai_model import ModelModality
from .._paths import SCRIPTS_DIR
from .._router import ui_router

ASSETS = frozenset(
    ("load-test.js", *(f"{modality}.js" for modality in get_args(ModelModality)))
)


@ui_router.get("/assets/load-tests/{filename}", response_class=FileResponse)
async def get_load_test_asset(filename: str) -> FileResponse:
    """Return a load-test JavaScript module.

    :param filename: Name of a public load-test module.
    :returns: The packaged JavaScript file.
    :raises HTTPException: If the requested filename is outside the public modules.
    """
    if filename not in ASSETS:
        raise HTTPException(status_code=404, detail="Load test asset not found.")
    return FileResponse(
        SCRIPTS_DIR / "test" / filename,
        media_type="text/javascript",
    )
