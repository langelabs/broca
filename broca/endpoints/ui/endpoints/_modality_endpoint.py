"""Serve the model test page for each configured modality."""

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ....inference import MODELS
from .._paths import PAGES_DIR
from .._router import ui_router


@ui_router.get("/{modality}", response_class=HTMLResponse)
async def get_modality_page(modality: str) -> FileResponse:
    """Serve the test form for a supported inference modality.

    :param modality: Public inference modality from the page URL.
    :returns: The shared HTML used to render that modality's test form.
    :raises HTTPException: If the modality has no configured model.
    """
    if not any(model.type == modality for model in MODELS):
        raise HTTPException(status_code=404, detail="Modality not found.")
    return FileResponse(PAGES_DIR / "_base.html", media_type="text/html")
