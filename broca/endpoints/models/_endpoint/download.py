"""Download the pinned assets for a configured model."""

import logging
from typing import Annotated

from fastapi import Path, status
from fastapi.responses import JSONResponse

from .._catalog import find_model
from .._response import ModelDownloadResponse, ModelError, ModelErrorResponse
from .._router import models_router

logger = logging.getLogger(__name__)


@models_router.post(
    "/{model_id:path}/download",
    response_model=ModelDownloadResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ModelErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ModelErrorResponse},
    },
)
async def download_model(
    model_id: Annotated[str, Path(min_length=1)],
) -> ModelDownloadResponse | JSONResponse:
    """Wait until the selected model's assets are available on disk.

    :param model_id: Namespaced configured model ID.
    :returns: Download state or a public error response.
    """
    model = find_model(model_id)
    if isinstance(model, JSONResponse):
        return model
    try:
        return ModelDownloadResponse(id=model.id, downloaded=await model.download())
    except Exception as error:
        logger.error("Download failed for %s (%s)", model.id, type(error).__name__)
        return ModelErrorResponse(
            error=ModelError(
                message=f"Could not download model '{model.id}'.",
                type="server_error",
                code="model_download_failed",
            )
        ).to_response(status.HTTP_503_SERVICE_UNAVAILABLE)
