"""Change the loaded state of a configured inference runtime."""

import asyncio
import logging
from typing import Annotated

from fastapi import Path, status
from fastapi.responses import JSONResponse

from ....inference import get_pipeline
from .._catalog import find_model
from .._request import ModelLoadedRequest
from .._response import ModelError, ModelErrorResponse, ModelLoadedResponse
from .._router import models_router

logger = logging.getLogger(__name__)


@models_router.put(
    "/{model_id:path}/loaded",
    response_model=ModelLoadedResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ModelErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ModelErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ModelErrorResponse},
    },
)
async def set_model_loaded(
    model_id: Annotated[str, Path(min_length=1)],
    request: ModelLoadedRequest,
) -> ModelLoadedResponse | JSONResponse:
    """Load or unload the shared pipeline in this server process.

    :param model_id: Namespaced configured model ID.
    :param request: True to load the runtime, or false to unload it.
    :returns: Runtime state or a public error response.
    """
    model = find_model(model_id)
    if isinstance(model, JSONResponse):
        return model
    try:
        pipeline = await asyncio.to_thread(get_pipeline, model)
        if request.root:
            loaded = await asyncio.to_thread(pipeline.load)
        else:
            loaded = not await asyncio.to_thread(pipeline.unload)
        return ModelLoadedResponse(id=model.id, loaded=loaded)
    except FileNotFoundError:
        return ModelErrorResponse(
            error=ModelError(
                message=f"Download the assets for model '{model.id}' before loading it.",
                code="model_assets_missing",
            )
        ).to_response(status.HTTP_409_CONFLICT)
    except Exception as error:
        logger.error(
            "Could not change the loaded state for %s (%s)",
            model.id,
            type(error).__name__,
        )
        return ModelErrorResponse(
            error=ModelError(
                message=f"Could not change the loaded state for model '{model.id}'.",
                type="server_error",
                code="model_runtime_failed",
            )
        ).to_response(status.HTTP_503_SERVICE_UNAVAILABLE)
