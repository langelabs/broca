"""List the models configured for inference."""

import asyncio

from ....inference import MODELS
from .._response import ModelListResponse, ModelResponse
from .._router import models_router


@models_router.get("")
async def list_models() -> ModelListResponse:
    """Return all configured models from the inference catalog.

    :returns: Public metadata and lifecycle flags for the models in catalog order.
    """
    models = await asyncio.gather(
        *(ModelResponse.from_model(model) for model in MODELS)
    )
    return ModelListResponse(data=models)
