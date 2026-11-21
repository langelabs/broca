"""Shared model lookup for catalog and lifecycle routes."""

from fastapi import status
from fastapi.responses import JSONResponse

from ...ai_model import AiModel
from ...inference import get_model
from ._response import ModelError, ModelErrorResponse


def find_model(model_id: str) -> AiModel | JSONResponse:
    """Resolve a catalog ID or return the public model-not-found error.

    :param model_id: Namespaced configured model ID.
    :returns: Model configuration or the HTTP 404 response.
    """
    model = get_model(model_id)
    if model is not None:
        return model
    return ModelErrorResponse(
        error=ModelError(
            message=f"Model '{model_id}' cannot be found.", code="model_not_found"
        )
    ).to_response(status.HTTP_404_NOT_FOUND)
