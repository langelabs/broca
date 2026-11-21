"""Retrieve an individual configured model."""

from typing import Annotated

from fastapi import Depends, status
from fastapi.responses import JSONResponse

from .._catalog import find_model
from .._request import ModelRequest
from .._response import ModelErrorResponse, ModelResponse
from .._router import models_router


@models_router.get(
    "/{model:path}",
    response_model=ModelResponse,
    responses={status.HTTP_404_NOT_FOUND: {"model": ModelErrorResponse}},
)
async def retrieve_model(
    request: Annotated[ModelRequest, Depends()],
) -> ModelResponse | JSONResponse:
    """Retrieve public metadata for the requested catalog ID.

    :param request: Model identifier from the request path.
    :returns: Model metadata and lifecycle flags or an HTTP 404 error response.
    """
    model = find_model(request.model)
    if isinstance(model, JSONResponse):
        return model
    return await ModelResponse.from_model(model)
