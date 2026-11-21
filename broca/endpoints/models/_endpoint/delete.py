"""Handle fine-tuned model deletion requests against the configured catalog."""

from typing import Annotated

from fastapi import Depends, status
from fastapi.responses import JSONResponse
from .._request import ModelRequest
from .._response import ModelDeletedResponse, ModelError, ModelErrorResponse
from .._router import models_router


@models_router.delete(
    "/{model:path}",
    response_model=ModelDeletedResponse,
    responses={
        status.HTTP_403_FORBIDDEN: {"model": ModelErrorResponse},
    },
)
async def delete_model(request: Annotated[ModelRequest, Depends()]) -> JSONResponse:
    """Reject deletion of configured models from the read-only catalog.

    :param request: Model identifier from the request path.
    :returns: HTTP 403 for a configured model or HTTP 404 for an unknown ID.
    """
    return ModelErrorResponse(
        error=ModelError(
            message=f"Model '{request.model}' cannot be deleted.",
            code="model_not_deletable",
        )
    ).to_response(status.HTTP_403_FORBIDDEN)
