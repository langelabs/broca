"""Shared model checks and error handling for inference endpoints."""

import logging
from collections.abc import Awaitable
from typing import TypeVar

from fastapi import HTTPException
from PIL import UnidentifiedImageError

from ..ai_model import AiModel
from ..inference import is_model_loaded

logger = logging.getLogger(__name__)
Result = TypeVar("Result")


def require_loaded_model(model: AiModel, requested_id: str | None) -> None:
    """Require the configured model to match the request and be loaded.

    :param model: Model owned by the endpoint's inference module.
    :param requested_id: Requested model ID, or None for the default.
    :raises HTTPException: If the model is unknown or unloaded.
    """
    if requested_id not in (None, model.id):
        raise HTTPException(404, "Model not found.")
    if not is_model_loaded(model):
        raise HTTPException(409, "Load the model before running inference.")


async def run_inference(model: AiModel, result: Awaitable[Result]) -> Result:
    """Await the backend and translate inference errors into HTTP responses.

    :param model: Model handling the inference request.
    :param result: Awaitable returned by an inference function.
    :returns: The inference result.
    :raises HTTPException: If the input is invalid or the backend is unavailable.
    """
    try:
        return await result
    except (ValueError, UnidentifiedImageError) as error:
        raise HTTPException(422, "The model rejected the inference input.") from error
    except Exception as error:
        if not is_model_loaded(model):
            raise HTTPException(409, "The model was unloaded.") from error
        logger.error("Inference failed for %s (%s)", model.id, type(error).__name__)
        raise HTTPException(503, "Model inference failed.") from error
