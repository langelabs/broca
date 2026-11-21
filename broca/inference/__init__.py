"""Configured models available for local inference."""

from importlib import import_module
from sys import modules
from typing import TYPE_CHECKING, Any

from ..ai_model import AiModel
from .audio import _model as audio_model
from .decision import _model as decision_model
from .embedding import _model as embedding_model
from .images import _model as image_model
from .text import _model as text_model
from .video import _model as video_model

if TYPE_CHECKING:
    from ._pipelines import BasePipeline

MODELS: list[AiModel] = [
    audio_model,
    decision_model,
    embedding_model,
    image_model,
    text_model,
    video_model,
]


def get_model(model_id: str) -> AiModel | None:
    """Find a configured model by its catalog ID.

    :param model_id: The model identifier.
    :returns: The configuration, or None when the ID is unknown.
    """
    return next((model for model in MODELS if model.id == model_id), None)


def _pipeline_module_name(model: AiModel) -> str:
    """Identify the inference module for a catalog model.

    :param model: A configuration from the inference catalog.
    :returns: The full module name that owns the pipeline.
    :raises ValueError: If the model does not have a configured pipeline.
    """
    if get_model(model.id) is not model or model.type is None:
        raise ValueError("The model does not have a configured inference pipeline.")
    package = "images" if model.type == "image" else model.type
    return f"{__name__}.{package}._pipeline"


def get_pipeline(model: AiModel) -> "BasePipeline[object, Any, object]":
    """Resolve the pipeline retained by the configured inference module.

    :param model: A configuration from the inference catalog.
    :returns: The same pipeline instance used by the inference function.
    :raises ValueError: If the model does not have a configured pipeline.
    """
    return import_module(_pipeline_module_name(model)).pipeline


def is_model_loaded(model: AiModel) -> bool:
    """Read a configured model's runtime state without importing its backend.

    :param model: A configuration from the inference catalog.
    :returns: Whether its pipeline is loaded in this process.
    """
    module = modules.get(_pipeline_module_name(model))
    pipeline: BasePipeline[object, Any, object] | None = getattr(
        module, "pipeline", None
    )
    return pipeline is not None and pipeline.is_loaded()


__all__ = ["MODELS", "get_model", "get_pipeline", "is_model_loaded"]
