"""Video inference package."""

from functools import partial
from typing import TYPE_CHECKING

from .._exports import resolve_pipeline_export
from ._model import _model

if TYPE_CHECKING:
    from ._pipeline import VideoPipeline, pipeline

__all__ = ["VideoPipeline", "_model", "pipeline"]

__getattr__ = partial(resolve_pipeline_export, __name__, __all__)
