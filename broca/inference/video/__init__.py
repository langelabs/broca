"""Video inference package."""

from ._pipeline import VideoPipeline, pipeline
from ._model import _model

infer = pipeline.infer

__all__ = ["VideoPipeline", "_model", "pipeline","infer"]
