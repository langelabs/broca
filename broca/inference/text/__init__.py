"""Text inference package."""
from ._model import _model
from ._pipeline import TextPipeline, pipeline

infer = pipeline.infer

__all__ = ["TextPipeline", "_model", "pipeline","infer"]
