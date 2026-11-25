"""Image inference package."""
from ._model import _model
from ._pipeline import ImagePipeline, pipeline

infer = pipeline.infer

__all__ = ["ImagePipeline", "_model", "pipeline","infer"]
