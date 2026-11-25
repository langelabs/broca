"""Decision inference package."""
from ._pipeline import DecisionPipeline, pipeline
from ._model import _model

infer = pipeline.infer

__all__ = ["DecisionPipeline", "_model", "pipeline","infer"]
