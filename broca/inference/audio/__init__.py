"""Audio inference package."""
from ._pipeline import AudioPipeline, pipeline
from ._model import _model

infer = pipeline.infer

__all__ = ["AudioPipeline", "_model", "pipeline","infer"]
