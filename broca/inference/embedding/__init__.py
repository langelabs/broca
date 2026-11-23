"""Embedding inference package."""

from ._model import _model
from ._pipeline import EmbeddingPipeline, pipeline

infer = pipeline.infer

__all__ = ["EmbeddingPipeline", "_model", "pipeline", "infer"]
