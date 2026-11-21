"""Portable AI model configuration and local asset management."""

from .ai_model import AiModel
from .endpoints import router as broca_router
from .server import app as broca_app

__all__ = ["AiModel", "broca_router", "broca_app"]
