"""Public router for embedding inference endpoints."""

from . import _endpoint as _endpoint
from ._router import embedding_router

__all__ = ["embedding_router"]
