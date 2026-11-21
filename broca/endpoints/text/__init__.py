"""Public router for text inference endpoints."""

from . import _endpoint as _endpoint
from ._router import text_router

__all__ = ["text_router"]
