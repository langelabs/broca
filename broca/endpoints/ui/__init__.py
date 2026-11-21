"""Public router for the AI dashboard and model test pages."""

from . import endpoints as endpoints
from ._router import ui_router

__all__ = ["ui_router"]
