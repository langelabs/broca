"""Public router for image inference endpoints."""

from . import _endpoint as _endpoint
from ._router import images_router

__all__ = ["images_router"]
