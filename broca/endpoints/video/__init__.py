"""Public router for video inference endpoints."""

from . import _endpoint as _endpoint
from ._router import video_router

__all__ = ["video_router"]
