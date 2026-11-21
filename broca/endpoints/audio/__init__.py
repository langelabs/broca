"""Public router for audio inference endpoints."""

from . import _endpoint as _endpoint
from ._router import audio_router

__all__ = ["audio_router"]
