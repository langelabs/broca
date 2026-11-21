"""Public router for decision inference endpoints."""

from . import _endpoint as _endpoint
from ._router import decision_router

__all__ = ["decision_router"]
