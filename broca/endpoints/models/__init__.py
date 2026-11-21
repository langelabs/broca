"""Public router for the configured model catalog."""

from . import _endpoint as _endpoint
from ._router import models_router

__all__ = ["models_router"]
