"""Request for the system metrics endpoint."""

from pydantic import BaseModel


class SystemRequest(BaseModel):
    """Read system metrics without request parameters."""
