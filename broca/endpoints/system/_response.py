"""Response for the system metrics endpoint."""

from pydantic import BaseModel, Field, NonNegativeInt


class SystemResponse(BaseModel):
    """Return accelerator memory capacity and usage in bytes."""

    max_vram_bytes: NonNegativeInt = Field(
        description="Total visible CUDA capacity or the recommended Metal memory budget."
    )
    used_vram_bytes: NonNegativeInt = Field(
        description="Total visible CUDA usage or this process's Metal GPU allocation."
    )
