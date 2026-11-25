"""Report the server's accelerator memory metrics."""

from typing import Annotated

from fastapi import Query

from ._request import SystemRequest
from ._response import SystemResponse
from ._router import system_router
import torch


@system_router.get("")
def get_system(_request: Annotated[SystemRequest, Query()]) -> SystemResponse:
    """Read the current GPU memory metrics in a worker thread.

    :param _request: Parameterless query for system metrics.
    :returns: Capacity and usage in bytes, both zero without a supported accelerator.
    """
    if torch.cuda.is_available():
        max_vram_bytes = 0
        used_vram_bytes = 0
        for device in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            max_vram_bytes += total_bytes
            used_vram_bytes += total_bytes - free_bytes
        return SystemResponse(
            max_vram_bytes=max_vram_bytes, used_vram_bytes=used_vram_bytes
        )

    if torch.mps.is_available():
        return SystemResponse(
            max_vram_bytes=torch.mps.recommended_max_memory(),
            used_vram_bytes=torch.mps.driver_allocated_memory(),
        )

    return SystemResponse(max_vram_bytes=0, used_vram_bytes=0)
