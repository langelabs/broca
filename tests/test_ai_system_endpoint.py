"""Exercise the system route with and without available accelerators."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from broca.server import app


def test_system_reports_total_and_used_vram_across_visible_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sum capacity and calculate used bytes from each CUDA device's free memory.

    :param monkeypatch: Replace the accelerator API with fixed readings.
    """
    readings = [(512, 1024), (2048, 4096)]
    cuda = SimpleNamespace(
        is_available=Mock(return_value=True),
        device_count=Mock(return_value=len(readings)),
        mem_get_info=Mock(side_effect=readings.__getitem__),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    with TestClient(app) as client:
        response = client.get("/system")

    assert response.status_code == 200
    assert response.json() == {"max_vram_bytes": 5120, "used_vram_bytes": 2560}


def test_system_reports_metal_budget_and_driver_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the recommended budget and process-level Metal GPU allocations.

    :param monkeypatch: Replace the accelerator API with fixed readings.
    """
    mps = SimpleNamespace(
        is_available=Mock(return_value=True),
        recommended_max_memory=Mock(return_value=4096),
        driver_allocated_memory=Mock(return_value=1024),
    )
    cuda = SimpleNamespace(is_available=Mock(return_value=False))
    torch = SimpleNamespace(cuda=cuda, mps=mps)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with TestClient(app) as client:
        response = client.get("/system")

    assert response.status_code == 200
    assert response.json() == {"max_vram_bytes": 4096, "used_vram_bytes": 1024}


@pytest.mark.parametrize("torch_installed", [True, False])
def test_system_reports_zero_without_a_supported_accelerator(
    monkeypatch: pytest.MonkeyPatch, torch_installed: bool
) -> None:
    """Serve zero VRAM on CPU-only installs and without the optional dependency.

    :param monkeypatch: Replace the accelerator API or simulate a missing package.
    :param torch_installed: Whether the optional PyTorch package is present.
    """
    cpu = SimpleNamespace(is_available=Mock(return_value=False))
    torch = SimpleNamespace(cuda=cpu, mps=cpu) if torch_installed else None
    monkeypatch.setitem(sys.modules, "torch", torch)

    with TestClient(app) as client:
        response = client.get("/system")

    assert response.status_code == 200
    assert response.json() == {"max_vram_bytes": 0, "used_vram_bytes": 0}
