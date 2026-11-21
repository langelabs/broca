"""Tests for the catalog's public model representation."""

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from broca import AiModel
from broca.endpoints.models._response import ModelResponse
from broca.inference.audio import _model as audio_model


def test_model_response_reports_current_lifecycle_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the size estimate, download progress, and runtime state.

    :param monkeypatch: Replaces the download tracker and pipeline module.
    """
    name = "broca.inference.audio._pipeline"
    monkeypatch.setitem(sys.modules, name, None)
    progress = AsyncMock(return_value=0.0)
    monkeypatch.setattr(AiModel, "download_progress", progress)

    response = asyncio.run(ModelResponse.from_model(audio_model))
    assert response.modality == "audio"
    assert response.size == audio_model.size
    assert response.download_progress == 0.0
    assert response.downloaded is False
    assert response.loaded is False

    progress.return_value = 0.25
    response = asyncio.run(ModelResponse.from_model(audio_model))
    assert response.download_progress == 0.25
    assert response.downloaded is False

    calls = ModuleType(name)
    pipeline = Mock(spec_set=["is_loaded"])
    monkeypatch.setattr(calls, "pipeline", pipeline, raising=False)
    monkeypatch.setitem(sys.modules, name, calls)
    progress.return_value = 1.0
    pipeline.is_loaded.return_value = True
    response = asyncio.run(ModelResponse.from_model(audio_model))
    assert response.download_progress == 1.0
    assert response.downloaded is True
    assert response.loaded is True

    pipeline.is_loaded.return_value = False
    response = asyncio.run(ModelResponse.from_model(audio_model))
    assert response.downloaded is True
    assert response.loaded is False
