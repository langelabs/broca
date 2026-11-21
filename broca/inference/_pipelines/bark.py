"""Load Bark models and their speaker preset processor with Transformers."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import BarkModel, BarkProcessor

from .__base import BasePipeline, Input, Result


@dataclass(frozen=True, slots=True)
class BarkRuntime:
    """Hold the speech generator and processor for one loaded Bark model."""

    model: BarkModel
    processor: BarkProcessor


class BarkPipeline(BasePipeline[BarkRuntime, Input, Result]):
    """Load Bark from a local snapshot on the selected inference device."""

    @contextmanager
    def _open_runtime(self) -> Generator[BarkRuntime, None, None]:
        """Load Bark weights and speaker presets from local model assets.

        :returns: The model and processor for the lifetime of the loaded pipeline.
        """
        config = self.model
        path = str(config.local_path)
        device = self.get_device()
        dtype = self.get_dtype(config.dtype, torch.float32)
        processor = BarkProcessor.from_pretrained(path, local_files_only=True)
        model = BarkModel.from_pretrained(
            path,
            dtype=dtype,
            local_files_only=True,
            use_safetensors=config.model_format == "safetensors",
        )
        # Bark passes a new-token limit for each coarse sliding window.
        coarse = getattr(model.generation_config, "coarse_acoustics_config", None)
        if isinstance(coarse, dict) and coarse.get("max_length") is not None:
            coarse["max_length"] = None

        semantic = getattr(model.generation_config, "semantic_config", None)
        if isinstance(semantic, dict) and semantic.get("max_new_tokens") is not None:
            semantic["max_length"] = None

        yield BarkRuntime(model=model.eval().to(device), processor=processor)
