"""Shared checkpoint setup for Diffusers pipelines."""

from __future__ import annotations

from typing import TypeVar

from .__base import BasePipeline, Input, Result
import torch
from diffusers.loaders import FromOriginalModelMixin
from diffusers import GGUFQuantizationConfig

Pipeline = TypeVar("Pipeline")
Transformer = TypeVar("Transformer", bound="FromOriginalModelMixin")


class DiffusersPipeline(BasePipeline[Pipeline, Input, Result]):
    """Resolve model precision, device, and quantized transformer settings."""

    def _load_transformer(
        self, transformer_class: type[Transformer], dtype: torch.dtype
    ) -> Transformer:
        """Load a GGUF transformer using its local Diffusers configuration."""
        options = self.model.runtime_options or {}
        quantization = options.get("quantization_config")
        compute_dtype = (
            quantization.get("compute_dtype")
            if isinstance(quantization, dict)
            else None
        )
        return transformer_class.from_single_file(
            str(self.model.checkpoint_path),
            config=str(
                self.model.get_file_path(
                    str(options.get("transformer_subfolder", "transformer")),
                    weights=False,
                )
            ),
            quantization_config=GGUFQuantizationConfig(
                compute_dtype=self.get_dtype(
                    compute_dtype if isinstance(compute_dtype, str) else None, dtype
                ),
            ),
            dtype=dtype,
            local_files_only=True,
        )
