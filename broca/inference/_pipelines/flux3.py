"""FLUX 3 Image Klein pipeline."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

import torch
from diffusers.models.transformers.transformer_flux3 import (
    Flux3Transformer2DModel,
)
from diffusers.modular_pipelines.flux3.denoise import Flux3LoopAfterDenoiser
from diffusers.modular_pipelines.flux3.modular_blocks_flux3_distilled import (
    Flux3DistilledAutoBlocks,
)
from diffusers.modular_pipelines.flux3.modular_pipeline import (
    Flux3DistilledModularPipeline,
    Flux3ModularPipeline,
)
from diffusers.modular_pipelines.modular_pipeline import BlockState
from diffusers.quantizers.gguf.gguf_quantizer import GGUFQuantizer

from .__base import Input, Result
from .diffusers_pipeline import DiffusersPipeline


flux_step_callback: ContextVar[
    Callable[[Flux3ModularPipeline, torch.Tensor, int], None] | None
] = ContextVar("flux_step_callback", default=None)


class Flux3PreviewStep(Flux3LoopAfterDenoiser):
    """Expose a request's latents just after the scheduler updates them."""

    def __call__(
        self,
        components: Flux3ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[Flux3ModularPipeline, BlockState]:
        """Complete a denoising step and notify the active request.

        :param components: FLUX runtime components.
        :param block_state: Mutable state of the denoising loop.
        :param i: Zero-based denoising step.
        :param t: Scheduler timestep.
        :returns: The updated components and loop state.
        """
        components, block_state = super().__call__(components, block_state, i, t)
        callback = flux_step_callback.get()
        if callback is not None:
            callback(components, block_state.latents, i + 1)
        return components, block_state


class Flux3GGUFTransformer(Flux3Transformer2DModel):
    """Report the activation dtype expected by FLUX's GGUF linear layers."""

    @property
    def dtype(self) -> torch.dtype:
        """Keep denoising inputs in the GGUF compute dtype.

        :returns: The floating-point dtype used to dequantize the weights.
        """
        quantizer = getattr(self, "hf_quantizer", None)
        if isinstance(quantizer, GGUFQuantizer):
            return quantizer.compute_dtype
        return super().dtype


class Flux3Pipeline(DiffusersPipeline["Flux3DistilledModularPipeline", Input, Result]):
    """Combine the distilled GGUF with the local FLUX 3 encoder and VAE."""

    @contextmanager
    def _open_runtime(self) -> Generator[Flux3DistilledModularPipeline, None, None]:
        """Load the configured generator with request-scoped preview steps.

        :yields: The local FLUX runtime with the configured weights.
        """
        dtype = self.get_dtype(self.model.dtype, torch.bfloat16)
        # Diffusers declares some block instances at class scope.
        blocks = deepcopy(Flux3DistilledAutoBlocks())
        blocks.sub_blocks.pop("prompt_upsampler")
        denoise = blocks.sub_blocks["denoise"]
        for branch in denoise.sub_blocks.values():
            branch.sub_blocks["denoise"].sub_blocks["after_denoiser"] = (
                Flux3PreviewStep()
            )
        runtime = Flux3DistilledModularPipeline(
            blocks=blocks,
            pretrained_model_name_or_path=str(self.model.local_path),
            local_files_only=True,
        )
        runtime.update_components(
            transformer=self._load_transformer(Flux3GGUFTransformer, dtype)
        )
        runtime.load_components(
            names=["text_encoder", "tokenizer", "vae"],
            pretrained_model_name_or_path=str(self.model.local_path),
            dtype=dtype,
            local_files_only=True,
        )
        if self.model.vae_tiling:
            runtime.vae.enable_tiling()
        runtime.to(self.get_device())
        yield runtime
