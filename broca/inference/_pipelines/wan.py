"""Wan 2.2 TI2V pipeline."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from .diffusers_pipeline import DiffusersPipeline
import torch
from diffusers import AutoencoderKLWan
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.pipelines.wan.pipeline_wan import WanPipeline as WanRuntime
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline

from ...ai_model import AiModel
from .__base import Input, Result


class WanPipeline(DiffusersPipeline["WanRuntime", Input, Result]):
    """Combine Wan's quantized transformer, text encoder, scheduler, and VAE."""

    def __init__(self, model: AiModel) -> None:
        """Store the configuration for both video generation modes.

        :param model: Configuration for the shared Wan weights.
        """
        super().__init__(model)
        self._image_runtime: WanImageToVideoPipeline | None = None

    @property
    def image_runtime(self) -> WanImageToVideoPipeline:
        """Access the first-frame conditioning pipeline while Wan is loaded.

        :returns: The runtime that shares the loaded text-to-video components.
        :raises RuntimeError: If the model is not loaded.
        """
        with self._lock:
            if self._image_runtime is None:
                raise RuntimeError("Call load() before accessing the pipeline runtime.")
            return self._image_runtime

    @contextmanager
    def _open_runtime(self) -> Generator[WanRuntime, None, None]:
        """Create the local Diffusers runtime and remove its hooks at unload."""

        device = torch.device(self.get_device())
        dtype = self.get_dtype(self.model.dtype, torch.bfloat16)
        local_path = str(self.model.local_path)
        vae_dtype = (self.model.runtime_options or {}).get("vae_dtype")
        vae = AutoencoderKLWan.from_pretrained(
            str(self.model.get_file_path("vae", weights=False)),
            dtype=self.get_dtype(
                vae_dtype if isinstance(vae_dtype, str) else None, torch.float32
            ),
            local_files_only=True,
        )
        runtime = WanRuntime.from_pretrained(
            local_path,
            transformer=self._load_transformer(WanTransformer3DModel, dtype),
            vae=vae,
            dtype=dtype,
            local_files_only=True,
        )
        try:
            if self.model.vae_tiling:
                vae.enable_tiling()
            if self.model.cpu_offload and device.type != "cpu":
                runtime.enable_model_cpu_offload(device=device)
            else:
                runtime.to(device)
            # from_pipe() casts every component, which GGUF transformers reject.
            self._image_runtime = WanImageToVideoPipeline(
                tokenizer=runtime.tokenizer,
                text_encoder=runtime.text_encoder,
                vae=runtime.vae,
                transformer=runtime.transformer,
                scheduler=type(runtime.scheduler).from_config(runtime.scheduler.config),
                expand_timesteps=bool(runtime.config.get("expand_timesteps", False)),
            )
            yield runtime
        finally:
            self._image_runtime = None
            runtime.remove_all_hooks()
