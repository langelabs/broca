"""Shared native GGUF loading for Gemma text and embedding models."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import closing, contextmanager
from typing import TypedDict, cast

from .__base import BasePipeline, Input, Result
from llama_cpp import (
    LLAMA_POOLING_TYPE_MEAN,
    LLAMA_POOLING_TYPE_UNSPECIFIED,
    Llama,
    llama_max_parallel_sequences,
    llama_model_n_embd_out,
)
from llama_cpp.llama_chat_format import Gemma4ChatHandler, Jinja2ChatFormatter

class LlamaOptions(TypedDict, total=False):
    """Map optional model settings to llama.cpp constructor arguments."""

    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    main_gpu: int
    tensor_split: list[float]
    n_threads: int
    n_threads_batch: int
    use_mmap: bool
    use_mlock: bool
    flash_attn: bool
    offload_kqv: bool
    seed: int


class LlamaPipeline(BasePipeline["Llama", Input, Result]):
    """Load and release a llama.cpp checkpoint and its optional projector."""

    _runtime_class: type[Llama] = Llama

    """
    def get_device(self) -> str:
        Select a device with support for the configured llama.cpp GPU offload.

        :returns: The configured device, a native GPU target, or CPU.

        if self.model.device is not None:
            return self.model.device
        if not self.model.gpu_layers:
            return "cpu"

        from llama_cpp import llama_supports_gpu_offload

        if not llama_supports_gpu_offload():
            return "cpu"
        device = super().get_device()
        return "gpu" if device == "cpu" else device
    """


    @contextmanager
    def _open_runtime(self) -> Generator[Llama, None, None]:
        """Translate loader settings and open the local GGUF assets."""

        class ProjectedEmbeddingRuntime(Llama):
            """Expose the projected width used by the embedding buffer."""

            def n_embd(self) -> int:
                """Read the output width, which differs from the input width."""
                return llama_model_n_embd_out(self.model)

        model = self.model
        runtime_options = model.runtime_options or {}
        embedding = runtime_options.get("embedding", model.type == "embedding") is True
        verbose = bool(runtime_options.get("verbose", False))
        n_gpu_layers = 0 if self.get_device() == "cpu" else model.gpu_layers
        handler = (
            Gemma4ChatHandler(
                clip_model_path=str(model.get_file_path(model.clip, weights=False)),
                verbose=verbose,
                use_gpu=n_gpu_layers != 0 if n_gpu_layers is not None else False,
            )
            if model.clip is not None
            else None
        )
        options = cast(
            LlamaOptions,
            {
                key: value
                for key, value in {
                    "n_ctx": model.context_window,
                    "n_batch": model.batch_size,
                    "n_ubatch": model.physical_batch_size,
                    "n_gpu_layers": n_gpu_layers,
                    "main_gpu": model.main_gpu,
                    "tensor_split": model.tensor_split,
                    "n_threads": model.cpu_threads,
                    "n_threads_batch": model.cpu_batch_threads,
                    "use_mmap": model.use_mmap,
                    "use_mlock": model.use_mlock,
                    "flash_attn": model.flash_attention,
                    "offload_kqv": model.offload_kqv,
                    "seed": model.seed,
                }.items()
                if value is not None
            },
        )
        if embedding:
            if model.context_window is not None:
                options.setdefault("n_batch", model.context_window)
                options.setdefault("n_ubatch", options["n_batch"])
            # Non-causal embedding calls must fit in one physical batch. The native
            # graph reserves whole groups of sequence slots, so align its token budget.
            limit = min(
                options.get("n_ctx", 512),
                options.get("n_batch", 512),
                options.get("n_ubatch", 512),
            )
            slots = min(limit, llama_max_parallel_sequences())
            options["n_batch"] = limit - limit % slots
        runtime_class = ProjectedEmbeddingRuntime if embedding else self._runtime_class
        with closing(
            runtime_class(
                model_path=str(model.checkpoint_path),
                embedding=embedding,
                pooling_type=(
                    LLAMA_POOLING_TYPE_MEAN
                    if embedding
                    else LLAMA_POOLING_TYPE_UNSPECIFIED
                ),
                chat_handler=handler,
                verbose=verbose,
                **options,
            )
        ) as runtime:
            if handler is not None:
                # The binding normally initializes the projector on the first chat.
                handler._init_mtmd_context(runtime)
            if model.chat_template is not None and handler is None:
                runtime.chat_handler = Jinja2ChatFormatter(
                    template=model.chat_template,
                    eos_token=runtime.detokenize([runtime.token_eos()]).decode(),
                    bos_token=runtime.detokenize([runtime.token_bos()]).decode(),
                    stop_token_ids=model.stop_token_ids,
                ).to_chat_handler()
            yield runtime
