"""Native runtime configuration and per-request state for batched text decoding."""

from dataclasses import dataclass, field
from typing import TypedDict

from llama_cpp import Llama, llama_cpp as native


class CompletionOptions(TypedDict):
    """Share sampling settings between native decoding and Gemma's media handler."""

    max_tokens: int | None
    temperature: float
    top_p: float
    top_k: int
    stop: list[str]
    seed: int | None


class BatchLlama(Llama):
    """Allocate sequence slots in one KV cache shared by text and media inference."""

    @property
    def context_params(self) -> native.llama_context_params:
        """Return the context parameters populated by the binding during construction.

        :returns: Parameters used by the native llama.cpp context.
        """
        return self._batch_context_params

    @context_params.setter
    def context_params(self, params: native.llama_context_params) -> None:
        """Configure slots before the binding constructs the native context.

        :param params: Context parameters initialized by ``Llama.__init__``.
        """
        # llama-cpp-python 0.3.37 has no constructor arguments for text sequence slots.
        params.n_seq_max = min(4, native.llama_max_parallel_sequences())
        params.kv_unified = True
        self._batch_context_params = params


@dataclass
class TextSequence:
    """Track the prompt, sampler settings, and progress of one native sequence."""

    index: int
    prompt: list[int]
    options: CompletionOptions
    stop: tuple[bytes, ...]
    seed: int
    max_tokens: int = 0
    position: int = 0
    generated: list[int] = field(default_factory=list)
