"""Shared lifecycle and batch dispatch for model runtimes."""

import asyncio
import gc
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import dataclass
from itertools import count, islice
from threading import Condition, RLock, Thread
from time import monotonic
from typing import Any, ClassVar, Generic, TypeVar, cast, Sequence

import torch

from ...ai_model import AiModel

Runtime = TypeVar("Runtime")
Input = TypeVar("Input")
Result = TypeVar("Result")

_BATCH_WINDOW = 0.010


@dataclass(frozen=True)
class InferenceCall:
    """Carry call arguments that cannot be represented by a single positional input."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class _QueuedInference(Generic[Input, Result]):
    """Associate a queued input and its submission time with one caller's result."""

    item: Input | InferenceCall
    submitted_at: float
    future: Future[Result]


class BasePipeline(ABC, Generic[Runtime, Input, Result]):
    """Own one lazily initialized runtime for an AI model configuration."""

    _max_batch_size: ClassVar[int] = 8

    def __init__(self, model: AiModel) -> None:
        """Store the configuration without importing a backend or loading weights.

        :param model: Configuration for the pipeline's local model assets.
        """
        self.model = model
        self._compute_lock = RLock()
        self._runtime: Runtime | None = None
        self._context: AbstractContextManager[Runtime] | None = None

        self._batch_ready = Condition()
        self._batch: deque[_QueuedInference[Input, Result]] = deque()
        self._batch_results: dict[int, Future[Result]] = {}
        self._batch_indices = count()
        self._batch_worker_active = False

    def is_loaded(self) -> bool:
        """Read the runtime state while inference can continue.

        :returns: Whether the instance has published an initialized runtime.
        """
        return self._runtime is not None

    def get_device(self) -> str:
        """Select the configured device or detect an available accelerator.

        :returns: The configured device, CUDA, MPS, or CPU.
        """
        if self.model.device is not None:
            return self.model.device
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"

    @staticmethod
    def get_dtype(name: str | None, default: torch.dtype) -> torch.dtype:
        """Resolve a configured floating-point dtype against PyTorch.

        :param name: Optional name from the model configuration.
        :param default: Backend dtype to use when no name is configured.
        :returns: The configured or default dtype.
        :raises ValueError: If the configured dtype is unsupported.
        """

        if name is None:
            return default
        if name not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported model dtype: {name}")
        return cast(torch.dtype, getattr(torch, name))

    @property
    def runtime(self) -> Runtime:
        """Access the initialized backend.

        :raises RuntimeError: If the pipeline has no loaded runtime.
        """
        with self._compute_lock:
            if self._runtime is None or not self.is_loaded():
                raise RuntimeError("Call load() before accessing the pipeline runtime.")
            return self._runtime

    def load(self) -> bool:
        """Open local model assets and retain the backend until unload.

        :returns: True once the runtime is initialized.
        :raises Exception: If the local assets or backend cannot be loaded.
        """
        with self._compute_lock:
            if self.is_loaded():
                return True
            if self._context is not None:
                self.unload()
            context = self._open_runtime()
            self._runtime = context.__enter__()
            self._context = context
            return True

    @abstractmethod
    async def _infer(self, item: Input, /, *args: Any, **kwargs: Any) -> Result:
        """Run one inference operation on the worker's event loop.

        :param item: Input accepted by the concrete pipeline.
        :param args: Additional positional arguments accepted by the concrete pipeline.
        :param kwargs: Keyword options accepted by the concrete pipeline.
        :returns: The concrete pipeline's inference result.
        """
        ...

    async def _infer_batched(self, inputs: list[Input | InferenceCall]) -> list[Result]:
        """Run the single-input hook sequentially for each item.

        :param inputs: Inputs or bound arguments for the single-input hook.
        :returns: Inference results in input order.
        """
        return [
            await self._infer(*item.args, **item.kwargs)
            if isinstance(item, InferenceCall)
            else await self._infer(item)
            for item in inputs
        ]

    async def infer(self, *args: Any, **kwargs: Any) -> Result:
        """Queue one inference request and return its result from the batch worker.

        :param args: Positional inputs forwarded to the model hook.
        :param kwargs: Keyword options forwarded to the model hook.
        :returns: The result corresponding to this call's batch position.
        """
        index = self._add_to_batch(*args, **kwargs)
        return await self._get_batched_result(index)

    def _add_to_batch(self, *args: Any, **kwargs: Any) -> int:
        """Queue a request and assign an index that survives batch rollover.

        :param args: Positional inputs forwarded to the model hook.
        :param kwargs: Keyword options forwarded to the model hook.
        :returns: The index used to retrieve this request's result.
        """
        item = (
            cast(Input, args[0])
            if len(args) == 1 and not kwargs
            else InferenceCall(args, kwargs)
        )
        with self._batch_ready:
            index = next(self._batch_indices)
            future: Future[Result] = Future()
            self._batch_results[index] = future
            self._batch.append(_QueuedInference(item, monotonic(), future))
            self._batch_ready.notify()
            return index

    async def _get_batched_result(self, index: int) -> Result:
        """Start the worker if needed and wait for the indexed result.

        :param index: Index returned by :meth:`_add_to_batch`.
        :returns: The result for the indexed inference request.
        :raises KeyError: If the index is unknown or has already been consumed.
        """
        with self._batch_ready:
            future = self._batch_results[index]
        try:
            with self._batch_ready:
                if not self._batch_worker_active:
                    Thread(target=self._compute_batches, daemon=True).start()
                    self._batch_worker_active = True
            return await asyncio.wrap_future(future)
        finally:
            with self._batch_ready:
                self._batch_results.pop(index, None)
                if future.cancel():
                    self._batch_ready.notify()

    def _compute_batches(self) -> None:
        """Dispatch queued requests using this pipeline's batch size limit."""
        while True:
            with self._batch_ready:
                while self._batch and self._batch[0].future.cancelled():
                    self._batch.popleft()
                if not self._batch:
                    self._batch_worker_active = False
                    return

                deadline = self._batch[0].submitted_at + _BATCH_WINDOW
                size = sum(
                    entry.submitted_at < deadline
                    for entry in islice(self._batch, self._max_batch_size)
                )
                remaining = deadline - monotonic()
                if size < self._max_batch_size and remaining > 0:
                    self._batch_ready.wait(remaining)
                    continue
                entries = [self._batch.popleft() for _ in range(size)]

            batch = [
                entry
                for entry in entries
                if entry.future.set_running_or_notify_cancel()
            ]
            if not batch:
                continue
            try:
                with self._compute_lock:
                    results = asyncio.run(
                        self._infer_batched([entry.item for entry in batch])
                    )
                if len(results) != len(batch):
                    raise RuntimeError(
                        "Inference returned the wrong number of batch results."
                    )
            except BaseException as error:
                for entry in batch:
                    entry.future.set_exception(error)
            else:
                for entry, result in zip(batch, results, strict=True):
                    entry.future.set_result(result)

    def unload(self) -> bool:
        """Release the runtime, then return unused accelerator cache to the driver.

        :returns: True once the pipeline has no loaded runtime.
        """
        with self._compute_lock:
            context = self._context
            self._runtime = None
            self._context = None
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                finally:
                    del context
                    gc.collect()
                    device = self.get_device().partition(":")[0]
                    if device == "cuda" and torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                    elif device == "mps" and torch.mps.is_available():
                        torch.mps.empty_cache()
            return True

    @abstractmethod
    def _open_runtime(self) -> AbstractContextManager[Runtime]:
        """Create a backend context that releases resources when it exits."""
        raise NotImplementedError


    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs without adding sequence markers.

        :param text: Text to tokenize as UTF-8.
        :returns: Token IDs from the embedding model's vocabulary.
        :raises RuntimeError: If the embedding pipeline is unloaded.
        """
        with self._compute_lock:
            return self.runtime.tokenize(text.encode("utf-8"), add_bos=False)

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token IDs with the embedding model's vocabulary.

        :param tokens: Token IDs to decode.
        :returns: Text with replacement characters for incomplete UTF-8 sequences.
        :raises RuntimeError: If the embedding pipeline is unloaded.
        """
        with self._compute_lock:
            data = self.runtime.detokenize(list(tokens))
        return data.decode("utf-8", errors="replace")
