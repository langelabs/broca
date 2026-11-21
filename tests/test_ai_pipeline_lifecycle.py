"""Exercise lazy backend ownership and recovery after a failed load."""

import asyncio
import gc
import sys
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from inspect import iscoroutinefunction
from threading import Event, get_ident
from time import monotonic
from typing import TypeVar
from weakref import ref

import pytest

from broca import AiModel
from broca.inference._pipelines import BasePipeline
from broca.inference.audio._model import _model as audio_model


RuntimeType = TypeVar("RuntimeType")


class LifecyclePipeline(BasePipeline[RuntimeType, None, RuntimeType]):
    """Exercise pipeline ownership with a minimal inference method."""

    async def _infer(self, _item: None = None) -> RuntimeType:
        """Return the runtime through the guarded inference entry point.

        :param _item: Placeholder input for the lifecycle check.
        :returns: The currently loaded runtime.
        """
        return self.runtime


def test_inference_catalog_resolves_shared_unloaded_pipelines() -> None:
    """Resolve the pipeline owned by each inference module without loading it."""
    from broca.inference import MODELS, get_pipeline

    for model in MODELS:
        pipeline = get_pipeline(model)
        assert pipeline is get_pipeline(model)
        assert pipeline.model is model
        assert not pipeline.is_loaded()
        assert type(pipeline).__module__.endswith("._pipeline")
        assert type(pipeline).infer is BasePipeline.infer
        assert iscoroutinefunction(pipeline._infer)


def test_infer_awaits_the_hook_in_a_worker_under_the_lifecycle_lock() -> None:
    """Keep the event loop responsive and hold the lock until the hook returns."""
    started = Event()
    release = Event()

    class Pipeline(LifecyclePipeline[int]):
        """Expose the thread running the model hook and pause until inspected."""

        def _open_runtime(self) -> nullcontext[int]:
            """Provide an inert runtime.

            :returns: A context containing a placeholder integer.
            """
            return nullcontext(0)

        async def _infer(self, _item: None = None) -> int:
            """Wait for the caller to inspect the lock, then report the worker thread.

            :param _item: Placeholder input for the worker check.
            :returns: The ID of the thread running inference.
            """
            started.set()
            assert release.wait(5)
            return get_ident()

    async def check_inference() -> None:
        """Verify that inference holds the lock while the caller can keep running."""
        pipeline = Pipeline(AiModel(id="lab/model"))
        task = asyncio.create_task(pipeline.infer())
        try:
            assert await asyncio.to_thread(started.wait, 2)
            acquired = pipeline._compute_lock.acquire(blocking=False)
            if acquired:
                pipeline._compute_lock.release()
            assert not acquired
        finally:
            release.set()
            worker_id = await task
        assert worker_id != get_ident()

    asyncio.run(check_inference())


class RecordingBatchPipeline(BasePipeline[None, int, int]):
    """Record batch dispatches without loading model weights."""

    batches: list[list[int]]
    dispatch_times: list[float]

    def __init__(self) -> None:
        """Create a fresh pipeline and dispatch history for each test."""
        super().__init__(AiModel(id="lab/batch"))
        self.batches = []
        self.dispatch_times = []

    def _open_runtime(self) -> nullcontext[None]:
        """Provide a placeholder runtime."""
        return nullcontext()

    async def _infer(self, value: int) -> int:
        """Return a value that identifies its corresponding request.

        :param value: Input assigned to the caller.
        :returns: The transformed input.
        """
        return value * 10

    async def _infer_batched(self, inputs: list[int]) -> list[int]:
        """Record each dispatch and fail a batch containing the sentinel input.

        :param inputs: The inputs collected by the inference worker.
        :returns: Results in the same order as the inputs.
        """
        self.batches.append(inputs)
        self.dispatch_times.append(monotonic())
        if -1 in inputs:
            raise ValueError("Batch failed")
        return await super()._infer_batched(inputs)


def test_infer_collects_and_routes_batches_of_at_most_eight() -> None:
    """Route callers to matching results across successive full batches."""
    pipeline = RecordingBatchPipeline()

    async def collect() -> list[int]:
        """Submit concurrent requests from a single event loop.

        :returns: Results in submission order.
        """
        return await asyncio.gather(*(pipeline.infer(item) for item in range(19)))

    assert asyncio.run(collect()) == [item * 10 for item in range(19)]
    assert [len(inputs) for inputs in pipeline.batches] == [8, 8, 3]


def test_partial_batch_waits_for_the_collection_window() -> None:
    """Give a partially filled batch ten milliseconds to collect more inputs."""
    pipeline = RecordingBatchPipeline()

    async def collect() -> None:
        """Submit a second request during the first request's window."""
        submitted = monotonic()
        first = asyncio.create_task(pipeline.infer(1))
        await asyncio.sleep(0)
        assert await asyncio.gather(first, pipeline.infer(2)) == [10, 20]
        assert pipeline.dispatch_times[0] - submitted >= 0.010

    asyncio.run(collect())
    assert pipeline.batches == [[1, 2]]


def test_batch_failure_and_waiter_cancellation_do_not_strand_requests() -> None:
    """Propagate failures and keep serving after a waiter cancels."""
    pipeline = RecordingBatchPipeline()

    async def collect() -> None:
        """Cancel one waiter while two other callers wait on a failing batch."""
        cancelled = asyncio.create_task(pipeline.infer(1))
        failing = asyncio.create_task(pipeline.infer(-1))
        peer = asyncio.create_task(pipeline.infer(2))
        await asyncio.sleep(0)
        cancelled.cancel()
        failures = await asyncio.gather(
            cancelled, failing, peer, return_exceptions=True
        )
        assert isinstance(failures[0], asyncio.CancelledError)
        assert all(isinstance(error, ValueError) for error in failures[1:])
        assert await pipeline.infer(3) == 30

    asyncio.run(asyncio.wait_for(collect(), timeout=2))


@pytest.mark.parametrize("inputs", [[], ["second", "first", "second"]])
def test_default_batch_returns_one_result_per_input_in_order(inputs: list[str]) -> None:
    """Use the single-input hook for each item, including repeated inputs.

    :param inputs: Inputs supplied to the default batch hook.
    """
    seen: list[str] = []

    class Pipeline(BasePipeline[None, str, str]):
        """Record the inputs the default implementation forwards."""

        def _open_runtime(self) -> nullcontext[None]:
            """Provide an inert runtime.

            :returns: An empty runtime context.
            """
            return nullcontext()

        async def _infer(self, value: str) -> str:
            """Record the input and return its corresponding result.

            :param value: Input forwarded by the batch hook.
            :returns: The result for this input.
            """
            seen.append(value)
            return value.upper()

    pipeline = Pipeline(AiModel(id="lab/model"))
    assert asyncio.run(pipeline._infer_batched(inputs)) == [
        value.upper() for value in inputs
    ]
    assert seen == inputs


def test_inference_call_requires_loaded_runtime() -> None:
    """Prevent inference from reloading a runtime after it was unloaded."""
    from broca.inference.audio._pipeline import pipeline

    with pytest.raises(RuntimeError, match="load"):
        asyncio.run(pipeline.infer("Hello"))


def test_runtime_is_lazy_idempotent_and_released() -> None:
    """Keep one runtime per instance and release it on unload."""
    loads: list[AiModel] = []
    releases: list[object] = []

    class Pipeline(LifecyclePipeline[object]):
        """Record resource ownership without requiring model weights."""

        @contextmanager
        def _open_runtime(self) -> Generator[object, None, None]:
            """Allocate and release a distinct backend for each load."""
            loads.append(self.model)
            runtime = object()
            try:
                yield runtime
            finally:
                releases.append(runtime)

    config = AiModel(id="lab/model")
    pipeline = Pipeline(config)
    other = Pipeline(AiModel(id="lab/model", device="cuda:1"))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "torch", None)
        assert pipeline.get_device() == "cpu"
        assert other.get_device() == "cuda:1"
    assert not loads and not pipeline.is_loaded()
    with pytest.raises(RuntimeError, match="load"):
        _ = pipeline.runtime
    assert pipeline.load()
    runtime = pipeline.runtime
    assert pipeline.load()
    assert pipeline.runtime is runtime and loads == [config]
    assert asyncio.run(pipeline.infer()) is runtime
    assert not other.is_loaded()
    assert pipeline.unload() and pipeline.unload()
    assert releases == [runtime] and not pipeline.is_loaded()
    assert pipeline.load()
    assert pipeline.runtime is not runtime
    assert pipeline.unload()


def test_failed_load_can_be_retried() -> None:
    """Publish backend state only after successful initialization."""
    attempts = 0

    class Pipeline(LifecyclePipeline[object]):
        """Fail the first attempt to initialize the backend."""

        @contextmanager
        def _open_runtime(self) -> Generator[object, None, None]:
            """Simulate a transient error while acquiring resources."""
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("checkpoint unavailable")
            yield object()

    pipeline = Pipeline(AiModel(id="lab/model"))
    with pytest.raises(OSError, match="checkpoint unavailable"):
        pipeline.load()
    assert not pipeline.is_loaded()
    assert pipeline.load() and pipeline.is_loaded()
    assert pipeline.unload()


def test_unload_collects_cyclic_runtime() -> None:
    """Collect a runtime cycle after its context releases ownership."""

    class Runtime:
        """Mimic a backend with a reference cycle."""

        def __init__(self) -> None:
            """Retain a self-reference until garbage collection."""
            self.cycle = self

    class Pipeline(LifecyclePipeline[Runtime]):
        """Hold a runtime through a context manager."""

        def _open_runtime(self) -> nullcontext[Runtime]:
            """Return a context that keeps its runtime until it is discarded."""
            return nullcontext(Runtime())

    pipeline = Pipeline(AiModel(id="lab/model"))
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert pipeline.load()
        runtime_ref = ref(pipeline.runtime)
        assert pipeline.unload()
        assert runtime_ref() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("device", ["mps", "cuda:1", "cpu"])
def test_unload_releases_unused_cache_after_runtime_is_collected(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Flush only the selected accelerator after releasing the runtime.

    :param monkeypatch: Provides cache operations on hosts without a GPU.
    :param device: Configured runtime device.
    """
    import torch

    class Runtime:
        """Represent an accelerator runtime with a reference cycle."""

        cycle: "Runtime"

    class Pipeline(LifecyclePipeline[Runtime]):
        """Retain a small cyclic runtime until the pipeline unloads."""

        def _open_runtime(self) -> nullcontext[Runtime]:
            """Return a context that strongly owns the runtime."""
            runtime = Runtime()
            runtime.cycle = runtime
            return nullcontext(runtime)

    pipeline = Pipeline(AiModel(id="lab/model", device=device))
    assert pipeline.load()
    runtime_ref = ref(pipeline.runtime)
    flushed: list[str] = []

    def record_flush(backend: str) -> None:
        """Record a flush after checking that backend teardown has finished."""
        assert runtime_ref() is None
        assert not pipeline.is_loaded()
        flushed.append(backend)

    monkeypatch.setattr(torch.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: record_flush("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: record_flush("cuda"))
    assert pipeline.unload()
    assert pipeline.unload()
    assert flushed == ([] if device == "cpu" else [device.split(":")[0]])


def test_bark_loads_native_model_and_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Bark's model and voice processor together for the loaded lifecycle.

    :param monkeypatch: Replaces checkpoint reads with small runtime objects.
    """
    from unittest.mock import Mock

    import torch
    from transformers import BarkModel, BarkProcessor, GenerationConfig

    from broca.inference.audio._pipeline import AudioPipeline

    model = audio_model.model_copy(update={"device": "cpu"})
    assert model.id == "suno/bark-small"
    generator = Mock(spec=BarkModel)
    generator.generation_config = GenerationConfig(
        semantic_config={"max_length": 20, "max_new_tokens": 768},
        coarse_acoustics_config={"max_length": 20, "sliding_window_len": 60},
        fine_acoustics_config={"max_length": 20},
    )
    generator.eval.return_value = generator
    generator.to.return_value = generator
    processor = Mock(spec=BarkProcessor)
    load_model = Mock(return_value=generator)
    load_processor = Mock(return_value=processor)
    monkeypatch.setattr(BarkModel, "from_pretrained", load_model)
    monkeypatch.setattr(BarkProcessor, "from_pretrained", load_processor)

    pipeline = AudioPipeline(model)
    load_model.assert_not_called()
    load_processor.assert_not_called()
    assert pipeline.load() and pipeline.load()
    load_model.assert_called_once_with(
        str(model.local_path),
        dtype=torch.float32,
        local_files_only=True,
        use_safetensors=False,
    )
    load_processor.assert_called_once_with(str(model.local_path), local_files_only=True)
    generator.to.assert_called_once_with("cpu")
    assert pipeline.runtime.model is generator
    assert pipeline.runtime.processor is processor
    generation = generator.generation_config.to_dict()
    assert generation["semantic_config"] == {"max_length": None, "max_new_tokens": 768}
    assert generation["coarse_acoustics_config"] == {
        "max_length": None,
        "sliding_window_len": 60,
    }
    assert generation["fine_acoustics_config"] == {"max_length": 20}
    assert pipeline.unload() and not pipeline.is_loaded()
