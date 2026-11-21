"""Check request batching at the EmbeddingGemma pipeline boundary."""

import asyncio
from unittest.mock import Mock

import pytest

from broca.inference._pipelines import llama_pipeline
from broca.inference.embedding import _pipeline as embedding_pipeline
from broca.inference.embedding._model import _model
from broca.inference.embedding._pipeline import EmbeddingPipeline
from broca.inference.embedding._types import (
    AudioEmbeddingInput,
    EmbeddingOptions,
    EmbeddingResult,
    ImageEmbeddingInput,
)


def test_concurrent_requests_share_embedding_batches_and_keep_their_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Format and flatten mixed inputs, then restore request order and dimensions.

    :param monkeypatch: Install a lightweight runtime at the native model boundary.
    """
    pipeline = EmbeddingPipeline(_model)
    prompts = {
        "title: none | text: First document": [3.0, 4.0] + [0.0] * 766,
        "title: none | text: Second document": [5.0] + [0.0] * 767,
        "task: search result | query: Third input": [0.0, 9.0] + [0.0] * 766,
    }
    runtime = Mock()
    runtime.embed.side_effect = lambda texts, **_: [prompts[text] for text in texts]
    monkeypatch.setattr(pipeline, "_runtime", runtime)
    image = ImageEmbeddingInput(data=b"image")
    audio = AudioEmbeddingInput(data=b"audio")
    embed_media = Mock(return_value=[[0.0, 2.0] + [0.0] * 766, [2.0] + [0.0] * 767])
    monkeypatch.setattr(embedding_pipeline, "embed_media", embed_media)

    async def collect() -> list[EmbeddingResult]:
        """Submit calls with different task prefixes and output settings.

        :returns: Results in public call order.
        """
        return await asyncio.gather(
            pipeline.infer(["First document", image, "Second document"]),
            pipeline.infer(
                inputs=[audio, "Third input"],
                options=EmbeddingOptions(
                    task="search_query", dimensions=128, normalize=False
                ),
            ),
        )

    first, second = asyncio.run(collect())

    runtime.embed.assert_called_once_with(list(prompts), truncate=False)
    embed_media.assert_called_once_with(runtime, [image, audio])
    assert first == EmbeddingResult(
        vectors=[
            [0.6, 0.8] + [0.0] * 766,
            [0.0, 1.0] + [0.0] * 766,
            [1.0] + [0.0] * 767,
        ],
        dimensions=768,
        normalized=True,
    )
    assert second == EmbeddingResult(
        vectors=[[2.0] + [0.0] * 127, [0.0, 9.0] + [0.0] * 126],
        dimensions=128,
        normalized=False,
    )


@pytest.mark.parametrize(
    ("logical", "physical", "expected"), [(256, 128, 128), (300, 300, 256)]
)
def test_embedding_runtime_limits_text_batches_to_native_microbatch_size(
    monkeypatch: pytest.MonkeyPatch, logical: int, physical: int, expected: int
) -> None:
    """Keep complete non-causal sequences within the native physical batch.

    :param monkeypatch: Simulate a runtime with a smaller physical batch limit.
    :param logical: Requested logical batch size.
    :param physical: Requested physical batch size.
    :param expected: Allowed token budget for the native batch.
    """

    class Runtime:
        """Expose the logical batch size consumed by the binding's embed method."""

        def __init__(self, *, n_batch: int, **_: object) -> None:
            """Record the binding's initial logical batch size.

            :param n_batch: Model's configured logical batch size.
            :param _: Remaining model arguments.
            """
            self.n_batch = n_batch

        def close(self) -> None:
            """Close the placeholder runtime."""

    monkeypatch.setattr(llama_pipeline, "Llama", Runtime)
    monkeypatch.setattr(
        llama_pipeline, "llama_max_parallel_sequences", Mock(return_value=256)
    )
    pipeline = EmbeddingPipeline(
        _model.model_copy(
            update={
                "clip": None,
                "batch_size": logical,
                "physical_batch_size": physical,
            }
        )
    )

    assert pipeline.load()
    try:
        assert pipeline.runtime.n_batch == expected
    finally:
        pipeline.unload()
