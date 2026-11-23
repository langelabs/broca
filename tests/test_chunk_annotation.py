"""Verify the summaries and embeddings returned by chunk annotation."""

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, Mock, call

import pytest

from broca.inference.embedding import pipeline as embedding_pipeline
from broca.inference.embedding.types import EmbeddingResult
from broca.inference.text.types import TextResult
from broca.tools.chunk import Chunk, chunk
from broca.tools.chunk.pipeline import _annotate, annotate


def test_annotate_returns_new_chunks_with_ordered_summaries_and_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep source chunks intact and embed each summary with its original content.

    :param monkeypatch: Substitute the loaded models at the inference boundaries.
    """
    chunks = [
        Chunk(index=5, content="The schedule includes {draft} dates."),
        Chunk(index=2, content="The budget covers two workshops."),
    ]
    original = [chunk.model_copy(deep=True) for chunk in chunks]
    vectors = [[1.0, 0.0] + [0.0] * 126, [0.0, 1.0] + [0.0] * 126]
    text_model = AsyncMock(
        side_effect=[
            TextResult(text="  Draft schedule.  ", finish_reason="stop"),
            TextResult(text="Workshop budget.", finish_reason="stop"),
        ]
    )
    embedding_model = AsyncMock(
        side_effect=[
            EmbeddingResult(vectors=[vector], dimensions=128, normalized=True)
            for vector in vectors
        ]
    )
    monkeypatch.setattr(_annotate, "infer_text", text_model)
    monkeypatch.setattr(_annotate, "infer_embeddings", embedding_model)

    annotated = asyncio.run(annotate(chunks))

    assert annotated == [
        Chunk(
            index=5,
            content=chunks[0].content,
            summary="Draft schedule.",
            embedding=vectors[0],
        ),
        Chunk(
            index=2,
            content=chunks[1].content,
            summary="Workshop budget.",
            embedding=vectors[1],
        ),
    ]
    assert chunks == original
    assert all(
        chunk is not source for chunk, source in zip(annotated, chunks, strict=True)
    )
    assert chunks[0].content in text_model.await_args_list[0].args[0]
    assert chunks[1].content in text_model.await_args_list[1].args[0]
    embedding_model.assert_has_awaits(
        [
            call([f"Draft schedule.\n\n{chunks[0].content}"]),
            call([f"Workshop budget.\n\n{chunks[1].content}"]),
        ]
    )


def test_annotate_skips_inference_for_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an empty result when chunking finds no content.

    :param monkeypatch: Track calls to both inference boundaries.
    """
    text_model = AsyncMock()
    embedding_model = AsyncMock()
    monkeypatch.setattr(_annotate, "infer_text", text_model)
    monkeypatch.setattr(_annotate, "infer_embeddings", embedding_model)

    assert asyncio.run(annotate([])) == []
    text_model.assert_not_awaited()
    embedding_model.assert_not_awaited()


def test_chunk_annotates_a_named_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert and split Markdown before annotating it through the public entry point.

    :param monkeypatch: Substitute the text and embedding runtime boundaries.
    """
    vector = [1.0] + [0.0] * 127
    monkeypatch.setattr(
        _annotate,
        "infer_text",
        AsyncMock(
            return_value=TextResult(text="Workshop schedule.", finish_reason="stop")
        ),
    )
    monkeypatch.setattr(
        _annotate,
        "infer_embeddings",
        AsyncMock(
            return_value=EmbeddingResult(
                vectors=[vector], dimensions=128, normalized=True
            )
        ),
    )
    runtime = Mock()
    runtime.tokenize.side_effect = lambda data, *, add_bos: list(data)
    runtime.detokenize.side_effect = bytes
    monkeypatch.setattr(embedding_pipeline, "_runtime", runtime)

    result = asyncio.run(
        chunk(BytesIO(b"# Schedule\n\nWorkshop starts Monday."), name="schedule.md")
    )

    assert result.name == "schedule.md"
    (annotated,) = result.chunks
    assert "Workshop starts Monday." in annotated.content
    assert annotated.summary == "Workshop schedule."
    assert annotated.embedding == vector
