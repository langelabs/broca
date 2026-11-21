"""Embed text and encoded media with the configured EmbeddingGemma model."""

from collections.abc import Sequence
from math import hypot
from typing import cast

from .._pipelines import LlamaPipeline
from .._pipelines.__base import InferenceCall
from ._media import embed_media
from ._model import _model
from ._types import (
    EmbeddingInput,
    EmbeddingOptions,
    EmbeddingRequest,
    EmbeddingResult,
    MediaEmbeddingInput,
)
from ._utils import prepare_request


class EmbeddingPipeline(LlamaPipeline[list[EmbeddingInput], EmbeddingResult]):
    """Embed text and media using the configured EmbeddingGemma runtime."""

    async def _infer(
        self,
        inputs: list[EmbeddingInput],
        *,
        options: EmbeddingOptions | None = None,
    ) -> EmbeddingResult:
        """Embed one request through the batch computation.

        :param inputs: Nonempty list of text strings or encoded media inputs.
        :param options: Task, output dimension, and normalization overrides.
        :returns: Embedding vectors in the order of the supplied inputs.
        """
        request = prepare_request(
            self.model.runtime_options or {}, inputs, options=options
        )
        (result,) = await self._infer_batched([request])
        return result

    async def _infer_batched(
        self, inputs: Sequence[list[EmbeddingInput] | EmbeddingRequest | InferenceCall]
    ) -> list[EmbeddingResult]:
        """Coalesce embedding inputs and restore one result per caller.

        :param inputs: Embedding requests or queued call arguments.
        :returns: Per-request vectors in request and input order.
        :raises RuntimeError: If the model returns missing or undersized vectors.
        """
        if not inputs:
            return []
        runtime = self.runtime
        defaults = self.model.runtime_options or {}
        requests: list[EmbeddingRequest] = []
        texts: list[str] = []
        media: list[MediaEmbeddingInput] = []
        for item in inputs:
            if isinstance(item, InferenceCall):
                request = prepare_request(defaults, *item.args, **item.kwargs)
            elif isinstance(item, EmbeddingRequest):
                request = item
            else:
                request = prepare_request(defaults, item)
            requests.append(request)
            for value in request.inputs:
                if isinstance(value, str):
                    texts.append(value)
                else:
                    media.append(value)

        text_vectors = (
            cast(list[list[float]], runtime.embed(texts, truncate=False))
            if texts
            else []
        )
        media_vectors = embed_media(runtime, media) if media else []
        if len(text_vectors) != len(texts) or len(media_vectors) != len(media):
            raise RuntimeError(
                "The model returned the wrong number of embedding vectors."
            )

        texts_in_order = iter(text_vectors)
        media_in_order = iter(media_vectors)
        results: list[EmbeddingResult] = []
        for request in requests:
            vectors: list[list[float]] = []
            for item in request.inputs:
                vector = next(
                    texts_in_order if isinstance(item, str) else media_in_order
                )
                if len(vector) < request.dimensions:
                    raise RuntimeError(
                        "The model returned fewer embedding dimensions than requested."
                    )
                vector = vector[: request.dimensions]
                if request.normalize and (norm := hypot(*vector)):
                    vector = [value / norm for value in vector]
                vectors.append(vector)
            results.append(
                EmbeddingResult(
                    vectors=vectors,
                    dimensions=request.dimensions,
                    normalized=request.normalize,
                )
            )
        return results


pipeline = EmbeddingPipeline(_model)
