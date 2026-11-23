"""Prepare embedding requests and pool native embedding batch rows."""

from __future__ import annotations

from typing import cast

from llama_cpp import Llama, llama_cpp as native
from pydantic import JsonValue

from .types import EmbeddingInput, EmbeddingOptions, EmbeddingRequest


_TASK_PROMPTS = {
    "question_answering": "question answering",
    "fact_checking": "fact checking",
    "code_retrieval": "code retrieval",
    "classification": "classification",
    "clustering": "clustering",
    "similarity": "sentence similarity",
}


def prepare_request(
    defaults: dict[str, JsonValue],
    /,
    inputs: list[EmbeddingInput],
    *,
    options: EmbeddingOptions | None = None,
) -> EmbeddingRequest:
    """Format text inputs and resolve the output settings for one caller.

    :param defaults: Model runtime settings for prompts and normalization.
    :param inputs: Text or encoded media inputs in caller order.
    :param options: Optional task, output dimension, and normalization overrides.
    :returns: The prepared inputs and their output settings.
    :raises ValueError: If the request contains no inputs.
    """
    if not inputs:
        raise ValueError("At least one embedding input is required.")
    options = options or EmbeddingOptions()
    normalize = (
        defaults.get("normalize_embeddings", True) is True
        if options.normalize is None
        else options.normalize
    )
    prepared: list[EmbeddingInput] = []
    for item in inputs:
        if not isinstance(item, str):
            prepared.append(item)
        elif options.task == "document":
            template = cast(
                str,
                defaults.get("document_template", "title: {title} | text: {content}"),
            )
            prepared.append(
                template.format(
                    title=defaults.get("default_document_title", "none"), content=item
                )
            )
        elif options.task == "search_query":
            prefix = cast(
                str, defaults.get("query_prefix", "task: search result | query: ")
            )
            prepared.append(prefix + item)
        else:
            prepared.append(f"task: {_TASK_PROMPTS[options.task]} | query: {item}")
    return EmbeddingRequest(prepared, options.dimensions or 768, normalize)


def set_embedding_row(
    batch: native.llama_batch_ext_p, row: int, positions: list[int]
) -> None:
    """Set the position and pooling output flag for a token or projected row.

    :param batch: The native batch that owns the row.
    :param row: Index returned by a native token or embedding insertion.
    :param positions: Position within the sequence, including M-RoPE axes when needed.
    :raises RuntimeError: If the native batch rejected the row.
    """
    if (
        row < 0
        or not native.llama_batch_ext_set_pos(
            batch, row, (native.llama_pos * len(positions))(*positions)
        )
        or not native.llama_batch_ext_set_output_embd(batch, row, True)
    ):
        raise RuntimeError("Could not prepare an embedding batch row.")


def decode_media_batch(
    runtime: Llama, batch: native.llama_batch_ext_p, sequence_count: int
) -> list[list[float]]:
    """Decode one packed media batch and copy its pooled vectors in sequence order.

    :param runtime: Loaded EmbeddingGemma runtime that owns the decoder.
    :param batch: Native batch containing complete media sequences.
    :param sequence_count: Number of sequence IDs, starting at zero.
    :returns: One pooled vector per sequence.
    :raises RuntimeError: If decoding fails or a sequence has no pooled vector.
    """
    vectors: list[list[float]] = []
    runtime._ctx.kv_cache_clear()
    try:
        if (
            native.llama_process(runtime.ctx, native.LLAMA_PROCESS_TYPE_DECODE, batch)
            != 0
        ):
            raise RuntimeError("The model could not embed the media batch.")
        width = runtime.n_embd()
        for sequence_id in range(sequence_count):
            output = native.llama_get_embeddings_seq(runtime.ctx, sequence_id)
            if not output:
                raise RuntimeError("The model returned no pooled media embedding.")
            vectors.append(output[:width])
        return vectors
    finally:
        runtime._ctx.kv_cache_clear()
