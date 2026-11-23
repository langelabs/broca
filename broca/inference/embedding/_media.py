"""Prepare and pack complete multimodal sequences for EmbeddingGemma."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from io import BytesIO
from math import floor

import av
from llama_cpp import Llama, llama_cpp as native, mtmd_cpp as mtmd
from llama_cpp.llama_chat_format import Gemma4ChatHandler

from .types import MediaEmbeddingInput
from ._utils import decode_media_batch, set_embedding_row


@contextmanager
def _open_media_chunks(
    item: MediaEmbeddingInput, projector: mtmd.mtmd_context_p, limit: int
) -> Iterator[tuple[mtmd.mtmd_input_chunks_p, int]]:
    """Decode and tokenize one input while retaining its native resources.

    :param item: Encoded image, audio, or video input.
    :param projector: Loaded MTMD projector.
    :param limit: Maximum token count for a full sequence.
    :yields: Tokenized chunks and their total token count.
    :raises ValueError: If the media is invalid or exceeds the token budget.
    :raises RuntimeError: If the native chunks cannot be allocated.
    """
    marker = mtmd.mtmd_default_marker()
    with ExitStack() as resources:
        bitmaps: list[mtmd.mtmd_bitmap_p] = []
        if item.kind == "video":
            prompt_parts: list[bytes] = []
            next_frame = 0
            with av.open(
                BytesIO(item.data),
                mode="r",
                options={"format_whitelist": "mov,matroska,avi,mpegts,webm,ogg,mpeg"},
            ) as video:
                if not video.streams.video:
                    raise ValueError("The supplied video contains no video stream.")
                for frame in video.decode(video=0):
                    timestamp = frame.time
                    if timestamp is not None and timestamp < next_frame and bitmaps:
                        continue
                    rgb = frame.to_ndarray(format="rgb24").tobytes()
                    bitmap = mtmd.mtmd_bitmap_init(
                        frame.width,
                        frame.height,
                        (ctypes.c_uint8 * len(rgb)).from_buffer_copy(rgb),
                    )
                    if bitmap is None:
                        raise ValueError("Could not decode an embedding video frame.")
                    resources.callback(mtmd.mtmd_bitmap_free, bitmap)
                    bitmaps.append(bitmap)
                    if len(bitmaps) > limit // 70:
                        raise ValueError(
                            "The video exceeds the model's embedding context."
                        )
                    prompt_parts.append(f"<{next_frame:.1f} seconds>".encode() + marker)
                    next_frame = (
                        floor(timestamp) + 1
                        if timestamp is not None
                        else next_frame + 1
                    )
            if not bitmaps:
                raise ValueError("The supplied video contains no decodable frames.")
            prompt = b"".join(prompt_parts)
        else:
            media = mtmd.mtmd_helper_bitmap_init_from_buf_wrapper(
                projector,
                (ctypes.c_uint8 * len(item.data)).from_buffer_copy(item.data),
                len(item.data),
                False,
                mtmd.mtmd_helper_init_opt_default(),
            )
            if media.video_ctx is not None:
                resources.callback(mtmd.mtmd_helper_video_free, media.video_ctx)
            if media.bitmap is not None:
                resources.callback(mtmd.mtmd_bitmap_free, media.bitmap)
            if media.bitmap is None or mtmd.mtmd_bitmap_is_audio(media.bitmap) != (
                item.kind == "audio"
            ):
                raise ValueError(f"Could not decode the supplied {item.kind}.")
            bitmaps.append(media.bitmap)
            prompt = marker

        chunks = mtmd.mtmd_input_chunks_init()
        if chunks is None:
            raise RuntimeError("Could not allocate embedding input chunks.")
        resources.callback(mtmd.mtmd_input_chunks_free, chunks)
        text = mtmd.mtmd_input_text(prompt, len(prompt), True, True)
        bitmap_array = (mtmd.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
        if (
            mtmd.mtmd_tokenize(
                projector, chunks, ctypes.pointer(text), bitmap_array, len(bitmaps)
            )
            != 0
        ):
            raise ValueError("Could not tokenize the embedding media.")
        size = mtmd.mtmd_helper_get_n_tokens(chunks)
        if size > limit:
            raise ValueError("The media exceeds the model's embedding context.")
        if size < 1:
            raise ValueError("The media produced no embedding tokens.")
        yield chunks, size


def embed_media(
    runtime: Llama, inputs: Sequence[MediaEmbeddingInput]
) -> list[list[float]]:
    """Decode media sequences together as native token and slot capacity allows.

    :param runtime: Loaded EmbeddingGemma model and media projector.
    :param inputs: Encoded media inputs in desired output order.
    :returns: One pooled vector for each input, in the same order.
    :raises RuntimeError: If projection resources or pooled results are unavailable.
    :raises ValueError: If an input cannot be projected in one native microbatch.
    """
    handler = runtime.chat_handler
    projector = handler.mtmd_ctx if isinstance(handler, Gemma4ChatHandler) else None
    if projector is None:
        raise RuntimeError(
            "Load the EmbeddingGemma media projector before embedding media."
        )

    context = runtime.ctx
    limit = min(runtime.n_ctx(), runtime.n_batch, runtime.context_params.n_ubatch)
    sequence_limit = native.llama_n_seq_max(context)
    width = native.llama_model_n_embd_inp(runtime.model)
    mrope = mtmd.mtmd_decode_use_mrope(projector)
    batch = native.llama_batch_ext_init(context)
    if batch is None:
        raise RuntimeError("Could not allocate a multimodal embedding batch.")

    vectors: list[list[float]] = []
    total_tokens = 0
    sequences = 0
    try:
        for item in inputs:
            with _open_media_chunks(item, projector, limit) as (chunks, size):
                if total_tokens + size > limit or sequences >= sequence_limit:
                    vectors.extend(decode_media_batch(runtime, batch, sequences))
                    native.llama_batch_ext_clear(batch)
                    total_tokens = 0
                    sequences = 0

                position = 0
                for index in range(mtmd.mtmd_input_chunks_size(chunks)):
                    chunk = mtmd.mtmd_input_chunks_get(chunks, index)
                    if chunk is None:
                        raise RuntimeError(
                            "The embedding tokenizer returned an empty chunk."
                        )
                    if (
                        mtmd.mtmd_input_chunk_get_type(chunk)
                        == mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT
                    ):
                        count = ctypes.c_size_t()
                        tokens = mtmd.mtmd_input_chunk_get_tokens_text(
                            chunk, ctypes.pointer(count)
                        )
                        for offset in range(count.value):
                            row = native.llama_batch_ext_add_token(
                                batch, sequences, tokens[offset]
                            )
                            set_embedding_row(batch, row, [position + offset])
                    else:
                        if mtmd.mtmd_encode_chunk(projector, chunk) != 0:
                            raise ValueError("Could not encode the embedding media.")
                        projected = mtmd.mtmd_get_output_embd(projector)
                        if projected is None:
                            raise RuntimeError(
                                "The media projector returned no embeddings."
                            )
                        count = mtmd.mtmd_input_chunk_get_n_tokens(chunk)
                        # The native batch copies each row before the projector is reused.
                        rows = ctypes.cast(
                            projected, ctypes.POINTER(ctypes.c_float * (count * width))
                        ).contents
                        image_tokens = mtmd.mtmd_input_chunk_get_tokens_image(chunk)
                        for offset in range(count):
                            pointer = ctypes.cast(
                                ctypes.byref(
                                    rows, offset * width * ctypes.sizeof(ctypes.c_float)
                                ),
                                ctypes.POINTER(ctypes.c_float),
                            )
                            row = native.llama_batch_ext_add_embd(
                                batch, sequences, native.llama_embd(pointer, 1, width)
                            )
                            if mrope and image_tokens is not None:
                                relative = mtmd.mtmd_image_tokens_get_decoder_pos(
                                    image_tokens, position, offset
                                )
                                positions = [
                                    relative.t,
                                    relative.y,
                                    relative.x,
                                    relative.z,
                                ]
                            else:
                                positions = [position + offset] * (4 if mrope else 1)
                            set_embedding_row(batch, row, positions)
                    position += mtmd.mtmd_input_chunk_get_n_pos(chunk)
                total_tokens += size
                sequences += 1
        if sequences:
            vectors.extend(decode_media_batch(runtime, batch, sequences))
    finally:
        native.llama_batch_ext_free(batch)
    return vectors
