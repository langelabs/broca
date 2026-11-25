"""OpenAI-compatible embeddings backed by the embedding inference export."""

from base64 import b64encode
from struct import pack

from fastapi import HTTPException
from pydantic import ValidationError

from ...inference import embedding
from ...inference.embedding.types import EmbeddingInput, EmbeddingOptions
from .._inference import require_loaded_model, run_inference
from ._request import EmbeddingRequest
from ._response import EmbeddingData, EmbeddingResponse, EmbeddingUsage
from ._router import embedding_router


@embedding_router.post("")
async def create_embeddings(body: EmbeddingRequest) -> EmbeddingResponse:
    """Embed a text input or a batch of text inputs.

    :param body: Text inputs, dimensions, and vector encoding.
    :returns: Ordered vectors and the number of tokens processed.
    :raises HTTPException: If the runtime or request options are unsupported.
    """
    require_loaded_model(embedding._model, body.model)
    if isinstance(body.input, str):
        inputs: list[EmbeddingInput] = [body.input]
    else:
        inputs = [item for item in body.input if isinstance(item, str)]
        if len(inputs) != len(body.input):
            raise HTTPException(
                422, "This model accepts text input in place of token IDs."
            )
    try:
        options = EmbeddingOptions.model_validate({"dimensions": body.dimensions})
    except ValidationError as error:
        raise HTTPException(
            422, "Supported embedding dimensions are 128, 256, 512, and 768."
        ) from error
    result = await run_inference(
        embedding._model, embedding.infer(inputs, options=options)
    )
    if result.prompt_tokens is None:
        raise HTTPException(503, "Embedding token usage is unavailable.")
    return EmbeddingResponse(
        model=body.model,
        data=[
            EmbeddingData(
                index=index,
                embedding=(
                    b64encode(pack(f"<{len(vector)}f", *vector)).decode("ascii")
                    if body.encoding_format == "base64"
                    else vector
                ),
            )
            for index, vector in enumerate(result.vectors)
        ],
        usage=EmbeddingUsage(
            prompt_tokens=result.prompt_tokens, total_tokens=result.prompt_tokens
        ),
    )
