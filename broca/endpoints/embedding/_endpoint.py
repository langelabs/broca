"""OpenAI-compatible embeddings route declaration."""

from fastapi import HTTPException, status

from ._request import EmbeddingRequest
from ._response import EmbeddingResponse
from ._router import embedding_router


@embedding_router.post("")
async def create_embeddings(body: EmbeddingRequest) -> EmbeddingResponse:
    """Declare creation of embeddings for text or token inputs.

    :param body: Inputs and vector encoding settings.
    :returns: The embedding vectors and usage.
    :raises HTTPException: Embedding generation is not implemented.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
