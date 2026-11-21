"""Response for the OpenAI-compatible embeddings endpoint."""

from typing import Literal

from pydantic import BaseModel, FiniteFloat, NonNegativeInt


class EmbeddingData(BaseModel):
    """Return one vector and its position in the input batch."""

    object: Literal["embedding"] = "embedding"
    index: NonNegativeInt
    embedding: list[FiniteFloat] | str


class EmbeddingUsage(BaseModel):
    """Report token usage for the embedding request."""

    prompt_tokens: NonNegativeInt
    total_tokens: NonNegativeInt


class EmbeddingResponse(BaseModel):
    """Return the ordered embedding objects and token usage."""

    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
