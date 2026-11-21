"""Request for the OpenAI-compatible embeddings endpoint."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

EmbeddingText = Annotated[str, Field(min_length=1)]
EmbeddingTokens = Annotated[list[NonNegativeInt], Field(min_length=1)]


class EmbeddingRequest(BaseModel):
    """Select a model and one input or a batch to embed."""

    model: str = Field(min_length=1)
    input: (
        EmbeddingText
        | Annotated[list[EmbeddingText], Field(min_length=1, max_length=2048)]
        | EmbeddingTokens
        | Annotated[list[EmbeddingTokens], Field(min_length=1, max_length=2048)]
    )
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: PositiveInt | None = None
    user: str | None = None
