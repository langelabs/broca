"""Pydantic inputs, settings, and results for embedding inference."""

from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ImageEmbeddingInput(BaseModel):
    """Supply one encoded image for embedding."""

    model_config = ConfigDict(
        extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    kind: Literal["image"] = "image"
    data: bytes = Field(min_length=1)


class AudioEmbeddingInput(BaseModel):
    """Supply one encoded audio recording for embedding."""

    model_config = ConfigDict(
        extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    kind: Literal["audio"] = "audio"
    data: bytes = Field(min_length=1)


class VideoEmbeddingInput(BaseModel):
    """Supply one encoded video for embedding."""

    model_config = ConfigDict(
        extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    kind: Literal["video"] = "video"
    data: bytes = Field(min_length=1)


MediaEmbeddingInput: TypeAlias = (
    ImageEmbeddingInput | AudioEmbeddingInput | VideoEmbeddingInput
)
EmbeddingInput: TypeAlias = (
    str | Annotated[MediaEmbeddingInput, Field(discriminator="kind")]
)
EmbeddingDimensions: TypeAlias = Literal[128, 256, 512, 768]


class EmbeddingOptions(BaseModel):
    """Select text task formatting and optional embedding output overrides."""

    model_config = ConfigDict(extra="forbid")

    task: Literal[
        "document",
        "search_query",
        "question_answering",
        "fact_checking",
        "code_retrieval",
        "classification",
        "clustering",
        "similarity",
    ] = "document"
    dimensions: EmbeddingDimensions | None = None
    normalize: bool | None = None


@dataclass(frozen=True)
class EmbeddingRequest:
    """Keep formatted inputs and output settings together through batch dispatch."""

    inputs: list[EmbeddingInput]
    dimensions: EmbeddingDimensions
    normalize: bool


class EmbeddingResult(BaseModel):
    """Return vectors in input order, with one vector per supplied input."""

    model_config = ConfigDict(extra="forbid")

    vectors: list[list[FiniteFloat]] = Field(min_length=1)
    dimensions: EmbeddingDimensions
    normalized: bool
