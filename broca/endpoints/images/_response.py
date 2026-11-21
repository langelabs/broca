"""Response for the OpenAI-compatible image generation endpoint."""

from typing import Literal

from pydantic import BaseModel, NonNegativeInt


class GeneratedImage(BaseModel):
    """Carry an encoded image or a URL to the generated asset."""

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImageInputTokenDetails(BaseModel):
    """Break down image-generation input tokens by modality."""

    text_tokens: NonNegativeInt
    image_tokens: NonNegativeInt


class ImageUsage(BaseModel):
    """Report token usage for image generation when available."""

    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    total_tokens: NonNegativeInt
    input_tokens_details: ImageInputTokenDetails | None = None


class ImageGenerationResponse(BaseModel):
    """Return generated images and the output metadata."""

    created: NonNegativeInt
    data: list[GeneratedImage] | None = None
    background: Literal["transparent", "opaque"] | None = None
    output_format: Literal["png", "jpeg", "webp", "gif", "svg"] | None = None
    quality: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    size: str | None = None
    safety_identifier: str | None = None
    usage: ImageUsage | None = None
