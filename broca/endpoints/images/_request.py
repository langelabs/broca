"""Request for the OpenAI-compatible image generation endpoint."""

import re
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class ImageGenerationRequest(BaseModel):
    """Describe images to generate and their output settings."""

    prompt: str = Field(min_length=1, max_length=32000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=10)
    size: str | None = None
    quality: (
        Literal["auto", "standard", "hd", "low", "medium", "high", "xhigh", "max"]
        | None
    ) = None
    background: Literal["auto", "transparent", "opaque"] | None = None
    output_format: Literal["png", "jpeg", "webp", "gif", "svg"] | None = None
    output_compression: int | None = Field(default=None, ge=0, le=100)
    response_format: Literal["url", "b64_json"] | None = None
    stream: bool = False
    partial_images: int | None = Field(default=None, ge=0, le=3)
    user: str | None = None
    safety_identifier: str | None = None

    @model_validator(mode="after")
    def validate_flux_options(self) -> Self:
        """Reject settings the installed FLUX runtime cannot fulfill.

        :returns: The validated generation request.
        :raises ValueError: If a prompt or generation setting is unsupported.
        """
        if not self.prompt.strip():
            raise ValueError("Prompt must not be blank.")
        if self.n != 1:
            raise ValueError("FLUX generates one image per request.")
        if self.partial_images and not self.stream:
            raise ValueError("partial_images requires stream=true.")
        if self.output_format in {"gif", "svg"} or self.response_format == "url":
            raise ValueError("FLUX returns base64 PNG, JPEG, or WebP images.")
        if self.background == "transparent" or self.quality not in (None, "auto"):
            raise ValueError("FLUX supports opaque images with its default quality.")
        if self.output_compression is not None:
            raise ValueError("FLUX uses the default output compression.")
        if self.size is not None and self.size != "auto":
            if not re.fullmatch(r"[0-9]{1,4}x[0-9]{1,4}", self.size):
                raise ValueError("Image size must use WIDTHxHEIGHT.")
            width, height = map(int, self.size.split("x"))
            if (
                any(edge < 256 or edge > 1536 or edge % 16 for edge in (width, height))
                or width * height > 1024 * 1536
            ):
                raise ValueError(
                    "Image dimensions must be multiples of 16 from 256 to 1536 "
                    "with at most 1,572,864 pixels."
                )
        return self
