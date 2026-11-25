"""Pydantic settings and encoded results for image inference."""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

RasterFormat = Literal["png", "jpeg", "webp"]
ImagePreviewCallback = Callable[[int, bytes], None]


class ImageOptions(BaseModel):
    """Override configured generation defaults and select an encoded format."""

    model_config = ConfigDict(extra="forbid")

    width: PositiveInt | None = None
    height: PositiveInt | None = None
    num_inference_steps: PositiveInt | None = None
    seed: NonNegativeInt | None = None
    output_format: RasterFormat = "png"


class ImageResult(BaseModel):
    """Carry one encoded image and its format and pixel dimensions."""

    model_config = ConfigDict(
        extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    data: bytes = Field(min_length=1)
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    width: PositiveInt
    height: PositiveInt
    seed: NonNegativeInt | None = None
