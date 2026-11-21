"""Pydantic settings and encoded results for video inference."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


class VideoOptions(BaseModel):
    """Override generation defaults; dimensions and frames follow model limits."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    width: PositiveInt | None = None
    height: PositiveInt | None = None
    num_frames: PositiveInt | None = None
    num_inference_steps: PositiveInt | None = None
    guidance_scale: float | None = Field(default=None, ge=0)
    fps: PositiveFloat | None = None
    seed: NonNegativeInt | None = None
    output_format: Literal["mp4", "webm"] = "mp4"


class VideoResult(BaseModel):
    """Carry encoded video and the metadata required to play its frames."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    data: bytes = Field(min_length=1)
    media_type: Literal["video/mp4", "video/webm"]
    width: PositiveInt
    height: PositiveInt
    num_frames: PositiveInt
    fps: PositiveFloat
    seed: NonNegativeInt | None = None
