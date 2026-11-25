"""Pydantic inputs, settings, and results for audio inference."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
AudioMediaType = Literal[
    "audio/mpeg", "audio/ogg", "audio/aac", "audio/flac", "audio/wav", "audio/pcm"
]


class AudioVoice(BaseModel):
    """Select a built-in Bark voice preset."""

    model_config = ConfigDict(extra="forbid")

    preset: str = Field(min_length=1)


class AudioOptions(BaseModel):
    """Override generation defaults; omitted settings use the model configuration."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    semantic_max_new_tokens: PositiveInt | None = None
    temperature: float | None = Field(default=None, ge=0)
    seed: NonNegativeInt | None = None
    output_format: AudioFormat = "wav"


class AudioResult(BaseModel):
    """Carry encoded audio and playback metadata without requiring a file write."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    data: bytes = Field(min_length=1)
    media_type: AudioMediaType
    sample_rate: PositiveInt
    channels: PositiveInt
    duration_seconds: PositiveFloat
    seed: NonNegativeInt | None = None
