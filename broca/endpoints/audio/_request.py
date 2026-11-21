"""Request for the OpenAI-compatible speech endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class VoiceReference(BaseModel):
    """Identify a custom voice by its API ID."""

    id: str = Field(min_length=1)


class SpeechRequest(BaseModel):
    """Select a text-to-speech model, input, voice, and output encoding."""

    model: str = Field(min_length=1)
    input: str = Field(min_length=1, max_length=4096)
    voice: str | VoiceReference
    instructions: str | None = Field(default=None, max_length=4096)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Literal["audio", "sse"] = "audio"
