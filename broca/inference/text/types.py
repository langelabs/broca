"""Pydantic conversation, settings, and results for text inference."""

from base64 import b64decode
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)


_MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024


class _EncodedAttachment(BaseModel):
    """Carry inline media bounded to 10 MiB per attachment."""

    model_config = ConfigDict(extra="forbid")

    data: str = Field(min_length=1, max_length=4 * ((_MAX_ATTACHMENT_BYTES + 2) // 3))

    @field_validator("data")
    @classmethod
    def validate_data(cls, value: str) -> str:
        """Require inline base64 before forwarding to the native media decoder.

        :param value: Encoded attachment bytes.
        :returns: The validated base64 string.
        :raises ValueError: If the encoding or decoded size is invalid.
        """
        try:
            decoded = b64decode(value, validate=True)
        except ValueError as error:
            raise ValueError("Attachment data must be valid base64.") from error
        if not 0 < len(decoded) <= _MAX_ATTACHMENT_BYTES:
            raise ValueError("Attachment data must contain between 1 byte and 10 MiB.")
        return value


class ImageAttachment(_EncodedAttachment):
    """Provide an inline image for the current user turn."""

    type: Literal["image"]
    format: Literal["png", "jpeg"]


class AudioAttachment(_EncodedAttachment):
    """Provide inline audio for the current user turn."""

    type: Literal["input_audio"]
    format: Literal["wav", "mp3"]


TextAttachment = Annotated[
    ImageAttachment | AudioAttachment, Field(discriminator="type")
]


class TextMessage(BaseModel):
    """Represent a previous user or assistant turn in conversation order."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class TextOptions(BaseModel):
    """Override configured sampling defaults without selecting a runtime."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    max_tokens: PositiveInt | None = None
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: NonNegativeInt | None = None
    stop: list[str] | None = None
    seed: NonNegativeInt | None = None
    enable_thinking: bool | None = None


@dataclass(frozen=True)
class TextRequest:
    """Carry one conversation turn and its generation settings into the batch hook."""

    prompt: str
    system_prompt: str | None = None
    history: Sequence[TextMessage] | None = None
    options: TextOptions | None = None
    attachments: Sequence[TextAttachment] = ()


class TokenUsage(BaseModel):
    """Report prompt and generated token counts when the backend provides them."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: NonNegativeInt
    completion_tokens: NonNegativeInt


class TextResult(BaseModel):
    """Return one completed response with optional reasoning and token usage."""

    model_config = ConfigDict(extra="forbid")

    text: str
    finish_reason: Literal["stop", "length"]
    reasoning: str | None = None
    usage: TokenUsage | None = None
