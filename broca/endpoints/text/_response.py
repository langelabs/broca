"""Response for the OpenAI-compatible Chat Completions endpoint."""

from typing import Literal

from pydantic import BaseModel, JsonValue, NonNegativeInt

from ._request import ChatToolCall


class ChatCompletionMessage(BaseModel):
    """Return the assistant's text, refusal, or tool calls."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ChatToolCall] | None = None


class ChatCompletionChoice(BaseModel):
    """Identify a generated choice and why it ended."""

    index: NonNegativeInt
    finish_reason: Literal[
        "stop", "length", "tool_calls", "content_filter", "function_call"
    ]
    message: ChatCompletionMessage
    logprobs: dict[str, JsonValue] | None = None


class ChatCompletionUsage(BaseModel):
    """Report prompt, completion, and total token counts."""

    prompt_tokens: NonNegativeInt
    completion_tokens: NonNegativeInt
    total_tokens: NonNegativeInt
    prompt_tokens_details: dict[str, NonNegativeInt] | None = None
    completion_tokens_details: dict[str, NonNegativeInt] | None = None


class ChatCompletionResponse(BaseModel):
    """Return a Chat Completions response envelope."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: NonNegativeInt
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
