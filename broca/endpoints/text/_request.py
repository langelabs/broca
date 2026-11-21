"""Request for the OpenAI-compatible Chat Completions endpoint."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, JsonValue, NonNegativeInt, PositiveInt


class ChatTextPart(BaseModel):
    """Provide text within a multimodal message."""

    type: Literal["text"]
    text: str


class ChatImageURL(BaseModel):
    """Reference an image by URL or data URL."""

    url: str
    detail: Literal["auto", "low", "high", "original"] | None = None


class ChatImagePart(BaseModel):
    """Provide an image within a multimodal message."""

    type: Literal["image_url"]
    image_url: ChatImageURL


class ChatInputAudio(BaseModel):
    """Provide base64-encoded audio in a supported input format."""

    data: str
    format: Literal["wav", "mp3"]


class ChatAudioPart(BaseModel):
    """Provide audio within a multimodal message."""

    type: Literal["input_audio"]
    input_audio: ChatInputAudio


class ChatRefusalPart(BaseModel):
    """Preserve a refusal in assistant message history."""

    type: Literal["refusal"]
    refusal: str


ChatContentPart = Annotated[
    ChatTextPart | ChatImagePart | ChatAudioPart | ChatRefusalPart,
    Field(discriminator="type"),
]


class FunctionCall(BaseModel):
    """Preserve a function name and its serialized JSON arguments."""

    name: str
    arguments: str


class ChatToolCall(BaseModel):
    """Identify a function call for its eventual tool result."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessageRequest(BaseModel):
    """Represent a conversation message, tool call, or tool result."""

    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: str | list[ChatContentPart] | None = None
    name: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None


class FunctionDefinition(BaseModel):
    """Describe a function available to the model."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str | None = None
    parameters: dict[str, JsonValue] | None = None
    strict: bool | None = None


class ChatTool(BaseModel):
    """Declare a client-supplied function tool."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionName(BaseModel):
    """Identify a function selected for tool calling."""

    name: str


class NamedToolChoice(BaseModel):
    """Require a specific named function call."""

    type: Literal["function"] = "function"
    function: FunctionName


class TextResponseFormat(BaseModel):
    """Request plain text or a JSON object."""

    type: Literal["text", "json_object"]


class ResponseJsonSchema(BaseModel):
    """Supply a named JSON Schema for structured output."""

    name: str
    description: str | None = None
    schema_: dict[str, JsonValue] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class JsonSchemaResponseFormat(BaseModel):
    """Request output conforming to a JSON Schema."""

    type: Literal["json_schema"]
    json_schema: ResponseJsonSchema


class ChatStreamOptions(BaseModel):
    """Configure metadata included in a streaming response."""

    include_usage: bool | None = None


class ChatCompletionRequest(BaseModel):
    """Configure a chat completion from conversation history."""

    model: str = Field(min_length=1)
    messages: list[ChatMessageRequest] = Field(min_length=1)
    max_completion_tokens: PositiveInt | None = None
    max_tokens: PositiveInt | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    n: PositiveInt | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    logprobs: bool | None = None
    top_logprobs: NonNegativeInt | None = None
    tools: list[ChatTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: TextResponseFormat | JsonSchemaResponseFormat | None = None
    stream: bool | None = None
    stream_options: ChatStreamOptions | None = None
    user: str | None = None
