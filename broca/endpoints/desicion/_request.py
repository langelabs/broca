"""Request for the OpenAI-compatible Decisions endpoint."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, JsonValue, StrictBool, StrictStr

ChoiceValue = StrictStr | StrictBool
ToolStatus = Literal["in_progress", "completed", "incomplete"]


class DecisionText(BaseModel):
    """Provide a text content part."""

    type: Literal["input_text"]
    text: str


class DecisionImage(BaseModel):
    """Reference an image using a data URL or a public HTTP(S) URL."""

    type: Literal["input_image"]
    image_url: str = Field(pattern=r"^(data:|https?://)")
    detail: Literal["low", "high", "auto", "original"] | None = None


class DecisionJson(BaseModel):
    """Provide structured evidence as a JSON object or array of objects."""

    type: Literal["input_json"]
    data: dict[str, JsonValue] | list[dict[str, JsonValue]]


class DecisionCacheBreakpoint(BaseModel):
    """Mark the end of a reusable prefix."""

    mode: Literal["explicit"] = "explicit"


class DecisionInputText(DecisionText):
    """Provide text with an optional prompt cache breakpoint."""

    prompt_cache_breakpoint: DecisionCacheBreakpoint | None = None


class DecisionInputImage(DecisionImage):
    """Provide an image with an optional prompt cache breakpoint."""

    prompt_cache_breakpoint: DecisionCacheBreakpoint | None = None


DecisionToolOutputPart = Annotated[
    DecisionInputText | DecisionInputImage, Field(discriminator="type")
]
DecisionInputPart = Annotated[
    DecisionInputText | DecisionInputImage | DecisionJson, Field(discriminator="type")
]
DecisionQuestionPart = Annotated[
    DecisionText | DecisionImage | DecisionJson, Field(discriminator="type")
]


class DecisionUserMessage(BaseModel):
    """Provide shared text, image, or JSON evidence."""

    role: Literal["user"]
    content: str | list[DecisionInputPart]
    type: Literal["message"] = "message"


class DecisionInstructionMessage(BaseModel):
    """Provide developer or system instructions as text."""

    role: Literal["developer", "system"]
    content: str | list[DecisionInputText]
    type: Literal["message"] = "message"


class DecisionOutputText(BaseModel):
    """Include a previous assistant text output in the evidence."""

    type: Literal["output_text"]
    text: str


class DecisionInputRefusal(BaseModel):
    """Include a previous assistant refusal in the evidence."""

    type: Literal["refusal"]
    refusal: str


class DecisionAssistantMessage(BaseModel):
    """Provide a previous assistant message."""

    role: Literal["assistant"]
    content: str | list[DecisionOutputText | DecisionInputRefusal]
    phase: Literal["commentary", "final_answer"] | None = None
    type: Literal["message"] = "message"


DecisionMessage = Annotated[
    DecisionUserMessage | DecisionInstructionMessage | DecisionAssistantMessage,
    Field(discriminator="role"),
]


class DecisionToolItem(BaseModel):
    """Identify a tool call or result in the shared conversation."""

    call_id: str
    id: str | None = None
    status: ToolStatus | None = None


class DecisionFunctionCall(DecisionToolItem):
    """Include a previous call with serialized function arguments."""

    type: Literal["function_call"]
    name: str
    arguments: str
    namespace: str | None = None


class DecisionCustomToolCall(DecisionToolItem):
    """Include a previous call to a custom tool."""

    type: Literal["custom_tool_call"]
    name: str
    input: str
    namespace: str | None = None


class DecisionToolOutput(DecisionToolItem):
    """Include a tool's output and associate it with its previous call."""

    type: Literal["function_call_output", "custom_tool_call_output"]
    output: str | list[DecisionToolOutputPart]


DecisionInputItem = (
        DecisionMessage | DecisionFunctionCall | DecisionCustomToolCall | DecisionToolOutput
)


class DecisionQuestion(BaseModel):
    """Give one question its own instructions and optional name."""

    name: str | None = None
    instructions: str | Annotated[list[DecisionQuestionPart], Field(min_length=1)]


class PredicateQuestion(DecisionQuestion):
    """Ask how likely a statement about the evidence is to be true."""

    type: Literal["predicate"]


class DecisionChoice(BaseModel):
    """Identify an available answer by a string or boolean value."""

    value: ChoiceValue
    description: str | None = None


class ChoiceQuestion(DecisionQuestion):
    """Ask the model to choose one of the supplied options."""

    type: Literal["choice"]
    choices: list[DecisionChoice] = Field(min_length=2, max_length=255)


class DecisionScoreLevel(BaseModel):
    """Describe one level of an ordered rubric."""

    label: str
    description: str | None = None


class ScoreQuestion(DecisionQuestion):
    """Ask the model to rate the evidence against an ordered rubric."""

    type: Literal["score"]
    levels: list[DecisionScoreLevel] = Field(min_length=2, max_length=10)


class DecisionPromptCacheOptions(BaseModel):
    """Configure how reusable prompt prefixes are selected."""

    mode: Literal["implicit", "explicit"] | None = None
    ttl: Literal["30m"] = "30m"


class CreateDecisionRequest(BaseModel):
    """Evaluate several typed questions against shared evidence."""

    model: str = Field(min_length=1)
    input: str | list[DecisionInputItem]
    questions: list[
        Annotated[
            PredicateQuestion | ChoiceQuestion | ScoreQuestion,
            Field(discriminator="type"),
        ]
    ] = Field(min_length=1, max_length=400)
    prompt_cache_key: str | None = Field(default=None, max_length=64)
    prompt_cache_options: DecisionPromptCacheOptions | None = None
    safety_identifier: str | None = Field(default=None, max_length=64)
