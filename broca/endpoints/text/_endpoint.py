"""OpenAI-compatible chat completions backed by the text inference export."""

from time import time
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ...inference import text
from ...inference.text.types import (
    AudioAttachment,
    ImageAttachment,
    TextAttachment,
    TextMessage,
    TextOptions,
    TextRequest,
)
from .._events import event_frame
from .._inference import require_loaded_model, run_inference
from ._request import ChatCompletionRequest
from ._response import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionUsage,
)
from ._router import text_router


@text_router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
async def create_chat_completion(
    body: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Generate one completion for an ordered conversation.

    :param body: Messages, model, and generation settings.
    :returns: A completion object or SSE frames for the completed response.
    :raises HTTPException: If settings, messages, or the model are unavailable.
    """
    require_loaded_model(text._model, body.model)
    if (
        body.n not in (None, 1)
        or body.frequency_penalty not in (None, 0)
        or body.presence_penalty not in (None, 0)
        or body.logprobs
        or body.top_logprobs not in (None, 0)
        or (body.tools and body.tool_choice != "none")
        or body.tool_choice not in (None, "none", "auto")
        or (body.response_format and body.response_format.type != "text")
    ):
        raise HTTPException(
            422, "The text model does not support these generation settings."
        )
    if body.messages[-1].role != "user":
        raise HTTPException(422, "The conversation must end with a user message.")

    instructions: list[str] = []
    history: list[TextMessage] = []
    attachments: list[TextAttachment] = []
    prompt = ""
    try:
        for index, message in enumerate(body.messages):
            current = index == len(body.messages) - 1
            if message.role == "tool" or message.tool_calls or message.tool_call_id:
                raise HTTPException(
                    422, "The text model does not support tool messages."
                )
            content = message.content or ""
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    match part.type:
                        case "text":
                            parts.append(part.text)
                        case "refusal":
                            parts.append(part.refusal)
                        case _ if not current:
                            raise HTTPException(
                                422, "Media is supported on the final user message."
                            )
                        case "input_audio":
                            attachments.append(
                                AudioAttachment(
                                    type="input_audio", **part.input_audio.model_dump()
                                )
                            )
                        case "image_url":
                            prefix, _, data = part.image_url.url.partition(",")
                            if prefix not in (
                                "data:image/png;base64",
                                "data:image/jpeg;base64",
                            ):
                                raise HTTPException(
                                    422, "Use an inline PNG or JPEG data URL."
                                )
                            attachments.append(
                                ImageAttachment(
                                    type="image",
                                    format="png"
                                    if prefix.startswith("data:image/png")
                                    else "jpeg",
                                    data=data,
                                )
                            )
                content = "\n".join(parts)
            if not content and message.refusal:
                content = message.refusal
            if current:
                prompt = content
            elif message.role in ("system", "developer"):
                if history:
                    raise HTTPException(
                        422,
                        "Put system and developer messages before the conversation.",
                    )
                instructions.append(content)
            elif message.role in ("user", "assistant"):
                history.append(TextMessage(role=message.role, content=content))

        if not prompt.strip() and not attachments:
            raise HTTPException(422, "Provide text or media in the final user message.")
        options = TextOptions(
            max_tokens=body.max_completion_tokens or body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=[body.stop] if isinstance(body.stop, str) else body.stop,
            seed=body.seed,
        )
    except ValidationError as error:
        raise HTTPException(422, "Invalid chat settings or inline media.") from error

    result = await run_inference(
        text._model,
        text.infer(
            TextRequest(
                prompt, "\n\n".join(instructions) or None, history, options, attachments
            )
        ),
    )
    usage = None
    if result.usage is not None:
        usage = ChatCompletionUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.prompt_tokens + result.usage.completion_tokens,
        )
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time())
    if body.stream:
        chunk: dict[str, object] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
        }
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        if include_usage:
            chunk["usage"] = None
        frames = [
            event_frame(
                {
                    **chunk,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": result.text},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            event_frame(
                {
                    **chunk,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": result.finish_reason}
                    ],
                }
            ),
        ]
        if include_usage and usage is not None:
            frames.append(
                event_frame(
                    {
                        **chunk,
                        "choices": [],
                        "usage": usage.model_dump(exclude_none=True),
                    }
                )
            )
        frames.append("data: [DONE]\n\n")
        return StreamingResponse(iter(frames), media_type="text/event-stream")
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                finish_reason=result.finish_reason,
                message=ChatCompletionMessage(content=result.text),
            )
        ],
        usage=usage,
    )
