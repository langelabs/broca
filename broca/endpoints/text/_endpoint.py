"""OpenAI-compatible Chat Completions route declaration."""

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from ._request import ChatCompletionRequest
from ._response import ChatCompletionResponse
from ._router import text_router


@text_router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
async def create_chat_completion(
    body: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Declare a chat completion or a stream of completion events.

    :param body: Messages, model, and generation settings.
    :returns: A completion object or server-sent completion events.
    :raises HTTPException: Chat completion is not implemented.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
