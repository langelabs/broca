"""OpenAI-compatible speech route declaration."""

from fastapi import HTTPException, Response, status

from ._request import SpeechRequest
from ._response import SpeechResponse
from ._router import audio_router


@audio_router.post(
    "/speech",
    response_class=Response,
    responses={
        200: {
            "description": "Encoded speech or a stream of speech events.",
            "content": {
                "application/octet-stream": {
                    "schema": SpeechResponse.model_json_schema()
                },
                "text/event-stream": {"schema": {"type": "string"}},
            },
        }
    },
)
async def create_speech(body: SpeechRequest) -> Response:
    """Declare speech generation with a binary or event-stream response.

    :param body: Speech input and output settings.
    :returns: Raw audio bytes or server-sent speech events.
    :raises HTTPException: Speech generation is not implemented.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
