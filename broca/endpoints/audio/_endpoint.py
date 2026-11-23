"""OpenAI-compatible speech generation backed by the audio inference export."""

from base64 import b64encode

from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

from ...inference import audio
from ...inference.audio._types import AudioOptions, AudioVoice
from .._events import event_frame
from .._inference import require_loaded_model, run_inference
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
    """Generate speech and return encoded audio or server-sent events.

    :param body: Speech input and output settings.
    :returns: Raw audio bytes or server-sent speech events.
    :raises HTTPException: If the voice, settings, or model are unavailable.
    """
    require_loaded_model(audio._model, body.model)
    if body.instructions or body.speed != 1.0:
        raise HTTPException(
            422,
            "Bark uses its default speed and accepts speech text without separate instructions.",
        )
    if not isinstance(body.voice, str) or not body.voice.strip():
        raise HTTPException(
            422, "Use alloy for the default voice or specify a built-in Bark preset."
        )
    if not body.input.strip():
        raise HTTPException(422, "Speech text must not be blank.")

    voice = None if body.voice == "alloy" else AudioVoice(preset=body.voice)
    result = await run_inference(
        audio._model,
        audio.infer(
            body.input,
            voice=voice,
            options=AudioOptions(output_format=body.response_format),
        ),
    )
    if body.stream_format == "sse":
        return StreamingResponse(
            iter(
                (
                    event_frame(
                        {
                            "type": "speech.audio.delta",
                            "audio": b64encode(result.data).decode("ascii"),
                        },
                        event="speech.audio.delta",
                    ),
                    event_frame(
                        {"type": "speech.audio.done"}, event="speech.audio.done"
                    ),
                )
            ),
            media_type="text/event-stream",
        )
    return Response(result.data, media_type=result.media_type)
