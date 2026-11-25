"""Run the loaded model selected on a modality test page."""

import logging

from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from ....ai_model import ModelModality
from ....inference import (
    audio,
    decision,
    embedding,
    get_model,
    images,
    is_model_loaded,
    text,
    video,
)
from ....inference.decision.types import DecisionRequest
from .._request import ModelTestRequest
from .._router import ui_router

logger = logging.getLogger(__name__)


@ui_router.post("/{modality}")
async def run_model_test(modality: ModelModality, body: ModelTestRequest) -> Response:
    """Invoke the modality's public inference function using a minimal test input.

    :param modality: Inference modality selected by the test page.
    :param body: Configured model ID, prompt or decision JSON, and optional text attachments.
    :returns: A JSON result or the encoded output media.
    :raises HTTPException: If the model, input, or runtime is unavailable.
    """
    model = get_model(body.model)
    if model is None or model.type != modality:
        raise HTTPException(
            status_code=404, detail="Model not found for this modality."
        )
    if body.attachments and modality != "text":
        raise HTTPException(
            status_code=422,
            detail="Media attachments are available for text generation.",
        )
    if not is_model_loaded(model):
        raise HTTPException(
            status_code=409, detail="Load the model from the dashboard first."
        )

    try:
        match modality:
            case "audio":
                speech = await audio.infer(body.input)
                return Response(speech.data, media_type=speech.media_type)
            case "image":
                image = await images.infer(body.input)
                return Response(image.data, media_type=image.media_type)
            case "video":
                clip = await video.infer(body.input)
                return Response(clip.data, media_type=clip.media_type)
            case "decision":
                answer = await decision.infer(
                    DecisionRequest.model_validate_json(body.input)
                )
                return JSONResponse(answer.model_dump(mode="json"))
            case "embedding":
                vectors = await embedding.infer([body.input])
                return JSONResponse(
                    vectors.model_dump(mode="json", exclude={"prompt_tokens"})
                )
            case "text":
                completion = await text.infer(body.input, attachments=body.attachments)
                return JSONResponse(completion.model_dump(mode="json"))
    except ValidationError as error:
        message = error.errors(include_input=False)[0]["msg"]
        raise HTTPException(status_code=422, detail=message) from error
    except Exception as error:
        if not is_model_loaded(model):
            raise HTTPException(
                status_code=409, detail="The model was unloaded."
            ) from error
        logger.error("Model test failed for %s (%s)", model.id, type(error).__name__)
        raise HTTPException(
            status_code=503, detail="Model inference failed."
        ) from error
