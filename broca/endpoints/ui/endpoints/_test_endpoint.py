"""Run the loaded model selected on a modality test page."""

import logging

from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from ....ai_model import ModelModality
from ....inference import get_model, is_model_loaded
from .._request import ModelTestRequest
from .._router import ui_router

logger = logging.getLogger(__name__)


@ui_router.post("/{modality}")
async def run_model_test(modality: ModelModality, body: ModelTestRequest) -> Response:
    """Invoke the configured pipeline using a minimal test input.

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
                from ....inference.audio._pipeline import pipeline as audio_pipeline

                audio = await audio_pipeline.infer(body.input)
                return Response(audio.data, media_type=audio.media_type)
            case "image":
                from ....inference.images._pipeline import pipeline as image_pipeline

                image = await image_pipeline.infer(body.input)
                return Response(image.data, media_type=image.media_type)
            case "video":
                from ....inference.video._pipeline import pipeline as video_pipeline

                video = await video_pipeline.infer(body.input)
                return Response(video.data, media_type=video.media_type)
            case "decision":
                from ....inference.decision._pipeline import (
                    pipeline as decision_pipeline,
                )
                from ....inference.decision._types import DecisionRequest

                decision = await decision_pipeline.infer(
                    DecisionRequest.model_validate_json(body.input)
                )
                return JSONResponse(decision.model_dump(mode="json"))
            case "embedding":
                from ....inference.embedding._pipeline import (
                    pipeline as embedding_pipeline,
                )

                embedding = await embedding_pipeline.infer([body.input])
                return JSONResponse(embedding.model_dump(mode="json"))
            case "text":
                from ....inference.text._pipeline import pipeline as text_pipeline

                text = await text_pipeline.infer(
                    body.input, attachments=body.attachments
                )
                return JSONResponse(text.model_dump(mode="json"))
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
