"""Generate images with the loaded FLUX runtime."""

import logging
from base64 import b64encode
from time import time
from typing import cast

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ...inference import is_model_loaded
from ...inference.images import _model
from ...inference.images._types import ImageOptions, RasterFormat
from ._request import ImageGenerationRequest
from ._response import GeneratedImage, ImageGenerationResponse
from ._router import images_router
from ._stream import ImageEventStream

logger = logging.getLogger(__name__)


@images_router.post(
    "/generations",
    response_model=ImageGenerationResponse,
    response_model_exclude_none=True,
    responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}},
)
async def create_image(
    body: ImageGenerationRequest,
) -> ImageGenerationResponse | StreamingResponse:
    """Generate one image in the JSON or server-sent event format.

    :param body: Prompt and image output settings.
    :returns: Generated images or server-sent image events.
    :raises HTTPException: If the model is unavailable or inference fails.
    """
    if body.model not in (None, _model.id):
        raise HTTPException(404, "Image model not found.")
    if not is_model_loaded(_model):
        raise HTTPException(409, "Load the image model before generating an image.")

    from ...inference.images._pipeline import pipeline

    defaults = _model.generation_options or {}
    width = cast(int, defaults.get("width", 1024))
    height = cast(int, defaults.get("height", 1024))
    if body.size is not None and body.size != "auto":
        width, height = map(int, body.size.split("x"))
    options = ImageOptions(
        width=width,
        height=height,
        output_format=cast(RasterFormat, body.output_format or "png"),
    )
    if body.stream:
        return ImageEventStream(
            pipeline, body.prompt, options, body.partial_images or 0
        )
    try:
        image = await pipeline.infer(body.prompt, options=options)
    except Exception as error:
        logger.error("Image generation failed (%s)", type(error).__name__)
        raise HTTPException(503, "Image generation failed.") from error
    return ImageGenerationResponse(
        created=int(time()),
        data=[GeneratedImage(b64_json=b64encode(image.data).decode("ascii"))],
        background="opaque",
        output_format=options.output_format,
        size=f"{image.width}x{image.height}",
    )
