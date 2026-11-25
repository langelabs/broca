"""Generate images with the loaded FLUX runtime."""

from base64 import b64encode
from time import time
from typing import cast

from fastapi.responses import StreamingResponse

from ...inference import images
from ...inference.images.types import ImageOptions, RasterFormat
from .._inference import require_loaded_model, run_inference
from ._request import ImageGenerationRequest
from ._response import GeneratedImage, ImageGenerationResponse
from ._router import images_router
from ._stream import ImageEventStream


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
    require_loaded_model(images._model, body.model)
    defaults = images._model.generation_options or {}
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
        return ImageEventStream(body.prompt, options, body.partial_images or 0)
    image = await run_inference(
        images._model, images.infer(body.prompt, options=options)
    )
    return ImageGenerationResponse(
        created=int(time()),
        data=[GeneratedImage(b64_json=b64encode(image.data).decode("ascii"))],
        background="opaque",
        output_format=options.output_format,
        size=f"{image.width}x{image.height}",
    )
