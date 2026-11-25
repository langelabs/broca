"""Generate and return MP4 video using the video inference export."""

from typing import Annotated, cast

from fastapi import Form, HTTPException

from ...inference import video
from ...inference.video._types import VideoOptions
from .._inference import require_loaded_model, run_inference
from ._request import VideoRequest
from ._response import VideoResponse
from ._router import video_router

_MAX_REFERENCE_BYTES = 10 * 1024 * 1024


@video_router.post(
    "",
    response_class=VideoResponse,
    responses={
        200: {
            "content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}}}
        }
    },
)
async def create_video(
    body: Annotated[VideoRequest, Form(media_type="multipart/form-data")],
) -> VideoResponse:
    """Generate an MP4 from multipart form fields and an optional uploaded image.

    :param body: Prompt, output dimensions, duration, and optional reference image.
    :returns: Encoded MP4 bytes.
    :raises HTTPException: If the model, generation settings, or upload are invalid.
    """
    model = video._model
    require_loaded_model(model, body.model)
    if (
        body.input_reference_file_id is not None
        or body.input_reference_image_url is not None
    ):
        raise HTTPException(422, "Upload the reference image as a multipart file.")
    if not body.prompt.strip():
        raise HTTPException(422, "Video prompt must not be blank.")

    options = VideoOptions()
    runtime_options = model.runtime_options or {}
    if body.size:
        sizes = cast(list[list[int]], runtime_options["supported_resolutions"])
        supported_sizes = {f"{width}x{height}" for width, height in sizes}
        if body.size not in supported_sizes:
            raise HTTPException(
                422, f"Supported video sizes are {', '.join(sorted(supported_sizes))}."
            )
        width, height = map(int, body.size.split("x"))
        options.width, options.height = width, height
    if body.seconds:
        fps = cast(int, runtime_options["fps"])
        multiple = cast(int, runtime_options["frame_count_multiple"])
        remainder = cast(int, runtime_options["frame_count_remainder"])
        options.num_frames = (
            int(body.seconds) * fps // multiple
        ) * multiple + remainder

    image = None
    if body.input_reference is not None:
        image = await body.input_reference.read(_MAX_REFERENCE_BYTES + 1)
        if not image:
            raise HTTPException(422, "The reference image is empty.")
        if len(image) > _MAX_REFERENCE_BYTES:
            raise HTTPException(413, "The reference image exceeds 10 MiB.")
    result = await run_inference(
        model, video.infer(body.prompt, image=image, options=options)
    )
    return VideoResponse(result.data)
