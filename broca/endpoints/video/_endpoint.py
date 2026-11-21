"""OpenAI-compatible video creation route declaration."""

from typing import Annotated

from fastapi import Form, HTTPException, status

from ._request import VideoRequest
from ._response import VideoResponse
from ._router import video_router


@video_router.post("")
async def create_video(
    body: Annotated[VideoRequest, Form(media_type="multipart/form-data")],
) -> VideoResponse:
    """Declare creation of an asynchronous video generation job.

    :param body: Form fields and optional input reference.
    :returns: The initial video job object.
    :raises HTTPException: Video generation is not implemented.
    """
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
