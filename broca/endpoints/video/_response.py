"""Response for the OpenAI-compatible video creation endpoint."""

from typing import Literal

from pydantic import BaseModel, Field, NonNegativeInt


class VideoError(BaseModel):
    """Describe a failed video job."""

    code: str
    message: str


class VideoResponse(BaseModel):
    """Return the status of the asynchronous video generation job."""

    id: str
    object: Literal["video"] = "video"
    model: str
    status: Literal["queued", "in_progress", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    created_at: NonNegativeInt
    size: str
    seconds: str
    prompt: str | None = None
    completed_at: NonNegativeInt | None = None
    expires_at: NonNegativeInt | None = None
    error: VideoError | None = None
    remixed_from_video_id: str | None = None
