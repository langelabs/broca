"""Multipart request for video generation."""

from typing import Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field


class VideoRequest(BaseModel):
    """Describe a video to generate from multipart form data."""

    prompt: str = Field(min_length=1, max_length=32000)
    model: str | None = None
    seconds: Literal["4", "8", "12"] | None = None
    size: str | None = Field(default=None, pattern=r"^[1-9][0-9]*x[1-9][0-9]*$")
    input_reference: UploadFile | None = None
    input_reference_file_id: str | None = Field(
        default=None, alias="input_reference[file_id]"
    )
    input_reference_image_url: str | None = Field(
        default=None, alias="input_reference[image_url]"
    )
