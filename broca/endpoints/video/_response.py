"""Binary MP4 response for video generation."""

from fastapi import Response


class VideoResponse(Response):
    """Send encoded MP4 bytes as the HTTP response body."""

    media_type = "video/mp4"
