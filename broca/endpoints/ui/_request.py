"""Request for a single model test from the dashboard."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from ...inference.text._types import TextAttachment


class ModelTestRequest(BaseModel):
    """Select a model, provide its input, and attach optional media for text generation."""

    model: str
    input: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    attachments: list[TextAttachment] = Field(default_factory=list, max_length=2)
