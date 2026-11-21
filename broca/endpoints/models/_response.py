"""Public response models for the OpenAI-compatible catalog."""

from datetime import date
from typing import Literal, Self

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, NonNegativeInt

from ...ai_model import AiModel, ModelModality
from ...inference import is_model_loaded


class ModelResponse(BaseModel):
    """Expose the public Model fields without serializing internal settings."""

    id: str
    modality: ModelModality
    object: Literal["model"] = "model"
    created: NonNegativeInt
    owned_by: str
    shutdown_date: date | None = None
    size: NonNegativeInt | None = Field(
        description="Estimated resident memory in bytes, when known."
    )
    downloaded: bool
    download_progress: float = Field(
        ge=0.0, le=1.0, description="Current download progress as a fraction."
    )
    loaded: bool

    @classmethod
    async def from_model(cls, model: AiModel) -> Self:
        """Build public metadata with the model's current lifecycle state.

        :param model: Inference model configuration from the catalog.
        :returns: Public metadata, including size, download, and runtime state.
        :raises ValueError: If the model configuration has no modality.
        """
        modality = "text" if model.type == "chat" else model.type
        if modality is None:
            raise ValueError("Catalog models require an inference modality.")
        progress = await model.download_progress()
        return cls(
            id=model.id,
            modality=modality,
            created=model.created_at or 0,
            owned_by=model.id.partition("/")[0],
            size=model.size,
            downloaded=progress == 1.0,
            download_progress=progress,
            loaded=is_model_loaded(model),
        )


class ModelListResponse(BaseModel):
    """Return all configured models in the Models list envelope."""

    object: Literal["list"] = "list"
    data: list[ModelResponse]


class ModelDeletedResponse(BaseModel):
    """Describe successful deletion of a fine-tuned model."""

    id: str
    object: Literal["model"] = "model"
    deleted: bool


class ModelDownloadResponse(BaseModel):
    """Describe whether the configured assets are locally available."""

    id: str
    downloaded: bool


class ModelLoadedResponse(BaseModel):
    """Describe the resulting runtime state for a configured model."""

    id: str
    loaded: bool


class ModelError(BaseModel):
    """Describe an unsuccessful Models request in the OpenAI wire format."""

    message: str
    type: Literal["invalid_request_error", "server_error"] = "invalid_request_error"
    param: str | None = None
    code: Literal[
        "model_not_found",
        "model_not_deletable",
        "model_download_failed",
        "model_assets_missing",
        "model_runtime_failed",
    ]


class ModelErrorResponse(BaseModel):
    """Wrap the public error fields under the `error` key."""

    error: ModelError

    def to_response(self, status_code: int) -> JSONResponse:
        """Send the error envelope with the status selected by the route.

        :param status_code: HTTP status of the failed Models request.
        :returns: JSON response with the public error envelope.
        """
        return JSONResponse(
            status_code=status_code, content=self.model_dump(mode="json")
        )
