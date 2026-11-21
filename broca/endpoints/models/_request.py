"""Path parameters for operations on a configured model."""

from typing import Annotated

from fastapi import Path
from pydantic import BaseModel, RootModel, StrictBool


class ModelRequest(BaseModel):
    """Identify a model by its catalog ID, including the repository namespace."""

    model: Annotated[str, Path(min_length=1)]


class ModelLoadedRequest(RootModel[StrictBool]):
    """Set the desired loaded state using a JSON boolean."""
