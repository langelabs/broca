"""Tests for AI endpoint routers and their shared registration."""

import importlib

import pytest
from fastapi import APIRouter, FastAPI

from broca import endpoints

ENDPOINT_PACKAGES = (
    ("audio", "audio", "/audio/speech"),
    ("desicion", "decision", "/decisions"),
    ("embedding", "embedding", "/embeddings"),
    ("images", "images", "/images/generations"),
    ("text", "text", "/chat/completions"),
    ("video", "video", "/videos"),
)


@pytest.mark.parametrize("package_name,modality,path", ENDPOINT_PACKAGES)
def test_endpoint_subpackage_exposes_router(
    package_name: str, modality: str, path: str
) -> None:
    """Register each POST route on its own package router.

    :param package_name: Existing endpoint subpackage name.
    :param modality: Public inference modality name.
    :param path: Route path before including the versioned router.
    :returns: None.
    """
    module = importlib.import_module(f"broca.endpoints.{package_name}")
    router = getattr(module, f"{modality}_router", None)

    assert isinstance(router, APIRouter)
    app = FastAPI()
    app.include_router(router)
    assert "post" in app.openapi()["paths"][path]


def test_main_router_mounts_every_endpoint_subpackage() -> None:
    """Document the public inference routes under the API version prefix."""
    app = FastAPI()
    app.include_router(endpoints.ai_router)

    paths = app.openapi()["paths"]
    assert set(paths) == {
        *(f"/v1{path}" for _, _, path in ENDPOINT_PACKAGES),
        "/v1/models",
        "/v1/models/{model}",
        "/v1/models/{model_id}/download",
        "/v1/models/{model_id}/loaded",
    }
    assert "get" in paths["/v1/models"]
    assert {"get", "delete"} <= paths["/v1/models/{model}"].keys()
    assert "post" in paths["/v1/models/{model_id}/download"]
    assert "put" in paths["/v1/models/{model_id}/loaded"]
