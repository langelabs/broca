"""Exercise image generation and stream delivery without model weights."""

import asyncio
import json
from base64 import b64encode
from collections.abc import Callable
from threading import Event
from typing import Any, cast

import pytest
import torch
from diffusers.modular_pipelines.flux3.modular_pipeline import Flux3ModularPipeline
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from broca.endpoints.images import images_router
from broca.inference._pipelines.flux3 import flux_step_callback
from broca.inference.images._pipeline import pipeline
from broca.inference.images.types import ImageOptions, ImageResult


@pytest.fixture
def image_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Mount the image route and mark its shared pipeline as loaded.

    :param monkeypatch: Restores the pipeline after the test.
    :returns: Application containing the public image API.
    """
    monkeypatch.setattr(pipeline, "_runtime", object())
    app = FastAPI()
    app.include_router(images_router, prefix="/v1")
    return app


def test_image_generation_returns_json(
    image_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return the requested size and format with standard base64 output."""

    async def infer(prompt: str, *, options: ImageOptions) -> ImageResult:
        """Replace model computation with a known payload."""
        assert prompt == "a teapot"
        assert (options.width, options.height, options.output_format) == (
            512,
            768,
            "jpeg",
        )
        return ImageResult(
            data=b"\xff\xf8", media_type="image/jpeg", width=512, height=768
        )

    monkeypatch.setattr(pipeline, "_infer", infer)
    with TestClient(image_app) as client:
        response = client.post(
            "/v1/images/generations",
            json={
                "prompt": "a teapot",
                "size": "512x768",
                "output_format": "jpeg",
            },
        )
    assert response.status_code == 200
    assert response.json()["data"] == [{"b64_json": "//g="}]
    assert response.json()["size"] == "512x768"


def test_image_requests_return_results_independently(
    image_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release a completed image while the next request is still generating."""
    release_second = Event()

    async def infer(prompt: str) -> ImageResult:
        """Keep the second generation pending until the first result arrives."""
        if prompt == "second":
            assert release_second.wait(3)
        return ImageResult(
            data=prompt.encode(), media_type="image/png", width=1024, height=1024
        )

    monkeypatch.setattr(pipeline, "_infer", infer)

    async def check() -> None:
        """Submit two requests concurrently and require independent completion."""
        first = asyncio.create_task(pipeline.infer("first"))
        second = asyncio.create_task(pipeline.infer("second"))
        try:
            assert (await asyncio.wait_for(first, 1)).data == b"first"
            assert not second.done()
        finally:
            release_second.set()
            await second

    asyncio.run(check())


def test_disconnected_generation_stops_at_next_denoising_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop native generation on a denoising boundary even for a final-only stream."""
    stopped = Event()

    class Runtime:
        """Model a FLUX runtime advancing to one denoising step."""

        def __call__(self, **_kwargs: object) -> None:
            """Signal a disconnected client and call the active step callback."""
            stopped.set()
            callback = flux_step_callback.get()
            assert callback is not None
            callback(cast(Flux3ModularPipeline, self), torch.empty(0), 1)
            pytest.fail("Generation continued after the client disconnected")

    monkeypatch.setattr(pipeline, "_runtime", Runtime())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pipeline.infer("a teapot", cancelled=stopped))


@pytest.mark.parametrize("disconnect", [False, True])
def test_stream_sends_partial_before_inference_finishes(
    image_app: FastAPI, monkeypatch: pytest.MonkeyPatch, disconnect: bool
) -> None:
    """Deliver a partial frame and close the producer if sending it fails."""
    partial_delivered = Event()
    stop_observed = Event()

    async def infer(
        prompt: str,
        *,
        options: ImageOptions,
        partial_images: int,
        on_preview: Callable[[int, bytes], None],
        cancelled: Event,
    ) -> ImageResult:
        """Wait for ASGI delivery before returning the final result."""
        assert partial_images == 1
        assert not cancelled.is_set()
        on_preview(0, b"preview")
        assert partial_delivered.wait(2)
        if disconnect and cancelled.wait(2):
            stop_observed.set()
        return ImageResult(
            data=b"final", media_type="image/png", width=1024, height=1024
        )

    monkeypatch.setattr(pipeline, "_infer", infer)

    async def request() -> list[dict[str, Any]]:
        """Capture SSE messages as the application sends them."""
        messages: list[dict[str, Any]] = []
        incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        incoming.put_nowait(
            {
                "type": "http.request",
                "body": json.dumps(
                    {
                        "prompt": "a teapot",
                        "stream": True,
                        "partial_images": 1,
                    }
                ).encode(),
            }
        )

        async def send(message: dict[str, Any]) -> None:
            """Release the worker after a preview reaches the ASGI transport."""
            messages.append(message)
            if message.get("body", b"").startswith(
                b"event: image_generation.partial_image\n"
            ):
                partial_delivered.set()
                if disconnect:
                    raise OSError("Client closed the connection")

        try:
            await asyncio.wait_for(
                image_app(
                    {
                        "type": "http",
                        "asgi": {"spec_version": "2.4"},
                        "method": "POST",
                        "path": "/v1/images/generations",
                        "headers": [(b"content-type", b"application/json")],
                        "query_string": b"",
                        "scheme": "http",
                    },
                    incoming.get,
                    send,
                ),
                timeout=4,
            )
        except ClientDisconnect:
            assert disconnect
            assert await asyncio.to_thread(stop_observed.wait, 2)
        return messages

    messages = asyncio.run(request())
    assert messages[0]["status"] == 200
    assert dict(messages[0]["headers"])[b"content-type"].startswith(
        b"text/event-stream"
    )
    events = [
        json.loads(message["body"].splitlines()[1].removeprefix(b"data: "))
        for message in messages[1:]
        if message.get("body", b"").startswith(b"event: ")
    ]
    if disconnect:
        assert stop_observed.is_set()
        assert [event["type"] for event in events] == ["image_generation.partial_image"]
        return
    assert [event["type"] for event in events] == [
        "image_generation.partial_image",
        "image_generation.completed",
    ]
    assert events[0]["partial_image_index"] == 0
    assert events[0]["b64_json"] == b64encode(b"preview").decode("ascii")
    assert events[1]["b64_json"] == b64encode(b"final").decode("ascii")
    assert [event["quality"] for event in events] == ["auto", "auto"]


@pytest.mark.parametrize("fail", [False, True])
def test_stream_with_zero_partials_sends_one_terminal_event(
    image_app: FastAPI, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """End a final-only stream with either an image or a public error."""

    async def infer(
        prompt: str,
        *,
        options: ImageOptions,
        partial_images: int,
        on_preview: Callable[[int, bytes], None],
        cancelled: Event,
    ) -> ImageResult:
        """Return the final image or simulate a runtime failure."""
        assert partial_images == 0
        if fail:
            raise RuntimeError("private runtime details")
        return ImageResult(
            data=b"final", media_type="image/png", width=1024, height=1024
        )

    monkeypatch.setattr(pipeline, "_infer", infer)
    with TestClient(image_app) as client:
        response = client.post(
            "/v1/images/generations", json={"prompt": "a teapot", "stream": True}
        )
    assert response.status_code == 200
    event_names = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert event_names == ["error" if fail else "image_generation.completed"]
    assert "private runtime details" not in response.text


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"model": "lab/missing"}, 404),
        ({"size": "65536x65536"}, 422),
        ({"partial_images": 1}, 422),
        ({"output_format": "svg"}, 422),
        ({}, 409),
    ],
)
def test_image_validation_precedes_inference(
    fields: dict[str, object], expected: int
) -> None:
    """Reject unavailable models and unsupported requests before computing."""
    app = FastAPI()
    app.include_router(images_router, prefix="/v1")
    with TestClient(app) as client:
        response = client.post(
            "/v1/images/generations", json={"prompt": "a teapot", **fields}
        )
    assert response.status_code == expected
