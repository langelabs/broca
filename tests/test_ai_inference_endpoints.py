"""Exercise HTTP request and response mapping at the inference exports."""

from base64 import b64decode
from struct import unpack
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from broca.endpoints import router
from broca.inference import audio, decision, embedding, images, text, video
from broca.inference.audio._types import AudioResult
from broca.inference.decision.types import ChoiceResult, NoulResult, ScoreResult
from broca.inference.embedding.types import EmbeddingResult
from broca.inference.images.types import ImageResult
from broca.inference.text.types import TextResult, TokenUsage
from broca.inference.video._types import VideoResult


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Mount the full router with loaded pipeline sentinels and no model weights."""
    app = FastAPI()
    app.include_router(router)
    for module in (audio, decision, embedding, images, text, video):
        monkeypatch.setattr(module.pipeline, "_runtime", object())
    return TestClient(app)


@pytest.mark.parametrize("stream", [False, True])
def test_chat_completions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """Pass ordered turns and generation settings to text inference."""
    infer = AsyncMock(
        return_value=TextResult(
            text="Umbrella",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=8, completion_tokens=1),
        )
    )
    monkeypatch.setattr(text, "infer", infer)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": text.pipeline.model.id,
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "It is raining."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "What should I bring?"},
            ],
            "max_completion_tokens": 10,
            "stream": stream,
        },
    )

    assert response.status_code == 200, response.text
    request = infer.call_args.args[0]
    assert request.system_prompt == "Be concise."
    assert [message.content for message in request.history] == [
        "It is raining.",
        "Okay.",
    ]
    assert request.prompt == "What should I bring?"
    assert request.options.max_tokens == 10
    if stream:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"content":"Umbrella"' in response.text
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Umbrella"
        assert data["usage"]["total_tokens"] == 9


def test_chat_rejects_remote_media(client: TestClient) -> None:
    """Require inline media so chat input cannot trigger outbound URL fetching."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": text.pipeline.model.id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://example.test/image.png"},
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("stream_format", ["audio", "sse"])
def test_speech(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, stream_format: str
) -> None:
    """Deliver encoded audio through the requested response transport."""
    infer = AsyncMock(
        return_value=AudioResult(
            data=b"audio",
            media_type="audio/mpeg",
            sample_rate=24000,
            channels=1,
            duration_seconds=1.0,
        )
    )
    monkeypatch.setattr(audio, "infer", infer)
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": audio.pipeline.model.id,
            "input": "Hello",
            "voice": "alloy",
            "stream_format": stream_format,
        },
    )

    assert response.status_code == 200, response.text
    assert infer.call_args.args == ("Hello",)
    assert infer.call_args.kwargs["options"].output_format == "mp3"
    if stream_format == "sse":
        assert "speech.audio.delta" in response.text
        assert "YXVkaW8=" in response.text
        assert "speech.audio.done" in response.text
    else:
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"audio"


def test_decisions(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return predicate, choice, and score results in question order."""
    infer = AsyncMock(
        side_effect=[
            NoulResult(probability_yes=0.8),
            ChoiceResult(selected_id="1", probabilities={"0": 0.3, "1": 0.7}),
            ScoreResult(
                score=0.2, normalized_score=0.2, probabilities={"0": 0.8, "1": 0.2}
            ),
        ]
    )
    monkeypatch.setattr(decision, "infer", infer)
    response = client.post(
        "/v1/decisions",
        json={
            "model": decision.pipeline.model.id,
            "input": "The sunflower is thriving.",
            "questions": [
                {
                    "type": "predicate",
                    "name": "growing",
                    "instructions": "Is it growing?",
                },
                {
                    "type": "choice",
                    "instructions": "Pick its species.",
                    "choices": [{"value": "rose"}, {"value": "sunflower"}],
                },
                {
                    "type": "score",
                    "instructions": "Rate the photo.",
                    "levels": [{"label": "poor"}, {"label": "good"}],
                },
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert [call.args[0].decision.kind for call in infer.call_args_list] == [
        "noul",
        "choice",
        "score",
    ]
    assert all(
        call.args[0].state == "The sunflower is thriving."
        for call in infer.call_args_list
    )
    result = response.json()["answers"]
    assert result[0] == {"type": "predicate", "name": "growing", "probability": 0.8}
    assert result[1]["choice"] == "sunflower"
    assert result[2]["probabilities"][0] == {
        "label": "poor",
        "value": 0,
        "probability": 0.8,
    }


def test_embeddings(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return float32 base64 vectors and usage in input order."""
    infer = AsyncMock(
        return_value=EmbeddingResult(
            vectors=[[0.5] * 128, [-0.25] * 128],
            dimensions=128,
            normalized=True,
            prompt_tokens=12,
        )
    )
    monkeypatch.setattr(embedding, "infer", infer)
    response = client.post(
        "/v1/embeddings",
        json={
            "model": embedding.pipeline.model.id,
            "input": ["first", "second"],
            "dimensions": 128,
            "encoding_format": "base64",
        },
    )

    assert response.status_code == 200, response.text
    assert infer.call_args.args == (["first", "second"],)
    assert infer.call_args.kwargs["options"].dimensions == 128
    data = response.json()
    assert data["usage"] == {"prompt_tokens": 12, "total_tokens": 12}
    assert [item["index"] for item in data["data"]] == [0, 1]
    assert unpack("<128f", b64decode(data["data"][1]["embedding"])) == (-0.25,) * 128


@pytest.mark.parametrize("stream", [False, True])
def test_images(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """Use the same public image inference export for JSON and events."""
    infer = AsyncMock(
        return_value=ImageResult(
            data=b"image", media_type="image/png", width=1024, height=1024
        )
    )
    monkeypatch.setattr(images, "infer", infer)
    response = client.post(
        "/v1/images/generations", json={"prompt": "An apple", "stream": stream}
    )

    assert response.status_code == 200, response.text
    assert infer.call_args.args == ("An apple",)
    if stream:
        assert "image_generation.completed" in response.text
        assert "aW1hZ2U=" in response.text
    else:
        assert response.json()["data"][0]["b64_json"] == "aW1hZ2U="


def test_video(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return MP4 directly and adapt the requested duration to Wan frames."""
    infer = AsyncMock(
        return_value=VideoResult(
            data=b"mp4",
            media_type="video/mp4",
            width=704,
            height=1280,
            num_frames=97,
            fps=24,
        )
    )
    monkeypatch.setattr(video, "infer", infer)
    response = client.post(
        "/v1/videos",
        files={
            "prompt": (None, "A passing train"),
            "seconds": (None, "4"),
            "size": (None, "704x1280"),
            "input_reference": ("reference.png", b"reference", "image/png"),
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"mp4"
    assert infer.call_args.args == ("A passing train",)
    assert infer.call_args.kwargs["image"] == b"reference"
    options = infer.call_args.kwargs["options"]
    assert (options.width, options.height, options.num_frames) == (704, 1280, 97)


def test_video_rejects_remote_reference(client: TestClient) -> None:
    """Accept reference bytes only from multipart uploads."""
    response = client.post(
        "/v1/videos",
        files={
            "prompt": (None, "A passing train"),
            "input_reference[image_url]": (None, "http://example.test/reference.png"),
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("model_id", "expected_status"),
    [("missing/model", 404), (text.pipeline.model.id, 409)],
)
def test_chat_requires_a_loaded_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    expected_status: int,
) -> None:
    """Reject unknown and unloaded models before queueing inference."""
    monkeypatch.setattr(text.pipeline, "_runtime", None)
    infer = AsyncMock(side_effect=AssertionError("Inference should not be called."))
    monkeypatch.setattr(text, "infer", infer)
    response = client.post(
        "/v1/chat/completions",
        json={"model": model_id, "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == expected_status
    infer.assert_not_called()
