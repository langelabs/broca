"""Encode Bark output into the formats exposed by the speech endpoint."""

import asyncio
from io import BytesIO
from unittest.mock import Mock

import av
import pytest
import torch

from broca.inference.audio import AudioPipeline
from broca.inference.audio._types import AudioFormat, AudioOptions


@pytest.mark.parametrize(
    ("output_format", "media_type", "demuxer"),
    [
        ("mp3", "audio/mpeg", "mp3"),
        ("opus", "audio/ogg", "ogg"),
        ("aac", "audio/aac", "aac"),
        ("flac", "audio/flac", "flac"),
        ("wav", "audio/wav", "wav"),
        ("pcm", "audio/pcm", "s16le"),
    ],
)
def test_bark_audio_encodings(
    monkeypatch: pytest.MonkeyPatch,
    output_format: AudioFormat,
    media_type: str,
    demuxer: str,
) -> None:
    """Verify that each generated audio payload can be decoded.

    :param monkeypatch: Replaces model generation with a deterministic waveform.
    :param output_format: Audio encoding passed to the pipeline.
    :param media_type: MIME type expected from the pipeline.
    :param demuxer: Container reader used to verify the encoded bytes.
    """
    runtime = Mock()
    runtime.processor.return_value = {}
    runtime.model.device = torch.device("cpu")
    runtime.model.generate.return_value = torch.sin(torch.arange(12_000) * 0.03)
    runtime.model.generation_config.to_dict.return_value = {"sample_rate": 24_000}
    pipeline = AudioPipeline()
    monkeypatch.setattr(pipeline, "_runtime", runtime)

    result = asyncio.run(
        pipeline.infer("Hello", options=AudioOptions(output_format=output_format))
    )

    assert result.media_type == media_type
    assert result.sample_rate == 24_000
    assert result.duration_seconds == 0.5
    with av.open(BytesIO(result.data), format=demuxer) as container:
        assert next(container.decode(audio=0)).samples > 0
