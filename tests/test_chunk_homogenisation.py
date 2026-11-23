"""Exercise Docling conversion with Broca's multimodal inference boundary."""

import asyncio
from base64 import b64decode
from io import BytesIO
from unittest.mock import AsyncMock
import wave

import av
import pytest
from docling.datamodel import document as docling_input
from docling_core.types.io import DocumentStream
from docx import Document
from PIL import Image

from broca.inference import text
from broca.inference.text.types import AudioAttachment, ImageAttachment, TextResult
from broca.tools.chunk.pipeline._homogenisation import homogenise


@pytest.mark.parametrize("embedded", [False, True])
def test_homogenisation_describes_images(
    monkeypatch: pytest.MonkeyPatch, embedded: bool
) -> None:
    """Use Docling's Markdown output to verify image descriptions survive conversion.

    :param monkeypatch: Substitute the text model at the inference boundary.
    :param embedded: Put the image inside a Word document when true.
    """
    model = AsyncMock(
        return_value=TextResult(text="A small red square.", finish_reason="stop")
    )
    monkeypatch.setattr(text, "infer", model)
    source = BytesIO()
    Image.new("RGB", (64, 64), "red").save(source, format="PNG")
    name = "figure.png"
    if embedded:
        document = Document()
        document.add_paragraph("Before the figure.")
        source.seek(0)
        document.add_picture(source)
        document.add_paragraph("After the figure.")
        name, source = "report.docx", BytesIO()
        document.save(source)

    markdown = asyncio.run(homogenise(source, name=name))

    model.assert_awaited_once()
    attachment = model.await_args.kwargs["attachments"][0]
    assert isinstance(attachment, ImageAttachment)
    with Image.open(BytesIO(b64decode(attachment.data))) as image:
        assert image.size == (64, 64)
        assert image.getpixel((0, 0))[0] > 200
    assert "A small red square." in markdown
    if embedded:
        assert markdown.index("Before the figure.") < markdown.index(
            "A small red square."
        )
        assert markdown.index("A small red square.") < markdown.index(
            "After the figure."
        )


def test_homogenisation_transcribes_audio_from_url_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify format detection and bounded audio attachments for a long recording.

    :param monkeypatch: Substitute the URL fetch and text model boundaries.
    """
    model = AsyncMock(
        side_effect=[
            TextResult(text="First passage.", finish_reason="stop"),
            TextResult(text="Final passage.", finish_reason="stop"),
        ]
    )
    monkeypatch.setattr(text, "infer", model)
    source = BytesIO()
    with wave.open(source, "wb") as wav:
        wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * 8000 * 31)
    source.seek(0)
    monkeypatch.setattr(
        docling_input,
        "resolve_source_to_stream",
        lambda *_args, **_kwargs: DocumentStream(name="recording", stream=source),
    )

    markdown = asyncio.run(homogenise("https://example.com/recording"))

    assert markdown.index("First passage.") < markdown.index("Final passage.")
    clip_sizes = []
    for call in model.await_args_list:
        attachment = call.kwargs["attachments"][0]
        assert isinstance(attachment, AudioAttachment)
        assert attachment.format == "wav"
        with wave.open(BytesIO(b64decode(attachment.data))) as wav:
            assert wav.getframerate() == 16000
            clip_sizes.append(wav.getnframes())
    assert clip_sizes == [480000, 16000]


@pytest.mark.parametrize("with_audio", [False, True])
def test_homogenisation_describes_video_in_order(
    monkeypatch: pytest.MonkeyPatch, with_audio: bool
) -> None:
    """Verify video frames are sampled in bounded groups and audio is transcribed.

    :param monkeypatch: Substitute the text model at the inference boundary.
    :param with_audio: Add an audio track to the generated video when true.
    """
    results = [
        TextResult(text="Red appears first.", finish_reason="stop"),
        TextResult(text="Blue appears last.", finish_reason="stop"),
    ]
    if with_audio:
        results.append(TextResult(text="Spoken words.", finish_reason="stop"))
    model = AsyncMock(side_effect=results)
    monkeypatch.setattr(text, "infer", model)
    source = BytesIO()
    with av.open(source, mode="w", format="matroska") as output:
        video = output.add_stream("mpeg4", rate=2)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        audio = output.add_stream("pcm_s16le", rate=16000) if with_audio else None
        if audio is not None:
            audio.layout = "mono"
            samples = av.AudioFrame(format="s16", layout="mono", samples=16000)
            samples.sample_rate = 16000
            samples.planes[0].update(bytes(32000))
            output.mux(audio.encode(samples))
            output.mux(audio.encode(None))
        for index in range(18):
            color = "red" if index < 16 else "blue"
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), color))
            output.mux(video.encode(frame))
        output.mux(video.encode(None))

    markdown = asyncio.run(homogenise(source, name="recording.mkv"))

    assert markdown.index("Red appears first.") < markdown.index("Blue appears last.")
    first, second, *remaining = model.await_args_list
    assert len(first.kwargs["attachments"]) == 8
    assert len(second.kwargs["attachments"]) == 1
    for call, channel in ((first, 0), (second, 2)):
        for attachment in call.kwargs["attachments"]:
            assert isinstance(attachment, ImageAttachment)
            with Image.open(BytesIO(b64decode(attachment.data))) as image:
                assert image.getpixel((0, 0))[channel] > 200
    if with_audio:
        assert "Spoken words." in markdown
        assert len(remaining) == 1
        assert isinstance(remaining[0].kwargs["attachments"][0], AudioAttachment)
    else:
        assert not remaining


def test_homogenisation_preserves_text_only_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert Markdown through Docling without calling text inference.

    :param monkeypatch: Track any calls to the inference boundary.
    """
    model = AsyncMock()
    monkeypatch.setattr(text, "infer", model)

    markdown = asyncio.run(
        homogenise(BytesIO(b"# Notes\n\nText only."), name="notes.md")
    )

    assert markdown == "# Notes\n\nText only."
    model.assert_not_awaited()
