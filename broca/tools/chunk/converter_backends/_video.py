"""Convert video to timestamped descriptions and an optional transcript."""

from asyncio import Runner
from collections.abc import Sequence
from io import BytesIO
from itertools import chain
from math import floor

import av
from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DocItemLabel, DoclingDocument

from broca.inference import text
from broca.inference.text.types import ImageAttachment, TextOptions

from ._audio import AudioBackend
from ._image import ImageBackend

_FRAMES_PER_REQUEST = 8


class VideoBackend(NoOpBackend, DeclarativeDocumentBackend):
    """Build Docling documents from a video's images and audio track."""

    @staticmethod
    async def interpret(frames: Sequence[tuple[float, ImageAttachment]]) -> str:
        """Describe the visible events across one group of sampled frames.

        :param frames: Elapsed times in seconds and their corresponding frames.
        :returns: A concise description in playback order.
        """
        timestamps = ", ".join(f"{timestamp:.1f}s" for timestamp, _ in frames)
        response = await text.infer(
            f"These video frames are in playback order at {timestamps}. Describe the "
            "visible events and changes in order. Include important legible text. "
            "Return only the description.",
            attachments=[attachment for _, attachment in frames],
            options=TextOptions(temperature=0, enable_thinking=False),
        )
        return response.text.strip()

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        """Return the Docling input formats this backend handles."""
        return {InputFormat.VIDEO}

    def convert(self) -> DoclingDocument:
        """Sample one frame per second and transcribe the first audio track.

        :returns: A document with timestamped descriptions and any transcript.
        :raises ValueError: If the video has no decodable frames.
        :raises RuntimeError: If the input has already been unloaded.
        """
        source = self.path_or_stream
        if source is None:
            raise RuntimeError("The video source has already been unloaded.")
        if isinstance(source, BytesIO):
            source.seek(0)
        document = DoclingDocument(name=self.file.name)
        options = {"format_whitelist": "mov,matroska,avi,mpegts,webm,ogg,mpeg"}

        with Runner() as runner:
            with av.open(source, mode="r", options=options) as video:
                if not video.streams.video:
                    raise ValueError("The file contains no video stream.")
                has_audio = bool(video.streams.audio)
                frame_rate = float(video.streams.video[0].average_rate or 1)
                origin: float | None = None
                last_second = -1
                frames: list[tuple[float, ImageAttachment]] = []
                for index, frame in enumerate(chain(video.decode(video=0), (None,))):
                    if frame is not None:
                        elapsed = index / frame_rate
                        if frame.time is not None:
                            if origin is None:
                                origin = frame.time - elapsed
                            elapsed = max(0.0, frame.time - origin)
                        second = floor(elapsed)
                        if second <= last_second:
                            continue
                        last_second = second
                        image = frame.to_image()
                        if frame.rotation:
                            image = image.rotate(frame.rotation, expand=True)
                        frames.append((elapsed, ImageBackend.create_attachment(image)))

                    if frames and (frame is None or len(frames) == _FRAMES_PER_REQUEST):
                        description = runner.run(self.interpret(frames))
                        if description:
                            document.add_text(
                                label=DocItemLabel.PARAGRAPH,
                                text=f"[{frames[0][0]:.1f}s] {description}",
                            )
                        frames.clear()

                if last_second < 0:
                    raise ValueError("The video stream contains no decodable frames.")

            if has_audio:
                if isinstance(source, BytesIO):
                    source.seek(0)
                with av.open(source, mode="r", options=options) as audio:
                    transcripts = [
                        transcript
                        for transcript in AudioBackend.transcribe(audio, runner)
                        if transcript
                    ]
                if transcripts:
                    document.add_heading("Transcript")
                    for transcript in transcripts:
                        document.add_text(label=DocItemLabel.PARAGRAPH, text=transcript)

        return document
