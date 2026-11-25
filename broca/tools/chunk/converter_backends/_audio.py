"""Convert documents to Markdown and enrich their media with Broca inference."""

from asyncio import Runner
from base64 import b64encode
from collections.abc import Iterator
from io import BytesIO
from itertools import chain
import wave

import av
from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
)
from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DocItemLabel, DoclingDocument

from broca.inference import text
from broca.inference.text.types import AudioAttachment, TextOptions


_AUDIO_SAMPLE_RATE = 16_000
_AUDIO_CLIP_SECONDS = 30
_AUDIO_PROMPT = (
    "Transcribe speech verbatim in its original language. If there is no speech, briefly "
    "describe only the sounds you can hear. Return only the transcript or sound description."
)


class AudioBackend(NoOpBackend, DeclarativeDocumentBackend):
    """Build Docling transcript documents from standalone recordings."""

    @staticmethod
    async def interpret(attachment: AudioAttachment) -> str:
        """Transcribe one audio attachment with Broca's loaded text model.

        :param attachment: Inline audio clip to transcribe.
        :returns: The model's transcript or sound description.
        """
        response = await text.infer(
            _AUDIO_PROMPT,
            attachments=[attachment],
            options=TextOptions(temperature=0, enable_thinking=False),
        )
        return response.text.strip()

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        """Return the input formats registered to this media backend."""
        return {InputFormat.AUDIO}

    @staticmethod
    def transcribe(audio: av.container.InputContainer, runner: Runner) -> Iterator[str]:
        """Transcribe the first audio track in clips of up to 30 seconds.

        :param audio: Open media container positioned at the start.
        :param runner: Event loop runner for the text inference calls.
        :yields: The transcription of each clip, including empty responses.
        :raises ValueError: If there is no audio track.
        """
        if not audio.streams.audio:
            raise ValueError("The file contains no audio stream.")
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=_AUDIO_SAMPLE_RATE,
            frame_size=_AUDIO_SAMPLE_RATE * _AUDIO_CLIP_SECONDS,
        )
        for frame in chain(audio.decode(audio=0), (None,)):
            for clip in resampler.resample(frame):
                with BytesIO() as data:
                    with wave.open(data, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(_AUDIO_SAMPLE_RATE)
                        wav.writeframes(clip.to_ndarray().tobytes())
                    attachment = AudioAttachment(
                        type="input_audio",
                        format="wav",
                        data=b64encode(data.getvalue()).decode("ascii"),
                    )
                yield runner.run(AudioBackend.interpret(attachment))

    def convert(self) -> DoclingDocument:
        """Transcribe audio in playback order.

        :returns: A Docling document with transcript paragraphs.
        :raises ValueError: If an audio file has no decodable samples.
        :raises RuntimeError: If the input has already been unloaded.
        """
        source = self.path_or_stream
        if source is None:
            raise RuntimeError("The media source has already been unloaded.")
        if isinstance(source, BytesIO):
            source.seek(0)
        document = DoclingDocument(name=self.file.name)
        has_audio = False
        with (
            av.open(
                source,
                mode="r",
                options={"format_whitelist": "wav,mp3,mov,aac,ogg,flac"},
            ) as audio,
            Runner() as runner,
        ):
            for transcript in self.transcribe(audio, runner):
                has_audio = True
                if transcript:
                    document.add_text(label=DocItemLabel.PARAGRAPH, text=transcript)
        if not has_audio:
            raise ValueError("The audio stream contains no decodable samples.")
        return document
