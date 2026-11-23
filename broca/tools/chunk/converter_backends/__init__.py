"""Convert documents to Markdown and enrich their media with Broca inference."""

from docling.datamodel.backend_options import (
    HTMLBackendOptions,
    MarkdownBackendOptions,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    FormatOption,
    HTMLFormatOption,
    ImageFormatOption,
    MarkdownFormatOption,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline

from broca.inference.text.types import AudioAttachment, ImageAttachment
from ._audio import AudioBackend
from ._image import ImageBackend
from ._video import VideoBackend

converter = DocumentConverter(
    format_options={
        InputFormat.AUDIO: FormatOption(
            pipeline_cls=SimplePipeline, backend=AudioBackend
        ),
        InputFormat.IMAGE: ImageFormatOption(pipeline_cls=ImageBackend),
        InputFormat.VIDEO: FormatOption(
            pipeline_cls=SimplePipeline, backend=VideoBackend
        ),
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(generate_picture_images=True)
        ),
        InputFormat.HTML: HTMLFormatOption(
            backend_options=HTMLBackendOptions(fetch_images=True)
        ),
        InputFormat.MD: MarkdownFormatOption(
            backend_options=MarkdownBackendOptions(fetch_images=True)
        ),
    }
)


async def interpret_media(attachment: AudioAttachment | ImageAttachment) -> str:
    """Interpret a single audio or image attachment with Broca's text model.

    :param attachment: Inline media to describe or transcribe.
    :returns: The description or transcript.
    """
    if isinstance(attachment, AudioAttachment):
        return await AudioBackend.interpret(attachment)
    elif isinstance(attachment, ImageAttachment):
        return await ImageBackend.interpret(attachment)
    else:
        raise NotImplementedError("Only audio and image are supported")


__all__ = ["ImageBackend", "converter", "interpret_media"]
