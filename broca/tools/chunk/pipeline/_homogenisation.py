"""Convert documents to Markdown and enrich their media with Broca inference."""

from asyncio import to_thread
from base64 import b64decode
from io import BytesIO

from docling.datamodel.backend_options import (
    DEFAULT_MAX_IMAGE_PIXELS,
)

from docling.utils.image_utils import check_image_pixels, open_image
from docling_core.types.doc import DescriptionMetaField, PictureMeta
from docling_core.types.io import DocumentStream

from ..converter_backends import ImageBackend, converter, interpret_media


async def homogenise(source: str | BytesIO, name: str = "document") -> str:
    """Convert one source and describe embedded pictures before Markdown export.

    :param source: HTTPS URL or an in-memory document stream.
    :param name: Filename used by Docling for sources supplied as streams.
    :returns: Markdown containing the document text and media interpretations.
    """
    if isinstance(source, BytesIO):
        source = DocumentStream(stream=source, name=name)
        source.stream.seek(0)
    elif not source.startswith("https://"):
        raise ValueError("url must start with 'https://'")
    result = await to_thread(converter.convert, source)
    document = result.document
    for picture in document.pictures:
        if (
            picture.image is None
            or getattr(picture.image.uri, "scheme", None) != "data"
        ):
            continue
        if picture.meta is not None and picture.meta.description is not None:
            continue

        encoded_image = str(picture.image.uri).partition(",")[2]
        with open_image(b64decode(encoded_image, validate=True)) as image:
            check_image_pixels(image.size, DEFAULT_MAX_IMAGE_PIXELS)
            attachment = ImageBackend.create_attachment(image)
        description = await interpret_media(attachment)
        if description:
            picture.meta = picture.meta or PictureMeta()
            picture.meta.description = DescriptionMetaField(text=description)
    return document.export_to_markdown()
