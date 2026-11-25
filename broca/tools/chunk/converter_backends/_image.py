"""Convert documents to Markdown and enrich their media with Broca inference."""

from base64 import b64encode
from io import BytesIO

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
)
from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.document import ConversionResult
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.utils.image_utils import to_rgb
from docling_core.types.doc import DoclingDocument, ImageRef
from PIL.Image import Image

from broca.inference import text
from broca.inference.text.types import ImageAttachment, TextOptions

_IMAGE_PROMPT = (
    "Describe this image for a reader who cannot see it. Include legible text and "
    "the important details of any chart, table, or diagram. Return only the description."
)


class ImageBackend(SimplePipeline):
    """Retain standalone image frames as pictures for Broca interpretation."""

    @staticmethod
    def create_attachment(image: Image) -> ImageAttachment:
        """Encode a picture or video frame for the Broca text model.

        :param image: Owned image, resized in place if necessary.
        :returns: A bounded inline JPEG attachment.
        """
        image.thumbnail((1600, 1600))
        with BytesIO() as output:
            to_rgb(image).save(output, format="JPEG", quality=90)
            return ImageAttachment(
                type="image",
                format="jpeg",
                data=b64encode(output.getvalue()).decode("ascii"),
            )

    @staticmethod
    async def interpret(attachment: ImageAttachment) -> str:
        """Describe one image with Broca's loaded text model.

        :param attachment: Inline media prepared for Broca text inference.
        :returns: The model's generated text, stripped of surrounding whitespace.
        """
        response = await text.infer(
            _IMAGE_PROMPT,
            attachments=[attachment],
            options=TextOptions(temperature=0, enable_thinking=False),
        )
        return response.text.strip()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend) -> bool:
        """Check that the backend can read standalone image frames.

        :param backend: Backend selected by Docling for the input document.
        :returns: Whether the backend reads images.
        """
        return isinstance(backend, ImageDocumentBackend)

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build a document with one picture per frame in source order.

        :param conv_res: The conversion result carrying the input image document.
        :returns: The result with the converted pictures.
        :raises RuntimeError: If the image backend is unavailable.
        """
        backend = conv_res.input._backend
        if not isinstance(backend, ImageDocumentBackend):
            raise RuntimeError("Standalone images require Docling's image backend.")

        document = DoclingDocument(name=conv_res.input.file.name)
        for index in range(backend.page_count()):
            page = backend.load_page(index)
            image = page.get_page_image()
            if image is not None:
                document.add_picture(image=ImageRef.from_pil(image, dpi=72))
        conv_res.document = document
        return conv_res
