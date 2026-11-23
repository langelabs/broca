"""Generate or edit encoded images with the configured FLUX model."""

from asyncio import CancelledError
from io import BytesIO
from threading import Event
from typing import Literal, cast

import torch
from diffusers.modular_pipelines.flux3.decoders import (
    Flux3DecodeStep,
    Flux3UnpackLatentsStep,
)
from diffusers.modular_pipelines.flux3.modular_pipeline import Flux3ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from PIL import Image

from ...ai_model import AiModel
from .._pipelines import Flux3Pipeline
from .._pipelines.flux3 import flux_step_callback
from ._model import _model
from .types import ImageOptions, ImagePreviewCallback, ImageResult, RasterFormat


def _encode_image(image: Image.Image, output_format: RasterFormat) -> bytes:
    """Encode a final or intermediate image in the requested raster format.

    :param image: Decoded pixel data.
    :param output_format: Image encoding for the response.
    :returns: Encoded image bytes.
    """
    output = BytesIO()
    image.save(output, format=output_format.upper())
    return output.getvalue()


class ImagePipeline(Flux3Pipeline[str, ImageResult]):
    """Generate and edit images using the configured FLUX runtime."""

    # FLUX generates one image at a time. Publish each final result independently.
    _max_batch_size = 1

    def __init__(self, model: AiModel = _model) -> None:
        """Initialize the pipeline with a compatible model configuration.

        :param model: Model to use. Defaults to the built-in configuration.
        """
        super().__init__(model)

    async def _infer(
        self,
        prompt: str,
        *,
        reference_images: list[bytes] | None = None,
        options: ImageOptions | None = None,
        partial_images: int = 0,
        on_preview: ImagePreviewCallback | None = None,
        cancelled: Event | None = None,
    ) -> ImageResult:
        """Generate one image, optionally conditioned on reference images.

        :param prompt: Nonblank image description or editing instruction.
        :param reference_images: Optional encoded reference images in input order.
        :param options: Optional generation overrides and output encoding.
        :param partial_images: Maximum number of intermediate images to decode.
        :param on_preview: Worker-thread callback receiving indexed encoded previews.
        :param cancelled: Signal to stop inference if the caller disconnects.
        :returns: Encoded image bytes with format and pixel dimensions.
        :raises ValueError: If the prompt is blank or image dimensions are invalid.
        :raises CancelledError: If the caller disconnects before generation finishes.
        """
        if cancelled is not None and cancelled.is_set():
            raise CancelledError
        runtime = self.runtime
        if not prompt.strip():
            raise ValueError("Image prompt must not be blank.")
        options = options or ImageOptions()
        defaults = self.model.generation_options or {}
        width = options.width or cast(int, defaults.get("width", 1024))
        height = options.height or cast(int, defaults.get("height", 1024))
        steps = options.num_inference_steps or cast(
            int, defaults.get("num_inference_steps", 16)
        )
        if width % 16 or height % 16:
            raise ValueError("FLUX image dimensions must be multiples of 16.")
        images: list[Image.Image] = []
        for data in reference_images or ():
            with Image.open(BytesIO(data)) as reference:
                images.append(reference.convert("RGB"))
        seed = options.seed if options.seed is not None else self.model.seed
        generator = torch.Generator()
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        preview_count = min(max(partial_images, 0), steps - 1, 3)
        checkpoints = {
            steps * (index + 1) // (preview_count + 1): index
            for index in range(preview_count)
        }

        def on_step(
            components: Flux3ModularPipeline,
            latents: torch.Tensor,
            step: int,
        ) -> None:
            """Check cancellation and decode selected steps in independent state."""
            if cancelled is not None and cancelled.is_set():
                raise CancelledError
            if on_preview is None or step not in checkpoints:
                return
            state = PipelineState(
                {
                    "latents": latents.clone(),
                    "height": height,
                    "width": width,
                    "output_type": "pil",
                }
            )
            _, state = Flux3UnpackLatentsStep()(components, state)
            _, state = Flux3DecodeStep()(components, state)
            on_preview(
                checkpoints[step],
                _encode_image(state.get("images")[0], options.output_format),
            )

        track_steps = cancelled is not None or (
            bool(checkpoints) and on_preview is not None
        )
        token = flux_step_callback.set(on_step if track_steps else None)
        try:
            with torch.inference_mode():
                image = cast(
                    list[Image.Image],
                    runtime(
                        prompt=prompt,
                        image=images or None,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        generator=generator,
                        output="images",
                    ),
                )[0]
        finally:
            flux_step_callback.reset(token)
        return ImageResult(
            data=_encode_image(image, options.output_format),
            media_type=cast(
                Literal["image/png", "image/jpeg", "image/webp"],
                f"image/{options.output_format}",
            ),
            width=image.width,
            height=image.height,
            seed=seed,
        )


pipeline = ImagePipeline()
