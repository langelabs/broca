"""Generate encoded videos with the configured Wan model."""

from fractions import Fraction
from io import BytesIO
from typing import cast

import av
import torch
from PIL import Image

from .._pipelines import WanPipeline
from ._model import _model
from ._types import VideoOptions, VideoResult


class VideoPipeline(WanPipeline[str, VideoResult]):
    """Generate video using the configured Wan runtime."""

    async def _infer(
        self,
        prompt: str,
        *,
        image: bytes | None = None,
        options: VideoOptions | None = None,
    ) -> VideoResult:
        """Generate one video with optional first-frame conditioning.

        :param prompt: Nonblank description of the requested video.
        :param image: Optional encoded image used to condition the first frame.
        :param options: Optional generation overrides and output encoding.
        :returns: Encoded video bytes with format, frame size, count, and rate.
        :raises ValueError: If the prompt, dimensions, or frame count are invalid.
        """
        runtime = self.runtime
        if not prompt.strip():
            raise ValueError("Video prompt must not be blank.")
        options = options or VideoOptions()
        config = self.model
        defaults = config.generation_options or {}
        runtime_options = config.runtime_options or {}
        width = options.width or cast(int, defaults.get("width", 1280))
        height = options.height or cast(int, defaults.get("height", 704))
        num_frames = options.num_frames or cast(int, defaults.get("num_frames", 121))
        spatial_factor = runtime.vae_scale_factor_spatial
        _, patch_height, patch_width = runtime.transformer.config.patch_size
        if width % (spatial_factor * patch_width) or height % (
            spatial_factor * patch_height
        ):
            raise ValueError(
                "Video dimensions must be divisible by the Wan spatial stride."
            )
        multiple = cast(int, runtime_options.get("frame_count_multiple", 4))
        remainder = cast(int, runtime_options.get("frame_count_remainder", 1))
        if num_frames % multiple != remainder:
            raise ValueError(
                f"Wan requires a frame count of {multiple}k + {remainder}."
            )

        seed = options.seed if options.seed is not None else config.seed
        generator = torch.Generator()
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        guidance = (
            options.guidance_scale
            if options.guidance_scale is not None
            else cast(float, defaults.get("guidance_scale", 5.0))
        )
        generation = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": options.num_inference_steps
            or cast(int, defaults.get("num_inference_steps", 50)),
            "guidance_scale": guidance,
            "generator": generator,
            "output_type": "pil",
        }
        with torch.inference_mode():
            if image is None:
                result = runtime(**generation)
            else:
                with Image.open(BytesIO(image)) as reference:
                    first_frame = reference.convert("RGB")
                try:
                    result = self.image_runtime(image=first_frame, **generation)
                finally:
                    runtime.maybe_free_model_hooks()

        frames = cast(list[list[Image.Image]], result.frames)[0]
        width, height = frames[0].size
        fps = options.fps or cast(float, runtime_options.get("fps", 24))
        rate = Fraction(str(fps)).limit_denominator()
        output = BytesIO()
        with av.open(output, "w", format=options.output_format) as container:
            codec = "libx264" if options.output_format == "mp4" else "libvpx-vp9"
            stream = container.add_stream(codec, rate=rate, width=width, height=height)
            stream.pix_fmt = "yuv420p"
            for frame in frames:
                container.mux(stream.encode(av.VideoFrame.from_image(frame)))
            container.mux(stream.encode())
        return VideoResult(
            data=output.getvalue(),
            media_type="video/mp4" if options.output_format == "mp4" else "video/webm",
            width=width,
            height=height,
            num_frames=len(frames),
            fps=float(rate),
            seed=seed,
        )


pipeline = VideoPipeline(_model)
