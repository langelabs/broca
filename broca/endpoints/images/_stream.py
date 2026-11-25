"""Bridge encoded image previews from the inference worker to HTTP."""

import asyncio
import logging
from base64 import b64encode
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from threading import Event
from time import time

from fastapi.responses import StreamingResponse
from starlette.types import Send

from ...inference import images
from ...inference.images.types import ImageOptions
from .._events import event_frame

logger = logging.getLogger(__name__)


class ImageEventStream(StreamingResponse):
    """Send previews and the final image while keeping the model worker unblocked."""

    def __init__(
        self,
        prompt: str,
        options: ImageOptions,
        partial_images: int,
    ) -> None:
        """Bind one request to a lazily started event stream.

        :param prompt: Validated generation prompt.
        :param options: Image dimensions and encoding.
        :param partial_images: Maximum number of previews.
        """
        self.events = self._events(prompt, options, partial_images)
        super().__init__(
            self.events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def stream_response(self, send: Send) -> None:
        """Close the producer even if sending an SSE frame fails.

        :param send: ASGI response sender.
        """
        async with aclosing(self.events):
            await super().stream_response(send)

    async def _events(
        self,
        prompt: str,
        options: ImageOptions,
        partial_images: int,
    ) -> AsyncGenerator[str, None]:
        """Yield bounded preview frames followed by the final image or an error.

        :param prompt: Validated generation prompt.
        :param options: Image dimensions and encoding.
        :param partial_images: Maximum number of previews.
        :yields: Complete server-sent event frames.
        """
        loop = asyncio.get_running_loop()
        stopped = Event()
        # Only the requested previews and one completion marker can be queued.
        queue: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue(
            partial_images + 1
        )
        metadata = {
            "output_format": options.output_format,
            "size": f"{options.width}x{options.height}",
            "background": "opaque",
            "quality": "auto",
        }

        def on_preview(index: int, data: bytes) -> None:
            """Schedule a preview without blocking the inference thread."""
            if not stopped.is_set():
                # A worker can finish a preview during event-loop shutdown.
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(queue.put_nowait, (index, data))

        task = asyncio.create_task(
            images.infer(
                prompt,
                options=options,
                partial_images=partial_images,
                on_preview=on_preview,
                cancelled=stopped,
            )
        )
        task.add_done_callback(lambda _: queue.put_nowait(None))
        try:
            while True:
                try:
                    preview = await asyncio.wait_for(queue.get(), timeout=10)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if preview is None:
                    break
                index, data = preview
                kind = "image_generation.partial_image"
                yield event_frame(
                    {
                        **metadata,
                        "type": kind,
                        "created_at": int(time()),
                        "partial_image_index": index,
                        "b64_json": b64encode(data).decode("ascii"),
                    },
                    event=kind,
                )
            image = await task
            kind = "image_generation.completed"
            yield event_frame(
                {
                    **metadata,
                    "type": kind,
                    "created_at": int(time()),
                    "size": f"{image.width}x{image.height}",
                    "b64_json": b64encode(image.data).decode("ascii"),
                },
                event=kind,
            )
        except Exception as error:
            logger.error("Image stream failed (%s)", type(error).__name__)
            yield event_frame(
                {
                    "error": {
                        "type": "server_error",
                        "message": "Image generation failed.",
                    }
                },
                event="error",
            )
        finally:
            stopped.set()
            if task.done() and not task.cancelled():
                task.exception()
            else:
                task.cancel()
