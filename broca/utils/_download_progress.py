"""Download utilities for model-labelled Hub bars and shared byte progress."""

from collections.abc import Callable
from typing import Any, ClassVar

from huggingface_hub.utils import tqdm


class ModelDownloadProgress(tqdm):
    """Bind a model label and progress callback through a transfer-local subclass."""

    model_label: ClassVar[str]
    report: ClassVar[Callable[[float, float], None]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Label byte bars while preserving Hub and user display preferences.

        :param args: Standard tqdm positional options.
        :param kwargs: Standard tqdm and Hub progress options.
        :returns: None.
        """
        self._reports_bytes = kwargs.get("unit") == "B" and not str(
            kwargs.get("name", "")
        ).endswith(".transfer")
        kwargs["desc"] = f"{self.model_label}: {kwargs.get('desc', 'Downloading')}"
        super().__init__(*args, **kwargs)
        self._publish()

    def _publish(self) -> None:
        """Publish available reconstruction counters, including resumed bytes.

        :returns: None.
        """
        if self._reports_bytes and hasattr(self, "n"):
            self.report(float(self.n), float(self.total or 0))

    def update(self, n: float | None = 1) -> bool | None:
        """Report reconstructed bytes even when console output is disabled.

        :param n: Increment in reconstructed bytes.
        :returns: Whether tqdm refreshed the display.
        """
        with self.get_lock():
            if self.disable:
                self.n += n or 0
                result = None
            else:
                result = super().update(n or 0)
            self._publish()
            return result

    def refresh(self, *args: Any, **kwargs: Any) -> bool | None:
        """Report changing aggregate snapshot totals.

        :param args: Standard tqdm refresh arguments.
        :param kwargs: Standard tqdm refresh options.
        :returns: Whether tqdm refreshed the display.
        """
        with self.get_lock():
            result = super().refresh(*args, **kwargs)
            self._publish()
            return result
