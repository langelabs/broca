"""Bekko System One typed decision pipeline."""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer
from .__base import BasePipeline, Input, Result
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class BekkoPipeline(BasePipeline["SentenceTransformer", Input, Result]):
    """Load Bekko's typed decision model from its pinned local snapshot."""

    @contextmanager
    def _open_runtime(self) -> Generator[SentenceTransformer, None, None]:
        """Load the exported custom class with explicit code trust."""
        model = self.model
        if model.trust_remote_code is not True or not re.fullmatch(
            r"[0-9a-f]{40}", model.revision or ""
        ):
            raise ValueError(
                "Bekko requires explicit code trust and a pinned revision."
            )
        options = model.runtime_options or {}
        model_class = options.get(
            "model_class", "inference_v0.BekkoSentenceTransformer"
        )
        if not isinstance(model_class, str) or model_class != (
            "inference_v0.BekkoSentenceTransformer"
        ):
            raise ValueError(
                "Bekko requires its exported BekkoSentenceTransformer class."
            )
        if not model.get_file_path("inference_v0.py").is_file():
            raise FileNotFoundError("Download the Bekko snapshot before loading it.")
        local_path = str(model.local_path)
        implementation = get_class_from_dynamic_module(
            model_class, local_path, local_files_only=True
        )
        runtime = implementation(
            local_path,
            local_files_only=True,
            trust_remote_code=True,
            device=self.get_device(),
            attn_implementation=options.get("attn_implementation", "sdpa"),
        )
        yield runtime
