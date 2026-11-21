"""Portable AI model configuration and asynchronous local asset downloads."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from contextlib import ExitStack
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from filelock import FileLock, Timeout
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.file_download import DryRunFileInfo
from huggingface_hub.utils import filter_repo_objects
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from .utils._download_progress import ModelDownloadProgress
from .utils._download_state import (
    DownloadManifest,
    FileRecord,
    assets_complete,
    read_manifest,
    safe_path,
    write_manifest,
)
from .utils._download_types import DownloadOptions, SnapshotOptions


ModelModality = Literal["audio", "decision", "embedding", "image", "text", "video"]


class AiModel(BaseModel):
    """Describe model assets and optional settings for an inference backend.

    Only ``id="lab/model"`` is required. Unspecified options remain ``None`` so
    a backend can keep its defaults. Engine and format names are unrestricted;
    runtime and generation options accommodate backend-specific JSON settings.
    These settings describe a future loader; this class does not run inference.

    Assets always live under ``Path.home() / "ai" / lab / model``. A separate
    ``weights_repo`` gets its own lab/model directory. Authentication uses the
    Hugging Face credential store or ``HF_TOKEN``; credentials are never part of
    this serializable config. Top-level configuration is frozen during transfers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    id: str = Field(description="Hugging Face repository in lab/model form.")
    name: str | None = Field(default=None, description="Optional display name.")
    created_at: NonNegativeInt | None = Field(
        default=None, description="Unix timestamp of the primary repository creation."
    )
    type: ModelModality | Literal["chat"] | None = Field(
        default=None, description="Primary inference modality."
    )
    description: str | None = Field(
        default=None, description="Human-readable model description."
    )
    family: str | None = Field(
        default=None, description="Model architecture or family."
    )
    languages: list[str] | None = Field(
        default=None, description="Supported language codes."
    )
    parameter_count_billions: PositiveFloat | None = Field(
        default=None,
        description="Parameter count in billions, including fractional sizes.",
    )
    model_engine: str | None = Field(
        default=None,
        description="Backend name, such as MLX, llama.cpp, vLLM, transformers, or diffusers.",
    )
    model_format: str | None = Field(
        default=None,
        description="Artifact format, such as gguf, mlx, safetensors, or onnx.",
    )
    quantization: str | None = Field(
        default=None,
        validation_alias=AliasChoices("quantization", "quantisation"),
        description="Weight quantization; also accepts the reference pack's quantisation spelling.",
    )

    # Artifact selection. Revisions apply independently to pipeline and weights.
    file: str | None = Field(
        default=None,
        description="Repository-relative checkpoint; omit to download a snapshot.",
    )
    clip: str | None = Field(
        default=None,
        description="Repository-relative multimodal projector in the primary repository.",
    )
    weights_repo: str | None = Field(
        default=None,
        description="Separate repository for file; also download the primary pipeline snapshot.",
    )
    revision: str | None = Field(
        default=None,
        description="Primary repository branch, tag, or commit; None uses the Hub default.",
    )
    weights_revision: str | None = Field(
        default=None,
        description="Separate weights revision; None uses that repository's default.",
    )
    allow_patterns: list[str] | None = Field(
        default=None,
        description="Snapshot inclusion globs; explicit file and clip downloads are unaffected.",
    )
    ignore_patterns: list[str] | None = Field(
        default=None,
        description="Snapshot exclusion globs, in addition to replaced transformer weights.",
    )
    local_files_only: bool | None = Field(
        default=None, description="Ask the Hub to use only locally available assets."
    )
    max_workers: PositiveInt | None = Field(
        default=None, description="Maximum concurrent snapshot transfer workers."
    )

    # Planning estimates, in bytes; no hardware probing or backend import needed.
    estimated_weights_bytes: NonNegativeInt | None = Field(
        default=None,
        description="Primary weight size when no local checkpoint is available.",
    )
    supporting_weights_bytes: NonNegativeInt | None = Field(
        default=None,
        description="Resident encoder/VAE weight allowance at loaded precision.",
    )
    projector_bytes: NonNegativeInt | None = Field(
        default=None, description="Projector size when no local projector is available."
    )
    runtime_bytes: NonNegativeInt | None = Field(
        default=None,
        description="Additional context, activation, and workspace reserve.",
    )

    # Runtime overrides are metadata, interpreted by a concrete inference loader.
    device: str | None = Field(
        default=None, description="Backend device, such as cpu, mps, cuda, or cuda:1."
    )
    dtype: str | None = Field(
        default=None,
        description="Compute precision, such as float32, float16, or bfloat16.",
    )
    context_window: PositiveInt | None = Field(
        default=None, description="Maximum context length in tokens."
    )
    enable_thinking: bool | None = Field(
        default=None,
        description="Enable model-specific reasoning behavior when supported.",
    )
    chat_template: str | None = Field(
        default=None, description="Backend chat-format template."
    )
    stop: list[str] | None = Field(
        default=None, description="Text sequences that end generation."
    )
    stop_token_ids: list[NonNegativeInt] | None = Field(
        default=None, description="Token identifiers that end generation."
    )
    gpu_layers: Annotated[int, Field(ge=-1)] | None = Field(
        default=None,
        description="GPU layer count; -1 requests all layers for supporting engines.",
    )
    main_gpu: NonNegativeInt | None = Field(
        default=None, description="Primary GPU index."
    )
    tensor_split: list[NonNegativeFloat] | None = Field(
        default=None, description="Relative model allocation across GPUs."
    )
    batch_size: PositiveInt | None = Field(
        default=None, description="Logical token batch size."
    )
    physical_batch_size: PositiveInt | None = Field(
        default=None, description="Physical token microbatch size."
    )
    cpu_threads: PositiveInt | None = Field(
        default=None, description="CPU threads for token generation."
    )
    cpu_batch_threads: PositiveInt | None = Field(
        default=None, description="CPU threads for prompt evaluation."
    )
    use_mmap: bool | None = Field(
        default=None, description="Memory-map model weights when supported."
    )
    use_mlock: bool | None = Field(
        default=None, description="Lock model pages in resident memory when supported."
    )
    flash_attention: bool | None = Field(
        default=None, description="Use flash attention when supported."
    )
    offload_kqv: bool | None = Field(
        default=None, description="Offload attention operations to the accelerator."
    )
    cpu_offload: bool | None = Field(
        default=None, description="Enable pipeline CPU offloading when supported."
    )
    vae_tiling: bool | None = Field(
        default=None,
        description="Enable tiled image/video VAE decoding when supported.",
    )
    cache_enabled: bool | None = Field(
        default=None, description="Enable the backend's prompt cache."
    )
    cache_type: Literal["ram", "disk"] | None = Field(
        default=None, description="Prompt cache storage medium."
    )
    cache_size: NonNegativeInt | None = Field(
        default=None, description="Prompt cache budget in bytes."
    )
    kv_bits: PositiveFloat | None = Field(
        default=None, description="KV-cache precision in bits."
    )
    kv_quant_scheme: str | None = Field(
        default=None, description="KV-cache quantization scheme."
    )
    kv_group_size: PositiveInt | None = Field(
        default=None, description="KV-cache quantization group size."
    )
    kv_max_size: PositiveInt | None = Field(
        default=None, description="Maximum cached tokens."
    )
    seed: int | None = Field(default=None, description="Backend random seed.")
    trust_remote_code: bool | None = Field(
        default=None,
        description="Explicit loader permission for remote model code; downloading never executes it.",
    )
    runtime_options: dict[str, JsonValue] | None = Field(
        default=None,
        description="Additional backend-specific loader settings; do not store credentials here.",
    )
    generation_options: dict[str, JsonValue] | None = Field(
        default=None,
        description="Generation defaults, such as sampling, image dimensions, steps, frames, or voice.",
    )

    @field_validator("id", "weights_repo")
    @classmethod
    def validate_repository(cls, value: str | None) -> str | None:
        """Require safe, namespaced Hub identifiers for the local layout.

        :param value: Primary or optional weights repository.
        :returns: Validated repository identifier.
        :raises ValueError: If the identifier is not a safe lab/model name.
        """
        if value is not None and (
            re.fullmatch(r"[\w][\w.-]*/[\w][\w.-]{0,95}", value, re.ASCII) is None
            or ".." in value
            or "--" in value
            or any(part.endswith((".", "-", ".git")) for part in value.split("/"))
        ):
            raise ValueError("Repositories must be safe lab/model identifiers")
        return value

    @field_validator("file", "clip")
    @classmethod
    def validate_filename(cls, value: str | None) -> str | None:
        """Reject filenames that escape repositories or overwrite Hub metadata.

        :param value: Optional repository-relative asset path.
        :returns: Validated path.
        :raises ValueError: If the filename is unsafe.
        """
        if value is not None and (
            not PurePosixPath(value).parts
            or PurePosixPath(value).is_absolute()
            or any(part in {"..", ".cache"} for part in PurePosixPath(value).parts)
            or "\\" in value
            or ":" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("Model files must be safe repository-relative paths")
        return value

    @model_validator(mode="after")
    def validate_weights_selection(self) -> Self:
        """Reject weights options without their associated artifact selection.

        :returns: Validated model configuration.
        :raises ValueError: If separate weights or their revision lack a target.
        """
        if self.weights_repo is not None and self.file is None:
            raise ValueError("weights_repo requires a checkpoint file")
        if self.weights_revision is not None and self.weights_repo is None:
            raise ValueError("weights_revision requires weights_repo")
        return self

    @property
    def local_path(self) -> Path:
        """Locate primary assets under the user's AI directory.

        :returns: Safe primary repository directory.
        :raises ValueError: If a symlink redirects the directory outside ~/ai.
        """
        return safe_path(Path.home() / "ai", self.id)

    def get_file_path(
        self, filename: str | None, *, weights: bool | None = None
    ) -> Path:
        """Resolve an asset inside its model repository.

        :param filename: Repository-relative path to the asset.
        :param weights: Override repository selection. When omitted, the configured
            checkpoint uses the weights repository and other assets use the primary
            repository.
        :returns: Safe absolute path to the requested asset.
        :raises ValueError: If the filename is missing or escapes the repository.
        """
        if filename is None:
            raise ValueError("No model filename was provided")
        self.validate_filename(filename)
        use_weights = filename == self.file if weights is None else weights
        if use_weights and self.weights_repo is not None:
            return safe_path(safe_path(Path.home() / "ai", self.weights_repo), filename)
        return safe_path(self.local_path, filename)

    @property
    def checkpoint_path(self) -> Path:
        """Locate the selected checkpoint in its owning repository.

        :returns: Safe absolute checkpoint path.
        :raises ValueError: If no file is configured or a symlink escapes ~/ai.
        """
        if self.file is None:
            raise ValueError("This model has no checkpoint file")
        return self.get_file_path(self.file, weights=True)

    @property
    def state_path(self) -> Path:
        """Identify persisted state for the selected revisions and assets.

        :returns: Manifest location under ~/ai/.downloads.
        """
        identity = [
            self.id,
            self.weights_repo,
            self.file,
            self.clip,
            self.revision,
            self.weights_revision,
            self.allow_patterns,
            self.ignore_patterns,
        ]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        return safe_path(Path.home() / "ai", f".downloads/{key}.json")

    @property
    def size(self) -> int | None:
        """Estimate resident memory using local sizes or configured allowances.

        Add primary, supporting, and projector weights, a 20 percent allocation
        allowance, and runtime reserve. This is a planning estimate, not measured
        accelerator memory; offloading and generation settings affect actual use.

        :returns: Estimated bytes, or None if primary weight size is unknown.
        """
        weights, projector = self.estimated_weights_bytes, self.projector_bytes or 0
        for filename, primary in ((self.file, True), (self.clip, False)):
            if filename is None:
                continue
            path = self.get_file_path(filename, weights=primary)
            try:
                local_bytes = path.stat().st_size
            except OSError:
                continue
            if local_bytes > 0:
                if primary:
                    weights = local_bytes
                else:
                    projector = local_bytes
        if weights is None:
            return None
        resident = weights + projector + (self.supporting_weights_bytes or 0)
        return (resident * 120 + 99) // 100 + (self.runtime_bytes or 0)

    @property
    def estimated_vram_bytes(self) -> int | None:
        """Estimate VRAM assuming all model weights are resident on the GPU.

        Includes a fixed 20 percent margin for KV cache and other allocations,
        plus the configured runtime reserve. Actual usage depends on offloading
        and generation settings.

        :returns: Estimated bytes, or None if primary weight size is unknown.
        """
        return self.size

    async def is_downloaded(self) -> bool:
        """Check a successful manifest and each asset without network access.

        Untracked files do not establish snapshot completeness. Deleted or changed
        assets invalidate completion; revisions and snapshot filters have distinct
        manifests. The check requires no inference engine or Hub requests.

        :returns: True only when every recorded asset is still present and unchanged.
        """
        return await asyncio.to_thread(
            assets_complete, Path.home() / "ai", read_manifest(self.state_path)
        )

    async def download_progress(self) -> float:
        """Read shared progress without network access.

        :returns: 0 for missing/failed/stale downloads, an active fraction between
            0 and 1, or 1 after all downloaded assets have been verified. Stages
            have equal weight; discovered byte totals may change the estimate.
        """
        manifest = read_manifest(self.state_path)
        if manifest.status == "complete":
            return 1.0 if await self.is_downloaded() else 0.0
        if manifest.status != "downloading":
            return 0.0
        try:
            with FileLock(str(self.state_path) + ".lock", timeout=0):
                return 1.0 if await self.is_downloaded() else 0.0
        except Timeout:
            return max(0.000001, min(manifest.progress, 0.999999))

    def _set_download_progress(
        self, completed: float, total: float, *, stage: int, stages: int
    ) -> None:
        """Publish byte progress for a snapshot or individual-file stage.

        :param completed: Bytes reconstructed in the current stage.
        :param total: Discovered stage byte total, or zero when unknown.
        :param stage: Zero-based stage index.
        :param stages: Total number of stages.
        :returns: None.
        """
        fraction = min(1.0, max(0.0, completed / total)) if total > 0 else 0.0
        progress = max(0.000001, min((stage + fraction) / stages, 0.999999))
        write_manifest(
            self.state_path, DownloadManifest(status="downloading", progress=progress)
        )

    async def download(self) -> bool:
        """Download all selected assets, sharing transfers across model instances.

        Hub I/O runs in a worker thread; process locks coordinate matching
        configs and serialize selections sharing a repository. Concurrent callers
        with matching configs wait for the same completed result.
        Cancellation retains the lock until active native I/O ends. Failures
        propagate and can be retried using the Hub's resumable downloads.

        Separate weights replace ``transformer/*.{safetensors,bin,gguf}`` in the
        primary snapshot. Snapshot filters never suppress explicit checkpoints or
        projectors. None-valued transfer options preserve Hub defaults.

        :returns: True when all selected assets have downloaded successfully.
        :raises OSError: If transfers, asset validation, or local storage fail.
        :raises ValueError: If a local symlink escapes the model storage root.
        :raises asyncio.CancelledError: After an active transfer finishes on cancellation.
        """
        if await self.is_downloaded():
            return True
        root = Path.home() / "ai"
        state = self.state_path
        state.parent.mkdir(parents=True, exist_ok=True)
        lock_paths = [str(state) + ".lock"]
        for repository in sorted({self.id, self.weights_repo or self.id}):
            key = hashlib.sha256(repository.encode()).hexdigest()
            lock_paths.append(str(safe_path(root, f".downloads/repository-{key}.lock")))
        with ExitStack() as locks:
            while True:
                try:
                    for lock_path in lock_paths:
                        locks.enter_context(FileLock(lock_path, timeout=0))
                    break
                except Timeout:
                    locks.close()
                    await asyncio.sleep(0.05)
            transfer: (
                asyncio.Task[str | list[DryRunFileInfo]]
                | asyncio.Task[str | DryRunFileInfo]
                | None
            ) = None
            try:
                if assets_complete(root, read_manifest(state)):
                    return True
                stages: list[tuple[str, str | None, str | None]] = []
                if self.file is None or self.weights_repo is not None:
                    stages.append((self.id, None, self.revision))
                if self.file is not None:
                    stages.append(
                        (
                            self.weights_repo or self.id,
                            self.file,
                            self.weights_revision
                            if self.weights_repo
                            else self.revision,
                        )
                    )
                if self.clip is not None:
                    stages.append((self.id, self.clip, self.revision))
                files: set[Path] = set()
                for stage, (repo, filename, revision) in enumerate(stages):
                    report = partial(
                        self._set_download_progress, stage=stage, stages=len(stages)
                    )
                    report(0, 0)
                    progress_class = type(
                        "BoundModelDownloadProgress",
                        (ModelDownloadProgress,),
                        {
                            "model_label": self.name or self.id,
                            "report": staticmethod(report),
                        },
                    )
                    directory = safe_path(root, repo)
                    common: DownloadOptions = {}
                    if self.local_files_only is not None:
                        common["local_files_only"] = self.local_files_only
                    if filename is None:
                        ignored = list(self.ignore_patterns or [])
                        if self.file is not None:
                            ignored.extend(
                                [
                                    "transformer/*.safetensors",
                                    "transformer/*.bin",
                                    "transformer/*.gguf",
                                ]
                            )
                        snapshot_options: SnapshotOptions = {}
                        if self.allow_patterns is not None:
                            snapshot_options["allow_patterns"] = self.allow_patterns
                        if ignored:
                            snapshot_options["ignore_patterns"] = ignored
                        if self.max_workers is not None:
                            snapshot_options["max_workers"] = self.max_workers
                        snapshot_transfer = asyncio.create_task(
                            asyncio.to_thread(
                                snapshot_download,
                                repo_id=repo,
                                local_dir=str(directory),
                                revision=revision,
                                tqdm_class=progress_class,
                                **common,
                                **snapshot_options,
                            )
                        )
                        transfer = snapshot_transfer
                        result = await asyncio.shield(snapshot_transfer)
                        if not isinstance(result, str):
                            raise OSError(
                                "Hub did not return a downloaded snapshot path"
                            )
                        if Path(result).resolve() != directory.resolve():
                            raise OSError("Hub snapshot is outside its model directory")
                        snapshot = directory
                        assets = [
                            snapshot / name
                            for name in filter_repo_objects(
                                (
                                    path.relative_to(snapshot).as_posix()
                                    for path in snapshot.rglob("*")
                                    if path.is_file()
                                    and ".cache" not in path.relative_to(snapshot).parts
                                ),
                                allow_patterns=self.allow_patterns,
                                ignore_patterns=ignored,
                            )
                        ]
                        if not assets:
                            raise OSError("Snapshot contains no model assets")
                        files.update(assets)
                    else:
                        expected = self.get_file_path(filename, weights=repo != self.id)
                        file_transfer = asyncio.create_task(
                            asyncio.to_thread(
                                hf_hub_download,
                                repo_id=repo,
                                filename=filename,
                                local_dir=str(directory),
                                revision=revision,
                                tqdm_class=progress_class,
                                **common,
                            )
                        )
                        transfer = file_transfer
                        result = await asyncio.shield(file_transfer)
                        if not isinstance(result, str):
                            raise OSError(
                                "Hub did not return a downloaded checkpoint path"
                            )
                        path = Path(result)
                        if path.resolve() != expected.resolve():
                            raise OSError(
                                "Hub checkpoint is outside its configured asset path"
                            )
                        files.add(directory / filename)
                    report(1, 1)
                records: list[FileRecord] = []
                for path in sorted(files):
                    relative = path.relative_to(root).as_posix()
                    stat = safe_path(root, relative).stat()
                    if not path.is_file():
                        raise OSError("Downloaded model asset is not a file")
                    records.append(
                        FileRecord(
                            path=relative, size=stat.st_size, mtime_ns=stat.st_mtime_ns
                        )
                    )
                write_manifest(
                    state, DownloadManifest(status="complete", files=records)
                )
                return True
            except BaseException:
                # A cancelled asyncio caller cannot stop native Hub I/O. Even repeated
                # cancellation must not release the cross-process transfer lock early.
                if transfer is not None:
                    while not transfer.done():
                        try:
                            await asyncio.shield(transfer)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not transfer.cancelled():
                        transfer.exception()
                write_manifest(state, DownloadManifest(status="failed"))
                raise
