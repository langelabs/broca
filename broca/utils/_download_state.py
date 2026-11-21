"""Download utilities for safe paths and atomic, credential-free manifests."""

from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt


class FileRecord(BaseModel):
    """Record an AI-root-relative asset and its completed filesystem metadata."""

    model_config = ConfigDict(extra="forbid")
    path: str
    size: NonNegativeInt
    mtime_ns: NonNegativeInt


class DownloadManifest(BaseModel):
    """Represent missing, active, failed, or fully verified download state."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["missing", "downloading", "failed", "complete"] = "missing"
    progress: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    files: list[FileRecord] = Field(default_factory=list)


def safe_path(root: Path, relative: str) -> Path:
    """Keep a relative path and resolved symlinks within their storage root.

    :param root: Absolute storage directory.
    :param relative: Storage-relative POSIX path.
    :returns: Validated absolute path without changing the readable layout.
    :raises ValueError: If the path traverses outside the root.
    """
    candidate = root / relative
    if (
        PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        or not candidate.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("Model paths must remain inside their storage directory")
    return candidate


def read_manifest(path: Path) -> DownloadManifest:
    """Read persistent state, treating missing or malformed data as incomplete.

    :param path: Manifest location.
    :returns: Validated manifest or a missing-state record.
    """
    try:
        return DownloadManifest.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return DownloadManifest()


def write_manifest(path: Path, manifest: DownloadManifest) -> None:
    """Atomically publish state so readers never observe a partial JSON write.

    :param path: Manifest location in an existing directory.
    :param manifest: Credential-free download metadata.
    :returns: None.
    :raises OSError: If local storage fails.
    """
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(manifest.model_dump_json())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def assets_complete(root: Path, manifest: DownloadManifest) -> bool:
    """Validate completed files without Hub access or hashing large checkpoints.

    :param root: AI storage root.
    :param manifest: Previously read download manifest.
    :returns: True when every completed asset retains its size and modification time.
    """
    if manifest.status != "complete" or not manifest.files:
        return False
    try:
        for record in manifest.files:
            path = safe_path(root, record.path)
            stat = path.stat()
            if not path.is_file() or (stat.st_size, stat.st_mtime_ns) != (
                record.size,
                record.mtime_ns,
            ):
                return False
    except (OSError, ValueError):
        return False
    return True
