"""Download utility types for Hub options that preserve native defaults."""

from typing import TypedDict


class DownloadOptions(TypedDict, total=False):
    """Optional settings shared by file and snapshot transfers."""

    local_files_only: bool


class SnapshotOptions(TypedDict, total=False):
    """Optional settings specific to a repository snapshot transfer."""

    allow_patterns: list[str]
    ignore_patterns: list[str]
    max_workers: int
