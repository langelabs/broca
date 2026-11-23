"""Resolve public pipeline exports without importing inference backends eagerly."""

from collections.abc import Collection
from importlib import import_module


def resolve_pipeline_export(package: str, exports: Collection[str], name: str) -> object:
    """Resolve a lazy pipeline attribute of an inference package.

    :param package: Full name of the inference package.
    :param exports: Public names exported by the package.
    :param name: Requested attribute name.
    :returns: The class or shared pipeline from the implementation module.
    :raises AttributeError: If the package does not export the name.
    """
    if name not in exports:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    return getattr(import_module("._pipeline", package), name)
