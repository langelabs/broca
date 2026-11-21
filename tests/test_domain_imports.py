"""Tests for the public Broca package boundaries."""

import importlib

import pytest

from broca import AiModel


def test_ai_domain_exports_model() -> None:
    """Expose model configuration without importing optional backends.

    :returns: None.
    """
    assert AiModel.__name__ == "AiModel"


@pytest.mark.parametrize(
    "module_name",
    [
        "broca.contracts",
        "broca.servers",
        "broca.plugin",
    ],
)
def test_legacy_domain_imports_are_removed(module_name: str) -> None:
    """Reject removed contract, server, and plugin import paths.

    :param module_name: Removed module path.
    :returns: None.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
