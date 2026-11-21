"""Verify byte reporting independently of Hub console rendering preferences."""

import io
from unittest.mock import Mock

import pytest

from broca.utils._download_progress import ModelDownloadProgress


@pytest.mark.parametrize("disabled", [True, False])
def test_resumed_and_dynamic_byte_totals_are_reported(disabled: bool) -> None:
    """Report resume offsets, increments, and newly discovered snapshot totals.

    :param disabled: Whether console rendering is disabled.
    :returns: None.
    """
    report = Mock()
    progress_type = type(
        "TestDownloadProgress",
        (ModelDownloadProgress,),
        {"model_label": "Test model", "report": staticmethod(report)},
    )
    with progress_type(
        total=100, initial=20, unit="B", disable=disabled, file=io.StringIO()
    ) as progress:
        report.assert_called_with(20.0, 100.0)
        progress.update(30)
        report.assert_called_with(50.0, 100.0)
        progress.total = 200
        progress.refresh()
        report.assert_called_with(50.0, 200.0)
        if not disabled:
            assert "Test model" in progress.desc


@pytest.mark.parametrize(
    "options", [{"unit": "it"}, {"unit": "B", "name": "huggingface_hub.transfer"}]
)
def test_file_counts_and_transport_bytes_do_not_overwrite_reconstruction(
    options: dict[str, str],
) -> None:
    """Ignore secondary bars to avoid double-counting compressed Xet transfers.

    :param options: File-count or transport-bar options.
    :returns: None.
    """
    report = Mock()
    progress_type = type(
        "TestDownloadProgress",
        (ModelDownloadProgress,),
        {"model_label": "Test model", "report": staticmethod(report)},
    )
    with progress_type(total=100, disable=True, **options) as progress:
        progress.update(20)
        progress.refresh()
    report.assert_not_called()
