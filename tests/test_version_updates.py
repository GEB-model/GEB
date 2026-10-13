"""Regression tests for karst input migrations."""

import logging

import pytest

from geb.build import version_updates


def test_karst_input_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Notify existing input users how to build karst coverage and rerun spinup.

    Args:
        monkeypatch: Fixture fixing the target version to the karst release.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b35")
    updates: list[str] = version_updates.get_and_maybe_do_version_updates(
        "1.0.0b34", logging.getLogger(__name__)
    )
    assert len(updates) == 1
    assert "setup_karst" in updates[0]
    assert "rerun spinup" in updates[0]
    assert "karst disabled require no input changes" in updates[0]
    assert "new karst store" in updates[0]
    assert "may contain only zeros" in updates[0]
    assert "1.0.0b37" in version_updates.VERSION_UPDATES
