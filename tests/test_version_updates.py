"""Regression tests for karst input migrations."""

import logging

import pytest

from geb.build import version_updates


def test_karst_input_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Notify existing input users how to build karst coverage and rerun spinup.

    Args:
        monkeypatch: Fixture fixing the target version to the karst release.

    Returns:
        None.

    Raises:
        AssertionError: If the required migration instructions are missing.
    """  # noqa: DOC202, DOC502
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b35")
    updates: list[str] = version_updates.get_and_maybe_do_version_updates(
        "1.0.0b34", logging.getLogger(__name__)
    )
    karst_notices: list[str] = [notice for notice in updates if "setup_karst" in notice]
    assert len(karst_notices) == 1
    notice: str = karst_notices[0]
    assert "every hydrology run" in notice
    assert "Remove hydrology.karst.enabled" in notice
    assert "reasonable_default_build.yml" in notice
    assert "rerun spinup and simulation" in notice
    assert "missing or invalid" in notice
