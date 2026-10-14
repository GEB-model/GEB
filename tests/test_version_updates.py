"""Regression tests for karst input migrations."""

import logging

import pytest

from geb.build import version_updates


@pytest.mark.parametrize("stored_version", ["1.0.0b33", "1.0.0b34"])
def test_karst_input_migration(
    monkeypatch: pytest.MonkeyPatch, stored_version: str
) -> None:
    """Notify existing input users how to build karst coverage and rerun spinup.

    Args:
        monkeypatch: Fixture fixing the target version to the karst release.
        stored_version: Input version before karst was added.

    Returns:
        None.

    Raises:
        AssertionError: If the required migration instructions are missing.
    """  # noqa: DOC202, DOC502
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b35")
    updates: list[str] = version_updates.get_and_maybe_do_version_updates(
        stored_version, logging.getLogger(__name__)
    )
    karst_notices: list[str] = [
        notice for notice in updates if "Karst is included" in notice
    ]
    assert len(karst_notices) == 1
    notice: str = karst_notices[0]
    assert "every hydrology run" in notice
    assert "Remove hydrology.karst.enabled" in notice
    assert "reasonable_default_build.yml" in notice
    assert "rerun spinup and simulation" in notice
    assert any("Re-run `setup_karst`" in update for update in updates)
