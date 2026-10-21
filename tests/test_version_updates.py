"""Regression tests for automatic input version update failures."""

import logging
from unittest.mock import Mock, call

import pytest

from geb.build import version_updates


def test_version_update_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the old input version when an automatic update fails.

    Args:
        monkeypatch: Fixture for isolating the migration registry.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b31")
    monkeypatch.setattr(
        version_updates,
        "VERSION_UPDATES",
        {"1.0.0b31": ["[update-method;setup_hydrography]"]},
    )
    builder: Mock = Mock()
    logger: logging.Logger = logging.getLogger(__name__)
    methods: dict[str, dict[str, str]] = {"setup_hydrography": {}}
    builder.update.side_effect = ValueError("Invalid source data")
    with pytest.raises(RuntimeError, match="error occurred during auto-update"):
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b30", logger, build_model=builder, methods=methods
        )
    builder.set_version.assert_not_called()
    builder.set_current_version.assert_not_called()
    builder.update.assert_called_once_with(methods)


def test_waterbody_input_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuild waterbody inputs and require a new spinup for this version.

    Args:
        monkeypatch: Fixture for fixing the target package version.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b37")
    builder: Mock = Mock()
    method_names: list[str] = ["setup_gtsm_station_data", "setup_waterbodies"]
    methods: dict[str, dict[str, str]] = {name: {} for name in method_names}
    with pytest.raises(RuntimeError, match="Rerun spinup"):
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b36",
            logging.getLogger(__name__),
            build_model=builder,
            methods=methods,
        )
    assert builder.update.call_args_list == [
        call({method_name: {}}) for method_name in method_names
    ]
    builder.set_version.assert_called_once_with("1.0.0b37")
