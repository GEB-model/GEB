"""Regression tests for automatic input version update failures."""

import logging
from unittest.mock import Mock, call

import pytest

from geb.build import version_updates


@pytest.mark.parametrize("update_fails", [False, True])
def test_version_update_failure_propagates(
    monkeypatch: pytest.MonkeyPatch, update_fails: bool
) -> None:
    """Only mark successful input migrations as current.

    Args:
        monkeypatch: Fixture for isolating the migration registry.
        update_fails: Whether the build method raises an error.

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
    if update_fails:
        builder.update.side_effect = ValueError("Invalid source data")
        with pytest.raises(RuntimeError, match="error occurred during auto-update"):
            version_updates.get_and_maybe_do_version_updates(
                "1.0.0b30", logger, build_model=builder, methods=methods
            )
        builder.set_version.assert_not_called()
        builder.set_current_version.assert_not_called()
    else:
        updates: list[str] = version_updates.get_and_maybe_do_version_updates(
            "1.0.0b30", logger, build_model=builder, methods=methods
        )
        assert updates == []
        builder.set_version.assert_called_once_with("1.0.0b31")
        builder.set_current_version.assert_called_once_with()
    builder.update.assert_called_once_with(methods)


def test_weir_input_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the required weir grid when updating b31 inputs.

    Args:
        monkeypatch: Fixture for fixing the target package version.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b32")
    builder: Mock = Mock()
    methods: dict[str, dict[str, list[int]]] = {
        "setup_waterbodies": {},
        "setup_weirs": {},
    }
    with pytest.raises(RuntimeError, match="Rerun spinup"):
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b31",
            logging.getLogger(__name__),
            build_model=builder,
            methods=methods,
        )
    assert builder.update.call_args_list == [
        call({"setup_waterbodies": {}}),
        call({"setup_weirs": {}}),
    ]
    builder.set_version.assert_called_once_with("1.0.0b32")
