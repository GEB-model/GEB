"""Regression tests for automatic input version update failures."""

import logging
from unittest.mock import Mock, call

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
    assert "1.0.0b36" not in version_updates.VERSION_UPDATES
    assert "1.0.0b37" not in version_updates.VERSION_UPDATES


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


@pytest.mark.parametrize(
    "stored_version,target_version,method_names",
    [
        ("1.0.0b31", "1.0.0b32", ["setup_waterbodies", "setup_weirs"]),
        ("1.0.0b32", "1.0.0b33", ["setup_waterbodies", "setup_weirs"]),
        ("1.0.0b33", "1.0.0b34", ["setup_weirs"]),
        ("1.0.0b34", "1.0.0b35", ["setup_waterbodies", "setup_weirs"]),
    ],
)
def test_waterbody_input_migration(
    monkeypatch: pytest.MonkeyPatch,
    stored_version: str,
    target_version: str,
    method_names: list[str],
) -> None:
    """Rebuild affected waterbody inputs and require rerunning spinup.

    Args:
        monkeypatch: Fixture for fixing the target package version.
        stored_version: Version of the existing inputs.
        target_version: Version to migrate to.
        method_names: Build methods required by the migration.
    """
    monkeypatch.setattr(version_updates, "__version__", target_version)
    builder: Mock = Mock()
    methods: dict[str, dict[str, list[int]]] = {
        method_name: {} for method_name in method_names
    }
    with pytest.raises(RuntimeError, match="Rerun spinup"):
        version_updates.get_and_maybe_do_version_updates(
            stored_version,
            logging.getLogger(__name__),
            build_model=builder,
            methods=methods,
        )
    assert builder.update.call_args_list == [
        call({method_name: {}}) for method_name in method_names
    ]
    builder.set_version.assert_called_once_with(target_version)


def test_barrier_file_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add the barrier file to existing inputs without requiring a new spinup.

    Args:
        monkeypatch: Fixture for fixing the package version.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b36")
    builder: Mock = Mock()
    updates: list[str] = version_updates.get_and_maybe_do_version_updates(
        "1.0.0b35",
        logging.getLogger(__name__),
        build_model=builder,
        methods={"setup_weirs": {}},
    )
    assert updates == []
    builder.update.assert_called_once_with({"setup_weirs": {}})
    builder.set_version.assert_called_once_with("1.0.0b36")
