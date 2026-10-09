"""Regression tests for automatic input version update failures."""

import logging
from contextlib import nullcontext
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
    "stored_version,target_version,method_names,manual_notice",
    [
        ("1.0.0b31", "1.0.0b32", ["setup_waterbodies", "setup_weirs"], "Rerun spinup"),
        ("1.0.0b32", "1.0.0b33", ["setup_groundwater"], None),
        ("1.0.0b33", "1.0.0b34", ["setup_waterbodies"], None),
        ("1.0.0b34", "1.0.0b35", [], "karst"),
    ],
)
def test_waterbody_input_migration(
    monkeypatch: pytest.MonkeyPatch,
    stored_version: str,
    target_version: str,
    method_names: list[str],
    manual_notice: str | None,
) -> None:
    """Run the methods and manual notices registered for each input version.

    Args:
        monkeypatch: Fixture for fixing the target package version.
        stored_version: Version of the existing inputs.
        target_version: Version to migrate to.
        method_names: Build methods required by the migration.
        manual_notice: Expected manual instruction, or None for automatic updates.
    """
    monkeypatch.setattr(version_updates, "__version__", target_version)
    builder: Mock = Mock()
    methods: dict[str, dict[str, list[int]]] = {
        method_name: {} for method_name in method_names
    }
    with (
        pytest.raises(RuntimeError, match=manual_notice)
        if manual_notice is not None
        else nullcontext()
    ):
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


@pytest.mark.parametrize("update_fails", [False, True])
def test_farmer_input_migration(
    monkeypatch: pytest.MonkeyPatch, update_fails: bool
) -> None:
    """Rebuild farmer inputs in order and require a new spinup and simulation.

    Args:
        monkeypatch: Fixture fixing the package version.
        update_fails: Whether rebuilding farms fails.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b36")
    method_names: list[str] = [
        "setup_well_prices_by_reference_year_global",
        "setup_create_farms",
        "setup_farmer_household_characteristics",
        "setup_crops",
        "setup_farmer_crop_calendar",
        "setup_farmer_characteristics",
        "setup_crop_prices",
    ]
    methods: dict[str, dict] = {name: {} for name in method_names}
    builder: Mock = Mock()
    if update_fails:
        builder.update.side_effect = [None, ValueError("Farm rebuild failed")]
        with pytest.raises(RuntimeError, match="error occurred during auto-update"):
            version_updates.get_and_maybe_do_version_updates(
                "1.0.0b35", logging.getLogger(__name__), builder, methods
            )
        builder.set_version.assert_not_called()
        builder.set_current_version.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="after rebuilding farmers"):
            version_updates.get_and_maybe_do_version_updates(
                "1.0.0b35", logging.getLogger(__name__), builder, methods
            )
        assert builder.update.call_args_list == [
            call({name: {}}) for name in method_names
        ]
        builder.set_version.assert_called_once_with("1.0.0b36")
        builder.set_current_version.assert_called_once_with()
        builder.reset_mock()
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b36", logging.getLogger(__name__), builder, methods
        )
        builder.update.assert_not_called()
