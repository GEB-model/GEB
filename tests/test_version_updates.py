"""Check input migration selection and propagation of automatic-update failures."""

import logging
from unittest.mock import Mock, call

import pytest

from geb.build import version_updates


@pytest.mark.parametrize("stored_version", ["1.0.0b38", "1.0.0b39", "1.0.0b40"])
def test_waterbody_input_migration(
    monkeypatch: pytest.MonkeyPatch, stored_version: str
) -> None:
    """Rebuild merged routing inputs even when a branch already recorded b39.

    Args:
        monkeypatch: Fixture for fixing the target package version.
        stored_version: Input version before the merged-code update.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b40")
    builder: Mock = Mock()
    method_names: list[str] = [
        "setup_hydrography",
        "setup_coastal_water_levels",
        "setup_retention_basins",
        "setup_floodplains",
        "setup_waterbodies",
    ]
    methods: dict[str, dict[str, str]] = {name: {} for name in method_names}
    if stored_version != "1.0.0b40":
        with pytest.raises(
            RuntimeError, match="Rerun spinup, simulation, and evaluation"
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
        builder.set_version.assert_called_once_with("1.0.0b40")
    else:
        version_updates.get_and_maybe_do_version_updates(
            stored_version,
            logging.getLogger(__name__),
            build_model=builder,
            methods=methods,
        )
        builder.update.assert_not_called()
        builder.set_version.assert_not_called()
    builder.set_current_version.assert_called_once()


@pytest.mark.parametrize("notice_first", [False, True])
def test_failed_migration_raises_without_advancing_version(
    monkeypatch: pytest.MonkeyPatch, notice_first: bool
) -> None:
    """Failures are fatal whether or not a manual notice was collected beforehand.

    Args:
        monkeypatch: Fixture for installing an isolated migration sequence.
        notice_first: Whether a manual instruction precedes the failing method.

    """
    instructions: list[str] = ["[update-method;setup_karst]", "[manual] Rerun spinup"]
    if notice_first:
        instructions.reverse()
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b36")
    monkeypatch.setattr(version_updates, "VERSION_UPDATES", {"1.0.0b36": instructions})
    builder: Mock = Mock()
    failure: ValueError = ValueError("Invalid karst geometry")
    builder.update.side_effect = failure
    with pytest.raises(
        RuntimeError, match="error occurred during auto-update"
    ) as caught:
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b35",
            logging.getLogger(__name__),
            build_model=builder,
            methods={"setup_karst": {}},
        )
    builder.set_version.assert_not_called()
    builder.set_current_version.assert_not_called()
    assert caught.value.__cause__ is failure
