"""Check input migration selection and propagation of automatic-update failures."""

import logging
from unittest.mock import Mock, call

import pytest

from geb.build import version_updates


@pytest.mark.parametrize(
    ("stored_version", "expected_methods", "expected_versions"),
    [
        (
            "1.0.0b36",
            [
                "setup_hydrography",
                "setup_coastal_water_levels",
                "setup_hydrography",
                "setup_floodplains",
            ],
            ["1.0.0b37", "1.0.0b38"],
        ),
        ("1.0.0b37", ["setup_hydrography", "setup_floodplains"], ["1.0.0b38"]),
        ("1.0.0b38", [], []),
    ],
)
def test_waterbody_input_migration(
    monkeypatch: pytest.MonkeyPatch,
    stored_version: str,
    expected_methods: list[str],
    expected_versions: list[str],
) -> None:
    """Apply only pending migrations while retaining package version b38.

    Args:
        monkeypatch: Fixture for fixing the target package version.
        stored_version: Input version before the update.
        expected_methods: Build methods that must run, in order.
        expected_versions: Successfully completed migration versions.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b38")
    builder: Mock = Mock()
    methods: dict[str, dict[str, str]] = {
        name: {}
        for name in [
            "setup_hydrography",
            "setup_coastal_water_levels",
            "setup_floodplains",
        ]
    }
    version_updates.get_and_maybe_do_version_updates(
        stored_version,
        logging.getLogger(__name__),
        build_model=builder,
        methods=methods,
    )
    assert builder.update.call_args_list == [
        call({name: {}}) for name in expected_methods
    ]
    assert builder.set_version.call_args_list == [
        call(version) for version in expected_versions
    ]
    builder.set_current_version.assert_called_once()


@pytest.mark.parametrize("stored_version", ["1.0.0b39", "1.0.0b40"])
def test_newer_inputs_are_not_downgraded(
    monkeypatch: pytest.MonkeyPatch, stored_version: str
) -> None:
    """Preserve newer input markers until an explicit rebuild succeeds.

    Args:
        monkeypatch: Fixture for fixing the running package version.
        stored_version: Input marker from a newer branch version.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b38")
    builder: Mock = Mock()
    with pytest.raises(ValueError, match="Refusing to downgrade"):
        version_updates.get_and_maybe_do_version_updates(
            stored_version, logging.getLogger(__name__), build_model=builder, methods={}
        )
    builder.update.assert_not_called()
    builder.set_version.assert_not_called()
    builder.set_current_version.assert_not_called()


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
