"""Check input migration selection and propagation of automatic-update failures."""

import logging
from unittest.mock import Mock, call

import pytest

from geb.build import version_updates


@pytest.mark.parametrize(
    ("stored_version", "expected_methods"),
    [
        ("1.0.0b36", []),
        ("1.0.0b37", []),
        ("1.0.0b38", []),
    ],
)
def test_waterbody_input_migration(
    monkeypatch: pytest.MonkeyPatch,
    stored_version: str,
    expected_methods: list[str],
) -> None:
    """Apply only pending migrations while retaining package version b38.

    Args:
        monkeypatch: Fixture for fixing the target package version.
        stored_version: Input version before the update.
        expected_methods: Build methods that must run, in order.
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
    builder.set_version.assert_not_called()
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


@pytest.mark.parametrize("apply", [False, True])
def test_duplicate_migration_methods_run_once_in_recipe_order(
    monkeypatch: pytest.MonkeyPatch, apply: bool
) -> None:
    """Deduplicate historical updates while preserving the current build recipe.

    Args:
        monkeypatch: Fixture installing an isolated set of historical updates.
        apply: Whether to execute updates or only report required changes.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b38")
    monkeypatch.setattr(
        version_updates,
        "VERSION_UPDATES",
        {
            "1.0.0b29": [
                "[update-method;setup_elevation]",
                "[update-method;setup_hydrography]",
            ],
            "1.0.0b35": ["[update-method;setup_elevation]"],
            "1.0.0b38": [
                "[update-method;setup_hydrography]",
                "[update-method;setup_floodplains]",
            ],
        },
    )
    recipe: dict[str, object] = {
        "setup_hydrography": {"custom": True},
        "setup_elevation": None,
        "setup_floodplains": {"dem_name": "fathomdem"},
    }
    builder: Mock = Mock()
    notices: list[str] = version_updates.get_and_maybe_do_version_updates(
        "1.0.0b28",
        logging.getLogger(__name__),
        build_model=builder if apply else None,
        methods=recipe,
    )
    if apply:
        assert builder.update.call_args_list == [
            call({name: args}) for name, args in recipe.items()
        ]
        builder.set_current_version.assert_called_once()
    else:
        assert len(notices) == 3
        assert len(set(notices)) == 3
        builder.update.assert_not_called()


def test_failed_batch_keeps_original_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not mark intermediate versions complete after reordered updates fail.

    Args:
        monkeypatch: Fixture defining two migrations and the running version.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b38")
    monkeypatch.setattr(
        version_updates,
        "VERSION_UPDATES",
        {
            "1.0.0b29": ["[update-method;setup_hydrography]"],
            "1.0.0b38": ["[update-method;setup_floodplains]"],
        },
    )
    builder: Mock = Mock()
    failure: RuntimeError = RuntimeError("Floodplain failed")
    builder.update.side_effect = [None, failure]
    with pytest.raises(
        RuntimeError, match="error occurred during auto-update"
    ) as caught:
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b28",
            logging.getLogger(__name__),
            build_model=builder,
            methods={"setup_hydrography": None, "setup_floodplains": None},
        )
    assert builder.update.call_count == 2
    assert caught.value.__cause__ is failure
    builder.set_version.assert_not_called()
    builder.set_current_version.assert_not_called()


def test_missing_update_method_fails_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the full pending method set before performing expensive updates.

    Args:
        monkeypatch: Fixture defining one required migration.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b38")
    monkeypatch.setattr(
        version_updates,
        "VERSION_UPDATES",
        {
            "1.0.0b38": ["[update-method;setup_floodplains]"],
        },
    )
    builder: Mock = Mock()
    with pytest.raises(RuntimeError, match="error occurred during auto-update"):
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b28", logging.getLogger(__name__), build_model=builder, methods={}
        )
    builder.update.assert_not_called()
    builder.set_current_version.assert_not_called()


def test_large_scale_retry_runs_only_remaining_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run group B from b28 and preserve the outstanding manual notices.

    Args:
        monkeypatch: Fixture for fixing the target package version.

    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b38")
    builder: Mock = Mock()
    remaining_methods: list[str] = [
        "setup_SPEI",
        "setup_discharge_observations",
        "setup_flood_observations",
        "setup_flood_protection_standards",
    ]
    methods: dict[str, dict[str, str]] = {
        name: {} for name in ["setup_forcing", *remaining_methods]
    }
    # Manual rerun notices remain actionable after the automatic batch succeeds.
    with pytest.raises(RuntimeError, match="IMPORTANT"):
        version_updates.get_and_maybe_do_version_updates(
            "1.0.0b28",
            logging.getLogger(__name__),
            build_model=builder,
            methods=methods,
        )
    assert builder.update.call_args_list == [
        call({name: {}}) for name in remaining_methods
    ]
    builder.set_current_version.assert_called_once()
