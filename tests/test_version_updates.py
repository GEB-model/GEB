"""Check the waterbody input update and required spinup notice."""

import logging
from unittest.mock import Mock, call

import pytest

from geb.build import version_updates


@pytest.mark.parametrize("stored_version", ["1.0.0b36", "1.0.0b37", "1.0.0b38"])
def test_waterbody_input_migration(
    monkeypatch: pytest.MonkeyPatch, stored_version: str
) -> None:
    """Rebuild waterbody inputs and require a new spinup for this version.

    Args:
        monkeypatch: Fixture for fixing the target package version.
        stored_version: Input version before the combined update.
    """
    monkeypatch.setattr(version_updates, "__version__", "1.0.0b39")
    builder: Mock = Mock()
    method_names: list[str] = [
        "setup_gtsm_station_data",
        "setup_hydrography",
        "setup_floodplains",
        "setup_waterbodies",
    ]
    methods: dict[str, dict[str, str]] = {name: {} for name in method_names}
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
    builder.set_version.assert_called_once_with("1.0.0b39")
