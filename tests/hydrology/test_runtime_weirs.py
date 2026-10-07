"""Runtime configuration for enabling and disabling river barriers."""

from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from geb.config_schema import RoutingConfig
from geb.hydrology.routing import Routing


@pytest.fixture
def routing() -> Mock:
    """Create routing dependencies without initializing a full model.

    Returns:
        Mock routing module with three river cells and stored barrier inputs.
    """
    routing_module: Mock = Mock(spec=Routing, model=Mock(), grid=Mock())
    routing_module.config = {}
    routing_module.ldd = np.array([6, 6, 5], dtype=np.uint8)
    routing_module.model.files = {
        "grid": {
            "routing/weir_height_m": "heights",
        }
    }
    routing_module.grid.load2d.return_value = np.array([-2, 2, 0], dtype=np.float32)
    return routing_module


@pytest.mark.parametrize("config", [{}, {"weirs": True}])
def test_enabled_weirs_load_stored_inputs(
    routing: Mock, config: dict[str, bool]
) -> None:
    """Load stored barrier heights when barriers are enabled.

    Args:
        routing: Mock routing module with stored barrier inputs.
        config: Runtime routing settings, including the default behavior.
    """
    routing.config = config
    Routing.load_weirs(cast(Routing, routing))

    np.testing.assert_array_equal(routing.weir_height_m, [-2, 2, 0])
    np.testing.assert_array_equal(routing.instream_dam, [False, False, False])
    assert routing.grid.load2d.call_count == 1


def test_disabled_weirs_do_not_require_input_files(routing: Mock) -> None:
    """Remove river barriers even if barrier files are absent.

    Args:
        routing: Mock routing module with stored barrier inputs.
    """
    routing.config = {"weirs": False}
    routing.model.files = {"grid": {}}
    Routing.load_weirs(cast(Routing, routing))

    np.testing.assert_array_equal(routing.weir_height_m, [0, 0, 0])
    assert routing.weir_height_m.dtype == np.float32
    assert not routing.instream_dam.any()
    routing.grid.load2d.assert_not_called()


@pytest.fixture
def routing_parameters() -> dict[str, object]:
    """Provide required channel geometry for routing configuration validation.

    Returns:
        Routing settings containing the required river depth parameters.
    """
    return {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
            }
        }
    }


def test_weirs_config_defaults_to_enabled(
    routing_parameters: dict[str, object],
) -> None:
    """Enable barriers by default and accept an explicit boolean override.

    Args:
        routing_parameters: Required river depth settings.
    """
    assert RoutingConfig.model_validate(routing_parameters).weirs is True
    assert (
        RoutingConfig.model_validate(routing_parameters | {"weirs": False}).weirs
        is False
    )


def test_weirs_config_rejects_non_boolean_values(
    routing_parameters: dict[str, object],
) -> None:
    """Reject ambiguous values for the runtime barrier switch.

    Args:
        routing_parameters: Required river depth settings.
    """
    with pytest.raises(ValidationError, match="weirs"):
        RoutingConfig.model_validate(routing_parameters | {"weirs": "false"})


@pytest.mark.parametrize("grid_name", ["routing/instream_dam", "routing/gated_dam"])
def test_enabled_weirs_load_instream_dams(routing: Mock, grid_name: str) -> None:
    """Read instream dams from the new grid name or the old one.

    Args:
        routing: Mock routing module with stored barrier inputs.
        grid_name: Current or old input grid name.
    """
    routing.model.files["grid"][grid_name] = "instream_dams"
    routing.grid.load2d.side_effect = [
        np.array([-2, 2, 0], dtype=np.float32),
        np.array([True, False, False]),
    ]
    Routing.load_weirs(cast(Routing, routing))
    np.testing.assert_array_equal(routing.instream_dam, [True, False, False])


def test_instream_dam_grid_takes_priority(routing: Mock) -> None:
    """Use the new grid if an older gated_dam grid is also present.

    Args:
        routing: Mock routing module with stored barrier inputs.
    """
    routing.model.files["grid"].update(
        {"routing/instream_dam": "new_dams", "routing/gated_dam": "old_dams"}
    )
    routing.grid.load2d.side_effect = [
        np.array([-2, 2, 0], dtype=np.float32),
        np.array([True, False, False]),
    ]
    Routing.load_weirs(cast(Routing, routing))
    assert routing.grid.load2d.call_args.args == ("new_dams",)
    np.testing.assert_array_equal(routing.instream_dam, [True, False, False])
