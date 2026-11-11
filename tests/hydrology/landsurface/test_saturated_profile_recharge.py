"""Regression checks for surface water crossing a saturated soil profile."""

from typing import Any

import numpy as np
import pytest

from geb.hydrology.landcovers import GRASSLAND_LIKE
from geb.hydrology.landsurface.water import generate_rainfall_lookup_table, infiltration


@pytest.fixture
def saturated_profile() -> dict[str, Any]:
    """Create a saturated six-layer profile with a front at the soil bottom.

    Returns:
        Infiltration arguments in meters, seconds and joules per square meter.
    """
    layer_height_m: np.ndarray = np.array(
        [0.05, 0.1, 0.15, 0.3, 0.4, 1.0], dtype=np.float32
    )
    saturated_water_m: np.ndarray = layer_height_m * np.float32(0.45)
    return {
        "seed": np.int64(42),
        "ws": saturated_water_m,
        "wres": layer_height_m * np.float32(0.05),
        "saturated_hydraulic_conductivity_m_per_s": np.full(6, 1e-4, dtype=np.float32),
        "groundwater_toplayer_conductivity_m_per_s": np.float32(1e-4),
        "land_use_type": np.int32(GRASSLAND_LIKE),
        "w": saturated_water_m.copy(),
        "topwater_m": np.float32(0.01),
        "wetting_front_depth_m": np.float32(layer_height_m.sum()),
        "wetting_front_suction_head_m": np.float32(0.2),
        "wetting_front_moisture_deficit": np.float32(0.1),
        "green_ampt_active_layer_idx": 5,
        "variable_runoff_shape_beta": np.float32(0.4),
        "bubbling_pressure_m_positive": np.full(6, 0.2, dtype=np.float32),
        "soil_layer_height_m": layer_height_m,
        "lambda_pore_size_distribution": np.full(6, 0.25, dtype=np.float32),
        "soil_enthalpy_top_layer_J_per_m2": np.float32(1e6),
        "solid_heat_capacity_top_layer_J_per_m2_K": np.float32(1e5),
        "rain_temperature_C": np.float32(0),
        "new_surface_water_input_m": np.float32(0.01),
        "rainfall_lookup_table": generate_rainfall_lookup_table(sigma=0.8),
        "distribute_rainfall_lognormally": False,
    }


@pytest.mark.parametrize("restrictive_layer", [0, 3, 5])
def test_saturated_recharge_respects_soil_bottleneck(
    saturated_profile: dict[str, Any], restrictive_layer: int
) -> None:
    """Keep a permeable aquifer from bypassing a low-conductivity soil layer.

    Args:
        saturated_profile: Saturated infiltration inputs.
        restrictive_layer: Index of the low-conductivity soil layer.

    Returns:
        None.

    Raises:
        AssertionError: If recharge exceeds soil capacity or water is lost.
    """  # noqa: DOC202, DOC502 - Document the test outcome and assertion failures.
    conductivity_m_per_s: float = 1e-8
    saturated_profile["saturated_hydraulic_conductivity_m_per_s"][restrictive_layer] = (
        conductivity_m_per_s
    )
    initial_soil_m: np.ndarray = saturated_profile["w"].copy()
    result: tuple = infiltration(**saturated_profile)
    crust_factor: float = 0.2 if restrictive_layer == 0 else 1.0
    capacity_m: float = conductivity_m_per_s * crust_factor * 3600
    assert float(result[2]) == pytest.approx(capacity_m, rel=1e-5)
    assert float(result[0] + result[1] + result[2]) == pytest.approx(0.01, abs=1e-8)
    np.testing.assert_array_equal(saturated_profile["w"], initial_soil_m)


def test_saturated_frozen_soil_blocks_surface_recharge(
    saturated_profile: dict[str, Any],
) -> None:
    """Prevent surface water bypassing frozen soil when the wetting front is deep.

    Args:
        saturated_profile: Saturated infiltration inputs.

    Returns:
        None.

    Raises:
        AssertionError: If frozen soil transmits surface water to groundwater.
    """  # noqa: DOC202, DOC502 - Document the test outcome and assertion failures.
    saturated_profile["soil_enthalpy_top_layer_J_per_m2"] = np.float32(-1e7)
    result: tuple = infiltration(**saturated_profile)
    assert result[2] == 0
    assert result[3] == 0
    assert float(result[0] + result[1]) == pytest.approx(0.01, abs=1e-8)


def test_permeable_saturated_soil_still_recharges(
    saturated_profile: dict[str, Any],
) -> None:
    """Pass available rain through an unfrozen, highly conductive soil profile.

    Args:
        saturated_profile: Saturated infiltration inputs.

    Returns:
        None.

    Raises:
        AssertionError: If valid throughflow is blocked or water is lost.
    """  # noqa: DOC202, DOC502 - Document the test outcome and assertion failures.
    result: tuple = infiltration(**saturated_profile)
    assert float(result[2]) == pytest.approx(0.01, abs=1e-8)
    assert float(result[0] + result[1] + result[2]) == pytest.approx(0.01, abs=1e-8)
