"""Regression checks for the water available to kinematic routing losses."""

import numpy as np
import pytest

from geb.hydrology.routing.kinematic import (
    _solve_kinematic_reach,
    update_node_kinematic,
)
from geb.hydrology.routing.topology import RetentionCoupling


@pytest.fixture
def no_retention() -> RetentionCoupling:
    """Supply empty basin tables for an isolated river reach.

    Returns:
        Retention state with no linked basins.
    """
    return RetentionCoupling(
        storage_m3=np.zeros(0, dtype=np.float32),
        inflow_m3=np.zeros(0, dtype=np.float32),
        outflow_m3=np.zeros(0, dtype=np.float32),
        max_storage_m3=np.zeros(0, dtype=np.float32),
        inertial_retention_k=np.zeros(0, dtype=np.int32),
        inertial_retention_id=np.zeros(0, dtype=np.int32),
        controlled_retention=np.zeros(0, dtype=bool),
        activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        release_threshold_factor=np.float32(0.85),
    )


@pytest.mark.parametrize("evaporation_rate_m3_s", [0.0, 0.5])
def test_losses_cannot_exceed_stored_water(evaporation_rate_m3_s: float) -> None:
    """Limit evaporation and seepage to storage when no new water enters.

    Args:
        evaporation_rate_m3_s: Potential evaporation rate (cubic meters/second).

    Returns:
        None.

    Raises:
        AssertionError: If reported fluxes exceed the initial 100 cubic meters.
    """  # noqa: DOC202, DOC502 - Document test outcomes and propagated assertions.
    result: tuple = update_node_kinematic(
        inflow_m3_s=np.float32(0),
        previous_discharge_m3_s=np.float32(1),
        sideflow_m3_s=np.float32(0),
        evaporation_m3_s=np.float32(evaporation_rate_m3_s),
        transmission_loss_m3_s=np.float32(0.5),
        river_storage_alpha=np.float32(1),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(3600),
        river_length_m=np.float32(100),
    )
    outgoing_volume_m3: float = float(sum(result)) * 3600
    assert outgoing_volume_m3 <= 100 + 1e-5
    assert outgoing_volume_m3 == pytest.approx(100, abs=1e-5)


@pytest.mark.parametrize("sideflow_volume_m3", [0.0, -5.0, -20.0])
def test_reach_uses_actual_storage_without_invented_losses(
    no_retention: RetentionCoupling, sideflow_volume_m3: float
) -> None:
    """Use saved storage and account only unmet abstraction as a deficit.

    Args:
        no_retention: Empty retention-basin state.
        sideflow_volume_m3: Signed lateral volume; negative values are abstraction (m3).

    Returns:
        None.

    Raises:
        AssertionError: If seepage consumes unavailable water or abstraction is misreported.
    """  # noqa: DOC202, DOC502 - Document test outcomes and propagated assertions.
    storage_m3: np.ndarray = np.array([10.0], dtype=np.float64)
    discharge_m3_s: np.ndarray = np.zeros(1, dtype=np.float32)
    evaporation_m3: np.ndarray = np.zeros(1, dtype=np.float32)
    seepage_m3: np.ndarray = np.zeros(1, dtype=np.float32)
    deficit_m3: np.ndarray = np.zeros(1, dtype=np.float32)
    _solve_kinematic_reach(
        reach_idx=0,
        upstream_inflow_m3_s=np.float32(0),
        node_sideflow=np.float32(sideflow_volume_m3),
        dt_f32=np.float32(3600),
        inv_dt_f32=np.float32(1 / 3600),
        previous_discharge=np.float32(1),
        evaporation_vol_m3=np.float32(0),
        transmission_loss_vol_m3=np.float32(1800),
        river_storage_alpha=np.float32(1),
        river_storage_beta=np.float32(0.6),
        river_length=np.float32(100),
        river_storage_m3=storage_m3,
        updated_discharge_m3_s=discharge_m3_s,
        actual_evaporation_m3=evaporation_m3,
        actual_transmission_loss_m3=seepage_m3,
        over_abstraction_m3=deficit_m3,
        ret_id=-1,
        ret=no_retention,
    )
    expected_deficit_m3: float = max(0, -sideflow_volume_m3 - 10)
    expected_seepage_m3: float = max(0, 10 + sideflow_volume_m3)
    assert float(deficit_m3[0]) == pytest.approx(expected_deficit_m3, abs=1e-5)
    assert float(seepage_m3[0]) == pytest.approx(expected_seepage_m3, abs=1e-5)
    assert float(storage_m3[0]) >= 0
    assert float(
        storage_m3[0] + evaporation_m3[0] + seepage_m3[0] + discharge_m3_s[0] * 3600
    ) == pytest.approx(10 + sideflow_volume_m3 + expected_deficit_m3, abs=1e-5)
