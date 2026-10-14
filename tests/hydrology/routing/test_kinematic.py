"""Tests for hydrological routing."""

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import numpy.typing as npt
import pytest

from geb.hydrology.routing import (
    get_channel_ratio,
)
from geb.hydrology.routing.kinematic import update_node_kinematic


def test_update_node_kinematic_1() -> None:
    """Test the update_node_kinematic function with known inputs and outputs.

    Test adopted from PCRaster implementation.
    """
    deltaX: int = 10
    Q_new, evaporation_m3_s = update_node_kinematic(
        inflow_m3_s=np.float32(0.000201343),
        previous_discharge_m3_s=np.float32(0.000115866),
        sideflow_m3_s=np.float32(-0.000290263 * deltaX),
        evaporation_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.73684),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(15),
        river_length_m=np.float32(deltaX),
        epsilon=np.float32(1e-12),
    )
    Q_check = 0.000031450866300937
    assert math.isclose(Q_new, Q_check, rel_tol=1e-5)


def test_update_node_kinematic_2() -> None:
    """Test the update_node_kinematic function with negative sideflow.

    In this function, the sideflow is so strongly negative that the discharge
    should be set to the minimum value by update_node_kinematic (1e-30).
    The 1e-30 is to avoid numerical issues.

    Test adopted from PCRaster implementation.
    """
    deltaX: int = 10
    Q_new, evaporation_m3_s = update_node_kinematic(
        inflow_m3_s=np.float32(0),
        previous_discharge_m3_s=np.float32(1.11659e-07),
        sideflow_m3_s=np.float32(-1.32678e-05 * deltaX),
        evaporation_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.6808),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(15),
        river_length_m=np.float32(deltaX),
        epsilon=np.float32(1e-12),
    )
    assert math.isclose(Q_new, 1e-30, abs_tol=1e-12)


def test_update_node_kinematic_no_flow() -> None:
    """Test kinematic wave update with zero flow conditions.

    Verifies that when all inflows are zero, the discharge
    is set to the minimum value (1e-30) to avoid numerical issues.
    """
    Q_new, evaporation_m3_s = update_node_kinematic(
        inflow_m3_s=np.float32(0),
        previous_discharge_m3_s=np.float32(0),
        sideflow_m3_s=np.float32(0),
        evaporation_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.6808),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(15),
        river_length_m=np.float32(10),
        epsilon=np.float32(1e-12),
    )
    assert math.isclose(Q_new, 1e-30, abs_tol=1e-12)


def test_update_node_kinematic_evaporation_limiter() -> None:
    """Test evaporation limiter behavior in kinematic routing."""
    # Case 1: Evaporation exceeds available flow -> capped at available flow.
    # Available flow = 0.5 * (1.0 + 1.0) + 0.5 = 1.5 m3/s.
    q_new_1, actual_evap_1 = update_node_kinematic(
        inflow_m3_s=np.float32(1.0),
        previous_discharge_m3_s=np.float32(1.0),
        sideflow_m3_s=np.float32(0.5),
        evaporation_m3_s=np.float32(5.0),
        river_storage_alpha=np.float32(1.7),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(60.0),
        river_length_m=np.float32(1000.0),
    )
    assert math.isclose(actual_evap_1, 1.5, rel_tol=1e-5)

    # Case 2: Evaporation is less than available flow -> actual evaporation equals potential evaporation.
    q_new_2, actual_evap_2 = update_node_kinematic(
        inflow_m3_s=np.float32(1.0),
        previous_discharge_m3_s=np.float32(1.0),
        sideflow_m3_s=np.float32(0.5),
        evaporation_m3_s=np.float32(0.3),
        river_storage_alpha=np.float32(1.7),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(60.0),
        river_length_m=np.float32(1000.0),
    )
    assert math.isclose(actual_evap_2, 0.3, rel_tol=1e-5)

    # Case 3: Negative sideflow (abstraction) does not contribute to available flow for evaporation.
    # Available flow = 0.5 * (1.0 + 1.0) + max(-0.5, 0.0) = 1.0 m3/s.
    q_new_3, actual_evap_3 = update_node_kinematic(
        inflow_m3_s=np.float32(1.0),
        previous_discharge_m3_s=np.float32(1.0),
        sideflow_m3_s=np.float32(-0.5),
        evaporation_m3_s=np.float32(3.0),
        river_storage_alpha=np.float32(1.7),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(60.0),
        river_length_m=np.float32(1000.0),
    )
    assert math.isclose(actual_evap_3, 1.0, rel_tol=1e-5)


def test_get_channel_ratio() -> None:
    """Test calculation of channel ratio for routing.

    Verifies that the channel ratio is correctly computed
    based on channel width and length parameters.
    """
    river_width = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    river_length = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.float32)
    cell_area = np.full_like(river_width, 10000, dtype=np.float32)

    channel_ratio = get_channel_ratio(
        river_width=river_width, river_length=river_length, cell_area=cell_area
    )

    assert np.allclose(channel_ratio, np.array([0.1, 0.4, 0.9, 1.0, 1.0]))


def test_update_node_kinematic_residual() -> None:
    """Test if update_node_kinematic converges to a solution with small residual.

    This test checks if the returned Q_new actually satisfies the kinematic wave equation
    within the specified epsilon tolerance.
    """
    deltaX: float = 100.0
    deltaT: float = 3600.0
    Qin: float = 10.0
    Qold: float = 8.0
    Qside: float = 1.0
    alpha: float = 1.5
    beta: float = 0.6
    epsilon: np.float32 = np.float32(1e-6)

    Q_new, _ = update_node_kinematic(
        inflow_m3_s=np.float32(Qin),
        previous_discharge_m3_s=np.float32(Qold),
        sideflow_m3_s=np.float32(Qside),
        evaporation_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(alpha),
        river_storage_beta=np.float32(beta),
        timestep_s=np.float32(deltaT),
        river_length_m=np.float32(deltaX),
        epsilon=epsilon,
    )

    deltaTX = np.float32(deltaT) / np.float32(deltaX)
    q = np.float32(Qside) / np.float32(deltaX)
    C = (
        deltaTX * np.float32(Qin)
        + np.float32(alpha) * np.float32(Qold) ** np.float32(beta)
        + np.float32(deltaT) * q
    )
    residual = deltaTX * Q_new + np.float32(alpha) * Q_new ** np.float32(beta) - C

    assert abs(residual) <= epsilon


@pytest.fixture
def ldd() -> npt.NDArray[np.uint8]:
    """Fixture providing a local drainage direction (ldd) array for routing tests.

    Returns:
        A 4x4 array with ldd values in PCRaster format.
    """
    return np.array(
        [
            [6, 5, 255, 2],
            [6, 8, 7, 2],
            [6, 8, 6, 5],
            [9, 8, 4, 4],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def mask() -> npt.NDArray[np.bool_]:
    """Fixture providing a mask array for routing tests.

    Returns:
        A 4x4 boolean array indicating valid cells.
    """
    return np.array(
        [
            [True, True, False, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
        dtype=bool,
    )


@pytest.fixture
def Q_initial() -> npt.NDArray[np.float32]:
    """Fixture providing a sample discharge array for testing.

    Returns:
        A 4x4 array with discharge values.
    """
    return np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )


def test_serial_and_parallel_kinematic_kernels_equivalence() -> None:
    """Test numerical equivalence and dynamic dispatch of serial and parallel kinematic wave kernels.

    Verifies that the serial and parallel compiled variants of the kinematic wave routing
    step produce numerically identical results, and that dispatching logic operates correctly.
    """
    from geb.hydrology.routing.inertial_substeps import (
        KinematicTopology,
        RetentionCoupling,
        WaterbodyCoupling,
    )
    from geb.hydrology.routing.local_inertial import (
        KINEMATIC_PARALLEL_THRESHOLD,
        _run_kinematic_step,
        _run_kinematic_step_parallel,
        _run_kinematic_step_serial,
    )

    assert KINEMATIC_PARALLEL_THRESHOLD == 5000

    # 4 reaches: reaches 0 and 1 are headwaters (n_kin_headwater=2), flowing into 2, which flows into 3
    n_kinematic: int = 4
    n_kin_headwater: int = 2
    dt_f32: np.float32 = np.float32(3600.0)
    inv_dt_f32: np.float32 = np.float32(1.0 / 3600.0)

    kin_ds_reach: npt.NDArray[np.int32] = np.array([2, 2, 3, -1], dtype=np.int32)
    kin_ds_wb_id: npt.NDArray[np.int32] = np.array([-1, -1, -1, -1], dtype=np.int32)

    def build_test_state() -> list[Any]:
        kin = KinematicTopology(
            n_kinematic=n_kinematic,
            n_kin_headwater=n_kin_headwater,
            kin_ds_reach=kin_ds_reach.copy(),
            kin_inflow_m3_s=np.zeros(n_kinematic, dtype=np.float32),
            kin_ds_waterbody_id=kin_ds_wb_id.copy(),
            wb_to_kin_target_reach=np.zeros(0, dtype=np.int32),
            wb_to_kin_wb_id=np.zeros(0, dtype=np.int32),
            kin_wb_inflow_m3=np.zeros(n_kinematic, dtype=np.float32),
            inertial_up_kin_offsets=np.zeros(1, dtype=np.int32),
            inertial_up_kin_indices=np.zeros(0, dtype=np.int32),
            inertial_up_kin_reach_idx=np.zeros(0, dtype=np.int32),
            river_storage_alpha=np.array([0.5, 0.6, 0.55, 0.52], dtype=np.float32),
            river_storage_beta=np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32),
            river_length=np.array([1000.0, 1200.0, 1500.0, 1100.0], dtype=np.float32),
            retention_node_id=np.array([0, -1, -1, -1], dtype=np.int32),
        )
        wb = WaterbodyCoupling(
            storage_m3=np.zeros(0, dtype=np.float64),
            prescribed_outflow_m3=np.zeros(0, dtype=np.float32),
            inflow_m3=np.zeros(0, dtype=np.float32),
            lake_area=np.zeros(0, dtype=np.float32),
            lake_factor=np.zeros(0, dtype=np.float32),
            outflow_height=np.zeros(0, dtype=np.float32),
            outflow_bed_elev=np.zeros(0, dtype=np.float32),
            inertial_wb_ds_k=np.zeros(0, dtype=np.int32),
            inertial_wb_ds_id=np.zeros(0, dtype=np.int32),
            lake_outflow_target_k=np.zeros(0, dtype=np.int32),
            lake_outflow_wb_id=np.zeros(0, dtype=np.int32),
            wb_to_wb_src_wb=np.zeros(0, dtype=np.int32),
            wb_to_wb_tgt_wb=np.zeros(0, dtype=np.int32),
            wb_terminal_wb_ids=np.zeros(0, dtype=np.int32),
        )
        ret = RetentionCoupling(
            storage_m3=np.array([100.0], dtype=np.float32),
            inflow_m3=np.zeros(1, dtype=np.float32),
            outflow_m3=np.zeros(1, dtype=np.float32),
            max_storage_m3=np.array([5000.0], dtype=np.float32),
            inertial_retention_k=np.zeros(0, dtype=np.int32),
            inertial_retention_id=np.zeros(0, dtype=np.int32),
            controlled_retention=np.array([True], dtype=bool),
            activation_threshold_m3_s=np.array([1.5], dtype=np.float32),
            release_threshold_factor=np.float32(0.2),
        )
        return [
            dt_f32,
            inv_dt_f32,
            np.array([2.0, 3.0, 5.0, 4.0], dtype=np.float32),
            np.array([5000.0, 6000.0, 10000.0, 8000.0], dtype=np.float64),
            np.array([1000.0, 1500.0, 500.0, 200.0], dtype=np.float32),
            np.array([50.0, 60.0, 30.0, 20.0], dtype=np.float32),
            np.zeros(n_kinematic, dtype=np.float32),
            np.zeros(n_kinematic, dtype=np.float32),
            np.zeros(n_kinematic, dtype=np.float32),
            kin,
            wb,
            ret,
        ]

    state_serial: list[Any] = build_test_state()
    state_parallel: list[Any] = build_test_state()
    state_raw: list[Any] = build_test_state()

    _run_kinematic_step_serial(*state_serial)
    _run_kinematic_step_parallel(*state_parallel)
    _run_kinematic_step(*state_raw)

    # updated_discharge_m3_s (index 8)
    np.testing.assert_allclose(state_serial[8], state_parallel[8], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_serial[8], state_raw[8], rtol=1e-5, atol=1e-5)

    # river_storage_m3 (index 3)
    np.testing.assert_allclose(state_serial[3], state_parallel[3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_serial[3], state_raw[3], rtol=1e-5, atol=1e-5)

    # actual_evaporation_m3 (index 7)
    np.testing.assert_allclose(state_serial[7], state_parallel[7], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_serial[7], state_raw[7], rtol=1e-5, atol=1e-5)

    # retention_storage_m3 (inside ret coupling at index 11)
    np.testing.assert_allclose(
        state_serial[11].storage_m3, state_parallel[11].storage_m3, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        state_serial[11].storage_m3, state_raw[11].storage_m3, rtol=1e-5, atol=1e-5
    )
