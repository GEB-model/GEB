"""Tests for hydrological routing."""

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import numpy.typing as npt
import pytest

from geb.hydrology.routing.kinematic import update_node_kinematic


def test_update_node_kinematic_1() -> None:
    """Test the update_node_kinematic function with known inputs and outputs.

    Test adopted from PCRaster implementation.
    """
    deltaX: int = 10
    Q_new, evaporation_m3_s, actual_trans_m3_s = update_node_kinematic(
        inflow_m3_s=np.float32(0.000201343),
        previous_discharge_m3_s=np.float32(0.000115866),
        sideflow_m3_s=np.float32(-0.000290263 * deltaX),
        evaporation_m3_s=np.float32(0.0),
        transmission_loss_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.73684),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(15),
        river_length_m=np.float32(deltaX),
        epsilon=np.float32(1e-12),
        stream_width_m=np.float32(0.0),
        bankfull_depth_m=np.float32(np.inf),
        floodplain_width_m=np.float32(0.0),
        floodplain_conveyance=np.float32(0.0),
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
    Q_new, evaporation_m3_s, actual_trans_m3_s = update_node_kinematic(
        inflow_m3_s=np.float32(0),
        previous_discharge_m3_s=np.float32(1.11659e-07),
        sideflow_m3_s=np.float32(-1.32678e-05 * deltaX),
        evaporation_m3_s=np.float32(0.0),
        transmission_loss_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.6808),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(15),
        river_length_m=np.float32(deltaX),
        epsilon=np.float32(1e-12),
        stream_width_m=np.float32(0.0),
        bankfull_depth_m=np.float32(np.inf),
        floodplain_width_m=np.float32(0.0),
        floodplain_conveyance=np.float32(0.0),
    )
    assert math.isclose(Q_new, 1e-30, abs_tol=1e-12)


def test_update_node_kinematic_no_flow() -> None:
    """Test kinematic wave update with zero flow conditions.

    Verifies that when all inflows are zero, the discharge
    is set to the minimum value (1e-30) to avoid numerical issues.
    """
    Q_new, evaporation_m3_s, actual_trans_m3_s = update_node_kinematic(
        inflow_m3_s=np.float32(0),
        previous_discharge_m3_s=np.float32(0),
        sideflow_m3_s=np.float32(0),
        evaporation_m3_s=np.float32(0.0),
        transmission_loss_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.6808),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(15),
        river_length_m=np.float32(10),
        epsilon=np.float32(1e-12),
        stream_width_m=np.float32(0.0),
        bankfull_depth_m=np.float32(np.inf),
        floodplain_width_m=np.float32(0.0),
        floodplain_conveyance=np.float32(0.0),
    )
    assert math.isclose(Q_new, 1e-30, abs_tol=1e-12)


def test_update_node_kinematic_evaporation_limiter() -> None:
    """Test evaporation limiter behavior in kinematic routing."""
    # Case 1: Evaporation exceeds available flow -> capped at available flow.
    # Available flow = 0.5 * (1.0 + 1.0) + 0.5 = 1.5 m3/s.
    q_new_1, actual_evap_1, actual_trans_1 = update_node_kinematic(
        inflow_m3_s=np.float32(1.0),
        previous_discharge_m3_s=np.float32(1.0),
        sideflow_m3_s=np.float32(0.5),
        evaporation_m3_s=np.float32(5.0),
        transmission_loss_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.7),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(60.0),
        river_length_m=np.float32(1000.0),
        stream_width_m=np.float32(0.0),
        bankfull_depth_m=np.float32(np.inf),
        floodplain_width_m=np.float32(0.0),
        floodplain_conveyance=np.float32(0.0),
    )
    assert math.isclose(actual_evap_1, 1.5, rel_tol=1e-5)

    # Case 2: Evaporation is less than available flow -> actual evaporation equals potential evaporation.
    q_new_2, actual_evap_2, actual_trans_2 = update_node_kinematic(
        inflow_m3_s=np.float32(1.0),
        previous_discharge_m3_s=np.float32(1.0),
        sideflow_m3_s=np.float32(0.5),
        evaporation_m3_s=np.float32(0.3),
        transmission_loss_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.7),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(60.0),
        river_length_m=np.float32(1000.0),
        stream_width_m=np.float32(0.0),
        bankfull_depth_m=np.float32(np.inf),
        floodplain_width_m=np.float32(0.0),
        floodplain_conveyance=np.float32(0.0),
    )
    assert math.isclose(actual_evap_2, 0.3, rel_tol=1e-5)

    # Case 3: Negative sideflow (abstraction) does not contribute to available flow for evaporation.
    # Available flow = 0.5 * (1.0 + 1.0) + max(-0.5, 0.0) = 1.0 m3/s.
    q_new_3, actual_evap_3, actual_trans_3 = update_node_kinematic(
        inflow_m3_s=np.float32(1.0),
        previous_discharge_m3_s=np.float32(1.0),
        sideflow_m3_s=np.float32(-0.5),
        evaporation_m3_s=np.float32(3.0),
        transmission_loss_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(1.7),
        river_storage_beta=np.float32(0.6),
        timestep_s=np.float32(60.0),
        river_length_m=np.float32(1000.0),
        stream_width_m=np.float32(0.0),
        bankfull_depth_m=np.float32(np.inf),
        floodplain_width_m=np.float32(0.0),
        floodplain_conveyance=np.float32(0.0),
    )
    assert math.isclose(actual_evap_3, 1.0, rel_tol=1e-5)


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

    Q_new, _, _ = update_node_kinematic(
        inflow_m3_s=np.float32(Qin),
        previous_discharge_m3_s=np.float32(Qold),
        sideflow_m3_s=np.float32(Qside),
        evaporation_m3_s=np.float32(0.0),
        transmission_loss_m3_s=np.float32(0.0),
        river_storage_alpha=np.float32(alpha),
        river_storage_beta=np.float32(beta),
        timestep_s=np.float32(deltaT),
        river_length_m=np.float32(deltaX),
        epsilon=epsilon,
        stream_width_m=np.float32(0.0),
        bankfull_depth_m=np.float32(np.inf),
        floodplain_width_m=np.float32(0.0),
        floodplain_conveyance=np.float32(0.0),
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
    from geb.hydrology.routing.kinematic import (
        KINEMATIC_PARALLEL_THRESHOLD,
        _run_kinematic_step,
        _run_kinematic_step_parallel,
        _run_kinematic_step_serial,
    )
    from geb.hydrology.routing.topology import (
        KinematicTopology,
        RetentionCoupling,
        WaterbodyCoupling,
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
            stream_width=np.zeros_like(
                np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32)
            ),
            bankfull_depth=np.full_like(
                np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32), np.inf
            ),
            floodplain_width=np.zeros_like(
                np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32)
            ),
            floodplain_conveyance=np.zeros_like(
                np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32)
            ),
            river_length=np.array([1000.0, 1200.0, 1500.0, 1100.0], dtype=np.float32),
            retention_node_id=np.array([0, -1, -1, -1], dtype=np.int32),
        )
        wb = WaterbodyCoupling(
            storage_m3=np.zeros(0, dtype=np.float64),
            prescribed_outflow_m3=np.zeros(0, dtype=np.float32),
            inflow_m3=np.zeros(0, dtype=np.float32),
            outflow_m3=np.zeros(0, dtype=np.float32),
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
            np.array([10.0, 20.0, 5.0, 15.0], dtype=np.float32),
            np.zeros(n_kinematic, dtype=np.float32),
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

    # updated_discharge_m3_s (index 10)
    np.testing.assert_allclose(
        state_serial[10], state_parallel[10], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(state_serial[10], state_raw[10], rtol=1e-5, atol=1e-5)

    # river_storage_m3 (index 3)
    np.testing.assert_allclose(state_serial[3], state_parallel[3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_serial[3], state_raw[3], rtol=1e-5, atol=1e-5)

    # actual_evaporation_m3 (index 8)
    np.testing.assert_allclose(state_serial[8], state_parallel[8], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_serial[8], state_raw[8], rtol=1e-5, atol=1e-5)

    # actual_transmission_loss_m3 (index 9)
    np.testing.assert_allclose(state_serial[9], state_parallel[9], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state_serial[9], state_raw[9], rtol=1e-5, atol=1e-5)

    # retention_storage_m3 (inside ret coupling at index 13)
    np.testing.assert_allclose(
        state_serial[13].storage_m3, state_parallel[13].storage_m3, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        state_serial[13].storage_m3, state_raw[13].storage_m3, rtol=1e-5, atol=1e-5
    )


def _single_reach_kinematic_state(
    previous_discharge: float, storage: float, sideflow: float
) -> list[Any]:
    """Build the arguments of _run_kinematic_step for one isolated kinematic reach.

    Returns:
        Positional arguments for the kinematic step kernel.
    """
    from geb.hydrology.routing.topology import (
        KinematicTopology,
        RetentionCoupling,
        WaterbodyCoupling,
    )

    kin = KinematicTopology(
        n_kinematic=1,
        n_kin_headwater=1,
        kin_ds_reach=np.array([-1], dtype=np.int32),
        kin_inflow_m3_s=np.zeros(1, dtype=np.float32),
        kin_ds_waterbody_id=np.array([-1], dtype=np.int32),
        wb_to_kin_target_reach=np.zeros(0, dtype=np.int32),
        wb_to_kin_wb_id=np.zeros(0, dtype=np.int32),
        kin_wb_inflow_m3=np.zeros(1, dtype=np.float32),
        inertial_up_kin_offsets=np.zeros(1, dtype=np.int32),
        inertial_up_kin_indices=np.zeros(0, dtype=np.int32),
        inertial_up_kin_reach_idx=np.zeros(0, dtype=np.int32),
        river_storage_alpha=np.array([0.5], dtype=np.float32),
        river_storage_beta=np.array([0.6], dtype=np.float32),
        stream_width=np.zeros_like(np.array([0.6], dtype=np.float32)),
        bankfull_depth=np.full_like(np.array([0.6], dtype=np.float32), np.inf),
        floodplain_width=np.zeros_like(np.array([0.6], dtype=np.float32)),
        floodplain_conveyance=np.zeros_like(np.array([0.6], dtype=np.float32)),
        river_length=np.array([1000.0], dtype=np.float32),
        retention_node_id=np.array([-1], dtype=np.int32),
    )
    wb = WaterbodyCoupling(
        storage_m3=np.zeros(0, dtype=np.float64),
        prescribed_outflow_m3=np.zeros(0, dtype=np.float32),
        inflow_m3=np.zeros(0, dtype=np.float32),
        outflow_m3=np.zeros(0, dtype=np.float32),
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
        storage_m3=np.zeros(0, dtype=np.float32),
        inflow_m3=np.zeros(0, dtype=np.float32),
        outflow_m3=np.zeros(0, dtype=np.float32),
        max_storage_m3=np.zeros(0, dtype=np.float32),
        inertial_retention_k=np.zeros(0, dtype=np.int32),
        inertial_retention_id=np.zeros(0, dtype=np.int32),
        controlled_retention=np.zeros(0, dtype=bool),
        activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        release_threshold_factor=np.float32(0.2),
    )
    return [
        np.float32(3600.0),
        np.float32(1.0 / 3600.0),
        np.array([previous_discharge], dtype=np.float32),
        np.array([storage], dtype=np.float64),
        np.array([sideflow], dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        kin,
        wb,
        ret,
    ]


def test_kinematic_outflow_limited_to_stored_water() -> None:
    """Kinematic outflow beyond the stored water is limited, not booked as over-abstraction."""
    from geb.hydrology.routing.kinematic import _run_kinematic_step_serial

    # Storage relation says much water (Q_prev = 10 m3/s), tracked storage is 100 m3.
    state = _single_reach_kinematic_state(
        previous_discharge=10.0, storage=100.0, sideflow=0.0
    )
    _run_kinematic_step_serial(*state)
    storage, over_abstraction, discharge = state[3], state[7], state[10]
    assert over_abstraction[0] == 0.0
    assert storage[0] >= 0.0
    # Mass balance: everything that left is the water that was there.
    np.testing.assert_allclose(discharge[0] * 3600.0 + storage[0], 100.0, rtol=1e-5)


def test_kinematic_over_abstraction_when_abstraction_exceeds_storage() -> None:
    """Abstraction (negative sideflow) beyond the stored water is over-abstraction."""
    from geb.hydrology.routing.kinematic import _run_kinematic_step_serial

    state = _single_reach_kinematic_state(
        previous_discharge=0.0, storage=100.0, sideflow=-150.0
    )
    _run_kinematic_step_serial(*state)
    storage, over_abstraction, discharge = state[3], state[7], state[10]
    np.testing.assert_allclose(over_abstraction[0], 50.0, rtol=1e-5)
    assert storage[0] == 0.0
    assert discharge[0] == 0.0


def test_update_node_kinematic_valley_floor() -> None:
    """Above the banks a stream with a valley floor stores more water and flows slower.

    Steady state: with a constant inflow the outflow equals the inflow, whether the
    stream has a valley floor or not. A flood wave is damped by the valley floor, and
    a stream without one behaves exactly as before.
    """
    alpha, beta = np.float32(0.6), np.float32(0.6)
    width, bankfull_depth = np.float32(2.0), np.float32(0.3)
    bankfull_discharge = float((width * bankfull_depth / alpha) ** (1 / beta))
    common = dict(
        sideflow_m3_s=np.float32(0.0),
        evaporation_m3_s=np.float32(0.0),
        transmission_loss_m3_s=np.float32(0.0),
        river_storage_alpha=alpha,
        river_storage_beta=beta,
        timestep_s=np.float32(3600.0),
        river_length_m=np.float32(700.0),
    )
    valley = dict(
        stream_width_m=width,
        bankfull_depth_m=bankfull_depth,
        floodplain_width_m=np.float32(40.0),
        floodplain_conveyance=np.float32(np.sqrt(0.01) * 40.0 / 0.06),
    )
    # a channel without a stream width, banks or valley floor
    channel_only = dict(
        stream_width_m=np.float32(0.0),
        bankfull_depth_m=np.float32(np.inf),
        floodplain_width_m=np.float32(0.0),
        floodplain_conveyance=np.float32(0.0),
    )

    def route(inflows: list[float], **kwargs: np.float32) -> list[float]:
        q = np.float32(inflows[0])
        out = []
        for inflow in inflows:
            q, _, _ = update_node_kinematic(
                inflow_m3_s=np.float32(inflow),
                previous_discharge_m3_s=q,
                **common,
                **kwargs,
            )
            out.append(float(q))
        return out

    # below the banks the valley floor plays no part
    low = [0.5 * bankfull_discharge] * 3 + [0.8 * bankfull_discharge] * 5
    np.testing.assert_allclose(
        route(low, **valley), route(low, **channel_only), rtol=1e-5
    )

    # steady flow above the banks: outflow equals inflow
    steady = route([4.0 * bankfull_discharge] * 200, **valley)
    assert steady[-1] == pytest.approx(4.0 * bankfull_discharge, rel=1e-3)

    # a flood wave: the valley floor lowers and delays the peak, and no water is lost
    wave = (
        [0.5 * bankfull_discharge] * 3
        + [6.0 * bankfull_discharge] * 2
        + [0.5 * bankfull_discharge] * 40
    )
    with_valley = route(wave, **valley)
    without = route(wave, **channel_only)
    assert max(with_valley) < max(without)
    assert int(np.argmax(with_valley)) >= int(np.argmax(without))
    assert with_valley[-1] == pytest.approx(0.5 * bankfull_discharge, rel=2e-2)
    assert all(q >= 0.0 for q in with_valley)
