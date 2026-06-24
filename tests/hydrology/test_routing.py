"""Tests for hydrological routing functions in GEB.

Tests for accuflux routing are quite nice and complete. Tests for kinematic wave routing are
more limited, as this is a more complex function. More tests should be added in the future.

"""

import math

import numpy as np
import numpy.typing as npt
import pyflwdir
import pytest

from geb.hydrology.routing import (
    Accuflux,
    KinematicWave,
    create_river_network,
    get_channel_ratio,
    update_node_kinematic,
)


def test_update_node_kinematic_1() -> None:
    """Test the update_node_kinematic function with known inputs and outputs.

    Test adopted from PCRaster implementation.
    """
    deltaX: int = 10
    Q_new, evaporation_m3_s = update_node_kinematic(
        Qin=0.000201343,
        Qold=0.000115866,
        Qside=-0.000290263 * deltaX,
        evaporation_m3_s=0.0,
        alpha=1.73684,
        beta=0.6,
        deltaT=15,
        deltaX=deltaX,
        epsilon=np.float32(1e-12),
    )
    Q_check = 0.000031450866300937
    assert math.isclose(Q_new, Q_check, abs_tol=1e-12)


def test_update_node_kinematic_2() -> None:
    """Test the update_node_kinematic function with negative sideflow.

    In this function, the sideflow is so strongly negative that the discharge
    should be set to the minimum value by update_node_kinematic (1e-30).
    The 1e-30 is to avoid numerical issues.

    Test adopted from PCRaster implementation.
    """
    deltaX: int = 10
    Q_new, evaporation_m3_s = update_node_kinematic(
        Qin=0,
        Qold=1.11659e-07,
        Qside=-1.32678e-05 * deltaX,
        evaporation_m3_s=0.0,
        alpha=1.6808,
        beta=0.6,
        deltaT=15,
        deltaX=deltaX,
        epsilon=np.float32(1e-12),
    )
    assert math.isclose(Q_new, 1e-30, abs_tol=1e-12)


def test_update_node_kinematic_no_flow() -> None:
    """Test kinematic wave update with zero flow conditions.

    Verifies that when all inflows are zero, the discharge
    is set to the minimum value (1e-30) to avoid numerical issues.
    """
    Q_new, evaporation_m3_s = update_node_kinematic(
        Qin=0,
        Qold=0,
        Qside=0,
        evaporation_m3_s=0.0,
        alpha=1.6808,
        beta=0.6,
        deltaT=15,
        deltaX=10,
        epsilon=np.float32(1e-12),
    )
    assert math.isclose(Q_new, 1e-30, abs_tol=1e-12)


def test_get_channel_ratio() -> None:
    """Test calculation of channel ratio for routing.

    Verifies that the channel ratio is correctly computed
    based on channel width and length parameters.
    """
    river_width = np.array([1, 2, 3, 4, 5])
    river_length = np.array([1000, 2000, 3000, 4000, 5000])
    cell_area = np.full_like(river_width, 10000)

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
        Qin=np.float32(Qin),
        Qold=np.float32(Qold),
        Qside=np.float32(Qside),
        evaporation_m3_s=np.float32(0.0),
        alpha=np.float32(alpha),
        beta=np.float32(beta),
        deltaT=np.float32(deltaT),
        deltaX=np.float32(deltaX),
        epsilon=epsilon,
    )

    # Recompute C and the residual using float32 to match internal precision
    deltaTX = np.float32(deltaT) / np.float32(deltaX)
    q = np.float32(Qside) / np.float32(deltaX)
    C = (
        deltaTX * np.float32(Qin)
        + np.float32(alpha) * np.float32(Qold) ** np.float32(beta)
        + np.float32(deltaT) * q
    )
    residual = deltaTX * Q_new + np.float32(alpha) * Q_new ** np.float32(beta) - C

    assert abs(residual) <= epsilon, (
        f"Residual {abs(residual)} exceeds epsilon {epsilon}"
    )


@pytest.fixture
def ldd() -> npt.NDArray[np.uint8]:
    """Fixture providing a local drainage direction (ldd) array for routing tests.

    Returns:
        np.ndarray: A 4x4 array with ldd values in PCRaster format.

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
        np.ndarray: A 4x4 boolean array indicating valid cells.
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
        np.ndarray: A 4x4 array with discharge values.
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


def test_accuflux(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test accumulation flux calculation for routing.

    Verifies that the accuflux function correctly accumulates
    water fluxes through the routing network over time.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)

    # Empty retention arrays for "no retention" case
    retention_max_storage_m3 = np.ndarray(0, dtype=np.float32)
    retention_node_id = np.full_like(mask[mask], -1, dtype=np.int32)
    controlled_retention = np.array([], dtype=bool)
    retention_activation_threshold_controlled_m3_s = np.ndarray(0, dtype=np.float32)
    retention_activation_threshold_uncontrolled_m3_s = np.ndarray(0, dtype=np.float32)

    router: Accuflux = Accuflux(
        dt=1,
        river_network=river_network,
        river_length=np.ones_like(mask[mask], dtype=np.float32),
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_activation_threshold_controlled_m3_s=retention_activation_threshold_controlled_m3_s,
        retention_activation_threshold_uncontrolled_m3_s=retention_activation_threshold_uncontrolled_m3_s,
    )

    sideflow = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )[mask]

    retention_storage_m3 = np.ndarray(0, dtype=np.float32)

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
        retention_storage_m3_out,
        retention_inflow_m3,
        retention_outflow_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
        retention_storage_m3=retention_storage_m3,
        river_storage_alpha=np.zeros_like(sideflow, dtype=np.float32),
        river_storage_beta=np.zeros_like(sideflow, dtype=np.float32),
    )

    assert (
        Q_new
        == np.array(
            [
                [0, 3, 0, 0],
                [0, 2, 0, 1],
                [0, 3, 0, 2],
                [0, 1, 1, 0],
            ]
        )[mask]
    ).all()
    assert (waterbody_inflow_m3 == 0).all()
    assert outflow_at_pits_m3 == 2
    assert waterbody_storage_m3.size == 0
    assert retention_storage_m3_out.size == 0
    assert retention_inflow_m3.size == 0
    assert retention_outflow_m3.size == 0


def test_accuflux_with_retention_basins(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test Accuflux routing with retention basins.

    Verifies that Water is correctly stored in retention basins without exceeding maximum storage.
    Inflow into each retention basin is correctly recorded.
    Retention outflow is zero when storage limits are not exceeded.
    Downstream discharge is reduced by the amount stored in the basins, ensuring mass balance.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)

    sideflow = np.zeros(mask.sum(), dtype=np.float32)

    retention_raster = -1 * np.ones_like(Q_initial, dtype=np.int32)
    retention_raster[2, 1] = 0  # controlled basin (receives Q_new = 3)
    retention_raster[0, 1] = 1  # uncontrolled basin (receieves Q_new = 3)
    # flatten according to mask
    retention_node_id = retention_raster[mask]

    # --- set initial storage and max storage ---
    retention_max_storage_m3 = np.array([2, 2], dtype=np.float32)  # max storage
    controlled_retention = np.array([True, False])  # 1 basin are controlled
    retention_activation_threshold_controlled_m3_s = np.array(
        [2.0, 0.0], dtype=np.float32
    )  # activation thresholds
    retention_activation_threshold_uncontrolled_m3_s = np.array(
        [0.0, 1.0], dtype=np.float32
    )

    router: Accuflux = Accuflux(
        dt=1,
        river_network=river_network,
        river_length=np.ones_like(mask[mask], dtype=np.float32),
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_activation_threshold_controlled_m3_s=retention_activation_threshold_controlled_m3_s,
        retention_activation_threshold_uncontrolled_m3_s=retention_activation_threshold_uncontrolled_m3_s,
    )

    retention_storage_m3 = np.zeros(2, dtype=np.float32)

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
        retention_storage_m3_out,
        retention_inflow_m3,
        retention_outflow_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
        retention_storage_m3=retention_storage_m3,
        river_storage_alpha=np.zeros_like(sideflow, dtype=np.float32),
        river_storage_beta=np.zeros_like(sideflow, dtype=np.float32),
    )

    # make sure retention storage is smaller than max storage
    assert (retention_storage_m3_out <= retention_max_storage_m3).all()
    # make sure that water is actually flowing into any of the two basins
    assert retention_storage_m3_out.sum() > 0.0
    # make sure that water is flowing into both basins
    assert (retention_storage_m3_out > 0.0).all()
    # make sure that there is discharge in next timestep (Qnew)
    assert Q_new.sum() > 0.0
    # make sure that mass balance is maintained: total initial flow should equal total flow after routing (Qnew + retention storage + outflow at pits)
    total_initial_flow = Q_initial[mask].sum()
    total_flow_after = Q_new.sum() + retention_storage_m3_out.sum() + outflow_at_pits_m3

    np.testing.assert_almost_equal(total_initial_flow, total_flow_after, decimal=5)


def test_accuflux_with_longer_dt(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test accumulation flux with longer time step.

    Verifies that accuflux correctly handles longer time steps
    in the routing calculations.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)

    # Empty retention arrays for "no retention" case
    retention_max_storage_m3 = np.ndarray(0, dtype=np.float32)
    retention_node_id = np.full_like(mask[mask], -1, dtype=np.int32)
    controlled_retention = np.array([], dtype=bool)
    retention_activation_threshold_controlled_m3_s = np.ndarray(0, dtype=np.float32)
    retention_activation_threshold_uncontrolled_m3_s = np.ndarray(0, dtype=np.float32)

    router: Accuflux = Accuflux(
        dt=15,
        river_network=river_network,
        river_length=np.ones_like(mask[mask], dtype=np.float32),
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_activation_threshold_controlled_m3_s=retention_activation_threshold_controlled_m3_s,
        retention_activation_threshold_uncontrolled_m3_s=retention_activation_threshold_uncontrolled_m3_s,
    )

    sideflow = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )[mask]

    retention_storage_m3 = np.ndarray(0, dtype=np.float32)

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
        retention_storage_m3_out,
        retention_inflow_m3,
        retention_outflow_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
        retention_storage_m3=retention_storage_m3,
        river_storage_alpha=np.zeros_like(sideflow, dtype=np.float32),
        river_storage_beta=np.zeros_like(sideflow, dtype=np.float32),
    )

    assert (
        Q_new
        == np.array(
            [
                [0, 3, 0, 0],
                [0, 2, 0, 1],
                [0, 3, 0, 2],
                [0, 1, 1, 0],
            ]
        )[mask]
    ).all()
    assert (waterbody_inflow_m3 == 0).all()
    assert outflow_at_pits_m3 == 30
    assert waterbody_storage_m3.size == 0


def test_accuflux_with_sideflow(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test accumulation flux with side flow inputs.

    Verifies that accuflux correctly incorporates side flow
    contributions into the routing calculations.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)

    # Empty retention arrays for "no retention" case
    retention_max_storage_m3 = np.ndarray(0, dtype=np.float32)
    retention_node_id = np.full_like(mask[mask], -1, dtype=np.int32)
    controlled_retention = np.array([], dtype=bool)
    retention_activation_threshold_controlled_m3_s = np.ndarray(0, dtype=np.float32)
    retention_activation_threshold_uncontrolled_m3_s = np.ndarray(0, dtype=np.float32)

    router = Accuflux(
        dt=1,
        river_network=river_network,
        river_length=np.ones_like(mask[mask], dtype=np.float32),
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_activation_threshold_controlled_m3_s=retention_activation_threshold_controlled_m3_s,
        retention_activation_threshold_uncontrolled_m3_s=retention_activation_threshold_uncontrolled_m3_s,
    )

    sideflow = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    )[mask]

    retention_storage_m3 = np.ndarray(0, dtype=np.float32)

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
        retention_storage_m3_out,
        retention_inflow_m3,
        retention_outflow_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
        retention_storage_m3=retention_storage_m3,
        river_storage_alpha=np.zeros_like(sideflow, dtype=np.float32),
        river_storage_beta=np.zeros_like(sideflow, dtype=np.float32),
    )

    assert (
        Q_new
        == np.array(
            [
                [0, 3, 0, 0],
                [0, 2, 0, 4],
                [0, 3, 0, 4],
                [0, 1, 1, 0],
            ]
        )[mask]
    ).all()
    assert outflow_at_pits_m3 == 3  # 2 + 1 from the sideflow
    assert waterbody_storage_m3.size == 0
    assert (waterbody_inflow_m3 == 0).all()

    assert (
        Q_initial[mask].sum() + sideflow.sum() - outflow_at_pits_m3
        == Q_new.sum() + waterbody_storage_m3.sum()
    )


def test_accuflux_with_waterbodies(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test accumulation flux with water bodies.

    Verifies that accuflux correctly handles routing through
    water bodies like lakes and reservoirs.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)

    waterbody_id = np.array(
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, 0, -1, -1],
            [1, -1, -1, -1],
        ]
    )
    Q_initial[waterbody_id != -1] = np.nan

    # Empty retention arrays for "no retention" case
    retention_max_storage_m3 = np.ndarray(0, dtype=np.float32)
    retention_node_id = np.full_like(mask[mask], -1, dtype=np.int32)
    controlled_retention = np.array([], dtype=bool)
    retention_activation_threshold_controlled_m3_s = np.ndarray(0, dtype=np.float32)
    retention_activation_threshold_uncontrolled_m3_s = np.ndarray(0, dtype=np.float32)

    router: Accuflux = Accuflux(
        dt=1,
        river_network=river_network,
        river_length=np.ones_like(mask[mask], dtype=np.float32),
        is_waterbody_outflow=np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, True, False, False],
                [True, False, False, False],
            ]
        )[mask],
        waterbody_id=waterbody_id[mask],
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_activation_threshold_controlled_m3_s=retention_activation_threshold_controlled_m3_s,
        retention_activation_threshold_uncontrolled_m3_s=retention_activation_threshold_uncontrolled_m3_s,
    )

    sideflow = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )[mask]

    waterbody_storage_m3 = np.array([10, 5])
    outflow_per_waterbody_m3 = np.array([7, 2])

    waterbody_storage_m3_pre = waterbody_storage_m3.copy()

    retention_storage_m3 = np.ndarray(0, dtype=np.float32)

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
        retention_storage_m3_out,
        retention_inflow_m3,
        retention_outflow_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=waterbody_storage_m3,
        outflow_per_waterbody_m3=outflow_per_waterbody_m3,
        retention_storage_m3=retention_storage_m3,
        river_storage_alpha=np.zeros_like(sideflow, dtype=np.float32),
        river_storage_beta=np.zeros_like(sideflow, dtype=np.float32),
    )

    np.testing.assert_array_equal(
        Q_new,
        np.array(
            [
                [0, 3, 0, 0],
                [0, 8, 0, 4],
                [0, np.nan, 0, 4],
                [np.nan, 1, 1, 0],
            ]
        )[mask],
    )
    assert outflow_at_pits_m3 == 2
    assert (
        waterbody_inflow_m3[0] == 4
    )  # 2 from upstream reservoir, 1 from both other river inflows
    assert waterbody_inflow_m3[1] == 0  # 2 no upstream cells

    assert waterbody_storage_m3[0] == 7  # 10 - 7 + 2 + 1 + 1
    assert waterbody_storage_m3[1] == 3  # 5 - 2
    assert (
        np.nansum(Q_initial[mask])
        + waterbody_storage_m3_pre.sum()
        + sideflow.sum()
        - outflow_at_pits_m3
        == np.nansum(Q_new) + waterbody_storage_m3.sum()
    )


def test_kinematic(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test kinematic wave routing implementation.

    Verifies that the kinematic wave routing correctly
    simulates water flow through river networks.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    router: KinematicWave = KinematicWave(
        river_network=river_network,
        river_length=np.full_like(mask, np.float32(5.0), dtype=np.float32)[mask],
        dt=15,
        waterbody_id=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
        # retention arrays: full arrays for all river cells
        retention_max_storage_m3=np.zeros(mask.sum(), dtype=np.float32),
        retention_node_id=np.full(mask.sum(), -1, dtype=np.int32),
        controlled_retention=np.zeros(mask.sum(), dtype=bool),
        retention_activation_threshold_controlled_m3_s=np.zeros(
            mask.sum(), dtype=np.float32
        ),
        retention_activation_threshold_uncontrolled_m3_s=np.zeros(
            mask.sum(), dtype=np.float32
        ),
    )

    sideflow = np.array(
        [
            [0, 0, 0, 3],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )[mask]

    (
        Q_new,
        actual_evaporation_m3,
        over_abstraction_m3,
        waterbody_storage_m3,
        waterbody_inflow_m3,
        outflow_at_pits_m3,
        retention_storage_m3_out,
        retention_inflow_m3,
        retention_outflow_m3,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
        retention_storage_m3=np.zeros(mask.sum(), dtype=np.float32),
        river_storage_alpha=np.full_like(mask[mask], np.float32(1.0), dtype=np.float32),
        river_storage_beta=np.full_like(mask[mask], np.float32(0.6), dtype=np.float32),
    )

    assert Q_new.shape[0] == mask.sum()
    assert not np.isnan(Q_new).any()
    assert (Q_new >= 0.0).all()


def test_accuflux_inverse_ops(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test if Accuflux's total_storage and discharge_from_river_storage are inverses."""
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    dt = 3600
    river_length = np.full_like(mask[mask], 100.0, dtype=np.float32)
    waterbody_id = np.full_like(mask[mask], -1, dtype=np.int32)
    is_waterbody_outflow = np.zeros_like(mask[mask], dtype=bool)

    # Empty retention arrays for "no retention" case
    retention_max_storage_m3 = np.ndarray(0, dtype=np.float32)
    retention_node_id = np.full_like(mask[mask], -1, dtype=np.int32)
    controlled_retention = np.array([], dtype=bool)
    retention_activation_threshold_controlled_m3_s = np.ndarray(0, dtype=np.float32)
    retention_activation_threshold_uncontrolled_m3_s = np.ndarray(0, dtype=np.float32)

    router: Accuflux = Accuflux(
        dt,
        river_network,
        river_length,
        waterbody_id,
        is_waterbody_outflow,
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_activation_threshold_controlled_m3_s=retention_activation_threshold_controlled_m3_s,
        retention_activation_threshold_uncontrolled_m3_s=retention_activation_threshold_uncontrolled_m3_s,
    )

    # Use Q_initial as dummy discharge values (m3/s)
    Q = Q_initial[mask]
    alpha = np.ones_like(Q, dtype=np.float32)  # Unused for Accuflux
    beta = np.ones_like(Q, dtype=np.float32)  # Unused for Accuflux

    # Q -> Storage
    storage = router.get_total_storage(Q, alpha, beta)

    # Storage -> Q
    Q_inv = router.calculate_discharge_from_river_storage(
        storage, alpha, beta, river_length, waterbody_id
    )

    # Check if Q_inv matches Q (Accuflux should be exact as it's linear: S = Q * dt)
    np.testing.assert_allclose(Q, Q_inv, rtol=1e-5)


def test_kinematic_wave_inverse_ops(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test if KinematicWave's total_storage and discharge_from_river_storage are inverses."""
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    dt = 3600
    river_length = np.full_like(mask[mask], 100.0, dtype=np.float32)
    waterbody_id = np.full_like(mask[mask], -1, dtype=np.int32)
    is_waterbody_outflow = np.zeros_like(mask[mask], dtype=bool)
    retention_storage_m3 = np.zeros(mask.sum(), dtype=np.float32)
    retention_max_storage_m3 = np.zeros(mask.sum(), dtype=np.float32)
    retention_node_id = np.full_like(mask[mask], -1, dtype=np.int32)
    controlled_retention = np.zeros(mask.sum(), dtype=bool)
    retention_activation_threshold_controlled_m3_s = np.zeros(
        mask.sum(), dtype=np.float32
    )
    retention_activation_threshold_uncontrolled_m3_s = np.zeros(
        mask.sum(), dtype=np.float32
    )

    router: KinematicWave = KinematicWave(
        dt,
        river_network,
        river_length,
        waterbody_id,
        is_waterbody_outflow,
        retention_max_storage_m3,
        retention_node_id,
        controlled_retention,
        retention_activation_threshold_controlled_m3_s,
        retention_activation_threshold_uncontrolled_m3_s,
    )

    # Use Q_initial as dummy discharge values (m3/s)
    Q = Q_initial[mask]
    # Typical alpha, beta values for kinematic wave
    alpha = np.full_like(Q, 1.5, dtype=np.float32)
    beta = np.full_like(Q, 0.6, dtype=np.float32)

    # Q -> Storage
    storage = router.get_total_storage(Q, alpha, beta)

    # Storage -> Q
    Q_inv = router.calculate_discharge_from_river_storage(
        storage, alpha, beta, river_length, waterbody_id
    )

    # Check if Q_inv matches Q
    np.testing.assert_allclose(Q, Q_inv, rtol=1e-5)


def test_kinematic_sudden_flood_wave(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
) -> None:
    """Test kinematic wave routing with a sudden massive flood wave.

    Verifies that the routing remains stable and doesn't exceed the total inflow
    when transitioning from extreme dry to extreme wet conditions.
    """
    river_network: pyflwdir.FlwdirRaster = create_river_network(ldd, mask)
    dt = 3600
    router: KinematicWave = KinematicWave(
        dt=dt,
        river_network=river_network,
        river_length=np.full(mask.sum(), np.float32(100.0), dtype=np.float32),
        waterbody_id=np.full(mask.sum(), -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(mask.sum(), dtype=bool),
        retention_max_storage_m3=np.zeros(mask.sum(), dtype=np.float32),
        retention_node_id=np.full(mask.sum(), -1, dtype=np.int32),
        controlled_retention=np.zeros(mask.sum(), dtype=bool),
        retention_activation_threshold_controlled_m3_s=np.zeros(
            mask.sum(), dtype=np.float32
        ),
        retention_activation_threshold_uncontrolled_m3_s=np.zeros(
            mask.sum(), dtype=np.float32
        ),
    )

    # Initial state: extremely dry
    Q_prev_m3_s = np.full(mask.sum(), 1e-30, dtype=np.float32)

    # Use a headwater cell with a longer path (3,3) which is the last index in our mask
    # This path is: (3,3) -> (3,2) -> (3,1) -> (2,1) -> (1,1) -> (0,1, PIT)
    injection_node = mask.sum() - 1
    side_flow_m3_s = 10000.0

    total_volume_in_m3 = 0.0
    total_volume_out_m3 = 0.0

    sideflow_m3 = np.zeros(mask.sum(), dtype=np.float32)
    sideflow_m3[injection_node] = side_flow_m3_s * dt
    for i in range(10):  # Run for 10 time steps to observe propagation
        total_volume_in_m3 += sideflow_m3.sum()

        (
            Q_new,
            _,
            _,
            _,
            _,
            outflow_step_m3,
            _,
            _,
            _,
        ) = router.step(
            Q_prev_m3_s=Q_prev_m3_s,
            sideflow_m3=sideflow_m3,
            evaporation_m3=np.zeros(mask.sum(), dtype=np.float32),
            waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
            retention_storage_m3=np.zeros(mask.sum(), dtype=np.float32),
            river_storage_alpha=np.full(mask.sum(), np.float32(1.0), dtype=np.float32),
            river_storage_beta=np.full(mask.sum(), np.float32(0.6), dtype=np.float32),
        )
        total_volume_out_m3 += outflow_step_m3
        Q_prev_m3_s = Q_new.copy()

    # Check stability: Q_new should be positive and finite
    assert np.isfinite(Q_new).all()
    assert (Q_new >= 1e-30).all()

    # Mass balance check
    river_storage_alpha = np.full(mask.sum(), np.float32(1.0), dtype=np.float32)
    river_storage_beta = np.full(mask.sum(), np.float32(0.6), dtype=np.float32)
    storage_end_m3 = router.get_total_storage(
        Q_new, river_storage_alpha, river_storage_beta
    ).sum()

    # total_in = total_out + total_stored_at_end (initial storage was ~0)
    np.testing.assert_allclose(
        total_volume_in_m3, total_volume_out_m3 + storage_end_m3, rtol=1e-5
    )

    print(Q_new.max(), Q_new.min(), storage_end_m3, total_volume_out_m3)


def _make_two_cell_router(
    dt: int,
    activation_threshold_m3_per_s: float,
    max_storage_m3: float,
    controlled: bool,
) -> tuple[Accuflux, np.ndarray]:
    """Build a minimal two-cell Accuflux router with a single retention basin.

    The network is a 2×1 grid where cell (0,0) drains south into cell (1,0),
    which is a pit and hosts the retention basin.

    Args:
        dt: Time step in seconds.
        activation_threshold_m3_per_s: Activation threshold for the basin (m³/s).
        max_storage_m3: Maximum storage of the retention basin (m³).
        controlled: Whether the basin is controlled (True) or uncontrolled (False).

    Returns:
        A tuple of (router, mask) where mask is the boolean 2×1 grid array.
    """
    ldd = np.array([[2], [5]], dtype=np.uint8)  # (0,0)→(1,0)→pit
    mask = np.ones((2, 1), dtype=bool)

    river_network = create_river_network(ldd, mask)

    n_cells = mask.sum()  # 2
    retention_node_id = np.array([-1, 0], dtype=np.int32)  # basin at cell (1,0)
    retention_max_storage_m3 = np.array([max_storage_m3], dtype=np.float32)
    controlled_retention = np.array([controlled], dtype=bool)

    if controlled:
        threshold_controlled = np.array(
            [activation_threshold_m3_per_s], dtype=np.float32
        )
        threshold_uncontrolled = np.array([0.0], dtype=np.float32)
    else:
        threshold_controlled = np.array([0.0], dtype=np.float32)
        threshold_uncontrolled = np.array(
            [activation_threshold_m3_per_s], dtype=np.float32
        )

    router = Accuflux(
        dt=dt,
        river_network=river_network,
        river_length=np.ones(n_cells, dtype=np.float32),
        waterbody_id=np.full(n_cells, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_cells, dtype=bool),
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_activation_threshold_controlled_m3_s=threshold_controlled,
        retention_activation_threshold_uncontrolled_m3_s=threshold_uncontrolled,
    )
    return router, mask


def _run_retention_step(
    router: Accuflux,
    mask: np.ndarray,
    upstream_discharge_m3_per_s: float,
    initial_retention_storage_m3: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single Accuflux step for the two-cell retention test network.

    Args:
        router: The Accuflux router to use.
        mask: Boolean grid mask (2×1).
        upstream_discharge_m3_per_s: Initial discharge at the upstream headwater cell (m³/s).
        initial_retention_storage_m3: Pre-existing storage in the retention basin (m³).

    Returns:
        A tuple of (retention_storage_m3, retention_inflow_m3, retention_outflow_m3).
    """
    # Only the upstream headwater cell has a non-zero initial discharge; the
    # retention-basin cell starts at 0 (it is a pit with no independent source).
    Q_prev = np.array([upstream_discharge_m3_per_s, 0.0], dtype=np.float32)
    sideflow = np.zeros(mask.sum(), dtype=np.float32)
    retention_storage = np.array([initial_retention_storage_m3], dtype=np.float32)

    (
        _,
        _,
        _,
        _,
        _,
        _,
        retention_storage_out,
        retention_inflow,
        retention_outflow,
    ) = router.step(
        Q_prev_m3_s=Q_prev,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float64),
        retention_storage_m3=retention_storage,
        river_storage_alpha=np.zeros_like(sideflow),
        river_storage_beta=np.zeros_like(sideflow),
    )
    return retention_storage_out, retention_inflow, retention_outflow


def test_retention_no_diversion_below_threshold() -> None:
    """No water is diverted when discharge is strictly below the activation threshold.

    With discharge < activation_threshold, the diversion condition is false
    and the retention basin should remain empty.
    """
    dt = 1
    discharge_m3_per_s = 5.0
    activation_threshold_m3_per_s = 7.0

    router, mask = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=activation_threshold_m3_per_s,
        max_storage_m3=100.0,
        controlled=True,
    )
    retention_storage, retention_inflow, retention_outflow = _run_retention_step(
        router, mask, upstream_discharge_m3_per_s=discharge_m3_per_s
    )

    assert retention_inflow[0] == pytest.approx(0.0), (
        "No inflow expected when discharge is below the activation threshold"
    )
    assert retention_storage[0] == pytest.approx(0.0)
    assert retention_outflow[0] == pytest.approx(0.0)


def test_retention_no_diversion_at_threshold() -> None:
    """No water is diverted when discharge equals the activation threshold exactly.

    The condition uses strict inequality (>), so discharge equal to the threshold
    must not trigger diversion.
    """
    dt = 1
    threshold = 7.0

    router, mask = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=threshold,
        max_storage_m3=100.0,
        controlled=True,
    )
    retention_storage, retention_inflow, retention_outflow = _run_retention_step(
        router, mask, upstream_discharge_m3_per_s=threshold
    )

    assert retention_inflow[0] == pytest.approx(0.0), (
        "No inflow expected when discharge equals the activation threshold"
    )
    assert retention_storage[0] == pytest.approx(0.0)


def test_retention_inflow_limited_to_discharge_above_threshold() -> None:
    """Diverted volume is capped at (discharge − threshold) × dt, not the full discharge.

    Setup:
    - discharge = 10 m³/s, activation_threshold = 7 m³/s  → excess = 3 m³/s
    - dt = 1 s  → discharge_above_threshold limit = 3 m³
    - max_storage = 100 m³  → inflow_limit = 0.05 × 100 = 5 m³  (not binding)
    - available storage = 100 m³  (not binding)
    - binding constraint: discharge_above_threshold × dt = 3 m³

    The basin must receive exactly 3 m³, not the full 10 m³.
    """
    dt = 1
    discharge_m3_per_s = 10.0
    activation_threshold_m3_per_s = 7.0
    expected_diversion_m3 = (discharge_m3_per_s - activation_threshold_m3_per_s) * dt

    router, mask = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=activation_threshold_m3_per_s,
        max_storage_m3=100.0,
        controlled=True,
    )
    retention_storage, retention_inflow, retention_outflow = _run_retention_step(
        router, mask, upstream_discharge_m3_per_s=discharge_m3_per_s
    )

    np.testing.assert_allclose(
        retention_inflow[0],
        expected_diversion_m3,
        rtol=1e-5,
        err_msg="Diverted volume must equal (discharge − threshold) × dt when that is the binding constraint",
    )
    # Storage equals inflow minus the small 1%-per-timestep outflow that fires
    # immediately once water enters the basin.
    assert retention_storage[0] < retention_inflow[0] + retention_outflow[0] + 1e-6
    assert retention_storage[0] > 0.0


def test_retention_inflow_limited_to_discharge_above_threshold_with_longer_dt() -> None:
    """Activation-threshold limit scales correctly with a longer time step.

    With dt = 3600 s:
    - discharge_above_threshold = 3 m³/s  → limit = 3 × 3600 = 10 800 m³
    - max_storage = 500 000 m³  → inflow_limit = 0.05 × 500 000 = 25 000 m³
    - available storage = 500 000 m³
    - binding constraint: 10 800 m³
    """
    dt = 3600
    discharge_m3_per_s = 10.0
    activation_threshold_m3_per_s = 7.0
    expected_diversion_m3 = (discharge_m3_per_s - activation_threshold_m3_per_s) * dt

    router, mask = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=activation_threshold_m3_per_s,
        max_storage_m3=500_000.0,
        controlled=True,
    )
    retention_storage, retention_inflow, _ = _run_retention_step(
        router, mask, upstream_discharge_m3_per_s=discharge_m3_per_s
    )

    np.testing.assert_allclose(
        retention_inflow[0],
        expected_diversion_m3,
        rtol=1e-5,
    )


def test_retention_controlled_uses_controlled_threshold() -> None:
    """A controlled retention basin uses the controlled activation-threshold array.

    The controlled threshold (6 m³/s) is set to be active for the given discharge
    (8 m³/s), while the uncontrolled threshold is set high (999 m³/s) to ensure it
    would never activate. With controlled=True, diversion must occur.
    """
    dt = 1
    discharge_m3_per_s = 8.0
    controlled_threshold = 6.0
    uncontrolled_threshold = 999.0
    expected_diversion_m3 = (discharge_m3_per_s - controlled_threshold) * dt

    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask)
    n_cells = mask.sum()

    router = Accuflux(
        dt=dt,
        river_network=river_network,
        river_length=np.ones(n_cells, dtype=np.float32),
        waterbody_id=np.full(n_cells, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_cells, dtype=bool),
        retention_max_storage_m3=np.array([100.0], dtype=np.float32),
        retention_node_id=np.array([-1, 0], dtype=np.int32),
        controlled_retention=np.array([True], dtype=bool),
        retention_activation_threshold_controlled_m3_s=np.array(
            [controlled_threshold], dtype=np.float32
        ),
        retention_activation_threshold_uncontrolled_m3_s=np.array(
            [uncontrolled_threshold], dtype=np.float32
        ),
    )
    _, retention_inflow, _ = _run_retention_step(
        router, mask, upstream_discharge_m3_per_s=discharge_m3_per_s
    )

    np.testing.assert_allclose(retention_inflow[0], expected_diversion_m3, rtol=1e-5)


def test_retention_uncontrolled_uses_uncontrolled_threshold() -> None:
    """An uncontrolled retention basin uses the uncontrolled activation-threshold array.

    The uncontrolled threshold (6 m³/s) is active for the given discharge (8 m³/s),
    while the controlled threshold is set high (999 m³/s). With controlled=False,
    diversion must occur using the uncontrolled threshold.
    """
    dt = 1
    discharge_m3_per_s = 8.0
    controlled_threshold = 999.0
    uncontrolled_threshold = 6.0
    expected_diversion_m3 = (discharge_m3_per_s - uncontrolled_threshold) * dt

    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask)
    n_cells = mask.sum()

    router = Accuflux(
        dt=dt,
        river_network=river_network,
        river_length=np.ones(n_cells, dtype=np.float32),
        waterbody_id=np.full(n_cells, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_cells, dtype=bool),
        retention_max_storage_m3=np.array([100.0], dtype=np.float32),
        retention_node_id=np.array([-1, 0], dtype=np.int32),
        controlled_retention=np.array([False], dtype=bool),
        retention_activation_threshold_controlled_m3_s=np.array(
            [controlled_threshold], dtype=np.float32
        ),
        retention_activation_threshold_uncontrolled_m3_s=np.array(
            [uncontrolled_threshold], dtype=np.float32
        ),
    )
    _, retention_inflow, _ = _run_retention_step(
        router, mask, upstream_discharge_m3_per_s=discharge_m3_per_s
    )

    np.testing.assert_allclose(retention_inflow[0], expected_diversion_m3, rtol=1e-5)


def test_retention_release_at_low_flow() -> None:
    """Water is released from the basin back into the river when flow is low.

    Setup:
    - Initial storage = 1000 m³
    - release_threshold = 5.0 m³/s (1/2 of activation_threshold=10)
    - Low river flow = 2.0 m³/s
    - Room for release = (5.0 - 2.0) = 3.0 m³/s
    - 1% storage release = 10 m³
    - dt = 1s -> max release speed constraint = 3.0 m³/s * 1s = 3 m³

    Expected:
    - Outflow = min(10, 3) = 3 m³
    - Final storage = 1000 - 3 = 997 m³
    """
    dt = 1
    activation_threshold = 10.0
    initial_storage = 1000.0
    low_discharge = 2.0

    router, mask = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=activation_threshold,
        max_storage_m3=2000.0,
        controlled=True,
    )

    storage_out, inflow, outflow = _run_retention_step(
        router,
        mask,
        upstream_discharge_m3_per_s=low_discharge,
        initial_retention_storage_m3=initial_storage,
    )

    assert outflow[0] == pytest.approx(3.0)
    assert storage_out[0] == pytest.approx(997.0)
    assert inflow[0] == pytest.approx(0.0)
