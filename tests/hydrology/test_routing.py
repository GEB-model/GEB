"""Tests for hydrological routing functions in GEB."""

import math

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pyflwdir
import pytest
from affine import Affine

from geb.hydrology.routing import (
    create_river_network,
    get_channel_ratio,
)
from geb.hydrology.routing.local_inertial import LocalInertial, update_node_kinematic


def _make_local_inertial(
    dt: float | int,
    river_network: pyflwdir.FlwdirRaster,
    river_length: np.ndarray,
    river_width: np.ndarray | None = None,
    waterbody_ids: np.ndarray | None = None,
    river_ids: np.ndarray | None = None,
    is_waterbody_outflow: np.ndarray | None = None,
    retention_max_storage_m3: np.ndarray | None = None,
    retention_node_id: np.ndarray | None = None,
    controlled_retention: np.ndarray | None = None,
    retention_basin_release_threshold_factor: float = 0.9,
    bankfull_river_elevation_m: np.ndarray | None = None,
    manning_n: np.ndarray | None = None,
    use_kinematic: np.ndarray | None = None,
    rivers_gdf: gpd.GeoDataFrame | None = None,
    min_slope: float = 1e-4,
) -> LocalInertial:
    """Helper to instantiate LocalInertial for unit tests with explicit required arrays.

    Returns:
        Configured LocalInertial router instance.
    """
    n_cells = river_length.size
    if river_width is None:
        river_width = np.full(n_cells, np.nan, dtype=np.float32)
    if waterbody_ids is None:
        waterbody_ids = np.full(n_cells, -1, dtype=np.int32)
    if river_ids is None:
        river_ids = np.arange(n_cells, dtype=np.int32)
    if is_waterbody_outflow is None:
        is_waterbody_outflow = np.zeros(n_cells, dtype=bool)
    if retention_max_storage_m3 is None:
        retention_max_storage_m3 = np.zeros(0, dtype=np.float32)
    if retention_node_id is None:
        retention_node_id = np.full(n_cells, -1, dtype=np.int32)
    if controlled_retention is None:
        controlled_retention = np.zeros(0, dtype=bool)
    if bankfull_river_elevation_m is None:
        bankfull_river_elevation_m = np.zeros(n_cells, dtype=np.float32)
    if manning_n is None:
        manning_n = np.full(n_cells, 0.03, dtype=np.float32)
    if use_kinematic is None:
        use_kinematic = np.isnan(river_width)
    if rivers_gdf is None:
        rivers_gdf = gpd.GeoDataFrame(
            {
                "downstream_ID": np.full(n_cells, -1, dtype=np.int32),
                "slope": np.full(n_cells, 0.001, dtype=np.float32),
            },
            index=river_ids,
        )

    return LocalInertial(
        dt=dt,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        waterbody_ids=waterbody_ids,
        river_ids=river_ids,
        is_waterbody_outflow=is_waterbody_outflow,
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_basin_release_threshold_factor=retention_basin_release_threshold_factor,
        bankfull_river_elevation_m=bankfull_river_elevation_m,
        manning_n=manning_n,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        min_slope=min_slope,
    )


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


def test_local_inertial_basic(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test the local inertial routing basic routing step."""
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    router = _make_local_inertial(
        dt=15,
        river_network=river_network,
        river_length=np.full_like(mask, 15.0, dtype=np.float32)[mask],
        waterbody_ids=np.full_like(mask, -1, dtype=np.int32)[mask],
        is_waterbody_outflow=np.zeros_like(mask, dtype=bool)[mask],
        retention_max_storage_m3=np.zeros(mask.sum(), dtype=np.float32),
        retention_node_id=np.full(mask.sum(), -1, dtype=np.int32),
        controlled_retention=np.zeros(mask.sum(), dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.zeros(mask.sum(), dtype=np.float32),
        manning_n=np.full(mask.sum(), 0.03, dtype=np.float32),
    )

    sideflow = np.zeros(mask.sum(), dtype=np.float32)
    river_storage = np.zeros(mask.sum(), dtype=np.float64)

    (
        Q_new,
        river_storage_out,
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
        river_storage_m3=river_storage,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(mask.sum(), dtype=np.float32),
        river_storage_alpha=np.full_like(mask[mask], 1.0, dtype=np.float32),
        river_storage_beta=np.full_like(mask[mask], 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(mask.sum(), dtype=np.float32),
    )

    assert Q_new.shape[0] == mask.sum()
    assert not np.isnan(Q_new).any()
    assert np.isfinite(Q_new).all()
    assert (Q_new >= 0.0).all()


def test_local_inertial_with_retention_basins(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test LocalInertial routing with retention basins."""
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells = mask.sum()

    sideflow = np.zeros(n_cells, dtype=np.float32)
    retention_raster = -1 * np.ones_like(Q_initial, dtype=np.int32)
    retention_raster[2, 1] = 0
    retention_raster[0, 1] = 1
    retention_node_id = retention_raster[mask]

    retention_max_storage_m3 = np.array([2.0, 2.0], dtype=np.float32)
    controlled_retention = np.array([True, False], dtype=bool)
    retention_activation_threshold_m3_s = np.array([2.0, 1.0], dtype=np.float32)

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 100.0, dtype=np.float32),
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_basin_release_threshold_factor=0.2,
        use_kinematic=np.ones(n_cells, dtype=bool),
    )

    retention_storage_m3 = np.zeros(2, dtype=np.float32)
    river_storage_m3 = np.zeros(n_cells, dtype=np.float64)

    (
        Q_new,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage,
        wb_inflow,
        outflow_at_pits,
        retention_storage_out,
        retention_inflow,
        retention_outflow,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        river_storage_m3=river_storage_m3,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=retention_storage_m3,
        river_storage_alpha=np.full(n_cells, 1.0, dtype=np.float32),
        river_storage_beta=np.full(n_cells, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
    )

    assert Q_new.shape[0] == n_cells
    assert not np.isnan(Q_new).any()
    assert (retention_storage_out <= retention_max_storage_m3).all()
    assert (retention_inflow >= 0.0).all()


def test_local_inertial_with_longer_dt(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test LocalInertial routing with longer time steps."""
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells = mask.sum()

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 1000.0, dtype=np.float32),
        river_width=np.full(n_cells, 20.0, dtype=np.float32),
    )

    sideflow = np.zeros(n_cells, dtype=np.float32)
    river_storage_m3 = np.zeros(n_cells, dtype=np.float64)

    (
        Q_new,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage,
        wb_inflow,
        outflow_at_pits,
        retention_storage_out,
        retention_inflow,
        retention_outflow,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        river_storage_m3=river_storage_m3,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        river_storage_alpha=np.full(n_cells, 1.0, dtype=np.float32),
        river_storage_beta=np.full(n_cells, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    assert Q_new.shape[0] == n_cells
    assert not np.isnan(Q_new).any()
    assert (Q_new >= 0.0).all()


def test_local_inertial_with_sideflow(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test LocalInertial routing incorporating side flow."""
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells = mask.sum()

    router = _make_local_inertial(
        dt=15,
        river_network=river_network,
        river_length=np.full(n_cells, 100.0, dtype=np.float32),
        use_kinematic=np.ones(n_cells, dtype=bool),
    )

    sideflow = np.ones(n_cells, dtype=np.float32) * 5.0
    river_storage_m3 = np.zeros(n_cells, dtype=np.float64)

    (
        Q_new,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage,
        wb_inflow,
        outflow_at_pits,
        retention_storage_out,
        retention_inflow,
        retention_outflow,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        river_storage_m3=river_storage_m3,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        river_storage_alpha=np.full(n_cells, 1.0, dtype=np.float32),
        river_storage_beta=np.full(n_cells, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    assert (Q_new > 0.0).all()
    assert outflow_at_pits > 0.0


def test_local_inertial_with_waterbodies(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test LocalInertial routing through water bodies."""
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells = mask.sum()

    waterbody_id = np.array(
        [
            [-1, -1, -1, 0],
            [-1, -1, 0, 0],
            [-1, 1, -1, -1],
            [1, -1, -1, -1],
        ],
        dtype=np.int32,
    )[mask]

    is_waterbody_outflow = np.array(
        [
            [False, False, False, False],
            [False, False, True, False],
            [False, False, False, False],
            [True, False, False, False],
        ],
        dtype=bool,
    )[mask]

    router = _make_local_inertial(
        dt=1,
        river_network=river_network,
        river_length=np.full(n_cells, 10.0, dtype=np.float32),
        waterbody_ids=waterbody_id,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=np.ones(n_cells, dtype=bool),
    )

    waterbody_storage_m3 = np.array([10.0, 5.0], dtype=np.float64)
    outflow_per_waterbody_m3 = np.array([2.0, 2.0], dtype=np.float32)
    sideflow = np.zeros(n_cells, dtype=np.float32)
    river_storage_m3 = np.zeros(n_cells, dtype=np.float64)

    (
        Q_new,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage_out,
        wb_inflow,
        outflow_at_pits,
        retention_storage_out,
        retention_inflow,
        retention_outflow,
    ) = router.step(
        Q_prev_m3_s=Q_initial[mask],
        river_storage_m3=river_storage_m3,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=waterbody_storage_m3,
        outflow_per_waterbody_m3=outflow_per_waterbody_m3,
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        river_storage_alpha=np.full(n_cells, 1.0, dtype=np.float32),
        river_storage_beta=np.full(n_cells, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    assert np.isnan(Q_new[waterbody_id != -1]).all()
    assert not np.isnan(Q_new[waterbody_id == -1]).any()
    assert (wb_inflow >= 0.0).all()


def test_local_inertial_inverse_ops(
    ldd: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    Q_initial: npt.NDArray[np.float32],
) -> None:
    """Test if LocalInertial's total_storage and discharge_from_river_storage are inverses."""
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    dt = 3600
    river_length = np.full_like(mask[mask], 100.0, dtype=np.float32)
    waterbody_id = np.full_like(mask[mask], -1, dtype=np.int32)
    is_waterbody_outflow = np.zeros_like(mask[mask], dtype=bool)
    retention_max_storage_m3 = np.zeros(mask.sum(), dtype=np.float32)
    retention_node_id = np.full(mask.sum(), -1, dtype=np.int32)
    controlled_retention = np.zeros(mask.sum(), dtype=bool)

    router = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=river_length,
        waterbody_ids=waterbody_id,
        is_waterbody_outflow=is_waterbody_outflow,
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.zeros(mask.sum(), dtype=np.float32),
        manning_n=np.full(mask.sum(), 0.03, dtype=np.float32),
    )

    Q = Q_initial[mask]
    alpha = np.full_like(Q, 1.5, dtype=np.float32)
    beta = np.full_like(Q, 0.6, dtype=np.float32)

    storage = router.get_total_storage(Q, alpha, beta)
    Q_inv = router.calculate_discharge_from_river_storage(
        storage, alpha, beta, river_length, waterbody_id
    )

    np.testing.assert_allclose(Q, Q_inv, rtol=1e-4)


def test_local_inertial_sudden_flood_wave(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
) -> None:
    """Test local inertial wave routing with a sudden massive flood wave."""
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    dt = 3600
    n_cells = mask.sum()

    router = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=np.full(n_cells, 1000.0, dtype=np.float32),
        river_width=np.full(n_cells, 20.0, dtype=np.float32),
        waterbody_ids=np.full(n_cells, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_cells, dtype=bool),
        retention_max_storage_m3=np.zeros(n_cells, dtype=np.float32),
        retention_node_id=np.full(n_cells, -1, dtype=np.int32),
        controlled_retention=np.zeros(n_cells, dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.zeros(n_cells, dtype=np.float32),
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
    )

    retention_activation_threshold_m3_s = np.zeros(n_cells, dtype=np.float32)
    Q_prev_m3_s = np.full(n_cells, 1e-30, dtype=np.float32)
    river_storage_m3 = np.zeros(n_cells, dtype=np.float64)

    injection_node = n_cells - 1
    side_flow_m3_s = 10000.0

    total_volume_in_m3 = 0.0
    total_volume_out_m3 = 0.0

    sideflow_m3 = np.zeros(n_cells, dtype=np.float32)
    sideflow_m3[injection_node] = side_flow_m3_s * dt

    for _ in range(10):
        total_volume_in_m3 += sideflow_m3.sum()

        (
            Q_new,
            river_storage_m3,
            _,
            _,
            _,
            _,
            outflow_pits_m3,
            _,
            _,
            _,
        ) = router.step(
            Q_prev_m3_s=Q_prev_m3_s,
            river_storage_m3=river_storage_m3,
            sideflow_m3=sideflow_m3,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
            retention_storage_m3=np.zeros(n_cells, dtype=np.float32),
            river_storage_alpha=np.full(n_cells, 1.0, dtype=np.float32),
            river_storage_beta=np.full(n_cells, 0.6, dtype=np.float32),
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
        )

        total_volume_out_m3 += outflow_pits_m3
        Q_prev_m3_s = Q_new

        assert not np.isnan(Q_new).any()
        assert not np.isinf(Q_new).any()

    assert np.isclose(
        total_volume_out_m3 + river_storage_m3.sum(), total_volume_in_m3, rtol=1e-4
    )


def _make_two_cell_router(
    dt: int,
    activation_threshold_m3_per_s: float,
    max_storage_m3: float,
    controlled: bool,
    release_threshold_factor: float = 0.9,
) -> tuple[LocalInertial, np.ndarray, np.ndarray, np.ndarray]:
    """Build a minimal two-cell LocalInertial router with a single retention basin.

    Returns:
        Tuple of (router, mask, controlled_threshold, uncontrolled_threshold).
    """
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)

    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    n_cells = mask.sum()
    retention_node_id = np.array([-1, 0], dtype=np.int32)
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

    router = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=np.ones(n_cells, dtype=np.float32),
        waterbody_ids=np.full(n_cells, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_cells, dtype=bool),
        retention_max_storage_m3=retention_max_storage_m3,
        retention_node_id=retention_node_id,
        controlled_retention=controlled_retention,
        retention_basin_release_threshold_factor=release_threshold_factor,
        use_kinematic=np.ones(n_cells, dtype=bool),
    )
    return router, mask, threshold_controlled, threshold_uncontrolled


def _run_retention_step(
    router: LocalInertial,
    mask: np.ndarray,
    upstream_discharge_m3_per_s: float,
    initial_retention_storage_m3: float = 0.0,
    retention_activation_threshold_m3_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single LocalInertial step for the two-cell retention test network.

    Returns:
        Tuple of (outflow, retention_inflow, retention_storage_out).
    """
    Q_prev = np.array(
        [upstream_discharge_m3_per_s, upstream_discharge_m3_per_s * 0.99999],
        dtype=np.float32,
    )
    sideflow = np.array(
        [upstream_discharge_m3_per_s * router.dt, 0.0], dtype=np.float32
    )
    river_storage = np.zeros(mask.sum(), dtype=np.float64)
    retention_storage = np.array([initial_retention_storage_m3], dtype=np.float32)
    if retention_activation_threshold_m3_s is None:
        retention_activation_threshold_m3_s = np.zeros(1, dtype=np.float32)

    (
        Q_out,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage_out,
        wb_inflow_out,
        outflow_at_pits,
        retention_storage_out,
        retention_inflow,
        retention_outflow,
    ) = router.step(
        Q_prev_m3_s=Q_prev,
        river_storage_m3=river_storage,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros_like(sideflow),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=retention_storage,
        river_storage_alpha=np.zeros_like(sideflow),
        river_storage_beta=np.zeros_like(sideflow),
        retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
    )
    return retention_storage_out, retention_inflow, retention_outflow


def test_retention_no_diversion_below_threshold() -> None:
    """No water is diverted when discharge is strictly below the activation threshold."""
    dt = 1
    discharge_m3_per_s = 5.0
    activation_threshold_m3_per_s = 7.0

    router, mask, threshold_controlled, threshold_uncontrolled = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=activation_threshold_m3_per_s,
        max_storage_m3=100.0,
        controlled=True,
    )
    retention_storage, retention_inflow, retention_outflow = _run_retention_step(
        router,
        mask,
        upstream_discharge_m3_per_s=discharge_m3_per_s,
        retention_activation_threshold_m3_s=threshold_controlled,
    )

    assert retention_inflow[0] == pytest.approx(0.0)
    assert retention_storage[0] == pytest.approx(0.0)
    assert retention_outflow[0] == pytest.approx(0.0)


def test_retention_no_diversion_at_threshold() -> None:
    """No water is diverted when discharge equals the activation threshold exactly."""
    dt = 1
    threshold = 7.0

    router, mask, threshold_controlled, threshold_uncontrolled = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=threshold,
        max_storage_m3=100.0,
        controlled=True,
    )
    retention_storage, retention_inflow, retention_outflow = _run_retention_step(
        router,
        mask,
        upstream_discharge_m3_per_s=threshold,
        retention_activation_threshold_m3_s=threshold_controlled,
    )

    assert retention_inflow[0] == pytest.approx(0.0)
    assert retention_storage[0] == pytest.approx(0.0)


def test_retention_inflow_limited_to_discharge_above_threshold() -> None:
    """Diverted volume is capped at (discharge − threshold) × dt."""
    dt = 1
    discharge_m3_per_s = 10.0
    activation_threshold_m3_per_s = 7.0
    expected_diversion_m3 = (discharge_m3_per_s - activation_threshold_m3_per_s) * dt

    router, mask, threshold_controlled, threshold_uncontrolled = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=activation_threshold_m3_per_s,
        max_storage_m3=100.0,
        controlled=False,
    )
    retention_storage, retention_inflow, retention_outflow = _run_retention_step(
        router,
        mask,
        upstream_discharge_m3_per_s=discharge_m3_per_s,
        retention_activation_threshold_m3_s=threshold_uncontrolled,
    )

    np.testing.assert_allclose(
        retention_inflow[0],
        expected_diversion_m3,
        rtol=1e-4,
    )
    assert retention_storage[0] > 0.0


def test_retention_inflow_limited_to_discharge_above_threshold_with_longer_dt() -> None:
    """Activation-threshold limit scales correctly with a longer time step."""
    dt = 3600
    discharge_m3_per_s = 10.0
    activation_threshold_m3_per_s = 7.0
    expected_diversion_m3 = (discharge_m3_per_s - activation_threshold_m3_per_s) * dt

    router, mask, threshold_controlled, threshold_uncontrolled = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=activation_threshold_m3_per_s,
        max_storage_m3=500_000.0,
        controlled=False,
    )
    retention_storage, retention_inflow, _ = _run_retention_step(
        router,
        mask,
        upstream_discharge_m3_per_s=discharge_m3_per_s,
        retention_activation_threshold_m3_s=threshold_uncontrolled,
    )

    np.testing.assert_allclose(
        retention_inflow[0],
        expected_diversion_m3,
        rtol=1e-4,
    )


def test_retention_controlled_uses_controlled_threshold() -> None:
    """A controlled retention basin uses the controlled activation-threshold array."""
    dt = 1
    discharge_m3_per_s = 8.0
    controlled_threshold = 6.0
    expected_diversion_m3 = (discharge_m3_per_s - controlled_threshold) * dt

    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells = mask.sum()

    threshold_controlled = np.array([controlled_threshold], dtype=np.float32)
    router = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=np.ones(n_cells, dtype=np.float32),
        waterbody_ids=np.full(n_cells, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_cells, dtype=bool),
        retention_max_storage_m3=np.array([100.0], dtype=np.float32),
        retention_node_id=np.array([-1, 0], dtype=np.int32),
        controlled_retention=np.array([True], dtype=bool),
        retention_basin_release_threshold_factor=0.9,
        use_kinematic=np.ones(n_cells, dtype=bool),
    )
    _, retention_inflow, _ = _run_retention_step(
        router,
        mask,
        upstream_discharge_m3_per_s=discharge_m3_per_s,
        retention_activation_threshold_m3_s=threshold_controlled,
    )

    np.testing.assert_allclose(retention_inflow[0], expected_diversion_m3, rtol=1e-4)


def test_retention_uncontrolled_uses_uncontrolled_threshold() -> None:
    """An uncontrolled retention basin uses the uncontrolled activation-threshold array."""
    dt = 1
    discharge_m3_per_s = 8.0
    controlled_threshold = 999.0
    uncontrolled_threshold = 6.0
    expected_diversion_m3 = (discharge_m3_per_s - uncontrolled_threshold) * dt

    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells = mask.sum()

    threshold_uncontrolled = np.array([uncontrolled_threshold], dtype=np.float32)
    router = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=np.ones(n_cells, dtype=np.float32),
        waterbody_ids=np.full(n_cells, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_cells, dtype=bool),
        retention_max_storage_m3=np.array([100.0], dtype=np.float32),
        retention_node_id=np.array([-1, 0], dtype=np.int32),
        controlled_retention=np.array([False], dtype=bool),
        retention_basin_release_threshold_factor=0.9,
        use_kinematic=np.ones(n_cells, dtype=bool),
    )
    _, retention_inflow, _ = _run_retention_step(
        router,
        mask,
        upstream_discharge_m3_per_s=discharge_m3_per_s,
        retention_activation_threshold_m3_s=threshold_uncontrolled,
    )

    np.testing.assert_allclose(retention_inflow[0], expected_diversion_m3, rtol=1e-4)


def test_retention_basin_evaporation_logic() -> None:
    """Test the calculation logic for retention basin evaporation."""
    retention_max_storage_m3 = np.array([300.0, 600.0], dtype=np.float32)
    retention_basin_ids = np.array([0, 0, 1, -1], dtype=np.int32)
    reference_evapotranspiration_water_m_hour = np.array(
        [0.01, 0.02, 0.03, 0.04], dtype=np.float32
    )
    retention_basin_storage_m3 = np.array([100.0, 50.0], dtype=np.float32)

    retention_basin_area = retention_max_storage_m3 / 3.0
    retention_mask = retention_basin_ids != -1
    count = np.bincount(
        retention_basin_ids[retention_mask], minlength=len(retention_basin_area)
    )

    et_sum = np.bincount(
        retention_basin_ids[retention_mask],
        weights=reference_evapotranspiration_water_m_hour[retention_mask],
        minlength=len(retention_basin_area),
    )

    avg_et = et_sum / np.maximum(count, 1)
    potential_evaporation_m3 = avg_et * retention_basin_area
    actual_evaporation_m3 = np.minimum(
        potential_evaporation_m3, retention_basin_storage_m3
    )

    assert np.allclose(actual_evaporation_m3, np.array([1.5, 6.0], dtype=np.float32))

    retention_basin_storage_m3 -= actual_evaporation_m3
    assert np.allclose(
        retention_basin_storage_m3, np.array([98.5, 44.0], dtype=np.float32)
    )


def test_retention_release_at_low_flow() -> None:
    """Water is released from the basin back into the river when flow is low."""
    dt = 1
    activation_threshold = 10.0
    initial_storage = 1000.0
    low_discharge = 2.0

    router, mask, threshold_controlled, threshold_uncontrolled = _make_two_cell_router(
        dt=dt,
        activation_threshold_m3_per_s=activation_threshold,
        max_storage_m3=2000.0,
        controlled=True,
        release_threshold_factor=0.75,
    )

    storage_out, inflow, outflow = _run_retention_step(
        router,
        mask,
        upstream_discharge_m3_per_s=low_discharge,
        initial_retention_storage_m3=initial_storage,
        retention_activation_threshold_m3_s=threshold_controlled,
    )

    assert outflow[0] == pytest.approx(5.5, rel=1e-4)
    assert storage_out[0] == pytest.approx(994.5, rel=1e-4)
    assert inflow[0] == pytest.approx(0.0)


def test_local_inertial_momentum_persistence(
    mask: npt.NDArray[np.bool_],
    ldd: npt.NDArray[np.uint8],
) -> None:
    """Test that LocalInertial preserves momentum across steps."""
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    dt = 3600
    n_cells = mask.sum()

    router = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=np.full(n_cells, 5000.0, dtype=np.float32),
        river_width=np.full(n_cells, 50.0, dtype=np.float32),
        waterbody_ids=np.full(n_cells, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_cells, dtype=bool),
        retention_max_storage_m3=np.zeros(n_cells, dtype=np.float32),
        retention_node_id=np.full(n_cells, -1, dtype=np.int32),
        controlled_retention=np.zeros(n_cells, dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.zeros(n_cells, dtype=np.float32),
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
    )

    Q_prev_m3_s = np.full(n_cells, 100.0, dtype=np.float32)
    river_storage_m3 = (
        np.full(n_cells, 5000.0, dtype=np.float64)
        * np.full(n_cells, 50.0, dtype=np.float64)
        * 3.0
    )

    (
        Q_step1,
        river_storage_step1,
        _,
        _,
        _,
        _,
        outflow_step1,
        _,
        _,
        _,
    ) = router.step(
        Q_prev_m3_s=Q_prev_m3_s,
        river_storage_m3=river_storage_m3,
        sideflow_m3=np.zeros(n_cells, dtype=np.float32),
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(n_cells, dtype=np.float32),
        river_storage_alpha=np.full(n_cells, 1.0, dtype=np.float32),
        river_storage_beta=np.full(n_cells, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(n_cells, dtype=np.float32),
    )

    assert not np.isnan(Q_step1).any()
    assert (Q_step1 > 0.0).any()
    assert (river_storage_step1 > 0.0).any()


def test_local_inertial_reverse_flow_mass_conservation() -> None:
    """Test two-cell network with adverse water surface gradient (reverse flow)."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.array([1000.0, 1000.0], dtype=np.float32),
        river_width=np.array([10.0, 10.0], dtype=np.float32),
        waterbody_ids=np.array([-1, -1], dtype=np.int32),
        is_waterbody_outflow=np.array([False, False], dtype=bool),
        retention_max_storage_m3=np.zeros(2, dtype=np.float32),
        retention_node_id=np.array([-1, -1], dtype=np.int32),
        controlled_retention=np.zeros(2, dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.array([0.0, 0.0], dtype=np.float32),
        manning_n=np.array([0.03, 0.03], dtype=np.float32),
    )

    Q_prev = np.array([0.0, 0.0], dtype=np.float32)
    river_storage_m3 = np.array(
        [1000.0 * 10.0 * 0.5, 1000.0 * 10.0 * 2.5], dtype=np.float64
    )

    total_storage_before = river_storage_m3.sum()

    (
        Q_new,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage,
        wb_inflow,
        outflow_at_pits,
        retention_storage_out,
        retention_inflow,
        retention_outflow,
    ) = router.step(
        Q_prev_m3_s=Q_prev,
        river_storage_m3=river_storage_m3,
        sideflow_m3=np.zeros(2, dtype=np.float32),
        evaporation_m3=np.zeros(2, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(2, dtype=np.float32),
        river_storage_alpha=np.full(2, 1.0, dtype=np.float32),
        river_storage_beta=np.full(2, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(2, dtype=np.float32),
    )

    total_storage_after = river_storage_out.sum()
    np.testing.assert_allclose(
        total_storage_before,
        total_storage_after + outflow_at_pits,
        rtol=1e-4,
    )


def test_local_inertial_head_gradient_overflow_resilience() -> None:
    """Test stability under extreme head gradient step changes."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.array([1000.0, 1000.0], dtype=np.float32),
        river_width=np.array([10.0, 10.0], dtype=np.float32),
        waterbody_ids=np.array([-1, -1], dtype=np.int32),
        is_waterbody_outflow=np.array([False, False], dtype=bool),
        retention_max_storage_m3=np.zeros(2, dtype=np.float32),
        retention_node_id=np.array([-1, -1], dtype=np.int32),
        controlled_retention=np.zeros(2, dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.array([0.0, 0.0], dtype=np.float32),
        manning_n=np.array([0.03, 0.03], dtype=np.float32),
    )

    Q_prev = np.array([0.0, 0.0], dtype=np.float32)
    river_storage_m3 = np.array(
        [1000.0 * 10.0 * 50.0, 1000.0 * 10.0 * 0.001], dtype=np.float64
    )

    (
        Q_new,
        river_storage_out,
        _,
        _,
        _,
        _,
        outflow_at_pits,
        _,
        _,
        _,
    ) = router.step(
        Q_prev_m3_s=Q_prev,
        river_storage_m3=river_storage_m3,
        sideflow_m3=np.zeros(2, dtype=np.float32),
        evaporation_m3=np.zeros(2, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(2, dtype=np.float32),
        river_storage_alpha=np.full(2, 1.0, dtype=np.float32),
        river_storage_beta=np.full(2, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(2, dtype=np.float32),
    )

    assert not np.isnan(Q_new).any()
    assert np.isfinite(Q_new).all()
    assert not np.isnan(river_storage_out).any()
    assert np.isfinite(river_storage_out).all()


def test_local_inertial_raises_on_non_finite_inputs() -> None:
    """Test that local inertial routing raises ValueError when non-finite inputs are passed."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.array([1000.0, 1000.0], dtype=np.float32),
        river_width=np.array([10.0, 10.0], dtype=np.float32),
        waterbody_ids=np.array([-1, -1], dtype=np.int32),
        is_waterbody_outflow=np.array([False, False], dtype=bool),
        retention_max_storage_m3=np.zeros(2, dtype=np.float32),
        retention_node_id=np.array([-1, -1], dtype=np.int32),
        controlled_retention=np.zeros(2, dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.array([0.0, 0.0], dtype=np.float32),
        manning_n=np.array([0.03, 0.03], dtype=np.float32),
    )

    # Test NaN in river_storage_m3
    with pytest.raises(ValueError, match="Non-finite"):
        router.step(
            Q_prev_m3_s=np.array([0.0, 0.0], dtype=np.float32),
            river_storage_m3=np.array([np.nan, 1000.0], dtype=np.float64),
            sideflow_m3=np.zeros(2, dtype=np.float32),
            evaporation_m3=np.zeros(2, dtype=np.float32),
            waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
            retention_storage_m3=np.zeros(2, dtype=np.float32),
            river_storage_alpha=np.full(2, 1.0, dtype=np.float32),
            river_storage_beta=np.full(2, 0.6, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(2, dtype=np.float32),
        )

    # Test NaN in sideflow_m3
    with pytest.raises(ValueError, match="Non-finite"):
        router.step(
            Q_prev_m3_s=np.array([0.0, 0.0], dtype=np.float32),
            river_storage_m3=np.array([1000.0, 1000.0], dtype=np.float64),
            sideflow_m3=np.array([np.nan, 0.0], dtype=np.float32),
            evaporation_m3=np.zeros(2, dtype=np.float32),
            waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
            retention_storage_m3=np.zeros(2, dtype=np.float32),
            river_storage_alpha=np.full(2, 1.0, dtype=np.float32),
            river_storage_beta=np.full(2, 0.6, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(2, dtype=np.float32),
        )


def test_local_inertial_backflow_into_dry_cell() -> None:
    """Test that downstream water level rise causes backflow into an initially dry upstream reach."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.array([1000.0, 1000.0], dtype=np.float32),
        river_width=np.array([10.0, 10.0], dtype=np.float32),
        waterbody_ids=np.array([-1, -1], dtype=np.int32),
        is_waterbody_outflow=np.array([False, False], dtype=bool),
        retention_max_storage_m3=np.zeros(2, dtype=np.float32),
        retention_node_id=np.array([-1, -1], dtype=np.int32),
        controlled_retention=np.zeros(2, dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.array([0.0, 0.0], dtype=np.float32),
        manning_n=np.array([0.03, 0.03], dtype=np.float32),
    )

    # Reach 0 is completely dry (0.0 m³), Reach 1 is flooded (30,000 m³ = 3m depth)
    Q_prev = np.array([0.0, 0.0], dtype=np.float32)
    river_storage_m3 = np.array([0.0, 1000.0 * 10.0 * 3.0], dtype=np.float64)

    (
        Q_new,
        river_storage_out,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = router.step(
        Q_prev_m3_s=Q_prev,
        river_storage_m3=river_storage_m3,
        sideflow_m3=np.zeros(2, dtype=np.float32),
        evaporation_m3=np.zeros(2, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(2, dtype=np.float32),
        river_storage_alpha=np.full(2, 1.0, dtype=np.float32),
        river_storage_beta=np.full(2, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(2, dtype=np.float32),
    )

    # Upstream cell must receive backflow (negative Q and non-zero storage)
    assert Q_new[0] < 0.0, (
        f"Expected backflow (negative discharge), but got Q_new[0] = {Q_new[0]}"
    )
    assert river_storage_out[0] > 0.0, (
        f"Expected upstream cell to receive water, but got {river_storage_out[0]}"
    )


def test_local_inertial_with_excess_abstraction() -> None:
    """Test that human abstraction exceeding available storage is tracked in over_abstraction and mass is conserved."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.array([1000.0, 1000.0], dtype=np.float32),
        river_width=np.array([10.0, 10.0], dtype=np.float32),
        waterbody_ids=np.array([-1, -1], dtype=np.int32),
        is_waterbody_outflow=np.array([False, False], dtype=bool),
        retention_max_storage_m3=np.zeros(2, dtype=np.float32),
        retention_node_id=np.array([-1, -1], dtype=np.int32),
        controlled_retention=np.zeros(2, dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.array([0.0, 0.0], dtype=np.float32),
        manning_n=np.array([0.03, 0.03], dtype=np.float32),
    )

    # Initial storage: Reach 0 has 1,000 m³, Reach 1 is dry (0 m³)
    river_storage_m3 = np.array([1000.0, 0.0], dtype=np.float64)
    # Requested sideflow abstraction: Reach 0 requests -5,000 m³ (deficit of 4,000 m³), Reach 1 has 0 m³
    sideflow_m3 = np.array([-5000.0, 0.0], dtype=np.float32)

    (
        Q_new,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage,
        wb_inflow,
        outflow_at_pits,
        ret_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=np.zeros(2, dtype=np.float32),
        river_storage_m3=river_storage_m3,
        sideflow_m3=sideflow_m3,
        evaporation_m3=np.zeros(2, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(2, dtype=np.float32),
        river_storage_alpha=np.full(2, 1.0, dtype=np.float32),
        river_storage_beta=np.full(2, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(2, dtype=np.float32),
    )

    # Over-abstraction in Reach 0 must capture the unfulfilled abstraction deficit
    assert over_abs[0] > 3900.0, f"Expected large over_abs[0], got {over_abs[0]}"
    # Storage in Reach 0 must be non-negative
    assert river_storage_out[0] >= 0.0
    # Mass conservation: initial storage = final storage + actual abstraction + pit outflow
    actual_abstraction = float(-sideflow_m3.sum() - over_abs.sum())
    total_final = float(river_storage_out.sum() + actual_abstraction + outflow_at_pits)
    assert np.isclose(total_final, 1000.0, rtol=1e-3)


def test_local_inertial_inertial_reaches_with_waterbodies() -> None:
    """Test local inertial wave reaches receiving waterbody releases and discharging to pits."""
    # 4-cell river: WB 0 (cell 0) -> Inertial 1 (cell 1) -> Inertial 2 (cell 2) -> River pit (cell 3)
    ldd = np.array([[2], [2], [2], [5]], dtype=np.uint8)
    mask = np.ones((4, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 4

    waterbody_id = np.array([0, -1, -1, -1], dtype=np.int32)
    is_waterbody_outflow = np.array([True, False, False, False], dtype=bool)

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 1000.0, dtype=np.float32),
        river_width=np.full(n_cells, 20.0, dtype=np.float32),
        waterbody_ids=waterbody_id,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=np.zeros(n_cells, dtype=bool),
        bankfull_river_elevation_m=np.array([30.0, 20.0, 10.0, 0.0], dtype=np.float32),
    )

    wb_storage_init = np.array([50000.0], dtype=np.float64)
    wb_outflow = np.array([7200.0], dtype=np.float32)
    river_storage_init = np.array([0.0, 5000.0, 5000.0, 5000.0], dtype=np.float64)

    total_initial: float = float(np.sum(wb_storage_init) + np.sum(river_storage_init))

    (
        Q_out,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage_out,
        wb_inflow,
        outflow_at_pits,
        ret_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=np.zeros(4, dtype=np.float32),
        river_storage_m3=river_storage_init.copy(),
        sideflow_m3=np.zeros(4, dtype=np.float32),
        evaporation_m3=np.zeros(4, dtype=np.float32),
        waterbody_storage_m3=wb_storage_init.copy(),
        outflow_per_waterbody_m3=wb_outflow,
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        river_storage_alpha=np.full(4, 1.0, dtype=np.float32),
        river_storage_beta=np.full(4, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Waterbody 0 should have released 7200 m3
    assert np.isclose(wb_storage_out[0], 50000.0 - 7200.0)
    # Mass balance for entire domain: total initial = total final + pit outflow
    total_final: float = float(
        np.sum(wb_storage_out) + np.sum(river_storage_out) + outflow_at_pits
    )
    assert np.isclose(total_initial, total_final, atol=1e-3)


def test_local_inertial_channel_evaporation() -> None:
    """Test evaporation in local inertial reaches under water-limited and unconstrained conditions."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.array([1000.0, 1000.0], dtype=np.float32),
        river_width=np.array([10.0, 10.0], dtype=np.float32),
        bankfull_river_elevation_m=np.array([10.0, 0.0], dtype=np.float32),
        use_kinematic=np.zeros(2, dtype=bool),
    )

    river_storage_init = np.array([500.0, 10000.0], dtype=np.float64)
    evaporation_m3 = np.array([2000.0, 1000.0], dtype=np.float32)

    (
        Q_out,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage_out,
        wb_inflow,
        outflow_at_pits,
        ret_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=np.array([0.0, 0.0], dtype=np.float32),
        river_storage_m3=river_storage_init,
        sideflow_m3=np.zeros(2, dtype=np.float32),
        evaporation_m3=evaporation_m3,
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        river_storage_alpha=np.full(2, 1.0, dtype=np.float32),
        river_storage_beta=np.full(2, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Actual evaporation on cell 0 is constrained by available water (<= 500 m3)
    assert actual_evap[0] <= 500.0 + 1e-4
    # Actual evaporation on cell 1 was fully satisfied (1000 m3)
    assert np.isclose(actual_evap[1], 1000.0, atol=1e-3)
    # Reach storage cannot be negative
    assert (river_storage_out >= 0.0).all()


def test_local_inertial_cascaded_waterbodies() -> None:
    """Test direct topological transfer between adjacent cascaded reservoirs."""
    ldd = np.array([[2], [2], [5]], dtype=np.uint8)
    mask = np.ones((3, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 3

    waterbody_id = np.array([0, 1, -1], dtype=np.int32)
    is_waterbody_outflow = np.array([True, True, False], dtype=bool)

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 1000.0, dtype=np.float32),
        waterbody_ids=waterbody_id,
        is_waterbody_outflow=is_waterbody_outflow,
        bankfull_river_elevation_m=np.array([20.0, 10.0, 0.0], dtype=np.float32),
    )

    wb_storage_init = np.array([10000.0, 5000.0], dtype=np.float64)
    wb_outflow = np.array([3000.0, 0.0], dtype=np.float32)

    (
        Q_out,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage_out,
        wb_inflow,
        outflow_at_pits,
        ret_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=np.zeros(3, dtype=np.float32),
        river_storage_m3=np.zeros(3, dtype=np.float64),
        sideflow_m3=np.zeros(3, dtype=np.float32),
        evaporation_m3=np.zeros(3, dtype=np.float32),
        waterbody_storage_m3=wb_storage_init,
        outflow_per_waterbody_m3=wb_outflow,
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        river_storage_alpha=np.full(3, 1.0, dtype=np.float32),
        river_storage_beta=np.full(3, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    assert np.isclose(wb_storage_out[0], 7000.0)
    assert np.isclose(wb_storage_out[1], 8000.0)
    assert np.isclose(wb_inflow[1], 3000.0)


def test_local_inertial_tributary_junction_multidirectional_scaling() -> None:
    """Test multidirectional backflow scaling at a tributary confluence junction."""
    # Junction topology: two upstream tributaries (cells 0 and 1) meet at cell 2
    ldd = np.array([[2, 4], [5, 5]], dtype=np.uint8)
    mask = np.array([[True, True], [True, False]], dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = int(mask.sum())

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 1000.0, dtype=np.float32),
        river_width=np.full(n_cells, 10.0, dtype=np.float32),
        bankfull_river_elevation_m=np.array([5.0, 5.0, 0.0], dtype=np.float32),
        use_kinematic=np.zeros(n_cells, dtype=bool),
    )

    # Junction (cell 2) has high water storage, tributaries have 0 storage
    river_storage_init = np.array([0.0, 0.0, 100000.0], dtype=np.float64)

    (
        Q_out,
        river_storage_out,
        actual_evap,
        over_abs,
        wb_storage_out,
        wb_inflow,
        outflow_at_pits,
        ret_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=np.zeros(n_cells, dtype=np.float32),
        river_storage_m3=river_storage_init,
        sideflow_m3=np.zeros(n_cells, dtype=np.float32),
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        river_storage_alpha=np.full(n_cells, 1.0, dtype=np.float32),
        river_storage_beta=np.full(n_cells, 0.6, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Both dry tributaries should receive reverse backflow from the high stage at the junction
    assert river_storage_out[0] > 0.0
    assert river_storage_out[1] > 0.0
    # Mass balance holds exactly
    total_after = float(np.sum(river_storage_out)) + float(outflow_at_pits)
    assert np.isclose(100000.0, total_after, rtol=1e-4)
