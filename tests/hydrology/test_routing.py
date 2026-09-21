"""Tests for hydrological routing functions in GEB."""

import math
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import numpy.typing as npt
import pyflwdir

matplotlib.use("Agg")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from affine import Affine

from geb.geb_types import ArrayFloat32
from geb.hydrology.routing import (
    RoutingVariables,
    create_river_network,
    get_channel_ratio,
    select_active_rivers,
)
from geb.hydrology.routing.geometry import (
    compute_cfl_area_and_top_width,
    compute_cross_section_from_depth,
    compute_overbank_depth,
    compute_static_geometry,
)
from geb.hydrology.routing.kinematic import update_node_kinematic
from geb.hydrology.routing.local_inertial import LocalInertial

from ..testconfig import output_folder


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
    shape_exponent: np.ndarray | float = 0.5,
    bankfull_depth_m: np.ndarray | float | None = None,
    floodplain_width_m: np.ndarray | None = None,
    waterbody_lake_area: np.ndarray | None = None,
    waterbody_lake_factor: np.ndarray | None = None,
    waterbody_outflow_height: np.ndarray | None = None,
    waterbody_outflow_bed_elev: np.ndarray | None = None,
    river_storage_alpha: np.ndarray | None = None,
    river_storage_beta: np.ndarray | None = None,
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
    if isinstance(shape_exponent, (int, float, np.floating)):
        shape_exponent = np.full(n_cells, shape_exponent, dtype=np.float32)
    if bankfull_depth_m is None:
        bankfull_depth_m = np.clip(
            np.where(
                np.isnan(river_width),
                np.float32(0.25),
                river_width / np.float32(20.0),
            ),
            np.float32(0.5),
            np.float32(3.0),
        ).astype(np.float32)
    elif isinstance(bankfull_depth_m, (int, float, np.floating)):
        bankfull_depth_m = np.full(n_cells, bankfull_depth_m, dtype=np.float32)
    if floodplain_width_m is None:
        floodplain_width_m = np.zeros(n_cells, dtype=np.float32)
    elif isinstance(floodplain_width_m, (int, float, np.floating)):
        floodplain_width_m = np.full(n_cells, floodplain_width_m, dtype=np.float32)

    if river_storage_alpha is None:
        river_storage_alpha = np.full(n_cells, 1.0, dtype=np.float32)
    if river_storage_beta is None:
        river_storage_beta = np.full(n_cells, 0.6, dtype=np.float32)

    n_wb: int = int(is_waterbody_outflow.sum())
    if waterbody_lake_area is None:
        waterbody_lake_area = np.ones(n_wb, dtype=np.float32) * np.float32(1e6)
    if waterbody_lake_factor is None:
        waterbody_lake_factor = np.ones(n_wb, dtype=np.float32)
    if waterbody_outflow_height is None:
        waterbody_outflow_height = np.zeros(n_wb, dtype=np.float32)
    if waterbody_outflow_bed_elev is None:
        waterbody_outflow_bed_elev = np.zeros(n_wb, dtype=np.float32)

    router = LocalInertial(
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
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        floodplain_width_m=floodplain_width_m,
        waterbody_lake_area=waterbody_lake_area,
        waterbody_lake_factor=waterbody_lake_factor,
        waterbody_outflow_height=waterbody_outflow_height,
        waterbody_outflow_bed_elev=waterbody_outflow_bed_elev,
        river_storage_alpha=river_storage_alpha,
        river_storage_beta=river_storage_beta,
        in_spinup=True,
    )
    router.initialize_stage(
        waterbody_storage_m3=np.zeros(n_wb, dtype=np.float64) if n_wb > 0 else None
    )
    return router


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
        sideflow_m3=np.array([0.0, 20000.0], dtype=np.float32),
        evaporation_m3=evaporation_m3,
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Actual evaporation on cell 0 is constrained by available water (<= 500 m3)
    assert actual_evap[0] <= 500.0 + 1e-4
    assert actual_evap[0] > 0.0
    # Actual evaporation on cell 1 occurs up to demand (<= 1000 m3)
    assert actual_evap[1] <= 1000.0 + 1e-4
    assert actual_evap[1] > 0.0
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
    river_storage_init = np.array([0.0, 0.0, 200000.0], dtype=np.float64)

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
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Both dry tributaries should receive reverse backflow from the high stage at the junction
    assert river_storage_out[0] > 0.0
    assert river_storage_out[1] > 0.0
    # Mass balance holds exactly
    total_after = float(np.sum(river_storage_out)) + float(outflow_at_pits)
    assert np.isclose(200000.0, total_after, rtol=1e-4)


def test_local_inertial_kinematic_pit_missing_river_id() -> None:
    """Test that a kinematic pit cell without a river ID initializes and routes without error."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 2

    # Cell 1 is a pit and kinematic (use_kinematic=True), with river_id=-1
    river_ids = np.array([0, -1], dtype=np.int32)
    use_kinematic = np.array([True, True], dtype=bool)
    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([-1], dtype=np.int32),
            "slope": np.array([0.001], dtype=np.float32),
        },
        index=[0],
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 1000.0, dtype=np.float32),
        river_ids=river_ids,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
    )

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
        river_storage_m3=np.array([1000.0, 500.0], dtype=np.float64),
        sideflow_m3=np.zeros(n_cells, dtype=np.float32),
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.ndarray(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.ndarray(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    assert (river_storage_out >= 0.0).all()
    assert outflow_at_pits >= 0.0


def test_local_inertial_inertial_pit_missing_river_id_raises_keyerror() -> None:
    """Test that an inertial pit cell without a river ID raises a KeyError upon initialization."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 2

    # Cell 1 is an inertial pit (use_kinematic=False), with river_id=-1 (not in rivers_gdf)
    river_ids = np.array([0, -1], dtype=np.int32)
    use_kinematic = np.array([False, False], dtype=bool)
    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([-1], dtype=np.int32),
            "slope": np.array([0.001], dtype=np.float32),
        },
        index=[0],
    )

    with pytest.raises(KeyError, match="Value with index 1 has river ID -1"):
        _make_local_inertial(
            dt=3600,
            river_network=river_network,
            river_length=np.full(n_cells, 1000.0, dtype=np.float32),
            river_width=np.full(n_cells, 10.0, dtype=np.float32),
            river_ids=river_ids,
            use_kinematic=use_kinematic,
            rivers_gdf=rivers_gdf,
            bankfull_river_elevation_m=np.array([10.0, 0.0], dtype=np.float32),
        )


def test_local_inertial_terminal_waterbody_on_pit() -> None:
    """Test that a waterbody located on a pit node initializes and routes correctly."""
    # Cell 0 drains to cell 1, which is a pit and part of a waterbody
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 2

    waterbody_ids = np.array([-1, 0], dtype=np.int32)
    is_waterbody_outflow = np.array([False, True], dtype=bool)
    river_ids = np.array([0, -1], dtype=np.int32)
    use_kinematic = np.array([False, False], dtype=bool)
    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([-1], dtype=np.int32),
            "slope": np.array([0.001], dtype=np.float32),
        },
        index=[0],
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 1000.0, dtype=np.float32),
        river_width=np.full(n_cells, 10.0, dtype=np.float32),
        waterbody_ids=waterbody_ids,
        river_ids=river_ids,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=np.array([10.0, 0.0], dtype=np.float32),
    )

    wb_storage_init = np.array([10000.0], dtype=np.float64)
    wb_outflow = np.array([2000.0], dtype=np.float32)
    river_storage_init = np.array([5000.0, 0.0], dtype=np.float64)

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
        Q_prev_m3_s=np.zeros(n_cells, dtype=np.float32),
        river_storage_m3=river_storage_init.copy(),
        sideflow_m3=np.zeros(n_cells, dtype=np.float32),
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=wb_storage_init.copy(),
        outflow_per_waterbody_m3=wb_outflow,
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Lake storage was reduced by the terminal release of 2000 m3 plus any inflow from upstream reach
    assert wb_storage_out[0] >= 0.0
    # Mass balance for the whole domain must hold
    total_final: float = float(
        np.sum(wb_storage_out) + np.sum(river_storage_out) + outflow_at_pits
    )
    assert np.isclose(total_initial, total_final, atol=1e-3)


def test_compute_cfl_area_and_top_width_and_overbank_depth() -> None:
    """Tests geometry derivation from stored volume for in-bank and overbank regimes."""
    reach_length_m = np.float32(1000.0)
    bankfull_width_m = np.float32(20.0)
    shape_exponent = np.float32(0.5)
    bankfull_depth_m = np.float32(2.0)
    floodplain_width_m = np.float32(80.0)

    (
        inverse_reach_length,
        inverse_shape_exponent_plus_one,
        bankfull_area_arr,
        bankfull_volume_arr,
        bankfull_perimeter_arr,
        floodplain_side_slope_arr,
        floodplain_depth_threshold_arr,
        floodplain_area_threshold_arr,
        sqrt_one_plus_floodplain_slope_squared_arr,
        width_over_sqrt_bankfull_depth_arr,
    ) = compute_static_geometry(
        river_length=np.array([reach_length_m], dtype=np.float32),
        river_width=np.array([bankfull_width_m], dtype=np.float32),
        shape_exponent=np.array([shape_exponent], dtype=np.float32),
        bankfull_depth=np.array([bankfull_depth_m], dtype=np.float32),
        floodplain_width=np.array([floodplain_width_m], dtype=np.float32),
    )
    bankfull_area = bankfull_area_arr[0]
    bankfull_volume = bankfull_volume_arr[0]

    # In-bank test at half bankfull volume (V = 13,333.33 m3)
    # Expected depth: h_bf * (0.5)^(1/1.5) = 2.0 * 0.5^(2/3) ≈ 1.25992 m
    expected_h = 2.0 * (0.5 ** (2.0 / 3.0))
    expected_top_width = 20.0 * ((expected_h / 2.0) ** 0.5)
    area_in, top_width_in = compute_cfl_area_and_top_width(
        volume_m3=np.float32(bankfull_volume * 0.5),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        floodplain_width_m=floodplain_width_m,
        inverse_reach_length=inverse_reach_length[0],
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area,
        bankfull_volume_m3=bankfull_volume,
        floodplain_side_slope=floodplain_side_slope_arr[0],
        floodplain_depth_threshold_m=floodplain_depth_threshold_arr[0],
        floodplain_area_threshold_m2=floodplain_area_threshold_arr[0],
    )
    assert np.isclose(area_in, bankfull_area * 0.5, atol=1e-4)
    assert np.isclose(top_width_in, expected_top_width, atol=1e-4)

    # Overbank test with trapezoidal floodplain widening:
    # W_bf=20.0, h_bf=2.0, W_fp=80.0 -> eff_z_fp = 80 / (2 * 2) = 20.0
    # Overbank volume = 20,000 m3 -> overbank area = 20 m2
    # h_fp = sqrt(20 / 20) = 1.0 m; Top width = 20 + 2 * 20 * 1.0 = 60 m; Area = a_bf + 20 m2
    h_fp_ob = compute_overbank_depth(
        overbank_volume_m3=np.float32(20000.0),
        inverse_reach_length=inverse_reach_length[0],
        floodplain_width_m=floodplain_width_m,
        floodplain_side_slope=floodplain_side_slope_arr[0],
        floodplain_depth_threshold_m=floodplain_depth_threshold_arr[0],
        floodplain_area_threshold_m2=floodplain_area_threshold_arr[0],
    )
    assert np.isclose(h_fp_ob, 1.0, atol=1e-4)

    area_ob, top_width_ob = compute_cfl_area_and_top_width(
        volume_m3=np.float32(bankfull_volume + 20000.0),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        floodplain_width_m=floodplain_width_m,
        inverse_reach_length=inverse_reach_length[0],
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area,
        bankfull_volume_m3=bankfull_volume,
        floodplain_side_slope=floodplain_side_slope_arr[0],
        floodplain_depth_threshold_m=floodplain_depth_threshold_arr[0],
        floodplain_area_threshold_m2=floodplain_area_threshold_arr[0],
    )
    assert np.isclose(area_ob, bankfull_area + 20.0, atol=1e-4)
    assert np.isclose(top_width_ob, 60.0, atol=1e-4)


def test_compute_cross_section_from_depth() -> None:
    """Tests cross-sectional area, width, and perimeter evaluation at interface flow depths."""
    bankfull_width_m = np.float32(20.0)
    shape_exponent = np.float32(0.5)
    bankfull_depth_m = np.float32(2.0)
    floodplain_width_m = np.float32(50.0)

    (
        inverse_reach_length,
        inverse_shape_exponent_plus_one,
        bankfull_area_arr,
        bankfull_volume_arr,
        bankfull_perimeter_arr,
        floodplain_side_slope_arr,
        floodplain_depth_threshold_arr,
        floodplain_area_threshold_arr,
        sqrt_one_plus_floodplain_slope_squared_arr,
        width_over_sqrt_bankfull_depth_arr,
    ) = compute_static_geometry(
        river_length=np.array([1000.0], dtype=np.float32),
        river_width=np.array([bankfull_width_m], dtype=np.float32),
        shape_exponent=np.array([shape_exponent], dtype=np.float32),
        bankfull_depth=np.array([bankfull_depth_m], dtype=np.float32),
        floodplain_width=np.array([floodplain_width_m], dtype=np.float32),
    )

    # In-bank interface depth (1.5 m)
    # Top width = 20 * (1.5 / 2.0)^0.5 = 20 * sqrt(0.75) ≈ 17.3205 m
    # Area = top_width * h / 1.5 = 17.3205 * 1.5 / 1.5 = 17.3205 m2
    area, top_width, p_wetted = compute_cross_section_from_depth(
        effective_depth_m=np.float32(1.5),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        floodplain_width_m=floodplain_width_m,
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area_arr[0],
        bankfull_perimeter_m=bankfull_perimeter_arr[0],
        floodplain_side_slope=floodplain_side_slope_arr[0],
        floodplain_depth_threshold_m=floodplain_depth_threshold_arr[0],
        floodplain_area_threshold_m2=floodplain_area_threshold_arr[0],
        sqrt_one_plus_floodplain_slope_squared=sqrt_one_plus_floodplain_slope_squared_arr[
            0
        ],
        width_over_sqrt_bankfull_depth=width_over_sqrt_bankfull_depth_arr[0],
    )
    expected_w = 20.0 * math.sqrt(0.75)
    assert np.isclose(top_width, expected_w, atol=1e-4)
    assert np.isclose(area, expected_w, atol=1e-4)

    # Overbank interface depth (2.5 m -> 0.5 m overbank)
    # W_bf=20.0, h_bf=2.0, W_fp=50.0 -> eff_z_fp = 50 / 4 = 12.5
    # Overbank top width = 20 + 2 * 12.5 * 0.5 = 32.5 m
    # Overbank area = 12.5 * (0.5^2) = 3.125 m2; Total area = (40.0 / 1.5) + 3.125 m2
    area_ob, top_width_ob, p_wetted_ob = compute_cross_section_from_depth(
        effective_depth_m=np.float32(2.5),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        floodplain_width_m=floodplain_width_m,
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area_arr[0],
        bankfull_perimeter_m=bankfull_perimeter_arr[0],
        floodplain_side_slope=floodplain_side_slope_arr[0],
        floodplain_depth_threshold_m=floodplain_depth_threshold_arr[0],
        floodplain_area_threshold_m2=floodplain_area_threshold_arr[0],
        sqrt_one_plus_floodplain_slope_squared=sqrt_one_plus_floodplain_slope_squared_arr[
            0
        ],
        width_over_sqrt_bankfull_depth=width_over_sqrt_bankfull_depth_arr[0],
    )
    assert np.isclose(area_ob, (40.0 / 1.5) + 3.125, atol=1e-4)
    assert np.isclose(top_width_ob, 32.5, atol=1e-4)


def test_compute_static_geometry_validation() -> None:
    """Tests that compute_static_geometry raises explicit ValueErrors for invalid inputs."""
    valid_length: ArrayFloat32 = np.array([1000.0], dtype=np.float32)
    valid_width: ArrayFloat32 = np.array([20.0], dtype=np.float32)
    valid_shape: ArrayFloat32 = np.array([0.5], dtype=np.float32)
    valid_depth: ArrayFloat32 = np.array([2.0], dtype=np.float32)
    valid_fp_width: ArrayFloat32 = np.array([50.0], dtype=np.float32)

    # Happy path succeeds
    results = compute_static_geometry(
        river_length=valid_length,
        river_width=valid_width,
        shape_exponent=valid_shape,
        bankfull_depth=valid_depth,
        floodplain_width=valid_fp_width,
    )
    assert len(results) == 10

    # Happy path with zero floodplain width
    zero_fp_results = compute_static_geometry(
        river_length=valid_length,
        river_width=valid_width,
        shape_exponent=valid_shape,
        bankfull_depth=valid_depth,
        floodplain_width=np.array([0.0], dtype=np.float32),
    )
    assert zero_fp_results[6][0] == 0.0  # h_fp_max

    # Mismatched shapes
    with pytest.raises(ValueError, match="All input arrays must have the same shape"):
        compute_static_geometry(
            river_length=np.array([1000.0, 500.0], dtype=np.float32),
            river_width=valid_width,
            shape_exponent=valid_shape,
            bankfull_depth=valid_depth,
            floodplain_width=valid_fp_width,
        )

    # Invalid river_length (NaN, inf, <= 0)
    for bad_length in [
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-10.0], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError, match="river_length must contain positive, finite numbers"
        ):
            compute_static_geometry(
                river_length=bad_length,
                river_width=valid_width,
                shape_exponent=valid_shape,
                bankfull_depth=valid_depth,
                floodplain_width=valid_fp_width,
            )

    # Invalid river_width (NaN, <= 0)
    for bad_width in [
        np.array([np.nan], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-5.0], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError, match="river_width must contain positive, finite numbers"
        ):
            compute_static_geometry(
                river_length=valid_length,
                river_width=bad_width,
                shape_exponent=valid_shape,
                bankfull_depth=valid_depth,
                floodplain_width=valid_fp_width,
            )

    # Invalid shape_exponent (NaN, <= 0)
    for bad_shape in [
        np.array([np.nan], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-0.5], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError, match="shape_exponent must contain positive, finite numbers"
        ):
            compute_static_geometry(
                river_length=valid_length,
                river_width=valid_width,
                shape_exponent=bad_shape,
                bankfull_depth=valid_depth,
                floodplain_width=valid_fp_width,
            )

    # Invalid bankfull_depth (NaN, <= 0)
    for bad_depth in [
        np.array([np.nan], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-1.0], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError, match="bankfull_depth must contain positive, finite numbers"
        ):
            compute_static_geometry(
                river_length=valid_length,
                river_width=valid_width,
                shape_exponent=valid_shape,
                bankfull_depth=bad_depth,
                floodplain_width=valid_fp_width,
            )

    # Invalid floodplain_width (NaN, < 0)
    for bad_fp in [
        np.array([np.nan], dtype=np.float32),
        np.array([-0.1], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError,
            match="floodplain_width must contain non-negative, finite numbers",
        ):
            compute_static_geometry(
                river_length=valid_length,
                river_width=valid_width,
                shape_exponent=valid_shape,
                bankfull_depth=valid_depth,
                floodplain_width=bad_fp,
            )


def test_compound_channel_floodplain_attenuation_and_mass_balance() -> None:
    """Tests flood wave attenuation and strict mass conservation in a compound channel network."""
    # 5-cell river channel
    ldd = np.array([[2], [2], [2], [2], [5]], dtype=np.uint8)
    mask = np.ones((5, 1), dtype=bool)
    river_network = pyflwdir.from_array(ldd, ftype="ldd", transform=Affine.identity())
    n_cells: int = 5

    # 1. Base channel without floodplain (narrow rectangular/trapezoidal)
    router_simple = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 2000.0, dtype=np.float32),
        river_width=np.full(n_cells, 15.0, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=10.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
        bankfull_river_elevation_m=np.array(
            [40.0, 30.0, 20.0, 10.0, 0.0], dtype=np.float32
        ),
    )

    # 2. Compound channel with wide floodplain (100 m wide floodplain activated above 1.5 m depth)
    router_compound = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.full(n_cells, 2000.0, dtype=np.float32),
        river_width=np.full(n_cells, 15.0, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=1.5,
        floodplain_width_m=np.full(n_cells, 100.0, dtype=np.float32),
        bankfull_river_elevation_m=np.array(
            [40.0, 30.0, 20.0, 10.0, 0.0], dtype=np.float32
        ),
    )

    # Simulate flood pulse over 10 hours
    n_hours: int = 10
    inflow_pulse = [10.0, 50.0, 150.0, 100.0, 40.0, 20.0, 10.0, 5.0, 5.0, 5.0]

    # Run compound channel
    storage_comp = np.full(n_cells, 5000.0, dtype=np.float64)
    q_prev_comp = np.zeros(n_cells, dtype=np.float32)
    total_sideflow_vol = 0.0
    total_pit_outflow = 0.0
    initial_comp_storage = float(np.sum(storage_comp))
    peak_outflow_compound = 0.0

    for h in range(n_hours):
        sideflow = np.zeros(n_cells, dtype=np.float32)
        sideflow[0] = np.float32(inflow_pulse[h] * 3600.0)
        total_sideflow_vol += float(sideflow[0])

        (
            q_prev_comp,
            storage_comp,
            _,
            _,
            _,
            _,
            pit_out,
            _,
            _,
            _,
        ) = router_compound.step(
            Q_prev_m3_s=q_prev_comp,
            river_storage_m3=storage_comp,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        total_pit_outflow += float(pit_out)
        peak_outflow_compound = max(peak_outflow_compound, float(q_prev_comp[-1]))

    final_comp_storage = float(np.sum(storage_comp))
    # Strict mass conservation check for compound channel
    expected_final_comp = initial_comp_storage + total_sideflow_vol - total_pit_outflow
    assert np.isclose(final_comp_storage, expected_final_comp, atol=1.0)


def test_floodplain_damping_scenarios_and_plots() -> None:
    """Overarching test comparing flood wave attenuation across confined vs moderate vs wide floodplains.

    Simulates a 24-hour flood hydrograph through a 20 km river network (10 reaches, 2 km each)
    and plots hydrographs and storage dynamics for manual review.
    """
    n_cells: int = 10
    reach_length_m: float = 2000.0
    dt: int = 3600

    ldd = np.full((n_cells, 1), 2, dtype=np.uint8)
    ldd[-1, 0] = 5
    mask = np.ones((n_cells, 1), dtype=bool)
    river_network = pyflwdir.from_array(ldd, ftype="ldd", transform=Affine.identity())
    river_ids = np.arange(n_cells, dtype=np.int32)
    bed_elevation = np.linspace(20.0, 2.0, n_cells, dtype=np.float32)
    slope_actual = float(
        (bed_elevation[0] - bed_elevation[-1]) / ((n_cells - 1) * reach_length_m)
    )
    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.concatenate(
                [np.arange(1, n_cells, dtype=np.int32), [-1]]
            ),
            "slope": np.full(n_cells, slope_actual, dtype=np.float32),
        },
        index=river_ids,
    )

    manning_n = np.full(n_cells, 0.035, dtype=np.float32)
    river_length = np.full(n_cells, reach_length_m, dtype=np.float32)
    river_width = np.full(n_cells, 20.0, dtype=np.float32)

    # 1. Confined / In-bank only (no floodplain)
    router_confined = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
        manning_n=manning_n,
        shape_exponent=0.5,
        bankfull_depth_m=20.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    # 2. Moderate Floodplain (100 m wide floodplain activated above 1.5 m depth)
    router_moderate = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
        manning_n=manning_n,
        shape_exponent=0.5,
        bankfull_depth_m=1.5,
        floodplain_width_m=np.full(n_cells, 100.0, dtype=np.float32),
    )

    # 3. Wide Floodplain (300 m wide floodplain activated above 1.5 m depth)
    router_wide = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
        manning_n=manning_n,
        shape_exponent=0.5,
        bankfull_depth_m=1.5,
        floodplain_width_m=np.full(n_cells, 300.0, dtype=np.float32),
    )

    n_hours: int = 24
    hours = np.arange(n_hours)
    # Triangular flood hydrograph: baseflow 10 m3/s, peaking at 200 m3/s at hour 6, returning to baseflow by hour 14
    inflow_m3_s = np.full(n_hours, 10.0, dtype=np.float32)
    for h in range(1, 7):
        inflow_m3_s[h] = 10.0 + (200.0 - 10.0) * (h / 6.0)
    for h in range(7, 15):
        inflow_m3_s[h] = 200.0 - (200.0 - 10.0) * ((h - 6) / 8.0)

    def simulate(
        router: LocalInertial,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q_outlet = []
        q_midpoint = []
        total_storage = []
        peak_stages = np.zeros(n_cells, dtype=np.float32)

        storage = np.full(n_cells, 20000.0, dtype=np.float64)
        q = np.full(n_cells, 10.0, dtype=np.float32)
        total_inflow_vol = 0.0
        total_outflow_vol = 0.0
        init_storage = float(np.sum(storage))

        for h in range(n_hours):
            sideflow = np.zeros(n_cells, dtype=np.float32)
            sideflow[0] = np.float32(inflow_m3_s[h] * dt)
            total_inflow_vol += float(sideflow[0])

            (
                q,
                storage,
                _,
                _,
                _,
                _,
                pit_out,
                _,
                _,
                _,
            ) = router.step(
                Q_prev_m3_s=q,
                river_storage_m3=storage,
                sideflow_m3=sideflow,
                evaporation_m3=np.zeros(n_cells, dtype=np.float32),
                waterbody_storage_m3=np.zeros(0, dtype=np.float64),
                outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
                retention_storage_m3=np.zeros(0, dtype=np.float32),
                retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
            )
            total_outflow_vol += float(pit_out)
            q_outlet.append(float(q[-1]))
            q_midpoint.append(float(q[n_cells // 2]))
            total_storage.append(float(np.sum(storage)))

            current_stages = router.get_water_stage()
            peak_stages = np.maximum(peak_stages, current_stages)

        # Mass balance verification
        final_storage = float(np.sum(storage))
        assert np.isclose(
            final_storage, init_storage + total_inflow_vol - total_outflow_vol, atol=2.0
        )

        return (
            np.array(q_outlet),
            np.array(q_midpoint),
            np.array(total_storage),
            peak_stages,
        )

    q_out_conf, q_mid_conf, s_conf, stages_conf = simulate(router_confined)
    q_out_mod, q_mid_mod, s_mod, stages_mod = simulate(router_moderate)
    q_out_wide, q_mid_wide, s_wide, stages_wide = simulate(router_wide)

    # Damping assertions:
    # 1. Floodplain slows and attenuates peak discharge at the catchment outlet
    assert np.max(q_out_wide) < np.max(q_out_mod) < np.max(q_out_conf)
    # 2. Time to peak is delayed as floodplain width increases
    assert np.argmax(q_out_wide) >= np.argmax(q_out_mod) >= np.argmax(q_out_conf)
    # 3. Peak volume stored in the channel network is higher for wider floodplains
    assert np.max(s_wide) > np.max(s_mod) > np.max(s_conf)

    # Generate diagnostic plot for manual review
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    # Panel 1: Catchment Outlet Hydrograph
    axes[0].plot(hours, inflow_m3_s, "k--", label="Inflow Hydrograph (Reach 0)")
    axes[0].plot(hours, q_out_conf, "r-", label="Outlet Q (Confined Channel)")
    axes[0].plot(hours, q_out_mod, "b-", label="Outlet Q (Moderate Floodplain W=100m)")
    axes[0].plot(hours, q_out_wide, "g-", label="Outlet Q (Wide Floodplain W=300m)")
    axes[0].set_title("Catchment Outlet Discharge Hydrographs")
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("Discharge (m³/s)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Panel 2: Total Channel Storage Dynamics
    axes[1].plot(hours, s_conf / 1e3, "r-", label="Confined Channel Storage")
    axes[1].plot(hours, s_mod / 1e3, "b-", label="Moderate Floodplain Storage (W=100m)")
    axes[1].plot(hours, s_wide / 1e3, "g-", label="Wide Floodplain Storage (W=300m)")
    axes[1].set_title("Total River Network Storage Volume")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Stored Volume (1000 m³)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Panel 3: Longitudinal Water Stage Profile at Peak
    reach_dist_km = np.arange(n_cells) * (reach_length_m / 1000.0)
    axes[2].plot(reach_dist_km, bed_elevation, "k-", label="River Bed Elevation")
    axes[2].plot(reach_dist_km, stages_conf, "r--o", label="Peak Stage (Confined)")
    axes[2].plot(
        reach_dist_km, stages_mod, "b--s", label="Peak Stage (Moderate FP W=100m)"
    )
    axes[2].plot(
        reach_dist_km, stages_wide, "g--^", label="Peak Stage (Wide FP W=300m)"
    )
    axes[2].set_title("Longitudinal Peak Water Stage Profile Along River Network")
    axes[2].set_xlabel("Downstream Distance (km)")
    axes[2].set_ylabel("Elevation above Datum (m)")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    fig_path = output_folder / "test_floodplain_attenuation_scenarios.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    assert fig_path.exists(), f"Expected plot file at {fig_path}"


def test_floodplain_convective_pulse_dynamics_and_plots() -> None:
    """Overarching test evaluating sharp convective storm pulses with floodplain attenuation.

    Simulates a sudden 400 m3/s storm pulse into middle reaches and saves longitudinal
    wave dynamics plots for manual review.
    """
    n_cells: int = 12
    reach_length_m: float = 1500.0
    dt: int = 1800  # 30-min time step

    ldd = np.full((n_cells, 1), 2, dtype=np.uint8)
    ldd[-1, 0] = 5
    mask = np.ones((n_cells, 1), dtype=bool)
    river_network = pyflwdir.from_array(ldd, ftype="ldd", transform=Affine.identity())
    river_ids = np.arange(n_cells, dtype=np.int32)
    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.full(n_cells, -1, dtype=np.int32),
            "slope": np.full(n_cells, 0.0008, dtype=np.float32),
        },
        index=river_ids,
    )

    bed_elevation = np.linspace(30.0, 3.0, n_cells, dtype=np.float32)
    manning_n = np.full(n_cells, 0.04, dtype=np.float32)
    river_length = np.full(n_cells, reach_length_m, dtype=np.float32)
    river_width = np.full(n_cells, 25.0, dtype=np.float32)

    router_confined = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
        manning_n=manning_n,
        shape_exponent=0.5,
        bankfull_depth_m=15.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    router_floodplain = _make_local_inertial(
        dt=dt,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
        manning_n=manning_n,
        shape_exponent=0.5,
        bankfull_depth_m=1.8,
        floodplain_width_m=np.full(n_cells, 250.0, dtype=np.float32),
    )

    n_steps: int = 30
    time_hours = np.arange(n_steps) * (dt / 3600.0)

    # Convective pulse entering reach 3 at t = 1.0 to 2.5 hours
    pulse_q = np.zeros(n_steps, dtype=np.float32)
    pulse_q[2:5] = 350.0

    def run_pulse(router: LocalInertial) -> tuple[np.ndarray, np.ndarray]:
        outlet_q = []
        reach6_q = []
        storage = np.full(n_cells, 15000.0, dtype=np.float64)
        q = np.full(n_cells, 5.0, dtype=np.float32)

        for step_idx in range(n_steps):
            sideflow = np.zeros(n_cells, dtype=np.float32)
            sideflow[3] = np.float32(pulse_q[step_idx] * dt)

            (
                q,
                storage,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = router.step(
                Q_prev_m3_s=q,
                river_storage_m3=storage,
                sideflow_m3=sideflow,
                evaporation_m3=np.zeros(n_cells, dtype=np.float32),
                waterbody_storage_m3=np.zeros(0, dtype=np.float64),
                outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
                retention_storage_m3=np.zeros(0, dtype=np.float32),
                retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
            )
            outlet_q.append(float(q[-1]))
            reach6_q.append(float(q[6]))

        return np.array(outlet_q), np.array(reach6_q)

    out_q_conf, r6_q_conf = run_pulse(router_confined)
    out_q_fp, r6_q_fp = run_pulse(router_floodplain)

    assert np.max(out_q_fp) < np.max(out_q_conf)
    assert np.max(r6_q_fp) < np.max(r6_q_conf)

    # Plot pulse wave propagation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time_hours, pulse_q, "k:", label="Storm Pulse Inflow (Reach 3)")
    ax1.plot(time_hours, r6_q_conf, "r-", label="Reach 6 Discharge (Confined Channel)")
    ax1.plot(time_hours, r6_q_fp, "b-", label="Reach 6 Discharge (Compound Floodplain)")
    ax1.set_title("Midstream (Reach 6) Response to Upstream Convective Storm Pulse")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Discharge (m³/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(time_hours, out_q_conf, "r-", label="Outlet Discharge (Confined Channel)")
    ax2.plot(time_hours, out_q_fp, "b-", label="Outlet Discharge (Compound Floodplain)")
    ax2.set_title("Catchment Outlet Flood Wave Attenuation")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Discharge (m³/s)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    fig_path = output_folder / "test_floodplain_pulse_wave_dynamics.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    assert fig_path.exists(), f"Expected plot file at {fig_path}"


def test_local_inertial_ocean_pit_negative_bed_elevation() -> None:
    """Test that an ocean pit with bed elevation below datum discharges water freely without trapping.

    When channel bed is below sea level datum (e.g. -2.0 m in delta reaches), water entering
    the reach must be able to drain into the sea rather than having discharge locked at 0.0 m³/s.
    """
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 2

    # Reach 0 is at -1.0 m, reach 1 is a pit at -2.0 m (below sea level)
    bed_elevation = np.array([-1.0, -2.0], dtype=np.float32)
    river_length = np.array([1000.0, 1000.0], dtype=np.float32)
    river_width = np.array([20.0, 20.0], dtype=np.float32)
    river_ids = np.array([0, 1], dtype=np.int32)
    use_kinematic = np.array([False, False], dtype=bool)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1], dtype=np.int32),
            "slope": np.array([0.001, 0.001], dtype=np.float32),
        },
        index=river_ids,
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        river_ids=river_ids,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )

    assert router._is_ocean_pit[router.inv_idxs[1]], (
        "Reach 1 must be classified as an ocean pit (downstream_ID == -1)."
    )

    # Initial storage in reaches corresponds to 1.5 m of water depth
    init_storage = np.array([30000.0, 30000.0], dtype=np.float64)
    # Add inflow to reach 0
    sideflow = np.array([18000.0, 0.0], dtype=np.float32)

    (
        q_out,
        storage_out,
        actual_evap,
        over_abs,
        wb_storage,
        wb_inflow,
        outflow_at_pits,
        ret_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=np.array([5.0, 5.0], dtype=np.float32),
        river_storage_m3=init_storage.copy(),
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Verify that pit discharges water to the ocean (not locked at 0)
    assert q_out[1] > 0.0, f"Expected positive pit discharge, got {q_out[1]}"
    assert outflow_at_pits > 0.0, (
        f"Expected positive outflow at pits, got {outflow_at_pits}"
    )

    # Verify domain mass conservation
    total_in = float(np.sum(init_storage) + np.sum(sideflow))
    total_out = float(np.sum(storage_out) + outflow_at_pits)
    assert np.isclose(total_in, total_out, rtol=1e-4)


def test_local_inertial_inland_pit_with_downstream_id() -> None:
    """Test that an inland basin outlet with downstream_ID != -1 is not treated as an ocean pit.

    An inland basin outlet that drains to a downstream river outside the local domain
    has downstream_ID != -1 (e.g. 2). It must use normal depth / free drainage slope
    (from pit_slope) rather than dropping to 0.0 m datum (which creates an artificial waterfall).
    """
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 2

    # High elevation inland basin (e.g. 350 m above sea level)
    bed_elevation = np.array([360.0, 350.0], dtype=np.float32)
    river_length = np.array([1000.0, 1000.0], dtype=np.float32)
    river_width = np.array([15.0, 15.0], dtype=np.float32)
    river_ids = np.array([0, 1], dtype=np.int32)
    use_kinematic = np.array([False, False], dtype=bool)

    # Outlet has downstream_ID = 2 (points to river outside domain) in vector network
    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, 2, -1], dtype=np.int32),
            "slope": np.array([0.005, 0.005, 0.005], dtype=np.float32),
        },
        index=np.array([0, 1, 2], dtype=np.int32),
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        river_ids=river_ids,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )

    # Pit with downstream_ID != -1 must NOT be an ocean pit
    assert not router._is_ocean_pit[router.inv_idxs[1]], (
        "Inland pit with downstream_ID != -1 must not be classified as ocean pit."
    )
    assert router._pit_slope[router.inv_idxs[1]] == np.float32(0.005)

    init_storage = np.array([15000.0, 15000.0], dtype=np.float64)
    sideflow = np.array([36000.0, 0.0], dtype=np.float32)

    (
        q_out,
        storage_out,
        actual_evap,
        over_abs,
        wb_storage,
        wb_inflow,
        outflow_at_pits,
        ret_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=np.array([10.0, 10.0], dtype=np.float32),
        river_storage_m3=init_storage.copy(),
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Discharge should be reasonable (not astronomical critical waterfall discharge)
    assert q_out[1] > 0.0
    assert q_out[1] < 1000.0, (
        f"Discharge {q_out[1]} is unreasonably large for 10 m3/s baseflow."
    )

    # Mass balance holds
    total_in = float(np.sum(init_storage) + np.sum(sideflow))
    total_out = float(np.sum(storage_out) + outflow_at_pits)
    assert np.isclose(total_in, total_out, rtol=1e-4)


def test_local_inertial_waterbody_boundary_uses_lake_water_level() -> None:
    """Test that an inertial reach entering a waterbody feels dynamic lake water level.

    When the lake water level is high (flooded), the backwater effect reduces the inflow gradient
    into the lake compared to when the lake is drawn down to its bed elevation.
    """
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 2

    # Reach 0 enters reach 1, which is a waterbody
    waterbody_ids = np.array([-1, 0], dtype=np.int32)
    is_waterbody_outflow = np.array([False, True], dtype=bool)
    river_ids = np.array([0, -1], dtype=np.int32)
    use_kinematic = np.array([False, False], dtype=bool)
    bed_elevation = np.array([10.0, 5.0], dtype=np.float32)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([-1], dtype=np.int32),
            "slope": np.array([0.005], dtype=np.float32),
        },
        index=[0],
    )

    lake_area = np.array([100000.0], dtype=np.float32)  # 100,000 m2
    outflow_height = np.array([5.0], dtype=np.float32)  # 5 m outflow sill
    outflow_bed_elev = np.array([5.0], dtype=np.float32)

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=np.array([1000.0, 1000.0], dtype=np.float32),
        river_width=np.array([10.0, 10.0], dtype=np.float32),
        waterbody_ids=waterbody_ids,
        river_ids=river_ids,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
        waterbody_lake_area=lake_area,
        waterbody_outflow_height=outflow_height,
        waterbody_outflow_bed_elev=outflow_bed_elev,
    )

    # Scenario A: Lake is low (empty, storage = 0 m3)
    # Lake stage is at bed elevation 5.0 - 5.0 = 0.0 m (well below river bed 10.0 m)
    river_storage = np.array([10000.0, 0.0], dtype=np.float64)
    wb_storage_low = np.array([0.0], dtype=np.float64)

    (
        q_low_arr,
        _,
        _,
        _,
        _,
        inflow_low,
        _,
        _,
        _,
        _,
    ) = router.step(
        Q_prev_m3_s=np.array([2.0, 0.0], dtype=np.float32),
        river_storage_m3=river_storage.copy(),
        sideflow_m3=np.zeros(n_cells, dtype=np.float32),
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=wb_storage_low.copy(),
        outflow_per_waterbody_m3=np.zeros(1, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    q_low = q_low_arr.copy()

    # Scenario B: Lake is flooded (high storage = 1,000,000 m3)
    # Lake stage is 5.0 - 5.0 + 10.0 = 10.0 m (matches river bed level, creating strong backwater)
    wb_storage_high = np.array([1000000.0], dtype=np.float64)

    (
        q_high,
        _,
        _,
        _,
        _,
        inflow_high,
        _,
        _,
        _,
        _,
    ) = router.step(
        Q_prev_m3_s=np.array([2.0, 0.0], dtype=np.float32),
        river_storage_m3=river_storage.copy(),
        sideflow_m3=np.zeros(n_cells, dtype=np.float32),
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=wb_storage_high.copy(),
        outflow_per_waterbody_m3=np.zeros(1, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # When the lake is flooded, backwater slows down channel discharge into the lake
    assert q_high[0] < q_low[0], (
        f"High lake stage should induce backwater and lower discharge: {q_high[0]} vs {q_low[0]}"
    )
    assert q_high[0] >= 0.0


def test_local_inertial_cfl_under_sudden_upstream_flood() -> None:
    """Test that a sudden large flood wave passing through inertial reaches maintains CFL stability without exploding."""
    # Linear network: 0 -> 1 -> 2 (pit)
    ldd = np.array([[2], [2], [5]], dtype=np.uint8)
    mask = np.ones((3, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 3

    bed_elevation = np.array([20.0, 10.0, 0.0], dtype=np.float32)
    river_length = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
    river_width = np.array([25.0, 25.0, 25.0], dtype=np.float32)
    floodplain_width = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    river_ids = np.arange(n_cells, dtype=np.int32)
    use_kinematic = np.zeros(n_cells, dtype=bool)

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        floodplain_width_m=floodplain_width,
        river_ids=river_ids,
        use_kinematic=use_kinematic,
        bankfull_river_elevation_m=bed_elevation,
    )

    # Reach 0 initially has a massive flood discharge (1,500 m3/s) while reaches 1 and 2 are at baseflow (5 m3/s)
    q_init = np.array([1500.0, 5.0, 5.0], dtype=np.float32)
    storage_init = np.array([2000000.0, 20000.0, 20000.0], dtype=np.float64)

    (
        q_out,
        storage_out,
        actual_evap,
        over_abs,
        wb_storage,
        wb_inflow,
        outflow_at_pits,
        ret_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=q_init,
        river_storage_m3=storage_init.copy(),
        sideflow_m3=np.zeros(n_cells, dtype=np.float32),
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Verify no exploding discharge or numerical oscillations
    assert np.all(np.isfinite(q_out)), "All discharges must be finite."
    assert np.max(q_out) < 400000.0, (
        f"Discharge exceeded upper threshold: {np.max(q_out)}"
    )
    assert np.all(storage_out >= 0.0), "Storage must remain non-negative."

    # Mass balance holds
    total_in = float(np.sum(storage_init))
    total_out = float(np.sum(storage_out) + outflow_at_pits)
    assert np.isclose(total_in, total_out, rtol=1e-4)


def test_local_inertial_zero_flow_startup_stability() -> None:
    """Test that dry/zero-flow startup under steep head gradient does not experience zero-friction shock overshoot."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())
    n_cells: int = 2

    bed_elevation = np.array([20.0, 0.0], dtype=np.float32)  # Steep gradient (2%)
    river_length = np.array([1000.0, 1000.0], dtype=np.float32)
    river_width = np.array([10.0, 10.0], dtype=np.float32)
    river_ids = np.arange(n_cells, dtype=np.int32)
    use_kinematic = np.zeros(n_cells, dtype=bool)

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        river_ids=river_ids,
        use_kinematic=use_kinematic,
        bankfull_river_elevation_m=bed_elevation,
    )

    # Start with near-dry reaches and zero previous discharge
    q_init = np.zeros(n_cells, dtype=np.float32)
    storage_init = np.array([500.0, 500.0], dtype=np.float64)  # 5 cm depth
    # Apply a modest sideflow
    sideflow = np.array([3600.0, 0.0], dtype=np.float32)  # 1 m3/s average

    (
        q_out,
        storage_out,
        _,
        _,
        _,
        _,
        outflow_at_pits,
        _,
        _,
        _,
    ) = router.step(
        Q_prev_m3_s=q_init,
        river_storage_m3=storage_init.copy(),
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Discharge should be smooth and not overshoot to crazy values
    assert np.all(np.isfinite(q_out))
    assert q_out[0] < 50.0, f"Discharge overshoot on dry start: {q_out[0]}"
    assert q_out[0] > 0.0
    assert np.all(storage_out >= 0.0)


def test_local_inertial_steep_drop_into_waterbody() -> None:
    """Test that an inertial reach with a steep drop into a lake routes successfully without error."""
    ldd = np.array([[2], [5]], dtype=np.uint8)
    mask = np.ones((2, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    # Reach 0 has bed elevation = 100 m, lake at reach 1 has bed elevation = 10 m (90 m drop over 1000 m, slope = 0.09)
    bed_elevation = np.array([100.0, 10.0], dtype=np.float32)
    river_length = np.array([1000.0, 1000.0], dtype=np.float32)
    river_width = np.array([10.0, 10.0], dtype=np.float32)
    waterbody_ids = np.array([-1, 0], dtype=np.int32)
    is_waterbody_outflow = np.array([False, True], dtype=bool)
    river_ids = np.array([0, -1], dtype=np.int32)
    use_kinematic = np.array([False, False], dtype=bool)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([-1], dtype=np.int32),
            "slope": np.array([0.09], dtype=np.float32),
        },
        index=[0],
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        waterbody_ids=waterbody_ids,
        river_ids=river_ids,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )
    assert router.n_inertial == 1


def test_kinematic_overland_flowing_into_downstream_inertial_reach() -> None:
    """Test that an upstream kinematic overland cell flows seamlessly into a downstream inertial river reach.

    Verifies hydraulic coupling, lateral flux accumulation, and exact mass conservation
    from overland kinematic wave routing into the local inertial river channel.
    """
    n_cells = 3
    # Linear chain of 3 reaches: 0 (overland) -> 1 (river reach) -> 2 (river pit)
    # Reach 0: overland cell -> Kinematic (river_id == -1, use_kinematic == True)
    # Reach 1: river reach -> Inertial (river_id == 1, use_kinematic == False)
    # Reach 2: river pit -> Inertial (river_id == 2, use_kinematic == False)
    ldd = np.array([[2], [2], [5]], dtype=np.uint8)
    mask = np.ones((3, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    bed_elevation = np.array([150.0, 100.0, 0.0], dtype=np.float32)
    river_length = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
    river_width = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    use_kinematic = np.array([True, False, False], dtype=bool)
    river_ids = np.array([-1, 1, 2], dtype=np.int32)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([2, -1], dtype=np.int32),
            "slope": np.array([0.05, 0.01], dtype=np.float32),
        },
        index=[1, 2],
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        river_ids=river_ids,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )

    storage = np.zeros(n_cells, dtype=np.float64)
    discharge = np.zeros(n_cells, dtype=np.float32)
    waterbody_storage = np.zeros(0, dtype=np.float64)
    retention_storage = np.zeros(0, dtype=np.float32)

    # Apply overland runoff (sideflow) to cell 0 (kinematic overland)
    sideflow = np.array([100.0 * 3600.0, 0.0, 0.0], dtype=np.float32)
    evaporation = np.zeros(n_cells, dtype=np.float32)
    alpha = np.full(n_cells, 1.0, dtype=np.float32)
    beta = np.full(n_cells, 0.6, dtype=np.float32)

    # Route for 5 hours
    for step in range(5):
        (
            discharge,
            storage,
            act_evap,
            over_abs,
            waterbody_storage,
            wb_inflow,
            outflow_pits,
            retention_storage,
            ret_inflow,
            ret_outflow,
        ) = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=evaporation,
            waterbody_storage_m3=waterbody_storage,
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=retention_storage,
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(discharge >= 0.0)
        assert np.all(storage >= 0.0)


def test_alternating_kinematic_inertial_network() -> None:
    """Test complex mixed topology: Kinematic -> Inertial -> Kinematic -> Inertial."""
    n_cells = 4
    ldd = np.array([[2], [2], [2], [5]], dtype=np.uint8)
    mask = np.ones((4, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    bed_elevation = np.array([300.0, 200.0, 100.0, 0.0], dtype=np.float32)
    river_length = np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32)
    river_width = np.array([15.0, 25.0, 15.0, 30.0], dtype=np.float32)
    use_kinematic = np.array([True, True, False, False], dtype=bool)
    river_ids = np.array([-1, -1, 2, 3], dtype=np.int32)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([3, -1], dtype=np.int32),
            "slope": np.array([0.05, 0.001], dtype=np.float32),
        },
        index=[2, 3],
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        river_ids=river_ids,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )

    storage = np.zeros(n_cells, dtype=np.float64)
    discharge = np.zeros(n_cells, dtype=np.float32)
    waterbody_storage = np.zeros(0, dtype=np.float64)
    retention_storage = np.zeros(0, dtype=np.float32)

    sideflow = np.array(
        [50.0 * 3600.0, 10.0 * 3600.0, 20.0 * 3600.0, 0.0], dtype=np.float32
    )
    evaporation = np.zeros(n_cells, dtype=np.float32)
    alpha = np.full(n_cells, 1.0, dtype=np.float32)
    beta = np.full(n_cells, 0.6, dtype=np.float32)

    for step in range(5):
        (
            discharge,
            storage,
            act_evap,
            over_abs,
            waterbody_storage,
            wb_inflow,
            outflow_pits,
            retention_storage,
            ret_inflow,
            ret_outflow,
        ) = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=evaporation,
            waterbody_storage_m3=waterbody_storage,
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=retention_storage,
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(discharge >= 0.0)


def test_local_inertial_unobserved_river_width_and_dynamic_width_update() -> None:
    """Test local inertial routing on channels without observed widths and dynamic width updates."""
    n_cells = 3
    ldd = np.array([[2], [2], [5]], dtype=np.uint8)
    mask = np.ones((3, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    bed_elevation = np.array([50.0, 40.0, 0.0], dtype=np.float32)
    river_length = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
    # Start with default initial width
    river_width = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    # All reaches use inertial routing
    use_kinematic = np.array([False, False, False], dtype=bool)
    river_ids = np.array([0, 1, 2], dtype=np.int32)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, 2, -1], dtype=np.int32),
            "slope": np.array([0.002, 0.002, 0.002], dtype=np.float32),
        },
        index=river_ids,
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        river_ids=river_ids,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )

    storage = np.zeros(n_cells, dtype=np.float64)
    discharge = np.zeros(n_cells, dtype=np.float32)
    waterbody_storage = np.zeros(0, dtype=np.float64)
    retention_storage = np.zeros(0, dtype=np.float32)

    sideflow = np.array([20.0 * 3600.0, 10.0 * 3600.0, 0.0], dtype=np.float32)
    evaporation = np.zeros(n_cells, dtype=np.float32)
    alpha = np.full(n_cells, 1.0, dtype=np.float32)
    beta = np.full(n_cells, 0.6, dtype=np.float32)

    # Initial steps
    for step in range(3):
        (
            discharge,
            storage,
            act_evap,
            over_abs,
            waterbody_storage,
            wb_inflow,
            outflow_pits,
            retention_storage,
            ret_inflow,
            ret_outflow,
        ) = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=evaporation,
            waterbody_storage_m3=waterbody_storage,
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=retention_storage,
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )

    # Dynamically update river width as discharge develops (e.g. from 0.5m to 12.0m)
    updated_widths = np.array([8.0, 12.0, 15.0], dtype=np.float32)
    router.update_channel_width(river_width=updated_widths)

    np.testing.assert_array_almost_equal(
        router._river_width, updated_widths[router.sorted_idxs]
    )

    # Run subsequent steps with updated channel geometry
    for step in range(3):
        (
            discharge,
            storage,
            act_evap,
            over_abs,
            waterbody_storage,
            wb_inflow,
            outflow_pits,
            retention_storage,
            ret_inflow,
            ret_outflow,
        ) = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=evaporation,
            waterbody_storage_m3=waterbody_storage,
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=retention_storage,
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(discharge >= 0.0)


def test_local_inertial_dynamic_lake_outflow_substepping() -> None:
    """Test dynamic lake outflow calculation integrated directly within local inertial sub-steps."""
    # Reach 0 (upstream river) -> Reach 1 (lake outlet) -> Reach 2 (downstream river pit)
    n_cells = 3
    ldd = np.array([[2], [2], [5]], dtype=np.uint8)
    mask = np.ones((3, 1), dtype=bool)
    river_network = create_river_network(ldd, mask, transform=Affine.identity())

    bed_elevation = np.array([20.0, 10.0, 5.0], dtype=np.float32)
    river_length = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
    river_width = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    waterbody_ids = np.array([-1, 0, -1], dtype=np.int32)
    is_waterbody_outflow = np.array([False, True, False], dtype=bool)
    river_ids = np.array([0, 1, 2], dtype=np.int32)
    use_kinematic = np.array([False, False, False], dtype=bool)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, 2, -1], dtype=np.int32),
            "slope": np.array([0.01, 0.005, 0.005], dtype=np.float32),
        },
        index=river_ids,
    )

    lake_area = np.array([100_000.0], dtype=np.float32)
    lake_factor = np.array([25.0], dtype=np.float32)
    outflow_height = np.array([5.0], dtype=np.float32)
    outflow_bed_elev = np.array([10.0], dtype=np.float32)

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        waterbody_ids=waterbody_ids,
        river_ids=river_ids,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
        waterbody_lake_area=lake_area,
        waterbody_lake_factor=lake_factor,
        waterbody_outflow_height=outflow_height,
        waterbody_outflow_bed_elev=outflow_bed_elev,
    )

    # Initial lake storage has water 10m deep (5m above outflow sill)
    initial_lake_storage = float(100_000.0 * 10.0)
    waterbody_storage = np.array([initial_lake_storage], dtype=np.float64)
    storage = np.zeros(n_cells, dtype=np.float64)
    discharge = np.zeros(n_cells, dtype=np.float32)
    sideflow = np.zeros(n_cells, dtype=np.float32)
    evaporation = np.zeros(n_cells, dtype=np.float32)
    retention_storage = np.zeros(0, dtype=np.float32)
    alpha = np.full(n_cells, 1.0, dtype=np.float32)
    beta = np.full(n_cells, 0.6, dtype=np.float32)

    # Outflow for lake is NaN to trigger dynamic calculation
    outflow_per_waterbody = np.array([np.nan], dtype=np.float32)

    (
        discharge,
        storage,
        act_evap,
        over_abs,
        waterbody_storage,
        wb_inflow,
        outflow_pits,
        retention_storage,
        ret_inflow,
        ret_outflow,
    ) = router.step(
        Q_prev_m3_s=discharge,
        river_storage_m3=storage,
        sideflow_m3=sideflow,
        evaporation_m3=evaporation,
        waterbody_storage_m3=waterbody_storage,
        outflow_per_waterbody_m3=outflow_per_waterbody,
        retention_storage_m3=retention_storage,
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    # Dynamic lake outflow must have reduced storage and flowed into reach 2
    assert waterbody_storage[0] < initial_lake_storage
    assert discharge[2] > 0.0
    assert outflow_pits > 0.0


def test_local_inertial_already_permuted_equivalence() -> None:
    """Verifies that already_permuted=True produces identical outputs to already_permuted=False."""
    n_cells: int = 3
    ldd = np.array([[8, 8, 5]], dtype=np.uint8)
    river_network = pyflwdir.from_array(ldd, ftype="ldd", transform=Affine.identity())

    bed_elevation = np.array([20.0, 10.0, 5.0], dtype=np.float32)
    river_length = np.array([500.0, 500.0, 500.0], dtype=np.float32)
    river_width = np.array([10.0, 20.0, 20.0], dtype=np.float32)
    waterbody_ids = np.array([-1, -1, -1], dtype=np.int32)
    is_waterbody_outflow = np.array([False, False, False], dtype=bool)
    river_ids = np.array([101, 102, 103], dtype=np.int32)
    use_kinematic = np.array([False, False, False], dtype=bool)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([102, 103, -1], dtype=np.int32),
            "slope": np.array([0.02, 0.01, 0.005], dtype=np.float32),
        },
        index=river_ids,
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        waterbody_ids=waterbody_ids,
        river_ids=river_ids,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )

    storage = np.array([100.0, 200.0, 150.0], dtype=np.float64)
    discharge = np.array([5.0, 10.0, 8.0], dtype=np.float32)
    sideflow = np.array([50.0, 100.0, 20.0], dtype=np.float32)
    evaporation = np.array([1.0, 2.0, 1.5], dtype=np.float32)
    wb_storage = np.zeros(0, dtype=np.float64)
    wb_outflow = np.zeros(0, dtype=np.float32)
    ret_storage = np.zeros(0, dtype=np.float32)
    alpha = np.full(n_cells, 1.0, dtype=np.float32)
    beta = np.full(n_cells, 0.6, dtype=np.float32)
    ret_thresh = np.zeros(0, dtype=np.float32)

    # Standard call (already_permuted=False)
    out_std = router.step(
        Q_prev_m3_s=discharge.copy(),
        river_storage_m3=storage.copy(),
        sideflow_m3=sideflow.copy(),
        evaporation_m3=evaporation.copy(),
        waterbody_storage_m3=wb_storage.copy(),
        outflow_per_waterbody_m3=wb_outflow.copy(),
        retention_storage_m3=ret_storage.copy(),
        retention_activation_threshold_m3_s=ret_thresh.copy(),
        already_permuted=False,
    )

    # Permute inputs manually
    p_idxs = router.sorted_idxs
    discharge_perm = discharge[p_idxs].copy()
    storage_perm = storage[p_idxs].copy()
    sideflow_perm = sideflow[p_idxs].copy()
    evaporation_perm = evaporation[p_idxs].copy()
    alpha_perm = alpha[p_idxs].copy()
    beta_perm = beta[p_idxs].copy()

    router_perm = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        waterbody_ids=waterbody_ids,
        river_ids=river_ids,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )

    # Optimized call (already_permuted=True)
    out_perm = router_perm.step(
        Q_prev_m3_s=discharge_perm,
        river_storage_m3=storage_perm,
        sideflow_m3=sideflow_perm,
        evaporation_m3=evaporation_perm,
        waterbody_storage_m3=wb_storage.copy(),
        outflow_per_waterbody_m3=wb_outflow.copy(),
        retention_storage_m3=ret_storage.copy(),
        retention_activation_threshold_m3_s=ret_thresh.copy(),
        already_permuted=True,
    )

    # Check discharge: out_std[0] is in grid order, out_perm[0] is in permuted order
    np.testing.assert_allclose(out_std[0][p_idxs], out_perm[0], rtol=1e-5, atol=1e-5)
    # Check river storage
    np.testing.assert_allclose(out_std[1][p_idxs], out_perm[1], rtol=1e-5, atol=1e-5)
    # Check evaporation
    np.testing.assert_allclose(out_std[2][p_idxs], out_perm[2], rtol=1e-5, atol=1e-5)
    # Check over abstraction
    np.testing.assert_allclose(out_std[3][p_idxs], out_perm[3], rtol=1e-5, atol=1e-5)
    # Check pit outflow
    assert np.isclose(out_std[6], out_perm[6])


def test_local_inertial_water_stage_state_variable() -> None:
    """Verifies initialization and update of water_stage_m state variable."""
    n_cells: int = 3
    ldd = np.array([[8, 8, 5]], dtype=np.uint8)
    river_network = pyflwdir.from_array(ldd, ftype="ldd", transform=Affine.identity())

    bed_elevation = np.array([20.0, 10.0, 5.0], dtype=np.float32)
    river_length = np.array([500.0, 500.0, 500.0], dtype=np.float32)
    river_width = np.array([10.0, 20.0, 20.0], dtype=np.float32)
    waterbody_ids = np.array([-1, -1, -1], dtype=np.int32)
    is_waterbody_outflow = np.array([False, False, False], dtype=bool)
    river_ids = np.array([101, 102, 103], dtype=np.int32)
    use_kinematic = np.array([False, False, False], dtype=bool)

    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([102, 103, -1], dtype=np.int32),
            "slope": np.array([0.02, 0.01, 0.005], dtype=np.float32),
        },
        index=river_ids,
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=river_width,
        waterbody_ids=waterbody_ids,
        river_ids=river_ids,
        is_waterbody_outflow=is_waterbody_outflow,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevation,
    )

    # Initial stage should equal bed elevation + bankfull depth
    stage_initial = router.get_water_stage()
    assert (stage_initial >= bed_elevation).all()
    assert np.all(np.isfinite(stage_initial))

    # Test explicit re-initialization with water_stage_m
    custom_stage = bed_elevation + np.array([1.5, 2.0, 2.5], dtype=np.float32)
    router.initialize_stage(water_stage_m=custom_stage)
    stage_readback = router.get_water_stage()
    np.testing.assert_allclose(stage_readback, custom_stage, rtol=1e-6)

    # Execute a step and verify stage is updated
    storage = np.array([500.0, 1000.0, 800.0], dtype=np.float64)
    discharge = np.array([2.0, 4.0, 3.0], dtype=np.float32)
    sideflow = np.zeros(n_cells, dtype=np.float32)
    evap = np.zeros(n_cells, dtype=np.float32)
    wb_storage = np.zeros(0, dtype=np.float64)
    wb_outflow = np.zeros(0, dtype=np.float32)
    ret_storage = np.zeros(0, dtype=np.float32)
    alpha = np.full(n_cells, 1.0, dtype=np.float32)
    beta = np.full(n_cells, 0.6, dtype=np.float32)
    ret_thresh = np.zeros(0, dtype=np.float32)

    router.step(
        Q_prev_m3_s=discharge,
        river_storage_m3=storage,
        sideflow_m3=sideflow,
        evaporation_m3=evap,
        waterbody_storage_m3=wb_storage,
        outflow_per_waterbody_m3=wb_outflow,
        retention_storage_m3=ret_storage,
        retention_activation_threshold_m3_s=ret_thresh,
    )

    stage_stepped = router.get_water_stage()
    assert (stage_stepped >= bed_elevation).all()
    assert np.all(np.isfinite(stage_stepped))


def test_local_inertial_waterbody_parameters_required(
    ldd: np.ndarray, mask: np.ndarray
) -> None:
    """Test that LocalInertial requires explicit waterbody parameters and validates their dimensions.

    Args:
        ldd: Fixture providing local drainage direction raster.
        mask: Fixture providing river mask raster.
    """
    flw: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=(0, 1, 0, 0, 0, -1), latlon=False
    )
    n_cells: int = int(mask.sum())
    river_len: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_w: np.ndarray = np.full(n_cells, 20.0, dtype=np.float32)
    wb_ids: np.ndarray = np.full(n_cells, -1, dtype=np.int32)
    riv_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    is_wb_out: np.ndarray = np.zeros(n_cells, dtype=bool)
    ret_storage: np.ndarray = np.zeros(0, dtype=np.float32)
    ret_node: np.ndarray = np.full(n_cells, -1, dtype=np.int32)
    ctrl_ret: np.ndarray = np.zeros(0, dtype=bool)
    bed_elev: np.ndarray = np.zeros(n_cells, dtype=np.float32)
    manning: np.ndarray = np.full(n_cells, 0.03, dtype=np.float32)
    use_kin: np.ndarray = np.zeros(n_cells, dtype=bool)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.full(n_cells, -1, dtype=np.int32),
            "slope": np.full(n_cells, 0.001, dtype=np.float32),
        },
        index=riv_ids,
    )
    shape_exp: np.ndarray = np.full(n_cells, 0.5, dtype=np.float32)
    bk_depth: np.ndarray = np.full(n_cells, 1.0, dtype=np.float32)
    fp_width: np.ndarray = np.zeros(n_cells, dtype=np.float32)

    # 1. Omitting any required waterbody parameter should raise TypeError
    router_cls: Any = LocalInertial
    incomplete_kwargs: dict[str, Any] = {
        "dt": 3600,
        "river_network": flw,
        "river_length": river_len,
        "river_width": river_w,
        "waterbody_ids": wb_ids,
        "river_ids": riv_ids,
        "is_waterbody_outflow": is_wb_out,
        "retention_max_storage_m3": ret_storage,
        "retention_node_id": ret_node,
        "controlled_retention": ctrl_ret,
        "retention_basin_release_threshold_factor": 0.9,
        "bankfull_river_elevation_m": bed_elev,
        "manning_n": manning,
        "use_kinematic": use_kin,
        "rivers_gdf": rivers_gdf,
        "shape_exponent": shape_exp,
        "bankfull_depth_m": bk_depth,
        "floodplain_width_m": fp_width,
        # Missing waterbody_lake_area, etc.
    }
    with pytest.raises(TypeError):
        router_cls(**incomplete_kwargs)

    # 2. Mismatched array length against n_wb should raise AssertionError
    with pytest.raises(AssertionError, match="waterbody_lake_area size"):
        LocalInertial(
            dt=3600,
            river_network=flw,
            river_length=river_len,
            river_width=river_w,
            waterbody_ids=wb_ids,
            river_ids=riv_ids,
            is_waterbody_outflow=is_wb_out,
            retention_max_storage_m3=ret_storage,
            retention_node_id=ret_node,
            controlled_retention=ctrl_ret,
            retention_basin_release_threshold_factor=0.9,
            bankfull_river_elevation_m=bed_elev,
            manning_n=manning,
            use_kinematic=use_kin,
            rivers_gdf=rivers_gdf,
            shape_exponent=shape_exp,
            bankfull_depth_m=bk_depth,
            floodplain_width_m=fp_width,
            waterbody_lake_area=np.ones(2, dtype=np.float32),  # n_wb is 0
            waterbody_lake_factor=np.zeros(0, dtype=np.float32),
            waterbody_outflow_height=np.zeros(0, dtype=np.float32),
            waterbody_outflow_bed_elev=np.zeros(0, dtype=np.float32),
            river_storage_alpha=np.full(n_cells, 1.0, dtype=np.float32),
            river_storage_beta=np.full(n_cells, 0.6, dtype=np.float32),
            in_spinup=True,
        )


def test_inertial_substeps_parallel_and_serial_equivalence() -> None:
    """Test numerical equivalence and dynamic dispatch of serial and parallel local inertial kernels.

    Verifies that the serial and parallel compiled variants of the 1D Saint-Venant momentum/continuity
    substepping kernel produce numerically identical results, and that dispatching works across the
    threshold boundary.
    """
    from geb.hydrology.routing.inertial_substeps import (
        INERTIAL_PARALLEL_THRESHOLD,
        _run_inertial_substeps,
        _run_inertial_substeps_parallel,
        _run_inertial_substeps_serial,
    )

    assert INERTIAL_PARALLEL_THRESHOLD == 5000

    n_reaches: int = 25
    ldd_grid: npt.NDArray[np.uint8] = np.full((n_reaches, 1), 2, dtype=np.uint8)
    ldd_grid[-1, 0] = 5
    active_mask: npt.NDArray[np.bool_] = np.ones((n_reaches, 1), dtype=bool)
    network: pyflwdir.FlwdirRaster = create_river_network(
        ldd_grid, active_mask, transform=Affine.identity()
    )

    router: LocalInertial = _make_local_inertial(
        dt=15,
        river_network=network,
        river_length=np.full(n_reaches, 100.0, dtype=np.float32),
        waterbody_ids=np.full(n_reaches, -1, dtype=np.int32),
        is_waterbody_outflow=np.zeros(n_reaches, dtype=bool),
        retention_max_storage_m3=np.zeros(n_reaches, dtype=np.float32),
        retention_node_id=np.full(n_reaches, -1, dtype=np.int32),
        controlled_retention=np.zeros(n_reaches, dtype=bool),
        retention_basin_release_threshold_factor=0.2,
        bankfull_river_elevation_m=np.linspace(
            100.0, 10.0, n_reaches, dtype=np.float32
        ),
        manning_n=np.full(n_reaches, 0.03, dtype=np.float32),
        river_width=np.full(n_reaches, 20.0, dtype=np.float32),
        bankfull_depth_m=np.full(n_reaches, 2.0, dtype=np.float32),
        floodplain_width_m=np.full(n_reaches, 100.0, dtype=np.float32),
    )

    def build_test_state() -> list[Any]:
        return [
            3,
            np.float32(router.dt),
            n_reaches,
            np.full(n_reaches, 10.0, dtype=np.float32),
            np.full(n_reaches, 5000.0, dtype=np.float64),
            np.full(n_reaches, 50.0, dtype=np.float32),
            np.full(n_reaches, 1.0, dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float32),
            router._geom_inbank,
            router._geom_overbank,
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.float32),
            np.float32(router.retention_basin_release_threshold_factor),
            router._pit_slope_inertial,
            np.zeros(n_reaches, dtype=np.float32),
            np.zeros(n_reaches, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(n_reaches, dtype=np.float32),
            np.zeros(n_reaches, dtype=np.float32),
            np.zeros(n_reaches, dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(n_reaches, dtype=np.float64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.linspace(101.0, 11.0, n_reaches, dtype=np.float32),
            router._ds_boundary_type,
            router._ds_inertial_k,
            router._ds_stage_idx,
            router._ds_bed_elevation,
            router._kin_ds_slope,
            router._inertial_up_offsets,
            router._inertial_up_indices,
            router._inertial_up_reach_idx,
            router._inertial_wb_ds_k,
            router._inertial_wb_ds_id,
            router._lake_outflow_target_k,
            router._lake_outflow_wb_id,
            router._geom_cfl,
            router._min_dt_buf,
            router._rev_demand_buf,
            router._wb_release_volume_substep,
            router._inertial_topo_order,
            router._total_flow_rate_buf,
        ]

    state_serial: list[Any] = build_test_state()
    state_parallel: list[Any] = build_test_state()
    state_dispatched: list[Any] = build_test_state()

    substeps_serial: int = _run_inertial_substeps_serial(*state_serial)
    substeps_parallel: int = _run_inertial_substeps_parallel(*state_parallel)
    substeps_dispatched: int = _run_inertial_substeps(*state_dispatched)

    assert substeps_serial == substeps_parallel == substeps_dispatched

    # Substep discharge buffer (index 3)
    np.testing.assert_allclose(state_serial[3], state_parallel[3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        state_serial[3], state_dispatched[3], rtol=1e-5, atol=1e-5
    )
    # Storage volume (index 4)
    np.testing.assert_allclose(state_serial[4], state_parallel[4], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        state_serial[4], state_dispatched[4], rtol=1e-5, atol=1e-5
    )
    # Time-averaged discharge output (index 24)
    np.testing.assert_allclose(
        state_serial[24], state_parallel[24], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        state_serial[24], state_dispatched[24], rtol=1e-5, atol=1e-5
    )
    # Stage buffer (index 37)
    np.testing.assert_allclose(
        state_serial[37], state_parallel[37], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        state_serial[37], state_dispatched[37], rtol=1e-5, atol=1e-5
    )


def test_serial_and_parallel_kinematic_kernels_equivalence() -> None:
    """Test numerical equivalence and dynamic dispatch of serial and parallel kinematic wave kernels.

    Verifies that the serial and parallel compiled variants of the kinematic wave routing
    step produce numerically identical results, and that dispatching logic operates correctly.
    """
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
        return [
            dt_f32,
            inv_dt_f32,
            n_kinematic,
            n_kin_headwater,
            kin_ds_reach.copy(),
            np.zeros(n_kinematic, dtype=np.float32),
            np.array([1000.0, 1500.0, 500.0, 200.0], dtype=np.float32),
            np.array([50.0, 60.0, 30.0, 20.0], dtype=np.float32),
            np.array([2.0, 3.0, 5.0, 4.0], dtype=np.float32),
            np.array([0.5, 0.6, 0.55, 0.52], dtype=np.float32),
            np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32),
            np.array([1000.0, 1200.0, 1500.0, 1100.0], dtype=np.float32),
            np.array([5000.0, 6000.0, 10000.0, 8000.0], dtype=np.float64),
            np.zeros(n_kinematic, dtype=np.float32),
            np.zeros(n_kinematic, dtype=np.float32),
            np.zeros(n_kinematic, dtype=np.float32),
            kin_ds_wb_id.copy(),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(n_kinematic, dtype=np.float32),
            np.array([0, -1, -1, -1], dtype=np.int32),
            np.array([100.0], dtype=np.float32),
            np.array([5000.0], dtype=np.float32),
            np.array([True], dtype=bool),
            np.array([1.5], dtype=np.float32),
            np.float32(0.2),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        ]

    state_serial: list[Any] = build_test_state()
    state_parallel: list[Any] = build_test_state()
    state_raw: list[Any] = build_test_state()

    _run_kinematic_step_serial(*state_serial)
    _run_kinematic_step_parallel(*state_parallel)
    _run_kinematic_step(*state_raw)

    # updated_discharge_m3_s (index 13)
    np.testing.assert_allclose(
        state_serial[13], state_parallel[13], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(state_serial[13], state_raw[13], rtol=1e-5, atol=1e-5)

    # river_storage_m3 (index 12)
    np.testing.assert_allclose(
        state_serial[12], state_parallel[12], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(state_serial[12], state_raw[12], rtol=1e-5, atol=1e-5)

    # actual_evaporation_m3 (index 14)
    np.testing.assert_allclose(
        state_serial[14], state_parallel[14], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(state_serial[14], state_raw[14], rtol=1e-5, atol=1e-5)

    # retention_storage_m3 (index 22)
    np.testing.assert_allclose(
        state_serial[22], state_parallel[22], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(state_serial[22], state_raw[22], rtol=1e-5, atol=1e-5)


def test_local_inertial_oscillation_suppression() -> None:
    """Verify numerical stability and oscillation suppression in demanding regimes.

    Tests two critical regimes known to provoke numerical oscillations in local inertial formulations:
    1. Low Manning roughness (0.015 s/m^(1/3)) on flat bed slopes (1e-4 m/m) subjected to a sharp flood pulse.
       Verifies monotonic recession without spurious rebound waves or 2*dx limit cycles.
    2. Severely irregular reach lengths (alternating from 400 m to 3000 m).
       Verifies that interface spatial weighting and reach-averaged spatial step prevent artificial
       pressure gradient spikes.
    """
    # -------------------------------------------------------------------------
    # Regime 1: Flat slope and low roughness
    # -------------------------------------------------------------------------
    n_cells_flat: int = 15
    reach_length_m: float = 1000.0
    dt_seconds: int = 3600

    ldd_flat: np.ndarray = np.full((n_cells_flat, 1), 2, dtype=np.uint8)
    ldd_flat[-1, 0] = 5
    river_network_flat: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd_flat, ftype="ldd", transform=Affine.identity()
    )
    river_ids_flat: np.ndarray = np.arange(n_cells_flat, dtype=np.int32)

    bed_slope_flat: float = 0.0001
    bed_elevation_flat: np.ndarray = (
        n_cells_flat - 1 - np.arange(n_cells_flat, dtype=np.float32)
    ) * (reach_length_m * bed_slope_flat) + 5.0

    rivers_gdf_flat: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.concatenate(
                [np.arange(1, n_cells_flat, dtype=np.int32), [-1]]
            ),
            "slope": np.full(n_cells_flat, bed_slope_flat, dtype=np.float32),
        },
        index=river_ids_flat,
    )

    manning_n_flat: np.ndarray = np.full(n_cells_flat, 0.015, dtype=np.float32)
    reach_lengths_flat: np.ndarray = np.full(
        n_cells_flat, reach_length_m, dtype=np.float32
    )
    river_width_flat: np.ndarray = np.full(n_cells_flat, 30.0, dtype=np.float32)

    router_flat: LocalInertial = _make_local_inertial(
        dt=dt_seconds,
        river_network=river_network_flat,
        river_length=reach_lengths_flat,
        river_width=river_width_flat,
        rivers_gdf=rivers_gdf_flat,
        bankfull_river_elevation_m=bed_elevation_flat,
        manning_n=manning_n_flat,
        shape_exponent=0.5,
        bankfull_depth_m=10.0,
        floodplain_width_m=np.zeros(n_cells_flat, dtype=np.float32),
    )

    n_simulation_hours: int = 30
    inflow_hydrograph_flat: np.ndarray = np.full(
        n_simulation_hours, 5.0, dtype=np.float32
    )
    inflow_hydrograph_flat[2:6] = [20.0, 50.0, 100.0, 60.0]

    storage_flat: np.ndarray = np.full(n_cells_flat, 10000.0, dtype=np.float64)
    q_flat: np.ndarray = np.full(n_cells_flat, 5.0, dtype=np.float32)
    outlet_discharges_flat: list[float] = []

    for hour_idx in range(n_simulation_hours):
        sideflow_flat: np.ndarray = np.zeros(n_cells_flat, dtype=np.float32)
        sideflow_flat[0] = np.float32(inflow_hydrograph_flat[hour_idx] * dt_seconds)
        step_result = router_flat.step(
            Q_prev_m3_s=q_flat,
            river_storage_m3=storage_flat,
            sideflow_m3=sideflow_flat,
            evaporation_m3=np.zeros(n_cells_flat, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        q_flat, storage_flat = step_result[0], step_result[1]
        outlet_discharges_flat.append(float(q_flat[-1]))

    outlet_q_array: np.ndarray = np.array(outlet_discharges_flat)
    peak_index_flat: int = int(np.argmax(outlet_q_array))
    recession_diffs_flat: np.ndarray = np.diff(outlet_q_array[peak_index_flat:])
    # During flood recession, discharge must decrease monotonically without rebounding
    assert np.all(recession_diffs_flat <= 1e-3), (
        f"Spurious oscillation in recession limbs: {recession_diffs_flat[recession_diffs_flat > 1e-3]}"
    )

    # -------------------------------------------------------------------------
    # Regime 2: Severely irregular reach lengths
    # -------------------------------------------------------------------------
    n_cells_irreg: int = 8
    reach_lengths_irreg: np.ndarray = np.array(
        [500.0, 2500.0, 400.0, 3000.0, 600.0, 2000.0, 800.0, 1500.0], dtype=np.float32
    )

    ldd_irreg: np.ndarray = np.full((n_cells_irreg, 1), 2, dtype=np.uint8)
    ldd_irreg[-1, 0] = 5
    river_network_irreg: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd_irreg, ftype="ldd", transform=Affine.identity()
    )
    river_ids_irreg: np.ndarray = np.arange(n_cells_irreg, dtype=np.int32)

    bed_slope_irreg: float = 0.0008
    cumulative_distance_m: np.ndarray = np.cumsum(
        np.concatenate([[0.0], reach_lengths_irreg[:-1]])
    )
    total_length_m: float = float(cumulative_distance_m[-1] + reach_lengths_irreg[-1])
    bed_elevation_irreg: np.ndarray = (
        20.0 + (total_length_m - cumulative_distance_m) * bed_slope_irreg
    ).astype(np.float32)

    rivers_gdf_irreg: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.concatenate(
                [np.arange(1, n_cells_irreg, dtype=np.int32), [-1]]
            ),
            "slope": np.full(n_cells_irreg, bed_slope_irreg, dtype=np.float32),
        },
        index=river_ids_irreg,
    )

    router_irreg: LocalInertial = _make_local_inertial(
        dt=dt_seconds,
        river_network=river_network_irreg,
        river_length=reach_lengths_irreg,
        river_width=np.full(n_cells_irreg, 25.0, dtype=np.float32),
        rivers_gdf=rivers_gdf_irreg,
        bankfull_river_elevation_m=bed_elevation_irreg,
        manning_n=np.full(n_cells_irreg, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=10.0,
        floodplain_width_m=np.zeros(n_cells_irreg, dtype=np.float32),
    )

    storage_irreg: np.ndarray = (reach_lengths_irreg * 25.0 * 1.5).astype(np.float64)
    q_irreg: np.ndarray = np.full(n_cells_irreg, 10.0, dtype=np.float32)
    inflow_hydrograph_irreg: np.ndarray = np.full(
        n_simulation_hours, 10.0, dtype=np.float32
    )
    inflow_hydrograph_irreg[2:6] = [30.0, 70.0, 120.0, 60.0]

    outlet_discharges_irreg: list[float] = []
    for hour_idx in range(n_simulation_hours):
        sideflow_irreg: np.ndarray = np.zeros(n_cells_irreg, dtype=np.float32)
        sideflow_irreg[0] = np.float32(inflow_hydrograph_irreg[hour_idx] * dt_seconds)
        step_result_irreg = router_irreg.step(
            Q_prev_m3_s=q_irreg,
            river_storage_m3=storage_irreg,
            sideflow_m3=sideflow_irreg,
            evaporation_m3=np.zeros(n_cells_irreg, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        q_irreg, storage_irreg = step_result_irreg[0], step_result_irreg[1]
        outlet_discharges_irreg.append(float(q_irreg[-1]))

    outlet_q_irreg_array: np.ndarray = np.array(outlet_discharges_irreg)
    peak_index_irreg: int = int(np.argmax(outlet_q_irreg_array))
    recession_diffs_irreg: np.ndarray = np.diff(outlet_q_irreg_array[peak_index_irreg:])
    assert np.all(recession_diffs_irreg <= 1e-3), (
        f"Spurious oscillation across irregular reaches: {recession_diffs_irreg[recession_diffs_irreg > 1e-3]}"
    )


def test_local_inertial_confluence_asymmetric_stability() -> None:
    """Verify stability and monotonic convergence at a confluence with asymmetric baseflow.

    A small tributary (2 m³/s) joins a large river trunk (50 m³/s). Under pure local inertia,
    the tributary reaches steady state without artificial momentum inflation or limit cycles.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([10.0, 30.0, 20.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([10.0, 5.0, 10.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.full(n_cells, 0.005, dtype=np.float32),
        },
        index=river_ids,
    )
    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=5.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    storage: np.ndarray = np.array([2000.0, 50000.0, 40000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([2.0, 52.0, 50.0], dtype=np.float32)
    dt_seconds: int = 3600

    for _ in range(15):
        sideflow: np.ndarray = np.array(
            [2.0 * dt_seconds, 0.0, 50.0 * dt_seconds], dtype=np.float32
        )
        step_result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = step_result[0], step_result[1]
        assert np.all(discharge >= 0.0), (
            f"Negative discharge observed at confluence: {discharge}"
        )
        assert np.all(storage > 0.0), f"Storage depleted at confluence: {storage}"

    # Verify steady-state values match expected inflows
    assert discharge[0] == pytest.approx(2.0, abs=0.05)
    assert discharge[2] == pytest.approx(50.0, abs=0.1)
    assert discharge[1] == pytest.approx(52.0, abs=0.1)


def test_local_inertial_volume_limiter_subcritical_preservation() -> None:
    """Verify that storage volume limiter preserves unthrottled subcritical flow and prevents negative storage.

    Tests two key behaviors of the rewritten threshold storage volume limiter:
    1. Normal subcritical flow where Q * dt <= 0.8 * V: discharge must be 100% unthrottled
       (eliminating the historical 10%-45% premature attenuation).
    2. Over-draft conditions where target discharge greatly exceeds available storage:
       discharge is smoothly clamped so storage volume remains positive.
    """
    from geb.hydrology.routing.inertial_substeps import (
        GEOM_IN_BANKFULL_DEPTH,
        GEOM_IN_BANKFULL_VOLUME,
        GEOM_IN_BED_ELEVATION,
        GEOM_IN_INTERFACE_BED_ELEVATION_MAX,
        GEOM_IN_INVERSE_INTERFACE_LENGTH,
        GEOM_IN_INVERSE_LENGTH,
        GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE,
        GEOM_IN_MANNING_N_SQUARED,
        GEOM_IN_NUM_COLS,
        GEOM_IN_STAGE_VOLUME_COEFFICIENT,
        GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH,
        GEOM_OV_BANKFULL_PERIMETER,
        GEOM_OV_FLOODPLAIN_AREA_THRESHOLD,
        GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD,
        GEOM_OV_FLOODPLAIN_SIDE_SLOPE,
        GEOM_OV_FLOODPLAIN_WIDTH,
        GEOM_OV_NUM_COLS,
        GEOM_OV_RIVER_WIDTH,
        GEOM_OV_SQRT_ONE_PLUS_FLOODPLAIN_SLOPE_SQUARED,
        _solve_inertial_momentum,
    )

    geom_inbank: np.ndarray = np.zeros((1, GEOM_IN_NUM_COLS), dtype=np.float32)
    geom_inbank[0, GEOM_IN_INVERSE_LENGTH] = np.float32(1.0 / 1000.0)
    geom_inbank[0, GEOM_IN_BED_ELEVATION] = np.float32(10.0)
    geom_inbank[0, GEOM_IN_BANKFULL_DEPTH] = np.float32(5.0)
    geom_inbank[0, GEOM_IN_INVERSE_SHAPE_EXPONENT_PLUS_ONE] = np.float32(1.0 / 1.5)
    geom_inbank[0, GEOM_IN_BANKFULL_VOLUME] = np.float32(
        1000.0 * 20.0 * 5.0 * (1.0 / 1.5)
    )
    geom_inbank[0, GEOM_IN_STAGE_VOLUME_COEFFICIENT] = np.float32(0.1)
    geom_inbank[0, GEOM_IN_INTERFACE_BED_ELEVATION_MAX] = np.float32(10.0)
    geom_inbank[0, GEOM_IN_WIDTH_OVER_SQRT_BANKFULL_DEPTH] = np.float32(
        20.0 / math.sqrt(5.0)
    )
    geom_inbank[0, GEOM_IN_MANNING_N_SQUARED] = np.float32(0.03**2)
    geom_inbank[0, GEOM_IN_INVERSE_INTERFACE_LENGTH] = np.float32(1.0 / 1000.0)

    geom_overbank: np.ndarray = np.zeros((1, GEOM_OV_NUM_COLS), dtype=np.float32)
    geom_overbank[0, GEOM_OV_RIVER_WIDTH] = np.float32(20.0)
    geom_overbank[0, GEOM_OV_FLOODPLAIN_WIDTH] = np.float32(0.0)
    geom_overbank[0, GEOM_OV_FLOODPLAIN_SIDE_SLOPE] = np.float32(0.0)
    geom_overbank[0, GEOM_OV_FLOODPLAIN_DEPTH_THRESHOLD] = np.float32(5.0)
    geom_overbank[0, GEOM_OV_FLOODPLAIN_AREA_THRESHOLD] = np.float32(100.0)
    geom_overbank[0, GEOM_OV_BANKFULL_PERIMETER] = np.float32(25.0)
    geom_overbank[0, GEOM_OV_SQRT_ONE_PLUS_FLOODPLAIN_SLOPE_SQUARED] = np.float32(1.0)

    inv_dt_substep: np.float32 = np.float32(0.1)  # dt = 10s
    g_dt_substep: np.float32 = np.float32(9.80665 * 10.0)
    sqrt_gravity: np.float32 = np.float32(math.sqrt(9.80665))

    # Case 1: Normal subcritical flow with plenty of storage (V = 10000 m3, V * inv_dt = 1000 m3/s)
    storage_subcritical: np.ndarray = np.array([10000.0], dtype=np.float64)
    q_subcritical: np.float32 = _solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(2.0),
        water_slope=np.float32(-0.001),
        curr_discharge=np.float32(20.0),
        min_wet_depth_m=np.float32(1e-3),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=g_dt_substep,
        sqrt_gravity=sqrt_gravity,
        can_reverse=False,
        river_storage_m3_inertial=storage_subcritical,
        inv_dt_substep=inv_dt_substep,
    )
    assert q_subcritical > np.float32(0.0)

    # Case 2: Near empty reach with high discharge demand (V = 100 m3, V * inv_dt = 10 m3/s)
    storage_near_empty: np.ndarray = np.array([100.0], dtype=np.float64)
    q_limited: np.float32 = _solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(3.0),
        water_slope=np.float32(-0.01),
        curr_discharge=np.float32(50.0),
        min_wet_depth_m=np.float32(1e-3),
        geom_inbank=geom_inbank,
        geom_overbank=geom_overbank,
        g_dt_substep=g_dt_substep,
        sqrt_gravity=sqrt_gravity,
        can_reverse=False,
        river_storage_m3_inertial=storage_near_empty,
        inv_dt_substep=inv_dt_substep,
    )
    max_allowable_drain: float = 0.95 * 100.0 * 0.1  # 9.5 m3/s
    assert q_limited <= max_allowable_drain, (
        f"Limiter allowed excessive outflow: {q_limited} > {max_allowable_drain}"
    )
    assert q_limited >= np.float32(8.0), (
        f"Limiter overly restricted flow below threshold: {q_limited} < 8.0"
    )


def test_local_inertial_short_reach_cfl_stability() -> None:
    """Verify that short reaches bordering long reaches constrain the CFL timestep properly.

    When a 50 m reach borders a 2500 m reach, the interface distance is (50 + 2500)/2 = 1275 m.
    The CFL spatial step must use min(dx_reach, dx_interface) = 50 m, preventing Courant condition
    violations in the short reach.
    """
    from geb.hydrology.routing.inertial_substeps import GEOM_CFL_CONSTANT

    n_cells: int = 3
    # Reach 0: 50 m, Reach 1: 2500 m, Reach 2: 2500 m
    reach_lengths: np.ndarray = np.array([50.0, 2500.0, 2500.0], dtype=np.float32)
    ldd: np.ndarray = np.array([[2], [2], [5]], dtype=np.uint8)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", transform=Affine.identity()
    )
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, 2, -1], dtype=np.int32),
            "slope": np.full(n_cells, 0.001, dtype=np.float32),
        },
        index=river_ids,
    )
    bed_elevations: np.ndarray = np.array([10.0, 9.95, 7.45], dtype=np.float32)

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=reach_lengths,
        river_width=np.full(n_cells, 20.0, dtype=np.float32),
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=5.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    # Verify that reach 0 CFL constant is based on its 50 m length, not the 1275 m interface
    k0: int = int(np.where(router._river_ids[router.n_kinematic :] == 0)[0][0])
    sqrt_g: float = math.sqrt(9.80665)
    expected_cfl_const_0: float = 0.7 * 50.0 / sqrt_g
    actual_cfl_const_0: float = float(router._geom_cfl[k0, GEOM_CFL_CONSTANT])
    assert actual_cfl_const_0 == pytest.approx(expected_cfl_const_0, rel=1e-3)

    # Run a flood wave step through the network and verify stable completion
    storage: np.ndarray = (reach_lengths * 20.0 * 2.0).astype(np.float64)
    discharge: np.ndarray = np.full(n_cells, 10.0, dtype=np.float32)
    sideflow: np.ndarray = np.zeros(n_cells, dtype=np.float32)
    sideflow[0] = np.float32(50.0 * 3600)  # Sudden flood wave into short reach

    step_result = router.step(
        Q_prev_m3_s=discharge,
        river_storage_m3=storage,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )
    q_out, storage_out = step_result[0], step_result[1]
    assert np.all(np.isfinite(q_out))
    assert np.all(np.isfinite(storage_out))
    assert np.all(storage_out > 0.0)


def test_local_inertial_confluence_flood_pulse_and_recession() -> None:
    """Verify confluence stability during a severe flood pulse on the main stem and subsequent recession.

    Tests dynamic backwater interaction at a confluence when a main stem undergoes a large flood pulse
    (rising to 250 m³/s) while a tributary maintains steady low baseflow (2 m³/s).
    Verifies monotonic recession of outlet peak, absence of spurious rebound oscillations,
    non-negative storage, and exact overall mass conservation.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([10.0, 30.0, 25.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([10.0, 5.0, 10.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.full(n_cells, 0.005, dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=5.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([2000.0, 20000.0, 18000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([2.0, 52.0, 50.0], dtype=np.float32)

    # Sharp flood wave on main stem (cell 2)
    hydrograph_main: list[float] = [
        50.0,
        100.0,
        250.0,
        180.0,
        100.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
    ]
    tributary_inflow: float = 2.0

    cum_inflow: float = 0.0
    cum_outflow: float = 0.0
    initial_storage: float = float(np.sum(storage))
    outlet_discharges: list[float] = []

    for q_main in hydrograph_main:
        sideflow: np.ndarray = np.array(
            [tributary_inflow * dt_seconds, 0.0, q_main * dt_seconds],
            dtype=np.float32,
        )
        cum_inflow += float(np.sum(sideflow))
        step_result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = step_result[0], step_result[1]
        pit_outflow: float = float(step_result[6])
        cum_outflow += pit_outflow
        outlet_discharges.append(float(discharge[1]))

        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(storage > 0.0), f"Storage depleted at confluence: {storage}"
        assert float(discharge[0]) > 0.0, (
            f"Spurious negative flow in tributary: {discharge[0]}"
        )

    # Verify monotonic recession of the outlet flood wave
    peak_idx: int = int(np.argmax(outlet_discharges))
    recession_limb: np.ndarray = np.array(outlet_discharges[peak_idx:])
    recession_diffs: np.ndarray = np.diff(recession_limb)
    assert np.all(recession_diffs <= 0.2), (
        f"Spurious rebound oscillation in confluence recession: {recession_diffs}"
    )

    # Verify overall mass conservation
    final_storage: float = float(np.sum(storage))
    mass_balance_error: float = abs(
        cum_inflow - (cum_outflow + final_storage - initial_storage)
    )
    assert mass_balance_error < 1.0, (
        f"Mass balance violated at confluence: {mass_balance_error} m³"
    )

    # Verify steady-state recovery at the end of the simulation
    assert discharge[0] == pytest.approx(2.0, abs=0.05)
    assert discharge[2] == pytest.approx(50.0, abs=0.1)
    assert discharge[1] == pytest.approx(52.0, abs=0.1)


def test_local_inertial_multi_tributary_confluence_stability() -> None:
    """Verify numerical stability at a multi-tributary confluence where 3 reaches merge into 1 trunk.

    Three distinct tributary reaches with different widths, slopes, and baseflow rates
    (3 m³/s, 10 m³/s, 25 m³/s) merge into a single confluence node.
    Verifies that all three tributaries converge to their individual inflow rates and
    the combined trunk reach cleanly discharges the sum of all inflows (38 m³/s)
    without cross-talk oscillations.
    """
    ldd: np.ndarray = np.array([[5, 2, 5], [6, 2, 4], [5, 5, 5]], dtype=np.uint8)
    mask: np.ndarray = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = river_network.ncells
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([8.0, 15.0, 35.0, 25.0, 40.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array(
        [15.0, 15.0, 10.0, 15.0, 5.0], dtype=np.float32
    )
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([2, 2, 4, 2, -1], dtype=np.int32),
            "slope": np.array([0.005, 0.005, 0.005, 0.005, 0.005], dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=5.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = (river_lengths * river_widths * 1.5).astype(np.float64)
    discharge: np.ndarray = np.array([3.0, 10.0, 38.0, 25.0, 38.0], dtype=np.float32)

    sideflow: np.ndarray = np.zeros(n_cells, dtype=np.float32)
    sideflow[0] = np.float32(3.0 * dt_seconds)
    sideflow[1] = np.float32(10.0 * dt_seconds)
    sideflow[3] = np.float32(25.0 * dt_seconds)

    for _ in range(12):
        step_result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = step_result[0], step_result[1]
        assert np.all(discharge >= 0.0), (
            f"Negative discharge in multi-tributary junction: {discharge}"
        )
        assert np.all(storage > 0.0), (
            f"Depleted storage in multi-tributary junction: {storage}"
        )

    # Verify steady-state convergence of all 3 incoming tributaries
    assert discharge[0] == pytest.approx(3.0, abs=0.05)
    assert discharge[1] == pytest.approx(10.0, abs=0.05)
    assert discharge[3] == pytest.approx(25.0, abs=0.05)
    # Verify confluence junction and outlet discharge
    assert discharge[2] == pytest.approx(38.0, abs=0.1)
    assert discharge[4] == pytest.approx(38.0, abs=0.2)


def test_local_inertial_3_river_confluence_pulse_and_recession_stability() -> None:
    """Verify stability of a 3-river confluence subjected to an asymmetric shock and recession.

    Three upstream tributary reaches join at a single confluence cell on a mild slope.
    A sudden severe shock on one tributary induces backwater reverse flows into the other
    tributaries. During the subsequent recession, discharges must decay monotonically without
    spurious rebound sloshing between tributaries.
    """
    ldd: np.ndarray = np.array([[5, 2, 5], [6, 2, 4], [5, 5, 5]], dtype=np.uint8)
    mask: np.ndarray = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = river_network.ncells
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array(
        [10.0, 15.0, 25.0, 12.0, 30.0], dtype=np.float32
    )
    bed_elevations: np.ndarray = np.array([5.1, 5.1, 5.0, 5.1, 4.9], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([2, 2, 4, 2, -1], dtype=np.int32),
            "slope": np.array(
                [0.0001, 0.0001, 0.0001, 0.0001, 0.0001], dtype=np.float32
            ),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.015, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=5.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = (river_lengths * river_widths * 1.0).astype(np.float64)
    discharge: np.ndarray = np.array([1.0, 1.0, 3.0, 1.0, 3.0], dtype=np.float32)

    sideflow_pulse: np.ndarray = np.zeros(n_cells, dtype=np.float32)
    sideflow_pulse[0] = np.float32(200.0 * dt_seconds)

    # Step 0: Severe asymmetric shock pulse
    res = router.step(
        Q_prev_m3_s=discharge,
        river_storage_m3=storage,
        sideflow_m3=sideflow_pulse,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )
    discharge, storage = res[0], res[1]
    assert np.all(storage > 0.0), f"Storage depleted during shock: {storage}"

    # Recession: zero inflows, verify monotonic decay without spurious rebound oscillations
    sideflow_zero: np.ndarray = np.zeros(n_cells, dtype=np.float32)
    recession_outflow: list[float] = [float(discharge[4])]

    for _ in range(8):
        res = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow_zero,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = res[0], res[1]
        assert np.all(storage > 0.0), f"Storage depleted during recession: {storage}"
        recession_outflow.append(float(discharge[4]))

    # Recession differences from the peak must be non-positive (no spurious rebounds)
    recession_arr: np.ndarray = np.array(recession_outflow)
    peak_idx: int = int(np.argmax(recession_arr))
    recession_diffs: np.ndarray = np.diff(recession_arr[peak_idx:])
    assert np.all(recession_diffs <= 1e-2), (
        f"Spurious rebound oscillation at 3-river confluence: {recession_diffs[recession_diffs > 1e-2]}"
    )


def test_local_inertial_confluence_flat_low_manning_stability() -> None:
    """Verify stability of a low-roughness flat-bed confluence subjected to an abrupt discharge step.

    Under flat slopes (1e-4 m/m) and low roughness (0.015 s/m^(1/3)), confluences are
    especially prone to spatial checkerboard oscillations. This test subjects a flat confluence
    to an abrupt 3x jump in tributary discharge and verifies smooth transition to the new equilibrium.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([10.0, 30.0, 20.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([5.1, 5.0, 5.1], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.full(n_cells, 0.0001, dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.015, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=10.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([10000.0, 30000.0, 20000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([5.0, 25.0, 20.0], dtype=np.float32)

    for step in range(25):
        q_trib: float = 15.0 if step >= 5 else 5.0
        sideflow: np.ndarray = np.array(
            [q_trib * dt_seconds, 0.0, 20.0 * dt_seconds], dtype=np.float32
        )
        step_result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = step_result[0], step_result[1]
        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(storage > 0.0)

    # Verify steady-state values after the step jump
    assert discharge[0] == pytest.approx(15.0, abs=0.05)
    assert discharge[2] == pytest.approx(20.0, abs=0.05)
    assert discharge[1] == pytest.approx(35.0, abs=0.1)


def test_local_inertial_confluence_monster_dam_break_backwater() -> None:
    """Verify stability under an extreme dam-break surge inducing deep reverse flow into a tributary.

    A small tributary (0.5 m³/s) meets a main stem that experiences an instantaneous 1500 m³/s
    monster surge / dam-break wave. The massive stage rise at the confluence forces strong
    reverse flow into the tributary, filling it with over 260,000 m³ of backwater.
    As the surge subsides, the trapped water cleanly discharges forward without oscillation,
    returning to baseflow with exact mass conservation.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([10.0, 50.0, 30.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([6.0, 5.0, 6.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.full(n_cells, 0.001, dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=8.0,
        floodplain_width_m=np.full(n_cells, 100.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([1000.0, 5000.0, 2000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([0.5, 2.5, 2.0], dtype=np.float32)

    # Extreme hydrograph: baseline (2 m3/s) -> 500 -> 1500 -> 1500 -> 300 -> 10 -> 2 m3/s
    hydrograph_main: list[float] = [
        2.0,
        500.0,
        1500.0,
        1500.0,
        300.0,
        10.0,
        2.0,
        2.0,
    ]
    tributary_inflow: float = 0.5

    cum_inflow: float = 0.0
    cum_outflow: float = 0.0
    initial_storage: float = float(np.sum(storage))
    observed_reverse_flow: bool = False

    for q_main in hydrograph_main:
        sideflow: np.ndarray = np.array(
            [tributary_inflow * dt_seconds, 0.0, q_main * dt_seconds],
            dtype=np.float32,
        )
        cum_inflow += float(np.sum(sideflow))
        step_result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = step_result[0], step_result[1]
        cum_outflow += float(step_result[6])

        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(storage > 0.0), (
            f"Negative storage during dam break surge: {storage}"
        )

        if discharge[0] < -1.0:
            observed_reverse_flow = True

    # Confirm that massive reverse backflow was actively exercised and handled
    assert observed_reverse_flow, (
        "Extreme surge failed to trigger expected backflow into tributary"
    )

    # Verify overall mass conservation
    final_storage: float = float(np.sum(storage))
    mass_balance_error: float = abs(
        cum_inflow - (cum_outflow + final_storage - initial_storage)
    )
    assert mass_balance_error < 5.0, (
        f"Mass balance error too high after dam break: {mass_balance_error} m³"
    )

    # Verify clean recovery to baseflow
    assert discharge[0] == pytest.approx(0.5, abs=0.1)
    assert discharge[2] == pytest.approx(2.0, abs=0.2)


def test_local_inertial_confluence_antiphase_tributary_clash() -> None:
    """Verify numerical resilience under alternating high-frequency antiphase flood pulses.

    Two equal tributaries clash at a confluence with alternating 400 m³/s shocks:
    Step 0: A=400, B=1 -> Step 1: A=1, B=400 -> Step 2: A=400, B=1 ...
    Verifies absence of resonance or numerical divergence, strict non-negativity of storage,
    and preservation of exact cross-channel anti-symmetry.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([20.0, 40.0, 20.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([6.0, 5.0, 6.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.full(n_cells, 0.001, dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=8.0,
        floodplain_width_m=np.full(n_cells, 50.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([5000.0, 10000.0, 5000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([10.0, 20.0, 10.0], dtype=np.float32)

    cum_inflow: float = 0.0
    cum_outflow: float = 0.0
    initial_storage: float = float(np.sum(storage))

    for step in range(8):
        q_a: float = 400.0 if (step % 2 == 0) else 1.0
        q_b: float = 1.0 if (step % 2 == 0) else 400.0
        sideflow: np.ndarray = np.array(
            [q_a * dt_seconds, 0.0, q_b * dt_seconds], dtype=np.float32
        )
        cum_inflow += float(np.sum(sideflow))
        step_result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = step_result[0], step_result[1]
        cum_outflow += float(step_result[6])

        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(storage > 0.0)

    # Verify that total outlet discharge remains steady and equal to the combined cycle inflow
    assert discharge[1] == pytest.approx(401.0, abs=1.0)
    # Verify mass conservation
    final_storage: float = float(np.sum(storage))
    mass_balance_error: float = abs(
        cum_inflow - (cum_outflow + final_storage - initial_storage)
    )
    assert mass_balance_error < 5.0


def test_local_inertial_confluence_extreme_geometric_disparity() -> None:
    """Verify confluence stability across a 500x reach length ratio and 7x Manning roughness disparity.

    A 20-meter ultra-short, narrow concrete chute (n=0.012, width 3 m) meets a 10,000-meter
    ultra-long, heavily vegetated river reach (n=0.08, width 60 m) at a 500-meter trunk reach.
    Initialized completely dry (zero storage, zero flow), the system is hit with sudden high inflows
    (15 m³/s into reach 0, 100 m³/s into reach 2).
    Verifies that the adaptive CFL substepping maintains stability without Courant failure.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    # Reach 0: 20 m, Reach 1: 500 m, Reach 2: 10,000 m
    reach_lengths: np.ndarray = np.array([20.0, 500.0, 10000.0], dtype=np.float32)
    river_widths: np.ndarray = np.array([3.0, 40.0, 60.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([6.0, 5.0, 15.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.array([0.05, 0.001, 0.001], dtype=np.float32),
        },
        index=river_ids,
    )

    manning_n: np.ndarray = np.array([0.012, 0.035, 0.08], dtype=np.float32)

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=reach_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=manning_n,
        shape_exponent=0.5,
        bankfull_depth_m=8.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.zeros(n_cells, dtype=np.float64)  # Start completely dry
    discharge: np.ndarray = np.zeros(n_cells, dtype=np.float32)

    sideflow: np.ndarray = np.array(
        [15.0 * dt_seconds, 0.0, 100.0 * dt_seconds], dtype=np.float32
    )

    for _ in range(10):
        step_result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = step_result[0], step_result[1]
        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(storage >= 0.0)

    # The 20 m reach reaches steady state quickly
    assert discharge[0] == pytest.approx(15.0, abs=0.05)
    # The 10 km reach is steadily routing water towards 100 m3/s
    assert discharge[2] > 95.0
    # Combined outlet carries their sum
    assert discharge[1] > 110.0


def test_local_inertial_confluence_overdeepened_scour_hole() -> None:
    """Verify confluence stability when the junction is an over-deepened plunge pool behind an adverse bed sill.

    The confluence cell has bed elevation 3.0 m, which is 2.0 m LOWER than the downstream exit sill (5.0 m).
    Two tributaries (bed 8.0 m) discharge 20 m³/s each into the dry depression.
    Verifies that water pools cleanly in the scour depression, overtops the adverse sill,
    and stably routes the combined 40 m³/s outflow without numerical trapping or oscillations.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([15.0, 30.0, 15.0], dtype=np.float32)
    # Confluence cell 1 has an overdeepened bed (3.0 m) relative to tributaries (8.0 m) and pit (5.0 m)
    bed_elevations: np.ndarray = np.array([8.0, 3.0, 8.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.array([0.005, 0.001, 0.005], dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=8.0,
        floodplain_width_m=np.zeros(n_cells, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.zeros(n_cells, dtype=np.float64)
    discharge: np.ndarray = np.zeros(n_cells, dtype=np.float32)

    sideflow: np.ndarray = np.array(
        [20.0 * dt_seconds, 0.0, 20.0 * dt_seconds], dtype=np.float32
    )

    for _ in range(15):
        step_result = router.step(
            Q_prev_m3_s=discharge,
            river_storage_m3=storage,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge, storage = step_result[0], step_result[1]
        assert np.all(np.isfinite(discharge))
        assert np.all(np.isfinite(storage))
        assert np.all(storage > 0.0)

    # Verify steady-state convergence over the adverse sill
    assert discharge[0] == pytest.approx(20.0, abs=0.05)
    assert discharge[2] == pytest.approx(20.0, abs=0.05)
    assert discharge[1] == pytest.approx(40.0, abs=0.1)


def test_local_inertial_cfl_weir_high_discharge_small_sill_volume() -> None:
    """Verify CFL stability when a lake releases high weir discharge with small volume above sill.

    A lake with a small area and sill volume releases an intense weir outflow
    (Q_weir = lake_factor * h² = 300 m³/s). Verifies that the initial macro CFL condition
    evaluates the physical instantaneous weir release rate rather than throttling it by the
    macro timestep, preventing downstream instability or numerical explosion.
    """
    ldd: np.ndarray = np.array([[6, 5]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 2
    river_lengths: np.ndarray = np.array([500.0, 500.0], dtype=np.float32)
    river_widths: np.ndarray = np.array([15.0, 15.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([10.0, 5.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1], dtype=np.int32),
            "slope": np.array([0.01, 0.01], dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=8.0,
        floodplain_width_m=np.full(n_cells, 30.0, dtype=np.float32),
        waterbody_ids=np.array([0, -1], dtype=np.int32),
        is_waterbody_outflow=np.array([True, False], dtype=bool),
        waterbody_lake_area=np.array([10000.0], dtype=np.float32),
        waterbody_lake_factor=np.array([300.0], dtype=np.float32),
        waterbody_outflow_height=np.array([9.0], dtype=np.float32),
        waterbody_outflow_bed_elev=np.array([10.0], dtype=np.float32),
    )

    wb_storage: np.ndarray = np.array([100000.0], dtype=np.float64)
    river_storage: np.ndarray = np.array([0.0, 1000.0], dtype=np.float64)
    discharge: np.ndarray = np.zeros(n_cells, dtype=np.float32)
    sideflow: np.ndarray = np.zeros(n_cells, dtype=np.float32)

    step_result = router.step(
        Q_prev_m3_s=discharge,
        river_storage_m3=river_storage,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=wb_storage,
        outflow_per_waterbody_m3=np.array([np.nan], dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    out_q: np.ndarray = step_result[0]
    out_s: np.ndarray = step_result[1]
    assert np.all(np.isfinite(out_q[~np.isnan(out_q)]))
    assert np.all(np.isfinite(out_s))
    assert np.all(out_s >= 0.0)
    assert out_q[1] > 0.0


def test_local_inertial_cfl_downstream_short_reach_upstream_surge() -> None:
    """Verify CFL stability when an upstream surge enters a dry short downstream reach.

    Reach 0 is 1000 m long and receives a sudden 150 m³/s flood surge, while Reach 1 is
    a short 50 m reach that starts completely dry (Q = 0, storage = 0).
    Verifies that topological inflow lookahead allows Reach 1 to anticipate the incoming
    upstream torrent during initial CFL timestep estimation, avoiding numerical blowup in
    the first substep.
    """
    ldd: np.ndarray = np.array([[6, 5]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 2
    river_lengths: np.ndarray = np.array([1000.0, 50.0], dtype=np.float32)
    river_widths: np.ndarray = np.array([10.0, 10.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([10.0, 5.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1], dtype=np.int32),
            "slope": np.array([0.005, 0.005], dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.03, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=8.0,
        floodplain_width_m=np.full(n_cells, 30.0, dtype=np.float32),
    )

    storage: np.ndarray = np.zeros(n_cells, dtype=np.float64)
    discharge: np.ndarray = np.zeros(n_cells, dtype=np.float32)
    sideflow: np.ndarray = np.array([150.0 * 3600.0, 0.0], dtype=np.float32)

    step_result = router.step(
        Q_prev_m3_s=discharge,
        river_storage_m3=storage,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    out_q: np.ndarray = step_result[0]
    out_s: np.ndarray = step_result[1]
    assert np.all(np.isfinite(out_q))
    assert np.all(np.isfinite(out_s))
    assert np.all(out_s >= 0.0)
    assert out_q[0] > 100.0
    assert out_q[1] > 100.0


def test_local_inertial_dynamic_cfl_with_intense_lateral_sideflow() -> None:
    """Verify dynamic micro-substepping stability under intense lateral runoff.

    Reaches with low baseflow are subjected to sudden lateral runoff (300 m³/s into
    reach 0, 200 m³/s into reach 1). Verifies that dynamic CFL re-evaluation accounts
    for lateral sideflow during substeps, remaining numerically stable and conserving mass.
    """
    ldd: np.ndarray = np.array([[6, 5]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 2
    river_lengths: np.ndarray = np.array([400.0, 400.0], dtype=np.float32)
    river_widths: np.ndarray = np.array([20.0, 20.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([10.0, 8.0], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1], dtype=np.int32),
            "slope": np.array([0.005, 0.005], dtype=np.float32),
        },
        index=river_ids,
    )

    router: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_lengths,
        river_width=river_widths,
        rivers_gdf=rivers_gdf,
        bankfull_river_elevation_m=bed_elevations,
        manning_n=np.full(n_cells, 0.035, dtype=np.float32),
        shape_exponent=0.5,
        bankfull_depth_m=6.0,
        floodplain_width_m=np.full(n_cells, 40.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([500.0, 500.0], dtype=np.float64)
    discharge: np.ndarray = np.array([1.0, 1.0], dtype=np.float32)
    sideflow: np.ndarray = np.array(
        [300.0 * dt_seconds, 200.0 * dt_seconds], dtype=np.float32
    )

    step_result = router.step(
        Q_prev_m3_s=discharge,
        river_storage_m3=storage,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )

    out_q: np.ndarray = step_result[0]
    out_s: np.ndarray = step_result[1]
    assert np.all(np.isfinite(out_q))
    assert np.all(np.isfinite(out_s))
    assert np.all(out_s >= 0.0)
    assert out_q[0] > 200.0
    assert out_q[1] > 400.0


def test_router_save_and_restore(tmp_path: Path) -> None:
    """Test router state save and restore via RoutingVariables bucket across a checkpoint boundary.

    Verifies that when a river network with dynamic channel widths and inertial routing is
    simulated, saved to disk via a RoutingVariables bucket, and restored into a new router
    instance (reproducing the transition between spinup and the main run):
    1. Dynamic channel widths, reach storage, stage, and previous substep discharges are preserved.
    2. The restored router continues simulation with exact numerical continuity (zero discontinuity).
    3. Discharges, storages, and water surface stages match continuous uninterrupted simulation step-for-step.

    Args:
        tmp_path: Temporary directory fixture provided by pytest for saving bucket data.
    """
    n_cells: int = 4
    ldd: np.ndarray = np.array([[2], [2], [2], [5]], dtype=np.uint8)
    mask: np.ndarray = np.ones((4, 1), dtype=bool)
    river_network: pyflwdir.FlwdirRaster = create_river_network(
        ldd, mask, transform=Affine.identity()
    )

    bed_elevation: ArrayFloat32 = np.array([40.0, 30.0, 20.0, 10.0], dtype=np.float32)
    river_length: ArrayFloat32 = np.array(
        [1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32
    )
    initial_width: ArrayFloat32 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    # Reach 0 is kinematic overland, Reaches 1-3 are inertial channels
    use_kinematic: np.ndarray = np.array([True, False, False, False], dtype=bool)
    river_ids: np.ndarray = np.array([0, 1, 2, 3], dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, 2, 3, -1], dtype=np.int32),
            "slope": np.array([0.002, 0.002, 0.002, 0.002], dtype=np.float32),
        },
        index=river_ids,
    )
    bankfull_depth: ArrayFloat32 = np.full(n_cells, 2.0, dtype=np.float32)

    router_A: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=initial_width,
        river_ids=river_ids,
        bankfull_river_elevation_m=bed_elevation,
        bankfull_depth_m=bankfull_depth,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
    )

    storage_A: np.ndarray = np.array([50.0, 100.0, 150.0, 200.0], dtype=np.float64)
    discharge_A: ArrayFloat32 = np.array([0.1, 0.5, 1.0, 1.5], dtype=np.float32)
    router_A.initialize_stage(river_storage_m3=storage_A)

    sideflow: ArrayFloat32 = np.array(
        [5.0 * 3600.0, 10.0 * 3600.0, 15.0 * 3600.0, 0.0], dtype=np.float32
    )

    # Step router A for 5 steps (simulating spinup steps)
    for _ in range(5):
        res = router_A.step(
            Q_prev_m3_s=discharge_A,
            river_storage_m3=storage_A,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge_A, storage_A = res[0], res[1]

    # Dynamic channel width calibration occurs during spinup
    dynamic_width: ArrayFloat32 = np.array([2.0, 6.0, 9.0, 12.0], dtype=np.float32)
    router_A.update_channel_width(dynamic_width, bankfull_depth)

    # Take step 5 with updated dynamic channel geometry
    res = router_A.step(
        Q_prev_m3_s=discharge_A,
        river_storage_m3=storage_A,
        sideflow_m3=sideflow,
        evaporation_m3=np.zeros(n_cells, dtype=np.float32),
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
        retention_storage_m3=np.zeros(0, dtype=np.float32),
        retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
    )
    discharge_A, storage_A = res[0], res[1]
    stage_A: ArrayFloat32 = router_A.get_water_stage()

    # Save state at end of spinup into RoutingVariables bucket
    checkpoint_dir: Path = tmp_path / "checkpoint_routing"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rv_saved: RoutingVariables = RoutingVariables()
    rv_saved.river_width = dynamic_width
    rv_saved.river_storage_m3 = storage_A.copy()
    rv_saved.discharge_in_rivers_m3_s_substep = discharge_A.copy()
    rv_saved.water_stage_m = stage_A.copy()

    with ThreadPoolExecutor() as executor:
        futures = rv_saved.save(checkpoint_dir, executor)
        for future in futures:
            future.result()

    # Continue uninterrupted run of Router A for 5 more steps (reference continuation)
    ref_discharges: list[np.ndarray] = []
    ref_storages: list[np.ndarray] = []
    ref_stages: list[np.ndarray] = []
    for _ in range(5):
        res = router_A.step(
            Q_prev_m3_s=discharge_A,
            river_storage_m3=storage_A,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge_A, storage_A = res[0], res[1]
        ref_discharges.append(discharge_A.copy())
        ref_storages.append(storage_A.copy())
        ref_stages.append(router_A.get_water_stage())

    # Restore state into a new RoutingVariables bucket (simulating run start)
    rv_restored: RoutingVariables = RoutingVariables()
    rv_restored.load(checkpoint_dir)

    # Validate restored variables match saved state
    np.testing.assert_allclose(rv_restored.river_width, dynamic_width)
    np.testing.assert_allclose(
        rv_restored.river_storage_m3, storage_A_copy := rv_saved.river_storage_m3
    )
    np.testing.assert_allclose(
        rv_restored.discharge_in_rivers_m3_s_substep,
        rv_saved.discharge_in_rivers_m3_s_substep,
    )
    np.testing.assert_allclose(rv_restored.water_stage_m, rv_saved.water_stage_m)

    # Initialize Router B with restored channel width and depth
    router_B: LocalInertial = _make_local_inertial(
        dt=3600,
        river_network=river_network,
        river_length=river_length,
        river_width=rv_restored.river_width,
        river_ids=river_ids,
        bankfull_river_elevation_m=bed_elevation,
        bankfull_depth_m=bankfull_depth,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
    )

    # Restore stage and storage on Router B
    router_B.initialize_stage(
        water_stage_m=rv_restored.water_stage_m,
        river_storage_m3=rv_restored.river_storage_m3,
    )

    storage_B: np.ndarray = rv_restored.river_storage_m3.copy()
    discharge_B: ArrayFloat32 = rv_restored.discharge_in_rivers_m3_s_substep.copy()

    # Step Router B for 5 steps and verify perfect step-by-step equivalence with Router A
    for step_idx in range(5):
        res = router_B.step(
            Q_prev_m3_s=discharge_B,
            river_storage_m3=storage_B,
            sideflow_m3=sideflow,
            evaporation_m3=np.zeros(n_cells, dtype=np.float32),
            waterbody_storage_m3=np.zeros(0, dtype=np.float64),
            outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
            retention_storage_m3=np.zeros(0, dtype=np.float32),
            retention_activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        )
        discharge_B, storage_B = res[0], res[1]
        stage_B: ArrayFloat32 = router_B.get_water_stage()

        # Assert exact agreement with continuous Router A run
        np.testing.assert_allclose(
            discharge_B, ref_discharges[step_idx], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            storage_B, ref_storages[step_idx], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(stage_B, ref_stages[step_idx], rtol=1e-5, atol=1e-5)

        # Assert no physical anomalies (no NaNs, non-negative flows and storages)
        assert np.all(np.isfinite(discharge_B))
        assert np.all(np.isfinite(storage_B))
        assert np.all(np.isfinite(stage_B))
        assert np.all(discharge_B >= 0.0)
        assert np.all(storage_B >= 0.0)


def test_select_active_rivers() -> None:
    """Test selecting active rivers inside the model domain.

    Validates that:
    1. Outflow segments marked as downstream are excluded.
    2. Segments represented in grid are retained.
    3. Segments not represented in grid are kept only if they connect to represented upstream reaches.
    """
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "is_downstream_outflow": [False, False, True, False, False],
            "is_further_downstream_outflow": [False, False, False, True, False],
            "represented_in_grid": [True, False, True, True, False],
            "downstream_ID": [1, -1, -1, -1, -1],
        },
        index=[0, 1, 2, 3, 4],
    )

    # Reach 0 is active and represented
    # Reach 1 is not represented, but reach 0 flows into it (downstream_ID of 0 is 1), so reach 1 is kept
    # Reach 2 is downstream outflow, excluded
    # Reach 3 is further downstream outflow, excluded
    # Reach 4 is not represented, and has no upstream represented reach, excluded
    active: gpd.GeoDataFrame = select_active_rivers(rivers)
    assert set(active.index) == {0, 1}
