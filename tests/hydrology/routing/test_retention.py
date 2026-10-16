"""Tests for hydrological routing."""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pyflwdir
import pytest
from affine import Affine

from geb.hydrology.routing import (
    create_river_network,
)
from geb.hydrology.routing.local_inertial import LocalInertial


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
        waterbody_storage_m3=np.zeros(0, dtype=np.float64),
        outflow_per_waterbody_m3=np.zeros(0, dtype=np.float32),
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
