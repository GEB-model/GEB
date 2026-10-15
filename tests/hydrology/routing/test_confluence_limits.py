"""Unit tests verifying confluence forward inflow limiting under diverse extreme and normal flow conditions."""

import geopandas as gpd
import numpy as np
import pyflwdir
import pytest
from affine import Affine

from geb.hydrology.routing.local_inertial import LocalInertial
from tests.hydrology.routing.test_local_inertial import _make_local_inertial


def test_confluence_normal_conditions_zero_limiting() -> None:
    """Verify that under normal conditions, the confluence limiter is completely inactive and does not throttle flow.

    Two tributaries carrying 10 m³/s and 15 m³/s baseflow join a confluence. Under normal conditions,
    available absorption and conveyance capacity exceed incoming flow, so the scaling factor is 1.0.
    The system must converge to exact unthrottled steady state:
        Q_trib1 = 10.0 m³/s, Q_trib2 = 15.0 m³/s, Q_confluence = 25.0 m³/s.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([20.0, 30.0, 20.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([5.5, 5.0, 5.5], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.array([0.0005, 0.0005, 0.0005], dtype=np.float32),
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
        bankfull_depth_m=5.0,
        floodplain_width_m=np.full(n_cells, 100.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([20000.0, 40000.0, 20000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([10.0, 25.0, 15.0], dtype=np.float32)
    sideflow: np.ndarray = np.array(
        [10.0 * dt_seconds, 0.0, 15.0 * dt_seconds], dtype=np.float32
    )

    for _ in range(8):
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

    # Must converge cleanly to exact analytical steady state without any throttle
    assert discharge[0] == pytest.approx(10.0, abs=0.05)
    assert discharge[2] == pytest.approx(15.0, abs=0.05)
    assert discharge[1] == pytest.approx(25.0, abs=0.1)


def test_confluence_extreme_symmetric_flash_flood() -> None:
    """Verify confluence stability when two equal tributaries experience an extreme flash flood surge.

    Two upstream tributary reaches (cells 0 and 2) simultaneously deliver a massive 300 m³/s
    flash flood wave (600 m³/s combined) into a narrow confluence reach (cell 1) on a mild slope.
    The inflow limiter must prevent stage overshooting without numerical checkerboard oscillation,
    ensuring 100% mass conservation and smooth downstream discharge propagation.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([20.0, 25.0, 20.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([5.2, 5.0, 5.2], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.array([0.0002, 0.0002, 0.0002], dtype=np.float32),
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
        bankfull_depth_m=6.0,
        floodplain_width_m=np.full(n_cells, 150.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([20000.0, 25000.0, 20000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([10.0, 20.0, 10.0], dtype=np.float32)

    # Flash flood surge: 300 m3/s on both tributaries simultaneously
    sideflow: np.ndarray = np.array(
        [300.0 * dt_seconds, 0.0, 300.0 * dt_seconds],
        dtype=np.float32,
    )

    init_storage: float = float(np.sum(storage))
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

    q_out: np.ndarray = step_result[0]
    storage_out: np.ndarray = step_result[1]
    outflow_pits: float = float(step_result[6])

    assert np.all(np.isfinite(q_out))
    assert np.all(np.isfinite(storage_out))
    assert np.all(storage_out > 0.0)

    # Confluence discharge must increase strongly to evacuate surge
    assert q_out[1] > 50.0

    # Inflows from both tributaries must remain positive (no negative slap-back)
    assert q_out[0] > 0.0
    assert q_out[2] > 0.0

    # Exact mass balance check for the step
    total_in: float = init_storage + float(np.sum(sideflow))
    total_out: float = float(np.sum(storage_out) + outflow_pits)
    assert np.isclose(total_in, total_out, rtol=1e-4)


def test_confluence_steep_mountain_drop_with_flat_river() -> None:
    """Verify confluence stability when a steep mountain torrent merges with a flat lowland river.

    Tributary 1 (cell 0) has a steep bed drop of 50 m over 1000 m (S0 = 0.05), while Tributary 2
    (cell 2) is a flat lowland reach with S0 = 0.0001 (0.1 m drop). Both discharge into a flat
    valley confluence reach (cell 1). A sudden 150 m³/s flood surge on the mountain stream must
    not cause a non-physical multi-meter stage spike at the confluence, and mass must be strictly conserved.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([15.0, 35.0, 30.0], dtype=np.float32)
    # Cell 0 bed elevation is 55.0m (50m above confluence at 5.0m); Cell 2 is 5.1m (0.1m drop)
    bed_elevations: np.ndarray = np.array([55.0, 5.0, 5.1], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.array([0.05, 0.0001, 0.0001], dtype=np.float32),
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
        bankfull_depth_m=5.0,
        floodplain_width_m=np.full(n_cells, 100.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([5000.0, 35000.0, 30000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([5.0, 25.0, 20.0], dtype=np.float32)

    # 150 m3/s torrential surge on the mountain stream, 5 m3/s on the flat stream
    sideflow: np.ndarray = np.array(
        [150.0 * dt_seconds, 0.0, 5.0 * dt_seconds],
        dtype=np.float32,
    )

    init_storage: float = float(np.sum(storage))
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

    q_out: np.ndarray = step_result[0]
    storage_out: np.ndarray = step_result[1]
    outflow_pits: float = float(step_result[6])

    assert np.all(np.isfinite(q_out))
    assert np.all(np.isfinite(storage_out))
    assert np.all(storage_out > 0.0)

    # Valley confluence water level must rise moderately, well bounded by depth cap
    # Depth in cell 1 (width 35m, length 1000m) should remain physically realistic (< 15m)
    h_conf: float = float(storage_out[1]) / (35.0 * 1000.0)
    assert h_conf < 15.0

    # Exact mass conservation across steep and flat reaches
    total_in: float = init_storage + float(np.sum(sideflow))
    total_out: float = float(np.sum(storage_out) + outflow_pits)
    assert np.isclose(total_in, total_out, rtol=1e-4)


def test_confluence_3_way_tributary_extreme_surge() -> None:
    """Verify stability of a 3-way tributary confluence subjected to an extreme simultaneous surge from dry bed.

    Three upstream tributaries (cells 0, 1, 3) merge into a single trunk reach (cell 2), which
    flows to an exit pit (cell 4). Starting from near-dry bed conditions (0.05 m depth), all three
    tributaries receive an abrupt 100 m³/s inflow pulse (300 m³/s total). The confluence limiter
    must throttle incoming fluxes during the initial filling phase, preventing negative depths or NaN values.
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
        [15.0, 20.0, 30.0, 15.0, 40.0], dtype=np.float32
    )
    bed_elevations: np.ndarray = np.array([5.2, 5.2, 5.0, 5.2, 4.8], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([2, 2, 4, 2, -1], dtype=np.int32),
            "slope": np.full(n_cells, 0.0002, dtype=np.float32),
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
        bankfull_depth_m=6.0,
        floodplain_width_m=np.full(n_cells, 100.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    # Near dry initial storage: 0.05m depth
    storage: np.ndarray = (river_widths * river_lengths * 0.05).astype(np.float64)
    discharge: np.ndarray = np.zeros(n_cells, dtype=np.float32)

    # 100 m3/s pulse into each of the 3 tributaries (cells 0, 1, 3)
    sideflow: np.ndarray = np.zeros(n_cells, dtype=np.float32)
    sideflow[0] = 100.0 * dt_seconds
    sideflow[1] = 100.0 * dt_seconds
    sideflow[3] = 100.0 * dt_seconds

    init_storage: float = float(np.sum(storage))
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

    q_out: np.ndarray = step_result[0]
    storage_out: np.ndarray = step_result[1]
    outflow_pits: float = float(step_result[6])

    assert np.all(np.isfinite(q_out))
    assert np.all(np.isfinite(storage_out))
    assert np.all(storage_out > 0.0)

    # All 3 incoming tributaries must be delivering forward flow into the confluence
    assert q_out[0] > 0.0
    assert q_out[1] > 0.0
    assert q_out[3] > 0.0
    # Confluence cell 2 and exit cell 4 must be actively routing water
    assert q_out[2] > 0.0
    assert q_out[4] > 0.0

    # Exact mass conservation check
    total_in: float = init_storage + float(np.sum(sideflow))
    total_out: float = float(np.sum(storage_out) + outflow_pits)
    assert np.isclose(total_in, total_out, rtol=1e-4)


def test_confluence_downstream_backwater_choke() -> None:
    """Verify confluence inflow limiter throttles tributary inflows when downstream is severely choked.

    Two tributaries (cells 0 and 2) flow into a confluence reach (cell 1). The downstream reach
    experiences an artificial severe stage elevation / backwater obstacle. The confluence stage
    rises towards the tributary stages. Inflow limiting must gracefully restrict incoming fluxes as
    head difference decreases, preventing adverse stage overshooting.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([20.0, 30.0, 20.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([5.1, 5.0, 5.1], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.array([0.0001, 0.0001, 0.0001], dtype=np.float32),
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
        floodplain_width_m=np.full(n_cells, 100.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    # Confluence cell 1 has high initial storage (depth = 4.0 m), close to tributaries (depth = 4.1 m)
    # Head difference is very small (0.2 m)
    storage: np.ndarray = np.array([82000.0, 120000.0, 82000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([10.0, 5.0, 10.0], dtype=np.float32)

    # Moderate continuous inflows
    sideflow: np.ndarray = np.array(
        [20.0 * dt_seconds, 0.0, 20.0 * dt_seconds],
        dtype=np.float32,
    )

    init_storage: float = float(np.sum(storage))
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

    q_out: np.ndarray = step_result[0]
    storage_out: np.ndarray = step_result[1]
    outflow_pits: float = float(step_result[6])

    assert np.all(np.isfinite(q_out))
    assert np.all(np.isfinite(storage_out))
    assert np.all(storage_out > 0.0)

    # Confluence stage must not overshoot the upstream tributary stages
    stage_trib_0: float = 5.1 + float(storage_out[0]) / (20.0 * 1000.0)
    stage_conf_1: float = 5.0 + float(storage_out[1]) / (30.0 * 1000.0)
    assert stage_conf_1 <= stage_trib_0 + 0.1

    # Exact mass balance check
    total_in: float = init_storage + float(np.sum(sideflow))
    total_out: float = float(np.sum(storage_out) + outflow_pits)
    assert np.isclose(total_in, total_out, rtol=1e-4)


def test_confluence_full_storm_hydrograph_exact_mass_balance() -> None:
    """Verify exact mass conservation and clean hydrograph recession across a multi-hour storm cycle.

    A confluence network is subjected to an entire storm hydrograph cycle: baseflow -> rising limb ->
    peak (400 m³/s) -> recession limb -> baseline. Total inflow, total outflow, and storage changes
    are tracked over 8 hours, verifying relative mass balance error < 1e-4.
    """
    ldd: np.ndarray = np.array([[2, 5], [5, 4]], dtype=np.uint8)
    mask: np.ndarray = np.array([[True, False], [True, True]], dtype=bool)
    river_network: pyflwdir.FlwdirRaster = pyflwdir.from_array(
        ldd, ftype="ldd", mask=mask, transform=Affine.identity()
    )
    n_cells: int = 3
    river_lengths: np.ndarray = np.full(n_cells, 1000.0, dtype=np.float32)
    river_widths: np.ndarray = np.array([20.0, 40.0, 20.0], dtype=np.float32)
    bed_elevations: np.ndarray = np.array([5.5, 5.0, 5.5], dtype=np.float32)
    river_ids: np.ndarray = np.arange(n_cells, dtype=np.int32)
    rivers_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "downstream_ID": np.array([1, -1, 1], dtype=np.int32),
            "slope": np.array([0.0005, 0.0005, 0.0005], dtype=np.float32),
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
        bankfull_depth_m=6.0,
        floodplain_width_m=np.full(n_cells, 120.0, dtype=np.float32),
    )

    dt_seconds: int = 3600
    storage: np.ndarray = np.array([10000.0, 20000.0, 10000.0], dtype=np.float64)
    discharge: np.ndarray = np.array([5.0, 10.0, 5.0], dtype=np.float32)

    # 10-hour storm hydrograph on Tributary 1 (Trib 2 maintains 5 m3/s baseflow)
    storm_hydrograph: list[float] = [
        5.0,
        50.0,
        200.0,
        400.0,
        250.0,
        80.0,
        15.0,
        5.0,
        5.0,
        5.0,
    ]

    initial_storage: float = float(np.sum(storage))
    cum_inflow: float = 0.0
    cum_outflow: float = 0.0

    for q_storm in storm_hydrograph:
        sideflow: np.ndarray = np.array(
            [q_storm * dt_seconds, 0.0, 5.0 * dt_seconds],
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
        assert np.all(storage > 0.0)

    # Verify exact mass balance across the entire storm event
    final_storage: float = float(np.sum(storage))
    mass_balance_error: float = abs(
        cum_inflow - (cum_outflow + final_storage - initial_storage)
    )
    assert mass_balance_error / cum_inflow < 1e-4

    # Verify clean recovery to near baseflow
    assert discharge[0] == pytest.approx(5.0, abs=2.0)
    assert discharge[2] == pytest.approx(5.0, abs=1.0)
    assert discharge[1] == pytest.approx(10.0, abs=2.0)
