"""Tests for lake and reservoir functions in GEB."""

import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from geb.hydrology.waterbodies import (
    estimate_lake_outflow,
    estimate_outflow_height,
    get_lake_factor,
    get_lake_height_above_outflow,
    get_lake_height_from_bottom,
    get_lake_outflow,
    get_lake_storage_from_height_above_bottom,
    get_river_width,
)

from ..testconfig import output_folder


def test_get_lake_height_from_bottom() -> None:
    """Test calculation of lake height from storage and area.

    Verifies that lake height is correctly calculated as storage divided by area,
    and that the inverse function produces consistent results.
    """
    lake_area = np.array([100]).astype(np.float32)
    lake_storage = np.linspace(0, 1000, 100).astype(np.float64)

    lake_height = get_lake_height_from_bottom(
        lake_storage=lake_storage, lake_area=lake_area
    )

    np.testing.assert_allclose(
        lake_height,
        np.linspace(0, 10, 100),
    )

    np.testing.assert_allclose(
        lake_storage,
        get_lake_storage_from_height_above_bottom(
            lake_height=lake_height, lake_area=lake_area
        ),
    )


def test_get_lake_storage_from_height_above_bottom() -> None:
    """Test calculation of lake storage from height and area.

    Verifies that lake storage is correctly calculated as height times area,
    and that the inverse function produces consistent results.
    """
    lake_area = np.array([100])
    lake_height = np.linspace(0, 10, 100).astype(np.float32)

    lake_storage = get_lake_storage_from_height_above_bottom(
        lake_height=lake_height, lake_area=lake_area
    )

    np.testing.assert_allclose(
        lake_storage,
        np.linspace(0, 1000, 100),
    )

    np.testing.assert_allclose(
        lake_height,
        get_lake_height_from_bottom(lake_storage=lake_storage, lake_area=lake_area),
    )


def test_estimate_initial_lake_storage_and_outflow_height() -> None:
    """Test estimation of lake outflow height and storage dynamics.

    Tests the complete lake outflow estimation workflow including:
    - Calculation of river width from average discharge
    - Lake factor computation using overflow coefficients
    - Outflow height estimation from capacity and average outflow
    - Verification of outflow-to-height inverse relationships
    - Storage dynamics simulation with inflow, outflow, and evaporation
    - Generation of diagnostic plots for lake behavior analysis
    """
    lake_area = np.array([3_480_000.0])
    lake_capacity = np.array([7_630_000.0])
    avg_outflow = np.array([2.494])
    river_width = get_river_width(avg_outflow)
    lake_factor = get_lake_factor(
        river_width=river_width,
        overflow_coefficient_mu=np.float32(0.577),
        lake_a_factor=np.float32(1),
    )

    outflow_height = estimate_outflow_height(
        lake_capacity=lake_capacity,
        lake_factor=lake_factor,
        lake_area=lake_area,
        avg_outflow=avg_outflow,
    )

    lake_storage = lake_capacity.copy()
    height_above_outflow = get_lake_height_above_outflow(
        lake_storage=lake_storage, lake_area=lake_area, outflow_height=outflow_height
    )

    assert math.isclose(
        ((height_above_outflow + outflow_height) * lake_area)[0],
        lake_storage[0],
        rel_tol=1e-5,
    )
    outflow = estimate_lake_outflow(
        lake_factor=lake_factor, height_above_outflow=height_above_outflow
    )

    # test if outflow_to_height_above_outflow is indeed the inverse of estimate_lake_outflow
    assert math.isclose(outflow[0], avg_outflow[0])

    lake_storage = np.linspace(0, lake_storage[0] * 1.2, 100)
    height_above_outflow = get_lake_height_above_outflow(
        lake_storage=lake_storage, lake_area=lake_area, outflow_height=outflow_height
    )
    outflow = estimate_lake_outflow(
        lake_factor=lake_factor, height_above_outflow=height_above_outflow
    )

    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        lake_storage, height_above_outflow, color="red", label="height_above_outflow"
    )
    ax_right.plot(lake_storage, outflow, color="blue", label="outflow")

    # plot vertical line of initial storage
    ax_left.axvline(x=lake_storage[0], color="green", linestyle="--")

    # plot horizontal line of average outflow
    ax_right.axhline(y=avg_outflow[0], color="black", linestyle="--")

    ax_left.set_ylim(0, None)
    ax_right.set_ylim(0, None)

    ax_left.set_xlim(lake_storage[0].item(), lake_storage[-1].item())

    ax_left.set_xlabel("Storage")
    ax_left.set_ylabel("Height above outflow (red)")
    ax_right.set_ylabel("Outflow (blue)")

    plt.savefig(output_folder / "estimate_outflow_height.png")
    plt.close()

    storage = lake_storage.copy()

    dt = 3600

    inflow_m3_s = np.full(1_000, 0, dtype=np.float32)
    inflow_m3_s[:200] = np.linspace(0, 10, 200)
    inflow_m3_s[200:400] = avg_outflow[0]
    inflow_m3_s[400:] = np.linspace(avg_outflow[0], 0, 600)
    inflow = inflow_m3_s * dt

    evaporation = avg_outflow[0] * 3600 * 0.5

    new_storages = np.full(inflow.size, np.nan)
    new_outflows = np.full(inflow.size, np.nan)
    new_height_above_outflows = np.full(inflow.size, np.nan)
    for i in range(inflow.size):
        storage += inflow[i]

        outflow, lake_height_above_outflow = get_lake_outflow(
            dt,
            storage,
            lake_factor,
            lake_area,
            outflow_height,
        )

        # evaporation
        storage -= outflow
        storage -= evaporation

        new_storages[i] = storage[0]
        new_outflows[i] = outflow[0]
        new_height_above_outflows[i] = lake_height_above_outflow[0]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

    ax0.plot(new_storages, color="red", label="storage (red)")
    ax0.axhline(y=lake_storage[0], color="red", linestyle="--")
    ax0.axhline(y=outflow_height[0] * lake_area[0], color="black", linestyle="--")

    ax1.plot(new_outflows, color="blue", label="outflow (blue)")
    ax1.plot(inflow, color="green", label="inflow (green)")
    ax2.plot(new_height_above_outflows, color="black", label="height_above_outflow")

    ax0.set_ylabel("Storage")
    ax1.set_ylabel("Outflow (blue) / Inflow (green)")
    ax2.set_ylabel("Height above outflow")

    ax0.set_ylim(0, None)
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)

    plt.savefig(output_folder / "get_lake_outflow_and_storage.png")

    plt.close()


def test_waterbodies_depth_and_stage_calculations() -> None:
    """Test WaterBodies.water_depth_from_bottom and WaterBodies.get_water_stage_m calculations."""
    from unittest.mock import MagicMock

    from geb.hydrology.waterbodies import WaterBodies

    mock_model = MagicMock()
    mock_model.in_spinup = False
    mock_model.config = {
        "general": {"hydrological_year_start_month": 10},
        "parameters": {"lake_outflow_multiplier": np.float32(1.0)},
    }
    mock_hydrology = MagicMock()

    wb = WaterBodies(model=mock_model, hydrology=mock_hydrology)
    wb.var.lake_area = np.array([200000.0], dtype=np.float32)
    wb.var.outflow_height = np.array([5.0], dtype=np.float32)
    wb.var.storage = np.array([1000000.0], dtype=np.float64)

    outflow_bed_elev = np.array([25.0], dtype=np.float32)

    # 1. Normal lake storage (storage = outflow_height * lake_area)
    wb.var.storage[0] = float(wb.var.outflow_height[0] * wb.var.lake_area[0])
    depth = wb.water_depth_from_bottom
    stage = wb.get_water_stage_m(outflow_bed_elev)
    assert np.isclose(depth[0], wb.var.outflow_height[0])
    assert np.isclose(stage[0], outflow_bed_elev[0])

    # 2. Flooded lake storage (storage increased by 200,000 m3 over 200,000 m2 area = +1.0 m)
    wb.var.storage[0] += 200000.0
    stage_flooded = wb.get_water_stage_m(outflow_bed_elev)
    assert np.isclose(stage_flooded[0], outflow_bed_elev[0] + 1.0)

    # 3. Dry lake storage (storage = 0 m3) -> stage should equal bottom elevation
    wb.var.storage[0] = 0.0
    stage_dry = wb.get_water_stage_m(outflow_bed_elev)
    bottom_elev = outflow_bed_elev[0] - wb.var.outflow_height[0]
    assert np.isclose(stage_dry[0], bottom_elev)


def test_flatten_waterbody_elevations() -> None:
    """Test that flatten_waterbody_elevations flattens all cells within a waterbody to its outlet elevation."""
    from unittest.mock import MagicMock

    from geb.hydrology.waterbodies import WaterBodies

    mock_model = MagicMock()
    mock_model.in_spinup = False
    mock_model.config = {
        "general": {"hydrological_year_start_month": 10},
        "parameters": {"lake_outflow_multiplier": np.float32(1.0)},
    }
    mock_hydrology = MagicMock()
    mock_grid = MagicMock()
    mock_grid.compressed_size = 5

    wb = WaterBodies(model=mock_model, hydrology=mock_hydrology)
    wb.grid = mock_grid

    # 4 cells: cell 0 and 1 are in waterbody 0; cell 2 is in waterbody 1; cell 3 is a normal river cell (-1)
    wb.grid.var.waterbody_ids = np.array([0, 0, 1, -1], dtype=np.int32)
    # Outlet for waterbody 0 is at cell 1; outlet for waterbody 1 is at cell 2
    wb.grid.var.waterbody_outflow_points = np.array([-1, 0, 1, -1], dtype=np.int32)
    wb.var.waterbody_outflow_linear_mapping = np.array([1, 2], dtype=np.int32)

    wb.var.waterbodies = gpd.GeoDataFrame({"elevation": [15.0, 80.0]})

    # Initial bed elevations: cell 0 (lake shore) is 300 m, cell 1 (lake outlet) is 15 m
    raw_elev = np.array([300.0, 15.0, 80.0, 120.0], dtype=np.float32)

    flattened = wb.flatten_waterbody_elevations(raw_elev)

    # Cell 0 must be flattened to match outlet cell 1 (15 m)
    assert flattened[0] == 15.0
    assert flattened[1] == 15.0
    assert flattened[2] == 80.0
    assert flattened[3] == 120.0


def test_off_waterbodies_filtered_out_at_spinup() -> None:
    """Test that waterbodies with waterbody_type == 0 (OFF) are excluded during spinup."""
    from unittest.mock import MagicMock

    import geopandas as gpd
    import pandas as pd

    from geb.hydrology.waterbodies import LAKE, OFF, RESERVOIR, WaterBodies

    mock_model = MagicMock()
    mock_model.in_spinup = False
    mock_model.config = {
        "general": {"hydrological_year_start_month": 10},
        "parameters": {"lake_outflow_multiplier": np.float32(1.0)},
    }
    mock_hydrology = MagicMock()
    mock_grid = MagicMock()
    mock_grid.compressed_size = 5

    # Raw grid waterbody_id has 3 waterbodies: 10, 20, 30
    # 10: LAKE (active), 20: OFF (disabled), 30: RESERVOIR (active)
    raw_wb_id = np.array([10, 10, 20, 30, -1], dtype=np.int32)
    mock_grid.load2d.return_value = raw_wb_id
    mock_hydrology.routing.var.discharge_in_rivers_m3_s_substep = np.zeros(
        5, dtype=np.float32
    )

    upstream_area = np.array([100, 200, 50, 300, 400], dtype=np.int32)
    mock_hydrology.routing.grid.var.upstream_area_n_cells = upstream_area

    wb_data = gpd.GeoDataFrame(
        {
            "waterbody_type": [LAKE, OFF, RESERVOIR],
            "average_area": [1e6, 2e6, 3e6],
            "volume_total": [5e6, 1e7, 2e7],
            "average_discharge": [10.0, 20.0, 30.0],
        },
        index=pd.Index([10, 20, 30], name="waterbody_id"),
    )

    wb = WaterBodies(model=mock_model, hydrology=mock_hydrology)
    wb.grid = mock_grid
    wb.model.files = {
        "grid": {
            "waterbodies/waterbody_id": "dummy_grid_path",
            "routing/bankfull_river_elevation_m": "dummy_grid_path",
        },
        "geom": {"waterbodies/waterbody_data": "dummy_geom_path"},
    }

    # Patch read_geom to return wb_data
    import pytest

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "geb.hydrology.waterbodies.read_geom", lambda path: wb_data.reset_index()
        )
        wb.spinup()

    # Active waterbodies should only be 10 (LAKE) and 30 (RESERVOIR) -> mapped to 0 and 1
    assert len(wb.var.waterbodies) == 2
    assert set(wb.var.waterbodies.index) == {10, 30}
    # Cell 2 (originally waterbody 20 which is OFF) must now be -1
    assert wb.grid.var.waterbody_ids[2] == -1
    # Cells for waterbody 10 and 30 must have valid mapped IDs (0 and 1)
    assert wb.grid.var.waterbody_ids[0] != -1
    assert wb.grid.var.waterbody_ids[1] != -1
    assert wb.grid.var.waterbody_ids[3] != -1
