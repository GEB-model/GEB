"""Tests for river snapping workflows."""

import geopandas as gpd
import numpy as np
import shapely.geometry
import xarray as xr

from geb.build.workflows.river_snapping import (
    select_river_segment,
    snap_point_to_river_network,
)


def test_select_river_segment_dominant_within_buffer() -> None:
    """Test selecting dominant river within buffer and falling back to closest river when none is within buffer."""
    point = shapely.geometry.Point(5.0, 50.0)

    # Distances in degrees: ~111,139 m per deg latitude
    # river1: distance 0.005 deg (~555 m), small uparea
    # river2: distance 0.015 deg (~1667 m), large uparea (dominant)
    # river3: distance 0.050 deg (~5557 m), huge uparea
    geom1 = shapely.geometry.LineString([(5.0, 50.005), (5.01, 50.005)])
    geom2 = shapely.geometry.LineString([(5.0, 50.015), (5.01, 50.015)])
    geom3 = shapely.geometry.LineString([(5.0, 50.050), (5.01, 50.050)])

    rivers = gpd.GeoDataFrame(
        {
            "geometry": [geom1, geom2, geom3],
            "station_distance": [0.005, 0.015, 0.050],
            "uparea_m2": [10_000.0, 1_000_000.0, 10_000_000.0],
            "shreve_stream_order": [1, 3, 4],
            "hydrography_xy": [[(0, 0)], [(1, 1)], [(2, 2)]],
        },
        index=[101, 102, 103],
        crs="EPSG:4326",
    )

    # Within 2000m buffer, river 2 should be selected as dominant over river 1
    selected = select_river_segment(
        rivers_sorted=rivers,
        max_spatial_difference_degrees=1.0,
        dominant_river_buffer_m=2000.0,
        point=point,
    )
    assert selected is not None
    assert selected.ID == 102
    assert selected.uparea_m2 == 1_000_000.0

    # Fallback to closest river if no candidate is within dominant buffer
    selected_outside = select_river_segment(
        rivers_sorted=rivers,
        max_spatial_difference_degrees=1.0,
        dominant_river_buffer_m=300.0,
        point=point,
    )
    assert selected_outside is not None
    assert selected_outside.ID == 101

    # Returns None if closest river exceeds max_spatial_difference_degrees
    selected_too_far = select_river_segment(
        rivers_sorted=rivers,
        max_spatial_difference_degrees=0.001,
        dominant_river_buffer_m=300.0,
        point=point,
    )
    assert selected_too_far is None


def test_snap_point_to_river_network_waterbody_handling() -> None:
    """Test that snapping excludes waterbodies and returns None without fallback if all cells are in waterbody."""
    point = shapely.geometry.Point(5.0, 50.0)

    # River line passing near point
    geom = shapely.geometry.LineString([(5.0, 50.001), (5.02, 50.001)])
    rivers = gpd.GeoDataFrame(
        {
            "geometry": [geom],
            "represented_in_grid": [True],
            "uparea_m2": [50_000.0],
            "shreve_stream_order": [2],
            "hydrography_xy": [[(0, 0), (1, 0)]],
        },
        index=[1],
        crs="EPSG:4326",
    )

    x_coords = np.array([5.0, 5.01])
    y_coords = np.array([50.0, 50.01])

    uparea_grid = xr.DataArray(
        np.array([[50_000.0, 50_000.0], [50_000.0, 50_000.0]]),
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
    )
    uparea_subgrid = uparea_grid.copy()

    # Case 1: All cells on river segment are inside a lake/reservoir (waterbody_id >= 0)
    wb_all = np.array([[42, 42], [-1, -1]], dtype=np.int32)
    res_all_wb = snap_point_to_river_network(
        point=point,
        rivers=rivers,
        upstream_area_grid=uparea_grid,
        upstream_area_subgrid=uparea_subgrid,
        waterbody_id_grid=wb_all,
    )
    # Must return None without falling back to the waterbody cells
    assert res_all_wb is None

    # Case 2: One cell is in waterbody, one cell is valid river
    wb_partial = np.array([[42, -1], [-1, -1]], dtype=np.int32)
    res_partial = snap_point_to_river_network(
        point=point,
        rivers=rivers,
        upstream_area_grid=uparea_grid,
        upstream_area_subgrid=uparea_subgrid,
        waterbody_id_grid=wb_partial,
    )
    assert res_partial is not None
    # Cell (1, 0) should be selected because (0, 0) is inside waterbody 42
    assert res_partial.snapped_grid_pixel_xy == (1, 0)
