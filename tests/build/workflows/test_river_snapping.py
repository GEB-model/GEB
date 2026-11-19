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
            "station_distance_m": [556.0, 1668.0, 5560.0],
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
    )
    assert selected is not None
    assert selected.ID == 102
    assert selected.uparea_m2 == 1_000_000.0

    # Fallback to closest river if no candidate is within dominant buffer
    selected_outside = select_river_segment(
        rivers_sorted=rivers,
        max_spatial_difference_degrees=1.0,
        dominant_river_buffer_m=300.0,
    )
    assert selected_outside is not None
    assert selected_outside.ID == 101

    # Returns None if closest river exceeds max_spatial_difference_degrees
    selected_too_far = select_river_segment(
        rivers_sorted=rivers,
        max_spatial_difference_degrees=0.001,
        dominant_river_buffer_m=300.0,
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
    wb_all = xr.DataArray(
        np.array([[42, 42], [-1, -1]], dtype=np.int32),
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
    )
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
    wb_partial = xr.DataArray(
        np.array([[42, -1], [-1, -1]], dtype=np.int32),
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
    )
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


def test_search_radius_is_the_same_in_every_direction() -> None:
    """A large river 1.6 km to the east is within a 2 km radius, like one 1.6 km to the north.

    At 49 degrees north, 1.6 km is 0.0144 degrees of latitude but 0.0219 degrees of
    longitude. A small stream runs 500 m south of the point in both cases; the large river
    must be selected whether it lies to the east or to the north, and not when it lies
    2.4 km away.
    """
    point = shapely.geometry.Point(8.0, 49.0)
    degrees_per_m_north = 1.0 / 111_195.0
    degrees_per_m_east = degrees_per_m_north / np.cos(np.radians(49.0))
    grid = xr.DataArray(
        np.full((2, 2), 1.0),
        coords={"y": [49.0, 49.1], "x": [8.0, 8.1]},
        dims=("y", "x"),
    )
    no_waterbodies = xr.full_like(grid, -1).astype(np.int32)
    small_stream = shapely.geometry.LineString(
        [
            (7.9, 49.0 - 500 * degrees_per_m_north),
            (8.1, 49.0 - 500 * degrees_per_m_north),
        ]
    )

    def selected_river(large_river: shapely.geometry.LineString) -> int:
        rivers = gpd.GeoDataFrame(
            {
                "geometry": [small_stream, large_river],
                "represented_in_grid": [True, True],
                "uparea_m2": [1e8, 5e10],
                "shreve_stream_order": [2, 500],
                "hydrography_xy": [[(0, 0)], [(1, 1)]],
            },
            index=[1, 2],
            crs="EPSG:4326",
        )
        result = snap_point_to_river_network(
            point=point,
            rivers=rivers,
            upstream_area_grid=grid,
            upstream_area_subgrid=grid,
            waterbody_id_grid=no_waterbodies,
            max_spatial_difference_degrees=180,
            dominant_river_buffer_m=2000.0,
        )
        assert result is not None
        return result.closest_river_segment.ID

    def river_to_the_east(distance_m: float) -> shapely.geometry.LineString:
        x = 8.0 + distance_m * degrees_per_m_east
        return shapely.geometry.LineString([(x, 48.9), (x, 49.1)])

    def river_to_the_north(distance_m: float) -> shapely.geometry.LineString:
        y = 49.0 + distance_m * degrees_per_m_north
        return shapely.geometry.LineString([(7.9, y), (8.1, y)])

    assert selected_river(river_to_the_east(1600.0)) == 2
    assert selected_river(river_to_the_north(1600.0)) == 2
    assert selected_river(river_to_the_east(2400.0)) == 1
    assert selected_river(river_to_the_north(2400.0)) == 1
