"""Tests for river water-surface profiles from a DEM."""

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString, Polygon

from geb.workflows.river_profiles import (
    estimate_river_surface_profiles,
    river_bed_profiles,
)

PIXEL_M: float = 30.0


def make_case() -> tuple[gpd.GeoDataFrame, xr.DataArray, gpd.GeoDataFrame]:
    """A main river (west to east, 12 km) with a tributary joining at 6 km.

    The DEM is land at 100 m with a water surface along the main river that falls
    1 m per km but is stored in 0.5 m terraces (as in FathomDEM). It contains a gravel
    pit next to the river (hole), a bridge (bump), a lake and the sea at the end (no
    data).

    Returns:
        Rivers, DEM and lake polygons (EPSG:32631).
    """
    nx, ny = 420, 120
    x = np.arange(nx) * PIXEL_M + PIXEL_M / 2
    y = (np.arange(ny) * PIXEL_M + PIXEL_M / 2)[::-1]
    dem = np.full((ny, nx), 100.0, dtype=np.float32)
    river_row = 60
    true_surface = 20.0 - x / 1000.0  # 1 m per km
    dem[river_row, :] = np.floor(true_surface / 0.5) * 0.5
    dem[river_row + 1, 60:70] = 5.0  # gravel pit next to the river (hole)
    dem[river_row - 1 : river_row + 2, 100] = 30.0  # bridge over the river (bump)
    lake_cols = slice(260, 300)
    dem[river_row, lake_cols] = 12.0  # lake level
    dem[:, 400:] = np.nan  # sea
    # tributary from the north joining at x = 6 km, falling 2 m per km
    trib_col = 200
    rows = np.arange(0, river_row)
    dem[rows, trib_col] = 14.0 + (river_row - rows) * PIXEL_M / 500.0
    elevation = xr.DataArray(dem, coords={"y": y, "x": x}, dims=("y", "x"))
    elevation = elevation.rio.write_crs(32631)

    yr = y[river_row]
    junction = (x[trib_col], yr)
    rivers = gpd.GeoDataFrame(
        {
            "downstream_ID": [2, -1, 2],
            "topological_stream_order": [1, 2, 1],
            "uparea_m2": [5e8, 6e8, 1e8],
            "depth": [2.0, 3.0, 1.0],
        },
        index=pd.Index([1, 2, 3], name="COMID"),
        geometry=[
            LineString([(x[0], yr), junction]),
            LineString([junction, (x[-1], yr)]),
            LineString([(x[trib_col], y[0]), junction]),
        ],
        crs=32631,
    )
    lake = gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (x[260] - 15, yr - 100),
                    (x[299] + 15, yr - 100),
                    (x[299] + 15, yr + 100),
                    (x[260] - 15, yr + 100),
                ]
            )
        ],
        crs=32631,
    )
    return rivers, elevation, lake


def test_river_surface_profiles() -> None:
    """Holes are filled, bumps cut, terraces smoothed, lakes flat, the sea at 0 m."""
    rivers, elevation, lake = make_case()
    profiles = estimate_river_surface_profiles(rivers, elevation, lakes=lake)
    z = profiles.set_index(["river_id", "position"])["water_surface_elevation_m"]
    x = profiles.geometry.x.to_numpy()

    # monotone along each river and across the junction
    for river_id, g in profiles.groupby("river_id"):
        assert (np.diff(g["water_surface_elevation_m"].to_numpy()) <= 1e-6).all()
    assert z.loc[1].iloc[-1] >= z.loc[2].iloc[0]
    assert z.loc[3].iloc[-1] >= z.loc[2].iloc[0]

    main = profiles[profiles["river_id"].isin([1, 2])]
    on_land = main[(main.geometry.x > 3500) & (main.geometry.x < 7500)]
    true_surface = 20.0 - on_land.geometry.x.to_numpy() / 1000.0
    # terraces are smoothed: within 0.5 m of the true surface and no 1 km flat
    assert (
        np.abs(on_land["water_surface_elevation_m"].to_numpy() - true_surface).max()
        < 0.5
    )
    zs = on_land["water_surface_elevation_m"].to_numpy()
    assert (zs[:-33] - zs[33:] > 0.1).all()
    # the gravel pit (5 m) does not pull the river down, the bridge (30 m) is cut
    near_pit = main[(x[main.index] > 1800) & (x[main.index] < 2100)]
    assert near_pit["water_surface_elevation_m"].min() > 17.0
    near_bridge = main[np.abs(x[main.index] - 3015) < 50]
    assert near_bridge["water_surface_elevation_m"].max() < 18.0
    # the lake stays flat at its level (the sampling window of the last lake point
    # reaches the river below the outlet)
    in_lake = main[main["is_lake"]]
    assert len(in_lake) > 10
    np.testing.assert_allclose(in_lake["water_surface_elevation_m"].iloc[:-1], 12.0)
    # no DEM data: sea level
    at_sea = main[main.geometry.x > 12100]
    np.testing.assert_allclose(at_sea["water_surface_elevation_m"], 0.0)


def test_river_bed_profiles() -> None:
    """The bed is the surface minus the depth and falls downstream."""
    rivers, elevation, lake = make_case()
    profiles = estimate_river_surface_profiles(rivers, elevation, lakes=lake)
    beds = river_bed_profiles(rivers, profiles)
    for river_id, g in beds.groupby("river_id"):
        bed = g["bed_elevation_m"].to_numpy()
        assert (np.diff(bed) <= 1e-6).all()
        assert (bed <= g["water_surface_elevation_m"].to_numpy() - 0.99).all()
    # junction depth is the largest of the meeting rivers (3 m)
    start_main = beds[(beds["river_id"] == 2) & (beds["position"] == 0)].iloc[0]
    assert (
        start_main["water_surface_elevation_m"] - start_main["bed_elevation_m"]
        >= 3.0 - 1e-5
    )
