"""Verify vectorized grid snapping and reuse of river coordinates."""

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point

from geb.build.workflows.river_snapping import (
    SnappingResults,
    snap_point_to_river_network,
)


@pytest.mark.parametrize("sample_subgrid", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_nearest_cell_tie_and_cache(use_cache: bool, sample_subgrid: bool) -> None:
    """Keep the first equidistant cell and reuse coordinates across barriers.

    Args:
        use_cache: Whether coordinate caching is enabled.
        sample_subgrid: Whether to sample subgrid diagnostics.

    Returns:
        None.

    Raises:
        AssertionError: If snapping or cache reuse changes the selected cell.
    """  # noqa: DOC202, DOC502
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "represented_in_grid": [True],
            "uparea_m2": [10.0],
            "hydrography_xy": [[(0, 0), (1, 0)]],
            "shreve_stream_order": [1],
        },
        geometry=[LineString([(0, 0), (0.02, 0)])],
        index=pd.Index([1], dtype="int64"),
        crs=4326,
    )
    area: xr.DataArray = xr.DataArray(
        [[10.0, 20.0]], dims=("y", "x"), coords={"y": [0.0], "x": [0.0, 0.02]}
    )
    cache: dict[int, npt.NDArray[np.float64]] | None = {} if use_cache else None
    result: SnappingResults | None = snap_point_to_river_network(
        Point(0.01, 0),
        rivers,
        area,
        area if sample_subgrid else None,
        river_coordinate_cache=cache,
    )
    assert result is not None
    assert result.snapped_grid_pixel_xy == (0, 0)
    saved_coordinates: npt.NDArray[np.float64] | None = (
        cache[1] if cache is not None else None
    )
    result = snap_point_to_river_network(
        Point(0.019, 0),
        rivers,
        area,
        area if sample_subgrid else None,
        river_coordinate_cache=cache,
    )
    assert result is not None
    assert result.snapped_grid_pixel_xy == (1, 0)
    assert result.geb_uparea_grid == 20.0
    if sample_subgrid:
        assert result.geb_uparea_subgrid == 20.0
        assert result.subgrid_pixel_coords == (0.02, 0.0)
    else:
        assert np.isnan(result.geb_uparea_subgrid)
        assert np.isnan(result.subgrid_pixel_coords).all()
    if cache is not None:
        assert cache[1] is saved_coordinates
    rivers.at[1, "hydrography_xy"] = []
    with pytest.raises(ValueError, match="no model grid cells"):
        snap_point_to_river_network(Point(0, 0), rivers, area, area)


def _unexpected_subgrid_read() -> npt.NDArray[np.float32]:
    """Fail if a diagnostic raster is read during weir placement.

    Returns:
        Never returns an array.

    Raises:
        RuntimeError: Whenever the raster is evaluated.
    """  # noqa: DOC202
    raise RuntimeError("Weir placement must not read subgrid diagnostics.")
