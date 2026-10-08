"""Tests for fractional WOKAM coverage in the model build."""

import logging
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from geb.build.modules.groundwater import GroundWater


@pytest.mark.parametrize("lazy", [False, True])
def test_setup_karst_fractional_coverage(lazy: bool) -> None:
    """Build coverage for both in-memory and lazily loaded model grids.

    Args:
        lazy: Whether the grid uses Dask, as when reading built inputs.
    """
    mask: xr.DataArray = xr.DataArray(
        [[False, False, False], [False, False, True]],
        coords={"y": [3.0, 1.0], "x": [1.0, 3.0, 5.0]},
        dims=("y", "x"),
    ).rio.write_crs(6933)
    if lazy:
        mask = mask.chunk({"y": 1, "x": 2})
    polygons: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"rock_type": [1, 1, 2, 4]},
        geometry=[box(0, 2, 2, 4), box(0, 2, 2, 4), box(2, 0, 3, 4), box(4, 2, 6, 4)],
        crs=6933,
    )
    builder: MagicMock = MagicMock()
    builder.logger = logging.getLogger(__name__)
    builder.grid = {"mask": mask}
    builder.data_catalog.fetch.return_value.read.return_value = polygons
    GroundWater.setup_karst(builder)
    result: xr.DataArray = builder.set_grid.call_args.args[0]
    np.testing.assert_allclose(
        result.values, [[0.825, 0.2, 0.65], [0, 0.2, np.nan]], rtol=1e-6
    )
    assert result.dtype == np.float32
    assert result.rio.crs == mask.rio.crs
    assert builder.set_grid.call_args.kwargs["name"] == "groundwater/karst_fraction"
    builder.data_catalog.fetch.assert_called_once_with("wokam")


def test_empty_and_invalid_wokam() -> None:
    """Return zero outside WOKAM coverage and reject unknown classes."""
    mask: xr.DataArray = xr.DataArray(
        np.zeros((2, 2), dtype=bool),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
        dims=("y", "x"),
    ).rio.write_crs(4326)
    polygons: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"rock_type": [1]}, geometry=[box(10, 10, 11, 11)], crs=4326
    )
    builder: MagicMock = MagicMock()
    builder.logger = logging.getLogger(__name__)
    builder.grid = {"mask": mask}
    builder.data_catalog.fetch.return_value.read.return_value = polygons
    GroundWater.setup_karst(builder)
    np.testing.assert_array_equal(builder.set_grid.call_args.args[0].values, 0)
    polygons["rock_type"] = 99
    with pytest.raises(ValueError, match="unknown rock classes"):
        GroundWater.setup_karst(builder)
    polygons["rock_type"] = 1
    polygons.set_crs(None, allow_override=True, inplace=True)
    with pytest.raises(ValueError, match="must have a CRS"):
        GroundWater.setup_karst(builder)
