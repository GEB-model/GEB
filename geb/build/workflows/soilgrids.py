"""Load soilgrids data from ISRIC SoilGrids."""

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr

from geb.workflows.io import get_window
from geb.workflows.raster import (
    convert_nodata,
    interpolate_na_2d,
    resample_chunked,
)


def process_soilgrids(
    da: xr.DataArray,
    mask: xr.DataArray,
    region: gpd.GeoDataFrame,
) -> xr.DataArray:
    """Load a SoilGrids variable from ISRIC SoilGrids.

    Args:
        da: The SoilGrids data array to load.
        mask: The grid to resample to.
        region: The region to load the data for.

    Returns:
        The requested SoilGrids variable with dimensions ``y`` and ``x``.
    """
    with rasterio.Env(
        GDAL_PAM_ENABLED="NO",
        GDAL_DISABLE_READDIR_ON_OPEN="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        da: xr.DataArray = (
            da.isel(
                get_window(
                    x=da.x,
                    y=da.y,
                    bounds=region.to_crs(da.rio.crs).total_bounds,
                    buffer=100,
                )
            )
            .compute()
            .chunk({"x": -1, "y": -1})
        )

    da = convert_nodata(da, np.nan)
    da = interpolate_na_2d(da)
    da = resample_chunked(
        da,
        mask.chunk({"x": 3000, "y": 3000}),
        method="nearest",
    )
    assert isinstance(da, xr.DataArray)
    return da
