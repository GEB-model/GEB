"""Load soilgrids data from ISRIC SoilGrids."""

import numpy as np
import xarray as xr

from geb.workflows.raster import convert_nodata, interpolate_na_2d, resample_chunked


def load_soilgrids_v2(
    data_catalog: NewDataCatalog,
    mask: xr.DataArray,
    variable_name: str,
    layer_name: str,
) -> xr.DataArray:
    """Load a SoilGrids variable from ISRIC SoilGrids.

    Args:
        data_catalog: A data catalog with soilgrids data sources.
        mask: The grid to resample to.
        variable_name: SoilGrids variable name to load.
        layer_name: SoilGrids layer name to load, e.g. "0-5cm". Must be one of "0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", or "100-200cm".

    Returns:
        The requested SoilGrids variable with dimensions ``y`` and ``x``.

    Raises:
        ValueError: If ``variable_name`` is not supported.
    """
    allowed_variables: tuple[str, ...] = ("bdod", "clay", "silt", "soc")

    if variable_name not in allowed_variables:
        raise ValueError(
            f"Unsupported SoilGrids variable '{variable_name}'. Expected one of {allowed_variables}."
        )

    soilgrids_source = data_catalog.fetch("soilgridsv2")
    da = soilgrids_source.read(variable=variable_name, depth=layer_name)
    da: xr.DataArray = convert_nodata(da, np.nan)
    da = da.astype(np.float32)
    da: xr.DataArray = resample_chunked(
        da,
        mask,
        method="nearest",
    )
    assert isinstance(da, xr.DataArray)
    da: xr.DataArray = da.chunk({"x": -1, "y": -1})
    da = interpolate_na_2d(da)
    return da
