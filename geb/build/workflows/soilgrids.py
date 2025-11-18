"""Load soilgrids data from ISRIC SoilGrids."""

import geopandas as gpd
import numpy as np
import xarray as xr

from geb.workflows.io import get_window
from geb.workflows.raster import convert_nodata, interpolate_na_2d, resample_chunked

from ..data_catalog import NewDataCatalog


def load_soilgrids(
    data_catalog: NewDataCatalog, mask: xr.Dataset, region: gpd.GeoDataFrame
) -> xr.Dataset:
    """Load soilgrids data from ISRIC SoilGrids.

    Args:
        data_catalog: A data catalog with soilgrids data sources.
        mask: The grid to resample to.
        region: The region of interest, matches with the subgrid.

    Returns:
        A dataset with soilgrids data.
    """
    variables: list[str] = ["bdod", "clay", "silt", "soc"]
    layers: list[str] = [
        "0-5cm",
        "5-15cm",
        "15-30cm",
        "30-60cm",
        "60-100cm",
        "100-200cm",
    ]

    ds: list[xr.DataArray] = []
    for variable_name in variables:
        variable_layers: list[xr.DataArray] = []
        for i, layer in enumerate(layers, start=1):
            da: xr.DataArray = data_catalog.fetch("soilgrids").read(
                variable=variable_name, depth=layer
            )
            da: xr.DataArray = (
                da.isel(
                    get_window(
                        da.x, da.y, region.to_crs(da.rio.crs).total_bounds, buffer=30
                    ),
                )
                .astype(np.float32)
                .compute()
            )
            da: xr.DataArray = interpolate_na_2d(da)
            da.assign_coords(soil_layer=i)
            variable_layers.append(da)
        ds_variable: xr.DataArray = xr.concat(
            variable_layers,
            dim=xr.Variable("soil_layer", [1, 2, 3, 4, 5, 6]),
            compat="equals",
        )
        ds_variable: xr.DataArray = ds_variable.chunk({"x": 30, "y": 30})
        ds_variable: xr.DataArray = convert_nodata(ds_variable, np.nan)
        ds_variable: xr.DataArray = resample_chunked(
            ds_variable,
            mask,
            method="nearest",
        )
        ds_variable: xr.DataArray = ds_variable.where(
            ~mask, ds_variable.attrs["_FillValue"]
        )
        ds_variable = ds_variable.rio.set_crs(4326)
        ds_variable.name = variable_name
        ds.append(ds_variable)

    ds: xr.Dataset = xr.merge(ds, join="exact").transpose("soil_layer", "y", "x")

    # soilgrids uses conversion factors as specified here:
    # https://www.isric.org/explore/soilgrids/faq-soilgrids
    ds["bdod"] = ds["bdod"] / 100  # cg/cm³ -> kg/dm³
    ds["clay"] = ds["clay"] / 10  # g/kg -> g/100g (%)
    ds["silt"] = ds["silt"] / 10  # g/kg -> g/100g (%)
    ds["soc"] = ds["soc"] / 100  # g/kg -> g/100g (%)

    # depth_to_bedrock = data_catalog.get_rasterdataset(
    #     "soilgrids_2017_BDTICM", geom=region
    # )
    # depth_to_bedrock = convert_nodata(depth_to_bedrock, np.nan)
    # depth_to_bedrock = resample_like(depth_to_bedrock, subgrid, method="bilinear")
    # depth_to_bedrock = interpolate_na_2d(depth_to_bedrock)
    soil_layer_height: xr.DataArray = xr.full_like(
        ds["silt"], fill_value=0.0, dtype=np.float32
    )
    for layer, height in enumerate((0.05, 0.10, 0.15, 0.30, 0.40, 1.00)):
        soil_layer_height[layer] = height
    ds["height"] = soil_layer_height
    return ds
