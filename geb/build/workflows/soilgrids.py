import numpy as np
import xarray as xr

from .general import resample_chunked


def load_soilgrids(data_catalog, subgrid, region):
    variables = ["bdod", "clay", "silt", "soc"]
    layers = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    subgrid_mask = subgrid["mask"]
    subgrid_mask = subgrid_mask.rio.set_crs(4326)

    ds = []
    for variable_name in variables:
        variable_layers = []
        for i, layer in enumerate(layers, start=1):
            da = (
                data_catalog.get_rasterdataset(
                    f"soilgrids_2020_{variable_name}_{layer}", geom=region, buffer=30
                )
                .astype(np.float32)
                .compute()
            )
            da = da.raster.interpolate_na("nearest")
            da.assign_coords(soil_layer=i)
            variable_layers.append(da)
        ds_variable = xr.concat(
            variable_layers,
            dim=xr.Variable("soil_layer", [1, 2, 3, 4, 5, 6]),
            compat="equals",
        )
        ds_variable = ds_variable.chunk({"x": 30, "y": 30})
        ds_variable = ds_variable.raster.mask_nodata()
        ds_variable = resample_chunked(
            ds_variable,
            subgrid_mask,
            method="nearest_neighbour",
        )
        ds_variable = ds_variable.where(~subgrid_mask, ds_variable.attrs["_FillValue"])
        ds_variable = ds_variable.rio.set_crs(4326)
        ds_variable.name = variable_name
        ds.append(ds_variable)

    ds = xr.merge(ds, join="exact").transpose("soil_layer", "y", "x")
    # depth_to_bedrock = data_catalog.get_rasterdataset(
    #     "soilgrids_2017_BDTICM", geom=region
    # )
    # depth_to_bedrock = depth_to_bedrock.raster.mask_nodata()
    # depth_to_bedrock = depth_to_bedrock.raster.reproject_like(
    #     subgrid, method="bilinear"
    # ).raster.interpolate_na("nearest")

    soil_layer_height = xr.full_like(ds["silt"], fill_value=0.0, dtype=np.float32)
    for layer, height in enumerate((0.05, 0.10, 0.15, 0.30, 0.40, 1.00)):
        soil_layer_height[layer] = height
    ds["height"] = soil_layer_height
    return ds
