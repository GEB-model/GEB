# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/Deltares/hydromt_wflow/
# Files:
# - https://github.com/Deltares/hydromt_wflow/blob/main/hydromt_wflow/workflows/ptf.py
# - https://github.com/Deltares/hydromt_wflow/blob/main/hydromt_wflow/workflows/soilgrids.py
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------

import xarray as xr
import numpy as np
from pyresample import geometry
from pyresample.resampler import resample_blocks
from pyresample.gradient import (
    gradient_resampler_indices_block,
    block_bilinear_interpolator,
    block_nn_interpolator,
)


def get_area_definition(da):
    return geometry.AreaDefinition(
        area_id="",
        description="",
        proj_id="",
        projection=da.rio.crs.to_proj4(),
        width=da.x.size,
        height=da.y.size,
        area_extent=da.rio.bounds(),
    )


def _fill_in_coords(target_coords, source_coords, data_dims):
    x_coord, y_coord = target_coords["x"], target_coords["y"]
    coords = []
    for key in data_dims:
        if key == "x":
            coords.append(x_coord)
        elif key == "y":
            coords.append(y_coord)
        else:
            coords.append(source_coords[key])
    return coords


def resample_chunked(source, target, method="bilinear"):
    if method == "nearest_neighbour":
        interpolator = block_nn_interpolator
    elif method == "bilinear":
        interpolator = block_bilinear_interpolator
    else:
        raise ValueError(
            f"Unknown method: {method}, must be 'bilinear' or 'nearest_neighbour'"
        )

    source_geo = get_area_definition(source)
    target_geo = get_area_definition(target)

    indices = resample_blocks(
        gradient_resampler_indices_block,
        source_geo,
        [],
        target_geo,
        chunk_size=(2, *target.chunks),
        dtype=np.float64,
    )

    resampled_data = resample_blocks(
        interpolator,
        source_geo,
        [source.data],
        target_geo,
        dst_arrays=[indices],
        chunk_size=(*source.data.shape[:-2], *target.chunks),
        dtype=source.dtype,
        fill_value=source.attrs["_FillValue"],
    )

    # Convert result back to xarray DataArray
    da = xr.DataArray(
        resampled_data,
        dims=source.dims,
        coords=_fill_in_coords(target.coords, source.coords, source.dims),
        name=source.name,
        attrs=source.attrs.copy(),
    )
    da.rio.set_crs(source.rio.crs)
    return da


def load_soilgrids(data_catalog, subgrid, region):
    variables = ["bdod", "clay", "silt", "soc"]
    layers = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    subgrid_mask = subgrid["areamaps/sub_grid_mask"]
    subgrid_mask = subgrid_mask.rio.set_crs(4326)

    ds = []
    for variable_name in variables:
        variable_layers = []
        for i, layer in enumerate(layers, start=1):
            da = data_catalog.get_rasterdataset(
                f"soilgrids_2020_{variable_name}_{layer}", geom=region, buffer=30
            ).compute()
            da = da.raster.interpolate_na("nearest")
            da.name = f"{variable_name}_{i}"
            variable_layers.append(da)
        ds_variable = xr.concat(variable_layers, dim="soil_layer", compat="equals")
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

    ds = xr.merge(ds, join="exact")

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

    ds.raster.set_crs(4326)
    return ds.rio.set_crs(4326)
