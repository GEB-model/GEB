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


def interpolate_soil_layers(ds):
    assert ds.ndim == 3
    for layer in range(ds.shape[0]):
        ds[layer] = ds[layer].raster.interpolate_na("nearest")
    return ds


def load_soilgrids(data_catalog, subgrid, region):
    variables = ["bdod", "clay", "silt", "sand", "soc"]
    layers = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    ds = []
    for variable in variables:
        variable_layers = []
        for i, layer in enumerate(layers, start=1):
            da = data_catalog.get_rasterdataset(
                f"soilgrids_2020_{variable}_{layer}", geom=region
            ).compute()
            da.name = f"{variable}_{i}"
            variable_layers.append(da)
        ds_variable = xr.concat(variable_layers, dim="soil_layers", compat="equals")
        ds_variable.name = variable
        ds.append(ds_variable)
    ds = xr.merge(ds, join="exact")

    ds = ds.raster.mask_nodata()  # set all nodata values to nan

    ds = ds.raster.reproject_like(subgrid, method="bilinear")

    ds["sand"] = interpolate_soil_layers(ds["sand"])
    ds["silt"] = interpolate_soil_layers(ds["silt"])
    ds["clay"] = interpolate_soil_layers(ds["clay"])
    ds["bdod"] = interpolate_soil_layers(ds["bdod"])
    ds["soc"] = interpolate_soil_layers(ds["soc"])

    total = ds["sand"] + ds["clay"] + ds["silt"]
    assert total.min() >= 99.8
    assert total.max() <= 100.2

    ds["sand"] = ds["sand"] / total * 100
    ds["clay"] = ds["clay"] / total * 100
    ds["silt"] = ds["silt"] / total * 100

    # the top 30 cm is considered as top soil (https://www.fao.org/uploads/media/Harm-World-Soil-DBv7cv_1.pdf)
    is_top_soil = np.zeros_like(ds["sand"], dtype=bool)
    is_top_soil[0:3] = True
    is_top_soil = xr.DataArray(
        is_top_soil, dims=ds["sand"].dims, coords=ds["sand"].coords
    )
    ds["is_top_soil"] = is_top_soil

    depth_to_bedrock = data_catalog.get_rasterdataset(
        "soilgrids_2017_BDTICM", geom=region
    )
    depth_to_bedrock = depth_to_bedrock.raster.mask_nodata()
    depth_to_bedrock = depth_to_bedrock.raster.reproject_like(
        subgrid, method="bilinear"
    ).raster.interpolate_na("nearest")

    soil_layer_height = xr.DataArray(
        np.zeros(ds["sand"].shape, dtype=np.float32),
        dims=ds["sand"].dims,
        coords=ds["sand"].coords,
    )
    for layer, height in enumerate((0.05, 0.10, 0.15, 0.30, 0.40, 1.00)):
        soil_layer_height[layer] = height

    assert (soil_layer_height.sum(axis=0) == 2.0).all()

    return ds["sand"], ds["silt"], ds["clay"], ds["bdod"], ds["soc"], soil_layer_height
