"""Module for building groundwater related datasets for GEB."""

import geopandas as gpd
import numpy as np
import xarray as xr

from geb.build.methods import build_method
from geb.workflows.io import get_window
from geb.workflows.raster import (
    convert_nodata,
    interpolate_na_2d,
    rasterize_like,
    resample_like,
)


class GroundWater:
    """Contains all build methods for the groundwater for GEB."""

    def __init__(self) -> None:
        """Initialize the GroundWater class."""
        pass

    @build_method(depends_on=["setup_elevation"])
    def setup_groundwater(
        self,
        minimum_thickness_confined_layer: int | float = 50,
        maximum_thickness_confined_layer: int | float = 1000,
        intial_heads_source: str = "GLOBGM",
        force_one_layer: bool = True,
    ) -> None:
        """Sets up the MODFLOW grid for GEB.

        This code is adopted from the GLOBGM model (https://github.com/UU-Hydro/GLOBGM). Also see ThirdPartyNotices.txt.

        More about GLOBGM: https://doi.org/10.5194/gmd-17-275-2024
        More about Fan: https://doi.org/10.1126/science.1229881

        Args:
            minimum_thickness_confined_layer: The minimum thickness of the confined layer in meters. Default is 50.
            maximum_thickness_confined_layer: The maximum thickness of the confined layer in meters. Default is 1000.
            intial_heads_source: The initial heads dataset to use, options are GLOBGM and Fan. Default is 'GLOBGM'.
            force_one_layer: If True, the model will be forced to use only one layer. Default is True.
        """
        aquifer_top_elevation = convert_nodata(
            self.grid["landsurface/elevation"], new_nodata=np.nan
        )
        aquifer_top_elevation = self.set_grid(
            aquifer_top_elevation, name="groundwater/aquifer_top_elevation"
        )

        # load total thickness
        total_thickness = self.new_data_catalog.fetch(
            "total_groundwater_thickness_globgm"
        ).read()
        total_thickness = total_thickness.isel(
            get_window(total_thickness.x, total_thickness.y, self.bounds, buffer=2)
        )

        total_thickness = np.clip(
            total_thickness,
            minimum_thickness_confined_layer,
            maximum_thickness_confined_layer,
        )

        confining_layer = self.new_data_catalog.fetch(
            "thickness_confining_layer_globgm"
        ).read()
        confining_layer = confining_layer.isel(
            get_window(confining_layer.x, confining_layer.y, self.bounds, buffer=2)
        )

        if not (confining_layer == 0).all() and not force_one_layer:  # two-layer-model
            two_layers = True
        else:
            two_layers = False

        if two_layers:
            # make sure that total thickness is at least 50 m thicker than confining layer
            total_thickness = np.maximum(
                total_thickness, confining_layer + minimum_thickness_confined_layer
            )
            # thickness of layer 2 is based on the predefined confiningLayerThickness
            relative_bottom_top_layer = -confining_layer
            # make sure that the minimum thickness of layer 2 is at least 0.1 m
            thickness_top_layer = np.maximum(0.1, -relative_bottom_top_layer)
            relative_bottom_top_layer = -thickness_top_layer
            # thickness of layer 1 is at least 5.0 m
            thickness_bottom_layer = np.maximum(
                5.0, total_thickness - thickness_top_layer
            )
            relative_bottom_bottom_layer = (
                relative_bottom_top_layer - thickness_bottom_layer
            )

            relative_layer_boundary_elevation = xr.concat(
                [
                    self.full_like(
                        relative_bottom_bottom_layer, fill_value=0, nodata=np.nan
                    ),
                    relative_bottom_top_layer,
                    relative_bottom_bottom_layer,
                ],
                dim=xr.Variable(
                    "boundary", ["relative_top", "boundary", "relative_bottom"]
                ),
                compat="equals",
            )
        else:
            relative_bottom_bottom_layer = -total_thickness
            relative_layer_boundary_elevation = xr.concat(
                [
                    self.full_like(
                        relative_bottom_bottom_layer, fill_value=0, nodata=np.nan
                    ),
                    relative_bottom_bottom_layer,
                ],
                dim=xr.Variable("boundary", ["relative_top", "relative_bottom"]),
                compat="equals",
            )

        layer_boundary_elevation = (
            resample_like(
                relative_layer_boundary_elevation,
                aquifer_top_elevation,
                method="bilinear",
            )
        ) + aquifer_top_elevation
        layer_boundary_elevation.attrs["_FillValue"] = np.nan

        self.set_grid(
            layer_boundary_elevation, name="groundwater/layer_boundary_elevation"
        )

        # load hydraulic conductivity
        hydraulic_conductivity = self.new_data_catalog.fetch(
            "hydraulic_conductivity_globgm"
        ).read()
        hydraulic_conductivity = hydraulic_conductivity.isel(
            get_window(
                hydraulic_conductivity.x,
                hydraulic_conductivity.y,
                self.bounds,
                buffer=2,
            )
        )

        # because hydraulic conductivity is log-normally distributed, we interpolate the log values
        # after log transformation and then back-transform after interpolation
        hydraulic_conductivity_log = np.log(hydraulic_conductivity)
        hydraulic_conductivity_log = resample_like(
            hydraulic_conductivity_log,
            aquifer_top_elevation,
            method="bilinear",
        )
        hydraulic_conductivity = np.exp(hydraulic_conductivity_log)

        if two_layers:
            hydraulic_conductivity = xr.concat(
                [hydraulic_conductivity, hydraulic_conductivity],
                dim=xr.Variable("layer", ["upper", "lower"]),
                compat="equals",
            )
        else:
            hydraulic_conductivity = hydraulic_conductivity.expand_dims(layer=["upper"])
        self.set_grid(hydraulic_conductivity, name="groundwater/hydraulic_conductivity")

        # load specific yield
        specific_yield = self.new_data_catalog.fetch(
            "specific_yield_aquifer_globgm"
        ).read()
        specific_yield = specific_yield.isel(
            get_window(specific_yield.x, specific_yield.y, self.bounds, buffer=2)
        )

        specific_yield = resample_like(
            specific_yield,
            aquifer_top_elevation,
            method="bilinear",
        )

        if two_layers:
            specific_yield = xr.concat(
                [specific_yield, specific_yield],
                dim=xr.Variable("layer", ["upper", "lower"]),
                compat="equals",
            )
        else:
            specific_yield = specific_yield.expand_dims(layer=["upper"])
        self.set_grid(specific_yield, name="groundwater/specific_yield")

        why_map: gpd.GeoDataFrame = self.new_data_catalog.fetch("why_map").read()
        why_map: gpd.GeoDataFrame = why_map[
            why_map["HYGEO2"] != 88
        ]  # remove areas under continuous ice cover
        why_map["aquifer_classification"] = why_map["HYGEO2"] // 10

        why_map_grid: xr.DataArray = rasterize_like(
            why_map,
            column="aquifer_classification",
            raster=aquifer_top_elevation,
            dtype=np.int16,
            nodata=-1,
            all_touched=False,
        )
        why_map_grid: xr.DataArray = interpolate_na_2d(why_map_grid)

        self.set_grid(why_map_grid, name="groundwater/why_map")

        if intial_heads_source == "GLOBGM":
            # the GLOBGM DEM has a slight offset, which we fix here before loading it

            reference_globgm_map = self.new_data_catalog.fetch(
                "head_upper_layer_globgm"
            ).read()

            dem_globgm = self.new_data_catalog.fetch("dem_globgm").read()

            dem_globgm = dem_globgm.assign_coords(
                x=reference_globgm_map.x.values,
                y=reference_globgm_map.y.values,
            )

            dem_globgm = dem_globgm.isel(
                get_window(dem_globgm.x, dem_globgm.y, self.bounds, buffer=2)
            )
            dem = convert_nodata(self.grid["landsurface/elevation"], new_nodata=np.nan)

            # heads
            head_upper_layer = self.new_data_catalog.fetch(
                "head_upper_layer_globgm"
            ).read()
            head_upper_layer = head_upper_layer.isel(
                get_window(
                    head_upper_layer.x, head_upper_layer.y, self.bounds, buffer=2
                ),
            )

            head_upper_layer = convert_nodata(head_upper_layer, new_nodata=np.nan)

            relative_head_upper_layer = head_upper_layer - dem_globgm

            relative_head_upper_layer = resample_like(
                relative_head_upper_layer, aquifer_top_elevation, method="bilinear"
            )

            head_upper_layer = dem + relative_head_upper_layer

            head_lower_layer = self.new_data_catalog.fetch(
                "head_lower_layer_globgm"
            ).read()
            head_lower_layer = head_lower_layer.isel(
                get_window(
                    head_lower_layer.x, head_lower_layer.y, self.bounds, buffer=2
                ),
            )

            head_lower_layer = convert_nodata(head_lower_layer, new_nodata=np.nan)

            relative_head_lower_layer = head_lower_layer - dem_globgm

            relative_head_lower_layer = resample_like(
                relative_head_lower_layer, aquifer_top_elevation, method="bilinear"
            )

            # TODO: Make sure head in lower layer is not lower than topography, but why is this needed?
            relative_head_lower_layer = xr.where(
                relative_head_lower_layer
                < layer_boundary_elevation.isel(boundary=-1) - dem,
                layer_boundary_elevation.isel(boundary=-1) - dem,
                relative_head_lower_layer,
            )
            head_lower_layer = dem + relative_head_lower_layer

            if two_layers:
                # combine upper and lower layer head in one dataarray
                heads = xr.concat(
                    [head_upper_layer, head_lower_layer],
                    dim=xr.Variable("layer", ["upper", "lower"]),
                    compat="equals",
                )
            else:
                heads = head_lower_layer.expand_dims(layer=["upper"])

        elif intial_heads_source == "Fan":
            # Load in the starting groundwater depth
            region_continent = np.unique(self.geom["regions"]["CONTINENT"])
            assert (
                np.size(region_continent) == 1
            )  # Transcontinental basins should not be possible

            if (
                np.unique(self.geom["regions"]["CONTINENT"])[0] == "Asia"
                or np.unique(self.geom["regions"]["CONTINENT"])[0] == "Europe"
            ):
                region_continent = "Eurasia"
            else:
                region_continent = region_continent[0]

            initial_depth = xr.open_dataarray(
                self.data_catalog.get_source(
                    f"initial_groundwater_depth_{region_continent}"
                ).path
            ).rename({"lon": "x", "lat": "y"})
            initial_depth = initial_depth.isel(
                get_window(initial_depth.x, initial_depth.y, self.bounds, buffer=1)
            )

            initial_depth_static = initial_depth.isel(time=0)

            initial_depth = resample_like(
                initial_depth_static,
                self.grid,
                method="bilinear",
            )

            raise NotImplementedError(
                "Need to convert initial depth to heads for all layers"
            )

        assert heads.shape == hydraulic_conductivity.shape
        self.set_grid(heads, name="groundwater/heads")
