import numpy as np
import xarray as xr

from geb.workflows.io import get_window

from ..workflows.general import (
    resample_like,
)


class GroundWater:
    def __init__(self):
        pass

    def setup_groundwater(
        self,
        minimum_thickness_confined_layer=50,
        maximum_thickness_confined_layer=1000,
        intial_heads_source="GLOBGM",
        force_one_layer=True,
    ):
        """Sets up the MODFLOW grid for GEB.

        This code is adopted from the GLOBGM model (https://github.com/UU-Hydro/GLOBGM). Also see ThirdPartyNotices.txt.

        Parameters
        ----------
        minimum_thickness_confined_layer : float, optional
            The minimum thickness of the confined layer in meters. Default is 50.
        maximum_thickness_confined_layer : float, optional
            The maximum thickness of the confined layer in meters. Default is 1000.
        intial_heads_source : str, optional
            The initial heads dataset to use, options are GLOBGM and Fan. Default is 'GLOBGM'.
            - More about GLOBGM: https://doi.org/10.5194/gmd-17-275-2024
            - More about Fan: https://doi.org/10.1126/science.1229881
        """
        self.logger.info("Setting up MODFLOW")

        aquifer_top_elevation = self.grid["landsurface/elevation"].raster.mask_nodata()
        aquifer_top_elevation.raster.set_crs(4326)
        aquifer_top_elevation = self.set_grid(
            aquifer_top_elevation, name="groundwater/aquifer_top_elevation"
        )

        # load total thickness
        total_thickness = (
            xr.open_dataarray(
                self.data_catalog.get_source("total_groundwater_thickness_globgm").path
            )
            .rename({"lon": "x", "lat": "y"})
            .rio.write_crs(4326)
        )
        total_thickness = total_thickness.isel(
            get_window(total_thickness.x, total_thickness.y, self.bounds, buffer=2)
        )

        total_thickness = np.clip(
            total_thickness,
            minimum_thickness_confined_layer,
            maximum_thickness_confined_layer,
        )

        confining_layer = (
            xr.open_dataarray(
                self.data_catalog.get_source("thickness_confining_layer_globgm").path
            )
            .rename({"lon": "x", "lat": "y"})
            .rio.write_crs(4326)
        )
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
        hydraulic_conductivity = (
            xr.open_dataarray(
                self.data_catalog.get_source("hydraulic_conductivity_globgm").path
            )
            .rename({"lon": "x", "lat": "y"})
            .rio.write_crs(4326)
        )
        hydraulic_conductivity = hydraulic_conductivity.isel(
            get_window(
                hydraulic_conductivity.x,
                hydraulic_conductivity.y,
                self.bounds,
                buffer=2,
            )
        )
        hydraulic_conductivity.attrs["_FillValue"] = np.nan

        # because
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
        specific_yield = (
            xr.open_dataarray(
                self.data_catalog.get_source("specific_yield_aquifer_globgm").path
            )
            .rename({"lon": "x", "lat": "y"})
            .rio.write_crs(4326)
        )
        specific_yield = specific_yield.isel(
            get_window(specific_yield.x, specific_yield.y, self.bounds, buffer=2)
        )
        specific_yield.attrs["_FillValue"] = np.nan

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

        # load aquifer classification from why_map and write it as a grid
        why_map = xr.open_dataarray(self.data_catalog.get_source("why_map").path)
        why_map = why_map.isel(
            band=0, **get_window(why_map.x, why_map.y, self.bounds, buffer=5)
        )

        why_map.x.attrs = {"long_name": "longitude", "units": "degrees_east"}
        why_map.y.attrs = {"long_name": "latitude", "units": "degrees_north"}
        why_map.attrs["_FillValue"] = np.nan

        original_dtype = why_map.dtype
        why_interpolated = resample_like(
            why_map.astype(np.float64),
            aquifer_top_elevation,
            method="nearest",
        )
        why_interpolated = why_interpolated.astype(original_dtype)

        self.set_grid(why_interpolated, name="groundwater/why_map")

        if intial_heads_source == "GLOBGM":
            # the GLOBGM DEM has a slight offset, which we fix here before loading it

            reference_globgm_map = xr.open_dataarray(
                self.data_catalog.get_source("head_upper_globgm").path
            )

            dem_globgm = xr.open_dataarray(
                self.data_catalog.get_source("dem_globgm").path
            ).rename({"lon": "x", "lat": "y"})

            dem_globgm = dem_globgm.assign_coords(
                x=reference_globgm_map.x.values,
                y=reference_globgm_map.y.values,
            )

            dem_globgm = dem_globgm.isel(
                get_window(dem_globgm.x, dem_globgm.y, self.bounds, buffer=2)
            )

            dem = self.grid["landsurface/elevation"].raster.mask_nodata()

            # heads
            head_upper_layer = xr.open_dataarray(
                self.data_catalog.get_source("head_upper_globgm").path
            )
            head_upper_layer = head_upper_layer.isel(
                band=0,
                **get_window(
                    head_upper_layer.x, head_upper_layer.y, self.bounds, buffer=2
                ),
            )

            head_upper_layer = head_upper_layer.raster.mask_nodata()
            relative_head_upper_layer = head_upper_layer - dem_globgm

            relative_head_upper_layer = resample_like(
                relative_head_upper_layer, aquifer_top_elevation, method="bilinear"
            )

            head_upper_layer = dem + relative_head_upper_layer

            head_lower_layer = xr.open_dataarray(
                self.data_catalog.get_source("head_lower_globgm").path
            )
            head_lower_layer = head_lower_layer.isel(
                band=0,
                **get_window(
                    head_lower_layer.x, head_lower_layer.y, self.bounds, buffer=2
                ),
            )

            head_lower_layer = head_lower_layer.raster.mask_nodata()
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
            region_continent = np.unique(self.geoms["regions"]["CONTINENT"])
            assert (
                np.size(region_continent) == 1
            )  # Transcontinental basins should not be possible

            if (
                np.unique(self.geoms["regions"]["CONTINENT"])[0] == "Asia"
                or np.unique(self.geoms["regions"]["CONTINENT"])[0] == "Europe"
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
        heads.attrs["_FillValue"] = np.nan
        self.set_grid(heads, name="groundwater/heads")
