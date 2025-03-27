import numpy as np
import xarray as xr


class Groundwater:
    def __init__(self):
        pass

    def setup_groundwater(
        self,
        minimum_thickness_confined_layer=50,
        maximum_thickness_confined_layer=1000,
        intial_heads_source="GLOBGM",
        force_one_layer=False,
    ):
        """
        Sets up the MODFLOW grid for GEB. This code is adopted from the GLOBGM
        model (https://github.com/UU-Hydro/GLOBGM). Also see ThirdPartyNotices.txt

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
        total_thickness = self.data_catalog.get_rasterdataset(
            "total_groundwater_thickness_globgm",
            bbox=self.bounds,
            buffer=2,
        ).rename({"lon": "x", "lat": "y"})

        total_thickness = np.clip(
            total_thickness,
            minimum_thickness_confined_layer,
            maximum_thickness_confined_layer,
        )

        confining_layer = self.data_catalog.get_rasterdataset(
            "thickness_confining_layer_globgm",
            bbox=self.bounds,
            buffer=2,
        ).rename({"lon": "x", "lat": "y"})

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

        aquifer_top_elevation.raster.set_crs(4326)
        layer_boundary_elevation = (
            relative_layer_boundary_elevation.raster.reproject_like(
                aquifer_top_elevation, method="bilinear"
            )
        ) + aquifer_top_elevation
        layer_boundary_elevation.attrs["_FillValue"] = np.nan

        self.set_grid(
            layer_boundary_elevation, name="groundwater/layer_boundary_elevation"
        )

        # load hydraulic conductivity
        hydraulic_conductivity = self.data_catalog.get_rasterdataset(
            "hydraulic_conductivity_globgm",
            bbox=self.bounds,
            buffer=2,
        ).rename({"lon": "x", "lat": "y"})

        # because
        hydraulic_conductivity_log = np.log(hydraulic_conductivity)
        hydraulic_conductivity_log = hydraulic_conductivity_log.raster.reproject_like(
            aquifer_top_elevation, method="bilinear"
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
        specific_yield = self.data_catalog.get_rasterdataset(
            "specific_yield_aquifer_globgm",
            bbox=self.bounds,
            buffer=2,
        ).rename({"lon": "x", "lat": "y"})
        specific_yield = specific_yield.raster.reproject_like(
            aquifer_top_elevation, method="bilinear"
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
        why_map = self.data_catalog.get_rasterdataset(
            "why_map",
            bbox=self.bounds,
            buffer=5,
        )

        why_map.x.attrs = {"long_name": "longitude", "units": "degrees_east"}
        why_map.y.attrs = {"long_name": "latitude", "units": "degrees_north"}
        why_interpolated = why_map.raster.reproject_like(
            aquifer_top_elevation, method="bilinear"
        )

        self.set_grid(why_interpolated, name="groundwater/why_map")

        if intial_heads_source == "GLOBGM":
            # the GLOBGM DEM has a slight offset, which we fix here before loading it
            dem_globgm = self.data_catalog.get_rasterdataset(
                "dem_globgm",
                variables=["dem_average"],
            )
            dem_globgm = dem_globgm.assign_coords(
                lon=self.data_catalog.get_rasterdataset("head_upper_globgm").x.values,
                lat=self.data_catalog.get_rasterdataset("head_upper_globgm").y.values,
            )

            # loading the globgm with fixed coordinates
            dem_globgm = self.data_catalog.get_rasterdataset(
                dem_globgm, geom=self.region, variables=["dem_average"], buffer=2
            ).rename({"lon": "x", "lat": "y"})
            # load digital elevation model that was used for globgm

            dem = self.grid["landsurface/elevation"].raster.mask_nodata()

            # heads
            head_upper_layer = self.data_catalog.get_rasterdataset(
                "head_upper_globgm",
                bbox=self.bounds,
                buffer=2,
            )

            head_upper_layer = head_upper_layer.raster.mask_nodata()
            relative_head_upper_layer = head_upper_layer - dem_globgm
            relative_head_upper_layer = relative_head_upper_layer.raster.reproject_like(
                aquifer_top_elevation, method="bilinear"
            )
            head_upper_layer = dem + relative_head_upper_layer

            head_lower_layer = self.data_catalog.get_rasterdataset(
                "head_lower_globgm",
                bbox=self.bounds,
                buffer=2,
            )
            head_lower_layer = head_lower_layer.raster.mask_nodata()
            relative_head_lower_layer = head_lower_layer - dem_globgm
            relative_head_lower_layer = relative_head_lower_layer.raster.reproject_like(
                aquifer_top_elevation, method="bilinear"
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

            initial_depth = self.data_catalog.get_rasterdataset(
                f"initial_groundwater_depth_{region_continent}",
                bbox=self.bounds,
                buffer=0,
            ).rename({"lon": "x", "lat": "y"})

            initial_depth_static = initial_depth.isel(time=0)
            initial_depth = initial_depth_static.raster.reproject_like(
                self.grid, method="average"
            )
            raise NotImplementedError(
                "Need to convert initial depth to heads for all layers"
            )

        assert heads.shape == hydraulic_conductivity.shape
        heads.attrs["_FillValue"] = np.nan
        self.set_grid(heads, name="groundwater/heads")
