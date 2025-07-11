import numpy as np
import pandas as pd
import xarray as xr

from geb.workflows.io import get_window

from ..workflows.general import (
    bounds_are_within,
    calculate_cell_area,
    pad_xy,
    repeat_grid,
    resample_chunked,
    resample_like,
)
from ..workflows.soilgrids import load_soilgrids


class LandSurface:
    def __init__(self):
        pass

    def setup_cell_area(self) -> None:
        """Sets up the cell area map for the model.

        Raises:
            ValueError: If the grid mask is not available.

        Notes:
            This method prepares the cell area map for the model by calculating the area of each cell in the grid. It first
            retrieves the grid mask from the `mask` attribute of the grid, and then calculates the cell area
            using the `calculate_cell_area()` function. The resulting cell area map is then set as the `cell_area`
            attribute of the grid.

            Additionally, this method sets up a subgrid for the cell area map by creating a new grid with the same extent as
            the subgrid, and then repeating the cell area values from the main grid to the subgrid using the `repeat_grid()`
            function, and correcting for the subgrid factor. Thus, every subgrid cell within a grid cell has the same value.
            The resulting subgrid cell area map is then set as the `cell_area` attribute of the subgrid.
        """
        self.logger.info("Preparing cell area map.")
        mask = self.grid["mask"]

        cell_area = self.full_like(
            mask, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )

        cell_area.data = calculate_cell_area(mask.rio.transform(), mask.shape)
        cell_area = cell_area.where(~mask, cell_area.attrs["_FillValue"])
        self.set_grid(cell_area, name="cell_area")

        sub_cell_area = self.full_like(
            self.subgrid["mask"],
            fill_value=np.nan,
            nodata=np.nan,
            dtype=np.float32,
        )

        sub_cell_area.data = (
            repeat_grid(cell_area.data, self.subgrid_factor) / self.subgrid_factor**2
        )
        self.set_subgrid(sub_cell_area, name="cell_area")

        region_subgrid_cell_area = self.full_like(
            self.region_subgrid["mask"],
            fill_value=np.nan,
            nodata=np.nan,
            dtype=np.float32,
        )

        region_subgrid_cell_area.data = calculate_cell_area(
            region_subgrid_cell_area.rio.transform(recalc=True),
            region_subgrid_cell_area.shape,
        )

        # set the cell area for the region subgrid
        self.set_region_subgrid(
            region_subgrid_cell_area,
            name="cell_area",
        )

    def setup_elevation(
        self,
        DEMs=[
            {
                "name": "fabdem",
                "zmin": 0.001,
            },
            {"name": "gebco"},
        ],
    ):
        """Sets up the elevation data for the model.

        For configuration of DEMs parameters, see
        https://deltares.github.io/hydromt_sfincs/latest/_generated/hydromt_sfincs.SfincsModel.setup_dep.html.
        """
        if not DEMs:
            DEMs = []

        assert isinstance(DEMs, list)
        # here we use the bounds of all subbasins, which may include downstream
        # subbasins that are not part of the study area
        bounds = tuple(self.geoms["routing/subbasins"].total_bounds)

        fabdem: xr.DataArray = xr.open_dataarray(
            self.data_catalog.get_source("fabdem").path
        )
        fabdem: xr.DataArray = fabdem.isel(
            band=0,
            **get_window(
                fabdem.x,
                fabdem.y,
                bounds,
                buffer=100,
            ),
        ).raster.mask_nodata()

        target: xr.DataArray = self.subgrid["mask"]
        target.raster.set_crs(4326)

        self.set_subgrid(
            resample_like(fabdem, target, method="bilinear"),
            name="landsurface/elevation",
        )

        for DEM in DEMs:
            if DEM["name"] == "fabdem":
                DEM_raster = fabdem
            else:
                DEM_raster = xr.open_dataarray(
                    self.data_catalog.get_source(DEM["name"]).path,
                )
                DEM_raster = DEM_raster.isel(
                    band=0,
                    **get_window(
                        DEM_raster.x,
                        DEM_raster.y,
                        tuple(
                            self.geoms["routing/subbasins"]
                            .to_crs(DEM_raster.rio.crs)
                            .total_bounds
                        ),
                        buffer=100,
                        raise_on_out_of_bounds=False,
                        raise_on_buffer_out_of_bounds=False,
                    ),
                ).raster.mask_nodata()

            DEM_raster = DEM_raster.astype(np.float32)
            self.set_other(
                DEM_raster,
                name=f"DEM/{DEM['name']}",
                byteshuffle=True,
            )
            DEM["path"] = f"DEM/{DEM['name']}"

        self.set_dict(DEMs, name="hydrodynamics/DEM_config")

    def setup_regions_and_land_use(
        self,
        region_database="GADM_level1",
        unique_region_id="GID_1",
        ISO3_column="GID_0",
        land_cover="esa_worldcover_2021_v200",
    ):
        """Sets up the (administrative) regions and land use data for GEB.

        The regions can be used for multiple purposes, for example for creating the
        agents in the model, assigning unique crop prices and other economic variables
        per region and for aggregating the results.

        Parameters
        ----------
        region_database : str, optional
            The name of the region database to use. Default is 'GADM_level1'.
        unique_region_id : str, optional
            The name of the column in the region database that contains the unique region ID. Default is 'UID',
            which is the unique identifier for the GADM database.

        Notes:
        -----
        This method sets up the regions and land use data for GEB. It first retrieves the region data from
        the specified region database and sets it as a geometry in the model. It then pads the subgrid to cover the entire
        region and retrieves the land use data from the ESA WorldCover dataset. The land use data is reprojected to the
        padded subgrid and the region ID is rasterized onto the subgrid. The cell area for each region is calculated and
        set as a grid in the model. The MERIT dataset is used to identify rivers, which are set as a grid in the model. The
        land use data is reclassified into five classes and set as a grid in the model. Finally, the cultivated land is
        identified and set as a grid in the model.

        The resulting grids are set as attributes of the model with names of the form '{grid_name}' or
        'landsurface/{grid_name}'.
        """
        self.logger.info("Preparing regions and land use data.")
        regions = self.data_catalog.get_geodataframe(
            region_database,
            geom=self.region,
            predicate="intersects",
        ).rename(columns={unique_region_id: "region_id", ISO3_column: "ISO3"})
        assert np.unique(regions["region_id"]).shape[0] == regions.shape[0], (
            f"Region database must contain unique region IDs ({self.data_catalog[region_database].path})"
        )

        assert bounds_are_within(
            self.region.total_bounds,
            regions.to_crs(self.region.crs).total_bounds,
        )

        region_id_mapping = {
            i: region_id for region_id, i in enumerate(regions["region_id"])
        }
        regions["region_id"] = regions["region_id"].map(region_id_mapping)
        self.set_dict(region_id_mapping, name="region_id_mapping")

        assert "ISO3" in regions.columns, (
            f"Region database must contain ISO3 column ({self.data_catalog[region_database].path})"
        )

        self.set_geoms(regions, name="regions")

        resolution_x, resolution_y = self.subgrid["mask"].rio.resolution()

        regions_bounds = self.geoms["regions"].total_bounds
        mask_bounds = self.grid["mask"].raster.bounds

        # The bounds should be set to a bit larger than the regions to avoid edge effects
        # and also larger than the mask, to ensure that the entire grid is covered.
        pad_minx = min(regions_bounds[0], mask_bounds[0]) - abs(resolution_x) / 2.0
        pad_miny = min(regions_bounds[1], mask_bounds[1]) - abs(resolution_y) / 2.0
        pad_maxx = max(regions_bounds[2], mask_bounds[2]) + abs(resolution_x) / 2.0
        pad_maxy = max(regions_bounds[3], mask_bounds[3]) + abs(resolution_y) / 2.0

        # TODO: Is there a better way to do this?
        region_mask, region_subgrid_slice = pad_xy(
            self.subgrid["mask"].rio,
            pad_minx,
            pad_miny,
            pad_maxx,
            pad_maxy,
            return_slice=True,
            constant_values=1,
        )
        region_mask.attrs["_FillValue"] = None
        region_mask = self.set_region_subgrid(region_mask, name="mask")

        bounds = self.geoms["regions"].total_bounds
        land_use = (
            xr.open_dataarray(
                self.data_catalog.get_source(land_cover).path,
                chunks={"x": 1000, "y": 1000},
                mask_and_scale=False,
            )
            .sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[3], bounds[1]))
            .isel(band=0)
        )

        reprojected_land_use = resample_chunked(
            land_use, region_mask.chunk({"x": 1000, "y": 1000}), method="nearest"
        )

        reprojected_land_use = self.set_region_subgrid(
            reprojected_land_use,
            name="landsurface/original_land_use",
        )

        region_ids = reprojected_land_use.raster.rasterize(
            self.geoms["regions"],
            col_name="region_id",
            all_touched=True,
        )
        region_ids.attrs["_FillValue"] = -1
        region_ids = self.set_region_subgrid(region_ids, name="region_ids")

        full_region_land_use_classes = reprojected_land_use.raster.reclassify(
            pd.DataFrame.from_dict(
                {
                    reprojected_land_use.raster.nodata: 5,  # no data, set to permanent water bodies because ocean
                    10: 0,  # tree cover
                    20: 1,  # shrubland
                    30: 1,  # grassland
                    40: 1,  # cropland, setting to non-irrigated. Initiated as irrigated based on agents
                    50: 4,  # built-up
                    60: 1,  # bare / sparse vegetation
                    70: 1,  # snow and ice
                    80: 5,  # permanent water bodies
                    90: 1,  # herbaceous wetland
                    95: 5,  # mangroves
                    100: 1,  # moss and lichen
                },
                orient="index",
                columns=["GEB_land_use_class"],
            ),
        )["GEB_land_use_class"].astype(np.int32)

        full_region_land_use_classes = self.set_region_subgrid(
            full_region_land_use_classes,
            name="landsurface/full_region_land_use_classes",
        )

        cultivated_land_full_region = xr.where(
            (full_region_land_use_classes == 1) & (reprojected_land_use == 40),
            True,
            False,
        )
        cultivated_land_full_region.attrs["_FillValue"] = None
        cultivated_land_full_region = self.set_region_subgrid(
            cultivated_land_full_region, name="landsurface/full_region_cultivated_land"
        )

        land_use_classes = full_region_land_use_classes.isel(region_subgrid_slice)
        land_use_classes = self.snap_to_grid(land_use_classes, self.subgrid)
        self.set_subgrid(land_use_classes, name="landsurface/land_use_classes")

        cultivated_land = cultivated_land_full_region.isel(region_subgrid_slice)
        cultivated_land = self.snap_to_grid(cultivated_land, self.subgrid)
        self.set_subgrid(cultivated_land, name="landsurface/cultivated_land")

    def setup_land_use_parameters(
        self,
        land_cover="esa_worldcover_2021_v200",
    ) -> None:
        """Sets up the land use parameters for the model.

        Parameters
        ----------
        interpolation_method : str, optional
            The interpolation method to use when interpolating the land use parameters. Default is 'nearest'.

        Notes:
        -----
        This method sets up the land use parameters for the model by retrieving land use data from the CWATM dataset and
        interpolating the data to the model grid. It first retrieves the land use dataset from the `data_catalog`, and
        then retrieves the maximum root depth and root fraction data for each land use type. It then
        interpolates the data to the model grid using the specified interpolation method and sets the resulting grids as
        attributes of the model with names of the form 'landcover/{land_use_type}/{parameter}_{land_use_type}', where
        {land_use_type} is the name of the land use type (e.g. 'forest', 'grassland', etc.) and {parameter} is the name of
        the land use parameter (e.g. 'maxRootDepth', 'rootFraction1', etc.).

        Additionally, this method sets up the crop coefficient and interception capacity data for each land use type by
        retrieving the corresponding data from the land use dataset and interpolating it to the model grid. The crop
        coefficient data is set as attributes of the model with names of the form 'landcover/{land_use_type}/cropCoefficient{land_use_type_netcdf_name}_10days',
        where {land_use_type_netcdf_name} is the name of the land use type in the CWATM dataset. The interception capacity
        data is set as attributes of the model with names of the form 'landcover/{land_use_type}/interceptCap{land_use_type_netcdf_name}_10days',
        where {land_use_type_netcdf_name} is the name of the land use type in the CWATM dataset.

        The resulting land use parameters are set as attributes of the model with names of the form 'landcover/{land_use_type}/{parameter}_{land_use_type}',
        where {land_use_type} is the name of the land use type (e.g. 'forest', 'grassland', etc.) and {parameter} is the name of
        the land use parameter (e.g. 'maxRootDepth', 'rootFraction1', etc.). The crop coefficient data is set as attributes
        of the model with names of the form 'landcover/{land_use_type}/cropCoefficient{land_use_type_netcdf_name}_10days',
        where {land_use_type_netcdf_name} is the name of the land use type in the CWATM dataset. The interception capacity
        data is set as attributes of the model with names of the form 'landcover/{land_use_type}/interceptCap{land_use_type_netcdf_name}_10days',
        where {land_use_type_netcdf_name} is the name of the land use type in the CWATM dataset.
        """
        self.logger.info("Setting up land use parameters")

        bounds = self.geoms["routing/subbasins"].total_bounds
        buffer = 0.1
        landcover_classification = (
            xr.open_dataarray(
                self.data_catalog.get_source(land_cover).path,
                chunks={"x": 3000, "y": 3000},
                mask_and_scale=False,
            )
            .sel(
                x=slice(bounds[0] - buffer, bounds[2] + buffer),
                y=slice(bounds[3] + buffer, bounds[1] - buffer),
            )
            .isel(band=0)
        )

        self.set_other(
            landcover_classification,
            name="landcover/classification",
        )

        target = self.grid["mask"]

        forest_kc = (
            xr.open_dataarray(
                self.data_catalog.get_source("cwatm_forest_5min").path.format(
                    variable="cropCoefficientForest_10days"
                ),
            )
            .rename({"lat": "y", "lon": "x"})
            .rio.write_crs(4326)
        )
        forest_kc = forest_kc.isel(
            **get_window(
                forest_kc.x,
                forest_kc.y,
                self.bounds,
                buffer=3,
            ),
        ).raster.mask_nodata()

        forest_kc = resample_like(forest_kc, target, method="nearest")

        forest_kc.attrs = {
            key: attr
            for key, attr in forest_kc.attrs.items()
            if not key.startswith("NETCDF_") and key != "units"
        }
        self.set_grid(
            forest_kc,
            name="landcover/forest/crop_coefficient",
        )

        for land_use_type in ("forest", "grassland"):
            self.logger.info(f"Setting up land use parameters for {land_use_type}")

            parameter = f"interceptCap{land_use_type.title()}_10days"
            interception_capacity = (
                xr.open_dataarray(
                    self.data_catalog.get_source(
                        f"cwatm_{land_use_type}_5min"
                    ).path.format(variable=parameter),
                )
                .rename({"lat": "y", "lon": "x"})
                .rio.write_crs(4326)
            )
            interception_capacity = interception_capacity.isel(
                **get_window(
                    interception_capacity.x,
                    interception_capacity.y,
                    self.bounds,
                    buffer=3,
                ),
            ).raster.mask_nodata()

            interception_capacity = resample_like(
                interception_capacity, target, method="nearest"
            )

            interception_capacity.attrs = {
                key: attr
                for key, attr in interception_capacity.attrs.items()
                if not key.startswith("NETCDF_") and key != "units"
            }
            self.set_grid(
                interception_capacity,
                name=f"landcover/{land_use_type}/interception_capacity",
            )

    def setup_soil_parameters(self) -> None:
        """Sets up the soil parameters for the model.

        Parameters
        ----------

        Notes:
        -----
        This method sets up the soil parameters for the model by retrieving soil data from the CWATM dataset and interpolating
        the data to the model grid. It first retrieves the soil dataset from the `data_catalog`, and
        then retrieves the soil parameters and storage depth data for each soil layer. It then interpolates the data to the
        model grid using the specified interpolation method and sets the resulting grids as attributes of the model.

        Additionally, this method sets up the percolation impeded and crop group data by retrieving the corresponding data
        from the soil dataset and interpolating it to the model grid.

        The resulting soil parameters are set as attributes of the model with names of the form 'soil/{parameter}{soil_layer}',
        where {parameter} is the name of the soil parameter (e.g. 'alpha', 'ksat', etc.) and {soil_layer} is the index of the
        soil layer (1-3; 1 is the top layer). The storage depth data is set as attributes of the model with names of the
        form 'soil/storage_depth{soil_layer}'. The percolation impeded and crop group data are set as attributes of the model
        with names 'soil/percolation_impeded' and 'soil/cropgrp', respectively.
        """
        self.logger.info("Setting up soil parameters")
        ds = load_soilgrids(self.data_catalog, self.subgrid, self.region)

        self.set_subgrid(ds["silt"], name="soil/silt")
        self.set_subgrid(ds["clay"], name="soil/clay")
        self.set_subgrid(ds["bdod"], name="soil/bulk_density")
        self.set_subgrid(ds["soc"], name="soil/soil_organic_carbon")
        self.set_subgrid(ds["height"], name="soil/soil_layer_height")

        crop_group = (
            xr.open_dataarray(
                self.data_catalog.get_source("cwatm_soil_5min").path.format(
                    variable="cropgrp"
                ),
            )
            .rename({"lat": "y", "lon": "x"})
            .rio.write_crs(4326)
        )
        crop_group = crop_group.isel(
            **get_window(
                crop_group.x,
                crop_group.y,
                self.bounds,
                buffer=10,
            ),
        )
        crop_group.attrs["_FillValue"] = crop_group.attrs["__FillValue"]
        del crop_group.attrs["__FillValue"]

        crop_group = crop_group.raster.mask_nodata()

        crop_group = crop_group.astype(np.float32)

        crop_group = resample_like(
            crop_group,
            self.grid["mask"],
            method="nearest",
        )

        self.set_grid(crop_group, name="soil/crop_group")
