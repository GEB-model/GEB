"""Implements build methods for the land surface submodel, responsible for land surface characteristics and processes."""

import geopandas as gpd
import numpy as np
import xarray as xr
from pyflwdir.dem import fill_depressions

from geb.build.methods import build_method
from geb.workflows.io import get_window, read_zarr
from geb.workflows.raster import (
    bounds_are_within,
    calculate_cell_area,
    convert_nodata,
    pad_xy,
    rasterize_like,
    reclassify,
    repeat_grid,
    resample_chunked,
    resample_like,
    snap_to_grid,
)

from ..workflows.soilgrids import load_soilgrids


class LandSurface:
    """Implements land surface submodel, responsible for land surface characteristics and processes."""

    def __init__(self) -> None:
        """Initialize the LandSurface class."""
        pass

    @build_method(depends_on=["setup_regions_and_land_use"])
    def setup_cell_area(self) -> None:
        """Sets up the cell area map for the model.

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
        mask = self.grid["mask"]

        cell_area = self.full_like(
            mask, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )

        cell_area.data = calculate_cell_area(
            mask.rio.transform(recalc=True), mask.shape
        )
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

    @build_method(depends_on=["setup_hydrography"])
    def setup_elevation(
        self,
        DEMs: list[dict[str, str | float]] = [
            {
                "name": "fabdem",
                "zmin": 0.001,
                "fill_depressions": True,
            },
            {"name": "gebco", "zmax": 0.0, "fill_depressions": False},
        ],
    ) -> None:
        """Sets up the elevation data for the model.

        For configuration of DEMs parameters, see
        https://deltares.github.io/hydromt_sfincs/latest/_generated/hydromt_sfincs.SfincsModel.setup_dep.html.

        Args:
            DEMs: A list of dictionaries containing the names and parameters of the DEMs to use. Each dictionary should have a 'name' key
                with the name of the DEM, and optionally other keys such as 'zmin' for minimum elevation.

        """
        if not DEMs:
            DEMs = []

        assert isinstance(DEMs, list)
        # here we use the bounds of all subbasins, which may include downstream
        # subbasins that are not part of the study area
        bounds: tuple[float, float, float, float] = tuple(
            self.geom["routing/subbasins"].total_bounds
        )

        buffer: float = 0.5
        xmin: float = bounds[0] - buffer
        ymin: float = bounds[1] - buffer
        xmax: float = bounds[2] + buffer
        ymax: float = bounds[3] + buffer
        fabdem: xr.DataArray = (
            self.new_data_catalog.fetch(
                "fabdem",
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                prefix="hydrodynamics",
            )
            .read(prefix="hydrodynamics")
            .compute()
        )

        target: xr.DataArray = self.subgrid["mask"]
        assert target.rio.crs is not None, "target grid must have a crs"

        self.set_subgrid(
            resample_like(fabdem, target, method="bilinear"),
            name="landsurface/elevation",
        )

        for DEM in DEMs:
            if DEM["name"] == "fabdem":
                DEM_raster = fabdem
            else:
                if DEM["name"] == "gebco":
                    DEM_raster = self.new_data_catalog.fetch("gebco").read()
                else:
                    if DEM["name"] == "geul_dem":
                        DEM_raster = read_zarr(
                            self.data_catalog.get_source(DEM["name"]).path
                        )
                    else:
                        DEM_raster = xr.open_dataarray(
                            self.data_catalog.get_source(DEM["name"]).path,
                        )
                if "bands" in DEM_raster.dims:
                    DEM_raster = DEM_raster.isel(band=0)

                DEM_raster = DEM_raster.isel(
                    get_window(
                        DEM_raster.x,
                        DEM_raster.y,
                        tuple(
                            self.geom["routing/subbasins"]
                            .to_crs(DEM_raster.rio.crs)
                            .total_bounds
                        ),
                        buffer=100,
                        raise_on_out_of_bounds=False,
                        raise_on_buffer_out_of_bounds=False,
                    ),
                )

            DEM_raster = convert_nodata(
                DEM_raster.astype(np.float32, keep_attrs=True), np.nan
            )

            if "fill_depressions" in DEM and DEM["fill_depressions"]:
                DEM_raster.values, d8 = fill_depressions(DEM_raster.values)

            self.set_other(
                DEM_raster,
                name=f"DEM/{DEM['name']}",
            )
            DEM["path"] = f"DEM/{DEM['name']}"
        low_elevation_coastal_zone = DEM_raster < 10
        low_elevation_coastal_zone.values = low_elevation_coastal_zone.values.astype(
            np.float32
        )
        self.set_other(
            low_elevation_coastal_zone, name="landsurface/low_elevation_coastal_zone"
        )  # Maybe remove this
        self.set_dict(DEMs, name="hydrodynamics/DEM_config")

    @build_method(depends_on=[])
    def setup_regions_and_land_use(
        self,
        region_database: str = "GADM_level1",
        unique_region_id: str = "GID_1",
        ISO3_column: str = "GID_0",
        land_cover: str = "esa_worldcover_2021",
    ) -> None:
        """Sets up the (administrative) regions and land use data for GEB.

        The regions can be used for multiple purposes, for example for creating the
        agents in the model, assigning unique crop prices and other economic variables
        per region and for aggregating the results.

        Args:
            region_database: The name of the region database to use. Default is 'GADM_level1'.
            unique_region_id: The name of a column in the region database that contains a unique region ID. Default is 'UID',
                which is the unique identifier for the GADM database.
            ISO3_column: The name of a column in the region database that contains the ISO3 code for the region. Default is 'ISO3'.
            land_cover: The name of the land cover dataset to use. Default is 'esa_worldcover_2021'.

        Notes:
            This method sets up the regions and land use data for GEB. It first retrieves the region data from
            the specified region database and sets it as a geometry in the model. It then pads the subgrid to cover the entire
            region and retrieves the land use data from the ESA WorldCover dataset. The land use data is reprojected to the
            padded subgrid and the region ID is rasterized onto the subgrid. The cell area for each region is calculated and
            set as a grid in the model. The MERIT dataset is used to identify rivers, which are set as a grid in the model. The
            land use data is reclassified into five classes and set as a grid in the model. Finally, the cultivated land is
            identified and set as a grid in the model.
        """
        regions: gpd.GeoDataFrame = (
            self.new_data_catalog.fetch(region_database)
            .read(geom=self.region.union_all())
            .rename(columns={unique_region_id: "region_id", ISO3_column: "ISO3"})
        )

        global_countries: gpd.GeoDataFrame = (
            self.new_data_catalog.fetch("GADM_level0")
            .read()
            .rename(columns={"GID_0": "ISO3"})
        )

        global_countries["geometry"] = global_countries.centroid
        global_countries = global_countries.set_index("ISO3")

        self.set_geom(global_countries, name="global_countries")

        assert np.unique(regions["region_id"]).shape[0] == regions.shape[0], (
            f"Region database must contain unique region IDs ({self.data_catalog[region_database].path})"
        )

        # allow some tolerance, especially for regions that coincide with coastlines, in which
        # case the region boundaries may be slightly outside the model region due to differences
        # in coastline representation. This is especially relevant for islands.
        assert bounds_are_within(
            self.region.total_bounds,
            regions.to_crs(self.region.crs).total_bounds,
            tolerance=0.1,
        )

        region_id_mapping = {
            i: region_id for region_id, i in enumerate(regions["region_id"])
        }
        regions["region_id"] = regions["region_id"].map(region_id_mapping)

        self.set_dict(region_id_mapping, name="region_id_mapping")

        assert "ISO3" in regions.columns, (
            f"Region database must contain ISO3 column ({self.data_catalog[region_database].path})"
        )

        self.set_geom(regions, name="regions")

        resolution_x, resolution_y = self.subgrid["mask"].rio.resolution()

        regions_bounds: tuple[float, float, float, float] = self.geom[
            "regions"
        ].total_bounds
        mask_bounds: tuple[float, float, float, float] = self.grid["mask"].rio.bounds(
            recalc=True
        )

        # The bounds should be set to a bit larger than the regions to avoid edge effects
        # and also larger than the mask, to ensure that the entire grid is covered.
        pad_minx = min(regions_bounds[0], mask_bounds[0]) - abs(resolution_x) / 2.0
        pad_miny = min(regions_bounds[1], mask_bounds[1]) - abs(resolution_y) / 2.0
        pad_maxx = max(regions_bounds[2], mask_bounds[2]) + abs(resolution_x) / 2.0
        pad_maxy = max(regions_bounds[3], mask_bounds[3]) + abs(resolution_y) / 2.0

        # TODO: Is there a better way to do this?
        region_mask, region_subgrid_slice = pad_xy(
            self.subgrid["mask"],
            pad_minx,
            pad_miny,
            pad_maxx,
            pad_maxy,
            return_slice=True,
            constant_values=1,
        )
        region_mask.attrs["_FillValue"] = None
        region_mask = self.set_region_subgrid(region_mask, name="mask")

        bounds = self.geom["regions"].total_bounds
        xmin: float = bounds[0] - 0.1
        ymin: float = bounds[1] - 0.1
        xmax: float = bounds[2] + 0.1
        ymax: float = bounds[3] + 0.1

        land_use: xr.DataArray = (
            self.new_data_catalog.fetch(land_cover)
            .read(xmin, ymin, xmax, ymax)
            .chunk({"x": 1000, "y": 1000})
        )

        reprojected_land_use: xr.DataArray = resample_chunked(
            land_use, region_mask.chunk({"x": 1000, "y": 1000}), method="nearest"
        )

        reprojected_land_use: xr.DataArray = self.set_region_subgrid(
            reprojected_land_use,
            name="landsurface/original_land_use",
        )

        region_ids: xr.DataArray = rasterize_like(
            gdf=self.geom["regions"],
            column="region_id",
            raster=region_mask,
            dtype=np.int32,
            nodata=-1,
            all_touched=True,
        ).compute()
        region_ids: xr.DataArray = self.set_region_subgrid(
            region_ids, name="region_ids"
        )

        full_region_land_use_classes = reclassify(
            reprojected_land_use,
            {
                reprojected_land_use.attrs[
                    "_FillValue"
                ]: 5,  # no data, set to permanent water bodies because ocean
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
        ).astype(np.int32)
        full_region_land_use_classes.attrs["_FillValue"] = -1

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
        land_use_classes = snap_to_grid(land_use_classes, self.subgrid)
        self.set_subgrid(land_use_classes, name="landsurface/land_use_classes")

        cultivated_land = cultivated_land_full_region.isel(region_subgrid_slice)
        cultivated_land = snap_to_grid(cultivated_land, self.subgrid)
        self.set_subgrid(cultivated_land, name="landsurface/cultivated_land")

    @build_method(depends_on=[])
    def setup_land_use_parameters(
        self,
        land_cover: str = "esa_worldcover_2021",
    ) -> None:
        """Sets up the land use parameters for the model.

        Args:
            land_cover: The name of the land cover dataset to use. Default is 'esa_worldcover_2021'.

        Notes:
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
        bounds = self.geom["routing/subbasins"].total_bounds
        buffer = 0.1

        xmin = bounds[0] - buffer
        ymin = bounds[1] - buffer
        xmax = bounds[2] + buffer
        ymax = bounds[3] + buffer

        landcover_classification: xr.DataArray = self.new_data_catalog.fetch(
            land_cover
        ).read(xmin, ymin, xmax, ymax)

        landcover_classification = self.set_other(
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
        forest_kc.attrs["_FillValue"] = np.nan
        forest_kc: xr.DataArray = forest_kc.isel(
            get_window(
                forest_kc.x,
                forest_kc.y,
                self.bounds,
                buffer=3,
            ),
        )
        forest_kc: xr.DataArray = resample_like(forest_kc, target, method="nearest")

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
            interception_capacity.attrs["_FillValue"] = np.nan
            interception_capacity: xr.DataArray = interception_capacity.isel(
                get_window(
                    interception_capacity.x,
                    interception_capacity.y,
                    self.bounds,
                    buffer=3,
                ),
            )
            interception_capacity: xr.DataArray = resample_like(
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

    @build_method(depends_on=[])
    def setup_soil_parameters(self) -> None:
        """Sets up the soil parameters for the model.

        Notes:
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
        ds: xr.Dataset = load_soilgrids(
            self.new_data_catalog, self.subgrid["mask"], self.region
        )

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
            get_window(
                crop_group.x,
                crop_group.y,
                self.bounds,
                buffer=10,
            ),
        )
        crop_group.attrs["_FillValue"] = crop_group.attrs["__FillValue"]
        del crop_group.attrs["__FillValue"]

        crop_group: xr.DataArray = crop_group.astype(np.float32)
        crop_group: xr.DataArray = convert_nodata(crop_group, np.nan)

        crop_group = resample_like(
            crop_group,
            self.grid["mask"],
            method="nearest",
        )

        self.set_grid(crop_group, name="soil/crop_group")
