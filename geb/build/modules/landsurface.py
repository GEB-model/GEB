"""Implements build methods for the land surface submodel, responsible for land surface characteristics and processes."""

import copy
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from pyflwdir.dem import fill_depressions

from geb.build.methods import build_method
from geb.workflows.io import get_window, parse_and_set_zarr_CRS
from geb.workflows.raster import (
    calculate_cell_area_m2,
    clip_with_geometry,
    convert_nodata,
    interpolate_na_2d,
    interpolate_na_along_dim,
    rasterize_like,
    reclassify,
    repeat_grid,
    resample_chunked,
    resample_like,
)

from ..workflows.soilgrids import load_soilgrids_v2
from .base import BuildModelBase


class LandSurface(BuildModelBase):
    """Implements land surface submodel, responsible for land surface characteristics and processes."""

    def __init__(self) -> None:
        """Initialize the LandSurface class."""
        pass

    @build_method(required=True)
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

        height, width = cell_area.shape
        cell_area.data = calculate_cell_area_m2(
            mask.rio.transform(recalc=True), height, width
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
        self.set_subgrid(sub_cell_area.chunk({"x": -1, "y": -1}), name="cell_area")

    @build_method(
        depends_on=[
            "setup_hydrography",
            "setup_coastlines",
        ],
        required=True,
    )
    def setup_elevation(
        self,
        DEMs: list[dict[str, str | float]] = [
            {
                "name": "fabdem",
                "zmin": 0.001,
                "coastal_zmin": 30.0,
                "fill_depressions": False,
            },
            {
                "name": "delta_dtm",
                "zmax": 30,
                "zmin": 0.001,
                "fill_depressions": False,
                "coastal_only": True,
            },
            {
                "name": "gebco",
                "zmax": 0.0,
                "fill_depressions": False,
                "coastal_only": True,
            },
        ],
    ) -> None:
        """Sets up the elevation data for the model.

        Configuration parameters:
            name: The name of the DEM to use. Supported names are 'fabdem', 'delta_dtm', 'gebco'. If it is not supported,
                the path must be set.
            path: The path to the DEM file. Only required if the name is not supported.
            zmin: The minimum elevation where to use the DEM. Elevations below this value will be set to NaN.
            zmin_coastal: The minimum elevation where to use the DEM for coastal subbasins. Elevations below this value will be set to NaN.
            zmax: The maximum elevation where to use the DEM. Elevations above this value will be set to NaN.
            zmax_coastal: The maximum elevation where to use the DEM for coastal subbasins. Elevations above this value will be set to NaN.
            fill_depressions: Whether to fill depressions in the DEM. Default is False. Note that this may use a lot of memory for large DEMs.
            nodata: The nodata value in the DEM. Optional, only required if the DEM does not have a nodata value defined.
            crs: The CRS to set for custom DEMs when the file does not define one (EPSG code or CRS string).
            coastal_only: DEMs with this value set to True will be skipped if there are no coastal subbasins in the model.
                Default is False.


        Args:
            DEMs: A list of dictionaries containing the names and parameters of the DEMs to use. Each dictionary should
                be configured as described above.

        Raises:
            ValueError: If no DEMs are provided.
            ValueError: If the DEMs are not provided as a list of dictionaries.
            ValueError: If CRS is missing or invalid in a custom DEM.
            ValueError: If nodata value is missing in a custom DEM.
            ValueError: If DeltaDTM DEM is not provided when coastal subbasins are present.
            ValueError: If a custom DEM CRS is not a valid EPSG code or CRS string.
        """
        DEMs = copy.deepcopy(DEMs)

        if not DEMs:
            raise ValueError("At least one DEM must be provided.")

        if not isinstance(DEMs, list) or not all(isinstance(DEM, dict) for DEM in DEMs):
            raise ValueError("DEMs must be provided as a list of dictionaries.")

        potential_flood_area_with_buffer = (
            self.geom["routing/subbasins"].union_all().buffer(0.1)
        )

        if self.geom["routing/subbasins"]["is_coastal"].any():
            # deltaDTM must be present if coastal DEMs are used
            if not any(DEM.get("name", "") == "delta_dtm" for DEM in DEMs):
                raise ValueError(
                    "DeltaDTM DEM must be provided when coastal DEMs are used."
                )

            for DEM in DEMs:
                if "coastal_zmin" in DEM:
                    DEM["zmin"] = DEM["coastal_zmin"]
                if "coastal_zmax" in DEM:
                    DEM["zmax"] = DEM["coastal_zmax"]

            coastlines = self.geom["coastal/coastlines"]
            potential_flood_area_with_buffer = potential_flood_area_with_buffer.union(
                coastlines.buffer(0.2).union_all()
            )

            delta_dtm: xr.DataArray = self.data_catalog.fetch(
                "delta_dtm",
                mask=potential_flood_area_with_buffer,
            ).read(mask=potential_flood_area_with_buffer)

            # Create low elevation coastal zone mask based on DeltaDTM
            low_elevation_coastal_zone = delta_dtm < 10
            low_elevation_coastal_zone = low_elevation_coastal_zone.astype(np.float32)
            self.set_other(
                low_elevation_coastal_zone,
                name="landsurface/low_elevation_coastal_zone",
            )  # Maybe remove this

        else:
            # Remove coastal DEMs if no coastal subbasins are present
            DEMs = [DEM for DEM in DEMs if not DEM.get("coastal_only", False)]

        fabdem: xr.DataArray = self.data_catalog.fetch(
            "fabdem",
            mask=potential_flood_area_with_buffer,
        ).read()

        target: xr.DataArray = self.subgrid["mask"].chunk({"x": 5000, "y": 5000})
        assert target.rio.crs is not None, "target grid must have a crs"

        self.set_subgrid(
            resample_chunked(fabdem, target, method="nearest"),
            name="landsurface/elevation",
        )

        DEM_raster: xr.DataArray
        for DEM in DEMs:
            # FABDEM is already handled above, so we just use it from there
            if DEM["name"] == "fabdem":
                DEM_raster: xr.DataArray = fabdem

            elif DEM["name"] == "delta_dtm":
                DEM_raster: xr.DataArray = delta_dtm

            elif DEM["name"] == "gebco":
                DEM_raster: xr.DataArray = self.data_catalog.fetch("gebco").read()
                # set maximum values for DEM_raster if zmax is set for the DEM
                if "zmax" in DEM:
                    DEM_raster = DEM_raster.where(
                        DEM_raster <= DEM["zmax"], DEM["zmax"]
                    )

            else:
                # custom DEMs must have a path
                if "path" not in DEM:
                    raise ValueError(
                        f"DEM name '{DEM['name']}' is not supported by default. Please provide a valid path."
                    )
                if not isinstance(DEM["path"], str):
                    raise ValueError("DEM path must be a string.")

                DEM_raster: xr.DataArray = xr.open_dataarray(Path(DEM["path"]))

                # Handle CRS for custom DEMs
                # Zarrs need special handling to set the CRS
                if DEM["path"].endswith(".zarr") or DEM["path"].endswith(".zarr.zip"):
                    DEM_raster = parse_and_set_zarr_CRS(DEM_raster)

                if "crs" in DEM:
                    crs_value = DEM["crs"]
                    if not isinstance(crs_value, (int, str)):
                        raise ValueError(
                            "Custom DEM CRS must be an EPSG code (int) or CRS string."
                        )
                    DEM_raster = DEM_raster.rio.write_crs(crs_value)

                if DEM_raster.rio.crs is None:
                    raise ValueError(
                        f"DEM at path '{DEM['path']}' does not have a valid CRS."
                    )

                # Handle nodata for custom DEMs
                if "nodata" in DEM:
                    DEM_raster.attrs["_FillValue"] = DEM["nodata"]
                else:
                    if "_FillValue" not in DEM_raster.attrs:
                        raise ValueError(
                            f"DEM at path '{DEM['path']}' does not have a nodata value defined."
                        )

            if "band" in DEM_raster.dims:
                DEM_raster: xr.DataArray = DEM_raster.isel(band=0)

            DEM_raster = clip_with_geometry(
                DEM_raster,
                gpd.GeoDataFrame(geometry=[potential_flood_area_with_buffer], crs=4326),
                all_touched=True,
                drop=True,
            )

            DEM_raster = convert_nodata(
                DEM_raster.astype(np.float32, keep_attrs=True), np.nan
            )

            if DEM.get("fill_depressions", False):
                DEM_raster.values, d8 = fill_depressions(
                    DEM_raster.values, nodata=DEM_raster.attrs["_FillValue"]
                )

            self.set_other(
                DEM_raster.chunk({"x": 5000, "y": 5000}),
                name=f"DEM/{DEM['name']}",
            )
            DEM["path"] = f"DEM/{DEM['name']}"

        self.set_params(DEMs, name="hydrodynamics/DEM_config")

    @build_method(depends_on=["setup_coastal_sfincs_model_regions"], required=True)
    def setup_regions_and_land_use(
        self,
        region_database: str = "GADM_level1",
        unique_region_id: str = "GID_1",
        ISO3_column: str = "GID_0",
        land_cover: str = "esa_worldcover_2021",
    ) -> None:
        """Sets up the (administrative) regions and land use data for GEB.

        Args:
            region_database: The name of the region database to use. Default is 'GADM_level1'.
            unique_region_id: The name of a column in the region database that contains a unique region ID. Default is 'UID',
                which is the unique identifier for the GADM database.
            ISO3_column: The name of a column in the region database that contains the ISO3 code for the region. Default is 'ISO3'.
            land_cover: The name of the land cover dataset to use. Default is 'esa_worldcover_2021'.
        """
        regions: gpd.GeoDataFrame = (
            self.data_catalog.fetch(region_database)
            .read(geom=self.region.union_all())
            .rename(columns={unique_region_id: "region_id", ISO3_column: "ISO3"})
        )

        global_countries: gpd.GeoDataFrame = (
            self.data_catalog.fetch("GADM_level0")
            .read()
            .rename(columns={"GID_0": "ISO3"})
        )

        global_countries["geometry"] = global_countries.to_crs(
            "ESRI:54009"
        ).centroid.to_crs(global_countries.crs)
        global_countries = global_countries.set_index("ISO3")

        self.set_geom(global_countries, name="global_countries")

        assert np.unique(regions["region_id"]).shape[0] == regions.shape[0], (
            f"Region database must contain unique region IDs"
        )

        region_id_mapping = {
            i: region_id for region_id, i in enumerate(regions["region_id"])
        }
        regions["region_id"] = regions["region_id"].map(region_id_mapping)

        self.set_params(region_id_mapping, name="region_id_mapping")

        assert "ISO3" in regions.columns, f"Region database must contain ISO3 column)"

        self.set_geom(regions, name="regions")

        subgrid_region_ids: xr.DataArray = rasterize_like(
            gdf=self.geom["regions"],
            column="region_id",
            raster=self.subgrid["mask"],
            dtype=np.int32,
            nodata=-1,
            all_touched=True,
        )

        self.set_subgrid(
            subgrid_region_ids, name="region_ids", shards={"x": 10, "y": 10}
        )

        potential_flood_area_with_buffer = (
            self.geom["routing/subbasins"].union_all().buffer(0.1)
        )
        if self.geom["routing/subbasins"]["is_coastal"].any():
            potential_flood_area_with_buffer = potential_flood_area_with_buffer.union(
                self.geom["coastal/low_elevation_coastal_zone_mask"]
                .union_all()
                .buffer(0.2)
            )

        land_use_classification_source: xr.DataArray = self.data_catalog.fetch(
            land_cover
        ).read(potential_flood_area_with_buffer)

        land_use_classification_source_within_potential_flood_area = clip_with_geometry(
            land_use_classification_source,
            gpd.GeoDataFrame(geometry=[potential_flood_area_with_buffer], crs=4326),
            all_touched=True,
            drop=True,
        )

        land_use_classification_source_within_potential_flood_area = self.set_other(
            land_use_classification_source_within_potential_flood_area,
            name="landcover/classification",
            shards={"x": 10, "y": 10},
        )

        land_use_classification_source_subgrid: xr.DataArray = resample_chunked(
            land_use_classification_source_within_potential_flood_area,
            self.subgrid["mask"].chunk({"x": 500, "y": 500}),
            method="nearest",
        )

        land_use_classification_source_subgrid = self.set_subgrid(
            land_use_classification_source_subgrid,
            name="landcover/classification",
            shards={"x": 5, "y": 5},
        )

        land_use_classes_subgrid = reclassify(
            land_use_classification_source_subgrid,
            remap_dict={
                0: np.int8(
                    5
                ),  # map nodata in source to permanent water bodies, as these are mostly ocean in the land cover dataset
                10: np.int8(0),  # tree cover
                20: np.int8(1),  # shrubland
                30: np.int8(1),  # grassland
                40: np.int8(
                    1
                ),  # cropland, setting to non-irrigated. Initiated as irrigated based on agents
                50: np.int8(4),  # built-up
                60: np.int8(1),  # bare / sparse vegetation
                70: np.int8(1),  # snow and ice
                80: np.int8(5),  # permanent water bodies
                90: np.int8(1),  # herbaceous wetland
                95: np.int8(5),  # mangroves
                100: np.int8(1),  # moss and lichen
            },
            method="lookup",
        )

        land_use_classes_subgrid.attrs["_FillValue"] = -1
        self.set_subgrid(land_use_classes_subgrid, name="landsurface/land_use_classes")

        cultivated_land_subgrid = xr.where(
            land_use_classification_source_subgrid == 40,
            True,
            False,
        )

        cultivated_land_subgrid.attrs["_FillValue"] = None
        self.set_subgrid(cultivated_land_subgrid, name="landsurface/cultivated_land")

    @build_method(depends_on=[], required=False)
    def setup_land_use_parameters(
        self,
        land_cover: str = "esa_worldcover_2021",
    ) -> None:
        """This method is removed."""
        self.logger.warning(
            "setup_land_use_parameters is removed, please remove it from your build configuration"
        )

    @build_method(depends_on=[], required=True)
    def setup_soil(self) -> None:
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
        # Keep this commented code because we want to include differentiation between valley and hillslope soils later
        # soil_depth = self.data_catalog.fetch("GlobalSoilRegolithSediment").read()
        # assert isinstance(soil_depth, xr.Dataset)
        # soil_depth = soil_depth.isel(
        #     get_window(
        #         soil_depth.x,
        #         soil_depth.y,
        #         self.bounds,
        #         buffer=10,
        #     ),
        # )

        subgrid_mask: xr.DataArray = self.subgrid["mask"]
        soilgrids_conversion_factors: dict[str, float] = {
            "silt": 0.1,  # g/kg -> g/100g (%)
            "clay": 0.1,  # g/kg -> g/100g (%)
            "bdod": 0.01,  # cg/cm³ -> gr/cm³
            "soc": 0.01,  # dg/kg -> g/100g (%)
        }
        soilgrids_output_names: dict[str, str] = {
            "silt": "soil/silt_percentage",
            "clay": "soil/clay_percentage",
            "bdod": "soil/bulk_density_kg_per_dm3",
            "soc": "soil/soil_organic_carbon_percentage",
        }
        soil_layer_names: list[str] = [
            "0-5cm",
            "5-15cm",
            "15-30cm",
            "30-60cm",
            "60-100cm",
            "100-200cm",
        ]

        for variable_name, conversion_factor in soilgrids_conversion_factors.items():
            soilgrids_variables: list[xr.DataArray] = []
            for soil_layer, layer_name in enumerate(soil_layer_names, start=1):
                soilgrids_variables.append(
                    load_soilgrids_v2(
                        self.data_catalog,
                        subgrid_mask,
                        self.region,
                        variable_name=variable_name,
                        layer_name=layer_name,
                    )
                    * conversion_factor
                )

            soilgrids_variable: xr.DataArray = xr.concat(
                soilgrids_variables,
                dim=xr.Variable("soil_layer", [1, 2, 3, 4, 5, 6]),
                compat="equals",
            )
            self.set_subgrid(
                soilgrids_variable,
                name=soilgrids_output_names[variable_name],
            )

        soil_layer_height_m: xr.DataArray = xr.full_like(
            soilgrids_variable, fill_value=0.0, dtype=np.float32
        )
        for layer_index, layer_height_m in enumerate(
            (0.05, 0.10, 0.15, 0.30, 0.40, 1.00)
        ):
            soil_layer_height_m[layer_index] = layer_height_m

        self.set_subgrid(soil_layer_height_m, name="soil/soil_layer_height_m")

        depth_to_bedrock_cm = (
            self.data_catalog.fetch("soilgridsv1")
            .read(variable="BDTICM_M_250m_ll")
            .astype(np.float32)
        )
        assert isinstance(depth_to_bedrock_cm, xr.DataArray)
        depth_to_bedrock_cm: xr.DataArray = resample_like(
            depth_to_bedrock_cm, subgrid_mask
        ).chunk({"x": -1, "y": -1})
        depth_to_bedrock_cm: xr.DataArray = convert_nodata(depth_to_bedrock_cm, np.nan)

        depth_to_bedrock_m: xr.DataArray = (
            depth_to_bedrock_cm / 100
        )  # convert from cm to m

        depth_to_bedrock_m: xr.DataArray = interpolate_na_2d(depth_to_bedrock_m)

        self.set_subgrid(depth_to_bedrock_m, name="soil/depth_to_bedrock_m")

    @build_method(depends_on=[], required=True)
    def setup_vegetation(
        self,
    ) -> None:
        """Sets up the vegetation parameters for the model."""
        for vegetation_type in ("forest", "grassland_like"):
            crop_group_number: xr.DataArray = self.data_catalog.fetch(
                f"lisflood_crop_group_number_{vegetation_type}"
            ).read()
            crop_group_number = crop_group_number.isel(
                get_window(
                    crop_group_number.x,
                    crop_group_number.y,
                    self.bounds,
                    buffer=10,
                ),
            )

            crop_group_number = crop_group_number.astype(np.float32)
            crop_group_number = convert_nodata(crop_group_number, np.nan)
            crop_group_number = interpolate_na_2d(crop_group_number)
            crop_group_number = resample_like(
                crop_group_number,
                self.grid["mask"],
                method="nearest",
            )
            self.set_grid(
                crop_group_number,
                name=f"vegetation/crop_group_number_{vegetation_type}",
            )

            leaf_area_index: xr.DataArray = self.data_catalog.fetch(
                f"lisflood_leaf_area_index_{vegetation_type}"
            ).read()
            leaf_area_index = leaf_area_index.isel(
                get_window(
                    leaf_area_index.x,
                    leaf_area_index.y,
                    self.bounds,
                    buffer=10,
                ),
            )

            leaf_area_index = leaf_area_index.astype(np.float32)
            leaf_area_index = convert_nodata(leaf_area_index, np.nan)
            leaf_area_index = interpolate_na_along_dim(leaf_area_index, dim="time")
            leaf_area_index = resample_like(
                leaf_area_index,
                self.grid["mask"],
                method="nearest",
            ).compute()
            self.set_other(
                leaf_area_index, name=f"vegetation/leaf_area_index_{vegetation_type}"
            )

    @build_method(depends_on=[], required=False)
    def setup_forest_restoration_potential(self) -> None:
        """Sets up the forest restoration potential data for the model.

        Source data is in percentage, which is converted to ratio.
        """
        forest_restoration_potential_percentage = self.data_catalog.fetch(
            "forest_restoration_potential"
        ).read()
        assert isinstance(forest_restoration_potential_percentage, xr.DataArray)
        forest_restoration_potential_percentage = (
            forest_restoration_potential_percentage.isel(
                get_window(
                    forest_restoration_potential_percentage.x,
                    forest_restoration_potential_percentage.y,
                    self.bounds,
                    buffer=2,
                ),
            ).compute()
        )

        forest_restoration_potential_percentage = interpolate_na_2d(
            forest_restoration_potential_percentage
        )
        forest_restoration_potential_percentage = resample_like(
            forest_restoration_potential_percentage, self.grid["mask"]
        )
        forest_restoration_potential_ratio = (
            forest_restoration_potential_percentage / 100
        )  # convert from percentage to ratio
        self.set_grid(
            forest_restoration_potential_ratio,
            name="landsurface/forest_restoration_potential_ratio",
        )
