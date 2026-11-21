"""Implements build methods for the land surface submodel, responsible for land surface characteristics and processes."""

import copy
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyflwdir
import shapely
import xarray as xr
from numba import njit, prange
from pyflwdir.dem import fill_depressions
from shapely.ops import linemerge

from geb.build.methods import build_method
from geb.hydrology.landcovers import FOREST, GRASSLAND_LIKE, OPEN_WATER, SEALED
from geb.numba_config import NUMBA_CACHE
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

from ..workflows.soilgrids import process_soilgrids
from .base import BuildModelBase

# Maximum number of elevation zones per grid cell: bottom (0), midslope (1), top (2).
N_ELEVATION_ZONES: int = 3

# Order in which zones that cover too little of a cell are merged into a neighbour,
# as (small zone, zone it merges into). A small top or bottom zone joins the
# midslope; a small midslope joins the bottom zone, or the top zone when the cell
# has no bottom zone. Each step uses the zones as left by the previous steps.
ELEVATION_ZONE_MERGE_ORDER: np.ndarray = np.array(
    [[2, 1], [0, 1], [1, 0], [1, 2]], dtype=np.int8
)


@njit(cache=NUMBA_CACHE, parallel=True)
def assign_elevation_zones(
    height: np.ndarray,
    open_water: np.ndarray,
    factor: int,
    bottom_zone_height_m: float,
    top_zone_relief_factor: float,
    minimum_zone_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assigns every subgrid pixel to an elevation zone of its grid cell.

    See ``LandSurface.setup_elevation_zones`` for the rules. Each grid cell is a
    block of ``factor`` x ``factor`` subgrid pixels and is handled on its own.

    Args:
        height: Height above the nearest stream of every subgrid pixel, NaN outside
            the model domain (m).
        open_water: Whether a subgrid pixel is open water.
        factor: Number of subgrid pixels along one side of a grid cell.
        bottom_zone_height_m: Height above the stream of the upper edge of the
            bottom zone (m).
        top_zone_relief_factor: The land above the bottom zone is split in two
            zones when it spans more than this number of times the bottom zone
            height (-).
        minimum_zone_fraction: Smallest fraction of the cell a zone may cover (-).

    Returns:
        zones: Zone of every subgrid pixel, numbered from 0 without gaps in every
            cell; -1 outside the model domain.
        zone_height: Mean height above the stream of each zone (zone, y, x), NaN
            where the zone does not exist (m).
        zone_fraction: Fraction of the cell covered by each zone (zone, y, x) (-).
    """
    n_rows: int = height.shape[0] // factor
    n_cols: int = height.shape[1] // factor
    n_pixels: int = factor * factor

    zones = np.full(height.shape, -1, dtype=np.int8)
    zone_height = np.full((N_ELEVATION_ZONES, n_rows, n_cols), np.nan, np.float32)
    zone_fraction = np.zeros((N_ELEVATION_ZONES, n_rows, n_cols), np.float32)

    for row in prange(n_rows):
        # buffers with the pixels of one cell, reused for every cell in the row
        cell_height = np.empty(n_pixels, dtype=np.float32)
        cell_water = np.empty(n_pixels, dtype=np.bool_)
        cell_zone = np.empty(n_pixels, dtype=np.int8)
        count = np.empty(N_ELEVATION_ZONES, dtype=np.int64)
        new_number = np.empty(N_ELEVATION_ZONES, dtype=np.int8)

        for col in range(n_cols):
            for i in range(factor):
                for j in range(factor):
                    cell_height[i * factor + j] = height[
                        row * factor + i, col * factor + j
                    ]
                    cell_water[i * factor + j] = open_water[
                        row * factor + i, col * factor + j
                    ]
            n_inside: int = 0
            for k in range(n_pixels):
                if not np.isnan(cell_height[k]):
                    n_inside += 1
            if n_inside == 0:
                continue  # cell outside the model domain: all zones stay empty

            # Relief of the cell: the 95th percentile of the height, so that a few
            # high pixels do not decide the zones.
            relief: float = float(np.nanpercentile(cell_height, 95))
            # The land above the bottom zone spans from the bottom zone height up to
            # the relief. It forms a midslope zone when it has any height, and is
            # split halfway into a midslope and a top zone when it spans more than
            # top_zone_relief_factor times the bottom zone height.
            upper_span: float = relief - bottom_zone_height_m
            has_midslope: bool = upper_span > 0.0
            has_top: bool = upper_span > top_zone_relief_factor * bottom_zone_height_m
            top_edge: float = bottom_zone_height_m + upper_span / 2.0

            for k in range(n_pixels):
                if np.isnan(cell_height[k]):
                    cell_zone[k] = -1  # outside the model domain
                elif cell_water[k]:
                    cell_zone[k] = 0  # open water is always in the lowest zone
                elif has_top and cell_height[k] > top_edge:
                    cell_zone[k] = 2
                elif has_midslope and cell_height[k] > bottom_zone_height_m:
                    cell_zone[k] = 1
                else:
                    cell_zone[k] = 0

            # Merge zones that cover too little of the cell with their neighbour,
            # but only into a zone that exists in the cell.
            count[:] = 0
            for k in range(n_pixels):
                if cell_zone[k] >= 0:
                    count[cell_zone[k]] += 1
            for step in range(ELEVATION_ZONE_MERGE_ORDER.shape[0]):
                small: int = ELEVATION_ZONE_MERGE_ORDER[step, 0]
                into: int = ELEVATION_ZONE_MERGE_ORDER[step, 1]
                fraction: float = count[small] / n_inside
                if (
                    count[small] > 0
                    and fraction < minimum_zone_fraction
                    and count[into] > 0
                ):
                    for k in range(n_pixels):
                        if cell_zone[k] == small:
                            cell_zone[k] = into
                    count[into] += count[small]
                    count[small] = 0

            # Number the remaining zones from 0 without gaps, so the lowest zone of
            # every cell is zone 0.
            next_number: int = 0
            for zone in range(N_ELEVATION_ZONES):
                new_number[zone] = next_number
                if count[zone] > 0:
                    next_number += 1

            # Write the zones back to the subgrid and compute per zone the fraction
            # of the cell it covers and its mean height above the stream.
            height_sum = np.zeros(N_ELEVATION_ZONES, dtype=np.float64)
            for i in range(factor):
                for j in range(factor):
                    k: int = i * factor + j
                    zone_k: int = cell_zone[k]
                    if zone_k >= 0:
                        zone_k = new_number[zone_k]
                        height_sum[zone_k] += cell_height[k]
                    zones[row * factor + i, col * factor + j] = zone_k
            for zone in range(N_ELEVATION_ZONES):
                n_in_zone: int = count[zone]
                if n_in_zone > 0:
                    renumbered: int = new_number[zone]
                    zone_fraction[renumbered, row, col] = n_in_zone / n_inside
                    zone_height[renumbered, row, col] = (
                        height_sum[renumbered] / n_in_zone
                    )

    return zones, zone_height, zone_fraction


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
                "name": "fathomdem",
                "zmin_coastal": 0.001,
                "fill_depressions": False,
            },
            {
                "name": "delta_dtm",
                "zmax": 29.999999,
                "zmin": 0.001,
                "merge_method": "last",
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
            name: The name of the DEM to use. Supported names are 'fathomdem', 'delta_dtm', 'gebco'. If it is not supported,
                the path must be set.
            path: The path to the DEM file. Only required if the name is not supported.
            zmin: The minimum elevation where to use the DEM. Elevations below this value will be set to NaN.
            zmin_coastal: Used as zmin in models with coastal subbasins, where the coastal DEMs fill the sea below it;
                not used in models without coastal subbasins, so that land below sea level keeps its elevation.
            zmax: The maximum elevation where to use the DEM. Elevations above this value will be set to NaN.
            zmax_coastal: Used as zmax in models with coastal subbasins; not used in other models.
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

        has_coastal_subbasins: bool = bool(
            self.geom["routing/subbasins"]["is_coastal"].any()
        )
        for DEM in DEMs:
            zmin_coastal: str | float | None = DEM.pop("zmin_coastal", None)
            zmax_coastal: str | float | None = DEM.pop("zmax_coastal", None)
            if has_coastal_subbasins:
                if zmin_coastal is not None:
                    DEM["zmin"] = zmin_coastal
                if zmax_coastal is not None:
                    DEM["zmax"] = zmax_coastal

        if has_coastal_subbasins:
            # deltaDTM must be present if coastal DEMs are used
            if not any(DEM.get("name", "") == "delta_dtm" for DEM in DEMs):
                raise ValueError(
                    "DeltaDTM DEM must be provided when coastal DEMs are used."
                )

            coastlines = self.geom["coastal/coastlines"]

            # coastlines is a very complex geometry that can cause issues with clipping and masking operations
            # therefore we take various steps to simplify
            coastlines = gpd.GeoDataFrame(
                geometry=shapely.set_precision(
                    coastlines.geometry, grid_size=0.01, mode="keep_collapsed"
                ),
                crs=coastlines.crs,
            )

            # if the coastlines are not a single LineString, we merge them into one
            geom = coastlines.geometry.union_all()
            if geom.geom_type == "MultiLineString":
                geom = linemerge(geom)
            coastlines = gpd.GeoDataFrame(
                geometry=[geom],
                crs=coastlines.crs,
            )

            # buffer coastlines with 0.2 degrees
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                )
                coastlines_with_buffer = coastlines.buffer(0.2, resolution=8)

            coastlines_with_buffer = coastlines_with_buffer.union_all()

            # merge the buffered coastlines with the potential flood area with buffer
            potential_flood_area_with_buffer = potential_flood_area_with_buffer.union(
                coastlines_with_buffer
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

        fathomdem: xr.DataArray = self.data_catalog.fetch(
            "fathomdem",
            mask=potential_flood_area_with_buffer,
        ).read(mask=potential_flood_area_with_buffer)

        potential_flood_area_with_buffer_gdf = gpd.GeoDataFrame(
            geometry=[potential_flood_area_with_buffer], crs=4326
        )

        target: xr.DataArray = self.subgrid["mask"].chunk({"x": 5000, "y": 5000})
        assert target.rio.crs is not None, "target grid must have a crs"

        subgrid_elevation = resample_chunked(fathomdem, target, method="nearest")
        # fill nan values with 0
        subgrid_elevation = subgrid_elevation.where(
            ~np.isnan(subgrid_elevation) & ~self.subgrid["mask"], 0
        )
        self.set_subgrid(
            subgrid_elevation,
            name="landsurface/elevation",
        )

        DEM_raster: xr.DataArray
        for DEM in DEMs:
            if DEM["name"] == "fathomdem":
                DEM_raster = fathomdem

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

            DEM_raster = convert_nodata(
                DEM_raster.astype(np.float32, keep_attrs=True), np.nan
            )

            DEM_raster = clip_with_geometry(
                DEM_raster,
                potential_flood_area_with_buffer_gdf.to_crs(DEM_raster.rio.crs),
                all_touched=True,
                drop=True,
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
        global_countries = global_countries.set_index("ISO3")  # ty:ignore[invalid-assignment]

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
                # nodata in the source is mostly ocean, so it becomes open water
                0: np.int8(OPEN_WATER),
                10: np.int8(FOREST),  # tree cover
                20: np.int8(GRASSLAND_LIKE),  # shrubland
                30: np.int8(GRASSLAND_LIKE),  # grassland
                # cropland starts as non-irrigated; agents set irrigated land later
                40: np.int8(GRASSLAND_LIKE),
                50: np.int8(SEALED),  # built-up
                60: np.int8(GRASSLAND_LIKE),  # bare / sparse vegetation
                70: np.int8(GRASSLAND_LIKE),  # snow and ice
                80: np.int8(OPEN_WATER),  # permanent water bodies
                90: np.int8(GRASSLAND_LIKE),  # herbaceous wetland
                95: np.int8(OPEN_WATER),  # mangroves
                100: np.int8(GRASSLAND_LIKE),  # moss and lichen
            },
            method="lookup",
        )

        land_use_classes_subgrid.attrs["_FillValue"] = -1
        self.set_subgrid(land_use_classes_subgrid, name="landsurface/land_use_classes")

        cultivated_land_subgrid = xr.where(
            land_use_classification_source_subgrid == 40,
            True,
            False,
            keep_attrs=True,
        )
        cultivated_land_subgrid = cultivated_land_subgrid.rio.write_crs(
            land_use_classification_source_subgrid.rio.crs
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

    def _load_soilgrids_variable_by_layer(
        self,
        subgrid_mask: xr.DataArray,
        variable_name: str,
        conversion_factor: float,
        soil_layer_names: list[str],
    ) -> xr.DataArray:
        """Load a SoilGrids variable for all configured soil layers.

        Args:
            subgrid_mask: Target subgrid mask used to request SoilGrids data.
            variable_name: SoilGrids variable name.
            conversion_factor: Multiplier used to convert the source units to the
                model units.
            soil_layer_names: SoilGrids layer labels in top-to-bottom order.

        Returns:
            SoilGrids variable concatenated along the ``soil_layer`` dimension.
        """
        soilgrids_source = self.data_catalog.fetch("soilgridsv2")
        soilgrids_layers: list[xr.DataArray] = []
        for layer_name in soil_layer_names:
            da = soilgrids_source.read(variable=variable_name, depth=layer_name)
            da = da.astype(np.float32)
            soilgrids_layers.append(
                process_soilgrids(
                    da=da,
                    mask=subgrid_mask,
                    region=self.region,
                )
            )

        soil_layer_numbers: list[int] = list(range(1, len(soil_layer_names) + 1))
        return (
            xr.concat(
                soilgrids_layers,
                dim=xr.Variable("soil_layer", soil_layer_numbers),
                compat="equals",
            )
            * conversion_factor
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

            The volumetric coarse fragment (stone) content of each layer (SoilGrids v2
            ``cfvo``) is stored as 'soil/coarse_fragments_percentage' (% of layer volume).
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
            "cfvo": 0.1,  # cm3/dm3 -> % of volume
        }
        soilgrids_output_names: dict[str, str] = {
            "silt": "soil/silt_percentage",
            "clay": "soil/clay_percentage",
            "bdod": "soil/bulk_density_kg_per_dm3",
            "soc": "soil/soil_organic_carbon_percentage",
            "cfvo": "soil/coarse_fragments_percentage",
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
            soilgrids_variable: xr.DataArray = self._load_soilgrids_variable_by_layer(
                subgrid_mask=subgrid_mask,
                variable_name=variable_name,
                conversion_factor=conversion_factor,
                soil_layer_names=soil_layer_names,
            )
            if variable_name == "cfvo":
                # Stones hold no water, so the model scales the water storage and
                # conductivity of each layer by the fine-earth fraction.
                soilgrids_variable = soilgrids_variable.clip(0.0, 100.0)

            soilgrids_variable: xr.DataArray = self.set_subgrid(
                soilgrids_variable,
                name=soilgrids_output_names[variable_name],
            )

        soil_layer_height_per_layer_m: xr.DataArray = xr.DataArray(
            np.array((0.05, 0.10, 0.15, 0.30, 0.40, 1.00), dtype=np.float32),
            dims=("soil_layer",),
            coords={"soil_layer": soilgrids_variable["soil_layer"]},
        )
        soil_layer_height_m = soil_layer_height_per_layer_m.broadcast_like(
            soilgrids_variable
        )
        soil_layer_height_m = soil_layer_height_m.chunk({"soil_layer": 1})
        soil_layer_height_m.attrs["units"] = "m"
        soil_layer_height_m.attrs["description"] = "Height of each soil layer"
        soil_layer_height_m.attrs["_FillValue"] = np.nan
        soil_layer_height_m = soil_layer_height_m.rio.write_crs(
            soilgrids_variable.rio.crs
        )

        self.set_subgrid(soil_layer_height_m, name="soil/soil_layer_height_m")

        # clean up memory
        del soil_layer_height_m
        del soilgrids_variable

        depth_to_bedrock_cm = (
            self.data_catalog.fetch("soilgridsv1")
            .read(variable="BDTICM_M_250m_ll")
            .astype(np.float32)
        )
        depth_to_bedrock_cm = process_soilgrids(
            da=depth_to_bedrock_cm,
            mask=subgrid_mask,
            region=self.region,
        )
        assert isinstance(depth_to_bedrock_cm, xr.DataArray)
        depth_to_bedrock_m: xr.DataArray = (
            depth_to_bedrock_cm / 100
        )  # convert from cm to m

        self.set_subgrid(depth_to_bedrock_m, name="soil/depth_to_bedrock_m")

    @build_method(depends_on=["setup_elevation"], required=True)
    def setup_grid_elevation(self) -> None:
        """Sets up the mean and lowest elevation of each grid cell from the subgrid elevation.

        All elevations of the model grid come from the subgrid elevation, so that they
        are consistent with each other: the mean elevation of the cell
        (``landsurface/elevation_m``: top of the groundwater model, temperature lapse)
        and the lowest elevation in the cell (``landsurface/elevation_min_m``: the
        stream of cells without a represented river, see setup_river_surface_profiles).
        The hydrography DEM (MERIT Hydro) only sets the flow directions; it lies about
        2 m above FathomDEM.
        """
        outside: xr.DataArray = self.subgrid["mask"]
        elevation: np.ndarray = (
            self.subgrid["landsurface/elevation"]
            .where(~outside)
            .values.astype(np.float32)
        )
        factor: int = self.subgrid_factor
        grid_mask: xr.DataArray = self.grid["mask"]
        n_rows, n_cols = grid_mask.shape
        assert elevation.shape == (n_rows * factor, n_cols * factor)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cell_elevation: np.ndarray = elevation.reshape(
                n_rows, factor, n_cols, factor
            )
            cell_mean_elevation = np.nanmean(cell_elevation, axis=(1, 3))
            cell_minimum_elevation = np.nanmin(cell_elevation, axis=(1, 3))
        for values, name in (
            (cell_mean_elevation, "landsurface/elevation_m"),
            (cell_minimum_elevation, "landsurface/elevation_min_m"),
        ):
            grid_elevation = self.full_like(
                grid_mask, fill_value=np.nan, nodata=np.nan, dtype=np.float32
            )
            grid_elevation.data = values.astype(np.float32)
            self.set_grid(grid_elevation, name=name)

    @build_method(
        depends_on=["setup_elevation", "setup_regions_and_land_use"], required=True
    )
    def setup_elevation_zones(
        self,
        bottom_zone_height_m: float = 3.0,
        top_zone_relief_factor: float = 4.0,
        minimum_zone_fraction: float = 0.05,
        channel_initiation_area_km2: float = 0.2,
    ) -> None:
        """Divides every grid cell in up to three zones by height above the nearest stream.

        The land inside a grid cell differs in its position on the hillslope: the valley
        bottom lies next to the stream with the water table close to the surface, the
        slopes and plateaus lie well above it. Each subgrid pixel is assigned to one of
        at most three zones, counted from the stream upwards: the bottom zone (0), the
        midslope (1) and the top (2). The land surface model splits its hydrological
        response units by zone.

        The height above the nearest stream follows the flow directions of the subgrid
        elevation; a stream starts where the upstream area exceeds the channel
        initiation area (the same threshold as for the hillslope length).

        The number of zones depends on the relief of the cell:

        - the bottom zone is the land within ``bottom_zone_height_m`` of the stream;
        - the land above it is one zone, or two zones split halfway up to the highest
          land of the cell (95th percentile of the height) when it spans more than
          ``top_zone_relief_factor`` times ``bottom_zone_height_m``;
        - a zone smaller than ``minimum_zone_fraction`` of the cell is merged with the
          neighbouring zone;
        - open water is always part of the lowest zone of the cell.

        Zones are numbered from 0 without gaps in every cell, so the lowest zone of a
        cell is always zone 0.

        Args:
            bottom_zone_height_m: Height above the stream of the upper edge of the bottom zone (m).
            top_zone_relief_factor: The land above the bottom zone is split in two zones when it
                spans more than this number of times the bottom zone height (-).
            minimum_zone_fraction: Smallest fraction of the cell a zone may cover (-).
            channel_initiation_area_km2: Upstream area above which a subgrid pixel is a stream (km2).
        """
        # Height above the nearest stream (HAND) of every subgrid pixel. Water flows
        # along the D8 flow directions of the depression-filled subgrid elevation, and
        # a pixel is a stream once its upstream area exceeds the channel initiation
        # area. The height of a pixel is its elevation minus that of the stream pixel
        # it drains to.
        outside: np.ndarray = self.subgrid["mask"].values.astype(bool)
        elevation_da: xr.DataArray = self.subgrid["landsurface/elevation"]
        nodata: float = -9999.0
        elevation: np.ndarray = elevation_da.values.astype(np.float32)
        elevation[outside] = nodata
        _, flow_directions = fill_depressions(elevation, nodata=nodata)
        flow_raster = pyflwdir.from_array(
            flow_directions,
            ftype="d8",
            transform=elevation_da.rio.transform(recalc=True),
            latlon=True,
        )
        is_stream: np.ndarray = (
            flow_raster.upstream_area("km2") > channel_initiation_area_km2
        )
        height: np.ndarray = flow_raster.hand(is_stream, elevation).astype(np.float32)
        # Pixels outside the model domain get NaN; small negative heights from the
        # depression filling are set to 0 (the pixel lies at stream level).
        height = np.where(outside | (height == nodata), np.nan, np.maximum(height, 0.0))

        open_water: np.ndarray = (
            self.subgrid["landsurface/land_use_classes"].values == OPEN_WATER
        )
        grid_mask: xr.DataArray = self.grid["mask"]
        assert height.shape == (
            grid_mask.shape[0] * self.subgrid_factor,
            grid_mask.shape[1] * self.subgrid_factor,
        )

        zones, zone_height, zone_fraction = assign_elevation_zones(
            height,
            open_water,
            self.subgrid_factor,
            bottom_zone_height_m,
            top_zone_relief_factor,
            minimum_zone_fraction,
        )

        zones_da: xr.DataArray = self.full_like(
            self.subgrid["landsurface/land_use_classes"],
            fill_value=-1,
            nodata=-1,
            dtype=np.int8,
        )
        zones_da.data = zones
        self.set_subgrid(zones_da, name="landsurface/elevation_zones")

        for values, name in (
            (zone_height, "landsurface/elevation_zone_height_above_stream_m"),
            (zone_fraction, "landsurface/elevation_zone_fraction"),
        ):
            zone_da = xr.DataArray(
                values,
                dims=("zone", "y", "x"),
                coords={
                    "zone": np.arange(N_ELEVATION_ZONES),
                    "y": grid_mask.y,
                    "x": grid_mask.x,
                },
            )
            zone_da.attrs["_FillValue"] = np.nan
            self.set_grid(zone_da, name=name)

    @build_method(depends_on=[], required=True)
    def setup_vegetation(
        self,
    ) -> None:
        """Sets up the vegetation parameters for the model.

        Stores, per grid cell, the crop group number and the leaf area index per
        10-day period for forest and grassland-like vegetation (LISFLOOD), and the
        evergreen fraction of tree cover (ESA CCI Land Cover plant functional types).
        """
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
            leaf_area_index = interpolate_na_along_dim(leaf_area_index)
            leaf_area_index = resample_like(
                leaf_area_index,
                self.grid["mask"],
                method="nearest",
            ).compute()
            self.set_other(
                leaf_area_index, name=f"vegetation/leaf_area_index_{vegetation_type}"
            )

        # Evergreen fraction of tree cover from the ESA CCI Land Cover plant functional
        # type map (2020, 300 m): (needleleaved + broadleaved evergreen trees) / all
        # trees. The model approximates the evergreen part of forest as constant at its
        # annual maximum leaf area index. Satellite LAI underestimates the winter LAI of
        # evergreen canopies (snow, low sun angles, understory die-back), while their
        # true seasonal change is small (~10-30 %). Cells without trees get 0.
        tree_types: list[str] = ["TREES-NE", "TREES-BE", "TREES-ND", "TREES-BD"]
        pft: xr.Dataset = self.data_catalog.fetch("esa_cci_land_cover_pft_2020").read(
            variables=tree_types, bounds=self.bounds
        )
        evergreen_percentage: xr.DataArray = pft["TREES-NE"] + pft["TREES-BE"]
        tree_percentage: xr.DataArray = (
            evergreen_percentage + pft["TREES-ND"] + pft["TREES-BD"]
        )

        # Average numerator and denominator to the grid separately (area-weighted ratio).
        evergreen_on_grid: xr.DataArray = resample_like(
            evergreen_percentage, self.grid["mask"], method="conservative"
        )
        trees_on_grid: xr.DataArray = resample_like(
            tree_percentage, self.grid["mask"], method="conservative"
        )
        evergreen_fraction: xr.DataArray = xr.where(
            trees_on_grid > 0.0, evergreen_on_grid / trees_on_grid, 0.0
        ).astype(np.float32)
        evergreen_fraction = evergreen_fraction.fillna(0.0).clip(0.0, 1.0)
        evergreen_fraction = evergreen_fraction.rio.write_crs(self.grid["mask"].rio.crs)
        evergreen_fraction.attrs["_FillValue"] = np.nan
        self.set_grid(evergreen_fraction, name="vegetation/forest_evergreen_fraction")

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
