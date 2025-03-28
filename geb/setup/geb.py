"""This module contains the main setup for the GEB model.

Notes:
- All prices are in nominal USD (face value) for their respective years. That means that the prices are not adjusted for inflation.
"""

import inspect
import json
import logging
import math
import os
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import geopandas as gpd
import hydromt.workflows
import numpy as np
import pandas as pd
import pyflwdir
import rasterio
import requests
import xarray as xr
from affine import Affine
from dateutil.relativedelta import relativedelta
from honeybees.library.raster import pixels_to_coords, sample_from_map
from hydromt.data_catalog import DataCatalog
from hydromt.exceptions import NoDataException
from isimip_client.client import ISIMIPClient
from rasterio.env import defenv
from scipy.ndimage import value_indices
from tqdm import tqdm

from geb.agents.crop_farmers import (
    FIELD_EXPANSION_ADAPTATION,
    IRRIGATION_EFFICIENCY_ADAPTATION,
    SURFACE_IRRIGATION_EQUIPMENT,
    WELL_ADAPTATION,
)
from geb.hydrology.lakes_reservoirs import LAKE, RESERVOIR

from ..workflows.io import open_zarr, to_zarr
from .modules import Crops, Forcing
from .workflows.conversions import (
    COUNTRY_NAME_TO_ISO3,
    GLOBIOM_NAME_TO_ISO3,
    SUPERWELL_NAME_TO_ISO3,
)
from .workflows.farmers import create_farms, get_farm_distribution, get_farm_locations
from .workflows.general import (
    bounds_are_within,
    calculate_cell_area,
    clip_with_grid,
    fetch_and_save,
    pad_xy,
    repeat_grid,
)
from .workflows.hydrography import (
    create_river_raster_from_river_lines,
    get_downstream_subbasins,
    get_river_graph,
    get_rivers,
    get_sink_subbasin_id_for_geom,
    get_subbasin_id_from_coordinate,
    get_subbasins_geometry,
    get_SWORD_river_widths,
    get_SWORD_translation_IDs_and_lenghts,
    get_upstream_subbasin_ids,
)
from .workflows.population import load_GLOPOP_S
from .workflows.soilgrids import load_soilgrids

# Set environment options for robustness
GDAL_HTTP_ENV_OPTS = {
    "GDAL_HTTP_MAX_RETRY": "10",  # Number of retry attempts
    "GDAL_HTTP_RETRY_DELAY": "2",  # Delay (seconds) between retries
    "GDAL_HTTP_TIMEOUT": "30",  # Timeout in seconds
}
defenv(**GDAL_HTTP_ENV_OPTS)

XY_CHUNKSIZE = 3000  # chunksize for xy coordinates

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

logger = logging.getLogger(__name__)


def convert_timestamp_to_string(timestamp):
    return timestamp.isoformat()


@contextmanager
def suppress_logging_warning(logger):
    """
    A context manager to suppress logging warning messages temporarily.
    """
    current_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)  # Set level to ERROR to suppress WARNING messages
    try:
        yield
    finally:
        logger.setLevel(current_level)  # Restore the original logging level


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            obj = obj.as_posix()
            return str(obj)
        return super().default(obj)


class GEBModel(Forcing, Crops):
    def __init__(
        self,
        root: str = None,
        data_catalogs: List[str] = None,
        logger=logger,
        epsg=4326,
        data_provider: str = None,
    ):
        self.logger = logger
        self.data_catalog = DataCatalog(data_libs=data_catalogs, logger=self.logger)

        Forcing.__init__(self)
        Crops.__init__(self)

        self.root = root
        self.epsg = epsg
        self.data_provider = data_provider

        # the grid, subgrid, and region subgrids are all datasets, which should
        # have exactly matching coordinates
        self.grid = xr.Dataset()
        self.subgrid = xr.Dataset()
        self.region_subgrid = xr.Dataset()

        # all other data types are dictionaries because these entries don't
        # necessarily match the grid coordinates, shapes etc.
        self.geoms = {}
        self.table = {}
        self.array = {}
        self.dict = {}
        self.other = {}

        self.files = defaultdict(dict)

    def setup_region(
        self,
        region: dict,
        subgrid_factor: int,
        resolution_arcsec=30,
    ) -> xr.DataArray:
        """Creates a 2D regular grid or reads an existing grid.
        An 2D regular grid will be created from a geometry (geom_fn) or bbox. If an existing
        grid is given, then no new grid will be generated.

        Adds/Updates model layers:
        * **grid** grid mask: add grid mask to grid object

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:
            * {'basin': [x, y]}

            Region must be of kind [basin, subbasin].
        subgrid_factor : int
            GEB implements a subgrid. This parameter determines the factor by which the subgrid is smaller than the original grid.
        """

        assert resolution_arcsec % 3 == 0, (
            "resolution_arcsec must be a multiple of 3 to align with MERIT"
        )
        assert subgrid_factor >= 2

        river_graph = get_river_graph(self.data_catalog)

        if "subbasin" in region:
            sink_subbasin_ids = region["subbasin"]
        elif "outflow" in region:
            lon, lat = region["outflow"][0], region["outflow"][1]
            sink_subbasin_ids = get_subbasin_id_from_coordinate(
                self.data_catalog, lon, lat
            )
        elif "admin" in region:
            admin_regions = self.data_catalog.get_geodataframe(
                region["admin"]["source"]
            )
            admin_regions = admin_regions[
                admin_regions[region["admin"]["column"]] == region["admin"]["key"]
            ]
            sink_subbasin_ids = get_sink_subbasin_id_for_geom(
                self.data_catalog, admin_regions, river_graph
            )
        else:
            raise ValueError(f"Region {region} not understood.")

        # always make a list of the subbasin ids, such that the function always gets the same type of input
        if not isinstance(sink_subbasin_ids, (list, set)):
            sink_subbasin_ids = [sink_subbasin_ids]

        subbasin_ids = get_upstream_subbasin_ids(river_graph, sink_subbasin_ids)
        subbasin_ids.update(sink_subbasin_ids)

        downstream_subbasins = get_downstream_subbasins(river_graph, sink_subbasin_ids)
        subbasin_ids.update(downstream_subbasins)

        subbasins = get_subbasins_geometry(self.data_catalog, subbasin_ids).set_index(
            "COMID"
        )
        subbasins["is_downstream_outflow_subbasin"] = pd.Series(
            True, index=downstream_subbasins
        ).reindex(subbasins.index, fill_value=False)

        subbasins["associated_upstream_basins"] = pd.Series(
            downstream_subbasins.values(), index=downstream_subbasins
        ).reindex(subbasins.index, fill_value=[])

        self.set_geoms(subbasins, name="routing/subbasins")

        xmin, ymin, xmax, ymax = subbasins.total_bounds
        hydrography = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            bbox=[
                xmin,
                ymin,
                xmax,
                ymax,
            ],
            buffer=10,
        )

        self.logger.info("Preparing 2D grid.")
        if "outflow" in region:
            # get basin geometry
            geom, _ = hydromt.workflows.get_basin_geometry(
                ds=hydrography,
                flwdir_name="dir",
                kind="subbasin",
                logger=self.logger,
                xy=(lon, lat),
            )
        elif "subbasin" in region or "admin" in region:
            geom = gpd.GeoDataFrame(
                geometry=[
                    subbasins[~subbasins["is_downstream_outflow_subbasin"]].union_all()
                ],
                crs=subbasins.crs,
            )
        elif "geom" in region:
            geom = region["geom"]
            if geom.crs is None:
                raise ValueError('Model region "geom" has no CRS')
            # merge regions when more than one geom is given
            if isinstance(geom, gpd.GeoDataFrame):
                geom = gpd.GeoDataFrame(geometry=[geom.union_all()], crs=geom.crs)
        else:
            raise ValueError(f"Region {region} not understood.")

        # ESPG 6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global) is an equal area projection
        # while thhe shape of the polygons becomes vastly different, the area is preserved mostly.
        # usable between 86°S and 86°N.
        self.logger.info(
            f"Approximate basin size in km2: {round(geom.to_crs(epsg=6933).area.sum() / 1e6, 2)}"
        )

        hydrography = hydrography.raster.clip_geom(
            geom,
            align=resolution_arcsec
            / 60
            / 60,  # align grid to resolution of model grid. Conversion is to convert from arcsec to degrees
            mask=True,
        )
        flwdir = hydrography["dir"].values
        assert flwdir.dtype == np.uint8

        flow_raster = pyflwdir.from_array(
            flwdir,
            ftype="d8",
            transform=hydrography.rio.transform(),
            latlon=True,  # hydrography is specified in latlon
            mask=hydrography.mask,  # this mask is True within study area
        )
        hydrography = hydrography.drop_vars(["mask"])

        scale_factor = resolution_arcsec // 3

        # IHU = Iterative hydrography upscaling method, see https://doi.org/10.5194/hess-25-5287-2021
        flow_raster_upscaled, idxs_out = flow_raster.upscale(
            scale_factor=scale_factor,
            method="ihu",
        )
        flow_raster_upscaled.repair_loops()

        elevation_coarsened = (
            hydrography["elv"]
            .raster.mask_nodata()
            .coarsen(
                x=scale_factor, y=scale_factor, boundary="exact", coord_func="mean"
            )
        )

        # elevation (we only set this later, because it has to be done after setting the mask)
        elevation = elevation_coarsened.mean()

        flow_mask = ~flow_raster_upscaled.mask.reshape(flow_raster_upscaled.shape)

        mask = self.full_like(elevation, fill_value=False, nodata=None, dtype=bool)
        # we use the inverted mask, that is True outside the study area
        mask.data = flow_mask
        self.set_grid(mask, name="mask")
        self.set_grid(elevation, name="landsurface/elevation")

        mask_geom = list(
            rasterio.features.shapes(
                flow_mask.astype(np.uint8),
                mask=~flow_mask,
                connectivity=8,
                transform=elevation.rio.transform(recalc=True),
            ),
        )
        mask_geom = gpd.GeoDataFrame.from_features(
            [{"geometry": geom[0], "properties": {}} for geom in mask_geom],
            crs=hydrography.rio.crs,
        )

        self.set_geoms(mask_geom, name="mask")

        self.set_grid(
            elevation_coarsened.std(),
            name="landsurface/elevation_standard_deviation",
        )

        # outflow elevation
        outflow_elevation = elevation_coarsened.min()
        self.set_grid(outflow_elevation, name="routing/outflow_elevation")

        slope = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        slope.raster.set_nodata(np.nan)
        slope_data = pyflwdir.dem.slope(
            elevation.values,
            nodata=np.nan,
            latlon=True,
            transform=elevation.rio.transform(recalc=True),
        )
        # set slope to zero on the mask boundary
        slope_data[np.isnan(slope_data) & (~mask.data)] = 0
        slope.data = slope_data
        self.set_grid(slope, name="landsurface/slope")

        # flow direction
        ldd = self.full_like(
            outflow_elevation, fill_value=255, nodata=255, dtype=np.uint8
        )
        ldd.data = flow_raster_upscaled.to_array(ftype="ldd")
        self.set_grid(ldd, name="routing/ldd")

        # upstream area
        upstream_area = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        upstream_area_data = flow_raster_upscaled.upstream_area(unit="m2").astype(
            np.float32
        )
        upstream_area_data[upstream_area_data == -9999.0] = np.nan
        upstream_area.data = upstream_area_data
        self.set_grid(upstream_area, name="routing/upstream_area")

        # river length
        river_length = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_length_data = flow_raster.subgrid_rivlen(
            idxs_out, unit="m", direction="down"
        )
        river_length_data[river_length_data == -9999.0] = np.nan
        river_length.data = river_length_data
        self.set_grid(river_length, name="routing/river_length")

        # river slope
        river_slope = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_slope_data = flow_raster.subgrid_rivslp(idxs_out, hydrography["elv"])
        river_slope_data[river_slope_data == -9999.0] = np.nan
        river_slope.data = river_slope_data
        self.set_grid(
            river_slope,
            name="routing/river_slope",
        )

        # river width
        rivers = get_rivers(self.data_catalog, subbasin_ids).set_index("COMID")

        # remove all rivers that are both shorter than 1 km and have no upstream river
        rivers = rivers[~((rivers["lengthkm"] < 1) & (rivers["maxup"] == 0))]

        rivers = rivers.join(
            subbasins[["is_downstream_outflow_subbasin", "associated_upstream_basins"]],
            how="left",
        )

        COMID_IDs_raster_data = create_river_raster_from_river_lines(
            rivers, idxs_out, hydrography
        )

        assert set(
            np.unique(COMID_IDs_raster_data[COMID_IDs_raster_data != -1])
        ) == set(np.unique(COMID_IDs_raster_data[COMID_IDs_raster_data != -1]))

        # Derive the xy coordinates of the river network. Here the coordinates
        # are the PIXEL coordinates for the coarse drainage network.
        rivers["hydrography_xy"] = [[]] * len(rivers)
        xy_per_river_segment = value_indices(COMID_IDs_raster_data, ignore_value=-1)
        for COMID, (ys, xs) in xy_per_river_segment.items():
            upstream_area = upstream_area_data[ys, xs]
            up_to_downstream_ids = np.argsort(upstream_area)
            ys = ys[up_to_downstream_ids]
            xs = xs[up_to_downstream_ids]
            rivers.at[COMID, "hydrography_xy"] = list(zip(xs, ys))

        COMID_IDs_raster = self.full_like(
            outflow_elevation, fill_value=-1, nodata=-1, dtype=np.int32
        )
        COMID_IDs_raster.data = COMID_IDs_raster_data
        self.set_grid(COMID_IDs_raster, name="routing/river_ids")

        SWORD_reach_IDs, SWORD_reach_lengths = get_SWORD_translation_IDs_and_lenghts(
            self.data_catalog, rivers
        )

        SWORD_river_widths = get_SWORD_river_widths(self.data_catalog, SWORD_reach_IDs)
        MINIMUM_RIVER_WIDTH = 3.0
        rivers["width"] = np.nansum(
            SWORD_river_widths * SWORD_reach_lengths, axis=0
        ) / np.nansum(SWORD_reach_lengths, axis=0)
        # ensure that all rivers with a SWORD ID have a width
        assert (~np.isnan(rivers["width"][(SWORD_reach_IDs != -1).any(axis=0)])).all()

        # set initial width guess where width is not available from SWORD
        rivers.loc[rivers["width"].isnull(), "width"] = (
            rivers[rivers["width"].isnull()]["uparea"] / 10
        )
        rivers["width"] = rivers["width"].clip(lower=float(MINIMUM_RIVER_WIDTH))

        self.set_geoms(rivers, name="routing/rivers")

        river_width_data = np.vectorize(
            lambda ID: rivers["width"].to_dict().get(ID, float(MINIMUM_RIVER_WIDTH))
        )(COMID_IDs_raster).astype(np.float32)

        river_width = self.full_like(
            outflow_elevation, fill_value=np.nan, nodata=np.nan, dtype=np.float32
        )
        river_width.data = river_width_data
        self.set_grid(river_width, name="routing/river_width")

        dst_transform = mask.rio.transform(recalc=True) * Affine.scale(
            1 / subgrid_factor
        )

        submask = xr.DataArray(
            data=repeat_grid(mask.data, subgrid_factor),
            coords={
                "y": dst_transform.f
                + np.arange(mask.raster.shape[0] * subgrid_factor) * dst_transform.e,
                "x": dst_transform.c
                + np.arange(mask.raster.shape[1] * subgrid_factor) * dst_transform.a,
            },
            attrs={"_FillValue": None},
        )

        submask = self.set_subgrid(submask, name="mask")
        return None

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
        """For configuration of DEMs parameters, see
        https://deltares.github.io/hydromt_sfincs/latest/_generated/hydromt_sfincs.SfincsModel.setup_dep.html
        """
        if not DEMs:
            DEMs = []

        assert isinstance(DEMs, list)
        # here we use the bounds of all subbasins, which may include downstream
        # subbasins that are not part of the study area
        bounds = tuple(self.geoms["routing/subbasins"].total_bounds)

        fabdem = self.data_catalog.get_rasterdataset(
            "fabdem",
            bbox=bounds,
            buffer=100,
        ).raster.mask_nodata()

        target = self.subgrid["mask"]
        target.raster.set_crs(4326)

        self.set_subgrid(
            fabdem.raster.reproject_like(target, method="average"),
            name="landsurface/elevation",
        )

        for DEM in DEMs:
            if DEM["name"] == "fabdem":
                DEM_raster = fabdem
            else:
                DEM_raster = self.data_catalog.get_rasterdataset(
                    DEM["name"],
                    bbox=bounds,
                    buffer=100,
                ).raster.mask_nodata()

            DEM_raster = DEM_raster.astype(np.float32)
            self.set_other(
                DEM_raster,
                name=f"DEM/{DEM['name']}",
                byteshuffle=True,
            )
            DEM["path"] = f"DEM/{DEM['name']}"

        self.set_dict(DEMs, name="hydrodynamics/DEM_config")

    def setup_cell_area(self) -> None:
        """
        Sets up the cell area map for the model.

        Raises
        ------
        ValueError
            If the grid mask is not available.

        Notes
        -----
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

    def setup_cell_area_map(self) -> None:
        self.logger.warn(
            "setup_cell_area_map is deprecated, use setup_cell_area instead"
        )
        self.setup_cell_area()

    def process_additional_years(
        self, costs, total_years, unique_regions, lower_bound=None, upper_bound=None
    ):
        inflation = self.dict["socioeconomics/inflation_rates"]
        for _, region in unique_regions.iterrows():
            region_id = region["region_id"]

            if lower_bound:
                costs = self.process_region_years(
                    costs=costs,
                    inflation=inflation,
                    region_id=region_id,
                    start_year=total_years[0],
                    end_year=lower_bound,
                )

            if upper_bound:
                costs = self.process_region_years(
                    costs=costs,
                    inflation=inflation,
                    region_id=region_id,
                    start_year=total_years[-1],
                    end_year=upper_bound,
                )

        return costs

    def process_region_years(
        self,
        costs,
        inflation,
        region_id,
        start_year,
        end_year,
    ):
        assert end_year != start_year, (
            "extra processed years must not be the same as data years"
        )

        if end_year < start_year:
            operator = "div"
            step = -1
        else:
            operator = "mul"
            step = 1

        inflation_rate_region = inflation["data"][str(region_id)]

        for year in range(start_year, end_year, step):
            year_str = str(year)
            year_index = inflation["time"].index(year_str)
            inflation_rate = inflation_rate_region[year_index]

            # Check and add an empty row if needed
            if (region_id, year) not in costs.index:
                empty_row = pd.DataFrame(
                    {col: [None] for col in costs.columns},
                    index=pd.MultiIndex.from_tuples(
                        [(region_id, year)], names=["region_id", "year"]
                    ),
                )
                costs = pd.concat(
                    [costs, empty_row]
                ).sort_index()  # Ensure the index is sorted after adding new rows

            # Update costs based on inflation rate and operation
            if operator == "div":
                costs.loc[(region_id, year)] = (
                    costs.loc[(region_id, year + 1)] / inflation_rate
                )
            elif operator == "mul":
                costs.loc[(region_id, year)] = (
                    costs.loc[(region_id, year - 1)] * inflation_rate
                )

        return costs

    def setup_mannings(self) -> None:
        """
        Sets up the Manning's coefficient for the model.

        Notes
        -----
        This method sets up the Manning's coefficient for the model by calculating the coefficient based on the cell area
        and topography of the grid. It first calculates the upstream area of each cell in the grid using the
        `routing/upstream_area` attribute of the grid. It then calculates the coefficient using the formula:

            C = 0.025 + 0.015 * (2 * A / U) + 0.030 * (Z / 2000)

        where C is the Manning's coefficient, A is the cell area, U is the upstream area, and Z is the elevation of the cell.

        The resulting Manning's coefficient is then set as the `routing/mannings` attribute of the grid using the
        `set_grid()` method.
        """
        self.logger.info("Setting up Manning's coefficient")
        a = (2 * self.grid["cell_area"]) / self.grid["routing/upstream_area"]
        a = xr.where(a < 1, a, 1, keep_attrs=True)
        b = self.grid["routing/outflow_elevation"] / 2000
        b = xr.where(b < 1, b, 1, keep_attrs=True)

        mannings = 0.025 + 0.015 * a + 0.030 * b
        mannings.attrs["_FillValue"] = np.nan

        self.set_grid(mannings, "routing/mannings")

    def setup_soil_parameters(self) -> None:
        """
        Sets up the soil parameters for the model.

        Parameters
        ----------

        Notes
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

        crop_group = self.data_catalog.get_rasterdataset(
            "cwatm_soil_5min", bbox=self.bounds, buffer=10, variables=["cropgrp"]
        ).raster.mask_nodata()

        self.set_grid(self.interpolate(crop_group, "linear"), name="soil/crop_group")

    def setup_land_use_parameters(
        self,
        land_cover="esa_worldcover_2021_v200",
    ) -> None:
        """
        Sets up the land use parameters for the model.

        Parameters
        ----------
        interpolation_method : str, optional
            The interpolation method to use when interpolating the land use parameters. Default is 'nearest'.

        Notes
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

        # landcover
        landcover_classification = self.data_catalog.get_rasterdataset(
            land_cover,
            bbox=self.bounds,
            buffer=200,  # 2 km buffer
        )
        self.set_other(
            landcover_classification,
            name="landcover/classification",
        )

        target = self.grid["mask"]
        target.raster.set_crs(4326)

        for land_use_type, land_use_type_netcdf_name, simple_name in (
            ("forest", "Forest", "forest"),
            ("grassland", "Grassland", "grassland"),
            ("irrPaddy", "irrPaddy", "paddy_irrigated"),
            ("irrNonPaddy", "irrNonPaddy", "irrigated"),
        ):
            self.logger.info(f"Setting up land use parameters for {land_use_type}")
            land_use_ds = self.data_catalog.get_rasterdataset(
                f"cwatm_{land_use_type}_5min", bbox=self.bounds, buffer=10
            )

            parameter = f"cropCoefficient{land_use_type_netcdf_name}_10days"
            crop_coefficient = land_use_ds[parameter].raster.mask_nodata()
            crop_coefficient = crop_coefficient.raster.reproject_like(
                target, method="nearest"
            )
            self.set_grid(
                crop_coefficient,
                name=f"landcover/{simple_name}/crop_coefficient",
            )
            if land_use_type in ("forest", "grassland"):
                parameter = f"interceptCap{land_use_type_netcdf_name}_10days"
                interception_capacity = land_use_ds[parameter].raster.mask_nodata()
                interception_capacity = interception_capacity.raster.reproject_like(
                    target, method="nearest"
                )
                self.set_grid(
                    interception_capacity,
                    name=f"landcover/{simple_name}/interception_capacity",
                )

    def setup_waterbodies(
        self,
        command_areas=None,
        custom_reservoir_capacity=None,
    ):
        """
        Sets up the waterbodies for GEB.

        Notes
        -----
        This method sets up the waterbodies for GEB. It first retrieves the waterbody data from the
        specified data catalog and sets it as a geometry in the model. It then rasterizes the waterbody data onto the model
        grid and the subgrid using the `rasterize` method of the `raster` object. The resulting grids are set as attributes
        of the model with names of the form 'waterbodies/{grid_name}'.

        The method also retrieves the reservoir command area data from the data catalog and calculates the area of each
        command area that falls within the model region. The `waterbody_id` key is used to do the matching between these
        databases. The relative area of each command area within the model region is calculated and set as a column in
        the waterbody data. The method sets all lakes with a command area to be reservoirs and updates the waterbody data
        with any custom reservoir capacity data from the data catalog.

        TODO: Make the reservoir command area data optional.

        The resulting waterbody data is set as a table in the model with the name 'waterbodies/waterbody_data'.
        """
        self.logger.info("Setting up waterbodies")
        dtypes = {
            "waterbody_id": np.int32,
            "waterbody_type": np.int32,
            "volume_total": np.float64,
            "average_discharge": np.float64,
            "average_area": np.float64,
        }
        try:
            waterbodies = self.data_catalog.get_geodataframe(
                "hydro_lakes",
                geom=self.region,
                predicate="intersects",
                variables=[
                    "waterbody_id",
                    "waterbody_type",
                    "volume_total",
                    "average_discharge",
                    "average_area",
                ],
            )
            waterbodies = waterbodies.astype(dtypes)
            hydrolakes_to_geb = {
                1: LAKE,
                2: RESERVOIR,
            }
            waterbodies["waterbody_type"] = waterbodies["waterbody_type"].map(
                hydrolakes_to_geb
            )
        except NoDataException:
            self.logger.info(
                "No water bodies found in domain, skipping water bodies setup"
            )
            waterbodies = gpd.GeoDataFrame(
                columns=[
                    "waterbody_id",
                    "waterbody_type",
                    "volume_total",
                    "average_discharge",
                    "average_area",
                    "geometry",
                ],
                crs=4326,
            )
            waterbodies = waterbodies.astype(dtypes)
            water_body_id = xr.zeros_like(self.grid["mask"], dtype=np.int32)
            sub_water_body_id = xr.zeros_like(self.subgrid["mask"], dtype=np.int32)
        else:
            water_body_id = self.grid.raster.rasterize(
                waterbodies,
                col_name="waterbody_id",
                nodata=-1,
                all_touched=True,
                dtype=np.int32,
            )
            sub_water_body_id = self.subgrid.raster.rasterize(
                waterbodies,
                col_name="waterbody_id",
                nodata=-1,
                all_touched=True,
                dtype=np.int32,
            )

        water_body_id.attrs["_FillValue"] = -1
        sub_water_body_id.attrs["_FillValue"] = -1

        self.set_grid(water_body_id, name="waterbodies/water_body_id")
        self.set_subgrid(sub_water_body_id, name="waterbodies/sub_water_body_id")

        waterbodies["volume_flood"] = waterbodies["volume_total"]

        if command_areas:
            command_areas = self.data_catalog.get_geodataframe(
                command_areas, geom=self.region, predicate="intersects"
            )
            command_areas = command_areas[
                ~command_areas["waterbody_id"].isnull()
            ].reset_index(drop=True)
            command_areas["waterbody_id"] = command_areas["waterbody_id"].astype(
                np.int32
            )

            self.set_grid(
                self.grid.raster.rasterize(
                    command_areas,
                    col_name="waterbody_id",
                    nodata=-1,
                    all_touched=True,
                    dtype=np.int32,
                ),
                name="waterbodies/command_area",
            )
            self.set_subgrid(
                self.subgrid.raster.rasterize(
                    command_areas,
                    col_name="waterbody_id",
                    nodata=-1,
                    all_touched=True,
                    dtype=np.int32,
                ),
                name="waterbodies/subcommand_areas",
            )

            # set all lakes with command area to reservoir
            waterbodies.loc[
                waterbodies.index.isin(command_areas["waterbody_id"]), "waterbody_type"
            ] = RESERVOIR
        else:
            command_areas = self.full_like(
                self.grid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )
            subcommand_areas = self.full_like(
                self.subgrid["mask"],
                fill_value=-1,
                nodata=-1,
                dtype=np.int32,
            )

            self.set_grid(command_areas, name="waterbodies/command_area")
            self.set_subgrid(subcommand_areas, name="waterbodies/subcommand_areas")

        if custom_reservoir_capacity:
            custom_reservoir_capacity = self.data_catalog.get_dataframe(
                custom_reservoir_capacity
            )
            custom_reservoir_capacity = custom_reservoir_capacity[
                custom_reservoir_capacity.index != -1
            ]

            waterbodies.set_index("waterbody_id", inplace=True)
            waterbodies.update(custom_reservoir_capacity)
            waterbodies.reset_index(inplace=True)

        # spatial dimension is not required anymore, so drop it.
        waterbodies = waterbodies.drop("geometry", axis=1)

        assert "waterbody_id" in waterbodies.columns, "waterbody_id is required"
        assert "waterbody_type" in waterbodies.columns, "waterbody_type is required"
        assert "volume_total" in waterbodies.columns, "volume_total is required"
        assert "average_discharge" in waterbodies.columns, (
            "average_discharge is required"
        )
        assert "average_area" in waterbodies.columns, "average_area is required"
        self.set_table(waterbodies, name="waterbodies/waterbody_data")

    def setup_water_demand(self, starttime, endtime, ssp):
        """
        Sets up the water demand data for GEB.

        Notes
        -----
        This method sets up the water demand data for GEB. It retrieves the domestic, industry, and
        livestock water demand data from the specified data catalog and sets it as forcing data in the model. The domestic
        water demand and consumption data are retrieved from the 'cwatm_domestic_water_demand' dataset, while the industry
        water demand and consumption data are retrieved from the 'cwatm_industry_water_demand' dataset. The livestock water
        consumption data is retrieved from the 'cwatm_livestock_water_demand' dataset.

        The domestic water demand and consumption data are provided at a monthly time step, while the industry water demand
        and consumption data are provided at an annual time step. The livestock water consumption data is provided at a
        monthly time step, but is assumed to be constant over the year.

        The resulting water demand data is set as forcing data in the model with names of the form 'water_demand/{demand_type}'.
        """
        self.logger.info("Setting up water demand")

        def set(file, accessor, name, ssp, starttime, endtime):
            ds_historic = self.data_catalog.get_rasterdataset(
                f"cwatm_{file}_historical_year", bbox=self.bounds, buffer=2
            )
            if accessor:
                ds_historic = getattr(ds_historic, accessor)
            ds_future = self.data_catalog.get_rasterdataset(
                f"cwatm_{file}_{ssp}_year", bbox=self.bounds, buffer=2
            )
            if accessor:
                ds_future = getattr(ds_future, accessor)
            ds_future = ds_future.sel(
                time=slice(ds_historic.time[-1] + 1, ds_future.time[-1])
            )

            ds = xr.concat([ds_historic, ds_future], dim="time")
            # assert dataset in monotonicically increasing
            assert (ds.time.diff("time") == 1).all(), "not all years are there"

            ds["time"] = pd.date_range(
                start=datetime(1901, 1, 1)
                + relativedelta(years=int(ds.time[0].data.item())),
                periods=len(ds.time),
                freq="AS",
            )

            assert (ds.time.dt.year.diff("time") == 1).all(), "not all years are there"
            ds = ds.sel(time=slice(starttime, endtime))
            ds = ds.rename({"lat": "y", "lon": "x"})
            ds.attrs["_FillValue"] = np.nan
            self.set_other(ds, name=f"water_demand/{name}")

        set(
            "domestic_water_demand",
            "domWW",
            "domestic_water_demand",
            ssp,
            starttime,
            endtime,
        )
        set(
            "domestic_water_demand",
            "domCon",
            "domestic_water_consumption",
            ssp,
            starttime,
            endtime,
        )
        set(
            "industry_water_demand",
            "indWW",
            "industry_water_demand",
            ssp,
            starttime,
            endtime,
        )
        set(
            "industry_water_demand",
            "indCon",
            "industry_water_consumption",
            ssp,
            starttime,
            endtime,
        )
        set(
            "livestock_water_demand",
            None,
            "livestock_water_consumption",
            ssp,
            starttime,
            endtime,
        )

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

    def snap_to_grid(self, ds, reference, relative_tollerance=0.02, ydim="y", xdim="x"):
        # make sure all datasets have more or less the same coordinates
        assert np.isclose(
            ds.coords[ydim].values,
            reference[ydim].values,
            atol=abs(ds.rio.resolution()[1] * relative_tollerance),
            rtol=0,
        ).all()
        assert np.isclose(
            ds.coords[xdim].values,
            reference[xdim].values,
            atol=abs(ds.rio.resolution()[0] * relative_tollerance),
            rtol=0,
        ).all()
        return ds.assign_coords({ydim: reference[ydim], xdim: reference[xdim]})

    def setup_regions_and_land_use(
        self,
        region_database="GADM_level1",
        unique_region_id="GID_1",
        ISO3_column="GID_0",
        land_cover="esa_worldcover_2021_v200",
    ):
        """
        Sets up the (administrative) regions and land use data for GEB. The regions can be used for multiple purposes,
        for example for creating the agents in the model, assigning unique crop prices and other economic variables
        per region and for aggregating the results.

        Parameters
        ----------
        region_database : str, optional
            The name of the region database to use. Default is 'GADM_level1'.
        unique_region_id : str, optional
            The name of the column in the region database that contains the unique region ID. Default is 'UID',
            which is the unique identifier for the GADM database.

        Notes
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

        land_use = self.data_catalog.get_rasterdataset(
            land_cover,
            geom=self.geoms["regions"],
            buffer=200,  # 2 km buffer
        )
        region_mask.raster.set_crs(4326)
        reprojected_land_use = land_use.raster.reproject_like(
            region_mask, method="nearest"
        )

        region_ids = reprojected_land_use.raster.rasterize(
            self.geoms["regions"],
            col_name="region_id",
            all_touched=True,
        )
        region_ids.attrs["_FillValue"] = -1
        region_ids = self.set_region_subgrid(region_ids, name="region_ids")

        region_subgrid_cell_area = self.full_like(
            region_mask, fill_value=np.nan, nodata=np.nan, dtype=np.float32
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

    def setup_economic_data(
        self, project_future_until_year=False, reference_start_year=2000
    ):
        """
        Sets up the economic data for GEB.

        Notes
        -----
        This method sets up the lending rates and inflation rates data for GEB. It first retrieves the
        lending rates and inflation rates data from the World Bank dataset using the `get_geodataframe` method of the
        `data_catalog` object. It then creates dictionaries to store the data for each region, with the years as the time
        dimension and the lending rates or inflation rates as the data dimension.

        The lending rates and inflation rates data are converted from percentage to rate by dividing by 100 and adding 1.
        The data is then stored in the dictionaries with the region ID as the key.

        The resulting lending rates and inflation rates data are set as forcing data in the model with names of the form
        'socioeconomics/lending_rates' and 'socioeconomics/inflation_rates', respectively.
        """
        self.logger.info("Setting up economic data")
        assert (
            not project_future_until_year
            or project_future_until_year > reference_start_year
        ), (
            f"project_future_until_year ({project_future_until_year}) must be larger than reference_start_year ({reference_start_year})"
        )

        # lending_rates = self.data_catalog.get_dataframe("wb_lending_rate")
        inflation_rates = self.data_catalog.get_dataframe("wb_inflation_rate")
        price_ratio = self.data_catalog.get_dataframe("world_bank_price_ratio")

        def filter_and_rename(df, additional_cols):
            # Select columns: 'Country Name', 'Country Code', and columns containing "YR"
            columns_to_keep = additional_cols + [
                col
                for col in df.columns
                if col.isnumeric() and 1900 <= int(col) <= 3000
            ]
            filtered_df = df[columns_to_keep]
            return filtered_df

        def extract_years(df):
            # Extract years that are numerically valid between 1900 and 3000
            return [
                col
                for col in df.columns
                if col.isnumeric() and 1900 <= int(col) <= 3000
            ]

        # Assuming dataframes for PPP and LCU per USD have been initialized
        price_ratio_filtered = filter_and_rename(
            price_ratio, ["Country Name", "Country Code"]
        )
        years_price_ratio = extract_years(price_ratio_filtered)
        price_ratio_dict = {"time": years_price_ratio, "data": {}}  # price ratio

        # Assume lending_rates and inflation_rates are available
        # years_lending_rates = extract_years(lending_rates)
        years_inflation_rates = extract_years(inflation_rates)

        # lending_rates_dict = {"time": years_lending_rates, "data": {}}
        inflation_rates_dict = {"time": years_inflation_rates, "data": {}}

        # Create a helper to process rates and assert single row data
        def process_rates(df, rate_cols, ISO3, convert_percent_to_ratio=False):
            filtered_data = df.loc[df["Country Code"] == ISO3, rate_cols]
            assert len(filtered_data) == 1, (
                f"Expected one row for {ISO3}, got {len(filtered_data)}"
            )
            if convert_percent_to_ratio:
                return (filtered_data.iloc[0] / 100 + 1).tolist()
            return filtered_data.iloc[0].tolist()

        USA_inflation_rates = process_rates(
            inflation_rates,
            years_inflation_rates,
            "USA",
            convert_percent_to_ratio=True,
        )

        for _, region in self.geoms["regions"].iterrows():
            region_id = str(region["region_id"])

            # Store data in dictionaries
            # lending_rates_dict["data"][region_id] = process_rates(
            #     lending_rates,
            #     years_lending_rates,
            #     region["ISO3"],
            #     convert_percent_to_ratio=True,
            # )
            ISO3 = region["ISO3"]
            if (
                ISO3 == "AND"
            ):  # for Andorra (not available in World Bank data), use Spain's data
                self.logger.warning(
                    "Andorra's economic data not available, using Spain's data"
                )
                ISO3 = "ESP"
            elif ISO3 == "LIE":  # for Liechtenstein, use Switzerland's data
                self.logger.warning(
                    "Liechtenstein's economic data not available, using Switzerland's data"
                )
                ISO3 = "CHE"

            local_inflation_rates = process_rates(
                inflation_rates,
                years_inflation_rates,
                ISO3,
                convert_percent_to_ratio=True,
            )
            assert not np.isnan(local_inflation_rates).any(), (
                f"Missing inflation rates for {region['ISO3']}"
            )
            inflation_rates_dict["data"][region_id] = (
                np.array(local_inflation_rates) / np.array(USA_inflation_rates)
            ).tolist()

            price_ratio_dict["data"][region_id] = process_rates(
                price_ratio_filtered, years_price_ratio, region["ISO3"]
            )

        if project_future_until_year:
            # convert to pandas dataframe
            inflation_rates = pd.DataFrame(
                inflation_rates_dict["data"], index=inflation_rates_dict["time"]
            ).dropna()
            # lending_rates = pd.DataFrame(
            #     lending_rates_dict["data"], index=lending_rates_dict["time"]
            # ).dropna()

            inflation_rates.index = inflation_rates.index.astype(int)
            # extend inflation rates to future
            mean_inflation_rate_since_reference_year = inflation_rates.loc[
                reference_start_year:
            ].mean(axis=0)
            inflation_rates = inflation_rates.reindex(
                range(inflation_rates.index.min(), project_future_until_year + 1)
            ).fillna(mean_inflation_rate_since_reference_year)

            inflation_rates_dict["time"] = inflation_rates.index.astype(str).tolist()
            inflation_rates_dict["data"] = inflation_rates.to_dict(orient="list")

            # lending_rates.index = lending_rates.index.astype(int)
            # extend lending rates to future
            # mean_lending_rate_since_reference_year = lending_rates.loc[
            #     reference_start_year:
            # ].mean(axis=0)
            # lending_rates = lending_rates.reindex(
            #     range(lending_rates.index.min(), project_future_until_year + 1)
            # ).fillna(mean_lending_rate_since_reference_year)

            # # convert back to dictionary
            # lending_rates_dict["time"] = lending_rates.index.astype(str).tolist()
            # lending_rates_dict["data"] = lending_rates.to_dict(orient="list")

        self.set_dict(inflation_rates_dict, name="socioeconomics/inflation_rates")
        # self.set_dict(lending_rates_dict, name="socioeconomics/lending_rates")
        self.set_dict(price_ratio_dict, name="socioeconomics/price_ratio")

    def setup_irrigation_sources(self, irrigation_sources):
        self.set_dict(irrigation_sources, name="agents/farmers/irrigation_sources")

    def setup_well_prices_by_reference_year(
        self,
        irrigation_maintenance: float,
        pump_cost: float,
        borewell_cost_1: float,
        borewell_cost_2: float,
        electricity_cost: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ):
        """
        Sets up the well prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        well_price : float
            The price of a well in the reference year.
        upkeep_price_per_m2 : float
            The upkeep price per square meter of a well in the reference year.
        reference_year : int
            The reference year for the well prices and upkeep prices.
        start_year : int
            The start year for the well prices and upkeep prices.
        end_year : int
            The end year for the well prices and upkeep prices.

        Notes
        -----
        This method sets up the well prices and upkeep prices for the hydrological model based on a reference year. It first
        retrieves the inflation rates data from the `socioeconomics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'socioeconomics/well_prices' and 'socioeconomics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["socioeconomics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "irrigation_maintenance": irrigation_maintenance,
            "pump_cost": pump_cost,
            "borewell_cost_1": borewell_cost_1,
            "borewell_cost_2": borewell_cost_2,
            "electricity_cost": electricity_cost,
        }

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for region in regions:
                prices = pd.Series(index=range(start_year, end_year + 1))
                prices.loc[reference_year] = initial_price

                # Forward calculation from the reference year
                for year in range(reference_year + 1, end_year + 1):
                    prices.loc[year] = (
                        prices[year - 1]
                        * inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year))
                        ]
                    )
                # Backward calculation from the reference year
                for year in range(reference_year - 1, start_year - 1, -1):
                    prices.loc[year] = (
                        prices[year + 1]
                        / inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year + 1))
                        ]
                    )

                prices_dict["data"][region] = prices.tolist()

            # Set the calculated prices in the appropriate dictionary
            self.set_dict(prices_dict, name=f"socioeconomics/{price_type}")

    def setup_irrigation_prices_by_reference_year(
        self,
        operation_surface: float,
        operation_sprinkler: float,
        operation_drip: float,
        capital_cost_surface: float,
        capital_cost_sprinkler: float,
        capital_cost_drip: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ):
        """
        Sets up the well prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        well_price : float
            The price of a well in the reference year.
        upkeep_price_per_m2 : float
            The upkeep price per square meter of a well in the reference year.
        reference_year : int
            The reference year for the well prices and upkeep prices.
        start_year : int
            The start year for the well prices and upkeep prices.
        end_year : int
            The end year for the well prices and upkeep prices.

        Notes
        -----
        This method sets up the well prices and upkeep prices for the hydrological model based on a reference year. It first
        retrieves the inflation rates data from the `socioeconomics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'socioeconomics/well_prices' and 'socioeconomics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["socioeconomics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "operation_cost_surface": operation_surface,
            "operation_cost_sprinkler": operation_sprinkler,
            "operation_cost_drip": operation_drip,
            "capital_cost_surface": capital_cost_surface,
            "capital_cost_sprinkler": capital_cost_sprinkler,
            "capital_cost_drip": capital_cost_drip,
        }

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for region in regions:
                prices = pd.Series(index=range(start_year, end_year + 1))
                prices.loc[reference_year] = initial_price

                # Forward calculation from the reference year
                for year in range(reference_year + 1, end_year + 1):
                    prices.loc[year] = (
                        prices[year - 1]
                        * inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year))
                        ]
                    )
                # Backward calculation from the reference year
                for year in range(reference_year - 1, start_year - 1, -1):
                    prices.loc[year] = (
                        prices[year + 1]
                        / inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year + 1))
                        ]
                    )

                prices_dict["data"][region] = prices.tolist()

            # Set the calculated prices in the appropriate dictionary
            self.set_dict(prices_dict, name=f"socioeconomics/{price_type}")

    def setup_well_prices_by_reference_year_global(
        self,
        WHY_10: float,
        WHY_20: float,
        WHY_30: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ):
        """
        Sets up the well prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        well_price : float
            The price of a well in the reference year.
        upkeep_price_per_m2 : float
            The upkeep price per square meter of a well in the reference year.
        reference_year : int
            The reference year for the well prices and upkeep prices.
        start_year : int
            The start year for the well prices and upkeep prices.
        end_year : int
            The end year for the well prices and upkeep prices.

        Notes
        -----
        This method sets up the well prices and upkeep prices for the hydrological model based on a reference year. It first
        retrieves the inflation rates data from the `socioeconomics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'socioeconomics/well_prices' and 'socioeconomics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["socioeconomics/inflation_rates"]
        price_ratio = self.dict["socioeconomics/price_ratio"]

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "why_10": WHY_10,
            "why_20": WHY_20,
            "why_30": WHY_30,
        }

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for _, region in self.geoms["regions"].iterrows():
                region_id = str(region["region_id"])

                prices = pd.Series(index=range(start_year, end_year + 1))
                price_ratio_region_year = price_ratio["data"][region_id][
                    price_ratio["time"].index(str(reference_year))
                ]

                prices.loc[reference_year] = price_ratio_region_year * initial_price

                # Forward calculation from the reference year
                for year in range(reference_year + 1, end_year + 1):
                    prices.loc[year] = (
                        prices[year - 1]
                        * inflation_rates["data"][region_id][
                            inflation_rates["time"].index(str(year))
                        ]
                    )

                # Backward calculation from the reference year
                for year in range(reference_year - 1, start_year - 1, -1):
                    prices.loc[year] = (
                        prices[year + 1]
                        / inflation_rates["data"][region_id][
                            inflation_rates["time"].index(str(year + 1))
                        ]
                    )

                prices_dict["data"][region_id] = prices.tolist()

            # Set the calculated prices in the appropriate dictionary
            self.set_dict(prices_dict, name=f"socioeconomics/{price_type}")

        electricity_rates = self.data_catalog.get_dataframe("gcam_electricity_rates")
        electricity_rates["ISO3"] = electricity_rates["Country"].map(
            SUPERWELL_NAME_TO_ISO3
        )
        electricity_rates = electricity_rates.set_index("ISO3")["Rate"].to_dict()

        electricity_rates_dict = {
            "time": list(range(start_year, end_year + 1)),
            "data": {},
        }

        for _, region in self.geoms["regions"].iterrows():
            region_id = str(region["region_id"])

            prices = pd.Series(index=range(start_year, end_year + 1))
            prices.loc[reference_year] = electricity_rates[region["ISO3"]]

            # Forward calculation from the reference year
            for year in range(reference_year + 1, end_year + 1):
                prices.loc[year] = (
                    prices[year - 1]
                    * inflation_rates["data"][region_id][
                        inflation_rates["time"].index(str(year))
                    ]
                )

            # Backward calculation from the reference year
            for year in range(reference_year - 1, start_year - 1, -1):
                prices.loc[year] = (
                    prices[year + 1]
                    / inflation_rates["data"][region_id][
                        inflation_rates["time"].index(str(year + 1))
                    ]
                )

            electricity_rates_dict["data"][region_id] = prices.tolist()

        # Set the calculated prices in the appropriate dictionary
        self.set_dict(electricity_rates_dict, name="socioeconomics/electricity_cost")

    def setup_drip_irrigation_prices_by_reference_year(
        self,
        drip_irrigation_price: float,
        reference_year: int,
        start_year: int,
        end_year: int,
    ):
        """
        Sets up the drip_irrigation prices and upkeep prices for the hydrological model based on a reference year.

        Parameters
        ----------
        drip_irrigation_price : float
            The price of a drip_irrigation in the reference year.

        reference_year : int
            The reference year for the drip_irrigation prices and upkeep prices.
        start_year : int
            The start year for the drip_irrigation prices and upkeep prices.
        end_year : int
            The end year for the drip_irrigation prices and upkeep prices.

        Notes
        -----

        The drip_irrigation prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["socioeconomics/inflation_rates"]
        regions = list(inflation_rates["data"].keys())

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "drip_irrigation_price": drip_irrigation_price,
        }

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for region in regions:
                prices = pd.Series(index=range(start_year, end_year + 1))
                prices.loc[reference_year] = initial_price

                # Forward calculation from the reference year
                for year in range(reference_year + 1, end_year + 1):
                    prices.loc[year] = (
                        prices[year - 1]
                        * inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year))
                        ]
                    )
                # Backward calculation from the reference year
                for year in range(reference_year - 1, start_year - 1, -1):
                    prices.loc[year] = (
                        prices[year + 1]
                        / inflation_rates["data"][region][
                            inflation_rates["time"].index(str(year + 1))
                        ]
                    )

                prices_dict["data"][region] = prices.tolist()

            # Set the calculated prices in the appropriate dictionary
            self.set_dict(prices_dict, name=f"socioeconomics/{price_type}")

    def setup_farmers(self, farmers):
        """
        Sets up the farmers data for GEB.

        Parameters
        ----------
        farmers : pandas.DataFrame
            A DataFrame containing the farmer data.
        irrigation_sources : dict, optional
            A dictionary mapping irrigation source names to IDs.
        n_seasons : int, optional
            The number of seasons to simulate.

        Notes
        -----
        This method sets up the farmers data for GEB. It first retrieves the region data from the
        `regions` and `subgrid` grids. It then creates a `farms` grid with the same shape as the
        `region_subgrid` grid, with a value of -1 for each cell.

        For each region, the method clips the `cultivated_land` grid to the region and creates farms for the region using
        the `create_farms` function, using these farmlands as well as the dataframe of farmer agents. The resulting farms
        whose IDs correspondd to the IDs in the farmer dataframe are added to the `farms` grid for the region.

        The method then removes any farms that are outside the study area by using the `region_mask` grid. It then remaps
        the farmer IDs to a contiguous range of integers starting from 0.

        The resulting farms data is set as agents data in the model with names of the form 'agents/farmers/farms'. The
        crop names are mapped to IDs using the `crop_name_to_id` dictionary that was previously created. The resulting
        crop IDs are stored in the `season_#_crop` columns of the `farmers` DataFrame.

        If `irrigation_sources` is provided, the method sets the `irrigation_source` column of the `farmers` DataFrame to
        the corresponding IDs.

        Finally, the method sets the array data for each column of the `farmers` DataFrame as agents data in the model
        with names of the form 'agents/farmers/{column}'.
        """
        regions = self.geoms["regions"]
        region_ids = self.region_subgrid["region_ids"]
        full_region_cultivated_land = self.region_subgrid[
            "landsurface/full_region_cultivated_land"
        ]

        farms = self.full_like(region_ids, fill_value=-1, nodata=-1)
        for region_id in regions["region_id"]:
            self.logger.info(f"Creating farms for region {region_id}")
            region = region_ids == region_id
            region_clip, bounds = clip_with_grid(region, region)

            cultivated_land_region = full_region_cultivated_land.isel(bounds)
            cultivated_land_region = xr.where(
                region_clip, cultivated_land_region, 0, keep_attrs=True
            )
            # TODO: Why does nodata value disappear?
            # self.dict['areamaps/region_id_mapping'][farmers['region_id']]
            farmer_region_ids = farmers["region_id"]
            farmers_region = farmers[farmer_region_ids == region_id]
            farms_region = create_farms(
                farmers_region, cultivated_land_region, farm_size_key="area_n_cells"
            )
            assert (
                farms_region.min() >= -1
            )  # -1 is nodata value, all farms should be positive
            farms[bounds] = xr.where(
                region_clip, farms_region, farms.isel(bounds), keep_attrs=True
            )
            # farms = farms.compute()  # perhaps this helps with memory issues?

        farmers = farmers.drop("area_n_cells", axis=1)

        cut_farms = np.unique(
            xr.where(
                self.region_subgrid["mask"],
                farms.copy().values,
                -1,
                keep_attrs=True,
            )
        )
        cut_farm_indices = cut_farms[cut_farms != -1]

        assert farms.min() >= -1  # -1 is nodata value, all farms should be positive
        subgrid_farms = farms.raster.clip_bbox(self.subgrid["mask"].raster.bounds)

        subgrid_farms_in_study_area = xr.where(
            np.isin(subgrid_farms, cut_farm_indices), -1, subgrid_farms, keep_attrs=True
        )
        farmers = farmers[~farmers.index.isin(cut_farm_indices)]

        remap_farmer_ids = np.full(
            farmers.index.max() + 2, -1, dtype=np.int32
        )  # +1 because 0 is also a farm, +1 because no farm is -1, set to -1 in next step
        remap_farmer_ids[farmers.index] = np.arange(len(farmers))
        subgrid_farms_in_study_area = remap_farmer_ids[
            subgrid_farms_in_study_area.values
        ]

        farmers = farmers.reset_index(drop=True)

        assert np.setdiff1d(np.unique(subgrid_farms_in_study_area), -1).size == len(
            farmers
        )
        assert farmers.iloc[-1].name == subgrid_farms_in_study_area.max()

        subgrid_farms_in_study_area_ = self.full_like(
            self.subgrid["mask"],
            fill_value=-1,
            nodata=-1,
            dtype=np.int32,
        )
        subgrid_farms_in_study_area_[:] = subgrid_farms_in_study_area

        self.set_subgrid(subgrid_farms_in_study_area_, name="agents/farmers/farms")
        self.set_array(farmers.index.values, name="agents/farmers/id")
        self.set_array(farmers["region_id"].values, name="agents/farmers/region_id")

    def setup_farmers_from_csv(self, path=None):
        """
        Sets up the farmers data for GEB from a CSV file.

        Parameters
        ----------
        path : str
            The path to the CSV file containing the farmer data.

        Notes
        -----
        This method sets up the farmers data for GEB from a CSV file. It first reads the farmer data from
        the CSV file using the `pandas.read_csv` method.

        See the `setup_farmers` method for more information on how the farmer data is set up in the model.
        """
        if path is None:
            path = self.preprocessing_dir / "agents" / "farmers" / "farmers.csv"
        farmers = pd.read_csv(path, index_col=0)
        self.setup_farmers(farmers)

    def setup_create_farms(
        self,
        region_id_column="region_id",
        country_iso3_column="ISO3",
        farm_size_donor_countries=None,
        data_source="lowder",
        size_class_boundaries=None,
    ):
        """
        Sets up the farmers for GEB.

        Parameters
        ----------
        irrigation_sources : dict
            A dictionary of irrigation sources and their corresponding water availability in m^3/day.
        region_id_column : str, optional
            The name of the column in the region database that contains the region IDs. Default is 'UID'.
        country_iso3_column : str, optional
            The name of the column in the region database that contains the country ISO3 codes. Default is 'ISO3'.
        risk_aversion_mean : float, optional
            The mean of the normal distribution from which the risk aversion values are sampled. Default is 1.5.
        risk_aversion_standard_deviation : float, optional
            The standard deviation of the normal distribution from which the risk aversion values are sampled. Default is 0.5.

        Notes
        -----
        This method sets up the farmers for GEB. This is a simplified method that generates an example set of agent data.
        It first calculates the number of farmers and their farm sizes for each region based on the agricultural data for
        that region based on theamount of farm land and data from a global database on farm sizes per country. It then
        randomly assigns crops, irrigation sources, household sizes, and daily incomes and consumption levels to each farmer.

        A paper that reports risk aversion values for 75 countries is this one: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2646134
        """
        if data_source == "lowder":
            size_class_boundaries = {
                "< 1 Ha": (0, 10000),
                "1 - 2 Ha": (10000, 20000),
                "2 - 5 Ha": (20000, 50000),
                "5 - 10 Ha": (50000, 100000),
                "10 - 20 Ha": (100000, 200000),
                "20 - 50 Ha": (200000, 500000),
                "50 - 100 Ha": (500000, 1000000),
                "100 - 200 Ha": (1000000, 2000000),
                "200 - 500 Ha": (2000000, 5000000),
                "500 - 1000 Ha": (5000000, 10000000),
                "> 1000 Ha": (10000000, np.inf),
            }
        else:
            assert size_class_boundaries is not None
            assert farm_size_donor_countries is None, (
                "farm_size_donor_countries is only used for lowder data"
            )

        cultivated_land = self.region_subgrid["landsurface/full_region_cultivated_land"]
        assert cultivated_land.dtype == bool, "Cultivated land must be boolean"
        region_ids = self.region_subgrid["region_ids"]
        cell_area = self.region_subgrid["cell_area"]

        regions_shapes = self.geoms["regions"]
        if data_source == "lowder":
            assert country_iso3_column in regions_shapes.columns, (
                f"Region database must contain {country_iso3_column} column ({self.data_catalog['GADM_level1'].path})"
            )

            farm_sizes_per_region = (
                self.data_catalog.get_dataframe("lowder_farm_sizes")
                .dropna(subset=["Total"], axis=0)
                .drop(["empty", "income class"], axis=1)
            )
            farm_sizes_per_region["Country"] = farm_sizes_per_region["Country"].ffill()
            # Remove preceding and trailing white space from country names
            farm_sizes_per_region["Country"] = farm_sizes_per_region[
                "Country"
            ].str.strip()
            farm_sizes_per_region["Census Year"] = farm_sizes_per_region[
                "Country"
            ].ffill()

            farm_sizes_per_region["ISO3"] = farm_sizes_per_region["Country"].map(
                COUNTRY_NAME_TO_ISO3
            )
            assert not farm_sizes_per_region["ISO3"].isna().any(), (
                f"Found {farm_sizes_per_region['ISO3'].isna().sum()} countries without ISO3 code"
            )
        else:
            # load data source
            farm_sizes_per_region = pd.read_excel(
                data_source["farm_size"], index_col=(0, 1, 2)
            )
            n_farms_per_region = pd.read_excel(
                data_source["n_farms"],
                index_col=(0, 1, 2),
            )

        all_agents = []
        self.logger.info(f"Starting processing of {len(regions_shapes)} regions")
        for _, region in regions_shapes.iterrows():
            UID = region[region_id_column]
            if data_source == "lowder":
                country_ISO3 = region[country_iso3_column]
                if farm_size_donor_countries:
                    assert isinstance(farm_size_donor_countries, dict)
                    country_ISO3 = farm_size_donor_countries.get(
                        country_ISO3, country_ISO3
                    )
            else:
                state, district, tehsil = (
                    region["state_name"],
                    region["district_n"],
                    region["sub_dist_1"],
                )

            self.logger.info(f"Processing region {UID}")

            cultivated_land_region_total_cells = (
                ((region_ids == UID) & (cultivated_land)).sum().compute()
            )
            total_cultivated_land_area_lu = (
                (((region_ids == UID) & (cultivated_land)) * cell_area).sum().compute()
            )
            if (
                total_cultivated_land_area_lu == 0
            ):  # when no agricultural area, just continue as there will be no farmers. Also avoiding some division by 0 errors.
                continue

            average_cell_area_region = (
                cell_area.where(((region_ids == UID) & (cultivated_land)))
                .mean()
                .compute()
            )

            if data_source == "lowder":
                region_farm_sizes = farm_sizes_per_region.loc[
                    (farm_sizes_per_region["ISO3"] == country_ISO3)
                ].drop(["Country", "Census Year", "Total"], axis=1)
                assert len(region_farm_sizes) == 2, (
                    f"Found {len(region_farm_sizes) / 2} region_farm_sizes for {country_ISO3}"
                )

                # Extract holdings and agricultural area data
                region_n_holdings = (
                    region_farm_sizes.loc[
                        region_farm_sizes["Holdings/ agricultural area"] == "Holdings"
                    ]
                    .iloc[0]
                    .drop(["Holdings/ agricultural area", "ISO3"])
                    .replace("..", np.nan)
                    .astype(float)
                )
                agricultural_area_db_ha = (
                    region_farm_sizes.loc[
                        region_farm_sizes["Holdings/ agricultural area"]
                        == "Agricultural area (Ha) "
                    ]
                    .iloc[0]
                    .drop(["Holdings/ agricultural area", "ISO3"])
                    .replace("..", np.nan)
                    .astype(float)
                )

                # Calculate average sizes for each bin
                average_sizes = {}
                for bin_name in agricultural_area_db_ha.index:
                    bin_name = bin_name.strip()
                    if bin_name.startswith("<"):
                        # For '< 1 Ha', average is 0.5 Ha
                        average_size = 0.5
                    elif bin_name.startswith(">"):
                        # For '> 1000 Ha', assume average is 1500 Ha
                        average_size = 1500
                    else:
                        # For ranges like '5 - 10 Ha', calculate the midpoint
                        try:
                            min_size, max_size = bin_name.replace("Ha", "").split("-")
                            min_size = float(min_size.strip())
                            max_size = float(max_size.strip())
                            average_size = (min_size + max_size) / 2
                        except ValueError:
                            # Default average size if parsing fails
                            average_size = 1
                    average_sizes[bin_name] = average_size

                # Convert average sizes to a pandas Series
                average_sizes_series = pd.Series(average_sizes)

                # Handle cases where entries are zero or missing
                agricultural_area_db_ha_zero_or_nan = (
                    agricultural_area_db_ha.isnull() | (agricultural_area_db_ha == 0)
                )
                region_n_holdings_zero_or_nan = region_n_holdings.isnull() | (
                    region_n_holdings == 0
                )

                if agricultural_area_db_ha_zero_or_nan.all():
                    # All entries in agricultural_area_db_ha are zero or NaN
                    if not region_n_holdings_zero_or_nan.all():
                        # Calculate agricultural_area_db_ha using average sizes and region_n_holdings
                        region_n_holdings = region_n_holdings.fillna(1).replace(0, 1)
                        agricultural_area_db_ha = (
                            average_sizes_series * region_n_holdings
                        )
                    else:
                        raise ValueError(
                            "Cannot calculate agricultural_area_db_ha: both datasets are zero or missing."
                        )
                elif region_n_holdings_zero_or_nan.all():
                    # All entries in region_n_holdings are zero or NaN
                    if not agricultural_area_db_ha_zero_or_nan.all():
                        # Calculate region_n_holdings using agricultural_area_db_ha and average sizes
                        agricultural_area_db_ha = agricultural_area_db_ha.fillna(
                            1
                        ).replace(0, 1)
                        region_n_holdings = (
                            agricultural_area_db_ha / average_sizes_series
                        )
                    else:
                        raise ValueError(
                            "Cannot calculate region_n_holdings: both datasets are zero or missing."
                        )
                else:
                    # Replace zeros and NaNs in both datasets to avoid division by zero
                    region_n_holdings = region_n_holdings.fillna(1).replace(0, 1)
                    agricultural_area_db_ha = agricultural_area_db_ha.fillna(1).replace(
                        0, 1
                    )

                # Calculate total agricultural area in square meters
                agricultural_area_db = (
                    agricultural_area_db_ha * 10000
                )  # Convert Ha to m^2

                # Calculate region farm sizes
                region_farm_sizes = agricultural_area_db / region_n_holdings

            else:
                region_farm_sizes = farm_sizes_per_region.loc[(state, district, tehsil)]
                region_n_holdings = n_farms_per_region.loc[(state, district, tehsil)]
                agricultural_area_db = region_farm_sizes * region_n_holdings

            total_cultivated_land_area_db = agricultural_area_db.sum()

            n_cells_per_size_class = pd.Series(0, index=region_n_holdings.index)

            for size_class in agricultural_area_db.index:
                if (
                    region_n_holdings[size_class] > 0
                ):  # if no holdings, no need to calculate
                    region_n_holdings[size_class] = region_n_holdings[size_class] * (
                        total_cultivated_land_area_lu / total_cultivated_land_area_db
                    )
                    n_cells_per_size_class.loc[size_class] = (
                        region_n_holdings[size_class]
                        * region_farm_sizes[size_class]
                        / average_cell_area_region
                    )
                    assert not np.isnan(n_cells_per_size_class.loc[size_class])

            assert math.isclose(
                cultivated_land_region_total_cells,
                round(n_cells_per_size_class.sum().item()),
            )

            whole_cells_per_size_class = (n_cells_per_size_class // 1).astype(int)
            leftover_cells_per_size_class = n_cells_per_size_class % 1
            whole_cells = whole_cells_per_size_class.sum()
            n_missing_cells = cultivated_land_region_total_cells - whole_cells
            assert n_missing_cells <= len(agricultural_area_db)

            index = list(
                zip(
                    leftover_cells_per_size_class.index,
                    leftover_cells_per_size_class % 1,
                )
            )
            n_cells_to_add = sorted(index, key=lambda x: x[1], reverse=True)[
                : n_missing_cells.compute().item()
            ]
            whole_cells_per_size_class.loc[[p[0] for p in n_cells_to_add]] += 1

            region_agents = []
            for size_class in whole_cells_per_size_class.index:
                # if no cells for this size class, just continue
                if whole_cells_per_size_class.loc[size_class] == 0:
                    continue

                number_of_agents_size_class = round(
                    region_n_holdings[size_class].item()
                )
                # if there is agricultural land, but there are no agents rounded down, we assume there is one agent
                if (
                    number_of_agents_size_class == 0
                    and whole_cells_per_size_class[size_class] > 0
                ):
                    number_of_agents_size_class = 1

                min_size_m2, max_size_m2 = size_class_boundaries[size_class]
                if max_size_m2 in (np.inf, "inf", "infinity", "Infinity"):
                    max_size_m2 = region_farm_sizes[size_class] * 2

                min_size_cells = int(min_size_m2 / average_cell_area_region)
                min_size_cells = max(
                    min_size_cells, 1
                )  # farm can never be smaller than one cell
                max_size_cells = (
                    int(max_size_m2 / average_cell_area_region) - 1
                )  # otherwise they overlap with next size class
                mean_cells_per_agent = int(
                    region_farm_sizes[size_class] / average_cell_area_region
                )

                if (
                    mean_cells_per_agent < min_size_cells
                    or mean_cells_per_agent > max_size_cells
                ):  # there must be an error in the data, thus assume centred
                    mean_cells_per_agent = (min_size_cells + max_size_cells) // 2

                population = pd.DataFrame(index=range(number_of_agents_size_class))

                offset = (
                    whole_cells_per_size_class[size_class]
                    - number_of_agents_size_class * mean_cells_per_agent
                )

                if (
                    number_of_agents_size_class * mean_cells_per_agent + offset
                    < min_size_cells * number_of_agents_size_class
                ):
                    min_size_cells = (
                        number_of_agents_size_class * mean_cells_per_agent + offset
                    ) // number_of_agents_size_class
                if (
                    number_of_agents_size_class * mean_cells_per_agent + offset
                    > max_size_cells * number_of_agents_size_class
                ):
                    max_size_cells = (
                        number_of_agents_size_class * mean_cells_per_agent + offset
                    ) // number_of_agents_size_class + 1

                n_farms_size_class, farm_sizes_size_class = get_farm_distribution(
                    number_of_agents_size_class,
                    min_size_cells,
                    max_size_cells,
                    mean_cells_per_agent,
                    offset,
                    self.logger,
                )
                assert n_farms_size_class.sum() == number_of_agents_size_class
                assert (farm_sizes_size_class > 0).all()
                assert (
                    n_farms_size_class * farm_sizes_size_class
                ).sum() == whole_cells_per_size_class[size_class]
                farm_sizes = farm_sizes_size_class.repeat(n_farms_size_class)
                np.random.shuffle(farm_sizes)
                population["area_n_cells"] = farm_sizes
                region_agents.append(population)

                assert (
                    population["area_n_cells"].sum()
                    == whole_cells_per_size_class[size_class]
                )

            region_agents = pd.concat(region_agents, ignore_index=True)
            region_agents["region_id"] = UID
            all_agents.append(region_agents)

        farmers = pd.concat(all_agents, ignore_index=True)
        self.setup_farmers(farmers)

    def setup_household_characteristics(self, maximum_age=85, skip_countries_ISO3=[]):
        # load GDL region within model domain
        GDL_regions = self.data_catalog.get_geodataframe(
            "GDL_regions_v4", geom=self.region, variables=["GDLcode", "iso_code"]
        )
        # create list of attibutes to include (and include name to store to)
        attributes_to_include = {
            "HHSIZE_CAT": "household_type",
            "AGE": "age_household_head",
            "EDUC": "education_level",
            "WEALTH_INDEX": "wealth_index",
            "RURAL": "rural",
        }
        region_results = {}

        # get age class to age (head of household) mapping
        age_class_to_age = {
            1: (0, 4),
            2: (5, 14),
            3: (15, 24),
            4: (25, 34),
            5: (35, 44),
            6: (45, 54),
            7: (55, 64),
            8: (66, maximum_age + 1),
        }

        # iterate over regions and sample agents from GLOPOP-S
        for _, GDL_region in GDL_regions.iterrows():
            GDL_code = GDL_region["GDLcode"]
            if GDL_region["iso_code"] in skip_countries_ISO3:
                self.logger.info(
                    f"Skipping setting up household characteristics for {GDL_region['iso_code']}"
                )
                continue
            self.logger.info(
                f"Setting up household characteristics for {GDL_region['iso_code']}"
            )
            region_results[GDL_code] = {}
            GLOPOP_S_region, GLOPOP_GRID_region = load_GLOPOP_S(
                self.data_catalog, GDL_code
            )

            # clip grid to model bounds
            GLOPOP_GRID_region = GLOPOP_GRID_region.rio.clip_box(*self.bounds)

            # get unique cells in grid
            unique_grid_cells = np.unique(GLOPOP_GRID_region.values)

            # subset GLOPOP_households_region to unique cells for quicker search
            GLOPOP_S_region = GLOPOP_S_region[
                GLOPOP_S_region["GRID_CELL"].isin(unique_grid_cells)
            ]

            # create column WEALTH_INDEX (GLOPOP-S contains either INCOME or WEALTH data, depending on the region. Therefor we combine these.)
            GLOPOP_S_region["WEALTH_INDEX"] = (
                GLOPOP_S_region["WEALTH"] + GLOPOP_S_region["INCOME"] + 1
            )

            # create all households
            GLOPOP_households_region = np.unique(GLOPOP_S_region["HID"])
            n_households = GLOPOP_households_region.size

            # iterate over unique housholds and extract the variables we want
            household_characteristics = {}
            household_characteristics["sizes"] = np.full(
                n_households, -1, dtype=np.int32
            )
            household_characteristics["locations"] = np.full(
                (n_households, 2), -1, dtype=np.float32
            )
            for column in attributes_to_include:
                household_characteristics[column] = np.full(
                    n_households, -1, dtype=np.int32
                )

            # initiate indice tracker
            households_found = 0

            for i, HID in enumerate(GLOPOP_households_region):
                if not households_found % 1000:
                    print(f"searching household {households_found} of {n_households}")
                household = GLOPOP_S_region[GLOPOP_S_region["HID"] == HID]
                household_size = len(household)
                if len(household) > 1:
                    # if there are multiple people in the household
                    # take head household
                    household = household[household["RELATE_HEAD"] == 1]

                GRID_CELL = int(household["GRID_CELL"])
                if GRID_CELL in GLOPOP_GRID_region.values:
                    for column in attributes_to_include:
                        household_characteristics[column][households_found] = household[
                            column
                        ]
                        household_characteristics["sizes"][households_found] = (
                            household_size
                        )

                    # now find location of household
                    idx_household = np.where(GLOPOP_GRID_region.values[0] == GRID_CELL)
                    # get x and y from xarray
                    x_y = np.concatenate(
                        [
                            GLOPOP_GRID_region.x.values[idx_household[1]],
                            GLOPOP_GRID_region.y.values[idx_household[0]],
                        ]
                    )
                    household_characteristics["locations"][households_found, :] = x_y
                    households_found += 1

            print(f"searching household {households_found} of {n_households}")

            # clip away unused data:
            for household_attribute in household_characteristics:
                household_characteristics[household_attribute] = (
                    household_characteristics[household_attribute][:households_found]
                )
                for column in attributes_to_include:
                    if column == "AGE":
                        age_range = age_class_to_age[household[column].values[0]]
                        age_household_head = np.random.randint(
                            age_range[0], age_range[1]
                        )
                        household_characteristics[column][i] = age_household_head
                    else:
                        household_characteristics[column][i] = household[column]
                        household_characteristics["sizes"][i] = household_size

                # now find location of household
                idx_household = np.where(GLOPOP_GRID_region.values[0] == GRID_CELL)
                # get x and y from xarray
                x_y = np.concatenate(
                    [
                        GLOPOP_GRID_region.x.values[idx_household[1]],
                        GLOPOP_GRID_region.y.values[idx_household[0]],
                    ]
                )
                household_characteristics["locations"][i, :] = x_y

            region_results[GDL_code] = household_characteristics

        # concatenate all data
        data_concatenated = {}
        for household_attribute in household_characteristics:
            data_concatenated[household_attribute] = np.concatenate(
                [
                    region_results[GDL_code][household_attribute]
                    for GDL_code in region_results
                ]
            )

            # and store to binary
            if household_attribute in attributes_to_include:
                self.set_array(
                    data_concatenated[household_attribute],
                    name=f"agents/households/{attributes_to_include[household_attribute]}",
                )
            else:
                self.set_array(
                    data_concatenated[household_attribute],
                    name=f"agents/households/{household_attribute}",
                )

    def setup_farmer_household_characteristics(self, maximum_age=85):
        n_farmers = self.array["agents/farmers/id"].size
        farms = self.subgrid["agents/farmers/farms"]

        # get farmer locations
        vertical_index = (
            np.arange(farms.shape[0])
            .repeat(farms.shape[1])
            .reshape(farms.shape)[farms != -1]
        )
        horizontal_index = np.tile(np.arange(farms.shape[1]), farms.shape[0]).reshape(
            farms.shape
        )[farms != -1]
        farms_flattened = farms.values[farms.values != -1]

        pixels = np.zeros((n_farmers, 2), dtype=np.int32)
        pixels[:, 0] = np.round(
            np.bincount(farms_flattened, horizontal_index)
            / np.bincount(farms_flattened)
        ).astype(int)
        pixels[:, 1] = np.round(
            np.bincount(farms_flattened, vertical_index) / np.bincount(farms_flattened)
        ).astype(int)

        locations = pixels_to_coords(
            pixels + 0.5, farms.rio.transform(recalc=True).to_gdal()
        )
        locations = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(locations[:, 0], locations[:, 1]),
            crs="EPSG:4326",
        )  # convert locations to geodataframe

        # GLOPOP-S uses the GDL regions. So we need to get the GDL region for each farmer using their location
        GDL_regions = self.data_catalog.get_geodataframe(
            "GDL_regions_v4", geom=self.region, variables=["GDLcode"]
        )
        GDL_region_per_farmer = gpd.sjoin(
            locations, GDL_regions, how="left", predicate="within"
        )

        # ensure that each farmer has a region
        assert GDL_region_per_farmer["GDLcode"].notna().all()

        # Get list of unique GDL codes from farmer dataframe
        attributes_to_include = ["HHSIZE_CAT", "AGE", "EDUC", "WEALTH"]

        for column in attributes_to_include:
            GDL_region_per_farmer[column] = np.full(
                len(GDL_region_per_farmer), -1, dtype=np.int32
            )

        for GDL_region, farmers_GDL_region in GDL_region_per_farmer.groupby("GDLcode"):
            GLOPOP_S_region, _ = load_GLOPOP_S(self.data_catalog, GDL_region)

            # select farmers only
            GLOPOP_S_region = GLOPOP_S_region[GLOPOP_S_region["RURAL"] == 1].drop(
                "RURAL", axis=1
            )

            # shuffle GLOPOP-S data to avoid biases in that regard
            GLOPOP_S_household_IDs = GLOPOP_S_region["HID"].unique()
            np.random.shuffle(GLOPOP_S_household_IDs)  # shuffle array in-place
            GLOPOP_S_region = (
                GLOPOP_S_region.set_index("HID")
                .loc[GLOPOP_S_household_IDs]
                .reset_index()
            )

            # Select a sample of farmers from the database. Because the households were
            # shuflled there is no need to pick random households, we can just take the first n_farmers.
            # If there are not enough farmers in the region, we need to upsample the data. In this case
            # we will just take the same farmers multiple times starting from the top.
            GLOPOP_S_household_IDs = GLOPOP_S_region["HID"].values

            # first we mask out all consecutive duplicates
            mask = np.concatenate(
                ([True], GLOPOP_S_household_IDs[1:] != GLOPOP_S_household_IDs[:-1])
            )
            GLOPOP_S_household_IDs = GLOPOP_S_household_IDs[mask]

            GLOPOP_S_region_sampled = []
            if GLOPOP_S_household_IDs.size < len(farmers_GDL_region):
                n_repetitions = len(farmers_GDL_region) // GLOPOP_S_household_IDs.size
                max_household_ID = GLOPOP_S_household_IDs.max()
                for i in range(n_repetitions):
                    GLOPOP_S_region_copy = GLOPOP_S_region.copy()
                    # increase the household ID to avoid duplicate household IDs. Using (i + 1) so that the original household IDs are not changed
                    # so that they can be used in the final "topping up" below.
                    GLOPOP_S_region_copy["HID"] = GLOPOP_S_region_copy["HID"] + (
                        (i + 1) * max_household_ID
                    )
                    GLOPOP_S_region_sampled.append(GLOPOP_S_region_copy)
                requested_farmers = (
                    len(farmers_GDL_region) % GLOPOP_S_household_IDs.size
                )
            else:
                requested_farmers = len(farmers_GDL_region)

            GLOPOP_S_household_IDs = GLOPOP_S_household_IDs[:requested_farmers]
            GLOPOP_S_region_sampled.append(
                GLOPOP_S_region[GLOPOP_S_region["HID"].isin(GLOPOP_S_household_IDs)]
            )

            GLOPOP_S_region_sampled = pd.concat(
                GLOPOP_S_region_sampled, ignore_index=True
            )
            assert GLOPOP_S_region_sampled["HID"].unique().size == len(
                farmers_GDL_region
            )

            households_region = GLOPOP_S_region_sampled.groupby("HID")
            # select only household heads
            household_heads = households_region.apply(
                lambda x: x[x["RELATE_HEAD"] == 1]
            )
            assert len(household_heads) == len(farmers_GDL_region)

            # # age
            # household_heads["AGE_continuous"] = np.full(
            #     len(household_heads), -1, dtype=np.int32
            # )
            age_class_to_age = {
                1: (0, 16),
                2: (16, 26),
                3: (26, 36),
                4: (36, 46),
                5: (46, 56),
                6: (56, 66),
                7: (66, maximum_age + 1),
            }  # exclusive
            for age_class, age_range in age_class_to_age.items():
                household_heads_age_class = household_heads[
                    household_heads["AGE"] == age_class
                ]
                household_heads.loc[household_heads_age_class.index, "AGE"] = (
                    np.random.randint(
                        age_range[0],
                        age_range[1],
                        size=len(household_heads_age_class),
                        dtype=GDL_region_per_farmer["AGE"].dtype,
                    )
                )
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "AGE"] = (
                household_heads["AGE"].values
            )

            # education level
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "EDUC"] = (
                household_heads["EDUC"].values
            )

            # household size
            household_sizes_region = households_region.size().values.astype(np.int32)
            GDL_region_per_farmer.loc[farmers_GDL_region.index, "HHSIZE_CAT"] = (
                household_sizes_region
            )

        # assert none of the household sizes are placeholder value -1
        assert (GDL_region_per_farmer["HHSIZE_CAT"] != -1).all()
        assert (GDL_region_per_farmer["AGE"] != -1).all()
        assert (GDL_region_per_farmer["EDUC"] != -1).all()

        self.set_array(
            GDL_region_per_farmer["HHSIZE_CAT"].values,
            name="agents/farmers/household_size",
        )
        self.set_array(
            GDL_region_per_farmer["AGE"].values,
            name="agents/farmers/age_household_head",
        )
        self.set_array(
            GDL_region_per_farmer["EDUC"].values,
            name="agents/farmers/education_level",
        )

    def create_preferences(self):
        # Risk aversion
        preferences_country_level = self.data_catalog.get_dataframe(
            "preferences_country",
            variables=["country", "isocode", "patience", "risktaking"],
        ).dropna()

        preferences_individual_level = self.data_catalog.get_dataframe(
            "preferences_individual",
            variables=["country", "isocode", "patience", "risktaking"],
        ).dropna()

        def scale_to_range(x, new_min, new_max):
            x_min = x.min()
            x_max = x.max()
            # Avoid division by zero
            if x_max == x_min:
                return pd.Series(new_min, index=x.index)
            scaled_x = (x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min
            return scaled_x

        # Create 'risktaking_losses'
        preferences_individual_level["risktaking_losses"] = scale_to_range(
            preferences_individual_level["risktaking"] * -1,
            new_min=-0.88,
            new_max=-0.02,
        )

        # Create 'risktaking_gains'
        preferences_individual_level["risktaking_gains"] = scale_to_range(
            preferences_individual_level["risktaking"] * -1,
            new_min=0.04,
            new_max=0.94,
        )

        # Create 'discount'
        preferences_individual_level["discount"] = scale_to_range(
            preferences_individual_level["patience"] * -1,
            new_min=0.15,
            new_max=0.73,
        )

        # Create 'risktaking_losses'
        preferences_country_level["risktaking_losses"] = scale_to_range(
            preferences_country_level["risktaking"] * -1,
            new_min=-0.88,
            new_max=-0.02,
        )

        # Create 'risktaking_gains'
        preferences_country_level["risktaking_gains"] = scale_to_range(
            preferences_country_level["risktaking"] * -1,
            new_min=0.04,
            new_max=0.94,
        )

        # Create 'discount'
        preferences_country_level["discount"] = scale_to_range(
            preferences_country_level["patience"] * -1,
            new_min=0.15,
            new_max=0.73,
        )

        # List of variables for which to calculate the standard deviation
        variables = ["discount", "risktaking_gains", "risktaking_losses"]

        # Convert the variables to numeric, coercing errors to NaN to handle non-numeric entries
        for var in variables:
            preferences_individual_level[var] = pd.to_numeric(
                preferences_individual_level[var], errors="coerce"
            )

        # Group by 'isocode' and calculate the standard deviation for each variable
        std_devs = preferences_individual_level.groupby("isocode")[variables].std()

        # Add a suffix '_std' to the column names to indicate standard deviation
        std_devs = std_devs.add_suffix("_std").reset_index()

        # Merge the standard deviation data into 'preferences_country_level' on 'isocode'
        preferences_country_level = preferences_country_level.merge(
            std_devs, on="isocode", how="left"
        )

        preferences_country_level = preferences_country_level.drop(
            ["patience", "risktaking"], axis=1
        )

        return preferences_country_level

    def setup_farmer_characteristics(
        self,
        interest_rate=0.05,
    ):
        n_farmers = self.array["agents/farmers/id"].size

        preferences_global = self.create_preferences()
        preferences_global.rename(
            columns={
                "country": "Country",
                "isocode": "ISO3",
                "risktaking_losses": "Losses",
                "risktaking_gains": "Gains",
                "discount": "Discount",
                "discount_std": "Discount_std",
                "risktaking_losses_std": "Losses_std",
                "risktaking_gains_std": "Gains_std",
            },
            inplace=True,
        )

        GLOBIOM_regions = self.data_catalog.get_dataframe("GLOBIOM_regions_37")
        GLOBIOM_regions["ISO3"] = GLOBIOM_regions["Country"].map(GLOBIOM_NAME_TO_ISO3)
        # For my personal branch
        GLOBIOM_regions.loc[GLOBIOM_regions["Country"] == "Switzerland", "Region37"] = (
            "EU_MidWest"
        )
        assert not np.any(GLOBIOM_regions["ISO3"].isna()), "Missing ISO3 codes"

        ISO3_codes_region = self.geoms["regions"]["ISO3"].unique()
        GLOBIOM_regions_region = GLOBIOM_regions[
            GLOBIOM_regions["ISO3"].isin(ISO3_codes_region)
        ]["Region37"].unique()
        ISO3_codes_GLOBIOM_region = GLOBIOM_regions[
            GLOBIOM_regions["Region37"].isin(GLOBIOM_regions_region)
        ]["ISO3"]

        donor_data = {}
        for ISO3 in ISO3_codes_GLOBIOM_region:
            region_risk_aversion_data = preferences_global[
                preferences_global["ISO3"] == ISO3
            ]

            region_risk_aversion_data = region_risk_aversion_data[
                [
                    "Country",
                    "ISO3",
                    "Gains",
                    "Losses",
                    "Gains_std",
                    "Losses_std",
                    "Discount",
                    "Discount_std",
                ]
            ]
            region_risk_aversion_data.reset_index(drop=True, inplace=True)
            # Store pivoted data in dictionary with region_id as key
            donor_data[ISO3] = region_risk_aversion_data

        # Concatenate all regional data into a single DataFrame with MultiIndex
        donor_data = pd.concat(donor_data, names=["ISO3"])

        # Drop crops with no data at all for these regions
        donor_data = donor_data.dropna(axis=1, how="all")

        unique_regions = self.geoms["regions"]

        data = self.donate_and_receive_crop_prices(
            donor_data, unique_regions, GLOBIOM_regions
        )

        # Map to corresponding region
        data_reset = data.reset_index(level="region_id")
        data = data_reset.set_index("region_id")
        region_ids = self.array["agents/farmers/region_id"]

        # Set gains and losses
        gains_array = pd.Series(region_ids).map(data["Gains"]).to_numpy()
        gains_std = pd.Series(region_ids).map(data["Gains_std"]).to_numpy()
        losses_array = pd.Series(region_ids).map(data["Losses"]).to_numpy()
        losses_std = pd.Series(region_ids).map(data["Losses_std"]).to_numpy()
        discount_array = pd.Series(region_ids).map(data["Discount"]).to_numpy()
        discount_std = pd.Series(region_ids).map(data["Discount_std"]).to_numpy()

        education_levels = self.array["agents/farmers/education_level"]
        age = self.array["agents/farmers/age_household_head"]

        def normalize(array):
            return (array - np.min(array)) / (np.max(array) - np.min(array))

        combined_deviation_risk_aversion = ((2 / 6) * normalize(education_levels)) + (
            (4 / 6) * normalize(age)
        )

        # Generate random noise, positively correlated with education levels and age
        # (Higher age / education level means more risk averse)
        z = np.random.normal(
            loc=0, scale=1, size=combined_deviation_risk_aversion.shape
        )

        # Initial gains_variation proportional to risk aversion
        gains_variation = combined_deviation_risk_aversion * z

        current_std = np.std(gains_variation)
        gains_variation = gains_variation * (gains_std / current_std)

        # Similarly for losses_variation
        losses_variation = combined_deviation_risk_aversion * z
        current_std_losses = np.std(losses_variation)
        losses_variation = losses_variation * (losses_std / current_std_losses)

        # Add the generated noise to the original gains and losses arrays
        gains_array_with_variation = np.clip(gains_array + gains_variation, -0.99, 0.99)
        losses_array_with_variation = np.clip(
            losses_array + losses_variation, -0.99, 0.99
        )

        # Calculate intention factor based on age and education
        # Intention factor scales negatively with age and positively with education level
        intention_factor = normalize(education_levels) - normalize(age)

        # Adjust the intention factor to center it around a mean of 0.3
        # The total intention of age, education and neighbor effects can scale to 1
        intention_factor = intention_factor * 0.333 + 0.333

        neutral_risk_aversion = np.mean(
            [gains_array_with_variation, losses_array_with_variation], axis=0
        )

        self.set_array(neutral_risk_aversion, name="agents/farmers/risk_aversion")
        self.set_array(
            gains_array_with_variation, name="agents/farmers/risk_aversion_gains"
        )
        self.set_array(
            losses_array_with_variation, name="agents/farmers/risk_aversion_losses"
        )
        self.set_array(intention_factor, name="agents/farmers/intention_factor")

        # discount rate
        discount_rate_variation_factor = ((2 / 6) * normalize(education_levels)) + (
            (4 / 6) * normalize(age)
        )
        discount_rate_variation = discount_rate_variation_factor * z * -1

        # Adjust the variations to have the desired overall standard deviation
        current_std = np.std(discount_rate_variation)
        discount_rate_variation = discount_rate_variation * (discount_std / current_std)

        # Apply the variations to the original discount rates and clip the values
        discount_rate_with_variation = np.clip(
            discount_array + discount_rate_variation, 0, 2
        )
        self.set_array(
            discount_rate_with_variation,
            name="agents/farmers/discount_rate",
        )

        interest_rate = np.full(n_farmers, interest_rate, dtype=np.float32)
        self.set_array(interest_rate, name="agents/farmers/interest_rate")

    def setup_farmer_irrigation_source(self, irrigating_farmers, year):
        fraction_sw_irrigation = "aeisw"
        fraction_sw_irrigation_data = self.data_catalog.get_rasterdataset(
            f"global_irrigation_area_{fraction_sw_irrigation}",
            bbox=self.bounds,
            buffer=2,
        )
        fraction_gw_irrigation = "aeigw"
        fraction_gw_irrigation_data = self.data_catalog.get_rasterdataset(
            f"global_irrigation_area_{fraction_gw_irrigation}",
            bbox=self.bounds,
            buffer=2,
        )

        farmer_locations = get_farm_locations(
            self.subgrid["agents/farmers/farms"], method="centroid"
        )

        # Determine which farmers are irrigating
        grid_id_da = self.create_grid_cell_id_array(fraction_sw_irrigation_data)
        ny, nx = (
            fraction_sw_irrigation_data.sizes["y"],
            fraction_sw_irrigation_data.sizes["x"],
        )

        n_cells = grid_id_da.max().item()
        n_farmers = self.array["agents/farmers/id"].size

        farmer_cells = sample_from_map(
            grid_id_da.values,
            farmer_locations,
            grid_id_da.rio.transform(recalc=True).to_gdal(),
        )
        fraction_sw_irrigation_farmers = sample_from_map(
            fraction_sw_irrigation_data.values,
            farmer_locations,
            fraction_sw_irrigation_data.rio.transform(recalc=True).to_gdal(),
        )
        fraction_gw_irrigation_farmers = sample_from_map(
            fraction_gw_irrigation_data.values,
            farmer_locations,
            fraction_gw_irrigation_data.rio.transform(recalc=True).to_gdal(),
        )

        adaptations = np.full(
            (
                n_farmers,
                max(
                    [
                        SURFACE_IRRIGATION_EQUIPMENT,
                        WELL_ADAPTATION,
                        IRRIGATION_EFFICIENCY_ADAPTATION,
                        FIELD_EXPANSION_ADAPTATION,
                    ]
                )
                + 1,
            ),
            -1,
            dtype=np.int32,
        )

        for i in range(n_cells):
            farmers_cell_mask = farmer_cells == i  # Boolean mask for farmers in cell i
            farmers_cell_indices = np.where(farmers_cell_mask)[0]  # Absolute indices

            irrigating_farmers_mask = irrigating_farmers[farmers_cell_mask]
            num_irrigating_farmers = np.sum(irrigating_farmers_mask)

            if num_irrigating_farmers > 0:
                fraction_sw = fraction_sw_irrigation_farmers[farmers_cell_mask][0]
                fraction_gw = fraction_gw_irrigation_farmers[farmers_cell_mask][0]

                # Normalize fractions
                total_fraction = fraction_sw + fraction_gw

                # Handle edge cases if there are irrigating farmers but no data on sw/gw
                if total_fraction == 0:
                    # Find neighboring cells with valid data
                    neighbor_ids = self.get_neighbor_cell_ids(i, nx, ny)
                    found_valid_neighbor = False

                    for neighbor_id in neighbor_ids:
                        if neighbor_id not in np.unique(farmer_cells):
                            continue

                        neighbor_mask = farmer_cells == neighbor_id
                        fraction_sw_neighbor = fraction_sw_irrigation_farmers[
                            neighbor_mask
                        ][0]
                        fraction_gw_neighbor = fraction_gw_irrigation_farmers[
                            neighbor_mask
                        ][0]
                        neighbor_total_fraction = (
                            fraction_sw_neighbor + fraction_gw_neighbor
                        )

                        if neighbor_total_fraction > 0:
                            # Found valid neighbor
                            fraction_sw = fraction_sw_neighbor
                            fraction_gw = fraction_gw_neighbor
                            total_fraction = neighbor_total_fraction

                            found_valid_neighbor = True
                            break
                    if not found_valid_neighbor:
                        # No valid neighboring cells found, handle accordingly
                        print(f"No valid data found for cell {i} and its neighbors.")
                        continue  # Skip this cell

                # Normalize fractions
                probabilities = np.array([fraction_sw, fraction_gw], dtype=np.float64)
                probabilities_sum = probabilities.sum()
                probabilities /= probabilities_sum

                # Indices of irrigating farmers in the region (absolute indices)
                farmer_indices_in_region = farmers_cell_indices[irrigating_farmers_mask]

                # Assign irrigation sources using np.random.choice
                irrigation_equipment_per_farmer = np.random.choice(
                    [SURFACE_IRRIGATION_EQUIPMENT, WELL_ADAPTATION],
                    size=len(farmer_indices_in_region),
                    p=probabilities,
                )

                adaptations[
                    farmer_indices_in_region, irrigation_equipment_per_farmer
                ] = 1

        self.set_array(adaptations, name="agents/farmers/adaptations")

    def setup_assets(self, feature_types, source="geofabrik", overwrite=False):
        """
        Get assets from OpenStreetMap (OSM) data.

        Parameters
        ----------
        feature_types : str or list of str
            The types of features to download from OSM. Available feature types are 'buildings', 'rails' and 'roads'.
        source : str, optional
            The source of the OSM data. Options are 'geofabrik' or 'movisda'. Default is 'geofabrik'.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        """
        if isinstance(feature_types, str):
            feature_types = [feature_types]

        OSM_data_dir = self.preprocessing_dir / "osm"
        OSM_data_dir.mkdir(exist_ok=True, parents=True)

        if source == "geofabrik":
            index_file = OSM_data_dir / "geofabrik_region_index.geojson"
            fetch_and_save(
                "https://download.geofabrik.de/index-v1.json",
                index_file,
                overwrite=overwrite,
            )

            index = gpd.read_file(index_file)
            # remove Dach region as all individual regions within dach countries are also in the index
            index = index[index["id"] != "dach"]

            # find all regions that intersect with the bbox
            intersecting_regions = index[index.intersects(self.region.geometry[0])]

            def filter_regions(ID, parents):
                return ID not in parents

            intersecting_regions = intersecting_regions[
                intersecting_regions["id"].apply(
                    lambda x: filter_regions(x, intersecting_regions["parent"].tolist())
                )
            ]

            urls = (
                intersecting_regions["urls"]
                .apply(lambda x: json.loads(x)["pbf"])
                .tolist()
            )

        elif source == "movisda":
            minx, miny, maxx, maxy = self.bounds

            urls = []
            for x in range(int(minx), int(maxx) + 1):
                # Movisda seems to switch the W and E for the x coordinate
                EW_code = f"E{-x:03d}" if x < 0 else f"W{x:03d}"
                for y in range(int(miny), int(maxy) + 1):
                    NS_code = f"N{y:02d}" if y >= 0 else f"S{-y:02d}"
                    url = f"https://osm.download.movisda.io/grid/{NS_code}{EW_code}-latest.osm.pbf"

                    # some tiles do not exists because they are in the ocean. Therefore we check if they exist
                    # before adding the url
                    response = requests.head(url, allow_redirects=True)
                    if response.status_code != 404:
                        urls.append(url)

        else:
            raise ValueError(f"Unknown source {source}")

        # download all regions
        all_features = {}
        for url in tqdm(urls):
            filepath = OSM_data_dir / url.split("/")[-1]
            fetch_and_save(url, filepath, overwrite=overwrite)
            for feature_type in feature_types:
                if feature_type not in all_features:
                    all_features[feature_type] = []

                if feature_type == "buildings":
                    features = gpd.read_file(
                        filepath,
                        mask=self.region,
                        layer="multipolygons",
                        use_arrow=True,
                    )
                    features = features[features["building"].notna()]
                elif feature_type == "rails":
                    features = gpd.read_file(
                        filepath,
                        mask=self.region,
                        layer="lines",
                        use_arrow=True,
                    )
                    features = features[
                        features["railway"].isin(
                            ["rail", "tram", "subway", "light_rail", "narrow_gauge"]
                        )
                    ]
                elif feature_type == "roads":
                    features = gpd.read_file(
                        filepath,
                        mask=self.region,
                        layer="lines",
                        use_arrow=True,
                    )
                    features = features[
                        features["highway"].isin(
                            [
                                "motorway",
                                "trunk",
                                "primary",
                                "secondary",
                                "tertiary",
                                "unclassified",
                                "residential",
                                "motorway_link",
                                "trunk_link",
                                "primary_link",
                                "secondary_link",
                                "tertiary_link",
                            ]
                        )
                    ]
                else:
                    raise ValueError(f"Unknown feature type {feature_type}")

                all_features[feature_type].append(features)

        for feature_type in feature_types:
            features = pd.concat(all_features[feature_type], ignore_index=True)
            self.set_geoms(features, name=f"assets/{feature_type}")

    def interpolate(self, ds, interpolation_method, ydim="y", xdim="x"):
        out_ds = ds.interp(
            method=interpolation_method,
            **{
                ydim: self.grid.y.rename({"y": ydim}),
                xdim: self.grid.x.rename({"x": xdim}),
            },
        )
        if "inplace" in out_ds.coords:
            out_ds = out_ds.drop_vars(["dparams", "inplace"])
        assert len(ds.dims) == len(out_ds.dims)
        return out_ds

    def download_isimip(
        self,
        product,
        variable,
        forcing,
        starttime=None,
        endtime=None,
        simulation_round="ISIMIP3a",
        climate_scenario="obsclim",
        resolution=None,
        buffer=0,
    ):
        """
        Downloads ISIMIP climate data for GEB.

        Parameters
        ----------
        product : str
            The name of the ISIMIP product to download.
        variable : str
            The name of the climate variable to download.
        forcing : str
            The name of the climate forcing to download.
        starttime : date, optional
            The start date of the data. Default is None.
        endtime : date, optional
            The end date of the data. Default is None.
        resolution : str, optional
            The resolution of the data to download. Default is None.
        buffer : int, optional
            The buffer size in degrees to add to the bounding box of the data to download. Default is 0.

        Returns
        -------
        xr.Dataset
            The downloaded climate data as an xarray dataset.

        Notes
        -----
        This method downloads ISIMIP climate data for GEB. It first retrieves the dataset
        metadata from the ISIMIP repository using the specified `product`, `variable`, `forcing`, and `resolution`
        parameters. It then downloads the data files that match the specified `starttime` and `endtime` parameters, and
        extracts them to the specified `download_path` directory.

        The resulting climate data is returned as an xarray dataset. The dataset is assigned the coordinate reference system
        EPSG:4326, and the spatial dimensions are set to 'lon' and 'lat'.
        """
        # if starttime is specified, endtime must be specified as well
        assert (starttime is None) == (endtime is None)

        client = ISIMIPClient()
        download_path = self.preprocessing_dir / "climate" / forcing / variable
        download_path.mkdir(parents=True, exist_ok=True)

        # Code to get data from disk rather than server.
        parse_files = []
        for file in os.listdir(download_path):
            if file.endswith(".nc"):
                fp = download_path / file
                parse_files.append(fp)

        # get the dataset metadata from the ISIMIP repository
        response = client.datasets(
            simulation_round=simulation_round,
            product=product,
            climate_forcing=forcing,
            climate_scenario=climate_scenario,
            climate_variable=variable,
            resolution=resolution,
        )
        assert len(response["results"]) == 1
        dataset = response["results"][0]
        files = dataset["files"]

        xmin, ymin, xmax, ymax = self.bounds
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer

        if variable == "orog":
            assert len(files) == 1
            filename = files[
                0
            ][
                "name"
            ]  # global should be included due to error in ISIMIP API .replace('_global', '')
            parse_files = [filename]
            if not (download_path / filename).exists():
                download_files = [files[0]["path"]]
            else:
                download_files = []

        else:
            assert starttime is not None and endtime is not None
            download_files = []
            parse_files = []
            for file in files:
                name = file["name"]
                assert name.endswith(".nc")
                splitted_filename = name.split("_")
                date = splitted_filename[-1].split(".")[0]
                if "-" in date:
                    start_date, end_date = date.split("-")
                    start_date = datetime.strptime(start_date, "%Y%m%d").date()
                    end_date = datetime.strptime(end_date, "%Y%m%d").date()
                elif len(date) == 6:
                    start_date = datetime.strptime(date, "%Y%m").date()
                    end_date = (
                        start_date + relativedelta(months=1) - relativedelta(days=1)
                    )
                elif len(date) == 4:  # is year
                    assert splitted_filename[-2].isdigit()
                    start_date = datetime.strptime(splitted_filename[-2], "%Y").date()
                    end_date = datetime.strptime(date, "%Y").date()
                else:
                    raise ValueError(f"could not parse date {date} from file {name}")

                if not (end_date < starttime or start_date > endtime):
                    parse_files.append(file["name"].replace("_global", ""))
                    if not (
                        download_path / file["name"].replace("_global", "")
                    ).exists():
                        download_files.append(file["path"])

        if download_files:
            self.logger.info(f"Requesting download of {len(download_files)} files")
            while True:
                try:
                    response = client.cutout(download_files, [ymin, ymax, xmin, xmax])
                except requests.exceptions.HTTPError:
                    self.logger.warning(
                        "HTTPError, could not download files, retrying in 60 seconds"
                    )
                else:
                    if response["status"] == "finished":
                        break
                    elif response["status"] == "started":
                        self.logger.info(
                            f"{response['meta']['created_files']}/{response['meta']['total_files']} files prepared on ISIMIP server for {variable}, waiting 60 seconds before retrying"
                        )
                    elif response["status"] == "queued":
                        self.logger.info(
                            f"Data preparation queued for {variable} on ISIMIP server, waiting 60 seconds before retrying"
                        )
                    elif response["status"] == "failed":
                        self.logger.info(
                            "ISIMIP internal server error, waiting 60 seconds before retrying"
                        )
                    else:
                        raise ValueError(
                            f"Could not download files: {response['status']}"
                        )
                time.sleep(60)
            self.logger.info(f"Starting download of files for {variable}")
            # download the file when it is ready
            client.download(
                response["file_url"], path=download_path, validate=False, extract=False
            )
            self.logger.info(f"Download finished for {variable}")
            # remove zip file
            zip_file = download_path / Path(
                urlparse(response["file_url"]).path.split("/")[-1]
            )
            # make sure the file exists
            assert zip_file.exists()
            # Open the zip file
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                # Get a list of all the files in the zip file
                file_list = [f for f in zip_ref.namelist() if f.endswith(".nc")]
                # Extract each file one by one
                for i, file_name in enumerate(file_list):
                    # Rename the file
                    bounds_str = ""
                    if isinstance(ymin, float):
                        bounds_str += f"_lat{ymin}"
                    else:
                        bounds_str += f"_lat{ymin:.1f}"
                    if isinstance(ymax, float):
                        bounds_str += f"to{ymax}"
                    else:
                        bounds_str += f"to{ymax:.1f}"
                    if isinstance(xmin, float):
                        bounds_str += f"lon{xmin}"
                    else:
                        bounds_str += f"lon{xmin:.1f}"
                    if isinstance(xmax, float):
                        bounds_str += f"to{xmax}"
                    else:
                        bounds_str += f"to{xmax:.1f}"
                    assert bounds_str in file_name
                    new_file_name = file_name.replace(bounds_str, "")
                    zip_ref.getinfo(file_name).filename = new_file_name
                    # Extract the file
                    if os.name == "nt":
                        max_file_path_length = 260
                    else:
                        max_file_path_length = os.pathconf("/", "PC_PATH_MAX")
                    assert (
                        len(str(download_path / new_file_name)) <= max_file_path_length
                    ), (
                        f"File path too long: {download_path / zip_ref.getinfo(file_name).filename}"
                    )
                    zip_ref.extract(file_name, path=download_path)
            # remove zip file
            (
                download_path / Path(urlparse(response["file_url"]).path.split("/")[-1])
            ).unlink()

        datasets = [
            xr.open_dataset(download_path / file, chunks={}) for file in parse_files
        ]
        for dataset in datasets:
            assert "lat" in dataset.coords and "lon" in dataset.coords

        # make sure y is decreasing rather than increasing
        datasets = [
            (
                dataset.reindex(lat=dataset.lat[::-1])
                if dataset.lat[0] < dataset.lat[-1]
                else dataset
            )
            for dataset in datasets
        ]

        reference = datasets[0]
        for dataset in datasets:
            # make sure all datasets have more or less the same coordinates
            assert np.isclose(
                dataset.coords["lat"].values,
                reference["lat"].values,
                atol=abs(datasets[0].rio.resolution()[1] / 50),
                rtol=0,
            ).all()
            assert np.isclose(
                dataset.coords["lon"].values,
                reference["lon"].values,
                atol=abs(datasets[0].rio.resolution()[0] / 50),
                rtol=0,
            ).all()

        datasets = [
            ds.assign_coords(lon=reference["lon"].values, lat=reference["lat"].values)
            for ds in datasets
        ]
        if len(datasets) > 1:
            ds = xr.concat(datasets, dim="time")
        else:
            ds = datasets[0]

        if starttime is not None:
            ds = ds.sel(time=slice(starttime, endtime))
            # assert that time is monotonically increasing with a constant step size
            assert (
                ds.time.diff("time").astype(np.int64)
                == (ds.time[1] - ds.time[0]).astype(np.int64)
            ).all()

        ds.raster.set_spatial_dims(x_dim="lon", y_dim="lat")
        assert not ds.lat.attrs, "lat already has attributes"
        assert not ds.lon.attrs, "lon already has attributes"
        ds.lat.attrs = {
            "long_name": "latitude of grid cell center",
            "units": "degrees_north",
        }
        ds.lon.attrs = {
            "long_name": "longitude of grid cell center",
            "units": "degrees_east",
        }
        ds.raster.set_crs(4326)

        # check whether data is for noon or midnight. If noon, subtract 12 hours from time coordinate to align with other datasets
        if hasattr(ds, "time") and pd.to_datetime(ds.time[0].values).hour == 12:
            # subtract 12 hours from time coordinate
            self.logger.warning(
                "Subtracting 12 hours from time coordinate to align climate datasets"
            )
            ds = ds.assign_coords(time=ds.time - np.timedelta64(12, "h"))
        return ds

    def setup_coastal_water_levels(
        self,
    ):
        water_levels = self.data_catalog.get_dataset("GTSM")
        assert (
            water_levels.time.diff("time").astype(np.int64)
            == (water_levels.time[1] - water_levels.time[0]).astype(np.int64)
        ).all()
        # convert to geodataframe
        stations = gpd.GeoDataFrame(
            water_levels.stations,
            geometry=gpd.points_from_xy(
                water_levels.station_x_coordinate, water_levels.station_y_coordinate
            ),
        )
        # filter all stations within the bounds, considering a buffer
        station_ids = stations.cx[
            self.bounds[0] - 0.1 : self.bounds[2] + 0.1,
            self.bounds[1] - 0.1 : self.bounds[3] + 0.1,
        ].index.values

        water_levels = water_levels.sel(stations=station_ids)

        assert len(water_levels.stations) > 0, (
            "No stations found in the region. If no stations should be set, set include_coastal=False"
        )

        self.set_other(
            water_levels,
            name="waterlevels",
            time_chunksize=24 * 6,  # 10 minute data
            byteshuffle=True,
        )

    def setup_damage_parameters(self, parameters):
        for hazard, hazard_parameters in parameters.items():
            for asset_type, asset_parameters in hazard_parameters.items():
                for component, asset_compontents in asset_parameters.items():
                    curve = pd.DataFrame(
                        asset_compontents["curve"], columns=["severity", "damage_ratio"]
                    )

                    self.set_table(
                        curve,
                        name=f"damage_parameters/{hazard}/{asset_type}/{component}/curve",
                    )

                    maximum_damage = {
                        "maximum_damage": asset_compontents["maximum_damage"]
                    }

                    self.set_dict(
                        maximum_damage,
                        name=f"damage_parameters/{hazard}/{asset_type}/{component}/maximum_damage",
                    )

    def setup_precipitation_scaling_factors_for_return_periods(
        self, risk_scaling_factors
    ):
        risk_scaling_factors = pd.DataFrame(
            risk_scaling_factors, columns=["exceedance_probability", "scaling_factor"]
        )
        self.set_table(risk_scaling_factors, name="precipitation_scaling_factors")

    def setup_discharge_observations(self, files):
        transform = self.grid.rio.transform(recalc=True)

        discharge_data = []
        for i, file in enumerate(files):
            filename = file["filename"]
            longitude, latitude = file["longitude"], file["latitude"]
            data = pd.read_csv(filename, index_col=0, parse_dates=True)

            # assert data has one column
            assert data.shape[1] == 1

            px, py = ~transform * (longitude, latitude)
            px = math.floor(px)
            py = math.floor(py)

            discharge_data.append(
                xr.DataArray(
                    np.expand_dims(data.iloc[:, 0].values, 0),
                    dims=["pixel", "time"],
                    coords={
                        "time": data.index.values,
                        "pixel": [i],
                        "px": ("pixel", [px]),
                        "py": ("pixel", [py]),
                    },
                )
            )
        discharge_data = xr.concat(discharge_data, dim="pixel")
        self.set_other(
            discharge_data,
            name="observations/discharge",
            time_chunksize=1e99,  # no chunking
        )

    def set_table(self, table, name, write=True):
        self.table[name] = table
        if write:
            fn = Path("table") / (name + ".parquet")
            self.logger.info(f"Writing file {fn}")

            self.files["table"][name] = fn

            fp = Path(self.root, fn)
            fp.parent.mkdir(parents=True, exist_ok=True)
            # brotli is a bit slower but gives better compression,
            # gzip is faster to read. Higher compression levels
            # generally don't make it slower to read, therefore
            # we use the highest compression level for gzip
            table.to_parquet(
                fp, engine="pyarrow", compression="gzip", compression_level=9
            )

    def set_array(self, data, name, write=True):
        self.array[name] = data

        if write:
            fn = Path("array") / (name + ".npz")
            self.logger.info(f"Writing file {fn}")
            self.files["array"][name] = fn
            fp = Path(self.root, fn)
            fp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fp, data=data)

    def set_dict(self, data, name, write=True):
        self.dict[name] = data

        if write:
            fn = Path("dict") / (name + ".json")
            self.logger.info(f"Writing file {fn}")

            self.files["dict"][name] = fn

            fp = Path(self.root) / fn
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, "w") as f:
                json.dump(data, f, default=convert_timestamp_to_string)

    def set_geoms(self, geoms, name, write=True):
        self.geoms[name] = geoms

        if write:
            fn = Path("geom") / (name + ".geoparquet")
            self.logger.info(f"Writing file {fn}")
            self.files["geoms"][name] = fn
            fp = self.root / fn
            fp.parent.mkdir(parents=True, exist_ok=True)
            # brotli is a bit slower but gives better compression,
            # gzip is faster to read. Higher compression levels
            # generally don't make it slower to read, therefore
            # we use the highest compression level for gzip
            geoms.to_parquet(
                fp, engine="pyarrow", compression="gzip", compression_level=9
            )

        return self.geoms[name]

    def write_file_library(self):
        file_library = self.read_file_library()

        # merge file library from disk with new files, prioritizing new files
        for type_name, type_files in self.files.items():
            if type_name not in file_library:
                file_library[type_name] = type_files
            else:
                file_library[type_name].update(type_files)

        with open(Path(self.root, "files.json"), "w") as f:
            json.dump(self.files, f, indent=4, cls=PathEncoder)

        self.logger.info("Done")

    def read_file_library(self):
        fp = Path(self.root, "files.json")
        if not fp.exists():
            return {}
        else:
            with open(Path(self.root, "files.json"), "r") as f:
                files = json.load(f)
        return defaultdict(dict, files)  # convert dict to defaultdict

    def read_geoms(self):
        for name, fn in self.files["geoms"].items():
            geom = gpd.read_parquet(Path(self.root, fn))
            self.set_geoms(geom, name=name, write=False)

    def read_array(self):
        for name, fn in self.files["array"].items():
            array = np.load(Path(self.root, fn))["data"]
            self.set_array(array, name=name, write=False)

    def read_table(self):
        for name, fn in self.files["table"].items():
            table = pd.read_parquet(Path(self.root, fn))
            self.set_table(table, name=name, write=False)

    def read_dict(self):
        for name, fn in self.files["dict"].items():
            with open(Path(self.root, fn), "r") as f:
                d = json.load(f)
            self.set_dict(d, name=name, write=False)

    def _read_grid(self, fn) -> xr.Dataset:
        da = open_zarr(Path(self.root) / fn)
        return da

    def read_grid(self) -> None:
        for name, fn in self.files["grid"].items():
            data = self._read_grid(fn)
            self.set_grid(data, name=name, write=False)

    def read_subgrid(self) -> None:
        for name, fn in self.files["subgrid"].items():
            data = self._read_grid(fn)
            self.set_subgrid(data, name=name, write=False)

    def read_region_subgrid(self) -> None:
        for name, fn in self.files["region_subgrid"].items():
            data = self._read_grid(fn)
            self.set_region_subgrid(data, name=name, write=False)

    def read_other(self) -> None:
        for name, fn in self.files["other"].items():
            da = open_zarr(Path(self.root) / fn)
            self.set_other(da, name=name, write=False)
        return None

    def read(self):
        with suppress_logging_warning(self.logger):
            self.files = self.read_file_library()

            self.read_geoms()
            self.read_array()
            self.read_table()
            self.read_dict()

            self.read_subgrid()
            self.read_grid()
            self.read_region_subgrid()

            self.read_other()

    def set_other(
        self,
        da,
        name: str,
        write=True,
        x_chunksize=XY_CHUNKSIZE,
        y_chunksize=XY_CHUNKSIZE,
        time_chunksize=1,
        *args,
        **kwargs,
    ):
        assert isinstance(da, xr.DataArray)

        if write:
            self.logger.info(f"Write {name}")

            fp = Path("other") / (name + ".zarr")
            self.files["other"][name] = fp

            dst_file = Path(self.root, fp)
            da = to_zarr(
                da,
                dst_file,
                x_chunksize=x_chunksize,
                y_chunksize=y_chunksize,
                time_chunksize=time_chunksize,
                crs=da.rio.crs,
                *args,
                **kwargs,
            )
        self.other[name] = da
        return self.other[name]

    def _set_grid(
        self,
        grid_name,
        grid,
        data: xr.DataArray,
        name: str,
        write,
        x_chunksize=XY_CHUNKSIZE,
        y_chunksize=XY_CHUNKSIZE,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """

        assert isinstance(data, xr.DataArray)

        if name in grid:
            self.logger.warning(f"Replacing grid map: {name}")
            grid = grid.drop_vars(name)

        if len(grid) == 0:
            assert name == "mask", "First grid layer must be mask"
            assert data.dtype == bool
        else:
            assert np.array_equal(data.x.values, grid.x.values)
            assert np.array_equal(data.y.values, grid.y.values)

            # when updating, it is possible that the mask already exists.
            if name != "mask":
                # if the mask exists, mask the data, saving some valuable space on disk
                data_ = xr.where(
                    ~grid["mask"],
                    data,
                    data.attrs["_FillValue"] if data.dtype != bool else False,
                    keep_attrs=True,
                )
                # depending on the order of the dimensions, xr.where may change the dimension order
                # so we change it back
                data = data_.transpose(*data.dims)

        if write:
            fn = Path(grid_name) / (name + ".zarr")
            self.logger.info(f"Writing file {fn}")
            data = to_zarr(
                data,
                path=self.root / fn,
                x_chunksize=x_chunksize,
                y_chunksize=y_chunksize,
                crs=4326,
            )
            self.files[grid_name][name] = Path(grid_name) / (name + ".zarr")

        grid[name] = data
        return grid

    def set_grid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, write=True
    ) -> None:
        self._set_grid("grid", self.grid, data, write=write, name=name)
        return self.grid[name]

    def set_subgrid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, write=True
    ) -> None:
        self.subgrid = self._set_grid(
            "subgrid", self.subgrid, data, write=write, name=name
        )
        return self.subgrid[name]

    def set_region_subgrid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, write=True
    ) -> None:
        self.region_subgrid = self._set_grid(
            "region_subgrid",
            self.region_subgrid,
            data,
            write=write,
            name=name,
        )
        return self.region_subgrid[name]

    def set_alternate_root(self, root, mode):
        relative_path = Path(os.path.relpath(Path(self.root), root.resolve()))
        for data in self.files.values():
            for name, fn in data.items():
                data[name] = relative_path / fn
        super().set_root(root, mode)

    @property
    def subgrid_factor(self):
        subgrid_factor = self.subgrid.dims["x"] // self.grid.dims["x"]
        assert subgrid_factor == self.subgrid.dims["y"] // self.grid.dims["y"]
        return subgrid_factor

    @property
    def region(self):
        return self.geoms["mask"]

    @property
    def bounds(self):
        return self.grid.rio.bounds(recalc=True)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = Path(root).absolute()

    @property
    def preprocessing_dir(self):
        return Path(self.root).parent / "preprocessing"

    @property
    def report_dir(self):
        return Path(self.root).parent / "report"

    def full_like(self, data, fill_value, nodata, *args, **kwargs):
        ds = xr.full_like(data, fill_value, *args, **kwargs)
        ds.attrs = {
            "_FillValue": nodata,
        }
        return ds

    def check_methods(self, opt):
        """Check all opt keys and raise sensible error messages if unknown."""
        for method in opt.keys():
            if not callable(getattr(self, method, None)):
                raise ValueError(f'Model {self._NAME} has no method "{method}"')
        return opt

    def run_method(self, method, *args, **kwargs):
        """Log method parameters before running a method."""
        func = getattr(self, method)
        signature = inspect.signature(func)
        # combine user and default options
        params = {}
        for i, (k, v) in enumerate(signature.parameters.items()):
            if k in ["args", "kwargs"]:
                if k == "args":
                    params[k] = args[i:]
                else:
                    params.update(**kwargs)
            else:
                v = kwargs.get(k, v.default)
                if len(args) > i:
                    v = args[i]
                params[k] = v
        # log options
        for k, v in params.items():
            if v is not inspect._empty:
                self.logger.info(f"{method}.{k}: {v}")
        return func(*args, **kwargs)

    def run_methods(self, methods):
        # then loop over other methods
        for method in methods:
            kwargs = {} if methods[method] is None else methods[method]
            self.run_method(method, **kwargs)
            self.write_file_library()

    def build(self, region: dict, methods: dict):
        methods = methods or {}
        methods = self.check_methods(methods)
        methods["setup_region"].update(region=region)

        self.run_methods(methods)

    def update(
        self,
        methods: dict,
    ):
        methods = methods or {}
        methods = self.check_methods(methods)

        if "setup_region" in methods:
            raise ValueError('"setup_region" can only be called when building a model.')

        self.run_methods(methods)

    def create_grid_cell_id_array(self, area_fraction_da):
        # Get the sizes of the spatial dimensions
        ny, nx = area_fraction_da.sizes["y"], area_fraction_da.sizes["x"]

        # Create an array of sequential integers from 0 to ny*nx - 1
        grid_ids = np.arange(ny * nx).reshape(ny, nx)

        # Create a DataArray with the same coordinates and dimensions as your spatial grid
        grid_id_da = xr.DataArray(
            grid_ids,
            coords={
                "y": area_fraction_da.coords["y"],
                "x": area_fraction_da.coords["x"],
            },
            dims=["y", "x"],
        )

        return grid_id_da

    def get_neighbor_cell_ids(self, cell_id, nx, ny, radius=1):
        row = cell_id // nx
        col = cell_id % nx

        neighbor_cell_ids = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue  # Skip the cell itself
                r = row + dr
                c = col + dc
                if 0 <= r < ny and 0 <= c < nx:
                    neighbor_id = r * nx + c
                    neighbor_cell_ids.append(neighbor_id)
        return neighbor_cell_ids
