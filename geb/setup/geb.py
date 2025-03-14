"""This module contains the main setup for the GEB model.

Notes:
- All prices are in nominal USD (face value) for their respective years. That means that the prices are not adjusted for inflation.
"""

from tqdm import tqdm
from pathlib import Path
import hydromt.workflows
from datetime import date, datetime, timedelta
from typing import Union, Dict, List, Optional
import logging
import os
import math
import requests
import time
import random
import zipfile
import shutil
import tempfile
import json
from urllib.parse import urlparse
from hydromt.exceptions import NoDataException
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from affine import Affine
from pyproj import CRS
from rasterio.env import defenv
import xarray as xr
from dask.diagnostics import ProgressBar
import xclim.indices as xci
from dateutil.relativedelta import relativedelta
from contextlib import contextmanager
from calendar import monthrange
from numcodecs import Blosc
from scipy.ndimage import value_indices
import pyflwdir

from hydromt.models.model_grid import GridModel
from hydromt.data_catalog import DataCatalog
from hydromt.data_adapter import (
    RasterDatasetAdapter,
    DatasetAdapter,
)

from honeybees.library.raster import sample_from_map, pixels_to_coords
from isimip_client.client import ISIMIPClient

from .workflows.general import (
    repeat_grid,
    clip_with_grid,
    pad_xy,
    calculate_cell_area,
    fetch_and_save,
    bounds_are_within,
)
from .workflows.farmers import get_farm_locations, create_farms, get_farm_distribution
from .workflows.population import generate_locations, load_GLOPOP_S
from .workflows.crop_calendars import parse_MIRCA2000_crop_calendar
from .workflows.soilgrids import load_soilgrids
from .workflows.conversions import (
    M49_to_ISO3,
    SUPERWELL_NAME_TO_ISO3,
    GLOBIOM_NAME_TO_ISO3,
    COUNTRY_NAME_TO_ISO3,
)
from .workflows.forcing import (
    reproject_and_apply_lapse_rate_temperature,
    reproject_and_apply_lapse_rate_pressure,
    download_ERA5,
    open_ERA5,
)
from .workflows.hydrography import (
    get_upstream_subbasin_ids,
    get_subbasin_id_from_coordinate,
    get_sink_subbasin_id_for_geom,
    get_subbasins_geometry,
    get_river_graph,
    get_downstream_subbasins,
    get_rivers,
    create_river_raster_from_river_lines,
    get_SWORD_translation_IDs_and_lenghts,
    get_SWORD_river_widths,
)

from geb.agents.crop_farmers import (
    SURFACE_IRRIGATION_EQUIPMENT,
    WELL_ADAPTATION,
    IRRIGATION_EFFICIENCY_ADAPTATION,
    FIELD_EXPANSION_ADAPTATION,
)

# Set environment options for robustness
GDAL_HTTP_ENV_OPTS = {
    "GDAL_HTTP_MAX_RETRY": "10",  # Number of retry attempts
    "GDAL_HTTP_RETRY_DELAY": "2",  # Delay (seconds) between retries
    "GDAL_HTTP_TIMEOUT": "30",  # Timeout in seconds
}
defenv(**GDAL_HTTP_ENV_OPTS)

XY_CHUNKSIZE = 350

# temporary fix for ESMF on Windows
if os.name == "nt":
    os.environ["ESMFMKFILE"] = str(
        Path(os.__file__).parent.parent / "Library" / "lib" / "esmf.mk"
    )
else:
    os.environ["ESMFMKFILE"] = str(Path(os.__file__).parent.parent / "esmf.mk")

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

logger = logging.getLogger(__name__)


def create_grid_cell_id_array(area_fraction_da):
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


def get_neighbor_cell_ids(cell_id, nx, ny, radius=1):
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
            return str(obj)
        return super().default(obj)


class GEBModel(GridModel):
    _CLI_ARGS = {"region": "setup_grid"}

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = None,
        data_libs: List[str] = None,
        logger=logger,
        epsg=4326,
        data_provider: str = None,
    ):
        """Initialize a GridModel for distributed models with a regular grid."""
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

        self.epsg = epsg
        self.data_provider = data_provider

        self._subgrid = None
        self._region_subgrid = None
        self._MERIT_grid = None

        self.table = {}
        self.binary = {}
        self.dict = {}

        self.files = {
            "forcing": {},
            "geoms": {},
            "grid": {},
            "dict": {},
            "table": {},
            "binary": {},
            "subgrid": {},
            "region_subgrid": {},
            "MERIT_grid": {},
        }
        self.is_updated = {
            "forcing": {},
            "geoms": {},
            "grid": {},
            "dict": {},
            "table": {},
            "binary": {},
            "subgrid": {},
            "region_subgrid": {},
            "MERIT_grid": {},
        }

    @property
    def subgrid(self):
        """Model static gridded data as xarray.Dataset."""
        if self._subgrid is None:
            self._subgrid = xr.Dataset()
            if self._read:
                self.read_subgrid()
        return self._subgrid

    @subgrid.setter
    def subgrid(self, value):
        self._subgrid = value

    @property
    def region_subgrid(self):
        """Model static gridded data as xarray.Dataset."""
        if self._region_subgrid is None:
            self._region_subgrid = xr.Dataset()
            if self._read:
                self.read_region_subgrid()
        return self._region_subgrid

    @region_subgrid.setter
    def region_subgrid(self, value):
        self._region_subgrid = value

    @property
    def MERIT_grid(self):
        """Model static gridded data as xarray.Dataset."""
        if self._MERIT_grid is None:
            self._MERIT_grid = xr.Dataset()
            if self._read:
                self.read_MERIT_grid()
        return self._MERIT_grid

    @MERIT_grid.setter
    def MERIT_grid(self, value):
        self._MERIT_grid = value

    def setup_grid(
        self,
        region: dict,
        sub_grid_factor: int,
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
        sub_grid_factor : int
            GEB implements a subgrid. This parameter determines the factor by which the subgrid is smaller than the original grid.
        """

        assert resolution_arcsec % 3 == 0, (
            "resolution_arcsec must be a multiple of 3 to align with MERIT"
        )
        assert sub_grid_factor >= 2

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

        # Add region and grid to model
        self.set_geoms(geom, name="region")

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

        # elevation
        elevation = elevation_coarsened.mean()
        self.set_grid(elevation, name="landsurface/topo/elevation")

        elevation_std = elevation_coarsened.std()
        self.set_grid(elevation_std, name="landsurface/topo/elevation_STD")

        self.set_MERIT_grid(
            hydrography["elv"], name="landsurface/topo/subgrid_elevation"
        )

        # outflow elevation
        outflow_elevation = elevation_coarsened.min()
        self.set_grid(outflow_elevation, name="routing/outflow_elevation")

        mask = xr.full_like(outflow_elevation, False, dtype=bool)
        # we use the inverted mask, that is True outside the study area
        mask.data = ~flow_raster_upscaled.mask.reshape(flow_raster_upscaled.shape)
        self.set_grid(mask, name="areamaps/grid_mask")

        slope = xr.full_like(outflow_elevation, np.nan, dtype=np.float32)
        slope.raster.set_nodata(np.nan)
        slope_data = pyflwdir.dem.slope(
            elevation.values,
            nodata=np.nan,
            latlon=True,
            transform=elevation.raster.transform,
        )
        # set slope to zero on the mask boundary
        slope_data[np.isnan(slope_data) & (~mask.data)] = 0
        slope.data = slope_data
        self.set_grid(slope, name="landsurface/topo/slope")

        # flow direction
        ldd = xr.full_like(outflow_elevation, 255, dtype=np.uint8)
        ldd.raster.set_nodata(255)
        ldd.data = flow_raster_upscaled.to_array(ftype="ldd")
        self.set_grid(ldd, name="routing/ldd")

        # upstream area
        upstream_area = xr.full_like(outflow_elevation, np.nan, dtype=np.float32)
        upstream_area.raster.set_nodata(np.nan)
        upstream_area_data = flow_raster_upscaled.upstream_area(unit="m2").astype(
            np.float32
        )
        upstream_area_data[upstream_area_data == -9999.0] = np.nan
        upstream_area.data = upstream_area_data
        self.set_grid(upstream_area, name="routing/upstream_area")

        # river length
        river_length = xr.full_like(outflow_elevation, np.nan, dtype=np.float32)
        river_length.raster.set_nodata(np.nan)
        river_length_data = flow_raster.subgrid_rivlen(
            idxs_out, unit="m", direction="down"
        )
        river_length_data[river_length_data == -9999.0] = np.nan
        river_length.data = river_length_data
        self.set_grid(river_length, name="routing/river_length")

        # river slope
        river_slope = xr.full_like(outflow_elevation, np.nan, dtype=np.float32)
        river_slope.raster.set_nodata(np.nan)
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

        COMID_IDs_raster = xr.full_like(outflow_elevation, -1, dtype=np.int32)
        COMID_IDs_raster.raster.set_nodata(-1)
        COMID_IDs_raster.data = COMID_IDs_raster_data
        self.set_grid(COMID_IDs_raster, name="routing/COMID_IDs")

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

        river_width = xr.full_like(outflow_elevation, np.nan, dtype=np.float32)
        river_width.raster.set_nodata(np.nan)
        assert river_width_data.dtype == np.float32
        river_width.data = river_width_data
        self.set_grid(river_width, name="routing/river_width")

        dst_transform = mask.raster.transform * Affine.scale(1 / sub_grid_factor)

        submask = hydromt.raster.full_from_transform(
            dst_transform,
            (
                mask.raster.shape[0] * sub_grid_factor,
                mask.raster.shape[1] * sub_grid_factor,
            ),
            nodata=0,
            dtype=mask.dtype,
            crs=mask.raster.crs,
            name="areamaps/sub_grid_mask",
            lazy=True,
        )
        submask.data = repeat_grid(mask.data, sub_grid_factor)

        self.set_subgrid(submask, name=submask.name)

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
        retrieves the grid mask from the `areamaps/grid_mask` attribute of the grid, and then calculates the cell area
        using the `calculate_cell_area()` function. The resulting cell area map is then set as the `areamaps/cell_area`
        attribute of the grid.

        Additionally, this method sets up a subgrid for the cell area map by creating a new grid with the same extent as
        the subgrid, and then repeating the cell area values from the main grid to the subgrid using the `repeat_grid()`
        function, and correcting for the subgrid factor. Thus, every subgrid cell within a grid cell has the same value.
        The resulting subgrid cell area map is then set as the `areamaps/sub_cell_area` attribute of the subgrid.
        """
        self.logger.info("Preparing cell area map.")
        mask = self.grid["areamaps/grid_mask"].raster
        affine = mask.transform

        cell_area = hydromt.raster.full(
            mask.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="areamaps/cell_area",
            lazy=True,
            crs=self.crs,
        )
        cell_area.data = calculate_cell_area(affine, mask.shape)
        self.set_grid(cell_area, name=cell_area.name)

        sub_cell_area = hydromt.raster.full(
            self.subgrid["areamaps/sub_grid_mask"].raster.coords,
            nodata=cell_area.raster.nodata,
            dtype=cell_area.dtype,
            name="areamaps/sub_cell_area",
            crs=self.crs,
            lazy=True,
        )

        sub_cell_area.data = (
            repeat_grid(cell_area.data, self.subgrid_factor) / self.subgrid_factor**2
        )
        self.set_subgrid(sub_cell_area, sub_cell_area.name)

    def setup_cell_area_map(self) -> None:
        self.logger.warn(
            "setup_cell_area_map is deprecated, use setup_cell_area instead"
        )
        self.setup_cell_area()

    def setup_crops(
        self,
        crop_data: dict,
        type: str = "MIRCA2000",
    ):
        assert type in ("MIRCA2000", "GAEZ")
        for crop_id, crop_values in crop_data.items():
            assert "name" in crop_values
            assert "reference_yield_kg_m2" in crop_values
            assert "is_paddy" in crop_values
            assert "rd_rain" in crop_values  # root depth rainfed crops
            assert "rd_irr" in crop_values  # root depth irrigated crops
            assert (
                "crop_group_number" in crop_values
            )  # adaptation level to drought (see WOFOST: https://wofost.readthedocs.io/en/7.2/)
            assert 5 >= crop_values["crop_group_number"] >= 0
            assert (
                crop_values["rd_rain"] >= crop_values["rd_irr"]
            )  # root depth rainfed crops should be larger than irrigated crops

            if type == "GAEZ":
                crop_values["l_ini"] = crop_values["d1"]
                crop_values["l_dev"] = crop_values["d2a"] + crop_values["d2b"]
                crop_values["l_mid"] = crop_values["d3a"] + crop_values["d3b"]
                crop_values["l_late"] = crop_values["d4"]
                del crop_values["d1"]
                del crop_values["d2a"]
                del crop_values["d2b"]
                del crop_values["d3a"]
                del crop_values["d3b"]
                del crop_values["d4"]

                assert "KyT" in crop_values

            elif type == "MIRCA2000":
                assert "a" in crop_values
                assert "b" in crop_values
                assert "P0" in crop_values
                assert "P1" in crop_values
                assert "l_ini" in crop_values
                assert "l_dev" in crop_values
                assert "l_mid" in crop_values
                assert "l_late" in crop_values
                assert "kc_initial" in crop_values
                assert "kc_mid" in crop_values
                assert "kc_end" in crop_values

            assert (
                crop_values["l_ini"]
                + crop_values["l_dev"]
                + crop_values["l_mid"]
                + crop_values["l_late"]
                == 100
            ), "Sum of l_ini, l_dev, l_mid, and l_late must be 100[%]"

        crop_data = {
            "data": crop_data,
            "type": type,
        }

        self.set_dict(crop_data, name="crops/crop_data")

    def setup_crops_from_source(
        self,
        source: Union[str, None] = "MIRCA2000",
        crop_specifier: Union[str, None] = None,
    ):
        """
        Sets up the crops data for the model.
        """
        self.logger.info("Preparing crops data")

        assert source in ("MIRCA2000",), (
            f"crop_variables_source {source} not understood, must be 'MIRCA2000'"
        )
        if crop_specifier is None:
            crop_data = {
                "data": (
                    self.data_catalog.get_dataframe("MIRCA2000_crop_data")
                    .set_index("id")
                    .to_dict(orient="index")
                ),
                "type": "MIRCA2000",
            }
        else:
            crop_data = {
                "data": (
                    self.data_catalog.get_dataframe(
                        f"MIRCA2000_crop_data_{crop_specifier}"
                    )
                    .set_index("id")
                    .to_dict(orient="index")
                ),
                "type": "MIRCA2000",
            }
        self.set_dict(crop_data, name="crops/crop_data")

    def process_crop_data(
        self,
        crop_prices,
        project_past_until_year=False,
        project_future_until_year=False,
        translate_crop_names=None,
    ):
        """
        Processes crop price data, performing adjustments, variability determination, and interpolation/extrapolation as needed.

        Parameters
        ----------
        crop_prices : str, int, or float
            If 'FAO_stat', fetches crop price data from FAO statistics. Otherwise, it can be a constant value for crop prices.
        project_past_until_year : int, optional
            The year to project past data until. Defaults to False.
        project_future_until_year : int, optional
            The year to project future data until. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing processed crop data in a time series format or as a constant value.

        Raises
        ------
        ValueError
            If crop_prices is neither a valid file path nor an integer/float.

        Notes
        -----
        The function performs the following steps:
        1. Fetches and processes crop data from FAO statistics if crop_prices is 'FAO_stat'.
        2. Adjusts the data for countries with missing values using PPP conversion rates.
        3. Determines price variability and performs interpolation/extrapolation of crop prices.
        4. Formats the processed data into a nested dictionary structure.
        """

        if crop_prices == "FAO_stat":
            crop_data = self.data_catalog.get_dataframe(
                "FAO_crop_price",
                variables=["Area Code (M49)", "year", "crop", "price_per_kg"],
            )

            # Dropping 58 (Belgium-Luxembourg combined), 200 (former Czechoslovakia),
            # 230 (old code Ethiopia), 891 (Serbia and Montenegro), 736 (former Sudan)
            crop_data = crop_data[
                ~crop_data["Area Code (M49)"].isin([58, 200, 230, 891, 736])
            ]

            crop_data["ISO3"] = crop_data["Area Code (M49)"].map(M49_to_ISO3)
            crop_data = crop_data.drop(columns=["Area Code (M49)"])

            crop_data["crop"] = crop_data["crop"].str.lower()

            assert not crop_data["ISO3"].isna().any(), "Missing ISO3 codes"

            all_years = crop_data["year"].unique()
            all_years.sort()
            all_crops = crop_data["crop"].unique()

            GLOBIOM_regions = self.data_catalog.get_dataframe("GLOBIOM_regions_37")
            GLOBIOM_regions["ISO3"] = GLOBIOM_regions["Country"].map(
                GLOBIOM_NAME_TO_ISO3
            )
            # For my personal branch
            GLOBIOM_regions.loc[
                GLOBIOM_regions["Country"] == "Switzerland", "Region37"
            ] = "EU_MidWest"
            assert not np.any(GLOBIOM_regions["ISO3"].isna()), "Missing ISO3 codes"

            ISO3_codes_region = self.geoms["areamaps/regions"]["ISO3"].unique()
            GLOBIOM_regions_region = GLOBIOM_regions[
                GLOBIOM_regions["ISO3"].isin(ISO3_codes_region)
            ]["Region37"].unique()
            ISO3_codes_GLOBIOM_region = GLOBIOM_regions[
                GLOBIOM_regions["Region37"].isin(GLOBIOM_regions_region)
            ]["ISO3"]

            # Setup dataFrame for further data corrections
            donor_data = {}
            for ISO3 in ISO3_codes_GLOBIOM_region:
                region_crop_data = crop_data[crop_data["ISO3"] == ISO3]
                region_pivot = region_crop_data.pivot_table(
                    index="year",
                    columns="crop",
                    values="price_per_kg",
                    aggfunc="first",
                ).reindex(index=all_years, columns=all_crops)

                region_pivot["ISO3"] = ISO3
                # Store pivoted data in dictionary with region_id as key
                donor_data[ISO3] = region_pivot

            # Concatenate all regional data into a single DataFrame with MultiIndex
            donor_data = pd.concat(donor_data, names=["ISO3", "year"])

            # Drop crops with no data at all for these regions
            donor_data = donor_data.dropna(axis=1, how="all")

            # Filter out columns that contain the word 'meat'
            donor_data = donor_data[
                [
                    column
                    for column in donor_data.columns
                    if "meat" not in column.lower()
                ]
            ]

            national_data = False
            # Check whether there is national or subnational data
            duplicates = donor_data.index.duplicated(keep=False)
            if duplicates.any():
                # Data is subnational
                unique_regions = self.geoms["areamaps/regions"]
            else:
                # Data is national
                unique_regions = (
                    self.geoms["areamaps/regions"].groupby("ISO3").first().reset_index()
                )
                national_data = True

            data = self.donate_and_receive_crop_prices(
                donor_data, unique_regions, GLOBIOM_regions
            )

            prices_plus_crop_price_inflation = self.determine_price_variability(
                data, unique_regions
            )

            # combine and rename crops
            all_crop_names_model = [
                d["name"] for d in self.dict["crops/crop_data"]["data"].values()
            ]
            for crop_name in all_crop_names_model:
                if (
                    translate_crop_names is not None
                    and crop_name in translate_crop_names
                ):
                    sub_crops = [
                        crop
                        for crop in translate_crop_names[crop_name]
                        if crop in data.columns
                    ]
                    if sub_crops:
                        data[crop_name] = data[sub_crops].mean(axis=1, skipna=True)
                    else:
                        data[crop_name] = np.nan
                        self.logger.warning(
                            f"No crop price data available for crop {crop_name}"
                        )
                else:
                    if crop_name not in data.columns:
                        data[crop_name] = np.nan
                        self.logger.warning(
                            f"No crop price data available for crop {crop_name}"
                        )

            # Extract the crop names from the dictionary and convert them to lowercase
            crop_names = [
                crop["name"].lower()
                for idx, crop in self.dict["crops/crop_data"]["data"].items()
            ]

            # Filter the columns of the data DataFrame
            data = data[
                [
                    col
                    for col in data.columns
                    if col.lower() in crop_names or col == "_crop_price_inflation"
                ]
            ]

            data = self.inter_and_extrapolate_prices(
                prices_plus_crop_price_inflation, unique_regions
            )

            total_years = data.index.get_level_values("year").unique()

            if project_past_until_year:
                assert total_years[0] > project_past_until_year, (
                    f"Extrapolation targets must not fall inside available data time series. Current lower limit is {total_years[0]}"
                )
            if project_future_until_year:
                assert total_years[-1] < project_future_until_year, (
                    f"Extrapolation targets must not fall inside available data time series. Current upper limit is {total_years[-1]}"
                )

            if project_past_until_year or project_future_until_year:
                data = self.process_additional_years(
                    costs=data,
                    total_years=total_years,
                    unique_regions=unique_regions,
                    lower_bound=project_past_until_year,
                    upper_bound=project_future_until_year,
                )

            # Create a dictionary structure with regions as keys and crops as nested dictionaries
            # This is the required format for crop_farmers.py
            crop_data = self.dict["crops/crop_data"]["data"]
            formatted_data = {
                "type": "time_series",
                "data": {},
                "time": data.index.get_level_values("year")
                .unique()
                .tolist(),  # Extract unique years for the time key
            }

            # If national_data is True, create a mapping from ISO3 code to representative region_id
            if national_data:
                unique_regions = data.index.get_level_values("region_id").unique()
                iso3_codes = (
                    self.geoms["areamaps/regions"]
                    .set_index("region_id")
                    .loc[unique_regions]["ISO3"]
                )
                iso3_to_representative_region_id = dict(zip(iso3_codes, unique_regions))

            for _, region in self.geoms["areamaps/regions"].iterrows():
                region_dict = {}
                region_id = region["region_id"]
                region_iso3 = region["ISO3"]

                # Determine the region_id to use based on national_data
                if national_data:
                    # Use the representative region_id for this ISO3 code
                    selected_region_id = iso3_to_representative_region_id.get(
                        region_iso3
                    )
                else:
                    # Use the actual region_id
                    selected_region_id = region_id

                # Fetch the data for the selected region_id
                if selected_region_id in data.index.get_level_values("region_id"):
                    region_data = data.loc[selected_region_id]
                else:
                    # If data is not available for the region, fill with NaNs
                    region_data = pd.DataFrame(
                        np.nan, index=formatted_data["time"], columns=data.columns
                    )

                region_data.index.name = "year"  # Ensure index name is 'year'

                # Ensuring all crops are present according to the crop_data keys
                for crop_id, crop_info in crop_data.items():
                    crop_name = crop_info["name"]

                    if crop_name.endswith("_flood") or crop_name.endswith("_drought"):
                        crop_name = crop_name.rsplit("_", 1)[0]

                    if crop_name in region_data.columns:
                        region_dict[str(crop_id)] = region_data[crop_name].tolist()
                    else:
                        # Add NaN entries for the entire time period if crop is not present in the region data
                        region_dict[str(crop_id)] = [np.nan] * len(
                            formatted_data["time"]
                        )

                formatted_data["data"][str(region_id)] = region_dict

            data = formatted_data.copy()

        # data is a file path
        elif isinstance(crop_prices, str):
            crop_prices = Path(crop_prices)
            if not crop_prices.exists():
                raise ValueError(f"file {crop_prices.resolve()} does not exist")
            with open(crop_prices) as f:
                data = json.load(f)
            data = pd.DataFrame(
                {
                    crop_id: data["crops"][crop_data["name"]]
                    for crop_id, crop_data in self.dict["crops/crop_data"][
                        "data"
                    ].items()
                },
                index=pd.to_datetime(data["time"]),
            )
            # compute mean price per year, using start day as index
            data = data.resample("AS").mean()
            # extend dataframe to include start and end years
            data = data.reindex(
                index=pd.date_range(
                    start=datetime(project_past_until_year, 1, 1),
                    end=datetime(project_future_until_year, 1, 1),
                    freq="YS",
                )
            )
            # only use year identifier as index
            data.index = data.index.year

            data = data.reindex(
                index=pd.MultiIndex.from_product(
                    [
                        self.geoms["areamaps/regions"]["region_id"],
                        data.index,
                    ],
                    names=["region_id", "date"],
                ),
                level=1,
            )

            data = self.determine_price_variability(
                data, self.geoms["areamaps/regions"]
            )

            data = self.inter_and_extrapolate_prices(
                data, self.geoms["areamaps/regions"]
            )

            data = {
                "type": "time_series",
                "time": data.xs(
                    data.index.get_level_values(0)[0], level=0
                ).index.tolist(),
                "data": {
                    str(region_id): data.loc[region_id].to_dict(orient="list")
                    for region_id in self.geoms["areamaps/regions"]["region_id"]
                },
            }

        elif isinstance(crop_prices, (int, float)):
            data = {
                "type": "constant",
                "data": crop_prices,
            }
        else:
            raise ValueError(
                f"must be a file path or an integer, got {type(crop_prices)}"
            )

        return data

    def donate_and_receive_crop_prices(
        self, donor_data, recipient_regions, GLOBIOM_regions
    ):
        """
        If there are multiple countries in one selected basin, where one country has prices for a certain crop, but the other does not,
        this gives issues. This function adjusts crop data for those countries by filling in missing values using data from nearby regions
        and PPP conversion rates.

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing crop data with a 'ISO3' column and indexed by 'region_id'. The DataFrame
            contains crop prices for different regions.

        Returns
        -------
        DataFrame
            The updated DataFrame with missing crop data filled in using PPP conversion rates from nearby regions.

        Notes
        -----
        The function performs the following steps:
        1. Identifies columns where all values are NaN for each country and stores this information.
        2. For each country and column with missing values, finds a country/region within that study area that has data for that column.
        3. Uses PPP conversion rates to adjust and fill in missing values for regions without data.
        4. Drops the 'ISO3' column before returning the updated DataFrame.
        """

        # create a copy of the data to avoid using data that was adjusted in this function
        data_out = None

        for _, region in recipient_regions.iterrows():
            ISO3 = region["ISO3"]
            region_id = region["region_id"]
            self.logger.debug(f"Processing region {region_id}")
            # Filter the data for the current country
            country_data = donor_data[donor_data["ISO3"] == ISO3]

            GLOBIOM_region = GLOBIOM_regions.loc[
                GLOBIOM_regions["ISO3"] == ISO3, "Region37"
            ].item()
            GLOBIOM_region_countries = GLOBIOM_regions.loc[
                GLOBIOM_regions["Region37"] == GLOBIOM_region, "ISO3"
            ]

            for column in country_data.columns:
                if country_data[column].isna().all():
                    donor_data_region = donor_data.loc[
                        donor_data["ISO3"].isin(GLOBIOM_region_countries), column
                    ]

                    # get the country with the least non-NaN values
                    non_na_values = donor_data_region.groupby("ISO3").count()

                    if non_na_values.max() == 0:
                        continue

                    donor_country = non_na_values.idxmax()
                    donor_data_country = donor_data_region[donor_country]

                    new_data = pd.DataFrame(
                        donor_data_country.values,
                        index=pd.MultiIndex.from_product(
                            [[region["region_id"]], donor_data_country.index],
                            names=["region_id", "year"],
                        ),
                        columns=[donor_data_country.name],
                    )
                    if data_out is None:
                        data_out = new_data.copy()
                    else:
                        data_out = data_out.combine_first(new_data)
                else:
                    new_data = pd.DataFrame(
                        country_data[column].values,
                        index=pd.MultiIndex.from_product(
                            [
                                [region["region_id"]],
                                country_data.droplevel(level=0).index,
                            ],
                            names=["region_id", "year"],
                        ),
                        columns=[column],
                    )
                    if data_out is None:
                        data_out = new_data.copy()
                    else:
                        data_out = data_out.combine_first(new_data)

        data_out = data_out.drop(columns=["ISO3"])
        data_out = data_out.dropna(axis=1, how="all")
        data_out = data_out.dropna(axis=0, how="all")

        return data_out

    def determine_price_variability(self, costs, unique_regions):
        """
        Determines the price variability of all crops in the region and adds a column that describes this variability.

        Parameters
        ----------
        costs : DataFrame
            A DataFrame containing the cost data for different regions. The DataFrame should be indexed by region IDs.

        Returns
        -------
        DataFrame
            The updated DataFrame with a new column 'changes' that contains the average price changes for each region.
        """
        costs["_crop_price_inflation"] = np.nan
        # Determine the average changes of price of all crops in the region and add it to the data
        for _, region in unique_regions.iterrows():
            region_id = region["region_id"]
            region_data = costs.loc[region_id]
            changes = np.nanmean(
                region_data[1:].to_numpy() / region_data[:-1].to_numpy(), axis=1
            )

            changes = np.insert(changes, 0, np.nan)
            costs.at[region_id, "_crop_price_inflation"] = changes

            years_with_no_crop_inflation_data = costs.loc[
                region_id, "_crop_price_inflation"
            ]
            region_inflation_rates = self.dict["economics/inflation_rates"]["data"][
                str(region["region_id"])
            ]

            for year, crop_inflation_rate in years_with_no_crop_inflation_data.items():
                if np.isnan(crop_inflation_rate):
                    year_inflation_rate = region_inflation_rates[
                        self.dict["economics/inflation_rates"]["time"].index(str(year))
                    ]
                    costs.at[(region_id, year), "_crop_price_inflation"] = (
                        year_inflation_rate
                    )

        return costs

    def inter_and_extrapolate_prices(self, data, unique_regions):
        """
        Interpolates and extrapolates crop prices for different regions based on the given data and predefined crop categories.

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing crop price data for different regions. The DataFrame should be indexed by region IDs
            and have columns corresponding to different crops.

        Returns
        -------
        DataFrame
            The updated DataFrame with interpolated and extrapolated crop prices. Columns for 'others perennial' and 'others annual'
            crops are also added.

        Notes
        -----
        The function performs the following steps:
        1. Extracts crop names from the internal crop data dictionary.
        2. Defines additional crops that fall under 'others perennial' and 'others annual' categories.
        3. Processes the data to compute average prices for these additional crops.
        4. Filters and updates the original data with the computed averages.
        5. Interpolates and extrapolates missing prices for each crop in each region based on the 'changes' column.
        """

        # Interpolate and extrapolate missing prices for each crop in each region based on the 'changes' column
        for _, region in unique_regions.iterrows():
            region_id = region["region_id"]
            region_data = data.loc[region_id]

            n = len(region_data)
            for crop in region_data.columns:
                if crop == "_crop_price_inflation":
                    continue
                crop_data = region_data[crop].to_numpy()
                if np.isnan(crop_data).all():
                    continue
                changes_data = region_data["_crop_price_inflation"].to_numpy()
                k = -1
                while np.isnan(crop_data[k]):
                    k -= 1
                for i in range(k + 1, 0, 1):
                    crop_data[i] = crop_data[i - 1] * changes_data[i]
                k = 0
                while np.isnan(crop_data[k]):
                    k += 1
                for i in range(k - 1, -1, -1):
                    crop_data[i] = crop_data[i + 1] / changes_data[i + 1]
                for j in range(0, n):
                    if np.isnan(crop_data[j]):
                        k = j
                        while np.isnan(crop_data[k]):
                            k += 1
                        empty_size = k - j
                        step_crop_price_inflation = changes_data[j : k + 1]
                        total_crop_price_inflation = np.prod(step_crop_price_inflation)
                        real_crop_price_inflation = crop_data[k] / crop_data[j - 1]
                        scaled_crop_price_inflation = (
                            step_crop_price_inflation
                            * (real_crop_price_inflation ** (1 / empty_size))
                            / (total_crop_price_inflation ** (1 / empty_size))
                        )
                        for i, change in zip(range(j, k), scaled_crop_price_inflation):
                            crop_data[i] = crop_data[i - 1] * change
                data.loc[region_id, crop] = crop_data

        # assert no nan values in costs
        data = data.drop(columns=["_crop_price_inflation"])
        return data

    def process_additional_years(
        self, costs, total_years, unique_regions, lower_bound=None, upper_bound=None
    ):
        inflation = self.dict["economics/inflation_rates"]
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

    def setup_cultivation_costs(
        self,
        cultivation_costs: Optional[Union[str, int, float]] = 0,
        project_future_until_year: Optional[int] = False,
        project_past_until_year: Optional[int] = False,
        translate_crop_names: Optional[Dict[str, str]] = None,
    ):
        """
        Sets up the cultivation costs for the model.

        Parameters
        ----------
        cultivation_costs : str or int or float, optional
            The file path or integer of cultivation costs. If a file path is provided, the file is loaded and parsed as JSON.
            The dictionary should have a 'time' key with a list of time steps, and a 'crops' key with a dictionary of crop
            IDs and their cultivation costs. If .
        """
        self.logger.info("Preparing cultivation costs")
        cultivation_costs = self.process_crop_data(
            crop_prices=cultivation_costs,
            project_future_until_year=project_future_until_year,
            project_past_until_year=project_past_until_year,
            translate_crop_names=translate_crop_names,
        )
        self.set_dict(cultivation_costs, name="crops/cultivation_costs")

    def setup_crop_prices(
        self,
        crop_prices: Optional[Union[str, int, float]] = "FAO_stat",
        project_future_until_year: Optional[int] = False,
        project_past_until_year: Optional[int] = False,
        translate_crop_names: Optional[Dict[str, str]] = None,
    ):
        """
        Sets up the crop prices for the model.

        Parameters
        ----------
        crop_prices : str or int or float, optional
            The file path or integer of crop prices. If a file path is provided, the file is loaded and parsed as JSON.
            The dictionary should have a 'time' key with a list of time steps, and a 'crops' key with a dictionary of crop
            IDs and their prices.
        """
        self.logger.info("Preparing crop prices")
        crop_prices = self.process_crop_data(
            crop_prices=crop_prices,
            project_future_until_year=project_future_until_year,
            project_past_until_year=project_past_until_year,
            translate_crop_names=translate_crop_names,
        )
        self.set_dict(crop_prices, name="crops/crop_prices")
        self.set_dict(crop_prices, name="crops/cultivation_costs")

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
        a = (2 * self.grid["areamaps/cell_area"]) / self.grid["routing/upstream_area"]
        a = xr.where(a < 1, a, 1, keep_attrs=True)
        b = self.grid["routing/outflow_elevation"] / 2000
        b = xr.where(b < 1, b, 1, keep_attrs=True)

        mannings = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            crs=self.crs,
            name="routing/mannings",
            lazy=True,
        )
        mannings.data = 0.025 + 0.015 * a + 0.030 * b
        self.set_grid(mannings, mannings.name)

    def setup_river_width(self, minimum_width: float) -> None:
        raise ValueError("setup_river_width not needed anymore, please remove")

    def setup_channel_depth(self) -> None:
        raise ValueError("setup_channel_depth not needed anymore, please remove")

    def setup_channel_ratio(self) -> None:
        raise ValueError("setup_channel_ratio not needed anymore, please remove")

    def setup_elevation(self) -> None:
        raise ValueError("setup_elevation not needed anymore, please remove")

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
        (
            sand,
            silt,
            clay,
            bulk_density,
            soil_organic_carbon,
            soil_layer_height,
        ) = load_soilgrids(self.data_catalog, self.subgrid, self.region)

        self.set_subgrid(sand, name="soil/sand")
        self.set_subgrid(silt, name="soil/silt")
        self.set_subgrid(clay, name="soil/clay")
        self.set_subgrid(bulk_density, name="soil/bulk_density")
        self.set_subgrid(soil_organic_carbon, name="soil/soil_organic_carbon")
        self.set_subgrid(soil_layer_height, name="soil/soil_layer_height")

        soil_ds = self.data_catalog.get_rasterdataset(
            "cwatm_soil_5min", bbox=self.bounds, buffer=10
        )

        ds = soil_ds["cropgrp"]
        self.set_grid(self.interpolate(ds, "linear"), name="soil/cropgrp")

    def setup_land_use_parameters(self, interpolation_method="nearest") -> None:
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
        for land_use_type, land_use_type_netcdf_name in (
            ("forest", "Forest"),
            ("grassland", "Grassland"),
            ("irrPaddy", "irrPaddy"),
            ("irrNonPaddy", "irrNonPaddy"),
        ):
            self.logger.info(f"Setting up land use parameters for {land_use_type}")
            land_use_ds = self.data_catalog.get_rasterdataset(
                f"cwatm_{land_use_type}_5min", bbox=self.bounds, buffer=10
            )

            parameter = f"cropCoefficient{land_use_type_netcdf_name}_10days"
            self.set_forcing(
                self.interpolate(land_use_ds[parameter], interpolation_method),
                name=f"landcover/{land_use_type}/{parameter}",
            )
            if land_use_type in ("forest", "grassland"):
                parameter = f"interceptCap{land_use_type_netcdf_name}_10days"
                self.set_forcing(
                    self.interpolate(land_use_ds[parameter], interpolation_method),
                    name=f"landcover/{land_use_type}/{parameter}",
                )

    def setup_waterbodies(
        self,
        command_areas="reservoir_command_areas",
        custom_reservoir_capacity="custom_reservoir_capacity",
    ):
        """
        Sets up the waterbodies for GEB.

        Notes
        -----
        This method sets up the waterbodies for GEB. It first retrieves the waterbody data from the
        specified data catalog and sets it as a geometry in the model. It then rasterizes the waterbody data onto the model
        grid and the subgrid using the `rasterize` method of the `raster` object. The resulting grids are set as attributes
        of the model with names of the form 'routing/lakesreservoirs/{grid_name}'.

        The method also retrieves the reservoir command area data from the data catalog and calculates the area of each
        command area that falls within the model region. The `waterbody_id` key is used to do the matching between these
        databases. The relative area of each command area within the model region is calculated and set as a column in
        the waterbody data. The method sets all lakes with a command area to be reservoirs and updates the waterbody data
        with any custom reservoir capacity data from the data catalog.

        TODO: Make the reservoir command area data optional.

        The resulting waterbody data is set as a table in the model with the name 'routing/lakesreservoirs/basin_lakes_data'.
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
        except (IndexError, NoDataException):
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
                crs=self.crs,
            )
            waterbodies = waterbodies.astype(dtypes)
            lakesResID = xr.zeros_like(self.grid["areamaps/grid_mask"])
            sublakesResID = xr.zeros_like(self.subgrid["areamaps/sub_grid_mask"])

            # When hydroMT 1.0 is released this should not be needed anymore
            sublakesResID.raster.set_crs(self.subgrid.raster.crs)
        else:
            lakesResID = self.grid.raster.rasterize(
                waterbodies,
                col_name="waterbody_id",
                nodata=0,
                all_touched=True,
                dtype=np.int32,
            )
            sublakesResID = self.subgrid.raster.rasterize(
                waterbodies,
                col_name="waterbody_id",
                nodata=0,
                all_touched=True,
                dtype=np.int32,
            )

        self.set_grid(lakesResID, name="routing/lakesreservoirs/lakesResID")
        self.set_subgrid(sublakesResID, name="routing/lakesreservoirs/sublakesResID")

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
                name="routing/lakesreservoirs/command_areas",
            )
            self.set_subgrid(
                self.subgrid.raster.rasterize(
                    command_areas,
                    col_name="waterbody_id",
                    nodata=-1,
                    all_touched=True,
                    dtype=np.int32,
                ),
                name="routing/lakesreservoirs/subcommand_areas",
            )

            # set all lakes with command area to reservoir
            waterbodies.loc[
                waterbodies.index.isin(command_areas["waterbody_id"]), "waterbody_type"
            ] = 2
        else:
            command_areas = hydromt.raster.full(
                self.grid.raster.coords,
                nodata=-1,
                dtype=np.int32,
                name="routing/lakesreservoirs/command_areas",
                crs=self.grid.raster.crs,
            )
            command_areas[:] = -1
            subcommand_areas = hydromt.raster.full(
                self.subgrid.raster.coords,
                nodata=-1,
                dtype=np.int32,
                name="routing/lakesreservoirs/subcommand_areas",
                crs=self.subgrid.raster.crs,
            )
            subcommand_areas[:] = -1
            self.set_grid(command_areas, name="routing/lakesreservoirs/command_areas")
            self.set_subgrid(
                subcommand_areas, name="routing/lakesreservoirs/subcommand_areas"
            )

        if custom_reservoir_capacity:
            custom_reservoir_capacity = self.data_catalog.get_dataframe(
                "custom_reservoir_capacity"
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
        self.set_table(waterbodies, name="routing/lakesreservoirs/basin_lakes_data")

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
            ds.name = name
            self.set_forcing(
                ds.rename({"lat": "y", "lon": "x"}), name=f"water_demand/{name}"
            )

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

        aquifer_top_elevation = (
            self.grid["landsurface/topo/elevation"].raster.mask_nodata().compute()
        )
        self.set_grid(aquifer_top_elevation, name="groundwater/aquifer_top_elevation")

        # load total thickness
        total_thickness = (
            self.data_catalog.get_rasterdataset(
                "total_groundwater_thickness_globgm",
                bbox=self.bounds,
                buffer=2,
            )
            .rename({"lon": "x", "lat": "y"})
            .compute()
        )

        total_thickness = np.clip(
            total_thickness,
            minimum_thickness_confined_layer,
            maximum_thickness_confined_layer,
        )

        confining_layer = (
            self.data_catalog.get_rasterdataset(
                "thickness_confining_layer_globgm",
                bbox=self.bounds,
                buffer=2,
            )
            .rename({"lon": "x", "lat": "y"})
            .compute()
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
                    xr.full_like(relative_bottom_bottom_layer, 0),
                    relative_bottom_top_layer,
                    relative_bottom_bottom_layer,
                ],
                dim="boundary",
                compat="equals",
            ).compute()
        else:
            relative_bottom_bottom_layer = -total_thickness
            relative_layer_boundary_elevation = xr.concat(
                [
                    xr.full_like(relative_bottom_bottom_layer, 0),
                    relative_bottom_bottom_layer,
                ],
                dim="boundary",
                compat="equals",
            ).compute()

        layer_boundary_elevation = (
            relative_layer_boundary_elevation.raster.reproject_like(
                aquifer_top_elevation, method="bilinear"
            )
        ) + aquifer_top_elevation

        self.set_grid(
            layer_boundary_elevation, name="groundwater/layer_boundary_elevation"
        )

        # load hydraulic conductivity
        hydraulic_conductivity = (
            self.data_catalog.get_rasterdataset(
                "hydraulic_conductivity_globgm",
                bbox=self.bounds,
                buffer=2,
            )
            .rename({"lon": "x", "lat": "y"})
            .compute()
        )

        # because
        hydraulic_conductivity_log = np.log(hydraulic_conductivity)
        hydraulic_conductivity_log = hydraulic_conductivity_log.raster.reproject_like(
            aquifer_top_elevation, method="bilinear"
        )
        hydraulic_conductivity = np.exp(hydraulic_conductivity_log)

        if two_layers:
            hydraulic_conductivity = xr.concat(
                [hydraulic_conductivity, hydraulic_conductivity],
                dim="layer",
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
                [specific_yield, specific_yield], dim="layer", compat="equals"
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
            dem_globgm = (
                self.data_catalog.get_rasterdataset(
                    dem_globgm, geom=self.region, variables=["dem_average"], buffer=2
                )
                .rename({"lon": "x", "lat": "y"})
                .compute()
            )
            # load digital elevation model that was used for globgm

            dem = self.grid["landsurface/topo/elevation"].raster.mask_nodata()

            # heads
            head_upper_layer = self.data_catalog.get_rasterdataset(
                "head_upper_globgm",
                bbox=self.bounds,
                buffer=2,
            ).compute()

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
            ).compute()
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
                    [head_upper_layer, head_lower_layer], dim="layer", compat="equals"
                )
            else:
                heads = head_lower_layer.expand_dims(layer=["upper"])

        elif intial_heads_source == "Fan":
            # Load in the starting groundwater depth
            region_continent = np.unique(self.geoms["areamaps/regions"]["CONTINENT"])
            assert (
                np.size(region_continent) == 1
            )  # Transcontinental basins should not be possible

            if (
                np.unique(self.geoms["areamaps/regions"]["CONTINENT"])[0] == "Asia"
                or np.unique(self.geoms["areamaps/regions"]["CONTINENT"])[0] == "Europe"
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
        self.set_grid(heads, name="groundwater/heads")

    def setup_forcing(
        self,
        starttime: date,
        endtime: date,
        data_source: str = "isimip",
        resolution_arcsec: int = 30,
        forcing: str = "chelsa-w5e5",
        ssp=None,
    ):
        """
        Sets up the forcing data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        data_source : str, optional
            The data source to use for the forcing data. Default is 'isimip'.

        Notes
        -----
        This method sets up the forcing data for GEB. It first downloads the high-resolution variables
        (precipitation, surface solar radiation, air temperature, maximum air temperature, and minimum air temperature) from
        the ISIMIP dataset for the specified time period. The data is downloaded using the `setup_30arcsec_variables_isimip`
        method.

        The method then sets up the relative humidity, longwave radiation, pressure, and wind data for the model. The
        relative humidity data is downloaded from the ISIMIP dataset using the `setup_hurs_isimip_30arcsec` method. The longwave radiation
        data is calculated using the air temperature and relative humidity data and the `calculate_longwave` function. The
        pressure data is downloaded from the ISIMIP dataset using the `setup_pressure_isimip_30arcsec` method. The wind data is downloaded
        from the ISIMIP dataset using the `setup_wind_isimip_30arcsec` method. All these data are first downscaled to the model grid.

        The resulting forcing data is set as forcing data in the model with names of the form 'forcing/{variable_name}'.
        """
        assert starttime < endtime, "Start time must be before end time"

        if data_source == "isimip":
            if resolution_arcsec == 30:
                assert forcing == "chelsa-w5e5", (
                    "Only chelsa-w5e5 is supported for 30 arcsec resolution"
                )
                # download source data from ISIMIP
                self.logger.info("setting up forcing data")
                high_res_variables = ["pr", "rsds", "tas", "tasmax", "tasmin"]
                self.setup_30arcsec_variables_isimip(
                    high_res_variables, starttime, endtime
                )
                self.logger.info("setting up relative humidity...")
                self.setup_hurs_isimip_30arcsec(starttime, endtime)
                self.logger.info("setting up longwave radiation...")
                self.setup_longwave_isimip_30arcsec(
                    starttime=starttime, endtime=endtime
                )
                self.logger.info("setting up pressure...")
                self.setup_pressure_isimip_30arcsec(starttime, endtime)
                self.logger.info("setting up wind...")
                self.setup_wind_isimip_30arcsec(starttime, endtime)
            elif resolution_arcsec == 1800:
                variables = [
                    "pr",
                    "rsds",
                    "tas",
                    "tasmax",
                    "tasmin",
                    "hurs",
                    "rlds",
                    "ps",
                    "sfcwind",
                ]
                self.setup_1800arcsec_variables_isimip(
                    forcing, variables, starttime, endtime, ssp=ssp
                )
            else:
                raise ValueError(
                    "Only 30 arcsec and 1800 arcsec resolution is supported for ISIMIP data"
                )
        elif data_source == "era5":
            # # Create a thread pool and map the set_forcing function to the variables
            # # Wait for all threads to complete
            # concurrent.futures.wait(futures)
            mask = self.grid["areamaps/grid_mask"]

            files = download_ERA5(
                folder=self.preprocessing_dir / "climate" / "ERA5",
                variables=[
                    "total_precipitation",
                    "surface_solar_radiation_downwards",
                    "surface_thermal_radiation_downwards",
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "surface_pressure",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                starttime=starttime,
                endtime=endtime,
                bounds=mask.raster.bounds,
                logger=self.logger,
            )

            pr_hourly = open_ERA5(
                files,
                "tp",  # total_precipitation
                xy_chunksize=XY_CHUNKSIZE,
            )
            pr_hourly = pr_hourly * (1000 / 3600)  # convert from m/hr to kg/m2/s
            pr_hourly.attrs = {
                "standard_name": "precipitation_flux",
                "long_name": "Precipitation",
                "units": "kg m-2 s-1",
            }
            # ensure no negative values for precipitation, which may arise due to float precision
            pr_hourly = xr.where(pr_hourly > 0, pr_hourly, 0, keep_attrs=True)
            pr_hourly.name = "pr_hourly"
            self.set_forcing(
                pr_hourly, name="climate/pr_hourly", time_chunksize=7 * 24
            )  # weekly chunk size
            pr = pr_hourly.resample(time="D").mean()  # get daily mean
            pr = pr.raster.reproject_like(mask, method="average")
            pr.name = "pr"
            self.set_forcing(pr, name="climate/pr")

            hourly_rsds = open_ERA5(
                files,
                "ssrd",  # surface_solar_radiation_downwards
                xy_chunksize=XY_CHUNKSIZE,
            )
            rsds = hourly_rsds.resample(time="D").sum() / (
                24 * 3600
            )  # get daily sum and convert from J/m2 to W/m2
            rsds.attrs = {
                "standard_name": "surface_downwelling_shortwave_flux_in_air",
                "long_name": "Surface Downwelling Shortwave Radiation",
                "units": "W m-2",
            }

            rsds = rsds.raster.reproject_like(mask, method="average")
            rsds.name = "rsds"
            self.set_forcing(rsds, name="climate/rsds")

            hourly_rlds = open_ERA5(
                files,
                "strd",  # surface_thermal_radiation_downwards
                xy_chunksize=XY_CHUNKSIZE,
            )
            rlds = hourly_rlds.resample(time="D").sum() / (24 * 3600)
            rlds.attrs = {
                "standard_name": "surface_downwelling_longwave_flux_in_air",
                "long_name": "Surface Downwelling Longwave Radiation",
                "units": "W m-2",
            }
            rlds = rlds.raster.reproject_like(mask, method="average")
            rlds.name = "rlds"
            self.set_forcing(rlds, name="climate/rlds")

            hourly_tas = open_ERA5(files, "t2m", xy_chunksize=XY_CHUNKSIZE)

            DEM = self.data_catalog.get_rasterdataset(
                "fabdem",
                bbox=hourly_tas.raster.bounds,
                buffer=100,
                variables=["fabdem"],
            )

            hourly_tas_reprojected = reproject_and_apply_lapse_rate_temperature(
                hourly_tas, DEM, mask
            )

            tas_reprojected = hourly_tas_reprojected.resample(time="D").mean()
            tas_reprojected.attrs = {
                "standard_name": "air_temperature",
                "long_name": "Near-Surface Air Temperature",
                "units": "K",
            }
            tas_reprojected.name = "tas"
            self.set_forcing(tas_reprojected, name="climate/tas", byteshuffle=True)

            tasmax = hourly_tas_reprojected.resample(time="D").max()
            tasmax.attrs = {
                "standard_name": "air_temperature",
                "long_name": "Daily Maximum Near-Surface Air Temperature",
                "units": "K",
            }
            tasmax.name = "tasmax"
            self.set_forcing(tasmax, name="climate/tasmax", byteshuffle=True)

            tasmin = hourly_tas_reprojected.resample(time="D").min()
            tasmin.attrs = {
                "standard_name": "air_temperature",
                "long_name": "Daily Minimum Near-Surface Air Temperature",
                "units": "K",
            }
            tasmin.name = "tasmin"
            self.set_forcing(tasmin, name="climate/tasmin", byteshuffle=True)

            dew_point_tas = open_ERA5(
                files,
                "d2m",
                xy_chunksize=XY_CHUNKSIZE,
            )
            dew_point_tas_reprojected = reproject_and_apply_lapse_rate_temperature(
                dew_point_tas, DEM, mask
            )

            water_vapour_pressure = 0.6108 * np.exp(
                17.27
                * (dew_point_tas_reprojected - 273.15)
                / (237.3 + (dew_point_tas_reprojected - 273.15))
            )  # calculate water vapour pressure (kPa)
            saturation_vapour_pressure = 0.6108 * np.exp(
                17.27
                * (hourly_tas_reprojected - 273.15)
                / (237.3 + (hourly_tas_reprojected - 273.15))
            )

            assert water_vapour_pressure.shape == saturation_vapour_pressure.shape
            relative_humidity = (
                water_vapour_pressure / saturation_vapour_pressure
            ) * 100
            relative_humidity.attrs = {
                "standard_name": "relative_humidity",
                "long_name": "Near-Surface Relative Humidity",
                "units": "%",
            }
            relative_humidity = relative_humidity.resample(time="D").mean()
            relative_humidity = relative_humidity.raster.reproject_like(
                mask, method="average"
            )
            relative_humidity.name = "hurs"
            self.set_forcing(relative_humidity, name="climate/hurs", byteshuffle=True)

            pressure = open_ERA5(files, "sp", xy_chunksize=XY_CHUNKSIZE)
            pressure = reproject_and_apply_lapse_rate_pressure(pressure, DEM, mask)
            pressure.attrs = {
                "standard_name": "surface_air_pressure",
                "long_name": "Surface Air Pressure",
                "units": "Pa",
            }
            pressure = pressure.resample(time="D").mean()
            pressure.name = "ps"
            self.set_forcing(pressure, name="climate/ps", byteshuffle=True)

            u_wind = open_ERA5(
                files,
                "u10",
                xy_chunksize=XY_CHUNKSIZE,
            )
            u_wind = u_wind.resample(time="D").mean()

            v_wind = open_ERA5(
                files,
                "v10",
                xy_chunksize=XY_CHUNKSIZE,
            )
            v_wind = v_wind.resample(time="D").mean()
            wind_speed = np.sqrt(u_wind**2 + v_wind**2)
            wind_speed.attrs = {
                "standard_name": "wind_speed",
                "long_name": "Near-Surface Wind Speed",
                "units": "m s-1",
            }
            wind_speed = wind_speed.raster.reproject_like(mask, method="average")
            wind_speed.name = "sfcwind"
            self.set_forcing(wind_speed, name="climate/sfcwind", byteshuffle=True)

        elif data_source == "cmip":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(f"Unknown data source: {data_source}")

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

    def setup_1800arcsec_variables_isimip(
        self,
        forcing: str,
        variables: List[str],
        starttime: date,
        endtime: date,
        ssp: str,
    ):
        """
        Sets up the high-resolution climate variables for GEB.

        Parameters
        ----------
        variables : list of str
            The list of climate variables to set up.
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the high-resolution climate variables for GEB. It downloads the specified
        climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
        `download_isimip` method.

        The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
        then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

        The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """

        def download_variable(variable_name, forcing, ssp, starttime, endtime):
            self.logger.info(f"Setting up {variable_name}...")
            first_year_future_climate = 2015
            var = []
            if ssp == "picontrol":
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario=ssp,
                    variable=variable_name,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(ds[variable_name].raster.clip_bbox(ds.raster.bounds))
            if (
                (
                    endtime.year < first_year_future_climate
                    or starttime.year < first_year_future_climate
                )
                and ssp != "picontrol"
            ):  # isimip cutoff date between historic and future climate
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario="historical",
                    variable=variable_name,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(ds[variable_name].raster.clip_bbox(ds.raster.bounds))
            if (
                starttime.year >= first_year_future_climate
                or endtime.year >= first_year_future_climate
            ) and ssp != "picontrol":
                assert ssp is not None, "ssp must be specified for future climate"
                assert ssp != "historical", "historical scenarios run until 2014"
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario=ssp,
                    variable=variable_name,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(ds[variable_name].raster.clip_bbox(ds.raster.bounds))

            var = xr.concat(
                var, dim="time", combine_attrs="drop_conflicts", compat="equals"
            )  # all values and dimensions must be the same

            # assert that time is monotonically increasing with a constant step size
            assert (
                ds.time.diff("time").astype(np.int64)
                == (ds.time[1] - ds.time[0]).astype(np.int64)
            ).all(), "time is not monotonically increasing with a constant step size"

            var = var.rename({"lon": "x", "lat": "y"})
            if variable_name in ("tas", "tasmin", "tasmax", "ps"):
                byteshuffle = True
                DEM = self.data_catalog.get_rasterdataset(
                    "fabdem",
                    bbox=var.raster.bounds,
                    buffer=100,
                    variables=["fabdem"],
                )
                if variable_name in ("tas", "tasmin", "tasmax"):
                    var = reproject_and_apply_lapse_rate_temperature(
                        var, DEM, self.grid["areamaps/grid_mask"]
                    )
                elif variable_name == "ps":
                    var = reproject_and_apply_lapse_rate_pressure(
                        var, DEM, self.grid["areamaps/grid_mask"]
                    )
                else:
                    raise ValueError
            else:
                byteshuffle = False
                var = self.interpolate(var, "linear")
            self.logger.info(f"Completed {variable_name}")
            self.set_forcing(
                var, name=f"climate/{variable_name}", byteshuffle=byteshuffle
            )

        for variable in variables:
            download_variable(variable, forcing, ssp, starttime, endtime)

        # # Create a thread pool and map the set_forcing function to the variables
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(download_variable, variable, forcing, ssp, starttime, endtime) for variable in variables]

        # # Wait for all threads to complete
        # concurrent.futures.wait(futures)

    def setup_30arcsec_variables_isimip(
        self, variables: List[str], starttime: date, endtime: date
    ):
        """
        Sets up the high-resolution climate variables for GEB.

        Parameters
        ----------
        variables : list of str
            The list of climate variables to set up.
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the high-resolution climate variables for GEB. It downloads the specified
        climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
        `download_isimip` method.

        The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
        then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

        The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """

        def download_variable(variable, starttime, endtime):
            self.logger.info(f"Setting up {variable}...")
            ds = self.download_isimip(
                product="InputData",
                variable=variable,
                starttime=starttime,
                endtime=endtime,
                forcing="chelsa-w5e5",
                resolution="30arcsec",
            )
            ds = ds.rename({"lon": "x", "lat": "y"})
            var = ds[variable].raster.clip_bbox(ds.raster.bounds)
            var = self.snap_to_grid(var, self.grid)
            self.logger.info(f"Completed {variable}")
            self.set_forcing(var, name=f"climate/{variable}")

        for variable in variables:
            download_variable(variable, starttime, endtime)

        # # Create a thread pool and map the set_forcing function to the variables
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(download_variable, variable, starttime, endtime) for variable in variables]

        # # Wait for all threads to complete
        # concurrent.futures.wait(futures)

    def setup_hurs_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the relative humidity data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the relative humidity data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the relative humidity data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the relative humidity data for GEB. It first downloads the relative humidity
        data from the ISIMIP dataset for the specified time period using the `download_isimip` method. The data is downloaded
        at a 30 arcsec resolution.

        The method then downloads the monthly CHELSA-BIOCLIM+ relative humidity data at 30 arcsec resolution from the data
        catalog. The data is downloaded for each month in the specified time period and is clipped to the bounding box of
        the downloaded relative humidity data using the `clip_bbox` method of the `raster` object.

        The original ISIMIP data is then downscaled using the monthly CHELSA-BIOCLIM+ data. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting relative humidity data is set as forcing data in the model with names of the form 'climate/hurs'.
        """
        hurs_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        )  # some buffer to avoid edge effects / errors in ISIMIP API

        # just taking the years to simplify things
        start_year = starttime.year
        end_year = endtime.year

        chelsa_folder = self.preprocessing_dir / "climate" / "chelsa-bioclim+" / "hurs"
        chelsa_folder.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Downloading/reading monthly CHELSA-BIOCLIM+ hurs data at 30 arcsec resolution"
        )
        hurs_ds_30sec, hurs_time = [], []
        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                fn = chelsa_folder / f"hurs_{year}_{month:02d}.zarr.zip"
                if not fn.exists():
                    hurs = self.data_catalog.get_rasterdataset(
                        f"CHELSA-BIOCLIM+_monthly_hurs_{month:02d}_{year}",
                        bbox=hurs_30_min.raster.bounds,
                        buffer=1,
                    )
                    del hurs.attrs["_FillValue"]
                    hurs.name = "hurs"
                    hurs.to_zarr(fn, mode="w")
                else:
                    hurs = xr.open_dataset(fn, chunks={}, engine="zarr")["hurs"]
                # assert hasattr(hurs, "spatial_ref")
                hurs_ds_30sec.append(hurs)
                hurs_time.append(f"{year}-{month:02d}")

        hurs_ds_30sec = xr.concat(hurs_ds_30sec, dim="time").rename(
            {"x": "lon", "y": "lat"}
        )
        hurs_ds_30sec.rio.set_spatial_dims("lon", "lat", inplace=True)
        hurs_ds_30sec["time"] = pd.date_range(hurs_time[0], hurs_time[-1], freq="MS")

        hurs_output = xr.full_like(self.forcing["climate/tas"], np.nan)
        hurs_output.name = "hurs"
        hurs_output.attrs = {"units": "%", "long_name": "Relative humidity"}

        hurs_output = hurs_output.rename({"x": "lon", "y": "lat"}).rio.set_spatial_dims(
            "lon", "lat"
        )

        import xesmf as xe

        regridder = xe.Regridder(
            hurs_30_min.isel(time=0).drop_vars("time"),
            hurs_ds_30sec.isel(time=0).drop_vars("time"),
            "bilinear",
        )
        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                start_month = datetime(year, month, 1)
                end_month = datetime(year, month, monthrange(year, month)[1])

                w5e5_30min_sel = hurs_30_min.sel(time=slice(start_month, end_month))
                w5e5_regridded = (
                    regridder(w5e5_30min_sel, output_chunks=(-1, -1)) * 0.01
                )  # convert to fraction
                assert (w5e5_regridded >= 0.1).all(), (
                    "too low values in relative humidity"
                )
                assert (w5e5_regridded <= 1).all(), "relative humidity > 1"

                w5e5_regridded_mean = w5e5_regridded.mean(
                    dim="time"
                )  # get monthly mean
                w5e5_regridded_tr = np.log(
                    w5e5_regridded / (1 - w5e5_regridded)
                )  # assume beta distribuation => logit transform
                w5e5_regridded_mean_tr = np.log(
                    w5e5_regridded_mean / (1 - w5e5_regridded_mean)
                )  # logit transform

                chelsa = (
                    hurs_ds_30sec.sel(time=start_month) * 0.0001
                )  # convert to fraction
                assert (chelsa >= 0.1).all(), "too low values in relative humidity"
                assert (chelsa <= 1).all(), "relative humidity > 1"

                chelsa_tr = np.log(
                    chelsa / (1 - chelsa)
                )  # assume beta distribuation => logit transform

                difference = chelsa_tr - w5e5_regridded_mean_tr

                # apply difference to w5e5
                w5e5_regridded_tr_corr = w5e5_regridded_tr + difference
                w5e5_regridded_corr = (
                    1 / (1 + np.exp(-w5e5_regridded_tr_corr))
                ) * 100  # back transform
                w5e5_regridded_corr.raster.set_crs(4326)
                w5e5_regridded_corr_clipped = w5e5_regridded_corr[
                    "hurs"
                ].raster.clip_bbox(hurs_output.raster.bounds)
                w5e5_regridded_corr_clipped = (
                    w5e5_regridded_corr_clipped.rio.set_spatial_dims("lon", "lat")
                )

                hurs_output.loc[dict(time=slice(start_month, end_month))] = (
                    self.snap_to_grid(
                        w5e5_regridded_corr_clipped, hurs_output, xdim="lon", ydim="lat"
                    )
                )

        hurs_output = hurs_output.rename({"lon": "x", "lat": "y"})
        self.set_forcing(hurs_output, "climate/hurs", byteshuffle=True)

    def setup_longwave_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the longwave radiation data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the longwave radiation data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the longwave radiation data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the longwave radiation data for GEB. It first downloads the relative humidity,
        air temperature, and downward longwave radiation data from the ISIMIP dataset for the specified time period using the
        `download_isimip` method. The data is downloaded at a 30 arcsec resolution.

        The method then regrids the downloaded data to the target grid using the `xe.Regridder` method. It calculates the
        saturation vapor pressure, water vapor pressure, clear-sky emissivity, all-sky emissivity, and cloud-based component
        of emissivity for the coarse and fine grids. It then downscales the longwave radiation data for the fine grid using
        the calculated all-sky emissivity and Stefan-Boltzmann constant. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting longwave radiation data is set as forcing data in the model with names of the form 'climate/rlds'.
        """
        x1 = 0.43
        x2 = 5.7
        sbc = 5.67e-8  # stefan boltzman constant [Js−1 m−2 K−4]

        es0 = 6.11  # reference saturation vapour pressure  [hPa]
        T0 = 273.15
        lv = 2.5e6  # latent heat of vaporization of water
        Rv = 461.5  # gas constant for water vapour [J K kg-1]

        target = self.forcing["climate/hurs"].rename({"x": "lon", "y": "lat"})

        hurs_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).hurs  # some buffer to avoid edge effects / errors in ISIMIP API
        tas_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="tas",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).tas  # some buffer to avoid edge effects / errors in ISIMIP API
        rlds_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="rlds",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).rlds  # some buffer to avoid edge effects / errors in ISIMIP API

        import xesmf as xe

        regridder = xe.Regridder(
            hurs_coarse.isel(time=0).drop_vars("time"), target, "bilinear"
        )

        hurs_coarse_regridded = regridder(hurs_coarse, output_chunks=(-1, -1)).rename(
            {"lon": "x", "lat": "y"}
        )
        tas_coarse_regridded = regridder(tas_coarse, output_chunks=(-1, -1)).rename(
            {"lon": "x", "lat": "y"}
        )
        rlds_coarse_regridded = regridder(rlds_coarse, output_chunks=(-1, -1)).rename(
            {"lon": "x", "lat": "y"}
        )

        hurs_fine = self.forcing["climate/hurs"]
        tas_fine = self.forcing["climate/tas"]

        # now ready for calculation:
        es_coarse = es0 * np.exp(
            (lv / Rv) * (1 / T0 - 1 / tas_coarse_regridded)
        )  # saturation vapor pressure
        pV_coarse = (
            hurs_coarse_regridded * es_coarse
        ) / 100  # water vapor pressure [hPa]

        es_fine = es0 * np.exp((lv / Rv) * (1 / T0 - 1 / tas_fine))
        pV_fine = (hurs_fine * es_fine) / 100  # water vapour pressure [hPa]

        e_cl_coarse = 0.23 + x1 * ((pV_coarse * 100) / tas_coarse_regridded) ** (1 / x2)
        # e_cl_coarse == clear-sky emissivity w5e5 (pV needs to be in Pa not hPa, hence *100)
        e_cl_fine = 0.23 + x1 * ((pV_fine * 100) / tas_fine) ** (1 / x2)
        # e_cl_fine == clear-sky emissivity target grid (pV needs to be in Pa not hPa, hence *100)

        e_as_coarse = rlds_coarse_regridded / (
            sbc * tas_coarse_regridded**4
        )  # all-sky emissivity w5e5
        e_as_coarse = xr.where(
            e_as_coarse < 1, e_as_coarse, 1
        )  # constrain all-sky emissivity to max 1
        assert (e_as_coarse <= 1).all(), "all-sky emissivity should be <= 1"
        delta_e = e_as_coarse - e_cl_coarse  # cloud-based component of emissivity w5e5

        e_as_fine = e_cl_fine + delta_e
        e_as_fine = xr.where(
            e_as_fine < 1, e_as_fine, 1
        )  # constrain all-sky emissivity to max 1
        assert (e_as_fine <= 1).all(), "all-sky emissivity should be <= 1"
        lw_fine = (
            e_as_fine * sbc * tas_fine**4
        )  # downscaled lwr! assume cloud e is the same

        lw_fine.name = "rlds"
        lw_fine = self.snap_to_grid(lw_fine, self.grid)
        self.set_forcing(lw_fine, name="climate/rlds", byteshuffle=False)

    def setup_pressure_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the surface pressure data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the surface pressure data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the surface pressure data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the surface pressure data for GEB. It then downloads
        the orography data and surface pressure data from the ISIMIP dataset for the specified time period using the
        `download_isimip` method. The data is downloaded at a 30 arcsec resolution.

        The method then regrids the orography and surface pressure data to the target grid using the `xe.Regridder` method.
        It corrects the surface pressure data for orography using the gravitational acceleration, molar mass of
        dry air, universal gas constant, and sea level standard temperature. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting surface pressure data is set as forcing data in the model with names of the form 'climate/ps'.
        """
        g = 9.80665  # gravitational acceleration [m/s2]
        M = 0.02896968  # molar mass of dry air [kg/mol]
        r0 = 8.314462618  # universal gas constant [J/(mol·K)]
        T0 = 288.16  # Sea level standard temperature  [K]

        target = self.forcing["climate/hurs"].rename({"x": "lon", "y": "lat"})
        pressure_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="psl",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).psl  # some buffer to avoid edge effects / errors in ISIMIP API

        orography = self.download_isimip(
            product="InputData", variable="orog", forcing="chelsa-w5e5", buffer=1
        ).orog  # some buffer to avoid edge effects / errors in ISIMIP API
        import xesmf as xe

        regridder = xe.Regridder(orography, target, "bilinear")
        orography = regridder(orography, output_chunks=(-1, -1)).rename(
            {"lon": "x", "lat": "y"}
        )

        regridder = xe.Regridder(
            pressure_30_min.isel(time=0).drop_vars("time"), target, "bilinear"
        )
        pressure_30_min_regridded = regridder(
            pressure_30_min, output_chunks=(-1, -1)
        ).rename({"lon": "x", "lat": "y"})
        pressure_30_min_regridded_corr = pressure_30_min_regridded * np.exp(
            -(g * orography * M) / (T0 * r0)
        )

        pressure = xr.full_like(self.forcing["climate/hurs"], fill_value=np.nan)
        pressure.name = "ps"
        pressure.attrs = {"units": "Pa", "long_name": "surface pressure"}
        pressure.data = pressure_30_min_regridded_corr

        pressure = self.snap_to_grid(pressure, self.grid)
        self.set_forcing(pressure, name="climate/ps", byteshuffle=True)

    def setup_wind_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the wind data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the wind data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the wind data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the wind data for GEB. It first downloads the global wind atlas data and
        regrids it to the target grid using the `xe.Regridder` method. It then downloads the 30-minute average wind data
        from the ISIMIP dataset for the specified time period and regrids it to the target grid using the `xe.Regridder`
        method.

        The method then creates a diff layer by assuming that wind follows a Weibull distribution and taking the log
        transform of the wind data. It then subtracts the log-transformed 30-minute average wind data from the
        log-transformed global wind atlas data to create the diff layer.

        The method then downloads the wind data from the ISIMIP dataset for the specified time period and regrids it to the
        target grid using the `xe.Regridder` method. It applies the diff layer to the log-transformed wind data and then
        exponentiates the result to obtain the corrected wind data. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting wind data is set as forcing data in the model with names of the form 'climate/wind'.

        Currently the global wind atlas database is offline, so the correction is removed
        """
        import xesmf as xe

        global_wind_atlas = self.data_catalog.get_rasterdataset(
            "global_wind_atlas", bbox=self.grid.raster.bounds, buffer=10
        ).rename({"x": "lon", "y": "lat"})
        target = self.grid["areamaps/grid_mask"].rename({"x": "lon", "y": "lat"})

        regridder = xe.Regridder(global_wind_atlas.copy(), target, "bilinear")
        global_wind_atlas_regridded = regridder(
            global_wind_atlas, output_chunks=(-1, -1)
        )

        wind_30_min_avg = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            starttime=date(2008, 1, 1),
            endtime=date(2017, 12, 31),
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind.mean(
            dim="time"
        )  # some buffer to avoid edge effects / errors in ISIMIP API
        regridder_30_min = xe.Regridder(wind_30_min_avg, target, "bilinear")
        wind_30_min_avg_regridded = regridder_30_min(wind_30_min_avg)

        # create diff layer:
        # assume wind follows weibull distribution => do log transform
        wind_30_min_avg_regridded_log = np.log(wind_30_min_avg_regridded)

        global_wind_atlas_regridded_log = np.log(global_wind_atlas_regridded)

        diff_layer = (
            global_wind_atlas_regridded_log - wind_30_min_avg_regridded_log
        )  # to be added to log-transformed daily

        wind_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind  # some buffer to avoid edge effects / errors in ISIMIP API

        wind_30min_regridded = regridder_30_min(wind_30_min)
        wind_30min_regridded_log = np.log(wind_30min_regridded)

        wind_30min_regridded_log_corr = wind_30min_regridded_log + diff_layer
        wind_30min_regridded_corr = np.exp(wind_30min_regridded_log_corr)

        wind_output_clipped = wind_30min_regridded_corr.raster.clip_bbox(
            self.grid.raster.bounds
        )
        wind_output_clipped = wind_output_clipped.rename({"lon": "x", "lat": "y"})
        wind_output_clipped.name = "sfcwind"

        wind_output_clipped = self.snap_to_grid(wind_output_clipped, self.grid)
        self.set_forcing(wind_output_clipped, "climate/sfcwind", byteshuffle=True)

    def setup_SPEI(
        self,
        calibration_period_start: date = date(1981, 1, 1),
        calibration_period_end: date = date(2010, 1, 1),
        window: int = 12,
    ):
        """
        Sets up the Standardized Precipitation Evapotranspiration Index (SPEI). Note that
        due to the sliding window, the SPEI data will be shorter than the original data. When
        a sliding window of 12 months is used, the SPEI data will be shorter by 11 months.

        Also sets up the Generalized Extreme Value (GEV) parameters for the SPEI data, being
        the c shape (ξ), loc location (μ), and scale (σ) parameters.

        The chunks for the climate data are optimized for reading the data in xy-direction. However,
        for the SPEI calculation, the data is needs to be read in time direction. Therefore, we
        create an intermediate temporary file of the water balance wher chunks are in an intermediate
        size between the xy and time chunks.

        Parameters
        ----------
        calibration_period_start : date
            The start time of the reSPEI data in ISO 8601 format (YYYY-MM-DD).
        calibration_period_end : date
            The end time of the SPEI data in ISO 8601 format (YYYY-MM-DD). Endtime is exclusive.
        """
        self.logger.info("setting up SPEI...")

        # assert input data have the same coordinates
        assert np.array_equal(
            self.forcing["climate/pr"].x, self.forcing["climate/tasmin"].x
        )
        assert np.array_equal(
            self.forcing["climate/pr"].x, self.forcing["climate/tasmax"].x
        )
        assert np.array_equal(
            self.forcing["climate/pr"].y, self.forcing["climate/tasmin"].y
        )
        assert np.array_equal(
            self.forcing["climate/pr"].y, self.forcing["climate/tasmax"].y
        )
        if not self.forcing[
            "climate/pr"
        ].time.min().dt.date <= calibration_period_start and self.forcing[
            "climate/pr"
        ].time.max().dt.date >= calibration_period_end - timedelta(days=1):
            forcing_start_date = self.forcing["climate/pr"].time.min().dt.date.item()
            forcing_end_date = self.forcing["climate/pr"].time.max().dt.date.item()
            raise AssertionError(
                f"water data does not cover the entire calibration period, forcing data covers from {forcing_start_date} to {forcing_end_date}, "
                f"while requested calibration period is from {calibration_period_start} to {calibration_period_end}"
            )

        self.forcing["climate/tasmin"]["y"].attrs["standard_name"] = "latitude"
        self.forcing["climate/tasmin"]["x"].attrs["standard_name"] = "longitude"
        self.forcing["climate/tasmin"]["y"].attrs["units"] = "degrees_north"
        self.forcing["climate/tasmin"]["x"].attrs["units"] = "degrees_east"

        pet = xci.potential_evapotranspiration(
            tasmin=self.forcing["climate/tasmin"],
            tasmax=self.forcing["climate/tasmax"],
            method="BR65",
        )

        # Compute the potential evapotranspiration
        water_budget = xci.water_budget(pr=self.forcing["climate/pr"], evspsblpot=pet)

        water_budget.attrs = {"units": "kg m-2 s-1"}

        water_budget.name = "water_budget"
        chunks = {
            "time": 100,
            "y": XY_CHUNKSIZE,
            "x": XY_CHUNKSIZE,
        }
        water_budget = water_budget.chunk(chunks)

        with tempfile.NamedTemporaryFile(suffix=".zarr.zip") as tmp_water_budget_file:
            print("Exporting temporary water budget to zarr")
            with ProgressBar(dt=10):
                water_budget.to_zarr(
                    tmp_water_budget_file.name,
                    mode="w",
                    encoding={"water_budget": {"chunks": chunks.values()}},
                )

            water_budget = xr.open_zarr(tmp_water_budget_file.name, chunks={})[
                "water_budget"
            ]
            # xclim fails when dparams is present, thus remove it
            if "dparams" in water_budget.coords:
                water_budget = water_budget.drop("dparams")

            # Compute the SPEI
            SPEI = xci.standardized_precipitation_evapotranspiration_index(
                wb=water_budget,
                cal_start=calibration_period_start,
                cal_end=calibration_period_end,
                freq="MS",
                window=window,
                dist="gamma",
                method="ML",
            ).chunk(chunks)

            # remove all nan values as a result of the sliding window
            SPEI.attrs = {
                "units": "-",
                "long_name": "Standard Precipitation Evapotranspiration Index",
                "name": "spei",
            }
            SPEI.name = "spei"

            with tempfile.NamedTemporaryFile(suffix=".zarr.zip") as tmp_spei_file:
                print("Exporting temporary SPEI to zarr")
                with ProgressBar(dt=10):
                    SPEI.to_zarr(
                        tmp_spei_file.name,
                        mode="w",
                        encoding={"spei": {"chunks": chunks.values()}},
                    )

                SPEI = xr.open_zarr(
                    tmp_spei_file.name,
                    chunks={},
                )["spei"]

                self.set_forcing(SPEI, name="climate/spei")

                self.logger.info("calculating GEV parameters...")

                # Group the data by year and find the maximum monthly sum for each year
                SPEI_yearly_min = SPEI.groupby("time.year").min(dim="time", skipna=True)

                SPEI_yearly_min = SPEI_yearly_min.dropna(dim="year")

                SPEI_yearly_min = (
                    SPEI_yearly_min.rename({"year": "time"})
                    .chunk({"time": -1})
                    .compute()
                )

                GEV = xci.stats.fit(SPEI_yearly_min, dist="genextreme").compute()
                GEV.name = "gev"

                self.set_grid(GEV.sel(dparams="c"), name="climate/gev_c")
                self.set_grid(GEV.sel(dparams="loc"), name="climate/gev_loc")
                self.set_grid(GEV.sel(dparams="scale"), name="climate/gev_scale")

    def setup_regions_and_land_use(
        self,
        region_database="GADM_level1",
        unique_region_id="GID_1",
        ISO3_column="GID_0",
        river_threshold=100,
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
        river_threshold : int, optional
            The threshold value to use when identifying rivers in the MERIT dataset. Default is 100.

        Notes
        -----
        This method sets up the regions and land use data for GEB. It first retrieves the region data from
        the specified region database and sets it as a geometry in the model. It then pads the subgrid to cover the entire
        region and retrieves the land use data from the ESA WorldCover dataset. The land use data is reprojected to the
        padded subgrid and the region ID is rasterized onto the subgrid. The cell area for each region is calculated and
        set as a grid in the model. The MERIT dataset is used to identify rivers, which are set as a grid in the model. The
        land use data is reclassified into five classes and set as a grid in the model. Finally, the cultivated land is
        identified and set as a grid in the model.

        The resulting grids are set as attributes of the model with names of the form 'areamaps/{grid_name}' or
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
        self.set_dict(region_id_mapping, name="areamaps/region_id_mapping")

        assert "ISO3" in regions.columns, (
            f"Region database must contain ISO3 column ({self.data_catalog[region_database].path})"
        )

        self.set_geoms(regions, name="areamaps/regions")

        resolution_x, resolution_y = self.subgrid[
            "areamaps/sub_grid_mask"
        ].rio.resolution()

        regions_bounds = self.geoms["areamaps/regions"].total_bounds
        mask_bounds = self.grid["areamaps/grid_mask"].raster.bounds

        # The bounds should be set to a bit larger than the regions to avoid edge effects
        # and also larger than the mask, to ensure that the entire grid is covered.
        pad_minx = min(regions_bounds[0], mask_bounds[0]) - abs(resolution_x) / 2.0
        pad_miny = min(regions_bounds[1], mask_bounds[1]) - abs(resolution_y) / 2.0
        pad_maxx = max(regions_bounds[2], mask_bounds[2]) + abs(resolution_x) / 2.0
        pad_maxy = max(regions_bounds[3], mask_bounds[3]) + abs(resolution_y) / 2.0

        # TODO: Is there a better way to do this?
        region_subgrid, region_subgrid_slice = pad_xy(
            self.subgrid["areamaps/sub_grid_mask"].rio,
            pad_minx,
            pad_miny,
            pad_maxx,
            pad_maxy,
            return_slice=True,
            constant_values=1,
        )
        region_subgrid.raster.set_crs(self.subgrid.raster.crs)
        region_subgrid = region_subgrid.astype(np.int8)
        region_subgrid.raster.set_nodata(-1)
        self.set_region_subgrid(region_subgrid, name="areamaps/region_mask")

        land_use = self.data_catalog.get_rasterdataset(
            land_cover,
            geom=self.geoms["areamaps/regions"],
            buffer=200,  # 2 km buffer
        )
        reprojected_land_use = land_use.raster.reproject_like(
            region_subgrid, method="nearest"
        )

        region_raster = reprojected_land_use.raster.rasterize(
            self.geoms["areamaps/regions"],
            col_name="region_id",
            all_touched=True,
        ).compute()
        self.set_region_subgrid(region_raster, name="areamaps/region_subgrid")

        region_subgrid_cell_area = xr.full_like(region_subgrid, np.nan)

        region_subgrid_cell_area.data = calculate_cell_area(
            region_subgrid_cell_area.raster.transform, region_subgrid_cell_area.shape
        )
        region_subgrid_cell_area = region_subgrid_cell_area.compute()

        # set the cell area for the region subgrid
        self.set_region_subgrid(
            region_subgrid_cell_area,
            name="areamaps/region_cell_area_subgrid",
        )

        MERIT = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            variables=["upg"],
            bbox=region_subgrid.rio.bounds(),
            buffer=300,  # 3 km buffer
        )
        # There is a half degree offset in MERIT data
        MERIT = MERIT.assign_coords(
            x=MERIT.coords["x"] + MERIT.rio.resolution()[0] / 2,
            y=MERIT.coords["y"] - MERIT.rio.resolution()[1] / 2,
        )

        # Assume all cells with at least x upstream cells are rivers.
        rivers = MERIT > river_threshold
        rivers = rivers.astype(np.int32)
        rivers.raster.set_nodata(-1)
        rivers = rivers.raster.reproject_like(
            reprojected_land_use, method="nearest"
        ).compute()
        self.set_region_subgrid(rivers, name="landcover/rivers")

        hydro_land_use = reprojected_land_use.raster.reclassify(
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
        )["GEB_land_use_class"]
        hydro_land_use = xr.where(
            rivers != 1, hydro_land_use, 5, keep_attrs=True
        )  # set rivers to 5 (permanent water bodies)
        hydro_land_use.raster.set_nodata(-1)

        hydro_land_use = hydro_land_use.compute()
        self.set_region_subgrid(
            hydro_land_use, name="landsurface/full_region_land_use_classes"
        )

        cultivated_land = xr.where(
            (hydro_land_use == 1) & (reprojected_land_use == 40), 1, 0, keep_attrs=True
        )
        cultivated_land.raster.set_crs(self.subgrid.raster.crs)
        cultivated_land.raster.set_nodata(-1)

        cultivated_land = cultivated_land.compute()
        self.set_region_subgrid(
            cultivated_land, name="landsurface/full_region_cultivated_land"
        )

        hydro_land_use_region = hydro_land_use.isel(region_subgrid_slice)
        self.set_subgrid(hydro_land_use_region, name="landsurface/land_use_classes")

        cultivated_land_region = cultivated_land.isel(region_subgrid_slice)
        self.set_subgrid(cultivated_land_region, name="landsurface/cultivated_land")

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
        'economics/lending_rates' and 'economics/inflation_rates', respectively.
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

        for _, region in self.geoms["areamaps/regions"].iterrows():
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

        self.set_dict(inflation_rates_dict, name="economics/inflation_rates")
        # self.set_dict(lending_rates_dict, name="economics/lending_rates")
        self.set_dict(price_ratio_dict, name="economics/price_ratio")

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
        retrieves the inflation rates data from the `economics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'economics/well_prices' and 'economics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["economics/inflation_rates"]
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
            self.set_dict(prices_dict, name=f"economics/{price_type}")

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
        retrieves the inflation rates data from the `economics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'economics/well_prices' and 'economics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["economics/inflation_rates"]
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
            self.set_dict(prices_dict, name=f"economics/{price_type}")

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
        retrieves the inflation rates data from the `economics/inflation_rates` dictionary. It then creates dictionaries to
        store the well prices and upkeep prices for each region, with the years as the time dimension and the prices as the
        data dimension.

        The well prices and upkeep prices are calculated by applying the inflation rates to the reference year prices. The
        resulting prices are stored in the dictionaries with the region ID as the key.

        The resulting well prices and upkeep prices data are set as dictionary with names of the form
        'economics/well_prices' and 'economics/upkeep_prices_well_per_m2', respectively.
        """
        self.logger.info("Setting up well prices by reference year")

        # Retrieve the inflation rates data
        inflation_rates = self.dict["economics/inflation_rates"]
        price_ratio = self.dict["economics/price_ratio"]

        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "why_10": WHY_10,
            "why_20": WHY_20,
            "why_30": WHY_30,
        }

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for _, region in self.geoms["areamaps/regions"].iterrows():
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
            self.set_dict(prices_dict, name=f"economics/{price_type}")

        electricity_rates = self.data_catalog.get_dataframe("gcam_electricity_rates")
        electricity_rates["ISO3"] = electricity_rates["Country"].map(
            SUPERWELL_NAME_TO_ISO3
        )
        electricity_rates = electricity_rates.set_index("ISO3")["Rate"].to_dict()

        electricity_rates_dict = {
            "time": list(range(start_year, end_year + 1)),
            "data": {},
        }

        for _, region in self.geoms["areamaps/regions"].iterrows():
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
        self.set_dict(electricity_rates_dict, name="economics/electricity_cost")

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
        inflation_rates = self.dict["economics/inflation_rates"]
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
            self.set_dict(prices_dict, name=f"economics/{price_type}")

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
        `areamaps/regions` and `areamaps/region_subgrid` grids. It then creates a `farms` grid with the same shape as the
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

        Finally, the method sets the binary data for each column of the `farmers` DataFrame as agents data in the model
        with names of the form 'agents/farmers/{column}'.
        """
        regions = self.geoms["areamaps/regions"]
        regions_raster = self.region_subgrid["areamaps/region_subgrid"].compute()
        full_region_cultivated_land = self.region_subgrid[
            "landsurface/full_region_cultivated_land"
        ].compute()

        farms = hydromt.raster.full_like(regions_raster, nodata=-1, lazy=True)
        farms[:] = -1
        assert farms.min() >= -1  # -1 is nodata value, all farms should be positive

        for region_id in regions["region_id"]:
            self.logger.info(f"Creating farms for region {region_id}")
            region = regions_raster == region_id
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
            farms = farms.compute()  # perhaps this helps with memory issues?

        farmers = farmers.drop("area_n_cells", axis=1)

        region_mask = self.region_subgrid["areamaps/region_mask"].astype(bool)

        # TODO: Again why is dtype changed? And export doesn't work?
        cut_farms = np.unique(
            xr.where(region_mask, farms.copy().values, -1, keep_attrs=True)
        )
        cut_farms = cut_farms[cut_farms != -1]

        assert farms.min() >= -1  # -1 is nodata value, all farms should be positive
        subgrid_farms = farms.raster.clip_bbox(
            self.subgrid["areamaps/sub_grid_mask"].raster.bounds
        )

        subgrid_farms_in_study_area = xr.where(
            np.isin(subgrid_farms, cut_farms), -1, subgrid_farms, keep_attrs=True
        )
        farmers = farmers[~farmers.index.isin(cut_farms)]

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

        subgrid_farms_in_study_area_array = hydromt.raster.full_from_transform(
            self.subgrid["areamaps/sub_grid_mask"].raster.transform,
            self.subgrid["areamaps/sub_grid_mask"].raster.shape,
            nodata=-1,
            dtype=np.int32,
            crs=self.subgrid.raster.crs,
            name="agents/farmers/farms",
            lazy=True,
        )
        subgrid_farms_in_study_area_array[:] = subgrid_farms_in_study_area
        self.set_subgrid(subgrid_farms_in_study_area_array, name="agents/farmers/farms")

        self.set_binary(farmers.index.values, name="agents/farmers/id")
        self.set_binary(farmers["region_id"].values, name="agents/farmers/region_id")

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

    def determine_crop_area_fractions(self, resolution="5-arcminute"):
        output_folder = "plot/mirca_crops"
        os.makedirs(output_folder, exist_ok=True)

        crops = [
            "Wheat",  # 0
            "Maize",  # 1
            "Rice",  # 2
            "Barley",  # 3
            "Rye",  # 4
            "Millet",  # 5
            "Sorghum",  # 6
            "Soybeans",  # 7
            "Sunflower",  # 8
            "Potatoes",  # 9
            "Cassava",  # 10
            "Sugar_cane",  # 11
            "Sugar_beet",  # 12
            "Oil_palm",  # 13
            "Rapeseed",  # 14
            "Groundnuts",  # 15
            "Others_perennial",  # 23
            "Fodder",  # 24
            "Others_annual",  # 25,
        ]

        years = ["2000", "2005", "2010", "2015"]
        irrigation_types = ["ir", "rf"]

        # Initialize lists to collect DataArrays across years
        fraction_da_list = []
        irrigated_fraction_da_list = []

        # Initialize a dictionary to store datasets
        crop_data = {}

        for year in years:
            crop_data[year] = {}
            for crop in crops:
                crop_data[year][crop] = {}
                for irrigation in irrigation_types:
                    dataset_name = f"MIRCA2000_cropping_area_{year}_{resolution}_{crop}_{irrigation}"

                    crop_map = self.data_catalog.get_rasterdataset(
                        dataset_name,
                        bbox=self.bounds,
                        buffer=2,
                    )
                    crop_map = crop_map.fillna(0)

                    crop_data[year][crop][irrigation] = crop_map.assign_coords(
                        x=np.round(crop_map.coords["x"].values, decimals=6),
                        y=np.round(crop_map.coords["y"].values, decimals=6),
                    )

            # Initialize variables for total calculations
            total_cropped_area = None
            total_crop_areas = {}

            # Calculate total crop areas and total cropped area
            for crop in crops:
                irrigated = crop_data[year][crop]["ir"]
                rainfed = crop_data[year][crop]["rf"]

                total_crop = irrigated + rainfed
                total_crop_areas[crop] = total_crop

                if total_cropped_area is None:
                    total_cropped_area = total_crop.copy()
                else:
                    total_cropped_area += total_crop

            # Initialize lists to collect DataArrays for this year
            fraction_list = []
            irrigated_fraction_list = []

            # Calculate the fraction of each crop to the total cropped area
            for crop in crops:
                fraction = total_crop_areas[crop] / total_cropped_area

                # Assign 'crop' as a coordinate
                fraction = fraction.assign_coords(crop=crop)

                # Append to the list
                fraction_list.append(fraction)

            # Concatenate the list of fractions into a single DataArray along the 'crop' dimension
            fraction_da = xr.concat(fraction_list, dim="crop")

            # Assign the 'year' coordinate and expand dimensions to include 'year'
            fraction_da = fraction_da.assign_coords(year=year).expand_dims(dim="year")

            # Append to the list of all years
            fraction_da_list.append(fraction_da)

            # Calculate irrigated fractions for each crop and collect them
            for crop in crops:
                irrigated = crop_data[year][crop]["ir"].compute()
                total_crop = total_crop_areas[crop]
                irrigated_fraction = irrigated / total_crop

                # Assign 'crop' as a coordinate
                irrigated_fraction = irrigated_fraction.assign_coords(crop=crop)

                # Append to the list
                irrigated_fraction_list.append(irrigated_fraction)

            # Concatenate the list of irrigated fractions into a single DataArray along the 'crop' dimension
            irrigated_fraction_da = xr.concat(irrigated_fraction_list, dim="crop")

            # Assign the 'year' coordinate and expand dimensions to include 'year'
            irrigated_fraction_da = irrigated_fraction_da.assign_coords(
                year=year
            ).expand_dims(dim="year")

            # Append to the list of all years
            irrigated_fraction_da_list.append(irrigated_fraction_da)

        # After processing all years, concatenate along the 'year' dimension
        all_years_fraction_da = xr.concat(fraction_da_list, dim="year")
        all_years_irrigated_fraction_da = xr.concat(
            irrigated_fraction_da_list, dim="year"
        )

        # Save the concatenated DataArrays as NetCDF files
        save_dir = self.preprocessing_dir / "crops" / "MIRCA2000"
        save_dir.mkdir(parents=True, exist_ok=True)

        output_filename = save_dir / "crop_area_fraction_all_years.nc"
        all_years_fraction_da.to_netcdf(output_filename)

        output_filename = save_dir / "crop_irrigated_fraction_all_years.nc"
        all_years_irrigated_fraction_da.to_netcdf(output_filename)

    def setup_create_farms_simple(
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

        cultivated_land = (
            self.region_subgrid["landsurface/full_region_cultivated_land"]
            .astype(bool)
            .compute()
        )
        regions_grid = self.region_subgrid["areamaps/region_subgrid"].compute()
        cell_area = self.region_subgrid["areamaps/region_cell_area_subgrid"].compute()

        regions_shapes = self.geoms["areamaps/regions"]
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
        self.logger.debug(f"Starting processing of {len(regions_shapes)} regions")
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

            self.logger.debug(f"Processing region {UID}")

            cultivated_land_region_total_cells = (
                ((regions_grid == UID) & (cultivated_land)).sum().compute()
            )
            total_cultivated_land_area_lu = (
                (((regions_grid == UID) & (cultivated_land)) * cell_area)
                .sum()
                .compute()
            )
            if (
                total_cultivated_land_area_lu == 0
            ):  # when no agricultural area, just continue as there will be no farmers. Also avoiding some division by 0 errors.
                continue

            average_cell_area_region = (
                cell_area.where(((regions_grid == UID) & (cultivated_land)))
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

    def setup_household_characteristics(self, maximum_age=85, v=True):
        # load GDL region within model domain
        GDL_regions = self.data_catalog.get_geodataframe(
            "GDL_regions_v4", geom=self.region, variables=["GDLcode"]
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
        for GDL_region in GDL_regions["GDLcode"]:
            region_results[GDL_region] = {}
            GLOPOP_S_region, GLOPOP_GRID_region = load_GLOPOP_S(
                self.data_catalog, GDL_region
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

            for i, HID in enumerate(GLOPOP_households_region):
                if v:
                    print(f"searching household {i} of {n_households}")
                household = GLOPOP_S_region[GLOPOP_S_region["HID"] == HID]
                household_size = len(household)
                if len(household) > 1:
                    # if there are multiple people in the household
                    # take head household
                    household = household[household["RELATE_HEAD"] == 1]

                GRID_CELL = int(household["GRID_CELL"])
                assert GRID_CELL in GLOPOP_GRID_region.values, (
                    f"{HID} should in in GLOPOP for this region"
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

            region_results[GDL_region] = household_characteristics

        # concatenate all data
        data_concatenated = {}
        for household_attribute in household_characteristics:
            data_concatenated[household_attribute] = np.concatenate(
                [
                    region_results[GDL_region][household_attribute]
                    for GDL_region in region_results
                ]
            )

            # and store to binary
            if household_attribute in attributes_to_include:
                self.set_binary(
                    data_concatenated[household_attribute],
                    name=f"agents/households/{attributes_to_include[household_attribute]}",
                )
            else:
                self.set_binary(
                    data_concatenated[household_attribute],
                    name=f"agents/households/{household_attribute}",
                )

    def setup_farmer_household_characteristics(self, maximum_age=85):
        n_farmers = self.binary["agents/farmers/id"].size
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

        locations = pixels_to_coords(pixels + 0.5, farms.raster.transform.to_gdal())
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

        self.set_binary(
            GDL_region_per_farmer["HHSIZE_CAT"].values,
            name="agents/farmers/household_size",
        )
        self.set_binary(
            GDL_region_per_farmer["AGE"].values,
            name="agents/farmers/age_household_head",
        )
        self.set_binary(
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

    def setup_farmer_characteristics_simple(
        self,
        interest_rate=0.05,
    ):
        n_farmers = self.binary["agents/farmers/id"].size

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

        ISO3_codes_region = self.geoms["areamaps/regions"]["ISO3"].unique()
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

        unique_regions = self.geoms["areamaps/regions"]

        data = self.donate_and_receive_crop_prices(
            donor_data, unique_regions, GLOBIOM_regions
        )

        # Map to corresponding region
        data_reset = data.reset_index(level="region_id")
        data = data_reset.set_index("region_id")
        region_ids = self.binary["agents/farmers/region_id"]

        # Set gains and losses
        gains_array = pd.Series(region_ids).map(data["Gains"]).to_numpy()
        gains_std = pd.Series(region_ids).map(data["Gains_std"]).to_numpy()
        losses_array = pd.Series(region_ids).map(data["Losses"]).to_numpy()
        losses_std = pd.Series(region_ids).map(data["Losses_std"]).to_numpy()
        discount_array = pd.Series(region_ids).map(data["Discount"]).to_numpy()
        discount_std = pd.Series(region_ids).map(data["Discount_std"]).to_numpy()

        try:
            # income = self.binary["agents/farmers/income"]
            pass
        except KeyError:
            self.logger.info("Income does not exist, generating random income..")
            daily_non_farm_income_family = random.choices(
                [50, 100, 200, 500], k=n_farmers
            )
            self.set_binary(
                daily_non_farm_income_family,
                name="agents/farmers/income",
            )

        try:
            household_size = self.binary["agents/farmers/household_size"]
        except KeyError:
            self.logger.info("Household size does not exist, generating random sizes..")
            household_size = random.choices([1, 2, 3, 4, 5, 6, 7], k=n_farmers)
            self.set_binary(household_size, name="agents/farmers/household_size")

            daily_consumption_per_capita = random.choices(
                [50, 100, 200, 500], k=n_farmers
            )
            self.set_binary(
                daily_consumption_per_capita,
                name="agents/farmers/daily_consumption_per_capita",
            )

        try:
            education_levels = self.binary["agents/farmers/education_level"]
        except KeyError:
            self.logger.info(
                "Education level does not exist, generating random levels.."
            )
            education_levels = random.choices(
                [1, 2, 3, 4, 5], k=n_farmers
            )  # Random levels from 0-4 or your preferred scale
            self.set_binary(education_levels, name="agents/farmers/education_level")

        try:
            age = self.binary["agents/farmers/age_household_head"]
        except KeyError:
            self.logger.info(
                "Age of household head does not exist, generating random ages.."
            )
            age = random.choices(
                range(18, 80), k=n_farmers
            )  # Random ages between 18 and 80
            self.set_binary(age, name="agents/farmers/age_household_head")

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

        self.set_binary(neutral_risk_aversion, name="agents/farmers/risk_aversion")
        self.set_binary(
            gains_array_with_variation, name="agents/farmers/risk_aversion_gains"
        )
        self.set_binary(
            losses_array_with_variation, name="agents/farmers/risk_aversion_losses"
        )
        self.set_binary(intention_factor, name="agents/farmers/intention_factor")

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
        self.set_binary(
            discount_rate_with_variation,
            name="agents/farmers/discount_rate",
        )

        interest_rate = np.full(n_farmers, interest_rate, dtype=np.float32)
        self.set_binary(interest_rate, name="agents/farmers/interest_rate")

    def setup_farmer_crop_calendar_multirun(
        self,
        year=2000,
        reduce_crops=False,
        replace_base=False,
        export=False,
    ):
        years = [2000, 2005, 2010, 2015]
        nr_runs = 20

        for year_nr in years:
            for run in range(nr_runs):
                self.setup_farmer_crop_calendar(
                    year_nr, reduce_crops, replace_base, export
                )

    def setup_farmer_crop_calendar(
        self,
        year=2000,
        reduce_crops=False,
        replace_base=False,
    ):
        n_farmers = self.binary["agents/farmers/id"].size

        MIRCA_unit_grid = self.data_catalog.get_rasterdataset(
            "MIRCA2000_unit_grid", bbox=self.bounds, buffer=2
        )

        crop_calendar = parse_MIRCA2000_crop_calendar(
            self.data_catalog,
            MIRCA_units=np.unique(MIRCA_unit_grid.values),
        )

        farmer_locations = get_farm_locations(
            self.subgrid["agents/farmers/farms"], method="centroid"
        )

        farmer_mirca_units = sample_from_map(
            MIRCA_unit_grid.values,
            farmer_locations,
            MIRCA_unit_grid.raster.transform.to_gdal(),
        )

        farmer_crops, is_irrigated = self.assign_crops_irrigation_farmers(year)
        self.setup_farmer_irrigation_source(is_irrigated, year)

        crop_calendar_per_farmer = np.zeros((n_farmers, 3, 4), dtype=np.int32)
        for mirca_unit in np.unique(farmer_mirca_units):
            farmers_in_unit = np.where(farmer_mirca_units == mirca_unit)[0]

            area_per_crop_rotation = []
            cropping_calenders_crop_rotation = []
            for crop_rotation in crop_calendar[mirca_unit]:
                area_per_crop_rotation.append(crop_rotation[0])
                crop_rotation_matrix = crop_rotation[1]
                starting_days = crop_rotation_matrix[:, 2]
                starting_days = starting_days[starting_days != -1]
                assert np.unique(starting_days).size == starting_days.size, (
                    "ensure all starting days are unique"
                )
                # TODO: Add check to ensure crop calendars are not overlapping.
                cropping_calenders_crop_rotation.append(crop_rotation_matrix)
            area_per_crop_rotation = np.array(area_per_crop_rotation)
            cropping_calenders_crop_rotation = np.stack(
                cropping_calenders_crop_rotation
            )

            crops_in_unit = np.unique(farmer_crops[farmers_in_unit])
            for crop_id in crops_in_unit:
                # Find rotations that include this crop
                rotations_with_crop_idx = []
                for idx, rotation in enumerate(cropping_calenders_crop_rotation):
                    # Get crop IDs in the rotation, excluding -1 entries
                    crop_ids_in_rotation = rotation[:, 0]
                    crop_ids_in_rotation = crop_ids_in_rotation[
                        crop_ids_in_rotation != -1
                    ]
                    if crop_id in crop_ids_in_rotation:
                        rotations_with_crop_idx.append(idx)

                if not rotations_with_crop_idx:
                    print(
                        f"No rotations found for crop ID {crop_id} in mirca unit {mirca_unit}"
                    )
                    continue

                # Get the area fractions and rotations for these indices
                areas_with_crop = area_per_crop_rotation[rotations_with_crop_idx]
                rotations_with_crop = cropping_calenders_crop_rotation[
                    rotations_with_crop_idx
                ]

                # Normalize the area fractions
                total_area_for_crop = areas_with_crop.sum()
                fractions = areas_with_crop / total_area_for_crop

                # Get farmers with this crop in the mirca_unit
                farmers_with_crop_in_unit = farmers_in_unit[
                    farmer_crops[farmers_in_unit] == crop_id
                ]

                # Assign crop rotations to these farmers
                assigned_rotation_indices = np.random.choice(
                    np.arange(len(rotations_with_crop)),
                    size=len(farmers_with_crop_in_unit),
                    replace=True,
                    p=fractions,
                )

                # Assign the crop calendars to the farmers
                for farmer_idx, rotation_idx in zip(
                    farmers_with_crop_in_unit, assigned_rotation_indices
                ):
                    assigned_rotation = rotations_with_crop[rotation_idx]
                    # Assign to farmer's crop calendar, taking columns [0, 2, 3, 4]
                    # Columns: [crop_id, planting_date, harvest_date, additional_attribute]
                    crop_calendar_per_farmer[farmer_idx] = assigned_rotation[
                        :, [0, 2, 3, 4]
                    ]

        # Define constants for crop IDs
        WHEAT = 0
        MAIZE = 1
        RICE = 2
        BARLEY = 3
        RYE = 4
        MILLET = 5
        SORGHUM = 6
        SOYBEANS = 7
        SUNFLOWER = 8
        POTATOES = 9
        CASSAVA = 10
        SUGAR_CANE = 11
        SUGAR_BEETS = 12
        OIL_PALM = 13
        RAPESEED = 14
        GROUNDNUTS = 15
        # PULSES = 16
        # CITRUS = 17
        # # DATE_PALM = 18
        # # GRAPES = 19
        # COTTON = 20
        COCOA = 21
        COFFEE = 22
        OTHERS_PERENNIAL = 23
        FODDER_GRASSES = 24
        OTHERS_ANNUAL = 25
        WHEAT_DROUGHT = 26
        WHEAT_FLOOD = 27
        MAIZE_DROUGHT = 28
        MAIZE_FLOOD = 29
        RICE_DROUGHT = 30
        RICE_FLOOD = 31
        SOYBEANS_DROUGHT = 32
        SOYBEANS_FLOOD = 33
        POTATOES_DROUGHT = 34
        POTATOES_FLOOD = 35

        # Manual replacement of certain crops
        def replace_crop(crop_calendar_per_farmer, crop_values, replaced_crop_values):
            # Find the most common crop value among the given crop_values
            crop_instances = crop_calendar_per_farmer[:, :, 0][
                np.isin(crop_calendar_per_farmer[:, :, 0], crop_values)
            ]

            # if none of the crops are present, no need to replace anything
            if crop_instances.size == 0:
                return crop_calendar_per_farmer

            crops, crop_counts = np.unique(crop_instances, return_counts=True)
            most_common_crop = crops[np.argmax(crop_counts)]

            # Determine if there are multiple cropping versions of this crop and assign it to the most common
            new_crop_types = crop_calendar_per_farmer[
                (crop_calendar_per_farmer[:, :, 0] == most_common_crop).any(axis=1),
                :,
                :,
            ]
            unique_rows, counts = np.unique(new_crop_types, axis=0, return_counts=True)
            max_index = np.argmax(counts)
            crop_replacement = unique_rows[max_index]

            crop_replacement_only_crops = crop_replacement[
                crop_replacement[:, -1] != -1
            ]
            if crop_replacement_only_crops.shape[0] > 1:
                assert (
                    np.unique(crop_replacement_only_crops[:, [1, 3]], axis=0).shape[0]
                    == crop_replacement_only_crops.shape[0]
                )

            for replaced_crop in replaced_crop_values:
                # Check where to be replaced crop is
                crop_mask = (crop_calendar_per_farmer[:, :, 0] == replaced_crop).any(
                    axis=1
                )
                # Replace the crop
                crop_calendar_per_farmer[crop_mask] = crop_replacement

            return crop_calendar_per_farmer

        def unify_crop_variants(crop_calendar_per_farmer, target_crop):
            # Create a mask for all entries whose first value == target_crop
            mask = crop_calendar_per_farmer[..., 0] == target_crop

            # If the crop does not appear at all, nothing to do
            if not np.any(mask):
                return crop_calendar_per_farmer

            # Extract only the rows/entries that match the target crop
            crop_entries = crop_calendar_per_farmer[mask]

            # Among these crop rows, find unique variants and their counts
            # (axis=0 ensures we treat each row/entry as a unit)
            unique_variants, variant_counts = np.unique(
                crop_entries, axis=0, return_counts=True
            )

            # The most common variant is the unique variant with the highest count
            most_common_variant = unique_variants[np.argmax(variant_counts)]

            # Replace all the target_crop rows with the most common variant
            crop_calendar_per_farmer[mask] = most_common_variant

            return crop_calendar_per_farmer

        def insert_other_variant_crop(
            crop_calendar_per_farmer, base_crops, resistant_crops
        ):
            # find crop rotation mask
            base_crop_rotation_mask = (
                crop_calendar_per_farmer[:, :, 0] == base_crops
            ).any(axis=1)

            # Find the indices of the crops to be replaced
            indices = np.where(base_crop_rotation_mask)[0]

            # Shuffle the indices to randomize the selection
            np.random.shuffle(indices)

            # Determine the number of crops for each category (stay same, first resistant, last resistant)
            n = len(indices)
            n_same = n // 3
            n_first_resistant = (n // 3) + (
                n % 3 > 0
            )  # Ensuring we account for rounding issues

            # Assign the new values
            crop_calendar_per_farmer[indices[:n_same], 0, 0] = base_crops
            crop_calendar_per_farmer[
                indices[n_same : n_same + n_first_resistant], 0, 0
            ] = resistant_crops[0]
            crop_calendar_per_farmer[indices[n_same + n_first_resistant :], 0, 0] = (
                resistant_crops[1]
            )

            return crop_calendar_per_farmer

        # Reduces certain crops of the same GCAM category to the one that is most common in that region
        # First line checks which crop is most common, second denotes which crops will be replaced by the most common one
        if reduce_crops:
            # Conversion based on the classification in table S1 by Yoon, J., Voisin, N., Klassert, C., Thurber, T., & Xu, W. (2024).
            # Representing farmer irrigated crop area adaptation in a large-scale hydrological model. Hydrology and Earth
            # System Sciences, 28(4), 899–916. https://doi.org/10.5194/hess-28-899-2024

            # Replace fodder with the most common grain crop
            most_common_check = [BARLEY, RYE, MILLET, SORGHUM]
            replaced_value = [FODDER_GRASSES]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change the grain crops to one
            most_common_check = [BARLEY, RYE, MILLET, SORGHUM]
            replaced_value = [BARLEY, RYE, MILLET, SORGHUM]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change other annual / misc to one
            most_common_check = [GROUNDNUTS, COCOA, COFFEE, OTHERS_ANNUAL]
            replaced_value = [GROUNDNUTS, COCOA, COFFEE, OTHERS_ANNUAL]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change oils to one
            most_common_check = [SOYBEANS, SUNFLOWER, RAPESEED]
            replaced_value = [SOYBEANS, SUNFLOWER, RAPESEED]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change tubers to one
            most_common_check = [POTATOES, CASSAVA]
            replaced_value = [POTATOES, CASSAVA]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Reduce sugar crops to one
            most_common_check = [SUGAR_CANE, SUGAR_BEETS]
            replaced_value = [SUGAR_CANE, SUGAR_BEETS]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            # Change perennial to annual, otherwise counted double in esa dataset
            most_common_check = [OIL_PALM, OTHERS_PERENNIAL]
            replaced_value = [OIL_PALM, OTHERS_PERENNIAL]
            crop_calendar_per_farmer = replace_crop(
                crop_calendar_per_farmer, most_common_check, replaced_value
            )

            unique_rows = np.unique(crop_calendar_per_farmer, axis=0)
            values = unique_rows[:, 0, 0]
            unique_values, counts = np.unique(values, return_counts=True)

            # this part asserts that the crop calendar is correctly set up
            # particulary that no two crops are planted at the same time
            for farmer_crop_calender in crop_calendar_per_farmer:
                farmer_crop_calender = farmer_crop_calender[
                    farmer_crop_calender[:, -1] != -1
                ]
                if farmer_crop_calender.shape[0] > 1:
                    assert (
                        np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                        == farmer_crop_calender.shape[0]
                    )

            # duplicates = unique_values[counts > 1]
            # if len(duplicates) > 0:
            #     for duplicate in duplicates:
            #         crop_calendar_per_farmer = unify_crop_variants(
            #             crop_calendar_per_farmer, duplicate
            #         )

            # this part asserts that the crop calendar is correctly set up
            # particulary that no two crops are planted at the same time
            for farmer_crop_calender in crop_calendar_per_farmer:
                farmer_crop_calender = farmer_crop_calender[
                    farmer_crop_calender[:, -1] != -1
                ]
                if farmer_crop_calender.shape[0] > 1:
                    assert (
                        np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                        == farmer_crop_calender.shape[0]
                    )

        if replace_base:
            base_crops = [WHEAT]
            resistant_crops = [WHEAT_DROUGHT, WHEAT_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [MAIZE]
            resistant_crops = [MAIZE_DROUGHT, MAIZE_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [RICE]
            resistant_crops = [RICE_DROUGHT, RICE_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [SOYBEANS]
            resistant_crops = [SOYBEANS_DROUGHT, SOYBEANS_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

            base_crops = [POTATOES]
            resistant_crops = [POTATOES_DROUGHT, POTATOES_FLOOD]

            crop_calendar_per_farmer = insert_other_variant_crop(
                crop_calendar_per_farmer, base_crops, resistant_crops
            )

        assert crop_calendar_per_farmer[:, :, 3].max() == 0

        # this part asserts that the crop calendar is correctly set up
        # particulary that no two crops are planted at the same time
        for farmer_crop_calender in crop_calendar_per_farmer:
            farmer_crop_calender = farmer_crop_calender[
                farmer_crop_calender[:, -1] != -1
            ]
            if farmer_crop_calender.shape[0] > 1:
                assert (
                    np.unique(farmer_crop_calender[:, [1, 3]], axis=0).shape[0]
                    == farmer_crop_calender.shape[0]
                )

        self.set_binary(crop_calendar_per_farmer, name="agents/farmers/crop_calendar")
        self.set_binary(
            np.full_like(is_irrigated, 1, dtype=np.int32),
            name="agents/farmers/crop_calendar_rotation_years",
        )

    def assign_crops_irrigation_farmers(self, year=2000):
        # Define the directory and file paths
        data_dir = self.preprocessing_dir / "crops" / "MIRCA2000"
        crop_area_file = data_dir / "crop_area_fraction_all_years.nc"
        crop_irr_fraction_file = data_dir / "crop_irrigated_fraction_all_years.nc"

        # Load the DataArrays
        all_years_fraction_da = xr.open_dataarray(crop_area_file)
        all_years_irrigated_fraction_da = xr.open_dataarray(crop_irr_fraction_file)

        farmer_locations = get_farm_locations(
            self.subgrid["agents/farmers/farms"], method="centroid"
        )

        crop_dict = {
            "Wheat": 0,
            "Maize": 1,
            "Rice": 2,
            "Barley": 3,
            "Rye": 4,
            "Millet": 5,
            "Sorghum": 6,
            "Soybeans": 7,
            "Sunflower": 8,
            "Potatoes": 9,
            "Cassava": 10,
            "Sugar_cane": 11,
            "Sugar_beet": 12,
            "Oil_palm": 13,
            "Rapeseed": 14,
            "Groundnuts": 15,
            "Pulses": 16,
            "Cotton": 20,
            "Cocoa": 21,
            "Coffee": 22,
            "Others_perennial": 23,
            "Fodder": 24,
            "Others_annual": 25,
        }

        area_fraction_2000 = all_years_fraction_da.sel(year=str(year))
        irrigated_fraction_2000 = all_years_irrigated_fraction_da.sel(year=str(year))
        # Fill nas as there is no diff between 0 or na in code and can cause issues
        area_fraction_2000 = area_fraction_2000.fillna(0)
        irrigated_fraction_2000 = irrigated_fraction_2000.fillna(0)

        crops_in_dataarray = area_fraction_2000.coords["crop"].values

        grid_id_da = create_grid_cell_id_array(all_years_fraction_da)

        ny, nx = area_fraction_2000.sizes["y"], area_fraction_2000.sizes["x"]

        n_cells = grid_id_da.max().item()

        farmer_cells = sample_from_map(
            grid_id_da.values,
            farmer_locations,
            grid_id_da.raster.transform.to_gdal(),
        )

        crop_area_fractions = sample_from_map(
            area_fraction_2000.values,
            farmer_locations,
            area_fraction_2000.raster.transform.to_gdal(),
        )
        crop_irrigated_fractions = sample_from_map(
            irrigated_fraction_2000.values,
            farmer_locations,
            irrigated_fraction_2000.raster.transform.to_gdal(),
        )

        n_farmers = self.binary["agents/farmers/id"].size

        # Prepare empty crop arrays
        farmer_crops = np.full(n_farmers, -1, dtype=np.int32)
        farmer_irrigated = np.full(n_farmers, 0, dtype=np.bool_)

        for i in range(n_cells):
            farmers_cell_mask = farmer_cells == i
            nr_farmers_cell = np.count_nonzero(farmers_cell_mask)
            if nr_farmers_cell == 0:
                continue
            crop_area_fraction = crop_area_fractions[farmer_cells == i][0]
            crop_irrigated_fraction = crop_irrigated_fractions[farmer_cells == i][0]

            if crop_area_fraction.sum() == 0:
                # Expand the search radius until valid data is found
                found_valid_neighbor = False
                max_radius = max(nx, ny)  # Maximum possible radius
                radius = 1
                while not found_valid_neighbor and radius <= max_radius:
                    neighbor_ids = get_neighbor_cell_ids(i, nx, ny, radius)
                    for neighbor_id in neighbor_ids:
                        if neighbor_id not in farmer_cells:
                            continue

                        neighbor_crop_area_fraction = crop_area_fractions[
                            farmer_cells == neighbor_id
                        ][0]
                        if neighbor_crop_area_fraction.sum() != 0:
                            # Found valid neighbor
                            crop_area_fraction = neighbor_crop_area_fraction
                            crop_irrigated_fraction = crop_irrigated_fractions[
                                farmer_cells == neighbor_id
                            ][0]
                            found_valid_neighbor = True
                            break
                    if not found_valid_neighbor:
                        radius += 1  # Increase the search radius
                if not found_valid_neighbor:
                    # No valid data found even after expanding radius
                    print(
                        f"No valid data found for cell {i} after searching up to radius {radius - 1}."
                    )
                    continue  # Skip this cell

            farmer_indices_in_cell = np.where(farmers_cell_mask)[0]

            # ensure fractions sum to 1
            area_per_crop_rotation = crop_area_fraction / crop_area_fraction.sum()

            farmer_crop_rotations_idx = np.random.choice(
                np.arange(len(area_per_crop_rotation)),
                size=len(farmer_indices_in_cell),
                replace=True,
                p=area_per_crop_rotation,
            )

            # Map sampled indices to crop names using crops_in_dataarray
            farmer_crop_names = crops_in_dataarray[farmer_crop_rotations_idx]
            # Map crop names to integer codes using crop_dict
            farmer_crop_codes = [
                crop_dict[crop_name] for crop_name in farmer_crop_names
            ]
            # assign to farmers
            farmer_crops[farmer_indices_in_cell] = farmer_crop_codes

            # Determine irrigating farmers
            chosen_crops = np.unique(farmer_crop_rotations_idx)

            for c in chosen_crops:
                # Indices of farmers in the cell assigned to crop c
                farmers_with_crop_c_in_cell = np.where(farmer_crop_rotations_idx == c)[
                    0
                ]
                N_c = len(farmers_with_crop_c_in_cell)
                f_c = crop_irrigated_fraction[c]
                if np.isnan(f_c) or f_c <= 0:
                    continue  # No irrigation for this crop
                N_irrigated = int(round(N_c * f_c))
                if N_irrigated > 0:
                    # Randomly select N_irrigated farmers from the N_c farmers
                    irrigated_indices_in_cell = np.random.choice(
                        farmers_with_crop_c_in_cell, size=N_irrigated, replace=False
                    )
                    # Get the overall farmer indices
                    overall_farmer_indices = farmer_indices_in_cell[
                        irrigated_indices_in_cell
                    ]
                    # Set irrigation status to True for these farmers
                    farmer_irrigated[overall_farmer_indices] = True

        assert not (farmer_crops == -1).any(), (
            "Error: some farmers have no crops assigned"
        )

        return farmer_crops, farmer_irrigated

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
        grid_id_da = create_grid_cell_id_array(fraction_sw_irrigation_data)
        ny, nx = (
            fraction_sw_irrigation_data.sizes["y"],
            fraction_sw_irrigation_data.sizes["x"],
        )

        n_cells = grid_id_da.max().item()
        n_farmers = self.binary["agents/farmers/id"].size

        farmer_cells = sample_from_map(
            grid_id_da.values,
            farmer_locations,
            grid_id_da.raster.transform.to_gdal(),
        )
        fraction_sw_irrigation_farmers = sample_from_map(
            fraction_sw_irrigation_data.values,
            farmer_locations,
            fraction_sw_irrigation_data.raster.transform.to_gdal(),
        )
        fraction_gw_irrigation_farmers = sample_from_map(
            fraction_gw_irrigation_data.values,
            farmer_locations,
            fraction_gw_irrigation_data.raster.transform.to_gdal(),
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
                    neighbor_ids = get_neighbor_cell_ids(i, nx, ny)
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

        self.set_binary(adaptations, name="agents/farmers/adaptations")

    def setup_population(self):
        populaton_map = self.data_catalog.get_rasterdataset(
            "ghs_pop_2020_54009_v2023a", bbox=self.bounds
        )
        populaton_map_values = np.round(populaton_map.values).astype(np.int32)
        populaton_map_values[populaton_map_values < 0] = 0  # -200 is nodata value

        locations, sizes = generate_locations(
            population=populaton_map_values,
            geotransform=populaton_map.raster.transform.to_gdal(),
            mean_household_size=5,
        )

        transformer = pyproj.Transformer.from_crs(
            populaton_map.raster.crs, self.epsg, always_xy=True
        )
        locations[:, 0], locations[:, 1] = transformer.transform(
            locations[:, 0], locations[:, 1]
        )

        # sample_locatons = locations[::10]
        # import matplotlib.pyplot as plt
        # from scipy.stats import gaussian_kde

        # xy = np.vstack([sample_locatons[:, 0], sample_locatons[:, 1]])
        # z = gaussian_kde(xy)(xy)
        # plt.scatter(sample_locatons[:, 0], sample_locatons[:, 1], c=z, s=100)
        # plt.savefig('population.png')

        self.set_binary(sizes, name="agents/households/sizes")
        self.set_binary(locations, name="agents/households/locations")

        return None

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
                        self.logger.debug(
                            f"{response['meta']['created_files']}/{response['meta']['total_files']} files prepared on ISIMIP server for {variable}, waiting 60 seconds before retrying"
                        )
                    elif response["status"] == "queued":
                        self.logger.debug(
                            f"Data preparation queued for {variable} on ISIMIP server, waiting 60 seconds before retrying"
                        )
                    elif response["status"] == "failed":
                        self.logger.debug(
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

    def setup_hydrodynamics(
        self,
        land_cover="esa_worldcover_2021_v200",
        include_coastal=True,
        DEMs=[{"elevtn": "fabdem", "zmin": 0.001}, {"elevtn": "gebco"}],
    ):
        assert isinstance(DEMs, list)

        hydrodynamics_data_catalog = DataCatalog()

        bounds = tuple(self.geoms["routing/subbasins"].total_bounds)

        self.set_dict(DEMs, name="hydrodynamics/DEM_config")
        for DEM in DEMs:
            DEM_raster = self.data_catalog.get_rasterdataset(
                DEM["elevtn"],
                bbox=bounds,
                buffer=100,
                single_var_as_array=False,
            ).compute()
            assert len(DEM_raster.data_vars) == 1
            DEM_raster = DEM_raster.rename({list(DEM_raster.data_vars)[0]: "elevtn"})

            # hydromt-sfincs requires the data to be a Dataset. This code here makes
            # data with only one variable a Dataarray, which is not supported in hydromt-sfincs
            # therefore we add a dummy variable to the data thus forcing the data to
            # be considered a Dataset
            DEM_raster["_dummy"] = 0
            self.set_forcing(
                DEM_raster,
                name=f"hydrodynamics/DEM/{DEM['elevtn']}",
                split_dataset=False,
                byteshuffle=True,
            )

            hydrodynamics_data_catalog.add_source(
                DEM["elevtn"],
                RasterDatasetAdapter(
                    path=Path(self.root)
                    / "hydrodynamics"
                    / "DEM"
                    / f"{DEM['elevtn']}.zarr.zip",
                    crs=self.data_catalog.get_source(
                        DEM["elevtn"]
                    ).crs,  # perhaps set crs in dataset itself
                    meta=self.data_catalog.get_source(DEM["elevtn"]).meta,
                    driver="zarr",
                ),  # hydromt likes absolute paths
            )

        # landcover
        esa_worldcover = self.data_catalog.get_rasterdataset(
            land_cover,
            bbox=bounds,
            buffer=200,  # 2 km buffer
        ).chunk({"x": XY_CHUNKSIZE, "y": XY_CHUNKSIZE})
        del esa_worldcover.attrs["_FillValue"]
        esa_worldcover.name = "lulc"
        esa_worldcover = esa_worldcover.to_dataset()
        esa_worldcover["_dummy"] = 0
        self.set_forcing(
            esa_worldcover,
            name="hydrodynamics/esa_worldcover",
            split_dataset=False,
            byteshuffle=False,
        )

        hydrodynamics_data_catalog.add_source(
            "esa_worldcover",
            RasterDatasetAdapter(
                path=Path(self.root) / "hydrodynamics" / "esa_worldcover.zarr.zip",
                meta=self.data_catalog.get_source(land_cover).meta,
                driver="zarr",
            ),  # hydromt likes absolute paths
        )

        if include_coastal:
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

            water_levels = water_levels.sel(stations=station_ids).compute()

            assert len(water_levels.stations) > 0, (
                "No stations found in the region. If no stations should be set, set include_coastal=False"
            )

            path = self.set_forcing(
                water_levels,
                name="hydrodynamics/waterlevel",
                split_dataset=False,
                is_spatial_dataset=False,
                time_chunksize=24 * 6,  # 10 minute data
                byteshuffle=True,
            )
            hydrodynamics_data_catalog.add_source(
                "waterlevel",
                DatasetAdapter(
                    path=Path(self.root) / path,
                    meta=self.data_catalog.get_source("GTSM").meta,
                    driver="zarr",
                ),  # hydromt likes absolute paths
            )

        # TEMPORARY HACK UNTIL HYDROMT IS FIXED
        # SEE: https://github.com/Deltares/hydromt/issues/832
        def to_yml(
            self,
            path: Union[str, Path],
            root: str = "auto",
            source_names: Optional[List] = None,
            used_only: bool = False,
            meta: Optional[Dict] = None,
        ) -> None:
            """Write data catalog to yaml format.

            Parameters
            ----------
            path: str, Path
                yaml output path.
            root: str, Path, optional
                Global root for all relative paths in yaml file.
                If "auto" (default) the data source paths are relative to the yaml
                output ``path``.
            source_names: list, optional
                List of source names to export, by default None in which case all sources
                are exported. This argument is ignored if `used_only=True`.
            used_only: bool, optional
                If True, export only data entries kept in used_data list, by default False.
            meta: dict, optional
                key-value pairs to add to the data catalog meta section, such as 'version',
                by default empty.
            """
            import yaml

            meta = meta or []
            yml_dir = os.path.dirname(os.path.abspath(path))
            if root == "auto":
                root = yml_dir
            data_dict = self.to_dict(
                root=root, source_names=source_names, meta=meta, used_only=used_only
            )
            if str(root) == yml_dir:
                data_dict["meta"].pop(
                    "root", None
                )  # remove root if it equals the yml_dir
            if data_dict:
                with open(path, "w") as f:
                    yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)
            else:
                self.logger.info("The data catalog is empty, no yml file is written.")

        hydrodynamics_data_catalog.to_yml = to_yml

        hydrodynamics_data_catalog.to_yml(
            hydrodynamics_data_catalog,
            Path(self.root) / "hydrodynamics" / "data_catalog.yml",
        )
        return None

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
        self.set_table(risk_scaling_factors, name="hydrodynamics/risk_scaling_factors")

    def setup_discharge_observations(self, files):
        transform = self.grid.raster.transform

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
        discharge_data.name = "discharge"
        self.set_forcing(
            discharge_data,
            name="observations/discharge",
            split_dataset=False,
            is_spatial_dataset=False,
            time_chunksize=1e99,  # no chunking
        )

    def _write_grid(
        self,
        grid,
        var,
        files,
        is_updated,
        y_chunksize=XY_CHUNKSIZE,
        x_chunksize=XY_CHUNKSIZE,
    ):
        if is_updated[var]["updated"]:
            self.logger.info(f"Writing {var}")
            filename = var + ".zarr.zip"
            files[var] = filename
            is_updated[var]["filename"] = filename
            filepath = Path(self.root, filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if grid.dtype == "float64":
                grid = grid.astype("float32")

            # zarr cannot handle / in variable names
            grid.name = "data"
            assert hasattr(grid, "spatial_ref")

            # Rechunk data variable if needed
            if grid.chunks is None:
                # Grid is not chunked, specify chunks and chunk the grid
                chunksizes = {
                    "y": min(grid.y.size, y_chunksize),
                    "x": min(grid.x.size, x_chunksize),
                }
                chunks_tuple = tuple(
                    (
                        chunksizes[dim]
                        if dim in chunksizes
                        else max(getattr(grid, dim).size, 1)
                    )
                    for dim in grid.dims
                )
                grid = grid.chunk(chunksizes)
                data_chunks = chunks_tuple
            else:
                # Grid is already chunked; use existing chunks
                data_chunks = tuple(
                    s[0] if len(set(s)) == 1 else s for s in grid.chunks
                )

            # Build the encoding dictionary for the data variable
            encoding = {
                grid.name: {
                    "compressor": Blosc(
                        cname="zstd", clevel=9, shuffle=Blosc.NOSHUFFLE
                    ),  # no shuffling is most efficient,
                    # Only specify chunks if the grid was not already chunked
                    **({"chunks": data_chunks} if grid.chunks is None else {}),
                }
            }

            # **Compute coordinate variables to avoid chunking issues**
            for coord in grid.coords:
                coord_da = grid.coords[coord]
                if coord_da.ndim > 0 and coord_da.chunks is not None:
                    # Option 1: Rechunk the coordinate variable to match data variable
                    # grid.coords[coord] = coord_da.rechunk(data_chunks[-coord_da.ndim:])
                    # Option 2: Compute the coordinate variable (since it's small)
                    grid.coords[coord] = coord_da.compute()
                # Include coordinate in encoding without specifying chunks
                encoding[coord] = {}

            # Now write to Zarr
            grid.to_zarr(
                filepath,
                mode="w",
                encoding=encoding,
            )

            # rasterio does not support boolean data, which is why
            # we convert it to uint8 before writing. These files are
            # only used for displaying purposes so they do not affect the
            # actual model (re-)building
            if grid.dtype == bool:
                grid = grid.astype(np.uint8)
                grid = grid.rio.set_nodata(255)
            grid.rio.to_raster(
                filepath.with_suffix(".tif"), compress="DEFLATE", zlevel=9
            )

    def write_grid(self):
        self._assert_write_mode
        for var, grid in self.grid.items():
            grid["spatial_ref"] = self.grid.spatial_ref
            if var == "spatial_ref":
                continue
            self._write_grid(grid, var, self.files["grid"], self.is_updated["grid"])

    def write_subgrid(self):
        self._assert_write_mode
        for var, grid in self.subgrid.items():
            if var == "spatial_ref":
                continue
            grid["spatial_ref"] = self.subgrid.spatial_ref
            self._write_grid(
                grid,
                var,
                self.files["subgrid"],
                self.is_updated["subgrid"],
                XY_CHUNKSIZE * self.subgrid_factor,
                XY_CHUNKSIZE * self.subgrid_factor,
            )

    def write_region_subgrid(self):
        self._assert_write_mode
        for var, grid in self.region_subgrid.items():
            grid["spatial_ref"] = self.region_subgrid.spatial_ref
            if var == "spatial_ref":
                continue
            self._write_grid(
                grid,
                var,
                self.files["region_subgrid"],
                self.is_updated["region_subgrid"],
                XY_CHUNKSIZE * self.subgrid_factor,
                XY_CHUNKSIZE * self.subgrid_factor,
            )

    def write_MERIT_grid(self):
        self._assert_write_mode
        for var, grid in self.MERIT_grid.items():
            if var == "spatial_ref":
                continue
            grid["spatial_ref"] = self.MERIT_grid.spatial_ref
            self._write_grid(
                grid, var, self.files["MERIT_grid"], self.is_updated["MERIT_grid"]
            )

    def write_forcing_to_zarr(
        self,
        var,
        forcing,
        y_chunksize=XY_CHUNKSIZE,
        x_chunksize=XY_CHUNKSIZE,
        byteshuffle=False,
        time_chunksize=1,
        is_spatial_dataset=True,
    ) -> None:
        self.logger.info(f"Write {var}")

        destination = var + ".zarr.zip"
        self.files["forcing"][var] = destination
        self.is_updated["forcing"][var]["filename"] = destination

        dst_file = Path(self.root, destination)
        if dst_file.exists():
            dst_file.unlink()
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if is_spatial_dataset and forcing.rio.crs is None:
            forcing = forcing.rio.write_crs(self.crs).rio.write_coordinate_system()

        if isinstance(forcing, xr.DataArray):
            # if data is float64, convert to float32
            if forcing.dtype == np.float64:
                forcing = forcing.astype(np.float32)
        elif isinstance(forcing, xr.Dataset):
            for var in forcing.data_vars:
                if forcing[var].dtype == np.float64:
                    forcing[var] = forcing[var].astype(np.float32)
        else:
            raise ValueError("forcing must be a DataArray or Dataset")

        # write netcdf to temporary file
        with tempfile.NamedTemporaryFile(suffix=".zarr.zip") as tmp_file:
            if "time" in forcing.dims:
                with ProgressBar(dt=10):  # print progress bar every 10 seconds
                    if is_spatial_dataset:
                        assert forcing.dims[1] == "y" and forcing.dims[2] == "x", (
                            "y and x dimensions must be second and third, otherwise xarray will not chunk correctly"
                        )
                        chunksizes = {
                            "time": min(forcing.time.size, time_chunksize),
                            "y": min(forcing.y.size, y_chunksize),
                            "x": min(forcing.x.size, x_chunksize),
                        }
                    else:
                        chunksizes = {"time": min(forcing.time.size, time_chunksize)}

                    forcing.chunk(chunksizes).to_zarr(
                        tmp_file.name,
                        mode="w",
                        encoding={
                            forcing.name: {
                                "compressor": Blosc(
                                    cname="zstd",
                                    clevel=9,
                                    shuffle=Blosc.SHUFFLE
                                    if byteshuffle
                                    else Blosc.NOSHUFFLE,
                                ),
                                "chunks": (
                                    (
                                        chunksizes[dim]
                                        if dim in chunksizes
                                        else max(getattr(forcing, dim).size, 1)
                                    )
                                    for dim in forcing.dims
                                ),
                            }
                        },
                    )

                    # move file to final location
                    shutil.copy(tmp_file.name, dst_file)
                return xr.open_dataset(dst_file, chunks={}, engine="zarr")[forcing.name]
            else:
                if isinstance(forcing, xr.DataArray):
                    name = forcing.name
                    encoding = {
                        forcing.name: {
                            "compressor": Blosc(
                                cname="zstd",
                                clevel=9,
                                shuffle=Blosc.SHUFFLE
                                if byteshuffle
                                else Blosc.NOSHUFFLE,
                            )
                        }
                    }
                elif isinstance(forcing, xr.Dataset):
                    assert len(forcing.data_vars) > 0, (
                        "forcing must have more than one variable or name must be set"
                    )
                    encoding = {
                        var: {
                            "compressor": Blosc(
                                cname="zstd",
                                clevel=9,
                                shuffle=Blosc.SHUFFLE
                                if byteshuffle
                                else Blosc.NOSHUFFLE,
                            )
                        }
                        for var in forcing.data_vars
                    }
                else:
                    raise ValueError("forcing must be a DataArray or Dataset")
                forcing.to_zarr(
                    tmp_file.name,
                    mode="w",
                    encoding=encoding,
                )

                if isinstance(forcing, xr.DataArray):
                    # also export to tif for easier visualization
                    forcing.rio.to_raster(dst_file.with_suffix(".tif"))
                elif isinstance(forcing, xr.Dataset) and len(forcing.data_vars) == 1:
                    # also export to tif for easier visualization, but only if there is one variable
                    forcing[list(forcing.data_vars)[0]].rio.to_raster(
                        dst_file.with_suffix(".tif")
                    )

                # move file to final location
                shutil.copy(tmp_file.name, dst_file)

                ds = xr.open_dataset(dst_file, chunks={}, engine="zarr")
                if isinstance(forcing, xr.DataArray):
                    return ds[name]
                else:
                    return ds

    def write_forcing(self) -> None:
        self._assert_write_mode
        for var in self.forcing:
            forcing = self.forcing[var]
            if self.is_updated["forcing"][var]["updated"]:
                self.write_forcing_to_zarr(var, forcing)

    def write_table(self):
        if len(self.table) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.table.items():
                if self.is_updated["table"][name]["updated"]:
                    fn = os.path.join(name + ".parquet")
                    self.logger.debug(f"Writing file {fn}")
                    self.files["table"][name] = fn
                    self.is_updated["table"][name]["filename"] = fn
                    self.logger.debug(f"Writing file {fn}")
                    fp = Path(self.root, fn)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    data.to_parquet(fp, engine="pyarrow")

    def write_binary(self):
        if len(self.binary) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.binary.items():
                if self.is_updated["binary"][name]["updated"]:
                    fn = os.path.join(name + ".npz")
                    self.logger.debug(f"Writing file {fn}")
                    self.files["binary"][name] = fn
                    self.is_updated["binary"][name]["filename"] = fn
                    self.logger.debug(f"Writing file {fn}")
                    fp = Path(self.root, fn)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(fp, data=data)

    def write_dict(self):
        def convert_timestamp_to_string(timestamp):
            return timestamp.isoformat()

        if len(self.dict) == 0:
            self.logger.debug("No table data found, skip writing.")
        else:
            self._assert_write_mode
            for name, data in self.dict.items():
                if self.is_updated["dict"][name]["updated"]:
                    fn = os.path.join(name + ".json")
                    self.files["dict"][name] = fn
                    self.is_updated["dict"][name]["filename"] = fn
                    self.logger.debug(f"Writing file {fn}")
                    output_path = Path(self.root) / fn
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(data, f, default=convert_timestamp_to_string)

    def write_geoms(self, fn: str = "{name}.geoparquet", **kwargs) -> None:
        """Write model geometries to a vector file (by default geoparquet) at <root>/<fn>

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'geoms/{name}.gpkg'
        """
        if len(self._geoms) == 0:
            self.logger.debug("No geoms data found, skip writing.")
            return
        else:
            self._assert_write_mode
            for name, gdf in self._geoms.items():
                if self.is_updated["geoms"][name]["updated"]:
                    self.logger.debug(f"Writing file {fn.format(name=name)}")
                    self.files["geoms"][name] = fn.format(name=name)
                    _fn = os.path.join(self.root, fn.format(name=name))
                    if not os.path.isdir(os.path.dirname(_fn)):
                        os.makedirs(os.path.dirname(_fn))
                    self.is_updated["geoms"][name]["filename"] = _fn
                    gdf.to_parquet(_fn, **kwargs)

    def set_table(self, table, name, update=True):
        self.is_updated["table"][name] = {"updated": update}
        self.table[name] = table

    def set_binary(self, data, name, update=True):
        self.is_updated["binary"][name] = {"updated": update}
        self.binary[name] = data

    def set_dict(self, data, name, update=True):
        self.is_updated["dict"][name] = {"updated": update}
        self.dict[name] = data

    def write_files(self):
        with open(Path(self.root, "files.json"), "w") as f:
            json.dump(self.files, f, indent=4, cls=PathEncoder)

    def write(self):
        self.write_geoms()
        self.write_binary()
        self.write_table()
        self.write_dict()

        self.write_grid()
        self.write_subgrid()
        self.write_region_subgrid()
        self.write_MERIT_grid()

        self.write_forcing()

        self.write_files()

    def read_files(self):
        files_is_empty = all(len(v) == 0 for v in self.files.values())
        if files_is_empty:
            with open(Path(self.root, "files.json"), "r") as f:
                self.files = json.load(f)

    def read_geoms(self):
        self.read_files()
        for name, fn in self.files["geoms"].items():
            geom = gpd.read_parquet(Path(self.root, fn))
            self.set_geoms(geom, name=name, update=False)

    def read_binary(self):
        self.read_files()
        for name, fn in self.files["binary"].items():
            binary = np.load(Path(self.root, fn))["data"]
            self.set_binary(binary, name=name, update=False)

    def read_table(self):
        self.read_files()
        for name, fn in self.files["table"].items():
            table = pd.read_parquet(Path(self.root, fn))
            self.set_table(table, name=name, update=False)

    def read_dict(self):
        self.read_files()
        for name, fn in self.files["dict"].items():
            with open(Path(self.root, fn), "r") as f:
                d = json.load(f)
            self.set_dict(d, name=name, update=False)

    def _read_grid(self, fn: str, name: str) -> xr.Dataset:
        if fn.endswith(".zarr.zip"):
            engine = "zarr"
            da = xr.load_dataset(
                Path(self.root) / fn, mask_and_scale=False, engine=engine
            )
            da = da.rename({"data": name})
        elif fn.endswith(".tif"):
            da = xr.load_dataset(Path(self.root) / fn, mask_and_scale=False)
            da = da.rename(  # deleted decode_cf=False
                {"band_data": name}
            )
            if "band" in da.dims and da.band.size == 1:
                # drop band dimension
                da = da.squeeze("band")
                # drop band coordinate
                da = da.drop_vars("band")
            da.x.attrs = {
                "long_name": "latitude of grid cell center",
                "units": "degrees_north",
            }
            da.y.attrs = {
                "long_name": "longitude of grid cell center",
                "units": "degrees_east",
            }
        else:
            raise ValueError(f"Unsupported file format: {fn}")
        return da

    def read_grid(self) -> None:
        for name, fn in self.files["grid"].items():
            data = self._read_grid(fn, name=name)
            self.set_grid(data, name=name, update=False)

    def read_subgrid(self) -> None:
        for name, fn in self.files["subgrid"].items():
            data = self._read_grid(fn, name=name)
            self.set_subgrid(data, name=name, update=False)

    def read_region_subgrid(self) -> None:
        for name, fn in self.files["region_subgrid"].items():
            data = self._read_grid(fn, name=name)
            self.set_region_subgrid(data, name=name, update=False)

    def read_MERIT_grid(self) -> None:
        for name, fn in self.files["MERIT_grid"].items():
            data = self._read_grid(fn, name=name)
            self.set_MERIT_grid(data, name=name, update=False)

    def read_forcing(self) -> None:
        self.read_files()
        for name, fn in self.files["forcing"].items():
            with xr.open_dataset(Path(self.root) / fn, chunks={}, engine="zarr") as ds:
                data_vars = set(ds.data_vars)
                data_vars.discard("spatial_ref")
                if len(data_vars) == 1:
                    self.set_forcing(ds[name.split("/")[-1]], name=name, update=False)
                else:
                    self.set_forcing(ds, name=name, update=False, split_dataset=False)
        return None

    def read(self):
        with suppress_logging_warning(self.logger):
            self.read_files()

            self.read_geoms()
            self.read_binary()
            self.read_table()
            self.read_dict()

            self.read_grid()
            self.read_subgrid()
            self.read_region_subgrid()
            self.read_MERIT_grid()

            self.read_forcing()

    def set_geoms(self, geoms, name, update=True):
        self.is_updated["geoms"][name] = {"updated": update}
        super().set_geoms(geoms, name=name)
        return self.geoms[name]

    def set_forcing(
        self,
        data,
        name: str,
        update=True,
        write=True,
        x_chunksize=XY_CHUNKSIZE,
        y_chunksize=XY_CHUNKSIZE,
        time_chunksize=1,
        byteshuffle=False,
        is_spatial_dataset=True,
        split_dataset=True,
        *args,
        **kwargs,
    ):
        if isinstance(data, xr.DataArray):
            assert data.name == name.split("/")[-1]
        self.is_updated["forcing"][name] = {"updated": update}
        if update and write:
            data = self.write_forcing_to_zarr(
                name,
                data,
                x_chunksize=x_chunksize,
                y_chunksize=y_chunksize,
                time_chunksize=time_chunksize,
                is_spatial_dataset=is_spatial_dataset,
                byteshuffle=byteshuffle,
            )
            self.is_updated["forcing"][name]["updated"] = False
        super().set_forcing(
            data, name=name, split_dataset=split_dataset, *args, **kwargs
        )
        return self.files["forcing"][name]

    def _set_grid(
        self,
        grid,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        assert grid is not None
        # NOTE: variables in a dataset are not longer renamed as used to be the case in
        # set_staticmaps
        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, np.ndarray):
            if data.shape != grid.raster.shape:
                raise ValueError("Shape of data and grid maps do not match")
            data = xr.DataArray(dims=grid.raster.dims, data=data, name=name)
        if isinstance(data, xr.DataArray):
            if name is not None:  # rename
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")

        if len(grid.data_vars) > 0:
            data = self.snap_to_grid(data, grid)
        # force read in r+ mode
        if len(grid) == 0:  # trigger init / read
            # copy spatial reference from data
            grid["spatial_ref"] = data["spatial_ref"]
            var = data[name]
            if "spatial_ref" in data[name].coords:
                grid[name] = var.drop_vars("spatial_ref")
            else:
                grid[name] = var
            # copy attributes from data
            grid[name].attrs = data[name].attrs
        else:
            for dvar in data.data_vars:
                if dvar == "spatial_ref":
                    continue
                if dvar in grid:
                    if self._read:
                        self.logger.warning(f"Replacing grid map: {dvar}")
                    assert grid[dvar].shape == data[dvar].shape
                    assert (grid[dvar].y.values == data[dvar].y.values).all()
                    assert (grid[dvar].x.values == data[dvar].x.values).all()
                    grid = grid.drop_vars(dvar)

                assert CRS.from_wkt(data.spatial_ref.crs_wkt) == CRS.from_wkt(
                    grid.spatial_ref.crs_wkt
                )
                var = data[dvar]
                var.raster.set_crs(grid.raster.crs)
                if "spatial_ref" in var.coords:
                    var = var.drop_vars("spatial_ref")

                grid[dvar] = var
                grid[dvar].attrs = data[dvar].attrs

        return grid

    def set_grid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["grid"][name] = {"updated": update}
        super().set_grid(data, name=name)
        return self.grid[name]

    def set_subgrid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["subgrid"][name] = {"updated": update}
        self.subgrid = self._set_grid(self.subgrid, data, name=name)
        return self.subgrid[name]

    def set_region_subgrid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["region_subgrid"][name] = {"updated": update}
        self.region_subgrid = self._set_grid(self.region_subgrid, data, name=name)
        return self.region_subgrid[name]

    def set_MERIT_grid(
        self, data: Union[xr.DataArray, xr.Dataset, np.ndarray], name: str, update=True
    ) -> None:
        self.is_updated["MERIT_grid"][name] = {"updated": update}
        self.MERIT_grid = self._set_grid(self.MERIT_grid, data, name=name)
        return self.MERIT_grid[name]

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
    def preprocessing_dir(self):
        return Path(self.root).parent / "preprocessing"
