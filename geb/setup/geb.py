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
import concurrent.futures
from hydromt.exceptions import NoDataException
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from affine import Affine
from pyproj import CRS
import xarray as xr
from dask.diagnostics import ProgressBar
import xclim.indices as xci
from dateutil.relativedelta import relativedelta
from contextlib import contextmanager
from calendar import monthrange
from numcodecs import Blosc

from hydromt.models.model_grid import GridModel
from hydromt.data_catalog import DataCatalog
from hydromt.data_adapter import (
    GeoDataFrameAdapter,
    RasterDatasetAdapter,
    DatasetAdapter,
)

from honeybees.library.raster import sample_from_map
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
from .workflows.population import generate_locations
from .workflows.crop_calendars import parse_MIRCA2000_crop_calendar
from .workflows.soilgrids import load_soilgrids
from .workflows.conversions import (
    M49_to_ISO3,
    SUPERWELL_NAME_TO_ISO3,
    GLOBIOM_NAME_TO_ISO3,
)
from .workflows.forcing import (
    reproject_and_apply_lapse_rate_temperature,
    reproject_and_apply_lapse_rate_pressure,
)

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
# Define the compressor using Blosc (e.g., Zstandard compression)

compressor = Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE)


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
        hydrography_fn: str,
        basin_index_fn: str,
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
        hydrography_fn : str
            Name of data source for hydrography data.
        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`.
        """

        assert (
            sub_grid_factor >= 10
        ), "sub_grid_factor must be larger than 10, because this is the resolution of the MERIT high-res DEM"
        assert sub_grid_factor % 10 == 0, "sub_grid_factor must be a multiple of 10"

        hydrography = self.data_catalog.get_rasterdataset(hydrography_fn)
        hydrography.x.attrs = {"long_name": "longitude", "units": "degrees_east"}
        hydrography.y.attrs = {"long_name": "latitude", "units": "degrees_north"}

        self.logger.info("Preparing 2D grid.")
        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        if kind in ["basin", "subbasin"]:
            # get basin geometry
            geom, xy = hydromt.workflows.get_basin_geometry(
                ds=hydrography, kind=kind, logger=self.logger, **region
            )
            region.update(xy=xy)
        elif "geom" in region:
            geom = region["geom"]
            if geom.crs is None:
                raise ValueError('Model region "geom" has no CRS')
            # merge regions when more than one geom is given
            if isinstance(geom, gpd.GeoDataFrame):
                geom = gpd.GeoDataFrame(geometry=[geom.unary_union], crs=geom.crs)
        else:
            raise ValueError(
                f"Region for grid must of kind [basin, subbasin], kind {kind} not understood."
            )

        self.logger.info(
            f"Approximate basin size in km2: {round(geom.to_crs(epsg=3857).area.sum() / 1e6, 2)}"
        )

        # Add region and grid to model
        self.set_geoms(geom, name="region")

        hydrography = hydrography.raster.clip_geom(geom, mask=True)

        ldd = hydrography["flwdir"].raster.reclassify(
            reclass_table=pd.DataFrame(
                index=[
                    0,
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    hydrography["flwdir"].raster.nodata,
                ],
                data={"ldd": [5, 6, 3, 2, 1, 4, 7, 8, 9, 0]},
            ),
            method="exact",
        )["ldd"]

        self.set_grid(ldd, name="routing/kinematic/ldd")
        self.set_grid(hydrography["uparea"], name="routing/kinematic/upstream_area")
        self.set_grid(hydrography["elevtn"], name="routing/kinematic/outflow_elevation")
        self.set_grid(
            xr.where(
                hydrography["rivlen_ds"] != -9999,
                hydrography["rivlen_ds"],
                np.nan,
                keep_attrs=True,
            ),
            name="routing/kinematic/channel_length",
        )
        self.set_grid(hydrography["rivslp"], name="routing/kinematic/channel_slope")

        # hydrography['mask'].raster.set_nodata(-1)
        self.set_grid((~hydrography["mask"]).astype(np.int8), name="areamaps/grid_mask")

        mask = self.grid["areamaps/grid_mask"]

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
        submask.raster.set_nodata(None)
        submask.data = repeat_grid(mask.data, sub_grid_factor)

        assert bounds_are_within(submask.raster.bounds, mask.raster.bounds)
        assert bounds_are_within(mask.raster.bounds, submask.raster.bounds)

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

        assert source in (
            "MIRCA2000",
        ), f"crop_variables_source {source} not understood, must be 'MIRCA2000'"
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

            total_years = donor_data.index.get_level_values("year").unique()

            if project_past_until_year:
                assert (
                    total_years[0] > project_past_until_year
                ), f"Extrapolation targets must not fall inside available data time series. Current lower limit is {total_years[0]}"
            if project_future_until_year:
                assert (
                    total_years[-1] < project_future_until_year
                ), f"Extrapolation targets must not fall inside available data time series. Current upper limit is {total_years[-1]}"

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
                if crop_name in translate_crop_names:
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

            data = self.inter_and_extrapolate_prices(
                prices_plus_crop_price_inflation, unique_regions
            )

            if (
                project_past_until_year is not None
                or project_future_until_year is not None
            ):
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
            data = data.reindex(
                columns=pd.MultiIndex.from_product(
                    [
                        self.geoms["areamaps/regions"]["region_id"],
                        data.columns,
                    ]
                ),
                level=1,
            )
            data = {
                "type": "time_series",
                "time": data.index.tolist(),
                "data": {
                    str(region_id): data[region_id].to_dict(orient="list")
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

        if "economics/ppp_conversion_rates" not in self.dict:
            raise ValueError("Please run setup_economic_data first")

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
                        data_out = new_data
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
                        data_out = new_data
                    else:
                        data_out = data_out.combine_first(new_data)

        # Drop columns that are all NaN
        data_out = data_out.dropna(axis=1, how="all")
        data_out = data_out.drop(columns=["ISO3"])
        return data_out

    def convert_price_using_ppp(
        self, price_source_LCU, ppp_factor_source, ppp_factor_target
    ):
        """
        Convert a price from one country's LCU to another's using PPP conversion factors.

        Parameters:
        - price_source_LCU (float): Array of the prices in the source country's local currency units (LCU).
        - ppp_factor_source (float): The PPP conversion factor for the source country.
        - ppp_factor_target (float): The PPP conversion factor for the target country.

        Returns:
        - float: The price in the target country's local currency units (LCU).
        """
        price_target_LCU = (price_source_LCU / ppp_factor_source) * ppp_factor_target
        return price_target_LCU

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
        assert (
            end_year != start_year
        ), "extra processed years must not be the same as data years"

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
            cultivation_costs, project_future_until_year=project_future_until_year
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
        `routing/kinematic/upstream_area` attribute of the grid. It then calculates the coefficient using the formula:

            C = 0.025 + 0.015 * (2 * A / U) + 0.030 * (Z / 2000)

        where C is the Manning's coefficient, A is the cell area, U is the upstream area, and Z is the elevation of the cell.

        The resulting Manning's coefficient is then set as the `routing/kinematic/mannings` attribute of the grid using the
        `set_grid()` method.
        """
        self.logger.info("Setting up Manning's coefficient")
        a = (2 * self.grid["areamaps/cell_area"]) / self.grid[
            "routing/kinematic/upstream_area"
        ]
        a = xr.where(a < 1, a, 1, keep_attrs=True)
        b = self.grid["routing/kinematic/outflow_elevation"] / 2000
        b = xr.where(b < 1, b, 1, keep_attrs=True)

        mannings = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            crs=self.crs,
            name="routing/kinematic/mannings",
            lazy=True,
        )
        mannings.data = 0.025 + 0.015 * a + 0.030 * b
        self.set_grid(mannings, mannings.name)

    def setup_channel_width(self, minimum_width: float) -> None:
        """
        Sets up the channel width for the model.

        Parameters
        ----------
        minimum_width : float
            The minimum channel width in meters.

        Notes
        -----
        This method sets up the channel width for the model by calculating the width of each channel based on the upstream
        area of each cell in the grid. It first retrieves the upstream area of each cell from the `routing/kinematic/upstream_area`
        attribute of the grid, and then calculates the channel width using the formula:

            W = A / 500

        where W is the channel width, and A is the upstream area of the cell. The resulting channel width is then set as
        the `routing/kinematic/channel_width` attribute of the grid using the `set_grid()` method.

        Additionally, this method sets a minimum channel width by replacing any channel width values that are less than the
        minimum width with the minimum width.
        """
        self.logger.info("Setting up channel width")
        channel_width_data = self.grid["routing/kinematic/upstream_area"] / 500
        channel_width_data = xr.where(
            channel_width_data > minimum_width,
            channel_width_data,
            minimum_width,
            keep_attrs=True,
        )

        channel_width = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="routing/kinematic/channel_width",
            lazy=True,
            crs=self.crs,
        )
        channel_width.data = channel_width_data

        self.set_grid(channel_width, channel_width.name)

    def setup_channel_depth(self) -> None:
        """
        Sets up the channel depth for the model.

        Raises
        ------
        AssertionError
            If the upstream area of any cell in the grid is less than or equal to zero.

        Notes
        -----
        This method sets up the channel depth for the model by calculating the depth of each channel based on the upstream
        area of each cell in the grid. It first retrieves the upstream area of each cell from the `routing/kinematic/upstream_area`
        attribute of the grid, and then calculates the channel depth using the formula:

            D = 0.27 * A ** 0.26

        where D is the channel depth, and A is the upstream area of the cell. The resulting channel depth is then set as
        the `routing/kinematic/channel_depth` attribute of the grid using the `set_grid()` method.

        Additionally, this method raises an `AssertionError` if the upstream area of any cell in the grid is less than or
        equal to zero. This is done to ensure that the upstream area is a positive value, which is required for the channel
        depth calculation to be valid.
        """
        self.logger.info("Setting up channel depth")
        assert (
            (self.grid["routing/kinematic/upstream_area"] > 0)
            | ~self.grid["areamaps/grid_mask"]
        ).all()
        channel_depth_data = 0.27 * self.grid["routing/kinematic/upstream_area"] ** 0.26
        channel_depth = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="routing/kinematic/channel_depth",
            lazy=True,
            crs=self.crs,
        )
        channel_depth.data = channel_depth_data
        self.set_grid(channel_depth, channel_depth.name)

    def setup_channel_ratio(self) -> None:
        """
        Sets up the channel ratio for the model.

        Raises
        ------
        AssertionError
            If the channel length of any cell in the grid is less than or equal to zero, or if the channel ratio of any
            cell in the grid is less than zero.

        Notes
        -----
        This method sets up the channel ratio for the model by calculating the ratio of the channel area to the cell area
        for each cell in the grid. It first retrieves the channel width and length from the `routing/kinematic/channel_width`
        and `routing/kinematic/channel_length` attributes of the grid, and then calculates the channel area using the
        product of the width and length. It then calculates the channel ratio by dividing the channel area by the cell area
        retrieved from the `areamaps/cell_area` attribute of the grid.

        The resulting channel ratio is then set as the `routing/kinematic/channel_ratio` attribute of the grid using the
        `set_grid()` method. Any channel ratio values that are greater than 1 are replaced with 1 (i.e., the whole cell is a channel).

        Additionally, this method raises an `AssertionError` if the channel length of any cell in the grid is less than or
        equal to zero, or if the channel ratio of any cell in the grid is less than zero. These checks are done to ensure
        that the channel length and ratio are positive values, which are required for the channel ratio calculation to be
        valid.
        """
        self.logger.info("Setting up channel ratio")
        assert (
            (self.grid["routing/kinematic/channel_length"] > 0)
            | ~self.grid["areamaps/grid_mask"]
        ).all()
        channel_area = (
            self.grid["routing/kinematic/channel_width"]
            * self.grid["routing/kinematic/channel_length"]
        )
        channel_ratio_data = channel_area / self.grid["areamaps/cell_area"]
        channel_ratio_data = xr.where(
            channel_ratio_data < 1, channel_ratio_data, 1, keep_attrs=True
        )
        assert ((channel_ratio_data >= 0) | ~self.grid["areamaps/grid_mask"]).all()
        channel_ratio = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="routing/kinematic/channel_ratio",
            lazy=True,
            crs=self.crs,
        )
        channel_ratio.data = channel_ratio_data
        self.set_grid(channel_ratio, channel_ratio.name)

    def setup_elevation(self) -> None:
        """
        Sets up the standard deviation of elevation for the model.

        Notes
        -----
        This method sets up the standard deviation of elevation for the model by retrieving high-resolution elevation data
        from the MERIT dataset and calculating the standard deviation of elevation for each cell in the grid.

        MERIT data has a half cell offset. Therefore, this function first corrects for this offset.  It then selects the
        high-resolution elevation data from the MERIT dataset using the grid coordinates of the model, and calculates the
        standard deviation of elevation for each cell in the grid using the `np.std()` function.

        The resulting standard deviation of elevation is then set as the `landsurface/topo/elevation_STD` attribute of
        the grid using the `set_grid()` method.
        """
        self.logger.info("Setting up elevation standard deviation")
        MERIT = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            variables=["elv"],
            provider=self.data_provider,
            bbox=self.grid.raster.bounds,
            buffer=50,
        ).compute()  # Why is compute needed here?
        # In some MERIT datasets, there is a half degree offset in MERIT data. We can detect this by checking the offset relative to the resolution.
        # This offset should be 0.5. If the offset instead is close to 0 or 1, then we need to correct for this offset.
        center_offset = (
            MERIT.coords["x"][0] % MERIT.rio.resolution()[0]
        ) / MERIT.rio.resolution()[0]
        # check whether offset is close to 0.5
        if not np.isclose(center_offset, 0.5, atol=MERIT.rio.resolution()[0] / 100):
            assert np.isclose(
                center_offset, 0, atol=MERIT.rio.resolution()[0] / 100
            ) or np.isclose(
                center_offset, 1, atol=MERIT.rio.resolution()[0] / 100
            ), "Could not detect offset in MERIT data"
            MERIT = MERIT.assign_coords(
                x=MERIT.coords["x"] + MERIT.rio.resolution()[0] / 2,
                y=MERIT.coords["y"] - MERIT.rio.resolution()[1] / 2,
            )
            center_offset = (
                MERIT.coords["x"][0] % MERIT.rio.resolution()[0]
            ) / MERIT.rio.resolution()[0]

        # we are going to match the upper left corners. So create a MERIT grid with the upper left corners as coordinates
        MERIT_ul = MERIT.assign_coords(
            x=MERIT.coords["x"] - MERIT.rio.resolution()[0] / 2,
            y=MERIT.coords["y"] - MERIT.rio.resolution()[1] / 2,
        )

        scaling = 10

        # find the upper left corner of the grid cells in self.grid
        y_step = self.grid.get_index("y")[1] - self.grid.get_index("y")[0]
        x_step = self.grid.get_index("x")[1] - self.grid.get_index("x")[0]
        upper_left_y = self.grid.get_index("y")[0] - y_step / 2
        upper_left_x = self.grid.get_index("x")[0] - x_step / 2

        ymin = np.isclose(
            MERIT_ul.get_index("y"), upper_left_y, atol=MERIT.rio.resolution()[1] / 100
        )
        assert (
            ymin.sum() == 1
        ), "Could not find the upper left corner of the grid cell in MERIT data"
        ymin = ymin.argmax()
        ymax = ymin + self.grid.y.size * scaling
        xmin = np.isclose(
            MERIT_ul.get_index("x"), upper_left_x, atol=MERIT.rio.resolution()[0] / 100
        )
        assert (
            xmin.sum() == 1
        ), "Could not find the upper left corner of the grid cell in MERIT data"
        xmin = xmin.argmax()
        xmax = xmin + self.grid.x.size * scaling

        # select data from MERIT using the grid coordinates
        high_res_elevation_data = MERIT.isel(y=slice(ymin, ymax), x=slice(xmin, xmax))
        self.set_MERIT_grid(
            MERIT.isel(y=slice(ymin - 1, ymax + 1), x=slice(xmin - 1, xmax + 1)),
            name="landsurface/topo/subgrid_elevation",
        )

        elevation_per_cell = high_res_elevation_data.values.reshape(
            high_res_elevation_data.shape[0] // scaling, scaling, -1, scaling
        ).swapaxes(1, 2)

        elevation = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="landsurface/topo/elevation",
            lazy=True,
            crs=self.crs,
        )
        elevation.data = np.mean(elevation_per_cell, axis=(2, 3))
        self.set_grid(elevation, elevation.name)

        standard_deviation = hydromt.raster.full(
            self.grid.raster.coords,
            nodata=np.nan,
            dtype=np.float32,
            name="landsurface/topo/elevation_STD",
            lazy=True,
        )
        standard_deviation.data = np.std(elevation_per_cell, axis=(2, 3))
        self.set_grid(standard_deviation, standard_deviation.name)

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
            hydraulic_conductivity,
            bubbling_pressure_cm,
            lambda_,
            thetas,
            thetar,
            soil_layer_height,
        ) = load_soilgrids(self.data_catalog, self.grid, self.region)

        self.set_grid(hydraulic_conductivity, name="soil/hydraulic_conductivity")
        self.set_grid(bubbling_pressure_cm, name="soil/bubbling_pressure_cm")
        self.set_grid(lambda_, name="soil/lambda")
        self.set_grid(thetas, name="soil/thetas")
        self.set_grid(thetar, name="soil/thetar")
        self.set_grid(soil_layer_height, name="soil/soil_layer_height")

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
        assert (
            "average_discharge" in waterbodies.columns
        ), "average_discharge is required"
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
                buffer=0,
            )
            .rename({"lon": "x", "lat": "y"})
            .compute()
        )
        total_thickness = self.snap_to_grid(total_thickness, self.grid)
        assert total_thickness.shape == self.grid.raster.shape

        total_thickness = np.clip(
            total_thickness,
            minimum_thickness_confined_layer,
            maximum_thickness_confined_layer,
        )

        confining_layer = (
            self.data_catalog.get_rasterdataset(
                "thickness_confining_layer_globgm",
                bbox=self.bounds,
                buffer=0,
            )
            .rename({"lon": "x", "lat": "y"})
            .compute()
        )
        confining_layer = self.snap_to_grid(confining_layer, self.grid)
        assert confining_layer.shape == self.grid.raster.shape

        if not (confining_layer == 0).all():  # two-layer-model
            two_layers = True
        else:
            two_layers = False

        if two_layers:
            # make sure that total thickness is at least 50 m thicker than confining layer
            total_thickness = np.maximum(
                total_thickness, confining_layer + minimum_thickness_confined_layer
            )
            # thickness of layer 2 is based on the predefined confiningLayerThickness
            bottom_top_layer = aquifer_top_elevation - confining_layer
            # make sure that the minimum thickness of layer 2 is at least 0.1 m
            thickness_top_layer = np.maximum(
                0.1, aquifer_top_elevation - bottom_top_layer
            )
            bottom_top_layer = aquifer_top_elevation - thickness_top_layer
            # thickness of layer 1 is at least 5.0 m
            thickness_bottom_layer = np.maximum(
                5.0, total_thickness - thickness_top_layer
            )
            bottom_bottom_layer = bottom_top_layer - thickness_bottom_layer

            layer_boundary_elevation = xr.concat(
                [aquifer_top_elevation, bottom_top_layer, bottom_bottom_layer],
                dim="boundary",
                compat="equals",
            ).compute()
        else:
            layer_boundary_elevation = xr.concat(
                [aquifer_top_elevation, aquifer_top_elevation - total_thickness],
                dim="boundary",
                compat="equals",
            ).compute()

        self.set_grid(
            layer_boundary_elevation, name="groundwater/layer_boundary_elevation"
        )

        # load hydraulic conductivity
        hydraulic_conductivity = self.data_catalog.get_rasterdataset(
            "hydraulic_conductivity_globgm",
            bbox=self.bounds,
            buffer=0,
        ).rename({"lon": "x", "lat": "y"})
        hydraulic_conductivity = self.snap_to_grid(hydraulic_conductivity, self.grid)
        assert hydraulic_conductivity.shape == self.grid.raster.shape

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
            buffer=0,
        ).rename({"lon": "x", "lat": "y"})
        specific_yield = self.snap_to_grid(specific_yield, self.grid)
        assert specific_yield.shape == self.grid.raster.shape

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
        why_interpolated = self.interpolate(why_map, "nearest").compute()

        self.set_grid(why_interpolated, name="groundwater/why_map")

        if intial_heads_source == "GLOBGM":
            # load digital elevation model that was used for globgm
            dem_globgm = (
                self.data_catalog.get_rasterdataset(
                    "dem_globgm",
                    geom=self.region,
                    buffer=0,
                    variables=["dem_average"],
                )
                .rename({"lon": "x", "lat": "y"})
                .compute()
            )
            dem_globgm = self.snap_to_grid(dem_globgm, self.grid)
            assert dem_globgm.shape == self.grid.raster.shape

            dem = self.grid["landsurface/topo/elevation"].raster.mask_nodata()

            # heads
            head_upper_layer = self.data_catalog.get_rasterdataset(
                "head_upper_globgm", bbox=self.bounds, buffer=0
            ).compute()
            head_upper_layer = head_upper_layer.raster.mask_nodata()
            head_upper_layer = self.snap_to_grid(head_upper_layer, self.grid)
            head_upper_layer = head_upper_layer - dem_globgm + dem
            assert head_upper_layer.shape == self.grid.raster.shape

            # assert concistency of datasets. If one layer, this layer should be all nan
            if not two_layers:
                assert np.isnan(head_upper_layer).all()

            head_lower_layer = self.data_catalog.get_rasterdataset(
                "head_lower_globgm", bbox=self.bounds, buffer=0
            ).compute()
            head_lower_layer = head_lower_layer.raster.mask_nodata()
            head_lower_layer = self.snap_to_grid(head_lower_layer, self.grid).compute()
            head_lower_layer = (head_lower_layer - dem_globgm + dem).compute()
            # TODO: Make sure head in lower layer is not lower than topography, but why is this needed?
            head_lower_layer = xr.where(
                head_lower_layer < layer_boundary_elevation[-1],
                layer_boundary_elevation[-1],
                head_lower_layer,
            )
            assert head_lower_layer.shape == self.grid.raster.shape

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
                assert (
                    forcing == "chelsa-w5e5"
                ), "Only chelsa-w5e5 is supported for 30 arcsec resolution"
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

            pr_hourly = self.download_ERA(
                "total_precipitation", starttime, endtime, method="accumulation"
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
            self.set_forcing(pr_hourly, name="climate/pr_hourly")
            pr = pr_hourly.resample(time="D").mean()  # get daily mean
            pr = pr.raster.reproject_like(mask, method="average")
            pr.name = "pr"
            self.set_forcing(pr, name="climate/pr")

            hourly_rsds = self.download_ERA(
                "surface_solar_radiation_downwards",
                starttime,
                endtime,
                method="accumulation",
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

            hourly_rlds = self.download_ERA(
                "surface_thermal_radiation_downwards",
                starttime,
                endtime,
                method="accumulation",
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

            hourly_tas = self.download_ERA(
                "2m_temperature", starttime, endtime, method="raw"
            )

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
            self.set_forcing(tas_reprojected, name="climate/tas")

            tasmax = hourly_tas_reprojected.resample(time="D").max()
            tasmax.attrs = {
                "standard_name": "air_temperature",
                "long_name": "Daily Maximum Near-Surface Air Temperature",
                "units": "K",
            }
            tasmax.name = "tasmax"
            self.set_forcing(tasmax, name="climate/tasmax")

            tasmin = hourly_tas_reprojected.resample(time="D").min()
            tasmin.attrs = {
                "standard_name": "air_temperature",
                "long_name": "Daily Minimum Near-Surface Air Temperature",
                "units": "K",
            }
            tasmin.name = "tasmin"
            self.set_forcing(tasmin, name="climate/tasmin")

            dew_point_tas = self.download_ERA(
                "2m_dewpoint_temperature", starttime, endtime, method="raw"
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
            self.set_forcing(relative_humidity, name="climate/hurs")

            pressure = self.download_ERA(
                "surface_pressure", starttime, endtime, method="raw"
            )
            pressure = reproject_and_apply_lapse_rate_pressure(pressure, DEM, mask)
            pressure.attrs = {
                "standard_name": "surface_air_pressure",
                "long_name": "Surface Air Pressure",
                "units": "Pa",
            }
            pressure = pressure.resample(time="D").mean()
            pressure.name = "ps"
            self.set_forcing(pressure, name="climate/ps")

            u_wind = self.download_ERA(
                "10m_u_component_of_wind", starttime, endtime, method="raw"
            )
            u_wind = u_wind.resample(time="D").mean()

            v_wind = self.download_ERA(
                "10m_v_component_of_wind", starttime, endtime, method="raw"
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
            self.set_forcing(wind_speed, name="climate/sfcwind")

        elif data_source == "cmip":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(f"Unknown data source: {data_source}")

    def download_ERA(
        self, variable, starttime: date, endtime: date, method: str, download_only=False
    ):
        # https://cds.climate.copernicus.eu/cdsapp#!/software/app-c3s-daily-era5-statistics?tab=appcode
        # https://earthscience.stackexchange.com/questions/24156/era5-single-level-calculate-relative-humidity
        import cdsapi

        """
        Download hourly ERA5 data for a specified time frame and bounding box.

        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

        """

        download_path = Path(self.root).parent / "preprocessing" / "climate" / "ERA5"
        download_path.mkdir(parents=True, exist_ok=True)

        def download(start_and_end_year):
            start_year, end_year = start_and_end_year
            output_fn = download_path / f"{variable}_{start_year}_{end_year}.nc"
            if output_fn.exists():
                self.logger.info(f"ERA5 data already downloaded to {output_fn}")
            else:
                (xmin, ymin, xmax, ymax) = self.bounds

                # add buffer to bounding box. Resolution is 0.1 degrees, so add 0.1 degrees to each side
                xmin -= 0.1
                ymin -= 0.1
                xmax += 0.1
                ymax += 0.1

                max_retries = 10
                retries = 0
                while retries < max_retries:
                    try:
                        request = {
                            "product_type": "reanalysis",
                            "format": "netcdf",
                            "variable": [
                                variable,
                            ],
                            "date": f"{start_year}-01-01/{end_year}-12-31",
                            "time": [
                                "00:00",
                                "01:00",
                                "02:00",
                                "03:00",
                                "04:00",
                                "05:00",
                                "06:00",
                                "07:00",
                                "08:00",
                                "09:00",
                                "10:00",
                                "11:00",
                                "12:00",
                                "13:00",
                                "14:00",
                                "15:00",
                                "16:00",
                                "17:00",
                                "18:00",
                                "19:00",
                                "20:00",
                                "21:00",
                                "22:00",
                                "23:00",
                            ],
                            "area": (
                                float(ymax),
                                float(xmin),
                                float(ymin),
                                float(xmax),
                            ),  # North, West, South, East
                        }
                        cdsapi.Client().retrieve(
                            "reanalysis-era5-land",
                            request,
                            output_fn,
                        )
                        break
                    except Exception as e:
                        print(
                            f"Download failed. Retrying... ({retries+1}/{max_retries})"
                        )
                        print(e)
                        print(request)
                        retries += 1
                if retries == max_retries:
                    raise Exception("Download failed after maximum retries.")
            return output_fn

        with concurrent.futures.ThreadPoolExecutor() as executor:
            multiple_years = 5

            range_start = starttime.year - starttime.year % 5
            range_end = endtime.year - endtime.year % 5 + 5
            years = []
            for year in range(range_start, range_end, multiple_years):
                years.append(
                    (
                        max(year, starttime.year),
                        min(year + multiple_years - 1, endtime.year),
                    )
                )
            files = list(executor.map(download, years))

        if download_only:
            return

        ds = xr.open_mfdataset(
            files,
            # chunks={
            #     "valid_time": 1,
            #     "latitude": XY_CHUNKSIZE,
            #     "longitude": XY_CHUNKSIZE,
            # },
            compat="equals",  # all values and dimensions must be the same,
            combine_attrs="drop_conflicts",  # drop conflicting attributes
        ).rio.set_crs(4326)

        assert "valid_time" in ds.dims
        assert "latitude" in ds.dims
        assert "longitude" in ds.dims

        # rename valid_time to time
        ds = ds.rename({"valid_time": "time"})

        # remove first time step.
        # This is an accumulation from the previous day and thus cannot be calculated
        ds = ds.isel(time=slice(1, None))
        ds = ds.chunk({"time": 24, "latitude": XY_CHUNKSIZE, "longitude": XY_CHUNKSIZE})
        # the ERA5 grid is sometimes not exactly regular. The offset is very minor
        # therefore we snap the grid to a regular grid, to save huge computational time
        # for a infenitesimal loss in accuracy
        ds = ds.assign_coords(
            latitude=np.linspace(
                ds["latitude"][0].item(),
                ds["latitude"][-1].item(),
                ds["latitude"].size,
                endpoint=True,
            ),
            longitude=np.linspace(
                ds["longitude"][0].item(),
                ds["longitude"][-1].item(),
                ds["longitude"].size,
                endpoint=True,
            ),
        )
        # assert that time is monotonically increasing with a constant step size
        assert (
            ds.time.diff("time").astype(np.int64)
            == (ds.time[1] - ds.time[0]).astype(np.int64)
        ).all()
        ds.raster.set_crs(4326)
        # the last few months of data may come from ERA5T (expver 5) instead of ERA5 (expver 1)
        # if so, combine that dimension
        if "expver" in ds.dims:
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

        # assert there is only one data variable
        assert len(ds.data_vars) == 1

        # select the variable and rename longitude and latitude variable
        ds = ds[list(ds.data_vars)[0]].rename({"longitude": "x", "latitude": "y"})

        if method == "accumulation":

            def xr_ERA5_accumulation_to_hourly(ds, dim):
                # Identify the axis number for the given dimension
                assert ds.time.dt.hour[0] == 1, "First time step must be at 1 UTC"
                # All chunksizes must be divisible by 24, except the last one
                assert all(
                    chunksize % 24 == 0 for chunksize in ds.chunksizes["time"][:-1]
                )

                def diff_with_prepend(data, dim):
                    # Assert dimension is a multiple of 24
                    # As the first hour is an accumulation from the first hour of the day, prepend a 0
                    # to the data array before taking the diff. In this way, the output is also 24 hours
                    return np.diff(data, prepend=0, axis=dim)

                # Apply the custom diff function using apply_ufunc
                return xr.apply_ufunc(
                    diff_with_prepend,  # The function to apply
                    ds,  # The DataArray or Dataset to which the function will be applied
                    kwargs={
                        "dim": ds.get_axis_num(dim)
                    },  # Additional arguments for the function
                    dask="parallelized",  # Enable parallelized computation
                    output_dtypes=[ds.dtype],  # Specify the output data type
                )

            # The accumulations in the short forecasts of ERA5-Land (with hourly steps from 01 to 24) are treated
            # the same as those in ERA-Interim or ERA-Interim/Land, i.e., they are accumulated from the beginning
            # of the forecast to the end of the forecast step. For example, runoff at day=D, step=12 will provide
            # runoff accumulated from day=D, time=0 to day=D, time=12. The maximum accumulation is over 24 hours,
            # i.e., from day=D, time=0 to day=D+1,time=0 (step=24).
            # forecasts are the difference between the current and previous time step
            hourly = xr_ERA5_accumulation_to_hourly(ds, "time")
        elif method == "raw":
            hourly = ds
        else:
            raise NotImplementedError

        return hourly

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
                var = self.interpolate(var, "linear")
            self.logger.info(f"Completed {variable_name}")
            self.set_forcing(var, name=f"climate/{variable_name}")

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

        chelsa_folder = (
            Path(self.root).parent
            / "preprocessing"
            / "climate"
            / "chelsa-bioclim+"
            / "hurs"
        )
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
                assert (
                    w5e5_regridded >= 0.1
                ).all(), "too low values in relative humidity"
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
        self.set_forcing(hurs_output, "climate/hurs")

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
        self.set_forcing(lw_fine, name="climate/rlds")

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
        self.set_forcing(pressure, name="climate/ps")

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
        self.set_forcing(wind_output_clipped, "climate/sfcwind")

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

                # negative_SPEI = SPEI.where(SPEI < 0)

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
        region_database="gadm_level1",
        unique_region_id="UID",
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
            The name of the region database to use. Default is 'gadm_level1'.
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

        assert bounds_are_within(
            self.region.total_bounds,
            regions.to_crs(self.region.crs).total_bounds,
        )
        assert np.issubdtype(
            regions["region_id"].dtype, np.integer
        ), "Region ID must be integer"

        region_id_mapping = {
            i: region_id for region_id, i in enumerate(regions["region_id"])
        }
        regions["region_id"] = regions["region_id"].map(region_id_mapping)
        self.set_dict(region_id_mapping, name="areamaps/region_id_mapping")

        assert (
            "ISO3" in regions.columns
        ), f"Region database must contain ISO3 column ({self.data_catalog[region_database].path})"

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
        padded_subgrid, region_subgrid_slice = pad_xy(
            self.subgrid["areamaps/sub_grid_mask"].rio,
            pad_minx,
            pad_miny,
            pad_maxx,
            pad_maxy,
            return_slice=True,
            constant_values=1,
        )
        padded_subgrid.raster.set_crs(self.subgrid.raster.crs)
        padded_subgrid.raster.set_nodata(-1)
        self.set_region_subgrid(padded_subgrid, name="areamaps/region_mask")

        land_use = self.data_catalog.get_rasterdataset(
            land_cover,
            geom=self.geoms["areamaps/regions"],
            buffer=200,  # 2 km buffer
        )
        reprojected_land_use = land_use.raster.reproject_like(
            padded_subgrid, method="nearest"
        )

        region_raster = reprojected_land_use.raster.rasterize(
            self.geoms["areamaps/regions"],
            col_name="region_id",
            all_touched=True,
        )
        self.set_region_subgrid(region_raster, name="areamaps/region_subgrid")

        padded_cell_area = self.grid["areamaps/cell_area"].rio.pad_box(*regions_bounds)
        # calculate the cell area for the grid for the entire region
        region_cell_area = calculate_cell_area(
            padded_cell_area.raster.transform, padded_cell_area.shape
        )

        # create subgrid for entire region
        region_cell_area_subgrid = hydromt.raster.full_from_transform(
            padded_cell_area.raster.transform * Affine.scale(1 / self.subgrid_factor),
            (
                padded_cell_area.raster.shape[0] * self.subgrid_factor,
                padded_cell_area.raster.shape[1] * self.subgrid_factor,
            ),
            nodata=np.nan,
            dtype=padded_cell_area.dtype,
            crs=padded_cell_area.raster.crs,
            name="areamaps/sub_grid_mask",
            lazy=False,
        )

        # calculate the cell area for the subgrid for the entire region
        region_cell_area_subgrid.data = (
            repeat_grid(region_cell_area, self.subgrid_factor) / self.subgrid_factor**2
        )

        # create new subgrid for the region without padding
        region_cell_area_subgrid_clipped_to_region = hydromt.raster.full(
            region_raster.raster.coords,
            nodata=np.nan,
            dtype=padded_cell_area.dtype,
            name="areamaps/sub_grid_mask_region",
            crs=region_raster.raster.crs,
            lazy=False,
        )

        # remove padding from region subgrid
        region_cell_area_subgrid_clipped_to_region.data = (
            region_cell_area_subgrid.raster.clip_bbox(
                (pad_minx, pad_miny, pad_maxx, pad_maxy)
            )
        )

        # set the cell area for the region subgrid
        self.set_region_subgrid(
            region_cell_area_subgrid_clipped_to_region,
            name="areamaps/region_cell_area_subgrid",
        )

        MERIT = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            variables=["upg"],
            bbox=padded_subgrid.rio.bounds(),
            buffer=300,  # 3 km buffer
            provider=self.data_provider,
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
        rivers = rivers.raster.reproject_like(reprojected_land_use, method="nearest")
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

        self.set_region_subgrid(
            hydro_land_use, name="landsurface/full_region_land_use_classes"
        )

        cultivated_land = xr.where(
            (hydro_land_use == 1) & (reprojected_land_use == 40), 1, 0, keep_attrs=True
        )
        cultivated_land.raster.set_crs(self.subgrid.raster.crs)
        cultivated_land.raster.set_nodata(-1)

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
        ), f"project_future_until_year ({project_future_until_year}) must be larger than reference_start_year ({reference_start_year})"

        lending_rates = self.data_catalog.get_dataframe("wb_lending_rate")
        inflation_rates = self.data_catalog.get_dataframe("wb_inflation_rate")

        ppp_conversion_rates = self.data_catalog.get_dataframe("wb_ppp_conversion_rate")
        lcu_per_usd_conversion_rates = self.data_catalog.get_dataframe(
            "lcu_per_usd_conversion_rate"
        )

        def filter_and_rename(df, additional_cols):
            # Select columns: 'Country Name', 'Country Code', and columns containing "YR"
            columns_to_keep = additional_cols + [
                col for col in df.columns if "YR" in col
            ]
            filtered_df = df[columns_to_keep]

            # Rename columns to just the year or keep the original name for specified columns
            filtered_df.columns = additional_cols + [
                col.split(" ")[0]
                for col in filtered_df.columns
                if col not in additional_cols
            ]
            return filtered_df

        def extract_years(df):
            # Extract years that are numerically valid between 1900 and 3000
            return [
                col
                for col in df.columns
                if col.isnumeric() and 1900 <= int(col) <= 3000
            ]

        # Assuming dataframes for PPP and LCU per USD have been initialized
        ppp_filtered = filter_and_rename(
            ppp_conversion_rates, ["Country Name", "Country Code"]
        )
        lcu_per_usd_filtered = filter_and_rename(
            lcu_per_usd_conversion_rates, ["Country Name", "Country Code"]
        )
        years_ppp_conversion_rates = extract_years(ppp_filtered)
        years_lcu_per_usd_conversion_rates = extract_years(lcu_per_usd_filtered)

        ppp_conversion_rates_dict = {"time": years_ppp_conversion_rates, "data": {}}
        lcu_per_usd_conversion_rates_dict = {
            "time": years_lcu_per_usd_conversion_rates,
            "data": {},
        }

        # Assume lending_rates and inflation_rates are available
        years_lending_rates = extract_years(lending_rates)
        years_inflation_rates = extract_years(inflation_rates)

        lending_rates_dict = {"time": years_lending_rates, "data": {}}
        inflation_rates_dict = {"time": years_inflation_rates, "data": {}}

        for _, region in self.geoms["areamaps/regions"].iterrows():
            region_id = str(region["region_id"])
            ISO3 = region["ISO3"]

            # Create a helper to process rates and assert single row data
            def process_rates(df, rate_cols, convert_percent=False):
                filtered_data = df.loc[df["Country Code"] == ISO3, rate_cols]
                assert (
                    len(filtered_data) == 1
                ), f"Expected one row for {ISO3}, got {len(filtered_data)}"
                if convert_percent:
                    return (filtered_data.iloc[0] / 100 + 1).tolist()
                return filtered_data.iloc[0].tolist()

            # Store data in dictionaries
            ppp_conversion_rates_dict["data"][region_id] = process_rates(
                ppp_filtered, years_ppp_conversion_rates
            )
            lcu_per_usd_conversion_rates_dict["data"][region_id] = process_rates(
                lcu_per_usd_filtered, years_lcu_per_usd_conversion_rates
            )
            lending_rates_dict["data"][region_id] = process_rates(
                lending_rates, years_lending_rates, True
            )
            inflation_rates_dict["data"][region_id] = process_rates(
                inflation_rates, years_inflation_rates, True
            )

        if project_future_until_year:
            # convert to pandas dataframe
            inflation_rates = pd.DataFrame(
                inflation_rates_dict["data"], index=inflation_rates_dict["time"]
            ).dropna()
            lending_rates = pd.DataFrame(
                lending_rates_dict["data"], index=lending_rates_dict["time"]
            ).dropna()

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

            lending_rates.index = lending_rates.index.astype(int)
            # extend lending rates to future
            mean_lending_rate_since_reference_year = lending_rates.loc[
                reference_start_year:
            ].mean(axis=0)
            lending_rates = lending_rates.reindex(
                range(lending_rates.index.min(), project_future_until_year + 1)
            ).fillna(mean_lending_rate_since_reference_year)

            # convert back to dictionary
            lending_rates_dict["time"] = lending_rates.index.astype(str).tolist()
            lending_rates_dict["data"] = lending_rates.to_dict(orient="list")

        self.set_dict(inflation_rates_dict, name="economics/inflation_rates")
        self.set_dict(lending_rates_dict, name="economics/lending_rates")
        self.set_dict(ppp_conversion_rates_dict, name="economics/ppp_conversion_rates")
        self.set_dict(
            lcu_per_usd_conversion_rates_dict,
            name="economics/lcu_per_usd_conversion_rates",
        )

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
        ppp_conversion_rates = self.dict["economics/ppp_conversion_rates"]

        full_years_array_ppp = np.array(ppp_conversion_rates["time"], dtype=str)
        years_index_ppp = np.isin(full_years_array_ppp, str(reference_year))
        source_conversion_rates = 1  # US ppp is 1

        electricity_rates = self.data_catalog.get_dataframe("gcam_electricity_rates")
        electricity_rates["ISO3"] = electricity_rates["Country"].map(
            SUPERWELL_NAME_TO_ISO3
        )
        # Create a dictionary to store the various types of prices with their initial reference year values
        price_types = {
            "why_10": WHY_10,
            "why_20": WHY_20,
            "why_30": WHY_30,
            "electricity_cost": electricity_rates,
        }

        # Iterate over each price type and calculate the prices across years for each region
        for price_type, initial_price in price_types.items():
            prices_dict = {"time": list(range(start_year, end_year + 1)), "data": {}}

            for _, region in self.geoms["areamaps/regions"].iterrows():
                region_id = str(region["region_id"])

                prices = pd.Series(index=range(start_year, end_year + 1))

                target_conversion_rates = np.array(
                    ppp_conversion_rates["data"][region_id], dtype=float
                )[years_index_ppp]

                prices.loc[reference_year] = self.convert_price_using_ppp(
                    initial_price,
                    source_conversion_rates,
                    target_conversion_rates,
                )

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
            path = (
                Path(self.root).parent
                / "preprocessing"
                / "agents"
                / "farmers"
                / "farmers.csv"
            )
        farmers = pd.read_csv(path, index_col=0)
        self.setup_farmers(farmers)

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
            assert (
                farm_size_donor_countries is None
            ), "farm_size_donor_countries is only used for lowder data"

        cultivated_land = (
            self.region_subgrid["landsurface/full_region_cultivated_land"]
            .astype(bool)
            .compute()
        )
        regions_grid = self.region_subgrid["areamaps/region_subgrid"].compute()
        cell_area = self.region_subgrid["areamaps/region_cell_area_subgrid"].compute()

        regions_shapes = self.geoms["areamaps/regions"]
        if data_source == "lowder":
            assert (
                country_iso3_column in regions_shapes.columns
            ), f"Region database must contain {country_iso3_column} column ({self.data_catalog['gadm_level1'].path})"

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

            # convert country names to ISO3 codes
            iso3_codes = {
                "Albania": "ALB",
                "Algeria": "DZA",
                "American Samoa": "ASM",
                "Argentina": "ARG",
                "Austria": "AUT",
                "Bahamas": "BHS",
                "Barbados": "BRB",
                "Belgium": "BEL",
                "Brazil": "BRA",
                "Bulgaria": "BGR",
                "Burkina Faso": "BFA",
                "Chile": "CHL",
                "Colombia": "COL",
                "Côte d'Ivoire": "CIV",
                "Croatia": "HRV",
                "Cyprus": "CYP",
                "Czech Republic": "CZE",
                "Democratic Republic of the Congo": "COD",
                "Denmark": "DNK",
                "Dominica": "DMA",
                "Ecuador": "ECU",
                "Egypt": "EGY",
                "Estonia": "EST",
                "Ethiopia": "ETH",
                "Fiji": "FJI",
                "Finland": "FIN",
                "France": "FRA",
                "French Polynesia": "PYF",
                "Georgia": "GEO",
                "Germany": "DEU",
                "Greece": "GRC",
                "Grenada": "GRD",
                "Guam": "GUM",
                "Guatemala": "GTM",
                "Guinea": "GIN",
                "Honduras": "HND",
                "India": "IND",
                "Indonesia": "IDN",
                "Iran (Islamic Republic of)": "IRN",
                "Ireland": "IRL",
                "Italy": "ITA",
                "Japan": "JPN",
                "Jamaica": "JAM",
                "Jordan": "JOR",
                "Korea, Rep. of": "KOR",
                "Kyrgyzstan": "KGZ",
                "Lao People's Democratic Republic": "LAO",
                "Latvia": "LVA",
                "Lebanon": "LBN",
                "Lithuania": "LTU",
                "Luxembourg": "LUX",
                "Malta": "MLT",
                "Morocco": "MAR",
                "Myanmar": "MMR",
                "Namibia": "NAM",
                "Nepal": "NPL",
                "Netherlands": "NLD",
                "Nicaragua": "NIC",
                "Northern Mariana Islands": "MNP",
                "Norway": "NOR",
                "Pakistan": "PAK",
                "Panama": "PAN",
                "Paraguay": "PRY",
                "Peru": "PER",
                "Philippines": "PHL",
                "Poland": "POL",
                "Portugal": "PRT",
                "Puerto Rico": "PRI",
                "Qatar": "QAT",
                "Romania": "ROU",
                "Saint Lucia": "LCA",
                "Saint Vincent and the Grenadines": "VCT",
                "Samoa": "WSM",
                "Senegal": "SEN",
                "Serbia": "SRB",
                "Sweden": "SWE",
                "Switzerland": "CHE",
                "Thailand": "THA",
                "Trinidad and Tobago": "TTO",
                "Turkey": "TUR",
                "Uganda": "UGA",
                "United Kingdom": "GBR",
                "United States of America": "USA",
                "Uruguay": "URY",
                "Venezuela (Bolivarian Republic of)": "VEN",
                "Virgin Islands, United States": "VIR",
                "Yemen": "YEM",
                "Cook Islands": "COK",
                "French Guiana": "GUF",
                "Guadeloupe": "GLP",
                "Martinique": "MTQ",
                "Réunion": "REU",
                "Canada": "CAN",
                "China": "CHN",
                "Guinea Bissau": "GNB",
                "Hungary": "HUN",
                "Lesotho": "LSO",
                "Libya": "LBY",
                "Malawi": "MWI",
                "Mozambique": "MOZ",
                "New Zealand": "NZL",
                "Slovakia": "SVK",
                "Slovenia": "SVN",
                "Spain": "ESP",
                "St. Kitts & Nevis": "KNA",
                "Viet Nam": "VNM",
                "Australia": "AUS",
                "Djibouti": "DJI",
                "Mali": "MLI",
                "Togo": "TGO",
                "Zambia": "ZMB",
            }
            farm_sizes_per_region["ISO3"] = farm_sizes_per_region["Country"].map(
                iso3_codes
            )
            assert (
                not farm_sizes_per_region["ISO3"].isna().any()
            ), f"Found {farm_sizes_per_region['ISO3'].isna().sum()} countries without ISO3 code"
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
                assert (
                    len(region_farm_sizes) == 2
                ), f"Found {len(region_farm_sizes) / 2} region_farm_sizes for {country_ISO3}"

                region_n_holdings = (
                    region_farm_sizes.loc[
                        region_farm_sizes["Holdings/ agricultural area"] == "Holdings"
                    ]
                    .iloc[0]
                    .drop(["Holdings/ agricultural area", "ISO3"])
                    .replace("..", "0")
                    .astype(np.int64)
                )
                agricultural_area_db_ha = (
                    region_farm_sizes.loc[
                        region_farm_sizes["Holdings/ agricultural area"]
                        == "Agricultural area (Ha) "
                    ]
                    .iloc[0]
                    .drop(["Holdings/ agricultural area", "ISO3"])
                    .replace("..", "0")
                    .astype(np.int64)
                )
                agricultural_area_db = agricultural_area_db_ha * 10000
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
                    region_n_holdings[size_class].compute().item()
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

    def setup_farmer_characteristics_simple(
        self,
        irrigation_choice={
            "no": 1.0,
        },
        risk_aversion_mean=0,
        risk_aversion_standard_deviation=0.387,
        interest_rate=0.05,
        discount_rate=0.1,
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

        farmer_mirca_units = sample_from_map(
            MIRCA_unit_grid.values,
            get_farm_locations(self.subgrid["agents/farmers/farms"], method="centroid"),
            MIRCA_unit_grid.raster.transform.to_gdal(),
        )

        # initialize the is_irrigated as -1 for all farmers
        is_irrigated = np.full(n_farmers, -1, dtype=np.int32)

        crop_calendar_per_farmer = np.zeros((n_farmers, 3, 4), dtype=np.int32)
        for mirca_unit in np.unique(farmer_mirca_units):
            n_farmers_mirca_unit = (farmer_mirca_units == mirca_unit).sum()

            area_per_crop_rotation = []
            cropping_calenders_crop_rotation = []
            for crop_rotation in crop_calendar[mirca_unit]:
                area_per_crop_rotation.append(crop_rotation[0])
                crop_rotation_matrix = crop_rotation[1]
                starting_days = crop_rotation_matrix[:, 2]
                starting_days = starting_days[starting_days != -1]
                assert (
                    np.unique(starting_days).size == starting_days.size
                ), "ensure all starting days are unique"
                # TODO: Add check to ensure crop calendars are not overlapping.
                cropping_calenders_crop_rotation.append(crop_rotation_matrix)
            area_per_crop_rotation = np.array(area_per_crop_rotation)
            cropping_calenders_crop_rotation = np.stack(
                cropping_calenders_crop_rotation
            )

            # select n crop rotations weighted by the area for each crop rotation
            farmer_crop_rotations_idx = np.random.choice(
                np.arange(len(area_per_crop_rotation)),
                size=n_farmers_mirca_unit,
                replace=True,
                p=area_per_crop_rotation / area_per_crop_rotation.sum(),
            )
            crop_calendar_per_farmer_mirca_unit = cropping_calenders_crop_rotation[
                farmer_crop_rotations_idx
            ]
            is_irrigated[farmer_mirca_units == mirca_unit] = (
                crop_calendar_per_farmer_mirca_unit[:, :, 1] == 1
            ).any(axis=1)

            crop_calendar_per_farmer[farmer_mirca_units == mirca_unit] = (
                crop_calendar_per_farmer_mirca_unit[:, :, [0, 2, 3, 4]]
            )

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
            CITRUS = 17
            DATE_PALM = 18
            GRAPES = 19
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
            def replace_crop(
                crop_calendar_per_farmer, crop_values, replaced_crop_values
            ):
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
                unique_rows, counts = np.unique(
                    new_crop_types, axis=0, return_counts=True
                )
                max_index = np.argmax(counts)
                crop_replacement = unique_rows[max_index]

                for replaced_crop in replaced_crop_values:
                    # Check where to be replaced crop is
                    crop_mask = (
                        crop_calendar_per_farmer[:, :, 0] == replaced_crop
                    ).any(axis=1)
                    # Replace the crop
                    crop_calendar_per_farmer[crop_mask] = crop_replacement

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
                crop_calendar_per_farmer[
                    indices[n_same + n_first_resistant :], 0, 0
                ] = resistant_crops[1]

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
                most_common_check = [GROUNDNUTS, CITRUS, COCOA, COFFEE, OTHERS_ANNUAL]
                replaced_value = [GROUNDNUTS, CITRUS, COCOA, COFFEE, OTHERS_ANNUAL]
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

                # Change perennial to one
                most_common_check = [OIL_PALM, GRAPES, DATE_PALM, OTHERS_PERENNIAL]
                replaced_value = [OIL_PALM, GRAPES, DATE_PALM, OTHERS_PERENNIAL]
                crop_calendar_per_farmer = replace_crop(
                    crop_calendar_per_farmer, most_common_check, replaced_value
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

        self.set_binary(crop_calendar_per_farmer, name="agents/farmers/crop_calendar")
        assert crop_calendar_per_farmer[:, :, 3].max() == 0
        self.set_binary(
            np.full_like(is_irrigated, 1, dtype=np.int32),
            name="agents/farmers/crop_calendar_rotation_years",
        )

        if irrigation_choice == "random":
            # randomly sample from irrigation sources
            irrigation_source = np.random.choice(
                list(self.dict["agents/farmers/irrigation_sources"].values()),
                size=n_farmers,
            )
        else:
            assert isinstance(irrigation_choice, dict)
            # convert irrigation sources to integers based on irrigation sources dictionary
            # which was set previously
            irrigation_choice_int = {
                self.dict["agents/farmers/irrigation_sources"][i]: k
                for i, k in irrigation_choice.items()
            }
            # pick irrigation source based on the probabilities
            irrigation_source = np.random.choice(
                list(irrigation_choice_int.keys()),
                size=n_farmers,
                p=np.array(list(irrigation_choice_int.values()))
                / sum(irrigation_choice_int.values()),
            )
        self.set_binary(irrigation_source, name="agents/farmers/irrigation_source")

        household_size = random.choices([1, 2, 3, 4, 5, 6, 7], k=n_farmers)
        self.set_binary(household_size, name="agents/farmers/household_size")

        daily_non_farm_income_family = random.choices([50, 100, 200, 500], k=n_farmers)
        self.set_binary(
            daily_non_farm_income_family,
            name="agents/farmers/daily_non_farm_income_family",
        )

        daily_consumption_per_capita = random.choices([50, 100, 200, 500], k=n_farmers)
        self.set_binary(
            daily_consumption_per_capita,
            name="agents/farmers/daily_consumption_per_capita",
        )

        risk_aversion = np.random.normal(
            loc=risk_aversion_mean,
            scale=risk_aversion_standard_deviation,
            size=n_farmers,
        )
        self.set_binary(risk_aversion, name="agents/farmers/risk_aversion")

        interest_rate = np.full(n_farmers, interest_rate, dtype=np.float32)
        self.set_binary(interest_rate, name="agents/farmers/interest_rate")

        discount_rate = np.full(n_farmers, discount_rate, dtype=np.float32)
        self.set_binary(discount_rate, name="agents/farmers/discount_rate")

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

        OSM_data_dir = Path(self.root).parent / "preprocessing" / "osm"
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
        download_path = (
            Path(self.root).parent / "preprocessing" / "climate" / forcing / variable
        )
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
                    ), f"File path too long: {download_path / zip_ref.getinfo(file_name).filename}"
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
        DEM=["fabdem", "gebco"],
    ):
        if isinstance(DEM, str):
            DEM = [DEM]

        hydrodynamics_data_catalog = DataCatalog()

        # hydrobasins
        hydrobasins = self.data_catalog.get_geodataframe(
            "hydrobasins_8",
            geom=self.region,
            predicate="intersects",
        )
        self.set_geoms(hydrobasins, name="hydrodynamics/hydrobasins")

        hydrodynamics_data_catalog.add_source(
            "hydrobasins_level_8",
            GeoDataFrameAdapter(
                path=Path(self.root) / "hydrodynamics" / "hydrobasins.gpkg",
                meta=self.data_catalog.get_source("hydrobasins_8").meta,
            ),  # hydromt likes absolute paths
        )

        bounds = tuple(hydrobasins.total_bounds)

        gcn250 = self.data_catalog.get_rasterdataset(
            "gcn250", bbox=bounds, buffer=100, variables=["cn_avg"]
        )
        gcn250.name = "gcn250"
        self.set_forcing(gcn250, name="hydrodynamics/gcn250")

        hydrodynamics_data_catalog.add_source(
            "gcn250",
            RasterDatasetAdapter(
                path=Path(self.root) / "hydrodynamics" / "gcn250.zarr.zip",
                meta=self.data_catalog.get_source("gcn250").meta,
                driver="zarr",
            ),  # hydromt likes absolute paths
        )

        for DEM_name in DEM:
            DEM_raster = self.data_catalog.get_rasterdataset(
                DEM_name,
                bbox=bounds,
                buffer=100,
                variables=["elevation"],
                single_var_as_array=False,
            ).compute()
            DEM_raster = DEM_raster.rename({"elevation": "elevtn"})

            # hydromt-sfincs requires the data to be a Dataset. This code here makes
            # data with only one variable a Dataarray, which is not supported in hydromt-sfincs
            # therefore we add a dummy variable to the data thus forcing the data to
            # be considered a Dataset
            DEM_raster["_dummy"] = 0
            self.set_forcing(
                DEM_raster, name=f"hydrodynamics/DEM/{DEM_name}", split_dataset=False
            )

            hydrodynamics_data_catalog.add_source(
                DEM_name,
                RasterDatasetAdapter(
                    path=Path(self.root)
                    / "hydrodynamics"
                    / "DEM"
                    / f"{DEM_name}.zarr.zip",
                    meta=self.data_catalog.get_source(DEM_name).meta,
                    driver="zarr",
                ),  # hydromt likes absolute paths
            )

        # merit hydro
        merit_hydro = self.data_catalog.get_rasterdataset(
            "merit_hydro",
            bbox=bounds,
            buffer=100,
            variables=["uparea", "flwdir", "elevtn"],
            provider=self.data_provider,
        )
        del merit_hydro["flwdir"].attrs["_FillValue"]
        self.set_forcing(
            merit_hydro, name="hydrodynamics/merit_hydro", split_dataset=False
        )

        hydrodynamics_data_catalog.add_source(
            "merit_hydro",
            RasterDatasetAdapter(
                path=Path(self.root) / "hydrodynamics" / "merit_hydro.zarr.zip",
                meta=self.data_catalog.get_source("merit_hydro").meta,
                driver="zarr",
            ),  # hydromt likes absolute paths
        )

        # glofas discharge
        glofas_discharge = self.data_catalog.get_rasterdataset(
            "glofas_4_0_discharge_yearly",
            bbox=bounds,
            buffer=1,
            variables=["discharge"],
        )
        glofas_discharge = glofas_discharge.rename({"latitude": "y", "longitude": "x"})
        glofas_discharge.name = "discharge_yearly"
        self.set_forcing(glofas_discharge, name="hydrodynamics/discharge_yearly")

        hydrodynamics_data_catalog.add_source(
            "glofas_discharge_Yearly_Resampled_Global",
            RasterDatasetAdapter(
                path=Path(self.root) / "hydrodynamics" / "discharge_yearly.zarr.zip",
                meta=self.data_catalog.get_source("glofas_4_0_discharge_yearly").meta,
                driver="zarr",
            ),  # hydromt likes absolute paths
        )

        glofas_uparea = self.data_catalog.get_rasterdataset(
            "glofas_uparea",
            bbox=bounds,
            buffer=1,
            variables=["uparea"],
        )
        glofas_uparea = glofas_uparea.rename({"latitude": "y", "longitude": "x"})
        glofas_uparea.name = "uparea"
        self.set_forcing(glofas_uparea, name="hydrodynamics/uparea")

        hydrodynamics_data_catalog.add_source(
            "glofas_uparea",
            RasterDatasetAdapter(
                path=Path(self.root) / "hydrodynamics" / "uparea.zarr.zip",
                meta=self.data_catalog.get_source("glofas_uparea").meta,
                driver="zarr",
            ),  # hydromt likes absolute paths
        )

        # river centerlines
        river_centerlines = self.data_catalog.get_geodataframe(
            "river_centerlines_MERIT_Basins",
            bbox=bounds,
            predicate="intersects",
        )
        self.set_geoms(river_centerlines, name="hydrodynamics/river_centerlines")

        hydrodynamics_data_catalog.add_source(
            "river_centerlines_MERIT_Basins",
            GeoDataFrameAdapter(
                path=Path(self.root)
                / "hydrodynamics"
                / "river_centerlines.gpkg",  # hydromt likes absolute paths
                meta=self.data_catalog.get_source(
                    "river_centerlines_MERIT_Basins"
                ).meta,
            ),
        )

        # landcover
        esa_worldcover = self.data_catalog.get_rasterdataset(
            land_cover,
            geom=self.geoms["areamaps/regions"],
            buffer=200,  # 2 km buffer
        ).chunk({"x": XY_CHUNKSIZE, "y": XY_CHUNKSIZE})
        del esa_worldcover.attrs["_FillValue"]
        esa_worldcover.name = "lulc"
        esa_worldcover = esa_worldcover.to_dataset()
        esa_worldcover["_dummy"] = 0
        self.set_forcing(
            esa_worldcover, name="hydrodynamics/esa_worldcover", split_dataset=False
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

            assert (
                len(water_levels.stations) > 0
            ), "No stations found in the region. If no stations should be set, set include_coastal=False"

            path = self.set_forcing(
                water_levels,
                name="hydrodynamics/waterlevel",
                split_dataset=False,
                is_spatial_dataset=False,
                time_chunksize=24 * 6,  # 10 minute data
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

    def _write_grid(self, grid, var, files, is_updated):
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
            grid.to_zarr(filepath, mode="w")

            # also export to tif for easier visualization
            grid.rio.to_raster(filepath.with_suffix(".tif"))

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
                grid, var, self.files["subgrid"], self.is_updated["subgrid"]
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

        if is_spatial_dataset:
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
                        assert (
                            forcing.dims[1] == "y" and forcing.dims[2] == "x"
                        ), "y and x dimensions must be second and third, otherwise xarray will not chunk correctly"
                        chunksizes = {
                            "time": min(forcing.time.size, time_chunksize),
                            "y": min(forcing.y.size, y_chunksize),
                            "x": min(forcing.x.size, x_chunksize),
                        }
                        # forcing = forcing.chunk(chunksizes)
                    else:
                        chunksizes = {"time": min(forcing.time.size, time_chunksize)}
                        # forcing = forcing.chunk(chunksizes)

                    forcing.to_zarr(
                        tmp_file.name,
                        mode="w",
                        encoding={
                            forcing.name: {
                                "compressor": compressor,
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
                    encoding = {forcing.name: {"compressor": compressor}}
                elif isinstance(forcing, xr.Dataset):
                    assert (
                        len(forcing.data_vars) > 0
                    ), "forcing must have more than one variable or name must be set"
                    encoding = {
                        var: {"compressor": compressor} for var in forcing.data_vars
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

    def write_geoms(self, fn: str = "{name}.gpkg", **kwargs) -> None:
        """Write model geometries to a vector file (by default gpkg) at <root>/<fn>

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
            if "driver" not in kwargs:
                kwargs.update(driver="GPKG")
            for name, gdf in self._geoms.items():
                if self.is_updated["geoms"][name]["updated"]:
                    self.logger.debug(f"Writing file {fn.format(name=name)}")
                    self.files["geoms"][name] = fn.format(name=name)
                    _fn = os.path.join(self.root, fn.format(name=name))
                    if not os.path.isdir(os.path.dirname(_fn)):
                        os.makedirs(os.path.dirname(_fn))
                    self.is_updated["geoms"][name]["filename"] = _fn
                    gdf.to_file(_fn, **kwargs)

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
            geom = gpd.read_file(Path(self.root, fn))
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
                    assert data.dims == grid.dims
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
