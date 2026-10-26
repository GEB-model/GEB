"""Utilities for processing CLIMRISK data.

This module reads locally stored CLIMRISK MATLAB files, crops the data to
selected countries, and returns the result as an xarray Dataset.
"""

from __future__ import annotations

import warnings
from typing import Any

import geopandas as gpd
import h5py
import numpy as np
from scipy.io import loadmat
import shapely
import xarray as xr

from geb.build.data_catalog.base import Adapter



class CLIMRISK(Adapter):
    """Read and process locally stored CLIMRISK data."""

    # Keeping the URL on the class makes the data source easy to find and
    # override in tests without making it specific to any adapter instance.
    ISO_COUNTRY_DATABASE_URL: str = (
        "https://raw.githubusercontent.com/pycountry/pycountry/"
        "refs/heads/main/src/pycountry/databases/iso3166-1.json"
    )


    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the adapter.

        Args:
            args: Additional positional arguments passed to the Adapter.
            kwargs: Additional keyword arguments passed to the Adapter.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> CLIMRISK:
        """Return this adapter instance.

        CLIMRISK data must currently be obtained and stored locally.

        Args:
            url: URL required by the adapter interface. It is currently unused.

        Returns:
            This CLIMRISK adapter instance.
        """
        return self

    def read(
        self,
        rcp: str | None,
        damage_function: str | None,
        warming_percentile: str | None,
        region: gpd.GeoDataFrame | None = None,
        ssp: str = "2",
        reference_year: int = 2020,
        obtain_population: bool = False,
    ) -> xr.DataArray:
        """Load CLIMRISK data for the part of the grid covered by a region.

        Finds country codes at grid-cell centers within ``region``, crops the
        data to their bounding box (padded by one cell), and masks cells
        belonging to other countries. Returns population data when
        ``obtain_population`` is true; otherwise, returns GDP damage data.

        Args:
            rcp: RCP identifier. Supported values are ``"19"``, ``"26"``,
                ``"34"``, ``"45"``, ``"60"``, ``"70"``, and ``"85"``.
                Defaults to ``"70"`` when omitted for damage data.
            damage_function: Damage function identifier: ``"K"``, ``"KU"``,
                ``"KPU"``, ``"RU"``, or ``"RPU"``. Defaults to ``"K"``
                when omitted. ``"GDP"`` selects the GDP data file.
            warming_percentile: Global warming percentile: ``"10"``, ``"50"``, or ``"90"``.
                Defaults to ``"90"`` when omitted for damage data.
            region: Non-empty GeoDataFrame defining the study region. It must
                have a CRS and is reprojected to EPSG:4326 for grid matching.
            ssp: SSP scenario identifier, from ``"1"`` through ``"5"``.
                Defaults to ``"2"``.
            reference_year: First year to include, from 2010 through 2100.
            obtain_population: If true, return population data instead of
                damage data.

        Returns:
            An xarray DataArray with ``year``, ``lon``, and ``lat`` dimensions.
            Its name is ``"population"`` or ``"GDP"`` depending on the
            selected data. A ``country`` coordinate contains country-grid codes.

        Raises:
            FileNotFoundError: If the adapter path or a required map file is
                missing.
            ValueError: If the region, scenario parameters, or required data
                variables are invalid, or no country-grid cells are found
                within the region.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"The local path {self.path} does not exist. "
                "Please download and extract the data first."
            )
        if region is None or region.empty:
            raise ValueError("Provide a non-empty region GeoDataFrame.")
        if region.crs is None:
            raise ValueError("The region GeoDataFrame must have a CRS.")

        if ssp not in ["1", "2", "3", "4", "5"]:
            raise ValueError(f"Possible climrisk SSPs: 1, 2, 3, 4, 5; got {ssp!r}.")
        if not 2010 <= reference_year <= 2100:
            raise ValueError(
                "reference_year must be between 2010 and 2100; "
                f"got {reference_year}."
            )

        country_file = self.path / "KompasCountry.mat"
        if not country_file.exists():
            raise FileNotFoundError(f"Country map file not found: {country_file}")

        country_data: dict[str, Any] = loadmat(country_file)
        if "CountryMap" not in country_data:
            raise ValueError(
                f"The country map file has no 'CountryMap': {country_file}"
            )
        # The MATLAB map's latitude order is opposite to the ascending
        # latitude coordinate used to label the xarray data.
        country_grid: np.ndarray = (
            np.flip(country_data["CountryMap"], axis=0)
            .transpose(1, 0)
        )

        # These coordinates describe the grid represented by the source arrays.
        dimensions: tuple[str, str, str] = ("year", "lon", "lat")
        years: np.ndarray = np.arange(2010, 2101)
        longitudes: np.ndarray = np.arange(-179.75, 180.25, 0.5)
        latitudes: np.ndarray = np.arange(-89.75, 90.25, 0.5)

        if obtain_population:
            population_file = self.path / "OECD_POP_SSP2_SPATIAL13R.mat"
            if not population_file.exists():
                raise FileNotFoundError(f"Population map file not found: {population_file}")
            population_data: dict[str, Any] = loadmat(population_file)
            if "POP_MAP" not in population_data:
                raise ValueError(
                    f"The population map file has no 'POP_MAP': {population_file}"
                )
            # The MATLAB map's latitude order is opposite to the ascending
            # latitude coordinate used to label the xarray data.
            population_grid: np.ndarray = (
                np.flip(population_data["POP_MAP"], axis=0)
                .transpose(2, 1, 0)
            )
            climrisk_data: xr.DataArray = xr.DataArray(
                data=population_grid,
                dims=dimensions,
                coords={
                    "year": years,
                    "lon": longitudes,
                    "lat": latitudes,
                    "country": (("lon", "lat"), country_grid),
                },
                name="population"
            )
        else:
            if warming_percentile is None: warming_percentile = "90"
            if rcp is None: rcp = "70"
            if damage_function is None: damage_function = "K"
            if warming_percentile not in ["10", "50", "90"]:
                raise ValueError(f"Possible climrisk warming percentiles: 10, 50, 90; got {warming_percentile!r}.")
            if rcp not in ["19", "26", "34", "45", "60", "70", "85"]:
                raise ValueError(f"Possible climrisk RCPs: 19, 26, 34, 45, 60, 70, 85; got {rcp!r}.")
    
            # Multiple function identifiers share the same CLIMRISK output file.
            file_groups: dict[str, tuple[str, str]] = {
                "GDP": ("Kompas_", "K_KU"), 
                "K": ("Kompas_", "K_KU"),
                "KU": ("Kompas_", "K_KU"),
                "KPU": ("Kompas_", "K_KU"),
                "RU": ("", "RU_RPU"),
                "RPU": ("", "RU_RPU"),
            }
            if damage_function not in file_groups:
                raise ValueError(f"Possible climrisk damage functions: K, KU, KPU, RU, RPU; got {damage_function!r}.")
    
            file_prefix: str
            damage_function_abbreviation: str
            file_prefix, damage_function_abbreviation = file_groups[damage_function]
    
            if damage_function=="GDP":
                damage_file = (
                    self.path
                    / f"SSP{ssp}26_Kompas_results_p10"
                    / f"SSP{ssp}26_K_KU_results10.mat"
                )
            else:
                scenario: str = f"{ssp}{rcp}"
                damage_file = (
                    self.path
                    / f"SSP{scenario}_{file_prefix}results_p{warming_percentile}"
                    / f"SSP{scenario}_{damage_function_abbreviation}_results{warming_percentile}.mat"
                )
    
            with h5py.File(damage_file, "r") as file:
                if damage_function=="GDP":
                    damage_key: str = f"CLIMRISK_DOLLARS_K_GRID_WORLD_S"
                    proportion_key: str = f"CLIMRISK_PROP_K_GRID_WORLD_S"
                else:
                    damage_key: str = f"CLIMRISK_DOLLARS_{damage_function}_GRID_WORLD_S"
                    proportion_key: str = f"CLIMRISK_PROP_{damage_function}_GRID_WORLD_S"
    
                if damage_key not in file or proportion_key not in file:
                    raise ValueError(
                        f"Expected variables {damage_key!r} and {proportion_key!r} "
                        f"were not found in {damage_file}."
                    )
    
                with warnings.catch_warnings():
                    # Zero proportions can cause divide-by-zero values in the
                    # source formula; suppress the warning while retaining results.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    damage_dollars: np.ndarray = file[damage_key][:]
                    damage_proportion: np.ndarray = file[proportion_key][:].clip(
                        max=100
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        if damage_function=="GDP":
                            gdp_damage: np.ndarray = (
                                damage_dollars
                                *100/damage_proportion
                            )
                        else:
                            gdp_damage: np.ndarray = (
                                damage_dollars
                                * (100 - damage_proportion)
                                / damage_proportion
                            )
    
                climrisk_data: xr.DataArray = xr.DataArray(
                    data=gdp_damage,
                    dims=dimensions,
                    coords={
                        "year": years,
                        "lon": longitudes,
                        "lat": latitudes,
                        "country": (("lon", "lat"), country_grid),
                    },
                    name="GDP",
                )

        # Obtain country codes where the study region is located
        region_geometry = region.to_crs("EPSG:4326").geometry.union_all()
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes, indexing="ij")
        inside = shapely.intersects_xy(region_geometry, lon_grid, lat_grid)
        codes = country_grid[inside]
        codes = codes[np.isfinite(codes) & (codes > 0)]
        country_code_numbers = np.unique(codes).astype(int).tolist()

        # Find the smallest rectangular crop containing all requested countries.
        country_mask: xr.DataArray = climrisk_data["country"].isin(
            country_code_numbers
        )
        matching_longitudes: np.ndarray = np.flatnonzero(
            country_mask.any(dim="lat").to_numpy()
        )
        matching_latitudes: np.ndarray = np.flatnonzero(
            country_mask.any(dim="lon").to_numpy()
        )
        if not country_code_numbers or not matching_longitudes.size or not matching_latitudes.size:
            raise ValueError("No country-grid cells were found within the region.")

        # Pad values by one cell to avoid out of boundary errors in basins located at the extremes of our regions
        first_longitude_index: int = int( max(matching_longitudes[0]-1, 0))
        last_longitude_index: int = int( min(matching_longitudes[-1]+1, longitudes.shape[0]-1))
        first_latitude_index: int = int( max(matching_latitudes[0]-1, 0))
        last_latitude_index: int = int( min(matching_latitudes[-1]+1, latitudes.shape[0]-1))

        longitude_bounds: tuple[float, float] = (
            float(longitudes[first_longitude_index]),
            float(longitudes[last_longitude_index]),
        )
        latitude_bounds: tuple[float, float] = (
            float(latitudes[first_latitude_index]),
            float(latitudes[last_latitude_index]),
        )

        # Crop inclusively, then mask other countries inside the bounding box.
        climrisk_data = climrisk_data.sel(
            {
                "lon":slice(*longitude_bounds),
                "lat":slice(*latitude_bounds),
                "year":slice(reference_year, 2100),
            }
        )
        climrisk_data = climrisk_data.where(
            climrisk_data["country"].isin(country_code_numbers)
        )
        climrisk_data.attrs.update(
            {
                "lon_origin": longitude_bounds[0],
                "lon_resolution": 0.5,
                "lat_origin": latitude_bounds[0],
                "lat_resolution": 0.5,
                "_FillValue": np.nan
            }
        )
        climrisk_data = xr.apply_ufunc(
            np.nan_to_num,
            climrisk_data,
            kwargs={"nan": np.nan, "posinf": np.nan, "neginf": np.nan},
            dask="allowed",
        )
        return climrisk_data.astype(np.float32)