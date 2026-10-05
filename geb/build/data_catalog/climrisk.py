"""Utilities for processing CLIMRISK data.

This module reads locally stored CLIMRISK MATLAB files, crops the data to
selected countries, and returns the result as an xarray Dataset.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from typing import Any
from urllib.request import urlopen

import h5py
import numpy as np
import xarray as xr
from scipy.io import loadmat

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

        Returns:
            None.
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
        countries: list[str] | None = None,
        ssp: str = "2",
        rcp: str = "70",
        function: str = "K",
        p: str = "90",
        reference_year: int = 2020,
    ) -> xr.Dataset:
        """Load CLIMRISK data for selected countries and scenario settings.

        The returned dataset contains GDP damage data, cropped to the bounding
        rectangle of the selected countries and masked to those countries.

        Args:
            countries: Country names to include. Defaults to Mexico.
            ssp: SSP scenario identifier, such as ``"2"``.
            rcp: RCP scenario identifier. Supported values are ``"26"``,
                ``"45"``, and ``"70"``.
            function: Damage function identifier. Supported values are
                ``"K"``, ``"KU"``, ``"KPU"``, ``"RU"``, and ``"RPU"``.
            p: Damage percentile. Supported values are ``"10"``, ``"50"``,
                and ``"90"``.
            reference_year: First year to include, from 2010 through 2100.

        Returns:
            An xarray Dataset containing GDP damage data, country codes, and
            spatial metadata.

        Raises:
            FileNotFoundError: If the adapter path or a required data file
                does not exist.
            ValueError: If an input is invalid or no grid cells match the
                requested countries.
            RuntimeError: If the country database cannot be fetched or parsed.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"The local path {self.path} does not exist. "
                "Please download and extract the data first."
            )

        if countries is None:
            countries = ["Mexico"]
        if not countries:
            raise ValueError("At least one country must be specified.")
        if p not in ["10", "50", "90"]:
            raise ValueError(f"Available percentiles: 10, 50, 90; got {p}.")
        if not 2010 <= reference_year <= 2100:
            raise ValueError(
                "reference_year must be between 2010 and 2100; "
                f"got {reference_year}."
            )

        # Multiple function identifiers share the same CLIMRISK output file.
        file_groups: dict[str, tuple[str, str]] = {
            "K": ("Kompas_", "K_KU"),
            "KU": ("Kompas_", "K_KU"),
            "KPU": ("Kompas_", "K_KU"),
            "RU": ("", "RU_RPU"),
            "RPU": ("", "RU_RPU"),
        }
        if function not in file_groups:
            supported_functions: str = ", ".join(file_groups)
            raise ValueError(
                f"Unsupported damage function {function!r}. "
                f"Choose from: {supported_functions}."
            )

        file_prefix: str
        function_abbreviation: str
        file_prefix, function_abbreviation = file_groups[function]

        # ISO numeric codes are strings so leading zeros are preserved.
        # The MATLAB country map stores them as numbers, so convert for matching.
        country_codes: list[str] = self.get_country_codes(countries)
        country_code_numbers: list[int] = [int(code) for code in country_codes]

        country_file = self.path / "KompasCountry.mat"
        if not country_file.exists():
            raise FileNotFoundError(f"Country map file not found: {country_file}")

        country_data: dict[str, Any] = loadmat(country_file)
        if "CountryMap" not in country_data:
            raise ValueError(
                f"The country map file has no 'CountryMap': {country_file}"
            )

        scenario: str = f"{ssp}{rcp}"
        damage_file = (
            self.path
            / "climrisk_output"
            / f"SSP{scenario}_{file_prefix}results_p{p}"
            / f"SSP{scenario}_{function_abbreviation}_results{p}.mat"
        )

        # These coordinates describe the grid represented by the source arrays.
        dimensions: tuple[str, str, str] = ("time", "lon", "lat")
        years: np.ndarray = np.arange(2010, 2101)
        longitudes: np.ndarray = np.arange(-179.75, 180.25, 0.5)
        latitudes: np.ndarray = np.arange(-89.75, 90.25, 0.5)

        with h5py.File(damage_file, "r") as file:
            damage_key: str = f"CLIMRISK_DOLLARS_{function}_GRID_WORLD_S"
            proportion_key: str = f"CLIMRISK_PROP_{function}_GRID_WORLD_S"

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
                    gdp_damage: np.ndarray = (
                        damage_dollars
                        * (100 - damage_proportion)
                        / damage_proportion
                    )

            # The MATLAB map's latitude order is opposite to the ascending
            # latitude coordinate used to label the xarray data.
            country_grid: np.ndarray = np.flip(
                country_data["CountryMap"], axis=0
            )

            data_array: xr.DataArray = xr.DataArray(
                data=gdp_damage,
                dims=dimensions,
                coords={
                    "time": years,
                    "lon": longitudes,
                    "lat": latitudes,
                    "country": (("lat", "lon"), country_grid),
                },
                name="GDP",
            )

        # Find the smallest rectangular crop containing all requested countries.
        country_mask: xr.DataArray = data_array["country"].isin(
            country_code_numbers
        )
        matching_longitudes: np.ndarray = np.flatnonzero(
            country_mask.any(dim="lat").to_numpy()
        )
        matching_latitudes: np.ndarray = np.flatnonzero(
            country_mask.any(dim="lon").to_numpy()
        )

        if matching_longitudes.size == 0 or matching_latitudes.size == 0:
            raise ValueError(
                f"No grid cells found for the requested countries: {countries}"
            )

        first_longitude_index: int = int(matching_longitudes[0])
        last_longitude_index: int = int(matching_longitudes[-1])
        first_latitude_index: int = int(matching_latitudes[0])
        last_latitude_index: int = int(matching_latitudes[-1])

        longitude_bounds: tuple[float, float] = (
            float(longitudes[first_longitude_index]),
            float(longitudes[last_longitude_index]),
        )
        latitude_bounds: tuple[float, float] = (
            float(latitudes[first_latitude_index]),
            float(latitudes[last_latitude_index]),
        )

        # Crop inclusively, then mask other countries inside the bounding box.
        data_array = data_array.sel(
            lon=slice(*longitude_bounds),
            lat=slice(*latitude_bounds),
            time=slice(reference_year, 2100),
        )
        data_array = data_array.where(
            data_array["country"].isin(country_code_numbers)
        )

        dataset: xr.Dataset = data_array.to_dataset(name="GDP")
        dataset.attrs.update(
            {
                "lon_origin": longitude_bounds[0],
                "lon_resolution": 0.5,
                "lat_origin": latitude_bounds[0],
                "lat_resolution": 0.5,
            }
        )

        return dataset

    @staticmethod
    @cache
    def _load_country_name_to_code() -> dict[str, str]:
        """Fetch and parse the ISO 3166-1 country database.

        The parsed mapping is cached after a successful fetch, avoiding
        repeated network requests during this process.

        Returns:
            Mapping from case-folded country names to three-character numeric
            ISO codes.

        Raises:
            RuntimeError: If the database cannot be fetched or has an invalid
                format.
        """
        try:
            # The context manager closes the network response even if reading
            # or parsing the downloaded data fails.
            with urlopen(CLIMRISK.ISO_COUNTRY_DATABASE_URL, timeout=15) as response:
                payload: bytes = response.read()

            # Decode the response as UTF-8, then parse the JSON document.
            database: Any = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Could not fetch or parse the ISO country database at "
                f"{CLIMRISK.ISO_COUNTRY_DATABASE_URL}."
            ) from error

        # Check the top-level structure before accessing its country records.
        records_value: Any = database.get("3166-1") if isinstance(database, dict) else None
        if not isinstance(records_value, list):
            raise RuntimeError("The ISO country database has an unexpected format.")

        country_records: list[Any] = records_value
        name_to_code: dict[str, str] = {}

        # Check that all values are valid
        for country_record in country_records:
            if not isinstance(country_record, dict):
                raise RuntimeError(
                    "The ISO country database contains an invalid country record."
                )

            numeric_value: Any = country_record.get("numeric")
            if not isinstance(numeric_value, (str, int)):
                raise RuntimeError(
                    "The ISO country database contains an invalid numeric code."
                )

            numeric_code: str = str(numeric_value)
            if not numeric_code.isdecimal() or len(numeric_code) > 3:
                raise RuntimeError(
                    "The ISO country database contains an invalid numeric code."
                )

            # ISO numeric codes are three digits; padding preserves leading
            # zeros if a code was represented as an integer in the JSON.
            numeric_code = numeric_code.zfill(3)

            # Index all available name variants so callers can use either the
            # short name, official name, or common name.
            for field_name in ("name", "official_name", "common_name"):
                country_name: Any = country_record.get(field_name)
                if isinstance(country_name, str):
                    name_to_code[country_name.casefold()] = numeric_code

        return name_to_code

    @staticmethod
    def get_country_codes(country_names: list[str]) -> list[str]:
        """Convert country names to ISO 3166-1 numeric codes.

        Name matching is case-insensitive. Input order and duplicates are
        preserved, and codes remain strings so leading zeros are retained.

        Args:
            country_names: Country names to convert.

        Returns:
            Three-character ISO numeric codes for the supplied countries.

        Raises:
            ValueError: If a name is empty, is not a string, or is
                unrecognized.
            RuntimeError: If the country database cannot be fetched or parsed.
        """
        name_to_code: dict[str, str] = CLIMRISK._load_country_name_to_code()
        country_codes: list[str] = []

        for country_name in country_names:
            if not isinstance(country_name, str) or not country_name.strip():
                raise ValueError(f"Invalid country name: {country_name!r}")

            normalized_name: str = country_name.strip().casefold()
            if normalized_name not in name_to_code:
                raise ValueError(f"Unrecognized country name: {country_name}")

            country_codes.append(name_to_code[normalized_name])

        return country_codes