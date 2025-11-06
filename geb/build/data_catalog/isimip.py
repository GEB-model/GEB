"""ISIMIP CO2 data adapter."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
import requests

from .base import Adapter


class ISIMIPCO2(Adapter):
    """Adapter for ISIMIP CO2 data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ISIMIP CO2 data adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> ISIMIPCO2:
        """Set the URL for the ISIMIP CO2 data source.

        Args:
            url: The URL of the ISIMIP CO2 data source.

        Returns:
            The current instance of the ISIMIPCO2 adapter.
        """
        self.url = url
        return self

    def read(self, scenario: str, **kwargs: Any) -> pd.DataFrame:
        """Read the ISIMIP CO2 data for a specific scenario.

        Args:
            scenario: The scenario to read (e.g., "ssp126", "ssp370", "ssp585").
            **kwargs: Additional keyword arguments (not used).

        Returns:
            A pandas DataFrame containing the CO2 concentration data with years as index and a column 'co2_ppm'.

        Raises:
            ValueError: If the scenario is invalid.
        """
        if scenario not in ["ssp126", "ssp370", "ssp585"]:
            raise ValueError(
                "Invalid scenario. Choose from 'ssp126', 'ssp370', or 'ssp585'."
            )

        historical_url: str = f"{self.url}/ISIMIP3b/InputData/climate/atmosphere_composition/co2/historical/co2_historical_annual_1850_2014.txt"
        response = requests.get(historical_url)
        response.raise_for_status()
        historical_data = response.text

        future_url: str = f"{self.url}/ISIMIP3b/InputData/climate/atmosphere_composition/co2/{scenario}/co2_{scenario}_annual_2015_2100.txt"
        response = requests.get(future_url)
        response.raise_for_status()
        future_data = response.text

        combined_data = historical_data + "\n" + future_data

        df: pd.DataFrame = pd.read_csv(
            StringIO(combined_data), delim_whitespace=True, names=["year", "co2_ppm"]
        ).set_index("year")
        return df
