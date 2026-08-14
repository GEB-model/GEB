"""Utilities to process IISa SSP data.

This module provides a downloader that downloads and extracts the needed
IISa SSP data from the remote dataset. The data is downloaded
from the specified URL and stored to a local file in the GEB data catalog.

"""

from typing import Any

import pandas as pd

from geb.build.data_catalog.base import Adapter
import numpy as np
import scipy


class IIASA_SSP(Adapter):
    """Class to handle IISa SSP data.

    This class provides methods to download, extract, and process the IISa SSP data.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the IISaSSPData class.

        Args:
            url (str): The URL to download the IISa SSP data from.
            local_path (str): The local path to store the downloaded data.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> IIASA_SSP:
        """Download the IISa SSP data from the specified URL."""
        # Implementation for downloading the data goes here
        # by default, you receive the latest SSP projections (2024 release)
        return self

    def read(self, country="Mexico", ssp="SSP5", reference_year=2020) -> pd.DataFrame:
        """Process the extracted IISa SSP data and return it as a DataFrame.

        Args:
            country (str): The country for which to process data.
            ssp (str): The SSP scenario for which to process data.
            reference_year (int): The reference year for the SSPs. Default is 2020.
        Raises:
            FileNotFoundError: If the local path does not exist.
        Returns:
            pd.DataFrame: A DataFrame containing the processed IISa SSP data for the specified country and SSP scenario, with GDP values scaled to the reference year.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"The local path {self.path} does not exist. Please download and extract the data first."
            )
        # Read the csv file and process the data for the specified country
        df = pd.read_csv(self.path)
        df = df[
            (df["Region"] == country)
            & ((df["Scenario"] == ssp) | (df["Scenario"] == "Historical Reference"))
            # & (df["Model"] == "IIASA GDP 2023"),
            & (df["Variable"] == "GDP|PPP")
        ]
        # create a continuous series of GDP values for the specified country and SSP scenario
        historical_df = df[df["Scenario"] == "Historical Reference"]
        ssp_df = df[(df["Scenario"] == ssp) & (df["Model"] == "IIASA GDP 2023")]

        combined_df = pd.DataFrame(index=range(2020, 2101))
        combined_df.index.name = "Year"

        for year in range(2020, 2021):
            col = str(year)
            if col in historical_df.columns:
                combined_df.loc[year, "GDP"] = historical_df[col].iloc[0]

        for year in range(2021, 2101):
            col = str(year)
            if col in ssp_df.columns:
                combined_df.loc[year, "GDP"] = ssp_df[col].iloc[0]

        combined_df["GDP"] = combined_df["GDP"].interpolate()

        # add column scaled to reference year
        combined_df["GDP_scaled"] = (
            combined_df["GDP"] / combined_df.loc[reference_year, "GDP"]
        )
        # add column for GDP growth rate relative to reference year
        combined_df["GDP_growth_rate"] = combined_df["GDP"].pct_change().fillna(0)

        return combined_df
