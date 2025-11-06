"""The Lowder adapter for downloading and processing farm size distribution data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from geb.workflows.io import fetch_and_save

from ..workflows.conversions import (
    COUNTRY_NAME_TO_ISO3,
)
from .base import Adapter


class Lowder(Adapter):
    """The Lowder adapter for downloading and processing farm size distribution data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Lowder adapter."""
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> Lowder:
        """Fetch the Lowder farm size distribution data from the given URL.

        Args:
            url: The URL to download the Lowder farm size distribution data from.

        Returns:
            The instance of the Adapter after fetching the data.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path)
        return self

    def read(self, **kwargs: Any) -> pd.DataFrame:
        """Read and process the Lowder farm size distribution data.

        Args:
            **kwargs: Additional keyword arguments to pass to pandas.read_excel.

        Returns:
            A pandas DataFrame containing the processed farm size distribution data.
        """
        df = (
            super()
            .read(
                sheet_name="WEB table 3",
                skiprows=4,
                skipfooter=2,
                header=None,
                names=[
                    "Country",
                    "Census Year",
                    "Holdings/ agricultural area",
                    "Total",
                    "< 1 Ha",
                    "1 - 2 Ha",
                    "2 - 5 Ha",
                    "5 - 10 Ha",
                    "10 - 20 Ha",
                    "20 - 50 Ha",
                    "50 - 100 Ha",
                    "100 - 200 Ha",
                    "200 - 500 Ha",
                    "500 - 1000 Ha",
                    "> 1000 Ha",
                    "empty",
                    "income class",
                ],
                **kwargs,
            )
            .dropna(subset=["Total"], axis=0)
            .drop(["empty", "income class"], axis=1)
        )
        df["Country"] = df["Country"].ffill()
        # Remove preceding and trailing white space from country names
        df["Country"] = df["Country"].str.strip()
        df["Census Year"] = df["Country"].ffill()

        df["ISO3"] = df["Country"].map(COUNTRY_NAME_TO_ISO3)

        # Clean up agricultural area strings
        df["Holdings/ agricultural area"] = (
            df["Holdings/ agricultural area"].str.strip().str.replace("  ", " ")
        )

        assert not df["ISO3"].isna().any(), (
            f"Found {df['ISO3'].isna().sum()} countries without ISO3 code"
        )
        return df
