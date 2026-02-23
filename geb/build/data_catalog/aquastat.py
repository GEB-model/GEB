"""File adapter for AQUASTAT datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from geb.build.workflows.conversions import AQUASTAT_NAME_TO_ISO3
from geb.workflows.io import fetch_and_save

from .base import Adapter


class AQUASTAT(Adapter):
    """Adapter for AQUASTAT datasets."""

    def fetch(self, url: str) -> AQUASTAT:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The AQUASTAT adapter instance.
        """
        if not self.is_ready:
            temp_path = self.path.with_suffix(".csv")
            fetch_and_save(url=url, file_path=temp_path)

            df = pd.read_csv(temp_path, encoding="ISO-8859-1", low_memory=False)

            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

            # Save as parquet for faster reading later
            df.to_parquet(self.path, index=False)

            # Clean up
            temp_path.unlink()
        return self

    def read(self, indicator: str, **kwargs: Any) -> pd.DataFrame:
        """Read the processed AQUASTAT data.

        Args:
            indicator: The specific indicator to filter the data by (e.g., "Municipal water withdrawal per capita (total population) [m3/inhab/year]").
            **kwargs: Additional keyword arguments to pass to pandas read_parquet.

        Returns:
            The AQUASTAT data as a pandas DataFrame.
        """
        df = pd.read_parquet(self.path, **kwargs)
        df = (df[df["aquastatElement.1"] == indicator]).copy()
        df["ISO3"] = df["AREA"].map(AQUASTAT_NAME_TO_ISO3)
        df = (
            df.drop(
                [
                    "[flagObservationStatus] flagObservationStatus - flagObservationStatus",
                    "[flagMethod] flagMethod - flagMethod",
                    "aquastatElement",
                    "aquastatElement.1",
                    "REF_AREA",
                    "AREA",
                    "timePointYears.1",
                ],
                axis=1,
            )
            .set_index("ISO3")
            .rename(
                columns={
                    "timePointYears": "Year",
                    "Value": "Value",
                }
            )
        )

        return df
