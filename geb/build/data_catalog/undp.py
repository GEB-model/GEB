"""Adapter for UNDP datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class HumanDevelopmentIndex(Adapter):
    """Adapter for Human Development Index datasets."""

    def fetch(self, url: str) -> HumanDevelopmentIndex:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The HumanDevelopmentIndex adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path)
        return self

    def read(self, **kwargs: Any) -> pd.DataFrame:
        """Read the dataset into a pandas DataFrame.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.

        Returns:
            The dataset as a pandas DataFrame.

        Notes:
            The data is renamed to match the expected format in GEB:
            - 'code' -> 'Code' (ISO3 code)
            - 'year' -> 'Year'
            - 'hdi__sex_total' -> 'Human Development Index'
        """
        df = pd.read_csv(self.path, **kwargs)
        # Rename columns to match what's expected in the GEB workflows
        # OWID format: entity,code,year,hdi__sex_total,owid_region
        # Expected: Code, Year, Human Development Index (historical expected name)
        df = df.rename(
            columns={
                "code": "Code",
                "year": "Year",
                "hdi__sex_total": "Human Development Index",
            }
        )
        return df
