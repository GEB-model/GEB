"""Dataset adapters for SUPERWELL data."""

import pandas as pd

from geb.workflows.io import fetch_and_save

from ..workflows.conversions import SUPERWELL_NAME_TO_ISO3
from .base import Adapter


class GCAMElectricityRates(Adapter):
    """Adapter for GCAM Electricity Rates."""

    def fetch(self, url: str) -> GCAMElectricityRates:
        """Fetch the dataset from the given URL if not already present.

        Args:
            url: The URL to fetch the dataset from.

        Returns:
            The GCAMElectricityRates adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url=url, file_path=self.path, logger=self.logger)
        return self

    def read(self) -> dict[str, float]:
        """Read the dataset and map countries to ISO3.

        Returns:
            Dictionary mapping ISO3 country codes to electricity rates
            (USD, nominal 2006, per kWh).

        Raises:
            ValueError: If a country name cannot be mapped to ISO3.
        """
        # The source has no header; skipping a row would discard Afghanistan.
        df: pd.DataFrame = pd.read_csv(
            self.path,
            names=["country", "rate_usd_2006_per_kwh"],
            header=None,
            encoding="utf-8-sig",
        )
        # Source country names contain trailing spaces, unlike the ISO3 mapping.
        df["country"] = df["country"].str.strip()
        df["ISO3"] = df["country"].map(SUPERWELL_NAME_TO_ISO3)
        unknown_countries: list[str] = df.loc[df["ISO3"].isna(), "country"].tolist()
        if unknown_countries:
            raise ValueError(
                f"Electricity rate countries cannot be mapped to ISO3: {unknown_countries}"
            )
        return df.set_index("ISO3")["rate_usd_2006_per_kwh"].to_dict()
