"""Dataset adapters for SUPERWELL data."""

from __future__ import annotations

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
            fetch_and_save(url=url, file_path=self.path)
        return self

    def read(self) -> dict[str, float]:
        """Read the dataset and map countries to ISO3.

        Returns:
            Dictionary mapping ISO3 country codes to electricity rates (USD/kWh).
        """
        df = pd.read_csv(
            self.path, names=["country", "rate_usd_2006_per_kwh"], skiprows=1
        )
        df["ISO3"] = df["country"].map(SUPERWELL_NAME_TO_ISO3)
        # Drop rows where ISO3 could not be mapped if necessary,
        # but here we keep the original logic which just maps and sets index.
        return df.set_index("ISO3")["rate_usd_2006_per_kwh"].to_dict()
