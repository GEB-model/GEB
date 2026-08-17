"""Adapter for World Bank Commodity Price Data (Pink Sheet)."""

from typing import Any

import pandas as pd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class WorldBankPinkSheetData(Adapter):
    """Adapter for World Bank Commodity Price Data (Pink Sheet)."""

    def fetch(self, url: str) -> "WorldBankPinkSheetData":
        """Fetch the World Bank Pink Sheet Excel file.

        Args:
            url: URL of the World Bank Pink Sheet Excel file.

        Returns:
            The adapter instance.
        """
        if not self.is_ready:
            fetch_and_save(url, self.path)

        return self

    def read(
        self,
        sheet_name: str = "Annual Prices (Nominal)",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read data from the downloaded Pink Sheet workbook.

        Args:
            sheet_name: Excel sheet to read.
            **kwargs: Additional arguments passed to pandas.read_excel.

        Returns:
            Pink Sheet data as a pandas DataFrame.
        """
        return pd.read_excel(
            self.path,
            sheet_name=sheet_name,
            **kwargs,
        )
