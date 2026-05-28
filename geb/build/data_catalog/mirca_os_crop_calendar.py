"""Adapter for MIRCA-OS crop calendar datasets."""

import pandas as pd

from .base import Adapter


class MIRCAOSCropCalendar(Adapter):
    """Adapter for MIRCA-OS crop calendar datasets."""

    def fetch(self, url: str) -> MIRCAOSCropCalendar:
        """Fetch the MIRCA-OS crop calendar dataset.

        Args:
            url: The URL to download the dataset from.

        Returns:
            MIRCAOSCropCalendar: The adapter instance.
        """
        if not self.is_ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            while not self.path.parent.exists() or not any(
                self.path.parent.glob("*.csv")
            ):
                print(
                    "\033[91mThis dataset requires manual download and extraction. "
                    + f"Please download MIRCA-OS Crop Calendar from: {url}\n"
                    + f"Then extract the contents and place the CSV files (e.g., MIRCA-OS_2000_ir.csv) at: {self.path.parent}\033[0m"
                )
                input(
                    "\033[91mPress Enter after placing the folder to continue...\033[0m"
                )

        return self

    def read(self) -> pd.DataFrame:
        """Read the crop calendar CSV file.

        Returns:
            pd.DataFrame: The crop calendar data.
        """
        return pd.read_csv(self.path, encoding="ISO-8859-1")
