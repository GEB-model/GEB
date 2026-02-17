"""Adapter for GPS country and individual preference datasets in Stata format."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .base import Adapter


class GlobalPreferencesSurvey(Adapter):
    """Adapter for manually downloaded GPS preference Stata datasets.

    This adapter expects users to manually download a GPS dataset zip file and place
    it in the adapter cache root. It then extracts one Stata file from the zip and
    stores it as a CSV file for downstream use.
    """

    def __init__(
        self, zip_filename: str, zip_member_path: str, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the adapter.

        Args:
            zip_filename: Filename of the downloaded zip archive.
            zip_member_path: Relative file path inside the downloaded zip archive.
            *args: Positional arguments passed to :class:`Adapter`.
            **kwargs: Keyword arguments passed to :class:`Adapter`.
        """
        super().__init__(*args, **kwargs)
        self.zip_filename: str = zip_filename
        self.zip_member_path: str = zip_member_path

    def fetch(self, url: str) -> GlobalPreferencesSurvey:
        """Fetch and process manually downloaded GPS preference data.

        Notes:
            The data source requires manual download from the provider website.
            A specific zip file is expected in the dataset cache root.

        Args:
            url: Website URL where users can download the source zip file.

        Returns:
            The adapter instance with processed CSV output.

        Raises:
            ValueError: If the expected Stata file is missing in the selected zip.
        """
        if not self.is_ready:
            zip_path: Path = self.root / self.zip_filename

            while not zip_path.exists():
                print(
                    "\033[91mThis file requires manual download due to licensing restrictions. "
                    f"Please download the GPS dataset zip file '{self.zip_filename}' from: {url} "
                    f"and place it in: {self.root}. The archive must contain: "
                    f"{self.zip_member_path}\033[0m"
                )
                input(
                    "\033[91mPress Enter after placing the file to continue...\033[0m"
                )

            with zipfile.ZipFile(file=zip_path, mode="r") as zip_ref:
                if self.zip_member_path not in zip_ref.namelist():
                    raise ValueError(
                        f"Expected file {self.zip_member_path} not found in {zip_path}."
                    )
                with zip_ref.open(self.zip_member_path) as stata_file:
                    dataframe: pd.DataFrame = pd.read_stata(stata_file)

            dataframe.to_csv(self.path, index=False)

        return self
