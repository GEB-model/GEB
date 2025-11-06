"""Simple file adapter for downloading and saving files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class WorldBankData(Adapter):
    """Adapter for generic file download and save."""

    def fetch(self, url: str) -> WorldBankData:
        """Fetch file from URL and save to local path.

        Args:
            url (str): URL of the file to download.

        Returns:
            File: Instance of the File adapter with the downloaded file path.
        """
        if not self.is_ready:
            download_path: Path = self.root / "zipfile.zip"
            fetch_and_save(url, download_path)

            # list all files in the zip and extract the first one
            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                files = zip_ref.namelist()
                data_files = [file for file in files if file.startswith("API_")]
                assert len(data_files) == 1, (
                    "Expected exactly one data file in the zip."
                )
                data_file = data_files[0]
                zip_ref.extract(data_file, self.root)

            download_path.unlink()  # remove zip file

            extracted_file_path = self.root / data_file
            extracted_file_path.rename(self.path)  # rename to desired filename

        return self

    def read(self, **kwargs: Any) -> pd.DataFrame:
        """Read the downloaded file.

        Args:
            **kwargs: Additional keyword arguments for reading the file.

        Returns:
            The data as a pandas DataFrame.
        """
        return pd.read_csv(self.path, skiprows=4, **kwargs)
