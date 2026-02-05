"""Utilities to fetch and convert Global Exposure Model (GEM) CSV exposures.

This module provides an adapter that downloads CSV files from the
`gem/global_exposure_model` GitHub repository for one or more countries,
processes those CSVs to compute damages per square metre for residential
(`OCCUPANCY == 'Res'`) assets and writes the aggregated parameters to a
JSON file named `global_exposure_model.json` next to the adapter's path.

Operations performed by the adapter:
- Query the GitHub tree API to find CSV file paths for requested countries.
- Download matching raw CSV files from the `raw.githubusercontent.com` URL.
- Parse CSVs with pandas, filter residential rows and aggregate by
    `NAME_1` (admin_1) to compute damage per sqm for several damage columns.
- Merge results across files and write the final parameter mapping using
    `geb.workflows.io.write_params`.

The class is intentionally lightweight: network interactions use
`requests`, temporary files are used for pandas reads, and numerical
division is guarded against zero area values.
"""

from __future__ import annotations

import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from geb.workflows.io import write_params

from .base import Adapter


class GlobalExposureModel(Adapter):
    """Adapter for Global Exposure Model data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GlobalExposureModel adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def _filter_folders(
        self, tree: list[dict[str, Any]], csv_files: list[str], country: str
    ) -> None:
        """Filter the repository tree for folders matching the specified country.

        Args:
            tree: The repository tree from the GitHub API.
            csv_files: List to append found CSV file paths to.
            country: The country name to filter folders by.
        Raises:
            ValueError: If no matching country folder is found.
        """
        # Find folders in the repository tree whose leaf name matches the
        # requested country (case-insensitive). The GitHub tree entries of
        # type 'tree' represent directories.
        matching_paths = [
            item["path"]
            for item in tree
            if item["type"] == "tree"
            and item["path"].split("/")[-1].lower() == country.lower()
        ]

        # Raise an error if no matching folder is found
        if not matching_paths:
            raise ValueError(f"No folder found for country: {country}")

        # Search for CSV-like blobs under any matching folder. We further
        # restrict to paths that include 'exposure_res' (case-insensitive)
        # to focus on residential exposure result files.
        for item in tree:
            if item["type"] == "blob":
                for folder in matching_paths:
                    if (
                        item["path"].startswith(folder + "/")
                        and "exposure_res" in item["path"].lower()
                    ):
                        csv_files.append(item["path"])

    def _download_and_process_csv(self, raw_base: str, csv_files: list[str]) -> None:
        """Download and process CSV files from the repository.

        Args:
            raw_base: The base URL for raw file access in the GitHub repository.
            csv_files: List of CSV file paths to download and process.
        """
        # Will collect per-file dictionaries mapping admin_1 -> damage metrics
        damages_per_sqm = []
        for csv in csv_files:
            url = raw_base + quote(csv)

            r = requests.get(url)
            r.raise_for_status()

            with tempfile.TemporaryDirectory() as temp_dir_str:
                temp_dir: Path = Path(temp_dir_str)
                out_path = temp_dir / os.path.basename(csv)
                with open(out_path, "wb") as f:
                    f.write(r.content)
                # Read the downloaded CSV into a pandas DataFrame for
                # processing. Using a temporary file avoids holding raw
                # bytes in memory and keeps pandas happy with local paths.
                df = pd.read_csv(out_path)
                damages_per_sqm.append(self._process_csv(df))

        # Merge per-file dictionaries. Later files overwrite earlier keys
        # for the same admin_1; this behavior matches a simple update
        # aggregation strategy.
        merged = {}
        for d in damages_per_sqm:
            merged.update(d)

        # and write to file
        os.makedirs(self.path.parent, exist_ok=True)
        write_params(
            merged,
            self.path.with_name("global_exposure_model.json"),
        )

    def _process_csv(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Process a single CSV DataFrame to compute damages per square meter.

        Args:
            df: The DataFrame to process.
        Returns:
            A dictionary with damages per square meter by admin_1 region.
        """
        # Only consider residential occupancy rows for damage-per-sqm
        # calculations.
        result = {}
        df = df[df["OCCUPANCY"] == "Res"]
        for admin_1 in df["NAME_1"].dropna().unique():
            result[admin_1] = {}
            for damage_type in [
                "TOTAL_REPL_COST_USD",
                "COST_STRUCTURAL_USD",
                "COST_NONSTRUCTURAL_USD",
                "COST_CONTENTS_USD",
            ]:
                df_admin_1 = df[df["NAME_1"] == admin_1]
                total_damage = df_admin_1[damage_type].sum()
                total_area = df_admin_1["TOTAL_AREA_SQM"].sum()
                # Compute damage per square metre, guarding against
                # division-by-zero when total_area is zero or missing.
                result[admin_1][damage_type + "_SQM"] = float(
                    total_damage / total_area if total_area > 0 else 0
                )
        return result

    def fetch(self, url: str, countries: list[str]) -> GlobalExposureModel:
        """Fetch and process data for specific countries.

        Args:
            url: required but not used in this implementation.
            countries: The list of countries to fetch data for.
        Returns:
            GlobalExposureModel: The adapter instance with the processed data.
        """
        # Query the repository tree for all files in the `main` branch so we
        # can locate country-specific folders without cloning the repo.
        branch = "main"
        tree_url = f"https://api.github.com/repos/gem/global_exposure_model/git/trees/{branch}?recursive=1"
        resp = requests.get(tree_url)  # , headers=HEADERS)
        resp.raise_for_status()
        tree = resp.json()["tree"]
        # Base URL for fetching raw file contents by path
        raw_base = (
            f"https://raw.githubusercontent.com/gem/global_exposure_model/{branch}/"
        )

        csv_files = []
        for country in countries:
            # Normalize the country name to ASCII for matching against
            # repository folder names (avoids accented-character mismatches).
            country = (
                unicodedata.normalize("NFKD", country)
                .encode("ascii", "ignore")
                .decode("ascii")
            )

            self._filter_folders(tree, csv_files, country)
        self._download_and_process_csv(raw_base, csv_files)

        return self

    def read(self, **kwargs: Any) -> dict[str, dict[str, float]]:
        """Read the dataset into a dictionary.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.
        Returns:
            The dataset as a dictionary.
        """
        return super().read(**kwargs)
