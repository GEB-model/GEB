"""This script downloads all CSV files from a specific country folder in the global_exposure_model GitHub repository."""

from __future__ import annotations

import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from geb.workflows.io import write_dict

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
        matching_paths = [
            item["path"]
            for item in tree
            if item["type"] == "tree"
            and item["path"].split("/")[-1].lower() == country.lower()
        ]

        if not matching_paths:
            raise ValueError(f"No folder found for country: {country}")

        print("Found matching country folders:")
        for p in matching_paths:
            print(" -", p)

        # ============================
        # 3. Look for CSV files inside that folder
        # ============================

        for item in tree:
            if item["type"] == "blob":
                for folder in matching_paths:
                    if (
                        item["path"].startswith(folder + "/")
                        and "exposure_res" in item["path"].lower()
                    ):
                        csv_files.append(item["path"])

        print(f"\nCSV files found: {len(csv_files)}")

    def _download_and_process_csv(self, raw_base: str, csv_files: list[str]) -> None:
        """Download and process CSV files from the repository.

        Args:
            raw_base: The base URL for raw file access in the GitHub repository.
            csv_files: List of CSV file paths to download and process.
        """
        damages_per_sqm = []
        for csv in csv_files:
            url = raw_base + quote(csv)
            print("Downloading:", csv)

            r = requests.get(url)
            r.raise_for_status()

            with tempfile.TemporaryDirectory() as temp_dir_str:
                temp_dir: Path = Path(temp_dir_str)
                out_path = temp_dir / os.path.basename(csv)
                with open(out_path, "wb") as f:
                    f.write(r.content)
                # open the file with pandas process
                df = pd.read_csv(out_path)
                damages_per_sqm.append(self._process_csv(df))

        # merge the dictionaries stored in the list
        merged = {}
        for d in damages_per_sqm:
            merged.update(d)

        # and write to file
        os.makedirs(self.path.parent, exist_ok=True)
        write_dict(
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
        # filter on Res
        result = {}
        df = df[df["OCCUPANCY"] == "Res"]
        for admin_1 in df["NAME_1"].unique():
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
        country = countries[0]  # For simplicity, only handle the first country
        output_dir = f"./downloads/{country}"
        branch = "main"
        tree_url = f"https://api.github.com/repos/gem/global_exposure_model/git/trees/{branch}?recursive=1"
        os.makedirs(output_dir, exist_ok=True)

        # ============================
        # 1. Get entire repo tree in ONE request
        # ============================

        print("Downloading repository index...")
        resp = requests.get(tree_url)  # , headers=HEADERS)
        resp.raise_for_status()

        tree = resp.json()["tree"]

        # ============================
        # 4. Download CSVs via raw URLs
        # ============================

        raw_base = (
            f"https://raw.githubusercontent.com/gem/global_exposure_model/{branch}/"
        )

        csv_files = []
        for country in countries:
            # clean country name for matching
            country = (
                unicodedata.normalize("NFKD", country)
                .encode("ascii", "ignore")
                .decode("ascii")
            )

            self._filter_folders(tree, csv_files, country)
        self._download_and_process_csv(raw_base, csv_files)

        print("\nDone!")

        # Implementation would go here
        return self

    def read(self, **kwargs: Any) -> dict[str, dict[str, float]]:
        """Read the dataset into an xarray DataArray.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.
        Returns:
            The dataset as a dictionary.
        """
        return super().read(**kwargs)
