"""This script downloads all CSV files from a specific country folder in the global_exposure_model GitHub repository."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import requests

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

    def filter_folders(
        self, tree: list[dict[str, Any]], csv_files: list[str], country: str
    ) -> None:
        """Filter the repository tree for folders matching the specified country.

        Args:
            tree: The repository tree from the GitHub API.
            csv_files: List to append found CSV file paths to.
            country: The country name to filter folders by.
        """
        matching_paths = [
            item["path"]
            for item in tree
            if item["type"] == "tree"
            and item["path"].split("/")[-1].lower() == country.lower()
        ]

        if not matching_paths:
            print("No matching country folders found.")
            exit()

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

    def fetch(self, url: str, countries: list[str]) -> GlobalExposureModel:
        """Fetch and process data for specific countries.

        Args:
            countries: The list of countries to fetch data for.
        Returns:
            GlobalExposureModel: The adapter instance with the processed data.
        """
        country = countries[0]  # For simplicity, only handle the first country
        OUTPUT_DIR = f"./downloads/{country}"
        BRANCH = "main"
        TREE_URL = f"https://api.github.com/repos/gem/global_exposure_model/git/trees/{BRANCH}?recursive=1"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ============================
        # 1. Get entire repo tree in ONE request
        # ============================

        print("Downloading repository index...")
        resp = requests.get(TREE_URL)  # , headers=HEADERS)
        resp.raise_for_status()

        tree = resp.json()["tree"]

        # ============================
        # 4. Download CSVs via raw URLs
        # ============================

        RAW_BASE = (
            f"https://raw.githubusercontent.com/gem/global_exposure_model/{BRANCH}/"
        )

        csv_files = []
        for country in countries:
            self.filter_folders(tree, csv_files, country)

        for csv in csv_files:
            url = RAW_BASE + quote(csv)
            print("Downloading:", csv)

            r = requests.get(url)
            r.raise_for_status()

            out_path = os.path.join(OUTPUT_DIR, os.path.basename(csv))
            with open(out_path, "wb") as f:
                f.write(r.content)

        print("\nDone!")

        # Implementation would go here
        return self
