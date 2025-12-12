"""This script downloads all CSV files from a specific country folder in the global_exposure_model GitHub repository."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote
import requests
from .base import Adapter

GITHUB_TOKEN = "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"  # optional but recommended

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


class GlobalExposureModel(Adapter):
    """Adapter for Global Exposure Model data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GlobalExposureModel adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

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
        resp = requests.get(TREE_URL, headers=HEADERS)
        resp.raise_for_status()

        tree = resp.json()["tree"]

        # ============================
        # 2. Filter to folders matching the country
        # ============================

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

        csv_files = []

        for item in tree:
            if item["type"] == "blob":
                for folder in matching_paths:
                    if item["path"].startswith(folder + "/") and item[
                        "path"
                    ].lower().endswith(".csv"):
                        csv_files.append(item["path"])

        print(f"\nCSV files found: {len(csv_files)}")

        # ============================
        # 4. Download CSVs via raw URLs
        # ============================

        RAW_BASE = (
            f"https://raw.githubusercontent.com/gem/global_exposure_model/{BRANCH}/"
        )

        for csv in csv_files:
            url = RAW_BASE + quote(csv)
            print("Downloading:", csv)

            r = requests.get(url, headers=HEADERS)
            r.raise_for_status()

            out_path = os.path.join(OUTPUT_DIR, os.path.basename(csv))
            with open(out_path, "wb") as f:
                f.write(r.content)

        print("\nDone!")

        # Implementation would go here
        return self
