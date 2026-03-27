"""Utilities to download and setup damage functions."""

from __future__ import annotations

import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .base import Adapter


class DamageFunctions(Adapter):
    """Adapter to download and setup damage functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def map_damage_functions(self) -> dict[str, tuple]:
        return {
            "residential": [slice(0, 9), slice(1, 9)],
            "commercial": [slice(9, 18), slice(1, 9)],
            "industrial": [slice(18, 27), slice(1, 9)],
            "transport": [slice(27, 36), slice(1, 9)],
            "infrastructure": [slice(36, 45), slice(1, 9)],
            "agricultural": [slice(45, 54), slice(1, 9)],
        }

    def _clean_damage_functions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the residential damage functions dataframe.

        Args:
            df: The raw dataframe containing the damage functions.

        Returns:
            A cleaned dataframe with standardized columns and numeric values.
        """

        damage_functions = {}

        for damage_class in self.map_damage_functions:
            df_damage_class = df.iloc[
                self.map_damage_functions[damage_class][0],
                self.map_damage_functions[damage_class][1],
            ].copy()
            # Rename columns
            df_damage_class.columns = [
                "depth",
                "europe",
                "north america",
                "central&south america",
                "asia",
                "africa",
                "oceania",
                "global",
            ]

            df_damage_class = df_damage_class.apply(pd.to_numeric, errors="coerce")
            damage_functions[damage_class] = df_damage_class
        return damage_functions

    def fetch(self, *args, **kwargs):
        # download the file if it doesn't exist
        if not self.path.exists():
            url = "https://publications.jrc.ec.europa.eu/repository/bitstream/JRC105688/copy_of_global_flood_depth-damage_functions__30102017.xlsx"
            response = requests.get(url)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                # read the file into a dataframe
                df = pd.read_excel(
                    tmp_file.name,
                    sheet_name="Damage functions",
                    skiprows=2,
                )
            # clean the dataframe
            damage_functions = self._clean_damage_functions(df)

            # save the cleaned dataframe as parquet
            for damage_class, df_damage_class in damage_functions.items():
                output_path = self.path.parent / f"{damage_class}_damage_functions.csv"
                df_damage_class.to_csv(output_path, index=False)

    def read(self, *args, **kwargs):
        return super().read(*args, **kwargs)
