"""Utilities to download and setup the damage model."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from .base import Adapter

geul_damage_model = {
    "flood": {
        "land_use": {
            "forest": {
                "curve": [
                    [0, 0],
                    [0.1, 0.1],
                    [0.25, 0.3],
                    [0.50, 0.55],
                    [1.00, 0.65],
                    [1.50, 0.7],
                    [2.00, 0.9],
                    [2.50, 0.92],
                    [3.00, 0.95],
                    [4.00, 1.00],
                    [5.00, 1.00],
                ],
                "maximum_damage": 10.79,
            },
            "agriculture": {
                "curve": [
                    [0, 0],
                    [0.1, 0.1],
                    [0.25, 0.3],
                    [0.50, 0.55],
                    [1.00, 0.65],
                    [1.50, 0.7],
                    [2.00, 0.9],
                    [2.50, 0.92],
                    [3.00, 0.95],
                    [4.00, 1.00],
                    [5.00, 1.00],
                ],
                "maximum_damage": 1.83,
            },
        },
        "residential": {
            "structure": {
                "curve": [
                    [0, 0],
                    [0.1, 0.27],
                    [0.5, 0.35],
                    [1, 0.37],
                    [1.5, 0.42],
                    [2, 0.45],
                    [2.5, 0.47],
                    [3, 0.5],
                ],
                "maximum_damage": 1806,
            },
            "content": {
                "curve": [
                    [0, 0],
                    [0.10, 0.5],
                    [0.50, 0.55],
                    [1.00, 0.6],
                    [1.50, 0.65],
                    [2.00, 0.67],
                    [2.50, 0.7],
                    [3.00, 0.72],
                ],
                "maximum_damage": 78787,
            },
        },
        "rail": {
            "main": {
                "curve": [[0, 0], [0.05, 0.02], [0.20, 0.2], [1.40, 1], [6.00, 1]],
                "maximum_damage": 7022,
            }
        },
        "road": {
            "residential": {
                "curve": [
                    [0, 0],
                    [0.50, 0.015],
                    [1.00, 0.025],
                    [1.50, 0.03],
                    [2.00, 0.035],
                    [6.00, 0.05],
                ],
                "maximum_damage": 5,
            },
            "unclassified": {
                "curve": [
                    [0, 0],
                    [0.50, 0.015],
                    [1.00, 0.025],
                    [1.50, 0.03],
                    [2.00, 0.035],
                    [6.00, 0.05],
                ],
                "maximum_damage": 5,
            },
            "tertiary": {
                "curve": [
                    [0, 0],
                    [0.50, 0.015],
                    [1.00, 0.025],
                    [1.50, 0.03],
                    [2.00, 0.035],
                    [6.00, 0.05],
                ],
                "maximum_damage": 10,
            },
            "primary": {
                "curve": [
                    [0, 0],
                    [0.50, 0.015],
                    [1.00, 0.025],
                    [1.50, 0.03],
                    [2.00, 0.035],
                    [6.00, 0.05],
                ],
                "maximum_damage": 50,
            },
            "secondary": {
                "curve": [
                    [0, 0],
                    [0.50, 0.015],
                    [1.00, 0.025],
                    [1.50, 0.03],
                    [2.00, 0.035],
                    [6.00, 0.05],
                ],
                "maximum_damage": 25,
            },
            "motorway": {
                "curve": [
                    [0, 0],
                    [0.50, 0.01],
                    [1.00, 0.03],
                    [1.50, 0.075],
                    [2.00, 0.1],
                    [6.00, 0.2],
                ],
                "maximum_damage": 4000,
            },
            "motorway_link": {
                "curve": [
                    [0, 0],
                    [0.50, 0.01],
                    [1.00, 0.03],
                    [1.50, 0.075],
                    [2.00, 0.1],
                    [6.00, 0.2],
                ],
                "maximum_damage": 4000,
            },
            "trunk": {
                "curve": [
                    [0, 0],
                    [0.50, 0.01],
                    [1.00, 0.03],
                    [1.50, 0.075],
                    [2.00, 0.1],
                    [6.00, 0.2],
                ],
                "maximum_damage": 1000,
            },
            "trunk_link": {
                "curve": [
                    [0, 0],
                    [0.50, 0.01],
                    [1.00, 0.03],
                    [1.50, 0.075],
                    [2.00, 0.1],
                    [6.00, 0.2],
                ],
                "maximum_damage": 1000,
            },
            "primary_link": {
                "curve": [
                    [0, 0],
                    [0.50, 0.015],
                    [1.00, 0.025],
                    [1.50, 0.03],
                    [2.00, 0.035],
                    [6.00, 0.05],
                ],
                "maximum_damage": 50,
            },
            "secondary_link": {
                "curve": [
                    [0, 0],
                    [0.50, 0.015],
                    [1.00, 0.025],
                    [1.50, 0.03],
                    [2.00, 0.035],
                    [6.00, 0.05],
                ],
                "maximum_damage": 25,
            },
            "tertiary_link": {
                "curve": [
                    [0, 0],
                    [0.50, 0.015],
                    [1.00, 0.025],
                    [1.50, 0.03],
                    [2.00, 0.035],
                    [6.00, 0.05],
                ],
                "maximum_damage": 25,
            },
        },
    }
}


class GeulFloodDamageModel(Adapter):
    """Adapter to fetch and clean local damage functions data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GeulFloodDamageModel adapter.

        Args:
            *args: Positional arguments passed to the base Adapter class.
            **kwargs: Keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, *args: Any, **kwargs: Any) -> GeulFloodDamageModel:
        """Empty fetch method since local damage functions are hardcoded.

        Args:
            *args: Positional arguments (not used).
            **kwargs: Keyword arguments (not used).
        Returns:
            The GeulFloodDamageModel instance.
        """
        return self

    def read(self) -> dict:
        """Read and return the local damage functions data.

        Returns:
            A dictionary containing cleaned local damage functions data.
        """
        return geul_damage_model


class GlobalFloodDamageModel(Adapter):
    """Adapter to download and setup damage functions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DamageFunctions adapter.

        Args:
            *args: Additional positional arguments passed to the base Adapter class.
            **kwargs: Additional keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    @property
    def map_damage_functions(self) -> dict[str, tuple[slice, slice]]:
        """Mapping of damage classes to their corresponding row and column slices in the raw damage functions dataframe."""
        return {
            "residential": (slice(0, 9), slice(1, 9)),
            "commercial": (slice(9, 18), slice(1, 9)),
            "industrial": (slice(18, 27), slice(1, 9)),
            "transport": (slice(27, 36), slice(1, 9)),
            "infrastructure": (slice(36, 45), slice(1, 9)),
            "agricultural": (slice(45, 54), slice(1, 9)),
        }

    def _clean_damage_functions(
        self, df: pd.DataFrame, region: str
    ) -> dict[str, pd.DataFrame]:
        """Clean the damage functions dataframe for all damage classes.

        Args:
            df: The raw dataframe containing the damage functions.
            region: The region for which to extract damage functions.

        Returns:
            A dictionary where keys are damage classes and values are cleaned dataframes
            with standardized columns and numeric values for the specified region.
        Raises:
            ValueError: If the specified region is not found in the damage functions dataframe.
        """
        damage_functions: dict[str, pd.DataFrame] = {}

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
            if region == "global":
                if df_damage_class["global"].isna().sum() > 0:
                    damage_functions[damage_class] = pd.DataFrame(
                        {
                            "depth": df_damage_class["depth"].values,
                            "damage_ratio": df_damage_class.iloc[:, 1:-1].mean(axis=1),
                        }
                    )
                else:
                    damage_functions[damage_class] = df_damage_class[
                        ["depth", "global"]
                    ]

            elif region in df_damage_class.columns:
                damage_functions[damage_class] = df_damage_class[["depth", region]]
            else:
                raise ValueError(
                    f"Region '{region}' not found in damage functions dataframe. Either use the global column or one of the following: {', '.join(df_damage_class.columns[1:])}"
                )
            damage_functions[damage_class].columns = ["depth", "damage_ratio"]
        return damage_functions

    def fetch(self, url: str) -> GlobalFloodDamageModel:
        """Fetch the damage functions file from the specified URL and save it to the local path if it doesn't already exist.

        Returns:
            The GlobalFloodDamageModel instance with the damage functions file downloaded and saved locally.
        """
        # download the file if it doesn't exist
        if not self.path.exists():
            url = url
            response = requests.get(url)
            response.raise_for_status()
            # save the file to self.path
            with open(self.path, "wb") as f:
                f.write(response.content)
        return self

    def read(self, region: str = "global") -> dict[str, pd.DataFrame]:
        """Read the damage functions from the local file, clean them, and return them as a dictionary of dataframes.

        Args:
            region: The region for which to extract damage functions. Default is 'global'.
        Returns:
            A dictionary where keys are damage classes and values are dataframes containing depth and damage fraction for the specified region.
        """
        df = pd.read_excel(
            self.path,
            sheet_name="Damage functions",
            skiprows=2,
        )
        # clean the dataframe
        damage_functions = self._clean_damage_functions(df, region=region)

        return damage_functions
