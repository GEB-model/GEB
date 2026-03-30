"""Utilities to download and setup the local damage model."""

from __future__ import annotations

from typing import Any

from .base import Adapter

flood = {
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
        "buildings": {
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


class LocalFloodDamageModel(Adapter):
    """Adapter to fetch and clean local damage functions data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the LocalDamageFunctions adapter.

        Args:
            *args: Positional arguments passed to the base Adapter class.
            **kwargs: Keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, *args: Any, **kwargs: Any) -> LocalFloodDamageModel:
        """Empty fetch method since local damage functions are hardcoded.

        Args:
            *args: Positional arguments (not used).
            **kwargs: Keyword arguments (not used).
        Returns:
                The LocalDamageFunctions instance.
        """
        return self

    def read(self) -> dict:
        """Read and return the local damage functions data.

        Returns:
            A dictionary containing cleaned local damage functions data.
        """
        return flood
