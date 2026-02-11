"""General utility methods for workflows in GEB."""

import math
from typing import Any

import numpy as np
from pyproj import CRS


def multi_level_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merges two dictionaries, updating dict1 with values from dict2.

    If a key exists in both dictionaries and both values are dictionaries, it merges them recursively.
    Otherwise, it updates dict1 with the value from dict2. This means that dict2's values will
    overwrite dict1's values for the same keys.

    Args:
        dict1 (dict): The first dictionary to merge into.
        dict2 (dict): The second dictionary whose values will be merged into dict1.

    Returns:
        dict: The updated dict1 after merging with dict2.
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            multi_level_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def multi_set(dict_obj: dict, value: Any, *attrs: str) -> None:
    """Set a value in a nested dictionary using a sequence of keys.

    Args:
        dict_obj: The dictionary to modify.
        value: The value to set.
        *attrs: A sequence of keys representing the path to the target value.

    Raises:
        KeyError: If any key in the path does not exist in the dictionary.
    """
    d = dict_obj
    for attr in attrs[:-1]:
        d = d[attr]
    if attrs[-1] not in d:
        raise KeyError(f"Key {attrs} does not exist in config file.")

    # Check if the value is a numpy scalar and convert it if necessary
    if isinstance(value, np.generic):
        value = value.item()

    d[attrs[-1]] = value


def get_utm_zone(lat: float, lon: float) -> CRS:
    """Get the UTM zone CRS for a given latitude and longitude.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        CRS: The corresponding UTM zone CRS.
    """
    # Calculate zone number
    zone_number = math.floor((lon + 180) / 6) + 1

    # Determine hemisphere
    hemisphere = "north" if lat >= 0 else "south"

    # Create CRS object (WGS84)
    # EPSG for UTM is 32600 + zone (North) or 32700 + zone (South)
    base_epsg = 32600 if hemisphere == "north" else 32700
    epsg_code = base_epsg + zone_number

    return CRS.from_epsg(epsg_code)
