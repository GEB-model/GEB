"""Interception functions."""

# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/iiasa/CWatM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------

import numpy as np
import numpy.typing as npt
from numba import njit

from .landcovers import (
    FOREST,
    GRASSLAND_LIKE,
    NON_PADDY_IRRIGATED,
    OPEN_WATER,
    PADDY_IRRIGATED,
    SEALED,
)


def get_interception_capacity(
    land_use_type: npt.NDArray[np.int32],
    interception_capacity_m_forest_HRU: npt.NDArray[np.float32],
    interception_capacity_m_grassland_HRU: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Get interception capacity based on land use type.

    Args:
        land_use_type: Array of land use types.
        interception_capacity_m_forest_HRU: Interception capacity for forest land use type.
        interception_capacity_m_grassland_HRU: Interception capacity for grassland land use type

    Returns:
        interception_capacity_m: Array of interception capacities corresponding to land use types.
    """
    interception_capacity_m = np.full(land_use_type.shape, np.nan, dtype=np.float32)
    interception_capacity_m[land_use_type == OPEN_WATER] = 0.0
    interception_capacity_m[land_use_type == SEALED] = 0.0
    interception_capacity_m[land_use_type == PADDY_IRRIGATED] = 0.001  # 1 mm
    interception_capacity_m[land_use_type == NON_PADDY_IRRIGATED] = 0.001  # 1 mm
    interception_capacity_m[land_use_type == FOREST] = (
        interception_capacity_m_forest_HRU[land_use_type == FOREST]
    )
    interception_capacity_m[land_use_type == GRASSLAND_LIKE] = (
        interception_capacity_m_grassland_HRU[land_use_type == GRASSLAND_LIKE]
    )
    assert not np.isnan(interception_capacity_m).any()
    return interception_capacity_m


@njit(cache=True, inline="always")
def interception(
    rainfall_m: np.float32,
    storage_m: np.float32,
    capacity_m: np.float32,
    potential_evaporation_m: np.float32,
    potential_transpiration_m: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Calculate interception storage, throughfall, and evaporation.

    The potential transpiration is reduced by the amount of evaporation from
    the interception storage.

    Args:
        rainfall_m: Precipitation (rain) in m.
        storage_m: Current interception storage in m.
        capacity_m: Interception capacity of vegetation in m.
        potential_evaporation_m: Potential evaporation from a wet surface in m.
        potential_transpiration_m: Potential transpiration in m.

    Returns:
        new_storage: Updated interception storage in m.
        throughfall: Water reaching the ground after interception in m.
        evaporation: Evaporation from intercepted water in m.
        potential_transpiration_m: Updated potential transpiration in m.
    """
    # Calculate throughfall
    throughfall = max(np.float32(0.0), rainfall_m + storage_m - capacity_m)

    # Update interception storage after throughfall
    new_storage = storage_m + rainfall_m - throughfall

    # Calculate evaporation from intercepted water
    evaporation = min(
        new_storage,
        potential_evaporation_m * (new_storage / capacity_m) ** np.float32(2.0 / 3.0)
        if capacity_m > np.float32(0.0)
        else np.float32(0.0),
    )

    # Update interception storage after evaporation
    new_storage -= evaporation

    potential_transpiration_m -= evaporation
    potential_transpiration_m = max(np.float32(0.0), potential_transpiration_m)

    return new_storage, throughfall, evaporation, potential_transpiration_m
