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
from numba import njit


@njit(cache=True, inline="always")
def interception(
    rainfall_m: np.float32,
    storage_m: np.float32,
    capacity_m: np.float32,
    potential_interception_evaporation_m: np.float32,
    potential_transpiration_m: np.float32,
    potential_direct_evaporation_m: np.float32,
    leaf_area_index: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32, np.float32]:
    """Calculate interception storage, throughfall, and evaporation.

    Interception capture follows the storage-based formulation from Aston (1978)
    and Merriam (1960) as documented in the LISFLOOD model (van der Knijff & de Roo, 2008,
    Section 2, Eq. 2-6 to 2-8). See: https://publications.jrc.ec.europa.eu/repository/handle/JRC44410

    Evaporation from intercepted water is calculated following CWatM:
        E_int = min(storage, E_pot_int * (storage / S_max)**(2/3))

    Args:
        rainfall_m: Precipitation (rain) depth in the time step (m).
        storage_m: Current interception storage before time step (m).
        capacity_m: Maximum canopy interception storage capacity (S_max) (m).
        potential_interception_evaporation_m: Potential evaporation from wet canopy (m).
        potential_transpiration_m: Potential transpiration (m).
        potential_direct_evaporation_m: Potential direct evaporation (soil/water) (m).
        leaf_area_index: Average Leaf Area Index (LAI) (m2 m-2).

    Returns:
        new_storage: Updated interception storage after evaporation (m).
        throughfall: Water reaching the ground after interception (m).
        evaporation: Evaporation from intercepted water (m).
        potential_transpiration_m: Updated potential transpiration (m).
        potential_direct_evaporation_m: Updated potential direct evaporation (m).
    """
    # If no canopy capacity or no vegetated surface, all rainfall and storage pass through
    if capacity_m <= np.float32(0.0) or leaf_area_index <= np.float32(0.1):
        throughfall: np.float32 = rainfall_m + storage_m
        return (
            np.float32(0.0),
            throughfall,
            np.float32(0.0),
            potential_transpiration_m,
            potential_direct_evaporation_m,
        )

    # If initial storage exceeds current capacity (after reduced LAI), drain excess to throughfall
    excess_storage: np.float32 = max(np.float32(0.0), storage_m - capacity_m)
    current_storage: np.float32 = storage_m - excess_storage

    if rainfall_m > np.float32(0.0):
        k: np.float32 = np.float32(0.046) * leaf_area_index
        int_captured: np.float32 = capacity_m * (
            np.float32(1.0) - np.exp(-k * rainfall_m / capacity_m)
        )
        # Cannot intercept more water than fell as rain
        int_captured = min(rainfall_m, int_captured)
        # Int can never exceed remaining capacity: S_max - Int_cum
        available_capacity: np.float32 = max(
            np.float32(0.0), capacity_m - current_storage
        )
        int_captured = min(int_captured, available_capacity)
    else:
        int_captured = np.float32(0.0)

    throughfall = rainfall_m - int_captured + excess_storage
    new_storage: np.float32 = current_storage + int_captured

    evaporation: np.float32 = min(
        new_storage,
        potential_interception_evaporation_m
        * (new_storage / capacity_m) ** np.float32(2.0 / 3.0),
    )
    new_storage -= evaporation

    # Reduce potential transpiration and direct evaporation, prioritizing transpiration
    transpiration_reduction: np.float32 = min(potential_transpiration_m, evaporation)
    potential_transpiration_m -= transpiration_reduction
    remaining_evap_to_reduce: np.float32 = evaporation - transpiration_reduction
    potential_direct_evaporation_m = max(
        np.float32(0.0), potential_direct_evaporation_m - remaining_evap_to_reduce
    )

    return (
        new_storage,
        throughfall,
        evaporation,
        potential_transpiration_m,
        potential_direct_evaporation_m,
    )
