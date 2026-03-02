"""Shared physical constants for the land-surface energy/water coupling.

All constants are stored as `np.float32` to avoid implicit float64 promotion in
Numba-compiled kernels.

Notes:
    - Temperatures in the land-surface energy scheme are expressed in degrees Celsius
      (°C), but all enthalpy calculations are relative to a 0°C reference. Therefore,
      multiplying a heat capacity by a temperature in °C is consistent (it acts as a
      temperature difference to the 0°C reference).
"""

from __future__ import annotations

import numpy as np

# Densities
RHO_WATER_KG_PER_M3: np.float32 = np.float32(1000.0)  # kg/m3

# Specific heat capacities
SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K: np.float32 = np.float32(4186.0)
SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K: np.float32 = np.float32(
    2005.0
)  # Heat capacity around 0°C. This is a simplification, as the heat capacity of ice can vary with temperature.

# Volumetric heat capacities
VOLUMETRIC_HEAT_CAPACITY_WATER_J_PER_M3_K: np.float32 = (
    RHO_WATER_KG_PER_M3 * SPECIFIC_HEAT_CAPACITY_WATER_J_PER_KG_K
)
VOLUMETRIC_HEAT_CAPACITY_ICE_J_PER_M3_K: np.float32 = (
    RHO_WATER_KG_PER_M3 * SPECIFIC_HEAT_CAPACITY_ICE_J_PER_KG_K
)

# Latent heats
L_FUSION_J_PER_KG: np.float32 = np.float32(3.34e5)
L_VAPORIZATION_J_PER_KG: np.float32 = np.float32(2.45e6)
L_SUBLIMATION_J_PER_KG: np.float32 = L_FUSION_J_PER_KG + L_VAPORIZATION_J_PER_KG

# Radiation
STEFAN_BOLTZMANN_W_PER_M2_K4: np.float32 = np.float32(5.670374419e-8)

# Thermal Conductivities [W/(m K)]
# Values from Johansen (1975)
LAMBDA_WATER: np.float32 = np.float32(0.57)
LAMBDA_ICE: np.float32 = np.float32(2.2)
LAMBDA_QUARTZ: np.float32 = np.float32(7.7)
LAMBDA_OTHER_FINE: np.float32 = np.float32(2.0)

# Soil properties
C_MINERAL_VOLUMETRIC_J_PER_M3_K: np.float32 = np.float32(2.13e6)
RHO_MINERAL_KG_PER_M3: np.float32 = np.float32(2650.0)
