"""Shared physical constants for the land-surface energy/water coupling.

All constants are stored as `np.float32` to avoid implicit float64 promotion in
Numba-compiled kernels.

Notes:
    - Temperatures in the land-surface energy scheme are expressed in degrees Celsius
      (°C), but all enthalpy calculations are relative to a 0°C reference. Therefore,
      multiplying a heat capacity by a temperature in °C is consistent (it acts as a
      temperature difference to the 0°C reference).
"""

import numpy as np

KELVIN_OFFSET: np.float32 = np.float32(273.15)
SNOW_EMISSIVITY: np.float32 = np.float32(0.99)  # Emissivity of snow surface

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
LATENT_HEAT_FUSION_J_PER_KG: np.float32 = np.float32(334000.0)
LATENT_HEAT_VAPORIZATION_J_PER_KG: np.float32 = np.float32(2.501e6)
LATENT_HEAT_SUBLIMATION_J_PER_KG: np.float32 = np.float32(2.834e6)

# Radiation
STEFAN_BOLTZMANN_W_PER_M2_K4: np.float32 = np.float32(5.670374419e-8)

# Thermal Conductivities [W/(m K)]
# Values from Johansen (1975)
THERMAL_CONDUCTIVITY_WATER_WATT_PER_MKELVIN: np.float32 = np.float32(0.57)
THERMAL_CONDUCTIVITY_ICE_WATT_PER_MKELVIN: np.float32 = np.float32(2.2)
THERMAL_CONDUCTIVITY_QUARTZ_WATT_PER_MKELVIN: np.float32 = np.float32(7.7)
THERMAL_CONDUCTIVITY_NON_QUARTZ_WATT_PER_MKELVIN: np.float32 = np.float32(2.0)

# Soil properties
C_MINERAL_VOLUMETRIC_J_PER_M3_K: np.float32 = np.float32(2.13e6)
RHO_MINERAL_KG_PER_M3: np.float32 = np.float32(2650.0)

N_SOIL_LAYERS: int = 6
N_SNOW_LAYERS: int = 2
