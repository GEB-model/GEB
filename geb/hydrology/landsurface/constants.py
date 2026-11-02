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
MAX_SNOW_DENSITY_KG_PER_M3: np.float32 = np.float32(917.0)  # kg/m3 (pure ice density)

# Minimum snow water equivalent for an active thermal snow layer in the column solver (m).
# Snow layers with SWE below this threshold are treated as trace/inactive layers
# whose temperature is synchronized with the top soil layer for numerical stability.
MIN_ACTIVE_SNOW_SWE_M: np.float64 = np.float64(0.001)

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

# Packed static soil property indices
SOIL_PROP_WATER_CONTENT_RESIDUAL_M: int = 0
SOIL_PROP_WATER_CONTENT_WILTING_POINT_M: int = 1
SOIL_PROP_WATER_CONTENT_FIELD_CAPACITY_M: int = 2
SOIL_PROP_WATER_CONTENT_SATURATED_M: int = 3
SOIL_PROP_SOLID_HEAT_CAPACITY_J_PER_M2_K: int = 4
SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_UNFROZEN_W_PER_M_K: int = 5
SOIL_PROP_THERMAL_CONDUCTIVITY_SATURATED_FROZEN_W_PER_M_K: int = 6
SOIL_PROP_THERMAL_CONDUCTIVITY_DRY_SOIL_W_PER_M_K: int = 7
SOIL_PROP_SAND_PERCENTAGE: int = 8
SOIL_PROP_SOIL_LAYER_HEIGHT_M: int = 9
SOIL_PROP_BUBBLING_PRESSURE_M_POSITIVE: int = 10
SOIL_PROP_LAMBDA_PORE_SIZE_DISTRIBUTION: int = 11
SOIL_PROP_HORIZONTAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S: int = 12
SOIL_PROP_VERTICAL_SATURATED_HYDRAULIC_CONDUCTIVITY_M_PER_S: int = 13
N_STATIC_SOIL_PROPS: int = 14

STATIC_SOIL_PROPERTY_NAMES: tuple[str, ...] = (
    "water_content_residual_m",
    "water_content_wilting_point_m",
    "water_content_field_capacity_m",
    "water_content_saturated_m",
    "solid_heat_capacity_J_per_m2_K",
    "thermal_conductivity_saturated_unfrozen_W_per_m_K",
    "thermal_conductivity_saturated_frozen_W_per_m_K",
    "thermal_conductivity_dry_soil_W_per_m_K",
    "sand_percentage",
    "soil_layer_height",
    "bubbling_pressure_m_positive",
    "lambda_pore_size_distribution",
    "horizontal_saturated_hydraulic_conductivity_m_per_s",
    "vertical_saturated_hydraulic_conductivity_m_per_s",
)

N_SNOW_LAYERS: int = 2
