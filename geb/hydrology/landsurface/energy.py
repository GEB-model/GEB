"""Soil energy flow functions."""

import numpy as np
from numba import njit

from geb.geb_types import Shape
from geb.workflows.algebra import tdma_solver


def get_heat_capacity_solid_fraction(
    bulk_density_kg_per_dm3: np.ndarray[Shape, np.dtype[np.float32]],
    layer_thickness_m: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculate the heat capacity of the solid fraction of the soil layer [J/(m2·K)].

    This calculates the total heat capacity per unit area for the solid part of the soil layer.

    Args:
        bulk_density_kg_per_dm3: Soil bulk density [kg/dm3].
        layer_thickness_m: Thickness of the soil layer [m].

    Returns:
        The areal heat capacity of the solid fraction [J/(m2·K)].
    """
    # Constants for volumetric heat capacity [J/(m3·K)]
    C_MINERAL = np.float32(2.13e6)

    # Particle density of minerals [kg/m3]
    RHO_MINERAL = np.float32(2650.0)

    # Calculate total volume fraction of solids from bulk density
    # Convert bulk density from g/cm3 to kg/m3 (factor 1000)
    phi_s = (bulk_density_kg_per_dm3 * 1000.0) / RHO_MINERAL

    # Calculate volumetric heat capacity [J/(m3·K)]
    volumetric_heat_capacity_solid = phi_s * C_MINERAL

    # Calculate areal heat capacity [J/(m2·K)]
    areal_heat_capacity = volumetric_heat_capacity_solid * layer_thickness_m

    return areal_heat_capacity.astype(np.float32)


@njit(cache=True, inline="always")
def calculate_thermal_conductivity_solid_fraction_watt_per_meter_kelvin(
    sand_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    silt_percentage: np.ndarray[Shape, np.dtype[np.float32]],
    clay_percentage: np.ndarray[Shape, np.dtype[np.float32]],
) -> np.ndarray[Shape, np.dtype[np.float32]]:
    """Calculate the thermal conductivity of the solid fraction of soil [W/(m·K)].

    Based on: https://apps.dtic.mil/sti/tr/pdf/ADA044002.pdf

    The thermal conductivity of the solid fraction ($\lambda_s$) is calculated as a
    geometric mean of the conductivity of quartz ($\lambda_q$) and other minerals ($\lambda_o$):

    $$ \lambda_s = \lambda_q^q \cdot \lambda_o^{1-q} $$

    where $q$ is the quartz content fraction.

    We assume characteristic quartz content is equal to the sand content fraction.

    Args:
        sand_percentage: Percentage of sand [0-100].
        silt_percentage: Percentage of silt [0-100].
        clay_percentage: Percentage of clay [0-100].

    Returns:
        Thermal conductivity of the solid fraction [W/(m·K)].
    """
    # Quartz thermal conductivity [W/(m K)]
    # Johansen (1975) value for quartz
    LAMBDA_QUARTZ = np.float32(7.7)
    LAMBDA_OTHER_FINE = np.float32(2.0)

    # Estimate from sand if not provided (ensuring it is bounded between 0 and 1)
    # Ensure it's bounded [0, 1]
    quartz_ratio = np.minimum(
        np.maximum(sand_percentage / np.float32(100.0), np.float32(0.0)),
        np.float32(1.0),
    )

    # Geometric mean calculation
    # lambda_s = (lambda_quartz^q) * (lambda_other^(1-q))
    lambda_s = (LAMBDA_QUARTZ**quartz_ratio) * (
        LAMBDA_OTHER_FINE ** (np.float32(1.0) - quartz_ratio)
    )

    return lambda_s


@njit(cache=True, inline="always")
def calculate_net_radiation_flux(
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    soil_temperature_C: np.float32,
    leaf_area_index: np.float32,
    air_temperature_K: np.float32,
    soil_emissivity: np.float32,
    soil_albedo: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate the net radiation energy flux and its derivative.

    Calculates absorbed incoming radiation - outgoing longwave radiation.
    Also returns the derivative of the outgoing longwave radiation with respect to temperature,
    which can be used for stability calculations in explicit schemes or damping in implicit schemes.

    Args:
        shortwave_radiation_W_per_m2: Incoming shortwave [W/m2].
        longwave_radiation_W_per_m2: Incoming longwave [W/m2].
        soil_temperature_C: Current soil temperature [C].
        leaf_area_index: Leaf Area Index [-].
        air_temperature_K: Air temperature [K], used as proxy for canopy temperature.
        soil_emissivity: Soil emissivity [-].
        soil_albedo: Soil albedo [-].

    Returns:
        Tuple of:
            - Net radiation flux [W/m2]. Positive = warming (incoming > outgoing).
            - Derivative of outgoing radiation flux [W/m2/K] (Conductance equivalent).
    """
    # Constants
    STEFAN_BOLTZMANN_CONSTANT = np.float32(5.670374419e-8)
    EXTINCTION_COEFFICIENT = np.float32(0.5)  # Beer's law extinction coefficient

    # Calculate Fluxes
    temperature_K = soil_temperature_C + np.float32(273.15)

    # Beer's law attenuation factor
    attenuation_factor = np.exp(-EXTINCTION_COEFFICIENT * leaf_area_index)

    absorbed_shortwave_W = (
        (np.float32(1.0) - soil_albedo)
        * shortwave_radiation_W_per_m2
        * attenuation_factor
    )

    # Longwave radiation reaching the soil
    # Atmospheric longwave transmitted through canopy
    transmitted_longwave_W = longwave_radiation_W_per_m2 * attenuation_factor

    # Emissions from canopy
    # We assume canopy temperature ~= air temperature
    canopy_longwave_W = (
        STEFAN_BOLTZMANN_CONSTANT
        * (air_temperature_K**4)
        * (np.float32(1.0) - attenuation_factor)
    )

    incoming_longwave_at_soil_surface_W = transmitted_longwave_W + canopy_longwave_W

    absorbed_longwave_W = soil_emissivity * incoming_longwave_at_soil_surface_W

    incoming_W = absorbed_shortwave_W + absorbed_longwave_W
    outgoing_W = soil_emissivity * STEFAN_BOLTZMANN_CONSTANT * (temperature_K**4)

    net_flux_W = incoming_W - outgoing_W

    # Calculate Derivative of Outgoing Radiation with respect to T:
    # d(sigma * eps * T^4)/dT = 4 * sigma * eps * T^3
    conductance_W_per_m2_K = (
        np.float32(4.0)
        * soil_emissivity
        * STEFAN_BOLTZMANN_CONSTANT
        * (temperature_K**3)
    )

    return net_flux_W, conductance_W_per_m2_K


@njit(cache=True, inline="always")
def calculate_sensible_heat_flux(
    soil_temperature_C: np.float32,
    air_temperature_K: np.float32,
    wind_speed_10m_m_per_s: np.float32,
    surface_pressure_pa: np.float32,
) -> tuple[np.float32, np.float32]:
    """Calculate the sensible heat flux and aerodynamic conductance.

    Args:
        soil_temperature_C: Soil temperature in Celsius [C].
        air_temperature_K: Air temperature at 2m height [K].
        wind_speed_10m_m_per_s: Wind speed at 10m height [m/s].
        surface_pressure_pa: Surface air pressure [Pa].

    Returns:
        Tuple of:
            - Sensible heat flux [W/m2]. Positive = warming (Heat flow from Air to Soil).
            - Aerodynamic conductance [W/m2/K].
    """
    # Physics Constants
    SPECIFIC_HEAT_AIR_J_KG_K: np.float32 = np.float32(1005.0)
    GAS_CONSTANT_AIR_J_KG_K: np.float32 = np.float32(287.058)
    VON_KARMAN_CONSTANT: np.float32 = np.float32(0.41)

    # Assumptions for Aerodynamic Resistance over bare soil
    WIND_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(10.0)
    TEMP_MEASUREMENT_HEIGHT_M: np.float32 = np.float32(2.0)
    ROUGHNESS_LENGTH_M: np.float32 = np.float32(0.001)

    # Calculate Air Density [kg/m3]
    # Ideal Gas Law: rho = P / (R * T)
    # Using air temperature for density calculation
    air_density_kg_per_m3: np.float32 = surface_pressure_pa / (
        GAS_CONSTANT_AIR_J_KG_K * air_temperature_K
    )

    # Calculate Aerodynamic Resistance (ra) [s/m]
    # For neutral conditions: ra = (ln((zm-d)/z0m) * ln((zh-d)/z0h)) / (k^2 * u)
    # Assuming d=0, z0m=z0h=z0
    # where zm = wind measurement height, zh = temp measurement height, z0 = roughness length, u = wind speed

    # Ensure minimum wind speed to avoid division by zero
    wind_speed_safe = np.maximum(wind_speed_10m_m_per_s, np.float32(0.1))

    log_wind_height_over_roughness = np.log(
        WIND_MEASUREMENT_HEIGHT_M / ROUGHNESS_LENGTH_M
    )
    log_temp_height_over_roughness = np.log(
        TEMP_MEASUREMENT_HEIGHT_M / ROUGHNESS_LENGTH_M
    )

    aerodynamic_resistance_s_per_m = (
        log_wind_height_over_roughness * log_temp_height_over_roughness
    ) / (VON_KARMAN_CONSTANT**2 * wind_speed_safe)

    # Calculate Conductance [W/m2/K]
    # Conductance = rho * Cp / ra
    conductance_W_per_m2_K = (
        air_density_kg_per_m3 * SPECIFIC_HEAT_AIR_J_KG_K
    ) / aerodynamic_resistance_s_per_m

    # Calculate Explicit Sensible Heat Flux [W/m2]
    # H = Conductance * (Ta - Ts)
    air_temperature_C = air_temperature_K - np.float32(273.15)
    temperature_difference_C = air_temperature_C - soil_temperature_C

    sensible_heat_flux_W_per_m2 = conductance_W_per_m2_K * temperature_difference_C

    return sensible_heat_flux_W_per_m2, conductance_W_per_m2_K


@njit(cache=True, inline="always")
def solve_energy_balance_implicit_iterative(
    soil_temperature_C: np.float32,
    solid_heat_capacity_J_per_m2_K: np.float32,
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    air_temperature_K: np.float32,
    wind_speed_10m_m_per_s: np.float32,
    surface_pressure_pa: np.float32,
    timestep_seconds: np.float32,
    soil_emissivity: np.float32,
    soil_albedo: np.float32,
    leaf_area_index: np.float32,
) -> np.float32:
    """Update soil temperature solving energy balance with an iterative implicit scheme.

    Solves the non-linear energy balance equation using Newton-Raphson iteration.
    Equation: C * (T_new - T_old) / dt = Q_rad(T_new) + Q_sens(T_new)

    This combines the radiation and sensible heat balances into a single
    implicit solution, which is robust and stable for large time steps.

    Args:
        soil_temperature_C: Initial soil temperature [C].
        solid_heat_capacity_J_per_m2_K: Heat capacity of the soil layer [J/m2/K].
        shortwave_radiation_W_per_m2: Incoming shortwave radiation [W/m2].
        longwave_radiation_W_per_m2: Incoming longwave radiation [W/m2].
        air_temperature_K: Air temperature [K].
        wind_speed_10m_m_per_s: Wind speed [m/s].
        surface_pressure_pa: Surface pressure [Pa].
        timestep_seconds: Total time to simulate [s] (e.g. 3600.0).
        soil_emissivity: Soil emissivity [-].
        soil_albedo: Soil albedo [-].
        leaf_area_index: Leaf Area Index [-].

    Returns:
        Updated soil temperature [C].
    """
    T_old = soil_temperature_C
    T_curr = soil_temperature_C

    # Newton-Raphson Configuration
    MAX_ITERATIONS = 10
    TOLERANCE_C = np.float32(0.01)

    for _ in range(MAX_ITERATIONS):
        # Calculate Fluxes and derivative/conductances at current estimate
        net_radiation_flux_W_per_m2, radiation_conductance_W_per_m2_K = (
            calculate_net_radiation_flux(
                shortwave_radiation_W_per_m2=shortwave_radiation_W_per_m2,
                longwave_radiation_W_per_m2=longwave_radiation_W_per_m2,
                soil_temperature_C=T_curr,
                leaf_area_index=leaf_area_index,
                air_temperature_K=air_temperature_K,
                soil_emissivity=soil_emissivity,
                soil_albedo=soil_albedo,
            )
        )

        sensible_heat_flux_W_per_m2, sensible_heat_conductance_W_per_m2_K = (
            calculate_sensible_heat_flux(
                soil_temperature_C=T_curr,
                air_temperature_K=air_temperature_K,
                wind_speed_10m_m_per_s=wind_speed_10m_m_per_s,
                surface_pressure_pa=surface_pressure_pa,
            )
        )

        # Function f(T_new) = C/dt * (T_new - T_old) - (Q_rad + Q_sens)
        # We want f(T_new) = 0

        storage_term_W_per_m2 = (solid_heat_capacity_J_per_m2_K / timestep_seconds) * (
            T_curr - T_old
        )
        total_flux_W_per_m2 = net_radiation_flux_W_per_m2 + sensible_heat_flux_W_per_m2

        f_val = storage_term_W_per_m2 - total_flux_W_per_m2

        # Derivative f'(T_new) = C/dt - (Q'_rad + Q'_sens)
        # Note: Q' terms are negative conductances (flux decreases as T increases)
        # Q'_rad = -radiation_conductance
        # Q'_sens = -sensible_heat_conductance
        # So f'(T_new) = C/dt + radiation_conductance + sensible_heat_conductance

        f_prime = (
            (solid_heat_capacity_J_per_m2_K / timestep_seconds)
            + radiation_conductance_W_per_m2_K
            + sensible_heat_conductance_W_per_m2_K
        )

        # Newton Step: T_next = T_curr - f(T_curr) / f'(T_curr)
        delta_T = f_val / f_prime

        T_curr -= delta_T

        if abs(delta_T) < TOLERANCE_C:
            break

    return T_curr


@njit(cache=True, inline="always")
def apply_evaporative_cooling(
    soil_temperature_top_layer_C: np.float32,
    evaporation_m: np.float32,
    solid_heat_capacity_J_per_m2_K: np.float32,
) -> np.float32:
    """Apply evaporative cooling to the top soil layer.

    Args:
        soil_temperature_top_layer_C: Temperature of the top layer (C).
        evaporation_m: Amount of evaporation (m).
        solid_heat_capacity_J_per_m2_K: Heat capacity of the top layer (J/m2/K).

    Returns:
        New temperature of top soil layer (C).
    """
    latent_heat_vaporization_J_per_kg = np.float32(2.45e6)
    density_water_kg_per_m3 = np.float32(1000.0)

    energy_loss_J_per_m2 = (
        evaporation_m * density_water_kg_per_m3 * latent_heat_vaporization_J_per_kg
    )

    cooling_K = energy_loss_J_per_m2 / solid_heat_capacity_J_per_m2_K

    return soil_temperature_top_layer_C - cooling_K


@njit(cache=True, inline="always")
def apply_advective_heat_transport(
    soil_temperature_top_layer_C: np.float32,
    infiltration_amount_m: np.float32,
    rain_temperature_C: np.float32,
    solid_heat_capacity_J_per_m2_K: np.float32,
) -> np.float32:
    """Apply advective heat transport from infiltrating rain to the top soil layer.

    Heat is added/removed based on the temperature difference between rain (air temp) and soil.
    Q = mass * cp * (T_rain - T_soil)
    dT = Q / C_soil

    Args:
        soil_temperature_top_layer_C: Temperature of the top layer (C).
        infiltration_amount_m: Amount of infiltration (m).
        rain_temperature_C: Temperature of the rain (C), assumed equal to air temperature.
        solid_heat_capacity_J_per_m2_K: Heat capacity of the top layer (J/m2/K).

    Returns:
        New temperature of top soil layer (C).
    """
    specific_heat_water_J_per_kg_K = np.float32(4186.0)
    density_water_kg_per_m3 = np.float32(1000.0)

    # Calculate energy flux into the soil layer: Q_advection (J/m2)
    # The water enters at T_rain and thermalizes to T_soil.
    # The energy change of the soil matrix is equal to the energy released by the water cooling down (or heating up).
    # dU_soil = m_water * cp_water * (T_rain - T_soil)
    # This energy dU_soil raises the temperature of the soil by dT = dU_soil / C_soil

    energy_added_J_per_m2 = (
        infiltration_amount_m
        * density_water_kg_per_m3
        * specific_heat_water_J_per_kg_K
        * (rain_temperature_C - soil_temperature_top_layer_C)
    )

    temperature_change_K = energy_added_J_per_m2 / solid_heat_capacity_J_per_m2_K

    return soil_temperature_top_layer_C + temperature_change_K


@njit(cache=True)
def solve_soil_temperature_column(
    soil_temperatures_C: np.ndarray,
    layer_thicknesses_m: np.ndarray,
    solid_heat_capacities_J_per_m2_K: np.ndarray,
    thermal_conductivities_W_per_m_K: np.ndarray,
    shortwave_radiation_W_per_m2: np.float32,
    longwave_radiation_W_per_m2: np.float32,
    air_temperature_K: np.float32,
    wind_speed_10m_m_per_s: np.float32,
    surface_pressure_pa: np.float32,
    timestep_seconds: np.float32,
    deep_soil_temperature_C: np.float32,
    soil_emissivity: np.float32,
    soil_albedo: np.float32,
    leaf_area_index: np.float32,
    snow_water_equivalent_m: np.float32 = np.float32(0.0),
) -> tuple[np.ndarray, np.float32]:
    """Solve the soil temperature profile using a fully implicit method with non-linear surface boundary.

    This function updates the soil temperature profile by solving the 1D heat diffusion equation.
    The surface boundary condition is a non-linear energy balance (radiation + sensible heat),
    which is handled via Newton-Raphson iteration coupled with a Tridiagonal Matrix Algorithm (TDMA) solver.
    The bottom boundary condition is a Dirichlet boundary with a provided deep soil temperature.

    Args:
        soil_temperatures_C: Current soil temperatures for each layer [C].
        layer_thicknesses_m: Thickness of each soil layer [m].
        solid_heat_capacities_J_per_m2_K: Areal heat capacity of the solid fraction for each layer [J/m2/K].
        thermal_conductivities_W_per_m_K: Thermal conductivity of the solid fraction for each layer [W/m-K].
        shortwave_radiation_W_per_m2: Incoming shortwave radiation [W/m2].
        longwave_radiation_W_per_m2: Incoming longwave radiation [W/m2].
        air_temperature_K: Air temperature [K].
        wind_speed_10m_m_per_s: Wind speed at 10m [m/s].
        surface_pressure_pa: Surface pressure [Pa].
        timestep_seconds: Time step length [s].
        deep_soil_temperature_C: Constant temperature at the bottom boundary [C].
        soil_emissivity: Soil emissivity [-].
        soil_albedo: Soil albedo [-].
        leaf_area_index: Leaf Area Index [-].
        snow_water_equivalent_m: Snow water equivalent [m]. If provided and > 0.001 m, the boundary condition is treated as adiabatic.

    Returns:
        Tuple of:
            - Updated soil temperatures for each layer [C].
            - Soil heat flux [W/m2]. Positive = flux into the soil.
    """
    n_soil_layers = len(soil_temperatures_C)
    temperatures_at_start_of_timestep_C = soil_temperatures_C.copy()
    # Initial guess for the implicit solution at current iteration k=0
    temperatures_current_iteration_C = soil_temperatures_C.copy()

    # Newton-Raphson iteration parameters
    MAX_ITERATIONS = 10
    TOLERANCE_C = np.float32(0.01)

    # Conductance K is the thermal coupling between centers of adjacent layers [W/m2/K].
    # It is derived from the thermal resistance of the two half-layers.
    thermal_conductances_between_layer_centers_W_per_m2_K = np.zeros(
        n_soil_layers - 1, dtype=soil_temperatures_C.dtype
    )
    for i in range(n_soil_layers - 1):
        resistance_upper_half_layer = (
            soil_temperatures_C.dtype.type(0.5) * layer_thicknesses_m[i]
        ) / thermal_conductivities_W_per_m_K[i]
        resistance_lower_half_layer = (
            soil_temperatures_C.dtype.type(0.5) * layer_thicknesses_m[i + 1]
        ) / thermal_conductivities_W_per_m_K[i + 1]

        thermal_conductances_between_layer_centers_W_per_m2_K[i] = (
            soil_temperatures_C.dtype.type(1.0)
            / (resistance_upper_half_layer + resistance_lower_half_layer)
        )

    # Matrix arrays for the Tridiagonal Matrix Algorithm (TDMA)
    # The system is structured as: a_i * T_{i-1} + b_i * T_i + c_i * T_{i+1} = d_i
    lower_diagonal_a = np.zeros(n_soil_layers, dtype=soil_temperatures_C.dtype)
    main_diagonal_b = np.zeros(n_soil_layers, dtype=soil_temperatures_C.dtype)
    upper_diagonal_c = np.zeros(n_soil_layers, dtype=soil_temperatures_C.dtype)
    rhs_vector_d = np.zeros(n_soil_layers, dtype=soil_temperatures_C.dtype)

    # Store fluxes to return G at the end
    final_net_radiation_flux_W_per_m2 = np.float32(0.0)
    final_sensible_heat_flux_W_per_m2 = np.float32(0.0)

    for _ in range(MAX_ITERATIONS):
        # Linearize Surface Boundary Conditions
        # Since radiation and sensible heat depend non-linearly on surface temperature (T0),
        # we linearize around the current guess: Flux(T0_new) ≈ Flux(T0) + dFlux/dT * (T0_new - T0)
        surface_temperature_guess_C = temperatures_current_iteration_C[0]

        if snow_water_equivalent_m > np.float32(0.001):
            net_radiation_flux_W_per_m2 = np.float32(0.0)
            derivative_net_radiation_W_per_m2_K = np.float32(0.0)
            sensible_heat_flux_W_per_m2 = np.float32(0.0)
            derivative_sensible_heat_W_per_m2_K = np.float32(0.0)
        else:
            net_radiation_flux_W_per_m2, derivative_net_radiation_W_per_m2_K = (
                calculate_net_radiation_flux(
                    shortwave_radiation_W_per_m2=shortwave_radiation_W_per_m2,
                    longwave_radiation_W_per_m2=longwave_radiation_W_per_m2,
                    soil_temperature_C=surface_temperature_guess_C,
                    leaf_area_index=leaf_area_index,
                    air_temperature_K=air_temperature_K,
                    soil_emissivity=soil_emissivity,
                    soil_albedo=soil_albedo,
                )
            )
            sensible_heat_flux_W_per_m2, derivative_sensible_heat_W_per_m2_K = (
                calculate_sensible_heat_flux(
                    soil_temperature_C=surface_temperature_guess_C,
                    air_temperature_K=air_temperature_K,
                    wind_speed_10m_m_per_s=wind_speed_10m_m_per_s,
                    surface_pressure_pa=surface_pressure_pa,
                )
            )

        # Store for final G calculation
        final_net_radiation_flux_W_per_m2 = net_radiation_flux_W_per_m2
        final_sensible_heat_flux_W_per_m2 = sensible_heat_flux_W_per_m2

        # Combine fluxes and conductances for the linearized boundary condition.
        # We use a first-order Taylor expansion: Flux(T_new) ≈ Flux(T_guess) + dFlux/dT * (T_new - T_guess)
        # Rearranged as: Flux(T_new) ≈ [Flux(T_guess) - dFlux/dT * T_guess] + dFlux/dT * T_new
        # Let G = -dFlux/dT (surface conductance).
        # Flux(T_new) ≈ [Flux(T_guess) + G * T_guess] - G * T_new
        # The term in brackets (flux_star) is independent of the unknown T_new in the current linear solve.
        flux_star_W_per_m2 = (
            net_radiation_flux_W_per_m2
            + sensible_heat_flux_W_per_m2
            + (
                derivative_net_radiation_W_per_m2_K
                + derivative_sensible_heat_W_per_m2_K
            )
            * surface_temperature_guess_C
        )
        surface_thermal_conductance_W_per_m2_K = (
            derivative_net_radiation_W_per_m2_K + derivative_sensible_heat_W_per_m2_K
        )

        # Build the system for the current iteration

        # Top soil layer (i = 0) with surface boundary condition
        heat_storage_capacity_normalized_0 = (
            solid_heat_capacities_J_per_m2_K[0] / timestep_seconds
        )
        conductance_to_layer_below = (
            thermal_conductances_between_layer_centers_W_per_m2_K[0]
        )

        # The lower diagonal represents the coupling to the node i-1.
        # Since node 0 is the top layer, there is no node -1, so a[0] is not used.
        lower_diagonal_a[0] = np.float32(
            0.0
        )  # No coupling to layer above because it's the surface
        main_diagonal_b[0] = (
            heat_storage_capacity_normalized_0
            + surface_thermal_conductance_W_per_m2_K
            + conductance_to_layer_below
        )  # the self-influence includes the surface conductance and coupling to layer below
        upper_diagonal_c[
            0
        ] = -conductance_to_layer_below  # The coupling to the layer below
        rhs_vector_d[0] = (
            heat_storage_capacity_normalized_0 * temperatures_at_start_of_timestep_C[0]
            + flux_star_W_per_m2
        )  # this "unchangeable" part of the flux is treated as a source term in the linear system
        # unchangeable only refers to the current iteration

        # intermediate soil layers
        for i in range(1, n_soil_layers - 1):
            heat_storage_capacity_normalized = (
                solid_heat_capacities_J_per_m2_K[i] / timestep_seconds
            )
            conductance_to_layer_above = (
                thermal_conductances_between_layer_centers_W_per_m2_K[i - 1]
            )
            conductance_to_layer_below = (
                thermal_conductances_between_layer_centers_W_per_m2_K[i]
            )

            lower_diagonal_a[i] = -conductance_to_layer_above
            main_diagonal_b[i] = (
                heat_storage_capacity_normalized
                + conductance_to_layer_above
                + conductance_to_layer_below
            )
            upper_diagonal_c[i] = -conductance_to_layer_below
            rhs_vector_d[i] = (
                heat_storage_capacity_normalized
                * temperatures_at_start_of_timestep_C[i]
            )

        # Bottom soil layer
        # We apply a Dirichlet boundary condition at the bottom using the provided deep soil temperature.
        last_idx = n_soil_layers - 1
        heat_storage_capacity_normalized_last = (
            solid_heat_capacities_J_per_m2_K[last_idx] / timestep_seconds
        )
        conductance_to_layer_above = (
            thermal_conductances_between_layer_centers_W_per_m2_K[last_idx - 1]
        )

        # Conductance from the center of the last layer to the boundary (distance = 0.5 * thickness)
        conductance_to_deep_soil_boundary_W_per_m2_K = thermal_conductivities_W_per_m_K[
            last_idx
        ] / (np.float32(0.5) * layer_thicknesses_m[last_idx])

        lower_diagonal_a[last_idx] = -conductance_to_layer_above
        main_diagonal_b[last_idx] = (
            heat_storage_capacity_normalized_last
            + conductance_to_layer_above
            + conductance_to_deep_soil_boundary_W_per_m2_K
        )
        # The upper diagonal represents the coupling to the node i+1.
        # Since last_idx is the bottom layer, there is no node n, so c[last_idx] is not used.
        upper_diagonal_c[last_idx] = np.float32(0.0)
        rhs_vector_d[last_idx] = (
            heat_storage_capacity_normalized_last
            * temperatures_at_start_of_timestep_C[last_idx]
            + conductance_to_deep_soil_boundary_W_per_m2_K * deep_soil_temperature_C
        )

        # Solve the Tridiagonal System
        temperatures_new_iteration_C = tdma_solver(
            lower_diagonal_a, main_diagonal_b, upper_diagonal_c, rhs_vector_d
        )

        # Check for Convergence
        max_temperature_correction_C = np.max(
            np.abs(temperatures_new_iteration_C - temperatures_current_iteration_C)
        )
        temperatures_current_iteration_C = temperatures_new_iteration_C

        # If the maximum correction is below the tolerance, we consider the solution converged.
        if max_temperature_correction_C < TOLERANCE_C:
            break

    soil_heat_flux_W_per_m2 = (
        final_net_radiation_flux_W_per_m2 + final_sensible_heat_flux_W_per_m2
    )

    return temperatures_current_iteration_C, soil_heat_flux_W_per_m2
