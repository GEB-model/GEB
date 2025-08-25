import numpy as np
import numpy.typing as npt
from numba import njit

from geb.module import Module

from .landcover import FOREST


@njit(cache=True, inline="always")
def get_saturated_vapour_pressure(
    temperature_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the saturated vapour pressure based on minimum and maximum temperature.

    Args:
        temperature_C: Temperature in Celsius.

    Returns:
        Saturated vapour pressure in kPa.
    """
    return np.float32(0.6108) * np.exp(
        (np.float32(17.27) * temperature_C) / (temperature_C + np.float32(237.3))
    )


@njit(cache=True, inline="always")
def get_actual_vapour_pressure(
    saturated_vapour_pressure_min: npt.NDArray[np.float32],
    saturated_vapour_pressure_max: npt.NDArray[np.float32],
    hurs: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the actual vapour pressure based on relative humidity and saturated vapour pressures.

    Args:
        saturated_vapour_pressure_min: Minimum saturated vapour pressure.
        saturated_vapour_pressure_max: Maximum saturated vapour pressure.
        hurs: Relative humidity in percentage.

    Returns:
        Actual vapour pressure.
    """
    return (
        hurs
        / np.float32(100.0)
        * (saturated_vapour_pressure_min + saturated_vapour_pressure_max)
        / np.float32(2.0)
    )


@njit(cache=True, inline="always")
def get_vapour_pressure_deficit(
    saturated_vapour_pressure_min: npt.NDArray[np.float32],
    saturated_vapour_pressure_max: npt.NDArray[np.float32],
    actual_vapour_pressure: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the vapour pressure deficit.

    Args:
        saturated_vapour_pressure_min: Minimum saturated vapour pressure.
        saturated_vapour_pressure_max: Maximum saturated vapour pressure.
        actual_vapour_pressure: Actual vapour pressure.

    Returns:
        Vapour pressure deficit.
    """
    return np.maximum(
        (saturated_vapour_pressure_min + saturated_vapour_pressure_max)
        / np.float32(2.0)
        - actual_vapour_pressure,
        np.float32(0.0),
    )


@njit(cache=True, inline="always")
def get_psychrometric_constant(
    ps_pascal: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the psychrometric constant.

    Args:
        ps_pascal: Surface pressure in Pascals.

    Returns:
        Psychrometric constant in kPa/°C.
    """
    return np.float32(0.665e-3) * np.float32(0.001) * ps_pascal  # Convert Pa to kPa


@njit(cache=True, inline="always")
def get_upwelling_long_wave_radiation(
    tasmin_C: npt.NDArray[np.float32], tasmax_C: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Calculate the upwelling long wave radiation based on minimum and maximum temperature.

    Args:
        tasmin_C: Minimum temperature in Celsius.
        tasmax_C: Maximum temperature in Celsius.

    Returns:
        Upwelling long wave radiation in MJ/m^2/day.
    """
    return (
        np.float32(4.903e-9)  # Stefan-Boltzmann constant [MJ m-2 K-4 day-1]
        * (
            ((tasmin_C + np.float32(273.16)) ** 4)
            + ((tasmax_C + np.float32(273.16)) ** 4)
        )
        / np.float32(2)
    )


@njit(cache=True, inline="always")
def W_m2_to_MJ_m2_day(
    solar_radiation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Convert solar radiation from W/m^2 to MJ/m^2/day.

    Args:
        solar_radiation: Solar radiation in W/m^2.

    Returns:
        Solar radiation in MJ/m^2/day.
    """
    return solar_radiation * (np.float32(86400) * np.float32(1e-6))


@njit(cache=True, inline="always")
def get_net_solar_radiation(
    solar_radiation: npt.NDArray[np.float32], albedo: np.float32
) -> npt.NDArray[np.float32]:
    """Calculate net solar radiation based on incoming solar radiation and albedo.

    Args:
        solar_radiation: Incoming solar radiation.
        albedo: Albedo of the surface (fraction of reflected solar radiation).

    Returns:
        Net solar radiation.
    """
    return (np.float32(1) - albedo) * solar_radiation


@njit(cache=True, inline="always")
def get_slope_of_saturation_vapour_pressure_curve(
    temperature_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the slope of the saturation vapour pressure curve.

    Args:
        temperature_C: Temperature in Celsius.

    Returns:
        Slope of the saturation vapour pressure curve in kPa/°C.
    """
    return (
        np.float32(4098.0)
        * get_saturated_vapour_pressure(temperature_C=temperature_C)
        / ((temperature_C + np.float32(237.3)) ** 2)
    )


@njit(cache=True)
def get_latent_heat_of_vaporization(
    temperature_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the latent heat of vaporization based on temperature.

    Args:
        temperature_C: Temperature in Celsius.

    Returns:
        Latent heat of vaporization in MJ/kg.
    """
    return np.float32(2.501) - np.float32(0.002361) * temperature_C


@njit(cache=True, inline="always")
def adjust_wind_speed(wind_speed: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Adjust wind speed to surface level.

    Args:
        wind_speed: Wind speed at 10 m height in m/s.

    Returns:
        Adjusted wind speed at 2 m height in m/s.
    """
    return wind_speed * np.float32(0.748)


@njit(cache=True, inline="always")
def get_reference_evapotranspiration(
    net_radiation_land: npt.NDArray[np.float32],
    net_radiation_water: npt.NDArray[np.float32],
    slope_of_saturated_vapour_pressure_curve: npt.NDArray[np.float32],
    psychrometric_constant: npt.NDArray[np.float32],
    wind_2m: npt.NDArray[np.float32],
    latent_heat_of_vaporarization: npt.NDArray[np.float32],
    termperature_C: npt.NDArray[np.float32],
    vapour_pressure_deficit: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Combine all terms of the Penman-Monteith equation to calculate reference evapotranspiration.

    Args:
        net_radiation_land: Net radiation for land in MJ/m^2/day.
        net_radiation_water: Net radiation for water in MJ/m^2/day.
        slope_of_saturated_vapour_pressure_curve: Slope of the saturation vapour pressure curve in kPa/°C.
        psychrometric_constant: Psychrometric constant in kPa/°C.
        wind_2m: Wind speed at 2 m height in m/s.
        latent_heat_of_vaporarization: Latent heat of vaporization in MJ/kg.
        termperature_C: Temperature in Celsius.
        vapour_pressure_deficit: Vapour pressure deficit in kPa.

    Returns:
        reference_evapotranspiration_land: Reference evapotranspiration for land in mm/day.
        reference_evapotranspiration_water: Reference evapotranspiration for water in mm/day.
    """
    denominator = slope_of_saturated_vapour_pressure_curve + psychrometric_constant * (
        np.float32(1) + np.float32(0.34) * wind_2m
    )

    common_energy_factor = (
        slope_of_saturated_vapour_pressure_curve
        / latent_heat_of_vaporarization
        / denominator
    )

    energy_term_land = net_radiation_land * common_energy_factor
    energy_term_water = net_radiation_water * common_energy_factor

    aerodynamic_term = (
        psychrometric_constant
        * np.float32(900)
        / (np.float32(273.16) + termperature_C)  # Assuming a temperature of 25 C
        * wind_2m
        * vapour_pressure_deficit
        / denominator
    )
    return energy_term_land + aerodynamic_term, energy_term_water + aerodynamic_term


@njit(cache=True, parallel=True)
def PET(
    tas: npt.NDArray[np.float32],
    tasmin: npt.NDArray[np.float32],
    tasmax: npt.NDArray[np.float32],
    hurs: npt.NDArray[np.float32],
    ps_pascal: npt.NDArray[np.float32],
    rlds: npt.NDArray[np.float32],
    rsds: npt.NDArray[np.float32],
    sfcWind: npt.NDArray[np.float32],
    albedo_canopy=np.float32(0.13),
    albedo_water=np.float32(0.05),
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Calculate potential evapotranspiration based on Penman-Monteith equation.

    Penman-Montheith equation:

        ET0 = (1 / λ * (Rn - G) + γ * (900 / (T + 273)) * u2 * (es - ea)) / (Δ + γ * (1 + 0.34 * u2))

    where:

        ET0   reference evapotranspiration [mm day-1],
        Rn    net radiation at the crop surface [MJ m-2 day-1],
        G     soil heat flux density [MJ m-2 day-1],
        T     mean daily air temperature at 2 m height [°C],
        es    saturation vapour pressure [kPa],
        ea    actual vapour pressure [kPa],
        es - ea saturation vapour pressure deficit [kPa],
        Δ     slope of the vapour pressure curve [kPa °C-1],
        γ     psychrometric constant [kPa °C-1],
        u2    wind speed at 2 m height [m s-1]
        λ latent heat of vaporization [MJ kg-1]

    Note:
        Note that in the daily time step, the soil heat flux density (G) is often assumed to be negligible (G = 0).
        When making this hourly, the soil heat flux should be included.

    Args:
        tas: average air temperature in Kelvin.
        tasmin: minimum air temperature in Kelvin.
        tasmax: maximum air temperature in Kelvin.
        hurs: relative humidity in percentage.
        ps_pascal: surface pressure in Pascals.
        rlds: long wave downward surface radiation fluxes in W/m^2.
        rsds: short wave downward surface radiation fluxes in W/m^2.
        sfcWind: wind speed at 10 m height in m/s.
        albedo_canopy: albedo of vegetation canopy (default = 0.13).
        albedo_water: albedo of water surface (default = 0.05).

    Returns:
        reference_evapotranspiration_land: reference evapotranspiration for land in m/day.
        reference_evapotranspiration_water: reference evapotranspiration for water in m/day.
        net_radiation_land: net radiation for land in MJ/m^2/day.
    """
    tas_C = tas - np.float32(273.15)
    tasmin_C = tasmin - np.float32(273.15)
    tasmax_C = tasmax - np.float32(273.15)

    saturated_vapour_pressure_min = get_saturated_vapour_pressure(
        temperature_C=tasmin_C,
    )
    saturated_vapour_pressure_max = get_saturated_vapour_pressure(
        temperature_C=tasmax_C,
    )
    actual_vapour_pressure = get_actual_vapour_pressure(
        saturated_vapour_pressure_min=saturated_vapour_pressure_min,
        saturated_vapour_pressure_max=saturated_vapour_pressure_max,
        hurs=hurs,
    )

    vapour_pressure_deficit = get_vapour_pressure_deficit(
        saturated_vapour_pressure_min=saturated_vapour_pressure_min,
        saturated_vapour_pressure_max=saturated_vapour_pressure_max,
        actual_vapour_pressure=actual_vapour_pressure,
    )

    psychrometric_constant = get_psychrometric_constant(ps_pascal=ps_pascal)

    rlus_MJ_m2_day = get_upwelling_long_wave_radiation(tasmin_C, tasmax_C)
    rlds_MJ_m2_day = W_m2_to_MJ_m2_day(rlds)
    net_longwave_radation_MJ_m2_day = rlus_MJ_m2_day - rlds_MJ_m2_day

    solar_radiation_MJ_m2_day = W_m2_to_MJ_m2_day(rsds)

    net_solar_radiation_MJ_m2_day_land = get_net_solar_radiation(
        solar_radiation_MJ_m2_day, albedo_canopy
    )

    net_radiation_land = np.maximum(
        net_solar_radiation_MJ_m2_day_land - net_longwave_radation_MJ_m2_day,
        np.float32(0.0),
    )

    net_solar_radiation_MJ_m2_day_water = get_net_solar_radiation(
        solar_radiation_MJ_m2_day, albedo_water
    )
    net_radiation_water = np.maximum(
        net_solar_radiation_MJ_m2_day_water - net_longwave_radation_MJ_m2_day,
        np.float32(0.0),
    )

    slope_of_saturated_vapour_pressure_curve = (
        get_slope_of_saturation_vapour_pressure_curve(temperature_C=tas_C)
    )

    wind_2m = adjust_wind_speed(sfcWind)

    # latent heat of vaporization [MJ/kg] -> joules required to evaporate 1 kg of water
    latent_heat_of_vaporarization = get_latent_heat_of_vaporization(
        temperature_C=tas_C,
    )

    reference_evapotranspiration_land_mm, reference_evapotranspiration_water_mm = (
        get_reference_evapotranspiration(
            net_radiation_land=net_radiation_land,
            net_radiation_water=net_radiation_water,
            slope_of_saturated_vapour_pressure_curve=slope_of_saturated_vapour_pressure_curve,
            psychrometric_constant=psychrometric_constant,
            wind_2m=wind_2m,
            latent_heat_of_vaporarization=latent_heat_of_vaporarization,
            termperature_C=tas_C,
            vapour_pressure_deficit=vapour_pressure_deficit,
        )
    )

    return (
        reference_evapotranspiration_land_mm / np.float32(1000),
        reference_evapotranspiration_water_mm / np.float32(1000),
        net_radiation_land,
    )


class PotentialEvapotranspiration(Module):
    """Calculate potential evapotranspiration from climate data mainly based on FAO 56."""

    def __init__(self, model, hydrology) -> None:
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        return "hydrology.potential_evapotranspiration"

    def spinup(self) -> None:
        pass

    def step(self) -> None:
        """Dynamic part of the potential evaporation module.

        Caluclation is based on Penman Monteith - FAO 56.
        """
        (
            self.HRU.var.reference_evapotranspiration_grass,
            self.HRU.var.reference_evapotranspiration_water,
            net_absorbed_radiation_vegetation_MJ_m2_day,
        ) = PET(
            tas=self.HRU.tas,
            tasmin=self.HRU.tasmin,
            tasmax=self.HRU.tasmax,
            hurs=self.HRU.hurs,
            ps_pascal=self.HRU.ps,
            rlds=self.HRU.rlds,
            rsds=self.HRU.rsds,
            sfcWind=self.HRU.sfcWind,
        )

        self.grid.var.reference_evapotranspiration_water = self.hydrology.to_grid(
            HRU_data=self.HRU.var.reference_evapotranspiration_water, fn="weightedmean"
        )
        net_absorbed_radiation_vegetation_MJ_m2_day[
            self.HRU.var.land_use_type != FOREST
        ] = np.nan

        self.grid.var.net_absorbed_radiation_vegetation_MJ_m2_day = (
            self.model.hydrology.to_grid(
                HRU_data=net_absorbed_radiation_vegetation_MJ_m2_day, fn="nanmax"
            )
        )

        assert self.HRU.var.reference_evapotranspiration_grass.dtype == np.float32
        assert self.HRU.var.reference_evapotranspiration_water.dtype == np.float32

        self.model.agents.crop_farmers.save_water_deficit()
        self.report(self, locals())
