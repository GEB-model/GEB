import numpy as np
import numpy.typing as npt
from numba import njit

from geb.module import Module


@njit(cache=True, inline="always")
def get_vapour_pressure(
    temperature_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the vapour pressure based on temperature.

    Args:
        temperature_C: Temperature in Celsius.

    Returns:
        Saturated vapour pressure in kPa.
    """
    return np.float32(0.6108) * np.exp(
        (np.float32(17.27) * temperature_C) / (temperature_C + np.float32(237.3))
    )


@njit(cache=True, inline="always")
def get_vapour_pressure_deficit(
    saturated_vapour_pressure: npt.NDArray[np.float32],
    actual_vapour_pressure: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the vapour pressure deficit.

    Args:
        saturated_vapour_pressure: Saturated vapour pressure.
        actual_vapour_pressure: Actual vapour pressure.

    Returns:
        Vapour pressure deficit in kPa.
    """
    return np.maximum(
        saturated_vapour_pressure - actual_vapour_pressure,
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
    tas_C: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Calculate the upwelling long wave radiation based on temperature.

    Args:
        tas_C: Temperature in Celsius.

    Returns:
        Upwelling long wave radiation in MJ/m^2/hour.
    """
    return (np.float32(4.903e-9) / 24) * ((tas_C + np.float32(273.16)) ** 4)


@njit(cache=True, inline="always")
def W_per_m2_to_MJ_per_m2_per_hour(
    solar_radiation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Convert solar radiation from W/m^2 to MJ/m^2/hour.

    Args:
        solar_radiation: Solar radiation in W/m^2.

    Returns:
        Solar radiation in MJ/m^2/hour.
    """
    return solar_radiation * (np.float32(3600) * np.float32(1e-6))


@njit(cache=True, inline="always")
def get_net_solar_radiation(
    solar_radiation: npt.NDArray[np.float32], albedo: np.float32
) -> npt.NDArray[np.float32]:
    """Calculate net solar radiation based on incoming solar radiation and albedo.

    Args:
        solar_radiation: Incoming solar radiation.
        albedo: Albedo of the surface (fraction of reflected solar radiation).

    Returns:
        Net solar radiation in MJ/m²/timestep.
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
        * get_vapour_pressure(temperature_C=temperature_C)
        / ((temperature_C + np.float32(237.3)) ** 2)
    )


@njit(cache=True, inline="always")
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
    latent_heat_of_vaporization: npt.NDArray[np.float32],
    temperature_C: npt.NDArray[np.float32],
    vapour_pressure_deficit: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Combine all terms of the Penman-Monteith equation to calculate reference evapotranspiration.

    Args:
        net_radiation_land: Net radiation for land in MJ/m^2/hour.
        net_radiation_water: Net radiation for water in MJ/m^2/hour.
        slope_of_saturated_vapour_pressure_curve: Slope of the saturation vapour pressure curve in kPa/°C.
        psychrometric_constant: Psychrometric constant in kPa/°C.
        wind_2m: Wind speed at 2 m height in m/s.
        latent_heat_of_vaporization: Latent heat of vaporization in MJ/kg.
        temperature_C: Temperature in Celsius.
        vapour_pressure_deficit: Vapour pressure deficit in kPa.

    Returns:
        reference_evapotranspiration_land: Reference evapotranspiration for land in mm/hour.
        reference_evapotranspiration_water: Reference evapotranspiration for water in mm/hour.
    """
    denominator = slope_of_saturated_vapour_pressure_curve + psychrometric_constant * (
        np.float32(1) + np.float32(0.34) * wind_2m
    )

    common_energy_factor = (
        slope_of_saturated_vapour_pressure_curve
        / latent_heat_of_vaporization
        / denominator
    )

    energy_term_land = net_radiation_land * common_energy_factor
    energy_term_water = net_radiation_water * common_energy_factor

    aerodynamic_term = (
        psychrometric_constant
        * np.float32(37)
        / (temperature_C + np.float32(273.16))
        * wind_2m
        * vapour_pressure_deficit
    ) / denominator
    return energy_term_land + aerodynamic_term, energy_term_water + aerodynamic_term


@njit(cache=True, parallel=True)
def PET(
    tas_K: npt.NDArray[np.float32],
    dewpoint_tas_K: npt.NDArray[np.float32],
    ps_pascal: npt.NDArray[np.float32],
    rlds_W_per_m2: npt.NDArray[np.float32],
    rsds_W_per_m2: npt.NDArray[np.float32],
    wind_u10m_m_per_s: npt.NDArray[np.float32],
    wind_v10m_m_per_s: npt.NDArray[np.float32],
    albedo_canopy: np.float32 = np.float32(0.13),
    albedo_water: np.float32 = np.float32(0.05),
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Calculate potential evapotranspiration based on Penman-Monteith equation.

    Penman-Montheith equation:

        ET0 = (1 / λ * (Rn - G) + γ * (37 / (T + 273)) * u2 * (es - ea)) / (Δ + γ * (1 + 0.34 * u2))

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
        TODO: Add soil heat flux density (G) term. Currently assumed to be 0.

    Args:
        tas_K: average air temperature in Kelvin.
        dewpoint_tas_K: dew point temperature in Kelvin.
        ps_pascal: surface pressure in Pascals.
        rlds_W_per_m2: long wave downward surface radiation fluxes in W/m^2.
        rsds_W_per_m2: short wave downward surface radiation fluxes in W/m^2.
        wind_u10m_m_per_s: wind speed at 10 m height in m/s (u component).
        wind_v10m_m_per_s: wind speed at 10 m height in m/s (v component).
        albedo_canopy: albedo of vegetation canopy (default = 0.13).
        albedo_water: albedo of water surface (default = 0.05).

    Returns:
        reference_evapotranspiration_land: reference evapotranspiration for land in m/day.
        reference_evapotranspiration_water: reference evapotranspiration for water in m/day.
        net_radiation_land: net radiation for land in MJ/m^2/day.
    """
    tas_C: npt.NDArray[np.float32] = tas_K - np.float32(273.15)
    dewpoint_tas_C: npt.NDArray[np.float32] = dewpoint_tas_K - np.float32(273.15)

    actual_vapour_pressure: npt.NDArray[np.float32] = get_vapour_pressure(
        temperature_C=dewpoint_tas_C
    )
    saturated_vapour_pressure: npt.NDArray[np.float32] = get_vapour_pressure(
        temperature_C=tas_C
    )

    vapour_pressure_deficit: npt.NDArray[np.float32] = get_vapour_pressure_deficit(
        saturated_vapour_pressure=saturated_vapour_pressure,
        actual_vapour_pressure=actual_vapour_pressure,
    )

    psychrometric_constant: npt.NDArray[np.float32] = get_psychrometric_constant(
        ps_pascal=ps_pascal
    )

    rlus_MJ_m2_dt: npt.NDArray[np.float32] = get_upwelling_long_wave_radiation(tas_C)
    rlds_MJ_m2_dt: npt.NDArray[np.float32] = W_per_m2_to_MJ_per_m2_per_hour(
        rlds_W_per_m2
    )

    net_longwave_radation_MJ_m2_dt: npt.NDArray[np.float32] = (
        rlus_MJ_m2_dt - rlds_MJ_m2_dt
    )

    solar_radiation_MJ_m2_dt: npt.NDArray[np.float32] = W_per_m2_to_MJ_per_m2_per_hour(
        rsds_W_per_m2
    )

    net_solar_radiation_MJ_m2_land_dt: npt.NDArray[np.float32] = (
        get_net_solar_radiation(solar_radiation_MJ_m2_dt, albedo_canopy)
    )

    net_radiation_land_dt: npt.NDArray[np.float32] = (
        net_solar_radiation_MJ_m2_land_dt - net_longwave_radation_MJ_m2_dt
    )
    net_solar_radiation_MJ_m2_water_dt: npt.NDArray[np.float32] = (
        get_net_solar_radiation(solar_radiation_MJ_m2_dt, albedo_water)
    )
    net_radiation_water_dt: npt.NDArray[np.float32] = np.maximum(
        net_solar_radiation_MJ_m2_water_dt - net_longwave_radation_MJ_m2_dt,
        np.float32(0.0),
    )

    slope_of_saturated_vapour_pressure_curve: npt.NDArray[np.float32] = (
        get_slope_of_saturation_vapour_pressure_curve(temperature_C=tas_C)
    )

    wind_10m: npt.NDArray[np.float32] = np.sqrt(
        wind_u10m_m_per_s**2 + wind_v10m_m_per_s**2
    )
    wind_2m: npt.NDArray[np.float32] = adjust_wind_speed(wind_10m)

    # latent heat of vaporization [MJ/kg] -> joules required to evaporate 1 kg of water
    latent_heat_of_vaporization: npt.NDArray[np.float32] = (
        get_latent_heat_of_vaporization(
            temperature_C=tas_C,
        )
    )

    reference_evapotranspiration_land_mm, reference_evapotranspiration_water_mm = (
        get_reference_evapotranspiration(
            net_radiation_land=net_radiation_land_dt,
            net_radiation_water=net_radiation_water_dt,
            slope_of_saturated_vapour_pressure_curve=slope_of_saturated_vapour_pressure_curve,
            psychrometric_constant=psychrometric_constant,
            wind_2m=wind_2m,
            latent_heat_of_vaporization=latent_heat_of_vaporization,
            temperature_C=tas_C,
            vapour_pressure_deficit=vapour_pressure_deficit,
        )
    )

    return (
        reference_evapotranspiration_land_mm / np.float32(1000),
        reference_evapotranspiration_water_mm / np.float32(1000),
        net_radiation_land_dt,
    )


class LandSurface(Module):
    def __init__(self, model: "GEBModel", hydrology: "Hydrology") -> None:
        """Initialize the potential evaporation module.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology module.
        """
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self) -> str:
        """Name of the module."""
        return "hydrology.landsurface"

    def spinup(self) -> None:
        pass

    def step(self) -> None:
        """Step function for the potential evaporation module."""
        tas_2m_K: npt.NDArray[np.float32] = self.HRU.tas_2m_K
        dewpoint_tas_2m_K: npt.NDArray[np.float32] = self.HRU.dewpoint_tas_2m_K
        ps_pascal: npt.NDArray[np.float32] = self.HRU.ps_pascal
        rlds_W_per_m2: npt.NDArray[np.float32] = self.HRU.rlds_W_per_m2
        rsds_W_per_m2: npt.NDArray[np.float32] = self.HRU.rsds_W_per_m2
        wind_u10m_m_per_s: npt.NDArray[np.float32] = self.HRU.wind_u10m_m_per_s
        wind_v10m_m_per_s: npt.NDArray[np.float32] = self.HRU.wind_v10m_m_per_s

        reference_evapotranspiration_grass_m_dt = np.full_like(tas_2m_K, np.nan)
        reference_evapotranspiration_water_m_dt = np.full_like(tas_2m_K, np.nan)
        net_absorbed_radiation_vegetation_MJ_m2_dt = np.full_like(tas_2m_K, np.nan)

        for hour in range(24):
            (
                reference_evapotranspiration_grass_m_dt[hour],
                reference_evapotranspiration_water_m_dt[hour],
                net_absorbed_radiation_vegetation_MJ_m2_dt[hour],
            ) = PET(
                tas_K=tas_2m_K[hour],
                dewpoint_tas_K=dewpoint_tas_2m_K[hour],
                ps_pascal=ps_pascal[hour],
                rlds_W_per_m2=rlds_W_per_m2[hour],
                rsds_W_per_m2=rsds_W_per_m2[hour],
                wind_u10m_m_per_s=wind_u10m_m_per_s[hour],
                wind_v10m_m_per_s=wind_v10m_m_per_s[hour],
            )

            assert reference_evapotranspiration_grass_m_dt.dtype == np.float32
            assert reference_evapotranspiration_water_m_dt.dtype == np.float32

        self.HRU.var.reference_evapotranspiration_grass_m_per_day_ = (
            reference_evapotranspiration_grass_m_dt.sum(axis=0)
        )
        self.HRU.var.reference_evapotranspiration_grass_m_per_day_[
            self.HRU.var.reference_evapotranspiration_grass_m_per_day_ < 0
        ] = 0
        self.HRU.var.reference_evapotranspiration_water_m_per_day_ = (
            reference_evapotranspiration_water_m_dt.sum(axis=0)
        )
        self.HRU.var.reference_evapotranspiration_water_m_per_day_[
            self.HRU.var.reference_evapotranspiration_water_m_per_day_ < 0
        ] = 0
        self.HRU.var.net_absorbed_radiation_vegetation_MJ_m2_per_day_ = (
            net_absorbed_radiation_vegetation_MJ_m2_dt.sum(axis=0)
        )
        assert (self.HRU.var.reference_evapotranspiration_grass_m_per_day_ >= 0).all()

        self.model.agents.crop_farmers.save_water_deficit(
            self.HRU.var.reference_evapotranspiration_grass_m_per_day_
        )
        self.report(locals())
