"""Land surface module for GEB."""

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from geb.module import Module
from geb.workflows import balance_check

from .potential_evapotranspiration import potential_evapotranspiration
from .snow_glaciers import snow_model


@njit(parallel=True, cache=True)
def land_surface_model(
    snow_water_equivalent_m: npt.NDArray[np.float32],
    liquid_water_in_snow_m: npt.NDArray[np.float32],
    snow_temperature_C: npt.NDArray[np.float32],
    pr_kg_per_m2_per_s: npt.NDArray[np.float32],
    tas_2m_K: npt.NDArray[np.float32],
    dewpoint_tas_2m_K: npt.NDArray[np.float32],
    ps_pascal: npt.NDArray[np.float32],
    rlds_W_per_m2: npt.NDArray[np.float32],
    rsds_W_per_m2: npt.NDArray[np.float32],
    wind_u10m_m_per_s: npt.NDArray[np.float32],
    wind_v10m_m_per_s: npt.NDArray[np.float32],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """The main land surface model of GEB.

    Args:
        snow_water_equivalent_m: Snow water equivalent in meters.
        liquid_water_in_snow_m: Liquid water in snow in meters.
        snow_temperature_C: Snow temperature in Celsius.
        pr_kg_per_m2_per_s: Precipitation rate in kg/m^2/s.
        tas_2m_K: 2m air temperature in Kelvin.
        dewpoint_tas_2m_K: 2m dewpoint temperature in Kelvin.
        ps_pascal: Surface pressure in Pascals.
        rlds_W_per_m2: Downward longwave radiation in W/m^2.
        rsds_W_per_m2: Downward shortwave radiation in W/m^2.
        wind_u10m_m_per_s: U component of 10m wind speed in m/s.
        wind_v10m_m_per_s: V component of 10m wind speed in m/s.

    Returns:
        Tuple of:
        - runoff_from_melt_m_per_hour: Runoff from snowmelt in meters per hour.
        - runoff_from_direct_rainfall_m_per_hour: Runoff from direct rainfall in
          meters per hour.
        - reference_evapotranspiration_grass_m: Reference evapotranspiration for
          grass in meters.
        - reference_evapotranspiration_water_m: Reference evapotranspiration for
          water in meters.
        - snow_water_equivalent_m: Updated snow water equivalent in meters.
        - liquid_water_in_snow_m: Updated liquid water in snow in meters.
        - sublimation_m: Sublimation in meters.
        - snow_temperature_C: Updated snow temperature in Celsius.
    """
    runoff_from_melt_m_per_hour = np.zeros_like(pr_kg_per_m2_per_s)
    runoff_from_direct_rainfall_m_per_hour = np.zeros_like(pr_kg_per_m2_per_s)
    reference_evapotranspiration_grass_m = np.zeros_like(pr_kg_per_m2_per_s)
    reference_evapotranspiration_water_m = np.zeros_like(pr_kg_per_m2_per_s)
    sublimation_m = np.zeros_like(snow_water_equivalent_m)

    for i in prange(snow_water_equivalent_m.size):
        pr_kg_per_m2_per_s_cell = pr_kg_per_m2_per_s[:, i]
        tas_2m_K_cell = tas_2m_K[:, i]
        dewpoint_tas_2m_K_cell = dewpoint_tas_2m_K[:, i]
        ps_pascal_cell = ps_pascal[:, i]
        rlds_W_per_m2_cell = rlds_W_per_m2[:, i]
        rsds_W_per_m2_cell = rsds_W_per_m2[:, i]
        wind_u10m_m_per_s_cell = wind_u10m_m_per_s[:, i]
        wind_v10m_m_per_s_cell = wind_v10m_m_per_s[:, i]

        snow_water_equivalent_m_cell = snow_water_equivalent_m[i]
        liquid_water_in_snow_m_cell = liquid_water_in_snow_m[i]
        snow_temperature_C_cell = snow_temperature_C[i]

        for hour in range(24):
            tas_C = tas_2m_K_cell[hour] - np.float32(273.15)
            dewpoint_tas_C = tas_2m_K_cell[hour] - np.float32(273.15)

            wind_10m_m_per_s: np.float32 = np.sqrt(
                wind_u10m_m_per_s_cell[hour] ** 2 + wind_v10m_m_per_s_cell[hour] ** 2
            )  # Wind speed at 10m height

            (
                reference_evapotranspiration_grass_m[hour, i],
                reference_evapotranspiration_water_m[hour, i],
                net_absorbed_radiation_vegetation_MJ_m2_dt,
                actual_vapour_pressure_Pa,
            ) = potential_evapotranspiration(
                tas_C=tas_C,
                dewpoint_tas_C=dewpoint_tas_C,
                ps_pa=ps_pascal_cell[hour],
                rlds_W_per_m2=rlds_W_per_m2_cell[hour],
                rsds_W_per_m2=rsds_W_per_m2_cell[hour],
                wind_10m_m_per_s=wind_10m_m_per_s,
            )

            (
                snow_water_equivalent_m_cell,
                liquid_water_in_snow_m_cell,
                snow_temperature_C_cell,
                _,  # melt (before refreezing)
                runoff_from_melt_m_per_hour[hour, i],  # after refreezing
                runoff_from_direct_rainfall_m_per_hour[hour, i],
                sublimation_cell_hour,
                _,  # refreezing
                _,  # snow surface temperature
                _,  # net shortwave radiation
                _,  # net longwave radiation
                _,  # sensible heat flux
                _,  # latent heat flux
            ) = snow_model(
                pr_kg_per_m2_per_s=pr_kg_per_m2_per_s_cell[hour],
                air_temperature_C=tas_C,
                snow_water_equivalent_m=snow_water_equivalent_m_cell,
                liquid_water_in_snow_m=liquid_water_in_snow_m_cell,
                snow_temperature_C=snow_temperature_C_cell,
                shortwave_radiation_W_per_m2=rsds_W_per_m2_cell[hour],
                downward_longwave_radiation_W_per_m2=rlds_W_per_m2_cell[hour],
                vapor_pressure_air_Pa=actual_vapour_pressure_Pa,
                air_pressure_Pa=ps_pascal_cell[hour],
                wind_10m_m_per_s=wind_10m_m_per_s,
            )

            sublimation_m[i] += sublimation_cell_hour

        snow_water_equivalent_m[i] = snow_water_equivalent_m_cell
        liquid_water_in_snow_m[i] = liquid_water_in_snow_m_cell
        snow_temperature_C[i] = snow_temperature_C_cell

    return (
        runoff_from_melt_m_per_hour,
        runoff_from_direct_rainfall_m_per_hour,
        reference_evapotranspiration_grass_m,
        reference_evapotranspiration_water_m,
        snow_water_equivalent_m,
        liquid_water_in_snow_m,
        sublimation_m,
        snow_temperature_C,
    )


class LandSurface(Module):
    """Land surface module for GEB."""

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
        """Spinup function for the land surface module."""
        self.HRU.var.snow_water_equivalent_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )
        self.HRU.var.liquid_water_in_snow_m = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )
        self.HRU.var.snow_temperature_C = self.HRU.full_compressed(
            0.0, dtype=np.float32
        )

    def step(
        self,
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        """Step function for the land surface module.

        Currently, this function calculates the reference evapotranspiration
        for grass and water surfaces using meteorological data.

        asfortranarray()

        Returns:
            A tuple containing:
            - snow_melt_m: Snow melt in meters.
            - rain_m: Rainfall in meters.
            - sublimation_m: Sublimation in meters.

        """
        if __debug__:
            snow_water_equivalent_prev = self.HRU.var.snow_water_equivalent_m.copy()
            liquid_water_in_snow_prev = self.HRU.var.liquid_water_in_snow_m.copy()

        pr_kg_per_m2_per_s = self.HRU.pr_kg_per_m2_per_s

        (
            snow_melt_m,
            rain_m,
            reference_evapotranspiration_grass_m_dt,
            reference_evapotranspiration_water_m_dt,
            self.HRU.var.snow_water_equivalent_m,
            self.HRU.var.liquid_water_in_snow_m,
            sublimation_m,
            self.HRU.var.snow_temperature_C,
        ) = land_surface_model(
            snow_water_equivalent_m=self.HRU.var.snow_water_equivalent_m,
            liquid_water_in_snow_m=self.HRU.var.liquid_water_in_snow_m,
            snow_temperature_C=self.HRU.var.snow_temperature_C,
            pr_kg_per_m2_per_s=np.asfortranarray(pr_kg_per_m2_per_s),
            tas_2m_K=np.asfortranarray(self.HRU.tas_2m_K),
            dewpoint_tas_2m_K=np.asfortranarray(self.HRU.dewpoint_tas_2m_K),
            ps_pascal=np.asfortranarray(self.HRU.ps_pascal),
            rlds_W_per_m2=np.asfortranarray(self.HRU.rlds_W_per_m2),
            rsds_W_per_m2=np.asfortranarray(self.HRU.rsds_W_per_m2),
            wind_u10m_m_per_s=np.asfortranarray(self.HRU.wind_u10m_m_per_s),
            wind_v10m_m_per_s=np.asfortranarray(self.HRU.wind_v10m_m_per_s),
        )

        assert balance_check(
            name="land surface 1",
            how="cellwise",
            influxes=[
                pr_kg_per_m2_per_s.sum(axis=0) * 3.6
            ],  # from kg/m2/s to m/hr and sum over hours to make it day (but then re-arranged for efficiency)
            outfluxes=[snow_melt_m.sum(axis=0), rain_m.sum(axis=0), -sublimation_m],
            prestorages=[snow_water_equivalent_prev, liquid_water_in_snow_prev],
            poststorages=[
                self.HRU.var.snow_water_equivalent_m,
                self.HRU.var.liquid_water_in_snow_m,
            ],
            tolerance=1e-6,
        )

        self.HRU.var.reference_evapotranspiration_water_m_per_day = np.sum(
            reference_evapotranspiration_water_m_dt, axis=0
        )
        self.HRU.var.reference_evapotranspiration_grass_m_per_day = np.sum(
            reference_evapotranspiration_grass_m_dt, axis=0
        )

        self.grid.var.reference_evapotranspiration_water_m_per_day = (
            self.hydrology.to_grid(
                HRU_data=self.HRU.var.reference_evapotranspiration_water_m_per_day,
                fn="weightedmean",
            )
        )

        print("WARNING: setting frost index to zero")
        self.HRU.var.frost_index = np.full_like(self.HRU.var.topwater, np.float32(0.0))

        self.model.agents.crop_farmers.save_water_deficit(
            self.HRU.var.reference_evapotranspiration_grass_m_per_day
        )
        self.report(locals())

        return snow_melt_m, rain_m, sublimation_m
