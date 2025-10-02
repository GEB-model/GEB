"""Land surface module for GEB."""

import numpy as np
import numpy.typing as npt

from geb.module import Module

from .potential_evapotranspiration import potential_evapotranspiration


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
        pass

    def step(self) -> None:
        """Step function for the land surface module.

        Currently, this function calculates the reference evapotranspiration
        for grass and water surfaces using meteorological data.
        """
        tas_2m_K: npt.NDArray[np.float32] = self.HRU.tas_2m_K
        dewpoint_tas_2m_K: npt.NDArray[np.float32] = self.HRU.dewpoint_tas_2m_K
        ps_pascal: npt.NDArray[np.float32] = self.HRU.ps_pascal
        rlds_W_per_m2: npt.NDArray[np.float32] = self.HRU.rlds_W_per_m2
        rsds_W_per_m2: npt.NDArray[np.float32] = self.HRU.rsds_W_per_m2
        wind_u10m_m_per_s: npt.NDArray[np.float32] = self.HRU.wind_u10m_m_per_s
        wind_v10m_m_per_s: npt.NDArray[np.float32] = self.HRU.wind_v10m_m_per_s

        (
            reference_evapotranspiration_grass_m_dt,
            reference_evapotranspiration_water_m_dt,
            net_absorbed_radiation_vegetation_MJ_m2_dt,
        ) = potential_evapotranspiration(
            tas_K=tas_2m_K,
            dewpoint_tas_K=dewpoint_tas_2m_K,
            ps_pascal=ps_pascal,
            rlds_W_per_m2=rlds_W_per_m2,
            rsds_W_per_m2=rsds_W_per_m2,
            wind_u10m_m_per_s=wind_u10m_m_per_s,
            wind_v10m_m_per_s=wind_v10m_m_per_s,
        )

        self.HRU.var.reference_evapotranspiration_grass_m_per_day = (
            reference_evapotranspiration_grass_m_dt.sum(axis=0)
        )
        self.HRU.var.reference_evapotranspiration_grass_m_per_day[
            self.HRU.var.reference_evapotranspiration_grass_m_per_day < 0
        ] = 0
        self.HRU.var.reference_evapotranspiration_water_m_per_day = (
            reference_evapotranspiration_water_m_dt.sum(axis=0)
        )
        self.HRU.var.reference_evapotranspiration_water_m_per_day[
            self.HRU.var.reference_evapotranspiration_water_m_per_day < 0
        ] = 0
        self.HRU.var.net_absorbed_radiation_vegetation_MJ_m2_per_day = (
            net_absorbed_radiation_vegetation_MJ_m2_dt.sum(axis=0)
        )
        assert (self.HRU.var.reference_evapotranspiration_grass_m_per_day >= 0).all()

        self.grid.var.reference_evapotranspiration_water_m_per_day = (
            self.hydrology.to_grid(
                HRU_data=self.HRU.var.reference_evapotranspiration_water_m_per_day,
                fn="weightedmean",
            )
        )

        self.model.agents.crop_farmers.save_water_deficit(
            self.HRU.var.reference_evapotranspiration_grass_m_per_day
        )
        self.report(locals())
