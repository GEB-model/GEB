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

from geb.module import Module


def get_CO2_induced_crop_factor_adustment(
    CO2_concentration_ppm: float,
) -> float:
    """Calculate the CO2 induced crop factor adjustment.

    For reference see:
        Reference Manual, Chapter 3 – AquaCrop, Version 7.1
        Eq. 3.10e/2

    Args:
        CO2_concentration_ppm: The CO2 concentration in ppm.
    """
    base_co2_concentration_ppm: float = 369.41
    return 1.0 - 0.05 * (CO2_concentration_ppm - base_co2_concentration_ppm) / (
        550 - base_co2_concentration_ppm
    )


class Evaporation(Module):
    """Calculate potential evaporation and pot. transpiration."""

    def __init__(self, model, hydrology):
        super().__init__(model)
        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    @property
    def name(self):
        return "hydrology.evaporation"

    def spinup(self):
        pass

    def step(
        self,
        ETRef: npt.NDArray[np.float32],
        snow_melt: npt.NDArray[np.float32],
        crop_factor: npt.NDArray[np.float32],
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        """Calculate potential transpiration, potential bare soil evaporation, potential evapotranspiration and corrects snow melt for evaporation.

        Args:
            ETRef: Reference evapotranspiration [m]
            snow_melt: Snow melt [m]
            crop_factor: Crop factor for each land use type [dimensionless]

        Returns:
            Potential transpiration, potential bare soil evaporation,
            potential evapotranspiration, remaining snow melt and snow evaporation.
        """
        # calculate potential bare soil evaporation
        potential_bare_soil_evaporation: npt.NDArray[np.float32] = (
            self.hydrology.crop_factor_calibration_factor * 0.2 * ETRef
        )

        # calculate snow evaporation
        snow_evaporation: npt.NDArray[np.float32] = np.minimum(
            snow_melt, potential_bare_soil_evaporation
        )
        snow_melt -= snow_evaporation
        potential_bare_soil_evaporation: npt.NDArray[np.float32] = (
            potential_bare_soil_evaporation - snow_evaporation
        )

        CO2_ppm: float = self.model.forcing.load("CO2")
        CO2_induced_crop_factor_adustment: float = (
            get_CO2_induced_crop_factor_adustment(CO2_ppm)
        )

        potential_evapotranspiration: npt.NDArray[np.float32] = (
            self.hydrology.crop_factor_calibration_factor * crop_factor * ETRef
        ) * np.float32(CO2_induced_crop_factor_adustment)

        potential_transpiration: npt.NDArray[np.float32] = np.maximum(
            0.0,
            potential_evapotranspiration
            - potential_bare_soil_evaporation
            - snow_evaporation,
        )

        self.report(self, locals())

        return (
            potential_transpiration,
            potential_bare_soil_evaporation,
            potential_evapotranspiration,
            snow_melt,
            snow_evaporation,
        )
