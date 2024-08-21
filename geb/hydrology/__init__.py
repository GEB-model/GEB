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

import os

import numpy as np
from geb.workflows import TimingModule

from .potential_evapotranspiration import PotentialEvapotranspiration
from .snow_frost import SnowFrost
from .soil import Soil
from .landcover import LandCover
from .sealed_water import SealedWater
from .evaporation import Evaporation
from .groundwater import GroundWater
from .water_demand import WaterDemand
from .interception import Interception
from .runoff_concentration import RunoffConcentration
from .lakes_res_small import SmallLakesReservoirs
from .routing import Routing
from .lakes_reservoirs import LakesReservoirs


class Hydrology:
    def __init__(self):
        """
        Init part of the initial part
        defines the mask map and the outlet points
        initialization of the hydrological modules
        """
        self.init_water_table_file = os.path.join(
            self.config["general"]["init_water_table"]
        )
        self.DynamicResAndLakes = False
        self.useSmallLakes = False
        self.crop_factor_calibration_factor = 1

        elevation_std = self.data.grid.load(
            self.files["grid"]["landsurface/topo/elevation_STD"]
        )
        elevation_std = self.data.to_HRU(data=elevation_std, fn=None)

        self.potential_evapotranspiration = PotentialEvapotranspiration(self)
        self.snowfrost = SnowFrost(self, elevation_std)
        self.landcover = LandCover(self)
        self.soil = Soil(self, elevation_std)
        self.evaporation = Evaporation(self)
        self.groundwater = GroundWater(self)
        self.interception = Interception(self)
        self.sealed_water = SealedWater(self)
        self.runoff_concentration = RunoffConcentration(self)
        self.lakes_res_small = SmallLakesReservoirs(self)
        self.routing_kinematic = Routing(self)
        self.lakes_reservoirs = LakesReservoirs(self)
        self.water_demand = WaterDemand(self)

    def step(self):
        """
        Dynamic part of CWATM
        calls the dynamic part of the hydrological modules
        Looping through time and space

        Note:
            if flags set the output on the screen can be changed e.g.

            * v: no output at all
            * l: time and first gauge discharge
            * t: timing of different processes at the end
        """

        timer = TimingModule("CWatM")

        self.potential_evapotranspiration.step()
        timer.new_split("PET")

        self.lakes_reservoirs.step()
        timer.new_split("Waterbodies")

        self.snowfrost.step()
        timer.new_split("Snow and frost")

        (
            interflow,
            directRunoff,
            groundwater_recharge,
            groundwater_abstraction,
            channel_abstraction,
            openWaterEvap,
            returnFlow,
        ) = self.landcover.step()
        timer.new_split("Landcover")

        self.groundwater.step(groundwater_recharge, groundwater_abstraction)
        timer.new_split("GW")

        self.runoff_concentration.step(interflow, directRunoff)
        timer.new_split("Runoff concentration")

        self.lakes_res_small.step()
        timer.new_split("Small waterbodies")

        self.routing_kinematic.step(openWaterEvap, channel_abstraction, returnFlow)
        timer.new_split("Routing")

        if self.timing:
            print(timer)

    def finalize(self) -> None:
        """
        Finalize the model
        """
        # finalize modflow model
        self.groundwater.modflow.finalize()

        if self.config["general"]["simulate_forest"]:
            for plantFATE_model in self.model.plantFATE:
                if plantFATE_model is not None:
                    plantFATE_model.finalize()

    def export_water_table(self) -> None:
        """Function to save required water table output to file."""
        dirname = os.path.dirname(self.init_water_table_file)
        os.makedirs(dirname, exist_ok=True)
        np.save(
            self.init_water_table_file,
            self.groundwater.modflow.decompress(self.groundwater.modflow.head),
        )

    @property
    def n_individuals_per_m2(self):
        n_invidiuals_per_m2_per_HRU = np.array(
            [model.n_individuals for model in self.plantFATE if model is not None]
        )
        land_use_ratios = self.data.HRU.land_use_ratio[self.soil.plantFATE_forest_RUs]
        return np.array(
            (n_invidiuals_per_m2_per_HRU * land_use_ratios).sum()
            / land_use_ratios.sum()
        )

    @property
    def biomass_per_m2(self):
        biomass_per_m2_per_HRU = np.array(
            [model.biomass for model in self.plantFATE if model is not None]
        )
        land_use_ratios = self.data.HRU.land_use_ratio[self.soil.plantFATE_forest_RUs]
        return np.array(
            (biomass_per_m2_per_HRU * land_use_ratios).sum() / land_use_ratios.sum()
        )
