from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pypfate import Patch as patch


class Model:
    """Implements the plantFATE model for a single patch of land.

    Args:
        param_file: the path to the plantFATE parameter file.
        acclim_forcing_file: the path to the acclimation forcing file.
        use_acclim: whether to use acclimation forcing or not.
    """

    def __init__(
        self, param_file: str | Path, acclim_forcing_file: str | Path, use_acclim: bool
    ):
        self.plantFATE_model = patch(str(param_file))
        self.time_unit_base = self.process_time_units()
        self.tcurrent = 0

        self.use_acclim = use_acclim
        if use_acclim:
            self.acclimation_forcing = self.read_acclimation_file(acclim_forcing_file)
            self.use_acclim = use_acclim

        self.first_step_was_run = False

    def read_acclimation_file(self, file):
        df = pd.read_csv(file)
        alldates = df["date"].map(
            lambda x: datetime.strptime(x, "%Y-%m-%d") - self.time_unit_base
        )
        alldates = alldates.map(lambda x: x.days - 1)
        df["date_jul"] = alldates
        return df

    def process_time_units(self):
        time_unit = self.plantFATE_model.config.time_unit
        time_unit = time_unit.split()
        if time_unit[0] != "days" or time_unit[1] != "since":
            raise ValueError(
                "incorrect plantFATE time unit; cwatm coupling supports only daily timescale"
            )
        time_unit = time_unit[2].split("-")
        return datetime(int(time_unit[0]), int(time_unit[1]), int(time_unit[2]))

    def runstep(
        self,
        soil_water_potential,
        vapour_pressure_deficit,
        photosynthetic_photon_flux_density,
        temperature,
        net_radiation,
        topsoil_volumetric_water_content,
    ):
        assert self.first_step_was_run, "first step must be run before running a step"

        self.plantFATE_model.update_climate(
            np.float64(368.9),  # co2
            np.float64(temperature),
            np.float64(vapour_pressure_deficit),  # kPa -> Pa
            np.float64(photosynthetic_photon_flux_density),
            np.float64(soil_water_potential),
            np.float64(net_radiation),
        )
        #
        # if (self.use_acclim):
        #     index_acclim = self.acclimation_forcing.index[
        #         self.acclimation_forcing['date_jul'] == self.tcurrent].tolist()
        #     self.plantFATE_model.update_climate_acclim(self.tcurrent,
        #                                      368.9,
        #                                      self.acclimation_forcing.loc[index_acclim, 'temp.C.'],
        #                                      self.acclimation_forcing.loc[index_acclim, 'vpd'],
        #                                      self.calculate_photosynthetic_photon_flux_density(
        #                                          self.acclimation_forcing.loc[index_acclim, 'shortwave.W.m2.'], albedo),
        #                                      soil_water_potential)

        self.plantFATE_model.simulate_to(self.tcurrent)
        trans = self.plantFATE_model.props.fluxes.trans
        # trans = trans/365
        potential_soil_evaporation = self.plantFATE_model.props.fluxes.pe_soil
        # print('potential soil evap')
        # print(potential_soil_evaporation)
        # print('plantFate transpiration')
        # print(trans)
        soil_evaporation = potential_soil_evaporation * topsoil_volumetric_water_content

        # return transpiration, soil_evaporation, soil_specific_depletion_1, soil_specific_depletion_2, soil_specific_depletion_3
        return trans, soil_evaporation, 0, 0, 0

    def first_step(
        self,
        tstart,
        soil_water_potential,
        vapour_pressure_deficit,
        photosynthetic_photon_flux_density,
        temperature,  # degrees Celcius, mean temperature
        topsoil_volumetric_water_content,
        net_radiation,
    ):
        datestart = datetime(tstart.year, tstart.month, tstart.day)
        datediff = datestart - self.time_unit_base
        datediff = datediff.days - 1
        # print("Running first step")
        self.tcurrent = datediff

        self.plantFATE_model.init(datediff, datediff + 1000)

        self.plantFATE_model.reset_time(datediff)

        # print("Running first step - after init")
        self.plantFATE_model.update_climate(
            368.9,
            temperature,
            vapour_pressure_deficit,
            photosynthetic_photon_flux_density,
            soil_water_potential,
            net_radiation,
        )
        # 250)
        # print("finished running first step")
        # if (self.use_acclim):
        #     index_acclim = self.acclimation_forcing.index[
        #         self.acclimation_forcing['date_jul'] == self.tcurrent].tolist()
        #     self.patch.update_climate_acclim(self.tcurrent,
        #                                      368.9,
        #                                      self.acclimation_forcing.loc[index_acclim, 'temp.C.'],
        #                                      self.acclimation_forcing.loc[index_acclim, 'vpd'],
        #                                      self.calculate_photosynthetic_photon_flux_density(
        #                                          self.acclimation_forcing.loc[index_acclim, 'shortwave.W.m2.'], albedo),
        #                                      soil_water_potential)

        self.first_step_was_run = True

    def step(
        self,
        soil_water_potential,
        vapour_pressure_deficit,
        photosynthetic_photon_flux_density,
        temperature,  # degrees Celcius, mean temperature
        topsoil_volumetric_water_content,
        net_radiation,
    ):
        self.tcurrent += 1

        (
            transpiration,
            soil_evaporation,
            soil_specific_depletion_1,
            soil_specific_depletion_2,
            soil_specific_depletion_3,
        ) = self.runstep(
            soil_water_potential,
            vapour_pressure_deficit,
            photosynthetic_photon_flux_density,
            temperature,
            net_radiation,
            # 250,
            topsoil_volumetric_water_content,
        )

        soil_specific_depletion_1 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion
        soil_specific_depletion_2 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion
        soil_specific_depletion_3 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion

        transpiration = transpiration / 1000  # kg H2O/m2/day to m/day

        return (
            transpiration,
            soil_evaporation,
            soil_specific_depletion_1,
            soil_specific_depletion_2,
            soil_specific_depletion_3,
        )

    def finalize(self):
        import os

        print(os.getcwd())
        self.plantFATE_model.close()

    @property
    def n_individuals(self):
        return sum(self.plantFATE_model.props.species.n_ind_vec)

    @property
    def biomass(self):
        return self.plantFATE_model.props.structure.biomass  # kgC / m2

    @property
    def npp(self):
        return self.plantFATE_model.props.fluxes.npp
