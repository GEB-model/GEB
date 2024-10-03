from datetime import datetime

import numpy as np
import pandas as pd

from pypfate import Patch as patch

class Model:
    def __init__(self, param_file, acclim_forcing_file, use_acclim):
        self.plantFATE_model = patch(str(param_file))
        self.time_unit_base = self.process_time_units()
        self.tcurrent = 0
        #
        # self.use_acclim = use_acclim
        # if use_acclim:
        #     self.acclimation_forcing = self.read_acclimation_file(acclim_forcing_file)
        #     self.use_acclim = use_acclim

    def read_acclimation_file(self, file):
        df = pd.read_csv(file)
        alldates = df['date'].map(lambda x: datetime.strptime(x, "%Y-%m-%d") - self.time_unit_base)
        alldates = alldates.map(lambda x: x.days - 1)
        df['date_jul'] = alldates
        return df

    def process_time_units(self):
        time_unit = self.plantFATE_model.config.time_unit
        time_unit = time_unit.split()
        if time_unit[0] != 'days' or time_unit[1] != 'since':
            print("incorrect plantFATE time unit; cwatm coupling supports only daily timescale")
            return
        else:
            time_unit = time_unit[2].split("-")
            return datetime(int(time_unit[0]),
                            int(time_unit[1]),
                            int(time_unit[2]))

    def runstep(
        self,
        soil_water_potential,
        vapour_pressure_deficit,
        photosynthetic_photon_flux_density,
        temperature,
        net_radiation,
        topsoil_volumetric_water_content
    ):

        self.plantFATE_model.update_climate(
            368.9,  # co2 - need to make it better
            temperature - 273.15,
            vapour_pressure_deficit * 1000,
            photosynthetic_photon_flux_density,
            soil_water_potential,
            net_radiation,
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
        potential_soil_evaporation = self.plantFATE_model.props.fluxes.pe_soil

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
            net_radiation
    ):
        datestart = datetime(tstart.year, tstart.month, tstart.day)
        datediff = datestart - self.time_unit_base
        datediff = datediff.days - 1

        self.patch.init(datediff, datediff + 1000)

        self.tcurrent = datediff
        self.patch.update_climate(368.9,
                                  temperature,
                                  vapour_pressure_deficit * 1000,
                                  photosynthetic_photon_flux_density,
                                  soil_water_potential,
                                  net_radiation)

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

    def step(
            self,
            soil_water_potential,
            vapour_pressure_deficit,
            photosynthetic_photon_flux_density,
            temperature,  # degrees Celcius, mean temperature
            topsoil_volumetric_water_content,
            net_radiation
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
            topsoil_volumetric_water_content
        )

        soil_specific_depletion_1 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion
        soil_specific_depletion_2 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion
        soil_specific_depletion_3 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion

        transpiration = transpiration / 1000  # kg H2O/m2/day to m/day - double check this value

        return (
            transpiration,
            soil_evaporation,
            soil_specific_depletion_1,
            soil_specific_depletion_2,
            soil_specific_depletion_3,
        )

    def finalize(self):
        self.plantFATE_model.close()

    @property
    def n_individuals(self):
        return sum(self.plantFATE_model.cwm.n_ind_vec)

    @property
    def biomass(self):
        return sum(self.plantFATE_model.cwm.biomass_vec)  # kgC / m2
