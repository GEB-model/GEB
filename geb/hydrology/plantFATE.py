from datetime import datetime

import numpy as np
import pandas as pd

from plantFATE import Simulator as sim
from plantFATE import Clim


class Model:
    def __init__(self, param_file):
        self.plantFATE_model = sim(str(param_file))
        self.environment = pd.DataFrame(
            columns=[
                "date",
                "tair",
                "ppfd_max",
                "ppfd",
                "vpd",
                "elv",
                "co2",
                "swp",
                "type",
            ]
        )
        self.emergentProps = pd.DataFrame()
        self.speciesProps = pd.DataFrame()

    def runstep(
        self,
        soil_water_potential,
        vapour_pressure_deficit,
        photosynthetic_photon_flux_density,
        temperature,
    ):
        self.plantFATE_model.update_environment(
            temperature - 273.15,
            photosynthetic_photon_flux_density * 4,
            photosynthetic_photon_flux_density,
            vapour_pressure_deficit * 1000,
            np.nan,
            np.nan,
            soil_water_potential,
        )
        self.plantFATE_model.simulate_step()

        # self.saveEnvironment()
        # self.saveEmergentProps()

        trans = self.plantFATE_model.props.trans / 365
        # return evapotranspiration, soil_specific_depletion_1, soil_specific_depletion_2, soil_specific_depletion_3
        return trans, 0, 0, 0

    def saveEnvironment(self):
        e = pd.DataFrame(
            {
                "date": [
                    self.plantFATE_model.E.tcurrent,
                    self.plantFATE_model.E.tcurrent,
                ],
                "tair": [
                    self.plantFATE_model.E.weightedAveClim.tc,
                    self.plantFATE_model.E.currentClim.tc,
                ],
                "ppfd_max": [
                    self.plantFATE_model.E.weightedAveClim.ppfd_max,
                    self.plantFATE_model.E.currentClim.ppfd_max,
                ],
                "ppfd": [
                    self.plantFATE_model.E.weightedAveClim.ppfd,
                    self.plantFATE_model.E.currentClim.ppfd,
                ],
                "vpd": [
                    self.plantFATE_model.E.weightedAveClim.vpd,
                    self.plantFATE_model.E.currentClim.vpd,
                ],
                "elv": [
                    self.plantFATE_model.E.weightedAveClim.elv,
                    self.plantFATE_model.E.currentClim.elv,
                ],
                "co2": [
                    self.plantFATE_model.E.weightedAveClim.co2,
                    self.plantFATE_model.E.currentClim.co2,
                ],
                "swp": [
                    self.plantFATE_model.E.weightedAveClim.swp,
                    self.plantFATE_model.E.currentClim.swp,
                ],
                "type": ["WeightedAverage", "Instantaneous"],
            }
        )

        self.environment = pd.concat([self.environment, e])

    def saveEmergentProps(self):
        e = pd.DataFrame(
            {
                "date": [self.plantFATE_model.tcurrent],
                "trans": [self.plantFATE_model.props.trans / 365],
                "gs": [self.plantFATE_model.props.gs],
                "gpp": [self.plantFATE_model.props.gpp * 0.5 / 365 * 1000],
                "lai": [self.plantFATE_model.props.lai],
                "npp": [self.plantFATE_model.props.npp * 0.5 / 365 * 1000],
                "cc_est": [self.plantFATE_model.props.cc_est],
                "croot_mass": [self.plantFATE_model.props.croot_mass * 1000 * 0.5],
                "froot_mass": [self.plantFATE_model.props.froot_mass * 1000 * 0.5],
                "lai_vert": [self.plantFATE_model.props.lai_vert],
                "leaf_mass": [self.plantFATE_model.props.leaf_mass * 1000 * 0.5],
                "resp_auto": [self.plantFATE_model.props.resp_auto * 0.5 / 365 * 1000],
                "stem_mass": [self.plantFATE_model.props.stem_mass * 1000 * 0.5],
            }
        )
        self.emergentProps = pd.concat([self.emergentProps, e])

    def exportEnvironment(self, out_file):
        self.environment.to_csv(out_file, sep=",", index=False, encoding="utf-8")

    def exportEmergentProps(self, out_file):
        self.emergentProps.to_csv(out_file, sep=",", index=False, encoding="utf-8")

    def exportSpeciesProps(self, out_file):
        self.speciesProps.to_csv(out_file, sep=",", index=False, encoding="utf-8")

    def first_step(
        self,
        tstart,
        vapour_pressure_deficit,
        soil_water_potential,
        photosynthetic_photon_flux_density,
        temperature,
    ):
        newclim = Clim()
        newclim.tc = temperature - 273.15  # C
        newclim.ppfd_max = photosynthetic_photon_flux_density * 4
        newclim.ppfd = photosynthetic_photon_flux_density
        newclim.vpd = vapour_pressure_deficit * 1000  # kPa -> Pa
        newclim.swp = soil_water_potential  # MPa

        datestart = tstart
        datediff = datestart - datetime(datestart.year, 1, 1)
        tstart = datestart.year + datediff.days / 365
        self.plantFATE_model.init(tstart, newclim)

    def step(
        self,
        soil_water_potential,
        vapour_pressure_deficit,
        photosynthetic_photon_flux_density,
        temperature,
    ):
        (
            evapotranspiration,
            soil_specific_depletion_1,
            soil_specific_depletion_2,
            soil_specific_depletion_3,
        ) = self.runstep(
            soil_water_potential,
            vapour_pressure_deficit,
            photosynthetic_photon_flux_density,
            temperature,
        )

        soil_specific_depletion_1 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion
        soil_specific_depletion_2 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion
        soil_specific_depletion_3 = np.nan  # this is currently not calculated in plantFATE, so just setting to np.nan to avoid confusion

        evapotranspiration = evapotranspiration / 1000  # kg H2O/m2/day to m/day

        return (
            evapotranspiration,
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
