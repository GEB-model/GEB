"""This module contains the Government agent class for GEB."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .general import AgentBaseClass

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel


class Government(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model: GEBModel, agents: Agents) -> None:
        """Initialize the government agent.

        Args:
            model: The GEB model.
            agents: The class that includes all agent types (allowing easier communication between agents).
        """
        super().__init__(model)
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["government"]
            if "government" in self.model.config["agent_settings"]
            else {}
        )
        self.ratio_farmers_to_provide_subsidies_per_year = 0.05

    @property
    def name(self) -> str:
        """Name of the module.

        Returns:
            The name of the module.
        """
        return "agents.government"

    def spinup(self) -> None:
        """This function is called during model spinup."""
        pass

    def set_irrigation_limit(self) -> None:
        """Set the irrigation limit for crop farmers based on the configuration.

        The irrigation limit can be set per capita, per area of fields, or per command area.
        """
        if "irrigation_limit" not in self.config:
            return None
        irrigation_limit = self.config["irrigation_limit"]
        if irrigation_limit["per"] == "capita":
            self.agents.crop_farmers.var.irrigation_limit_m3[:] = (
                self.agents.crop_farmers.var.household_size * irrigation_limit["limit"]
            )
        elif irrigation_limit["per"] == "area":  # limit per m2 of field
            self.agents.crop_farmers.var.irrigation_limit_m3[:] = (
                self.agents.crop_farmers.field_size_per_farmer
                * irrigation_limit["limit"]
            )
        elif irrigation_limit["per"] == "command_area":
            farmer_command_area = self.agents.crop_farmers.command_area
            farmers_per_command_area = np.bincount(
                farmer_command_area[farmer_command_area != -1],
                minlength=self.model.hydrology.waterbodies.n,
            )

            # get yearly usable release m3. We do not use the current year, as it
            # may not be complete yet, and we only use up to the history fill index
            yearly_usable_release_m3_per_command_area = np.full(
                self.model.hydrology.waterbodies.n, np.nan, dtype=np.float32
            )
            yearly_usable_release_m3_per_command_area[
                self.model.hydrology.waterbodies.is_reservoir
            ] = (self.agents.reservoir_operators.yearly_usuable_release_m3).mean(axis=1)

            irritation_limit_per_command_area = (
                yearly_usable_release_m3_per_command_area / farmers_per_command_area
            )

            # give all farmers there unique irrigation limit
            # all farmers without a command area get no irrigation limit (nan)
            irrigation_limit_per_farmer = irritation_limit_per_command_area[
                farmer_command_area
            ]
            irrigation_limit_per_farmer[farmer_command_area == -1] = np.nan

            # make sure all farmers in a command area have an irrigation limit
            assert not np.isnan(
                irrigation_limit_per_farmer[farmer_command_area != -1]
            ).any()

            self.agents.crop_farmers.var.irrigation_limit_m3[:] = (
                irrigation_limit_per_farmer
            )
        else:
            raise NotImplementedError(
                "Only 'capita' and 'area' are implemented for irrigation limit"
            )
        if "min" in irrigation_limit:
            self.agents.crop_farmers.var.irrigation_limit_m3[
                self.agents.crop_farmers.var.irrigation_limit_m3
                < irrigation_limit["min"]
            ] = irrigation_limit["min"]

    def step(self) -> None:
        """This function is run each timestep."""
        self.set_irrigation_limit()
        self.report(locals())

    def adaptation(self) -> None:
        """This function is used to decide whether adaptation is needed and what adaptation measures to apply."""
        # something to specify that this should only run when adaptation is turned on in the config file
        # should this step be skipped during spinup?
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January

            # calculate the water risk, equity and ecosystem health for the current year (adaptation is enabled and it is january first)
            water_risk_value = self.calculate_water_risk()  # this is defined by the EAD
            equity_indicator_value = (
                self.calculate_equity()
            )  # this is defined by the exposure inequalty
            ecosystem_indicator_value = (
                self.calculate_ecosystem_indicator()
            )  # this is defined by the ecosystem indicator

            # Set the indicator value and the thresholds for the adaptation decision.
            # but they should be defined in such a way that they trigger adaptation when threshold is crossed

            thresholds = {
                # maybe actually we should define these threshold in the config file, how to do this?
                # defines the threshold --> value in current situation, threshold
                "water_risk": (
                    water_risk_value,
                    self.config.get(
                        "water_risk_threshold", 1000000
                    ),  # this is incorrect
                ),  # currently arbitrary threshold of 1 million€
                "equity_indicator": (
                    equity_indicator_value,
                    1,
                ),  # this is defined by the equity indicator, if the value is >1 it is unequal as low-income groups experience more exposure than expected
                "ecosystem_indicator": (
                    ecosystem_indicator_value,
                    0.5,
                ),  # this is defined by the ecosystem indicator, should improve compared to current situation, which needs to be calculated
            }

            # if any of these the thresholds are crossed, we adapt the policies so a make a list of the thresholds that are crossed
            triggered = [
                key for key, value in thresholds.items() if value[0] > value[1]
            ]

            # if one of the threshold is triggered something needs to be done
            if triggered:
                print(
                    f"Adaptation needed, the following thresholds are crossed: {triggered}"
                )
                self.apply_adaptation(
                    triggered
                )  # we now know what has caused the need for adaptation, so now we move on and actually implement the adaptation through another function, what caused the trigger is passed to this function
            else:
                print(
                    "No adaptation needed, all thresholds are below the defined thresholds"
                )

    def calculate_water_risk(self) -> None | float:
        # should also only be calculated if adaptation is turned on in the config file otherwise it is not needed
        """Calculate the water risk for the current year.
        returns the expected annual damage (EAD) in euros, which is calculated as the product of the probability of a water-related hazard occurring and the potential damage caused by that hazard.
        """
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return
        # this is defined by the EAD
        ead = 1200000  # this should become an actual calculation
        return ead

    def calculate_equity(self) -> None | float:
        # should also only be calculated if adaptation is turned on in the config file otherwise it is not needed
        """Calculate the equity for the current year.
        Returns the equity index value.
        """
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return
        # this is defined by the exposure inequalty
        equity_index = 1.2  # this should become an actual calculation
        return equity_index

    def calculate_ecosystem_indicator(self) -> None | float:
        # should also only be calculated if adaptation is turned on in the config file otherwise it is not needed
        """Calculate the ecosystem health for the current year.
        returns the ecosystem index.
        """
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return
        # this is defined by the ecosystem indicator
        ecosystem_indicator = 0.4  # this should become an actual calculation
        return ecosystem_indicator

    def apply_adaptation(self, triggered) -> None | str:
        # should also only be applied if adaptation is turned on in the config file otherwise it is not needed
        """Apply the adaptation measures decided in the adaptation function."""
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January

        if "water_risk" in triggered:
            # apply adaptation measure, right now this is only one but aim to expand it to multiple measures so that the agent can decide which one to apply based on what triggered the need for adaptation
            # apply updating the building structure but this takes the number of households that are adapting as input so we fist need to define that
            adaptation_fraction = self.config.get(
                "adaptation_fraction", 0.1
            )  # default to 10% of households adapting, otherwise specifieed in config file
            # figure out which buildings are marked as flooded in this year
            household_agents = self.agents.households
            flooded_buildings_mask = set(
                household_agents.buildings.loc[
                    household_agents.buildings["flooded"] == True, "object_id"
                ].astype(int)
            )
            # figure out which households are in these flooded buildings
            households_in_flooded_buildings = np.where(
                pd.Series(household_agents.var.building_id_of_household.data).isin(
                    flooded_buildings_mask
                )
            )[0]
            # the government decides who of those are adapting, we only pick a fraction as specified in the config gilr
            n_to_adapt = int(len(households_in_flooded_buildings) * adaptation_fraction)
            adapting_households_sample = np.random.choice(
                households_in_flooded_buildings, size=n_to_adapt, replace=False
            )
            # use the function to floodproof the buildings of the households who are selected to adapt.
            household_agents.update_building_adaptation_status(
                adapting_households_sample
            )
            print(
                f"the government adapted {n_to_adapt} of the {len(households_in_flooded_buildings)} households in the floodzone by floodproofing their buildings"
            )
        if "equity_indicator" in triggered:
            # apply subsidies" --> maybe we can change who the subsidies are applied to but idk if that would make it better
            # this piece of code is still in Veerle's branch
            self.provide_subsidies()

        if "ecosystem_indicator" in triggered:
            # apply reforestation
            # needs to be updated to the main, was made by Tarun
            self.prepare_modified_soil_maps_for_forest()
            self._save_forest_planting_figure()
            self.remove_farmers_from_converted_forest_areas()
