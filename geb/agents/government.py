"""This module contains the Government agent class for GEB."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geb.hydrology.landcovers import FOREST
from geb.workflows.io import read_geom

from .general import AgentBaseClass

if TYPE_CHECKING:
    from geb.agents import Agents
    from geb.model import GEBModel

logger = logging.getLogger(__name__)


class Government(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model: GEBModel, agents: Agents) -> None:
        """Initialize the Government agent.

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
        if self.model.current_timestep == 0 and self.config.get("plant_forest", False):
            self.prepare_modified_soil_maps_for_forest()

        self.adaptation()
        self.set_irrigation_limit()

        self.report(locals())

    def prepare_modified_soil_maps_for_forest(self) -> None:
        """Plant forest: update soil properties in memory and remove displaced farmers.

        Loads the forest restoration potential at grid scale, applies a threshold
        to identify suitable HRUs, copies mean soil property values from existing forest
        HRUs to suitable HRUs, saves a figure, and removes farmers from converted areas.
        The threshold is read from the config key ``forest_restoration_potential_threshold``
        and defaults to 0.5.

        When ``increment_fraction`` is set, the model automatically determines which
        chunk to plant based on current land use. Suitable HRUs are sorted by potential
        value (highest first); those already classified as FOREST (from previous years
        or the initial state) are skipped, and the next
        ``increment_fraction * n_suitable`` HRUs are planted. Calling the function again
        the following year therefore plants the next batch automatically — compatible
        with the annual adaptation pathway that calls this on every January 1st.
        """
        hydrology = self.model.hydrology
        plant_forest_config = self.config.get("plant_forest", {})
        if isinstance(plant_forest_config, dict):
            threshold = plant_forest_config.get(
                "forest_restoration_potential_threshold", 0.5
            )
            increment_fraction = plant_forest_config.get("increment_fraction", None)
        else:
            threshold = 0.5
            increment_fraction = None

        forest_potential = hydrology.grid.load(
            self.model.files["grid"]["landsurface/forest_restoration_potential_ratio"]
        )
        suitability_grid = forest_potential >= threshold
        suitability_HRU = hydrology.to_HRU(data=suitability_grid).astype(bool)

        if increment_fraction is not None:
            # Sort suitable HRUs by potential value descending (best areas first).
            # Skip any already classified as FOREST (planted in prior years or
            # originally forest), then take the next chunk of
            # increment_fraction * n_suitable HRUs.
            forest_potential_HRU = hydrology.to_HRU(data=forest_potential)
            suitable_indices = np.where(suitability_HRU)[0]
            suitable_potentials = forest_potential_HRU[suitable_indices]
            sorted_indices = suitable_indices[np.argsort(suitable_potentials)[::-1]]

            n_suitable = len(sorted_indices)

            already_forest = hydrology.HRU.var.land_use_type == FOREST
            remaining = sorted_indices[~already_forest[sorted_indices]]

            if len(remaining) == 0:
                self.model.logger.warning(
                    "Incremental reforestation: all %d suitable HRUs are already "
                    "forest. No planting applied.",
                    n_suitable,
                )
                return

            # chunk is 10% of what remains, not 10% of the fixed total
            chunk_size = max(1, int(np.round(increment_fraction * len(remaining))))
            chunk_indices = remaining[:chunk_size]
            n_already = n_suitable - len(remaining)
            self.model.logger.info(
                "Incremental reforestation: planting %d HRUs (rank %d–%d of %d "
                "suitable; %d already forest, %d remaining).",
                len(chunk_indices),
                n_already,
                n_already + len(chunk_indices) - 1,
                n_suitable,
                n_already,
                len(remaining),
            )
            suitability_HRU = np.zeros(suitability_HRU.shape[0], dtype=bool)
            suitability_HRU[chunk_indices] = True

        land_use_type_before = hydrology.HRU.var.land_use_type.copy()

        forest_mask = hydrology.HRU.var.land_use_type == FOREST
        for prop in (
            "water_content_saturated_m",
            "water_content_field_capacity_m",
            "water_content_wilting_point_m",
            "water_content_residual_m",
            "saturated_hydraulic_conductivity_m_per_s",
            "bubbling_pressure_cm",
            "lambda_pore_size_distribution",
            "solid_heat_capacity_J_per_m2_K",
        ):
            arr = getattr(hydrology.HRU.var, prop)
            forest_mean = arr[:, forest_mask].mean(axis=1)
            arr[:, suitability_HRU] = forest_mean[:, np.newaxis]

        water_sat = hydrology.HRU.var.water_content_saturated_m
        water_res = hydrology.HRU.var.water_content_residual_m
        wc = hydrology.HRU.var.water_content_m

        # Case 1: wc > new saturation — route excess to topwater (water conserved).
        excess = np.maximum(0.0, wc[:, suitability_HRU] - water_sat[:, suitability_HRU])
        wc[:, suitability_HRU] -= excess
        hydrology.HRU.var.topwater_m[suitability_HRU] += excess.sum(axis=0)

        # Case 2: wc < new residual — raise wc to residual, sourcing from topwater (water conserved).
        deficit = np.maximum(
            0.0, water_res[:, suitability_HRU] - wc[:, suitability_HRU]
        )
        wc[:, suitability_HRU] += deficit
        topwater = hydrology.HRU.var.topwater_m[suitability_HRU]
        drawn = np.minimum(deficit.sum(axis=0), topwater)
        hydrology.HRU.var.topwater_m[suitability_HRU] -= drawn

        self.remove_farmers_from_converted_forest_areas(suitability_HRU)

        # Explicitly mark all planted HRUs as FOREST so that future calls to
        # prepare_modified_soil_maps_for_forest can detect them via the
        # already_forest check and advance to the next increment.
        hydrology.HRU.var.land_use_type[suitability_HRU] = FOREST

        output_folder = self.model.output_folder / "forest_planting"
        output_folder.mkdir(parents=True, exist_ok=True)
        self._save_forest_planting_figure(
            land_use_type_before, suitability_HRU, output_folder, threshold
        )

    def _save_forest_planting_figure(
        self,
        land_use_type_before: np.ndarray,
        suitability_HRU: np.ndarray,
        output_folder: Path,
        threshold: float = 0.5,
    ) -> None:
        """Save a 4-panel reforestation scenario figure."""
        hydrology = self.model.hydrology
        catchment_gdf = read_geom(self.model.files["geom"]["mask"])

        bounds = catchment_gdf.total_bounds  # [minx, miny, maxx, maxy]
        extent = [
            bounds[0],
            bounds[2],
            bounds[1],
            bounds[3],
        ]  # [left, right, bottom, top]

        current_2d = hydrology.HRU.decompress(land_use_type_before.astype(np.float32))
        future_2d = hydrology.HRU.decompress(
            hydrology.HRU.var.land_use_type.astype(np.float32)
        )
        suitability_2d = hydrology.HRU.decompress(suitability_HRU.astype(np.float32))
        change_2d = (future_2d != current_2d).astype(np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        im1 = axes[0, 0].imshow(
            current_2d, cmap="tab20", interpolation="nearest", extent=extent
        )
        axes[0, 0].set_title("Current Land Cover")
        catchment_gdf.boundary.plot(
            ax=axes[0, 0], color="black", linewidth=2, alpha=0.8
        )
        fig.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(
            future_2d, cmap="tab20", interpolation="nearest", extent=extent
        )
        axes[0, 1].set_title("Future Land Cover (with Reforestation)")
        catchment_gdf.boundary.plot(
            ax=axes[0, 1], color="black", linewidth=2, alpha=0.8
        )
        fig.colorbar(im2, ax=axes[0, 1])

        im3 = axes[1, 0].imshow(
            suitability_2d,
            cmap="Greens",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
        )
        axes[1, 0].set_title(f"Reforestation Suitability ({threshold:.0%} threshold)")
        catchment_gdf.boundary.plot(
            ax=axes[1, 0], color="black", linewidth=2, alpha=0.8
        )
        cbar3 = fig.colorbar(im3, ax=axes[1, 0])
        cbar3.set_ticks([0, 1])
        cbar3.set_ticklabels(["Unsuitable", "Suitable"])

        im4 = axes[1, 1].imshow(
            change_2d,
            cmap="Reds",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
        )
        axes[1, 1].set_title("Converted Areas")
        catchment_gdf.boundary.plot(
            ax=axes[1, 1], color="black", linewidth=2, alpha=0.8
        )
        cbar4 = fig.colorbar(im4, ax=axes[1, 1])
        cbar4.set_ticks([0, 1])
        cbar4.set_ticklabels(["No Change", "Converted"])

        plt.suptitle("Reforestation Scenario Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            output_folder / "reforestation_scenario.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def remove_farmers_from_converted_forest_areas(
        self, suitability_HRU: np.ndarray
    ) -> None:
        """Remove farmers from HRUs that are suitable for reforestation.

        Args:
            suitability_HRU: Boolean array at HRU scale (True = suitable for forest).
        """
        if not hasattr(self.agents, "crop_farmers"):
            return

        crop_farmers = self.agents.crop_farmers
        converted_HRU_indices = np.where(suitability_HRU)[0]
        if len(converted_HRU_indices) == 0:
            return

        land_owners = crop_farmers.HRU.var.land_owners[converted_HRU_indices]
        farmer_indices = land_owners[land_owners != -1]
        if len(farmer_indices) == 0:
            print("No farmers found in suitable areas, none removed")
            return

        unique_farmer_indices = np.unique(farmer_indices)
        farmers_before = crop_farmers.n
        crop_farmers.remove_agents(
            farmer_indices=unique_farmer_indices,
            new_land_use_type=FOREST,
        )
        print(
            f"Farmers removed: {len(unique_farmer_indices):,} ({farmers_before:,} → {crop_farmers.n:,})"
        )

    def adaptation(self) -> None:
        """Decide whether adaptation is needed and apply appropriate adaptation measures.

        Checks if adaptation is enabled and if it is January 1st, then calculates EAD,
        equity, and ecosystem indicators. If any thresholds are crossed, applies the corresponding
        adaptation measures (building floodproofing, subsidies, or reforestation).
        """
        # something to specify that this should only run when adaptation is turned on in the config file
        # should this step be skipped during spinup?
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exits because adaptation is not (enabled) in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January

        # calculate the water risk, equity and ecosystem health for the current year (adaptation is enabled and it is january first)
        EAD_value = self.calculate_EAD()  # this is defined by the EAD
        equity_indicator_value = (
            self.calculate_equity_indicator()
        )  # this is defined by the exposure inequalty
        ecosystem_indicator_value = (
            self.calculate_ecosystem_indicator()
        )  # this is defined by the ecosystem indicator

        # Set the indicator value and the thresholds for the adaptation decision.
        # but they should be defined in such a way that they trigger adaptation when threshold is crossed

        thresholds = {
            # defines the threshold --> value in current situation, threshold
            "EAD": (EAD_value, self.config["adaptation"].get("EAD_threshold")),
            "equity_indicator": (
                equity_indicator_value,
                self.config["adaptation"].get("equity_indicator_threshold"),
            ),
            "ecosystem_indicator": (
                ecosystem_indicator_value,
                self.config["adaptation"].get("ecosystem_indicator_threshold"),
            ),
        }

        # if any of these the thresholds are crossed, we adapt the policies so a make a list of the thresholds that are crossed
        # skip indicators where the value or threshold is None (not yet implemented)
        triggered = [
            key
            for key, value in thresholds.items()
            if value[0] is not None and value[1] is not None and value[0] > value[1]
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

    def calculate_EAD(self) -> None | float:
        """Calculate the EAD for the current year.

        Returns:
         the expected annual damage in euros, which is calculated as the product of the probability of a  hazard occurring and the potential damage caused by that hazard.
        """
        # should also only be calculated if adaptation is turned on in the config file and it is the first of jan otherwise it is not needed
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return

        households = self.agents.households
        if not hasattr(households, "flood"):
            # households.flood() is not yet implemented; skip EAD calculation
            return None
        return_periods = households.return_periods

        total_damage_per_rp = np.zeros(len(return_periods), dtype=np.float64)
        for i, return_period in enumerate(return_periods):
            flood_map = households.flood_maps[return_period]
            total_damage_per_rp[i] = households.flood(
                flood_map
            )  # the flood function in household agent calculates the damages for all assets and land use types.

        # sort ascending by return period so exceedance probability is descending
        sorted_rp = np.argsort(return_periods)
        exceedance_probabilities = 1.0 / return_periods[sorted_rp]
        sorted_damages = total_damage_per_rp[sorted_rp]

        EAD = np.trapezoid(sorted_damages, x=exceedance_probabilities)
        return EAD

    def calculate_equity_indicator(self) -> None | float:
        # should also only be calculated if adaptation is turned on in the config file otherwise it is not needed
        """Calculate the equity for the current year.

        Returns:
         the equity indicator value.
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
        equity_index = 0.7  # this should become an actual calculation
        return equity_index

    def calculate_ecosystem_indicator(self) -> None | float:
        # should also only be calculated if adaptation is turned on in the config file otherwise it is not needed
        """Calculate the ecosystem health for the current year.

        Returns:
        the ecosystem indicator value.
        """
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return

        # assign values per landuse type (maybe store in dictionary) (what do we base this on, tbd find literature)
        from geb.hydrology.landcovers import (
            FOREST,
            GRASSLAND_LIKE,
            NON_PADDY_IRRIGATED,
            OPEN_WATER,
            PADDY_IRRIGATED,
            SEALED,
        )

        values_per_land_use_type = {
            FOREST: 1.0,  # highest ecosystem quality
            GRASSLAND_LIKE: 0.7,
            OPEN_WATER: 0.6,
            PADDY_IRRIGATED: 0.3,
            NON_PADDY_IRRIGATED: 0.3,
            SEALED: 0.0,  # lowest ecosystem quality
        }

        # land use type per HRU and also area
        land_use_type_per_HRU = self.model.hydrology.HRU.var.land_use_type
        area_per_HRU = self.model.hydrology.HRU.var.cell_area

        # multiply the values with the area of each land use type
        # sum these
        weighted_sum = sum(
            np.sum(area_per_HRU[land_use_type_per_HRU == land_use_type]) * value
            for land_use_type, value in values_per_land_use_type.items()
        )

        # divide by total catchment area
        total_catchment_area = area_per_HRU.sum()
        ecosystem_indicator = weighted_sum / total_catchment_area

        return ecosystem_indicator

    def apply_adaptation(self, triggered: list) -> None:
        """Apply the adaptation measures decided in the adaptation function.

        Args:
            triggered: List of adaptation triggers that were activated.
        """
        if "adaptation" not in self.config or not self.config["adaptation"].get(
            "enabled", True
        ):
            return  # exit because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January

        if "EAD" in triggered:
            # the government decides which measure to apply based on what triggered the need for adaptation
            # apply updating the building structure but this takes the number of households that are adapting as input so we fist need to define that
            adaptation_fraction = self.config[
                "adaptation"
            ].get(
                "adaptation_fraction", 0.1
            )  # default to 10% of households adapting, otherwise specified in config file
            # figure out which buildings are marked as flooded in this year
            household_agents = self.agents.households
            flooded_buildings_mask = set(
                household_agents.buildings.loc[
                    household_agents.buildings["flooded"] == True, "id"
                ].astype(int)
            )
            # figure out which households are in these flooded buildings
            households_in_flooded_buildings = np.where(
                pd.Series(household_agents.var.building_id_of_household.data).isin(
                    flooded_buildings_mask
                )
            )[0]
            # the government decides who of those are adapting, we only pick a fraction as specified in the config file
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

        # if "equity_indicator" in triggered:
        #     # apply subsidies" --> maybe we can change who the subsidies are applied to but idk if that would make it better
        #     # this piece of code is still in Veerle's branch so can be uncommented when merged
        #     self.provide_subsidies()

        if "ecosystem_indicator" in triggered:
            # apply reforestation --> to do: but maybe a percentage of the suitable areas instead of all suitable areas
            # the threshold for the reforestation amount can be set in the config file under forest_reforestation_potential_threshold, set to 0.9 to convert only the top 10% suitable areas in the model.yml
            self.prepare_modified_soil_maps_for_forest()
