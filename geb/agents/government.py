"""This module contains the Government agent class for GEB."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geb.hydrology.landcovers import FOREST
from geb.workflows.io import read_geom, read_zarr

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
        adaptation_enabled = self.config["adaptation"]["enabled"]
        if (
            self.model.current_timestep == 0
            and self.config["plant_forest"]
            and not adaptation_enabled
        ):
            self.prepare_modified_soil_maps_for_forest()

        self.set_irrigation_limit()

        self.adaptation()

        if (
            self.model.current_time == self.model.run_start and adaptation_enabled
        ):  # plot the sigmoids that will be used ot make the adaptation decision for the run.
            steepness = self.config["sigmoid_parameters"]["steepness"]
            infliction_point = self.config["sigmoid_parameters"]["infliction_point"]
            self.plot_sigmoid(steepness, infliction_point)

        if self.model.current_time == self.model.run_end and adaptation_enabled:
            self.plot_indicators()
            self.plot_normalised_indicators()
            self.plot_adaptation_pathway()

        self.report(locals())

    def prepare_modified_soil_maps_for_forest(self) -> float:
        """Plant forest: update soil properties in memory and remove displaced farmers.

        Loads the forest restoration potential at grid scale, applies a threshold
        to identify suitable HRUs, copies mean soil property values from existing forest
        HRUs to suitable HRUs, saves a figure, and removes farmers from converted areas.
        The threshold is read from the config key ``forest_restoration_potential_threshold``
        and defaults to 0.5.

        When adaptation is enabled, the model automatically determines how much area to plant based on the available budget and reforestation costs. Suitable HRUs are sorted by potential
        value (highest first); those already classified as FOREST (from previous years
        or the initial state) are skipped, and HRUs are planted until there is no more affordable are for that year. Calling the function again
        the following year therefore plants the next batch automatically — compatible
        with the annual adaptation pathway that calls this on every January 1st.

        Returns:
            converted area in m2
        """
        hydrology = self.model.hydrology
        plant_forest_config = self.config["plant_forest"]
        if isinstance(plant_forest_config, dict):
            threshold = plant_forest_config.get(
                "forest_restoration_potential_threshold", 0.5
            )
        else:
            threshold = 0.5

        forest_potential = hydrology.grid.load2d(
            self.model.files["grid"]["landsurface/forest_restoration_potential_ratio"]
        )
        suitability_grid = forest_potential >= threshold
        suitability_HRU = hydrology.to_HRU(data=suitability_grid).astype(bool)

        # compute suitable area so we can use that for the incremental fraction --> based on the suitable area divided by the affordable area
        area_per_hru_m2: np.ndarray = hydrology.HRU.var.cell_area
        suitable_area_m2 = float(area_per_hru_m2[suitability_HRU].sum())

        adaptation_enabled = self.config.get("adaptation", {}).get("enabled", False)

        # Only use budget-driven incremental planting when adaptation is enabled.
        # Otherwise, keep legacy "all at once" behavior by setting None.
        incremental_planting: bool = False
        affordable_area_m2: float = 0.0
        if adaptation_enabled:
            incremental_planting = True
            available_budget = self.config["adaptation_costs"].get("available_budget")
            reforestation_cost_per_m2 = self.config["adaptation_costs"].get(
                "reforestation_cost_per_m2"
            )

            # if this is all defined we can calculate the affordable area.
            if suitable_area_m2 > 0 and reforestation_cost_per_m2:
                affordable_area_m2 = available_budget / reforestation_cost_per_m2

            print(f"Suitable area for reforestation: {suitable_area_m2:.2f} m2")
            print(f"Affordable area for reforestation: {affordable_area_m2:.2f} m2")

        if incremental_planting:
            # Sort suitable HRUs by potential value descending (best areas first).
            # Skip any already classified as FOREST (planted in prior years or
            # originally forest), then take the next chunk
            forest_potential_HRU = hydrology.to_HRU(data=forest_potential)
            suitable_indices = np.where(suitability_HRU)[0]
            suitable_potentials = forest_potential_HRU[suitable_indices]
            sorted_indices = suitable_indices[np.argsort(suitable_potentials)[::-1]]

            n_suitable = len(sorted_indices)
            if n_suitable == 0:
                self.model.logger.warning(
                    "Incremental reforestation: no suitable HRUs found. No planting applied."
                )
                return

            already_forest = hydrology.HRU.var.land_use_type == FOREST
            remaining = sorted_indices[~already_forest[sorted_indices]]

            if len(remaining) == 0:
                self.model.logger.warning(
                    "Incremental reforestation: all %d suitable HRUs are already "
                    "forest. No planting applied.",
                    n_suitable,
                )
                return 0.0

            # chunk is incremental_fraction of what remains, not 10% of the fixed total
            remaining_area_m2 = area_per_hru_m2[remaining]
            cumulative_area_m2 = np.cumsum(remaining_area_m2)

            n_to_convert = int(
                np.searchsorted(cumulative_area_m2, affordable_area_m2, side="right")
            )

            if n_to_convert == 0:
                self.model.logger.info(
                    "Incremental reforestation: budget allows 0 HRUs this year "
                    "(available area %.2f m2).",
                    affordable_area_m2,
                )
                return 0.0

            chunk_indices = remaining[:n_to_convert]
            n_already = n_suitable - len(remaining)
            self.model.logger.info(
                "Incremental reforestation: planting %d HRUs (rank %d–%d of %d "
                "suitable; %d already forest, %d remaining; %.2f m2 converted).",
                len(chunk_indices),
                n_already,
                n_already + len(chunk_indices) - 1,
                n_suitable,
                n_already,
                len(remaining),
                float(area_per_hru_m2[chunk_indices].sum()),
            )
            suitability_HRU = np.zeros(suitability_HRU.shape[0], dtype=bool)
            suitability_HRU[chunk_indices] = True
        else:
            self.model.logger.info(
                "Reforestation (all at once): planting %d suitable HRUs "
                "(threshold %.2f).",
                int(suitability_HRU.sum()),
                threshold,
            )

        land_use_type_before = hydrology.HRU.var.land_use_type.copy()

        converted_area_m2: float = float(area_per_hru_m2[suitability_HRU].sum())

        forest_mask = hydrology.HRU.var.land_use_type == FOREST
        for prop in (
            "water_content_saturated_m",
            "water_content_field_capacity_m",
            "water_content_wilting_point_m",
            "water_content_residual_m",
            "saturated_hydraulic_conductivity_m_per_s",
            "bubbling_pressure_m_positive",
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

        return converted_area_m2

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
        equity, and ecosystem indicators. The raw values for the indicators are normalised to ensure consistensy in their interpretation, which is needed for the sigmoid function.
        The sigmoid calculates the probability of adaptation, which is consequently combined with the weigth, that indicate government priorities, to generate an urgency score.
        The indicator with the highest urgency score gets the adaptation decision, this is passed to the apply adaptation function.
        """
        if not self.config["adaptation"].get("enabled", True):
            return  # exits because adaptation is not (enabled) in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January
        if self.model.in_spinup:
            return  # exits because the model is in spinup, we do not want adaptation during spinup

        # calculate the EAD, equity indicator value and ecosystem indicator value for the past year, before the adaptation decisions for this year are made.
        EAD_value = self.calculate_EAD()  # this is defined by the EAD
        equity_indicator_value = (
            self.calculate_equity_indicator()
        )  # this is defined by the exposure inequalty
        ecosystem_indicator_value = (
            self.calculate_ecosystem_indicator()
        )  # this is defined by the ecosystem indicator

        # normalise the values so we can use them in the sigmoid function. 1.0 means at threshold, above 1.0 means beyond threshold (urgent), below 1.0 means acceptable state.
        # EAD: value/threshold, at threshold = 1.0
        normalised_EAD_value = EAD_value / self.config["adaptation"]["EAD_threshold"]

        # equity: value/threshold, at threshold = 1.0, above = worse
        normalised_equity_value = (
            equity_indicator_value
            / self.config["adaptation"]["equity_indicator_threshold"]
        )

        # ecosystem: threshold/value, at threshold = 1.0, below threshold = worse (value < threshold means bad)
        normalised_ecosystem_value = (
            self.config["adaptation"]["ecosystem_indicator_threshold"]
            / ecosystem_indicator_value
        )

        # what adaptation measure should be implemented depends on what indicator most urgently needs adaptation as defined by
        # the probability of investment for the indicator value as calculated with a sigmoid function
        # the likelihood of adaptation is between 0 and 1, combine with weighting factor for each indicator to determine which indicator is the most urgent.

        steepness = self.config["sigmoid_parameters"]["steepness"]
        infliction_point = self.config["sigmoid_parameters"]["infliction_point"]

        EAD_sigmoid = self.sigmoid(steepness, normalised_EAD_value, infliction_point)
        equity_sigmoid = self.sigmoid(
            steepness, normalised_equity_value, infliction_point
        )
        ecosystem_sigmoid = self.sigmoid(
            steepness, normalised_ecosystem_value, infliction_point
        )

        EAD_urgency = self.config["indicator_weights"].get("EAD_weight") * EAD_sigmoid
        equity_urgency = (
            self.config["indicator_weights"].get("equity_weight") * equity_sigmoid
        )
        ecosystem_urgency = (
            self.config["indicator_weights"].get("ecosystem_weight") * ecosystem_sigmoid
        )

        # urgency score for decision making
        urgency_per_indicator = {
            "EAD": EAD_urgency,
            "equity_indicator": equity_urgency,
            "ecosystem_indicator": ecosystem_urgency,
        }

        # flip the sigmoid score so that higher score means more performance (and thus less need for adaptation) and lower score means worse performance (and thus more need for adaptation).
        EAD_performance = self.config["indicator_weights"]["EAD_weight"] - EAD_urgency
        equity_performance = (
            self.config["indicator_weights"]["equity_weight"] - equity_urgency
        )
        ecosystem_performance = (
            self.config["indicator_weights"]["ecosystem_weight"] - ecosystem_urgency
        )

        # performance score for logging and plotting in evaluation of pathway
        score_per_indicator = {
            "EAD": EAD_performance,
            "equity_indicator": equity_performance,
            "ecosystem_indicator": ecosystem_performance,
        }

        # save the values for each year of the model run to a csv file so that we can plot the values over time later on
        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        csv_file = output_folder / "adaptation_indicators_timeseries.csv"
        # Overwrite (not append) on the first Jan 1 of the run so that re-running the
        # model does not accumulate duplicate rows from previous runs.
        is_first_write = self.model.current_time.year == self.model.run_start.year
        file_mode = "w" if is_first_write else "a"
        write_header = is_first_write

        with open(csv_file, file_mode, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "year",
                    "EAD",
                    "equity_indicator",
                    "ecosystem_indicator",
                    "normalised_EAD",
                    "normalised_equity",
                    "normalised_ecosystem",
                    "EAD_urgency",
                    "equity_urgency",
                    "ecosystem_urgency",
                    "EAD_performance",
                    "equity_performance",
                    "ecosystem_performance",
                ],
            )
            if write_header:
                writer.writeheader()

            writer.writerow(
                {
                    "year": self.model.current_time.year,
                    "EAD": EAD_value,
                    "equity_indicator": equity_indicator_value,
                    "ecosystem_indicator": ecosystem_indicator_value,
                    "normalised_EAD": normalised_EAD_value,
                    "normalised_equity": normalised_equity_value,
                    "normalised_ecosystem": normalised_ecosystem_value,
                    "EAD_urgency": EAD_urgency,
                    "equity_urgency": equity_urgency,
                    "ecosystem_urgency": ecosystem_urgency,
                    "EAD_performance": EAD_performance,
                    "equity_performance": equity_performance,
                    "ecosystem_performance": ecosystem_performance,
                }
            )

        indicator_to_adapt = max(urgency_per_indicator, key=urgency_per_indicator.get)
        print("Indicator scores (weight * sigmoid): ")
        for indicator, score in urgency_per_indicator.items():
            print(f"  {indicator}: {score}")
        print(
            f"Government adaptation decision: {indicator_to_adapt} has the highest urgency score, applying adaptation for this indicator."
        )
        self.save_implemented_adaptation_measures(indicator_to_adapt)
        self.calculate_adaptation_effectiveness()
        self.apply_adaptation(indicator_to_adapt)

    def calculate_EAD(self) -> None | float:
        """Calculate the expected annual damage (EAD) for the current year.

        EAD is computed by integrating total flood damage over the exceedance
        probability curve (trapezoid rule across return periods).

        Returns:
         the expected annual damage in euros, which is calculated as the product of the probability of a hazard occurring and the potential damage caused by that hazard.

        Raises:
            RuntimeError: If the household flood risk module is not available.
        """
        households = self.agents.households
        return_periods = np.array(
            self.model.config["hazards"]["floods"]["return_periods"], dtype=float
        )

        if not hasattr(households, "flood_risk_module"):
            raise RuntimeError("Household flood risk module is not available.")

        fr = households.flood_risk_module
        # Ensure damage curves and maximum-damage values are loaded (idempotent)
        fr.load_damage_curves()
        fr.load_max_damage_values()
        # Load flood maps so downstream code can sample them
        fr.load_flood_maps()

        # Ensure the `flooded` building attribute exists. If not, compute it
        # using the households helper that populates building attributes.
        if "flooded" not in households.buildings.columns:
            if hasattr(households, "update_building_attributes"):
                households.update_building_attributes()
            else:
                self.model.logger.warning(
                    "Missing 'flooded' column and no update_building_attributes() available."
                )

        total_damage_per_rp = np.zeros(len(return_periods), dtype=np.float64)
        flood_maps = {}
        for i, return_period in enumerate(return_periods):
            file_path = (
                self.model.output_folder / "flood_maps" / f"{int(return_period)}.zarr"
            )
            flood_map = read_zarr(file_path)
            total_damage_per_rp[i] = households.flood_risk_module.flood(flood_map)

        exceedance_probabilities = 1.0 / return_periods

        sort_idx = np.argsort(exceedance_probabilities)

        EAD = np.trapezoid(
            total_damage_per_rp[sort_idx], x=exceedance_probabilities[sort_idx]
        )
        print(f"Calculated EAD: €{EAD}")
        return EAD

    def calculate_equity_indicator(self) -> None | float:
        """Calculate the equity for the current year.

        Returns:
         the equity indicator value.
        """
        # this is defined by the exposure inequalty
        households = self.agents.households

        # we define the low-income households based on EU standard definition of at-risk-of-poverty threshold, which is 60% of the median income (https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:At-risk-of-poverty_threshold)
        # as done in the UK (https://www.gov.uk/government/publications/how-low-income-is-measured/text-only-how-low-income-is-measured)
        # netherlands also (https://www.cbs.nl/en-gb/visualisations/monitor-of-wellbeing-caribbean-netherlands/indicator-descriptions)
        median_income = np.median(households.var.income.data)
        income_threshold = 0.6 * median_income
        low_income_households_mask = households.var.income.data <= income_threshold
        population_low_income_households = low_income_households_mask.sum()
        population_all_households = households.n

        population_share = population_low_income_households / population_all_households

        if "flooded" not in households.buildings.columns:
            households.update_building_attributes()

        flooded_buildings_mask = set(
            households.buildings.loc[
                households.buildings["flooded"] == True, "id"
            ].astype(int)
        )
        # figure out which households are in these flooded buildings
        households_in_flooded_buildings = np.where(
            pd.Series(households.var.building_id_of_household.data).isin(
                flooded_buildings_mask
            )
        )[0]
        exposure_all_households = len(households_in_flooded_buildings)

        exposure_low_income_households = np.intersect1d(
            households_in_flooded_buildings, np.where(low_income_households_mask)[0]
        ).size

        exposure_share = exposure_low_income_households / exposure_all_households

        equity_indicator = exposure_share / population_share

        print(f"Calculated equity indicator: {equity_indicator}")
        return equity_indicator

    def calculate_ecosystem_indicator(self) -> None | float:
        """Calculate the ecosystem health for the current year.

        Returns:
        the ecosystem indicator value.
        """
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
            FOREST: 0.9,  # highest ecosystem quality
            GRASSLAND_LIKE: 0.75,
            OPEN_WATER: 0.3,
            PADDY_IRRIGATED: 0.0,
            NON_PADDY_IRRIGATED: 0.3,
            SEALED: 0.025,  # lowest ecosystem quality
        }

        # land use type and area per HRU
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

        print(f"Calculated ecosystem indicator: {ecosystem_indicator}")
        return ecosystem_indicator

    def apply_adaptation(self, indicator_to_adapt) -> None:
        """Apply the adaptation measures decided in the adaptation function.

        Args:
            indicator_to_adapt: the indicator for which to apply adaptation measures.
        """
        # Only one adaptation measure is applied per year, this is the indicator with the
        # highest score as calculated by the sigmoid function multiplied by the weight for that indicator.
        available_budget = self.config["adaptation_costs"].get("available_budget")
        floodproofing_cost_per_household = self.config["adaptation_costs"].get(
            "floodproofing_cost_per_household"
        )
        reforestation_cost_per_m2 = self.config["adaptation_costs"].get(
            "reforestation_cost_per_m2"
        )
        subsidies_cost_per_household = self.config["adaptation_costs"].get(
            "subsidies_cost_per_household"
        )

        if indicator_to_adapt == "EAD":
            # the government decides which measure to apply based on what triggered the need for adaptation
            # apply updating the building structure but this takes the number of households that are adapting as input so we fist need to define that
            household_adaptation_fraction = self.config["adaptation"].get(
                "household_adaptation_fraction"
            )  # specified in config file
            # figure out which buildings are marked as flooded in this year
            households = self.agents.households

            if "flooded" not in households.buildings.columns:
                households.update_building_attributes()

            flooded_buildings_mask = set(
                households.buildings.loc[
                    households.buildings["flooded"] == True, "id"
                ].astype(int)
            )
            # figure out which households are in these flooded buildings
            households_in_flooded_buildings = np.where(
                pd.Series(households.var.building_id_of_household.data).isin(
                    flooded_buildings_mask
                )
            )[0]

            # Households that already adapted (dry-floodproofed) should not be
            # selected again for government-supported adaptation in later years.
            eligible_households = households_in_flooded_buildings[
                households.var.adapted.data[households_in_flooded_buildings] == 0
            ]

            if len(eligible_households) == 0:
                print(
                    "No eligible households for government floodproofing "
                    "(all flooded households already adapted)."
                )
                return

            # the government decides who of those are adapting, we only pick a fraction as specified in the config file
            # the number to adapt should actually be based on the available budget, but if the eligible households are less than budget allows, we can only adapt that numnber.
            potential_to_adapt = int(
                available_budget / floodproofing_cost_per_household
            )
            n_to_adapt = min(potential_to_adapt, len(eligible_households))
            if n_to_adapt == 0:
                print("No households selected for government floodproofing this year ")
                return
            # randomly select the households that are adapted based on the number of households that can be adapted.
            adapting_households_sample = np.random.choice(
                eligible_households, size=n_to_adapt, replace=False
            )
            # update the households that are adapted so they are not eligible for adaptation again.
            households.var.adapted[adapting_households_sample] = 1

            # use the function to floodproof the buildings of the households who are selected to adapt.
            households.update_building_adaptation_status(adapting_households_sample)
            # fr = households.flood_risk_module do we need to do this? Maybe now we only update the status of the building but we dont actually lower the damage curve for these buildings.
            # fr.alter_damage_curves_for_flood_proofed_buildings()
            print(
                f"the government adapted {n_to_adapt} of the "
                f"{len(eligible_households)} eligible households in the floodzone by floodproofing their buildings"
            )

        # if indicator_to_adapt == "equity_indicator":
        #     # apply subsidies" --> maybe we can change who the subsidies are applied to but idk if that would make it better
        #     # this piece of code is still in Veerle's branch so can be uncommented when merged
        #      print ("the government adapted by providing subsidies to the most vulnerable households to reduce inequality")
        #     self.provide_subsidies()

        if indicator_to_adapt == "ecosystem_indicator":
            # apply reforestation, the reforestation limited by budget is already implemented in the prepare_modified_soil_maps_for_forest_function.
            converted_area_m2 = self.prepare_modified_soil_maps_for_forest()
            print(
                f"the government adapted by planting {converted_area_m2} m2 of forest in the most suitable areas to improve the ecosystem health"
            )

    def plot_indicators(self):
        """Plot the evaluation criteria over the run time and calculate the final score of the government's adaptation strategy."""
        # plot the evaluation criteria over time
        csv_file = (
            self.model.output_folder
            / "adaptation"
            / "adaptation_indicators_timeseries.csv"
        )
        df = pd.read_csv(csv_file)

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle("evaluation criteria over time")

        ax1.plot(df["year"], df["EAD"])
        ax1.set_ylabel("EAD (millions of €)")
        ax1.set_xlabel("Year")
        ax1.set_title("Expected Annual Damage (€) over time")

        ax2.plot(df["year"], df["equity_indicator"])
        ax2.set_ylabel("Equity Indicator")
        ax2.set_xlabel("Year")
        ax2.set_title("Equity Indicator over time")

        ax3.plot(df["year"], df["ecosystem_indicator"])
        ax3.set_ylabel("Ecosystem Indicator")
        ax3.set_xlabel("Year")
        ax3.set_title("Ecosystem Indicator over time")

        fig.tight_layout()
        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_folder / "adaptation_indicators_over_time.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def plot_normalised_indicators(self):
        """Plot the normalised evaluation criteria over the run time and calculate the final score of the government's adaptation strategy."""
        # plot the normalised evaluation criteria over time
        csv_file = (
            self.model.output_folder
            / "adaptation"
            / "adaptation_indicators_timeseries.csv"
        )
        df = pd.read_csv(csv_file)

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle("normalised evaluation criteria over time")

        ax1.plot(df["year"], df["normalised_EAD"])
        ax1.set_ylabel("Normalised EAD (€)")
        ax1.set_xlabel("Year")
        ax1.set_title("Normalised Expected Annual Damage (€) over time")

        ax2.plot(df["year"], df["normalised_equity"])
        ax2.set_ylabel("Normalised Equity Indicator")
        ax2.set_xlabel("Year")
        ax2.set_title("Normalised Equity Indicator over time")

        ax3.plot(df["year"], df["normalised_ecosystem"])
        ax3.set_ylabel("Normalised Ecosystem Indicator")
        ax3.set_xlabel("Year")
        ax3.set_title("Normalised Ecosystem Indicator over time")

        fig.tight_layout()
        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_folder / "normalised_adaptation_indicators_over_time.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def calculate_adaptation_effectiveness(self):
        # the last value of the indicators in the csv file is the value after the adaptation measures have been implemented, so we can use these values to evaluate the effectiveness of the adaptation measures by comparing them with the first value of the indicators in the csv file, which is the value before the adaptation measures were implemented.
        csv_file = (
            self.model.output_folder
            / "adaptation"
            / "adaptation_indicators_timeseries.csv"
        )
        df = pd.read_csv(csv_file)

        initial_score = (  # these are calculated by multiplying the probability of adapting based on the sigmoid with the weight for that indicator.
            df["EAD_performance"].iloc[0]
            + df["equity_performance"].iloc[0]
            + df["ecosystem_performance"].iloc[0]
        )

        final_score = (
            df["EAD_performance"].iloc[-1]
            + df["equity_performance"].iloc[-1]
            + df["ecosystem_performance"].iloc[-1]
        )

        if len(df) < 2:
            improvement_achieved = 0.0
        else:
            improvement_achieved = final_score - (
                df["EAD_performance"].iloc[-2]
                + df["equity_performance"].iloc[-2]
                + df["ecosystem_performance"].iloc[-2]
            )

        # save the values for each year of the model run to a csv file so that we can plot the values over time later on
        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        csv_file = output_folder / "adaptation_scoring.csv"

        # Overwrite (not append) on the first Jan 1 of the run so that re-running the
        # model does not accumulate duplicate rows from previous runs.
        is_first_write = self.model.current_time.year == self.model.run_start.year
        file_mode = "w" if is_first_write else "a"
        write_header = is_first_write

        with open(csv_file, file_mode, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "year",
                    "initial_score",
                    "final_score",
                    "improvement_achieved",
                ],
            )
            if write_header:
                writer.writeheader()

            writer.writerow(
                {
                    "year": self.model.current_time.year,
                    "initial_score": initial_score,
                    "final_score": final_score,
                    "improvement_achieved": improvement_achieved,
                }
            )

    def sigmoid(self, steepness, indicator_value, infliction_point):
        """Calculate the sigmoid function value for a given indicator value.

        Values for steepness and infliction point are defined in the config file.

        Args:
            steepness: Steepness of the curve, where higher values
                represent more rigid government decisions.
            indicator_value: Current indicator value.
            infliction_point: Infliction point of the curve, where
                lower values represent a more proactive government.

        Returns:
            Sigmoid value.
        """
        sigmoid_value = 1 / (
            1 + np.exp(-steepness * (indicator_value - infliction_point))
        )
        return sigmoid_value

    def plot_sigmoid(self, steepness, infliction_point):
        """Plot the sigmoid function for a given steepness and infliction point, different per indicator."""
        x = np.linspace(-1, 2, 100)
        y = self.sigmoid(steepness, x, infliction_point)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.title(
            f"Sigmoid Function for decision-making (steepness={steepness}, infliction_point={infliction_point})"
        )
        plt.xlabel("Normalised Indicator Value")
        plt.ylabel("Adaptation Probability")
        plt.grid()
        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_folder / f"sigmoid_{steepness}_{infliction_point}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def save_implemented_adaptation_measures(self, indicator_to_adapt):
        """save the implemented adaptation measures in a csv file so that we can plot the adaptation pathway later on."""
        indicator_to_measure = {
            "EAD": "grey",
            "equity_indicator": "institutional",
            "ecosystem_indicator": "green",
        }

        measure_type = indicator_to_measure.get(indicator_to_adapt, "unknown")

        # save the values for each year of the model run to a csv file so that we can plot the values over time later on
        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        csv_file = output_folder / "adaptation_measures_implemented.csv"

        # Overwrite (not append) on the first Jan 1 of the run so that re-running the
        # model does not accumulate duplicate rows from previous runs.
        is_first_write = self.model.current_time.year == self.model.run_start.year
        file_mode = "w" if is_first_write else "a"
        write_header = is_first_write

        with open(csv_file, file_mode, newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["year", "indicator_to_adapt", "measure_type"]
            )
            if write_header:
                writer.writeheader()

            writer.writerow(
                {
                    "year": self.model.current_time.year,
                    "indicator_to_adapt": indicator_to_adapt,
                    "measure_type": measure_type,
                }
            )

    def plot_adaptation_pathway(self):
        """plot the adaptation pathway, showing the adaptation measures that are implemented over time."""
        csv_file_measures = (
            self.model.output_folder
            / "adaptation"
            / "adaptation_measures_implemented.csv"
        )
        df_measures = pd.read_csv(csv_file_measures)

        csv_file_scores = (
            self.model.output_folder / "adaptation" / "adaptation_scoring.csv"
        )
        df_scores = pd.read_csv(csv_file_scores)

        # merge on year so measures and scores are aligned
        df = pd.merge(df_measures, df_scores, on="year")

        colour_map = {
            "grey": "#5F5E5A",
            "green": "#3B8B2E",
            "institutional": "#C0392B",
        }

        fig, ax = plt.subplots(figsize=(10, 5))

        # plot each segment between adjacent years, coloured by measure type
        for i in range(len(df) - 1):
            x_seg = [df["year"].iloc[i], df["year"].iloc[i + 1]]
            y_seg = [df["final_score"].iloc[i], df["final_score"].iloc[i + 1]]
            colour = colour_map.get(df["measure_type"].iloc[i], "#aaaaaa")
            ax.plot(x_seg, y_seg, color=colour, linewidth=2.5)

        # add dots at each year coloured by measure
        for _, row in df.iterrows():
            colour = colour_map.get(row["measure_type"], "#aaaaaa")
            ax.scatter(row["year"], row["final_score"], color=colour, s=40, zorder=5)

        # legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="#5F5E5A",
                linewidth=2.5,
                label="Grey (hard infrastructure)",
            ),
            Line2D(
                [0], [0], color="#3B8B2E", linewidth=2.5, label="Green (nature-based)"
            ),
            Line2D(
                [0], [0], color="#C0392B", linewidth=2.5, label="Institutional (equity)"
            ),
        ]
        ax.legend(handles=legend_elements, fontsize=10, framealpha=0.3)

        ax.set_title("Adaptation pathway", fontsize=13)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Performance score", fontsize=11)

        # Auto-scale y-axis to show actual variation in scores rather than fixed 0–1 range.
        # This makes small improvements visible without distorting the data.
        score_min = df["final_score"].min()
        score_max = df["final_score"].max()
        score_range = score_max - score_min
        padding = max(score_range * 0.1, 0.01)  # 10% padding or 0.01 minimum
        ax.set_ylim(score_min - padding, score_max + padding)

        ax.grid(True, alpha=0.3)

        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_folder / "adaptation_pathway.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
