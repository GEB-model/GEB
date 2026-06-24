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

        if self.model.current_time == self.model.run_end and adaptation_enabled:
            self.plot_indicators()
            self.plot_adaptation_pathway()

        self.report(locals())

    def prepare_modified_soil_maps_for_forest(self) -> float:
        """Plant forest: update soil properties in memory and remove displaced farmers.

        Loads the forest restoration potential at grid scale, applies a threshold
        to identify suitable HRUs, copies mean soil property values from existing forest
        HRUs to suitable HRUs, saves a figure, and removes farmers from converted areas.
        The threshold is read from the config key ``forest_restoration_potential_threshold``
        and defaults to 0.5.

        When adaptation is enabled, the model automatically determines how much area to plant based on the available budget and reforestation costs. Suitable HRUs are sorted by area ascending, so cheapest HRUs to covert are chosen first; those already classified as FOREST (from previous years
        or the initial state) are skipped, and HRUs are planted until there is no more affordable area for that year. Calling the function again
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
        suitable_HRU = hydrology.to_HRU(data=suitability_grid).astype(bool)

        area_per_hru_m2: np.ndarray = hydrology.HRU.var.cell_area
        suitable_area_m2 = float(area_per_hru_m2[suitable_HRU].sum())

        adaptation_enabled = self.config.get("adaptation", {}).get("enabled", False)

        # select final_HRU (budget driven or all at once)

        if adaptation_enabled:
            reforestation_cost_per_m2 = self.config["adaptation_costs"].get(
                "reforestation_cost_per_m2"
            )
            available_budget = self.config["adaptation_costs"].get("initial_budget")
            affordable_area_m2 = (
                available_budget / reforestation_cost_per_m2
                if (suitable_area_m2 > 0 and reforestation_cost_per_m2)
                else 0.0
            )

            print(f"Suitable area for reforestation: {suitable_area_m2:.2f} m2")
            print(f"Affordable area for reforestation: {affordable_area_m2:.2f} m2")

            # incremental planting when adaptation is enabled.

            suitable_indices = np.where(suitable_HRU)[0]

            if len(suitable_indices) == 0:
                self.model.logger.warning(
                    "Incremental reforestation: no suitable HRUs found."
                )
                return 0.0

            already_forest = hydrology.HRU.var.land_use_type == FOREST
            remaining = suitable_indices[~already_forest[suitable_indices]]

            if len(remaining) == 0:
                self.model.logger.warning(
                    "Incremental reforestation: all suitable HRUs already forest."
                )
                return 0.0

            # sort ascending by area so the most HRUs are converted, also ties in with easiest options first.
            remaining = remaining[np.argsort(area_per_hru_m2[remaining])]

            remaining_budget = affordable_area_m2
            chunk_indices = []

            for idx in remaining:
                area = area_per_hru_m2[idx]
                if area <= remaining_budget:
                    chunk_indices.append(idx)
                    remaining_budget -= area

            if len(chunk_indices) == 0:
                self.model.logger.info(
                    "Incremental reforestation: budget insufficient."
                )
                return 0.0

            final_HRU = np.zeros_like(suitable_HRU, dtype=bool)
            final_HRU[chunk_indices] = True

            self.model.logger.info(
                "Incremental reforestation: planting %d HRUs (%.2f m2).",
                len(chunk_indices),
                float(area_per_hru_m2[final_HRU].sum()),
            )

        else:
            final_HRU = suitable_HRU

            self.model.logger.info(
                "Reforestation (all at once): planting %d HRUs (threshold %.2f).",
                int(final_HRU.sum()),
                threshold,
            )

        # modification for both incremental and all at once reforestation
        land_use_type_before = hydrology.HRU.var.land_use_type.copy()

        converted_area_m2: float = float(area_per_hru_m2[final_HRU].sum())

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
            arr[:, final_HRU] = forest_mean[:, np.newaxis]

        water_sat = hydrology.HRU.var.water_content_saturated_m
        water_res = hydrology.HRU.var.water_content_residual_m
        wc = hydrology.HRU.var.water_content_m

        # Case 1: wc > new saturation — route excess to topwater (water conserved).
        excess = np.maximum(0.0, wc[:, final_HRU] - water_sat[:, final_HRU])
        wc[:, final_HRU] -= excess
        hydrology.HRU.var.topwater_m[final_HRU] += excess.sum(axis=0)

        # Case 2: wc < new residual — raise wc to residual, sourcing from topwater (water conserved).
        deficit = np.maximum(0.0, water_res[:, final_HRU] - wc[:, final_HRU])
        wc[:, final_HRU] += deficit
        topwater = hydrology.HRU.var.topwater_m[final_HRU]
        drawn = np.minimum(deficit.sum(axis=0), topwater)
        hydrology.HRU.var.topwater_m[final_HRU] -= drawn

        self.remove_farmers_from_converted_forest_areas(final_HRU)

        # Explicitly mark all planted HRUs as FOREST so that future calls to
        # prepare_modified_soil_maps_for_forest can detect them via the
        # already_forest check and advance to the next increment.
        hydrology.HRU.var.land_use_type[final_HRU] = FOREST

        output_folder = self.model.output_folder / "forest_planting"
        output_folder.mkdir(parents=True, exist_ok=True)
        self._save_forest_planting_figure(
            land_use_type_before, final_HRU, output_folder, threshold
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

    # decision making logic for adaptation as implemented by the government agent
    # this function is the basis from which other functions are called that together make up the logic for adaptation
    def adaptation(self) -> None:
        """From this function all steps for the adaptation implementation are called.

        Checks if adaptation is enabled and if it is January 1st, then calculates EAD,
        equity, and ecosystem indicators. Then a mental simulation of all possible adaptation measures is done to see which one has the biggest improvement
        based on the sum of the normalised indicators. This one is then implemented.

        """
        if not self.config["adaptation"].get("enabled", True):
            return  # exits because adaptation is not enabled in the config file
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January
        if self.model.in_spinup:
            return  # exits because the model is in spinup, we do not want adaptation during spinup

        budget = self.config["adaptation_costs"].get("initial_budget")

        EAD: float = self.calculate_risk_reduction_indicator()
        exposure_inequality: float = self.calculate_equity_indicator()
        ecosystem_health: float = self.calculate_ecosystem_indicator()

        current_IPV = self.calculate_IPV(EAD, exposure_inequality, ecosystem_health)

        # do a mental simulation of the improvement (based on indicators) the implementation of all measures would generate. This will help the government to decide which
        # adaptation measure should be implemented. The one with the biggest improvement will get implemented. We should save the model, then do a mental simulation
        # so a hypothetical implementation of all measures, then calculate the indicators again, and see which one has the biggest improvement compared to the current state. This is the one that gets implemented in reality.

        adaptation_measure_to_implement: str | None = self.hypothetical_implementation(
            budget, current_IPV
        )

        # implement the adaptation measure that is selected
        self.apply_adaptation(budget, adaptation_measure_to_implement)

        # we want to save these things yearly so we can plot them
        self.save_adaptation_record(
            EAD,
            exposure_inequality,
            ecosystem_health,
            current_IPV,
            adaptation_measure_to_implement,
        )

    # def available_budget(self) -> float:
    #     """Calculate the available budget for adaptation measures based on the configuration.

    #     Returns:
    #         The available budget for adaptation measures in euros, or None if not defined.
    #     """
    #     # increase the initial budget with correction for the amount of years that have passed
    #     budget = self.config["adaptation_costs"].get("initial_budget") * (
    #         (1 + (self.config["adaptation_costs"].get("budget_yearly_increase") / 100))
    #         ** (self.model.current_time.year - self.model.run_start.year)
    #     )

    #     return budget

    # made into a function so it can also be used in the hypothetical implementation
    def calculate_IPV(
        self, EAD: float, exposure_inequality: float, ecosystem_health: float
    ) -> float:
        """Calculate the Integrated Performance Score based on the sum of the normalised indicators.

        Returns:
            The IPV.
        """
        risk_reduction_weight: float = self.config["priority_weights"].get(
            "risk_reduction"
        )
        equity_weight: float = self.config["priority_weights"].get("equity")
        ecosystem_weight: float = self.config["priority_weights"].get(
            "ecosystem_health"
        )

        IPV = sum(
            [
                (risk_reduction_weight * EAD),
                (equity_weight * exposure_inequality),
                (ecosystem_weight * ecosystem_health),
            ]
        )
        return IPV

    def hypothetical_implementation(
        self, budget: float, current_IPV: float
    ) -> str | None:
        """Calculate the improvement in IPV for each potential adaptation measure.

        The adaptation measure that generates the biggest improvement in IPV during
        the hypothetical implementation will actually be implemented.

        Returns:
            The adaptation measure to implement.
        """
        # save the state
        # implement reforestation in the hypothetical simulation
        self.apply_adaptation(budget, "reforestation")
        hypothetical_EAD = self.calculate_risk_reduction_indicator()
        hypothetical_exposure_inequality = self.calculate_equity_indicator()
        hypothetical_ecosystem_health = self.calculate_ecosystem_indicator()

        delta_reforestation = (
            self.calculate_IPV(
                hypothetical_EAD,
                hypothetical_exposure_inequality,
                hypothetical_ecosystem_health,
            )
            - current_IPV
        )
        # reset the state

        # implement floodproofing and calculate the indicators and IPV, save the IPV
        self.apply_adaptation(budget, "floodproofing")
        hypothetical_EAD = self.calculate_risk_reduction_indicator()
        hypothetical_exposure_inequality = self.calculate_equity_indicator()
        hypothetical_ecosystem_health = self.calculate_ecosystem_indicator()
        delta_floodproofing = (
            self.calculate_IPV(
                hypothetical_EAD,
                hypothetical_exposure_inequality,
                hypothetical_ecosystem_health,
            )
            - current_IPV
        )
        # reset the state

        # implement subsidies/risk communication and calculate the indicators, save IPV
        self.apply_adaptation(budget, "risk_communication")
        hypothetical_EAD = self.calculate_risk_reduction_indicator()
        hypothetical_exposure_inequality = self.calculate_equity_indicator()
        hypothetical_ecosystem_health = self.calculate_ecosystem_indicator()
        delta_risk_communication = (
            self.calculate_IPV(
                hypothetical_EAD,
                hypothetical_exposure_inequality,
                hypothetical_ecosystem_health,
            )
            - current_IPV
        )

        # select the adaptation measure with the biggest improvement and make that the adaptation that is to be implemented
        deltas = {
            "reforestation": delta_reforestation,
            "floodproofing": delta_floodproofing,
            "risk_communication": delta_risk_communication,
        }
        adaptation_measure_to_implement = max(deltas, key=deltas.get)

        return adaptation_measure_to_implement

    def calculate_risk_reduction_indicator(self) -> float:
        """Calculate the expected annual damage (EAD) for the current year.

        EAD is computed by integrating total flood damage over the exceedance
        probability curve (trapezoid rule across return periods). The EAD is normalised using the EAD from the first timestep as this is the
        theoretical max, since the EAD will go down over the years as adaptation is inplemented.

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

        for i, return_period in enumerate(return_periods):
            file_path = (
                Path(self.model.config["general"]["output_folder"])
                / "flood_maps"
                / f"{int(return_period)}.zarr"
            )
            flood_map = read_zarr(file_path)
            total_damage_per_rp[i] = households.flood_risk_module.flood(flood_map)

        exceedance_probabilities = 1.0 / return_periods

        sort_idx = np.argsort(exceedance_probabilities)

        EAD = np.trapezoid(
            total_damage_per_rp[sort_idx], x=exceedance_probabilities[sort_idx]
        )

        # we also need to normalise the EAD and use it in the decision making process for adaptation. Finding the maxium damage could be difficult, so we can use max_damage = initial damage (store in cache) since adaptation will lower the ead so it is a defensible max
        if not hasattr(self, "max_EAD"):
            self.max_EAD = EAD

        normalised_EAD = EAD / self.max_EAD
        print(f"Calculated EAD: {normalised_EAD}")
        return normalised_EAD

    def calculate_equity_indicator(self) -> float:
        """Calculate the exposure inequality for the current year.

        Equity indicator: proportion of low-income households NOT exposed to flooding.
        1.0 = no low-income households flooded (best equity)
        0.0 = all low-income households flooded (worst equity)

        Returns:
            the equity indicator value.
        """
        # this is defined by the exposure inequalty
        households = self.agents.households

        # we define the low-income households based on EU standard definition of at-risk-of-poverty threshold, which is 60% of the median income (https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:At-risk-of-poverty_threshold)
        # as done in the UK (https://www.gov.uk/government/publications/how-low-income-is-measured/text-only-how-low-income-is-measured)
        # high income is 200% of the median income (https://ec.europa.eu/eurostat/documents/3888793/7882117/KS-TC-16-027-EN-N.pdf/42d637e3-1386-40e1-845c-9aadad4ad2a1)
        # netherlands also (https://www.cbs.nl/en-gb/visualisations/monitor-of-wellbeing-caribbean-netherlands/indicator-descriptions)
        median_income = np.median(households.var.income.data)
        low_income_threshold = 0.6 * median_income
        low_income_households_mask = households.var.income.data <= low_income_threshold
        high_income_threshold = 2 * median_income
        high_income_households_mask = (
            households.var.income.data >= high_income_threshold
        )

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

        # the low income households that are exposed to flooding are the ones that are in the flooded buildings and are low income
        exposure_low_income_households = np.intersect1d(
            households_in_flooded_buildings, np.where(low_income_households_mask)[0]
        ).size

        exposure_high_income_households = np.intersect1d(
            households_in_flooded_buildings, np.where(high_income_households_mask)[0]
        ).size

        exposure_share_low = exposure_low_income_households / len(
            households_in_flooded_buildings
        )
        exposure_share_high = exposure_high_income_households / len(
            households_in_flooded_buildings
        )

        exposure_inequality = exposure_share_low / exposure_share_high

        equity_indicator = 1 / (1 + (np.log(exposure_inequality)))
        print(f"Calculated equity indicator: {equity_indicator}")
        return equity_indicator

    def calculate_ecosystem_indicator(self) -> float:
        """Calculate the ecosystem health for the current year.

        Returns:
        the ecosystem indicator value.
        """
        hydrology = self.model.hydrology

        # load the dataset
        zpath = self.model.files["subgrid"]["landcover/classification"]

        values_per_esa_land_use_type = {
            10: 1.0,  # tree cover
            20: 0.26,  # shrubland
            30: 0.79,  # grassland
            40: 0.21,  # cropland
            50: 0.04,  # built-up
            60: 0.18,  # bare / sparse vegetation
            70: 0.37,  # snow and ice
            80: 0.32,  # permanent water bodies
            90: 0.5,  # herbaceous wetland
            95: 0.21,  # mangroves
            100: 0.58,  # moss and lichen
        }

        # extract the ESA landcover codes
        esa_values = read_zarr(zpath).values

        # create a same shape filled with zeroes to score the ESA values based on the land use type
        esa_scores = np.full_like(esa_values, fill_value=0.0, dtype=float)
        for land_use_type, score in values_per_esa_land_use_type.items():
            esa_scores[esa_values == land_use_type] = score

        # get the total number of HRUs
        n_hrus = hydrology.HRU.var.land_use_type.size

        # create an array of HRU indices
        hru_index_1d = np.arange(n_hrus, dtype=np.int32)

        # create a 2D array of HRU in the same shape as the esa_values
        hru_index_2d = hydrology.HRU.decompress(hru_index_1d)

        valid_mask = hru_index_2d >= 0

        # create a 1D array with only the valid cells
        flat_hrus = hru_index_2d[valid_mask].ravel()
        flat_esa_scores = esa_scores[valid_mask].ravel()

        # sum scores given to ESA land use typs per HRU
        sum_score_per_hru = np.bincount(
            flat_hrus, weights=flat_esa_scores, minlength=n_hrus
        )

        # calculate the amount of pixels per hru
        count_per_hru = np.bincount(flat_hrus, minlength=n_hrus)

        # compute mean score per HRU, filter out zeroes
        mean_score_per_hru = np.zeros(n_hrus, dtype=float)
        nonzero_mask = count_per_hru > 0
        mean_score_per_hru[nonzero_mask] = (
            sum_score_per_hru[nonzero_mask] / count_per_hru[nonzero_mask]
        )

        # land use type and area per HRU
        area_per_HRU = self.model.hydrology.HRU.var.cell_area.astype(float)

        # compute the area-weighted sum of HRU mean scores across the catchment
        weighted_sum = float((mean_score_per_hru * area_per_HRU).sum())

        # divide by total area to get the area-weighted mean score as the ecosystem indicator
        ecosystem_indicator = weighted_sum / (area_per_HRU.sum())
        print(f"Calculated ecosystem indicator: {ecosystem_indicator}")
        return ecosystem_indicator

    def apply_adaptation(
        self, budget: float, adaptation_measure_to_implement: str | None
    ) -> None:
        """Apply the adaptation measures decided in the adaptation function.

        Args:
            budget: the available budget for adaptation.
            adaptation_measure_to_implement: the adaptation measure to implement.
        """
        floodproofing_cost_per_household = self.config["adaptation_costs"].get(
            "floodproofing_cost_per_household"
        )

        subsidies_cost_per_household = self.config["adaptation_costs"].get(
            "subsidies_cost_per_household"
        )

        if adaptation_measure_to_implement == "floodproofing":
            # the government decides which measure to apply based on what triggered the need for adaptation
            # apply updating the building structure but this takes the number of households that are adapting as input so we fist need to define that

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
            potential_to_adapt = int(budget / floodproofing_cost_per_household)
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
            fr = households.flood_risk_module
            fr.load_damage_curves()  # ensure damage curves are loaded before altering them
            fr.alter_damage_curves_for_flood_proofed_buildings()

            print(
                f"the government adapted {n_to_adapt} of the "
                f"{len(eligible_households)} eligible households in the floodzone by floodproofing their buildings"
            )

        # if adaptation_measure_to_implement == "risk_communication":
        #     # apply subsidies" --> maybe we can change who the subsidies are applied to but idk if that would make it better
        #     # this piece of code is still in Veerle's branch so can be uncommented when merged
        #      print ("the government adapted by providing subsidies to the most vulnerable households to reduce inequality")
        #     self.provide_subsidies()

        if adaptation_measure_to_implement == "reforestation":
            # apply reforestation, the reforestation limited by budget is already implemented in the prepare_modified_soil_maps_for_forest_function.
            converted_area_m2 = self.prepare_modified_soil_maps_for_forest()
            print(
                f"the government adapted by planting {converted_area_m2} m2 of forest in the most suitable areas to improve the ecosystem health"
            )

    def save_adaptation_record(
        self,
        EAD: float | None,
        exposure_inequality: float | None,
        ecosystem_health: float | None,
        current_IPV: float | None = None,
        adaptation_measure_to_implement: str | None = None,
    ) -> None:
        (
            """Save the values for the indicators, the IPV as well as the adaptation measure that is implemented each year to a csv."""
            ""
        )
        # save the values for each year of the model run to a csv file so that we can plot the values over time later on
        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        csv_file = output_folder / "adaptation_record_timeseries.csv"
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
                    "exposure_inequality",
                    "ecosystem_health",
                    "IPV",
                    "Implemented_measure",
                ],
            )
            if write_header:
                writer.writeheader()

            writer.writerow(
                {
                    "year": self.model.current_time.year,
                    "EAD": EAD,
                    "exposure_inequality": exposure_inequality,
                    "ecosystem_health": ecosystem_health,
                    "IPV": current_IPV,
                    "Implemented_measure": adaptation_measure_to_implement,
                }
            )

    def plot_indicators(self) -> None:
        """Plot the evaluation criteria over the run time and calculate the final score of the government's adaptation strategy."""
        # plot the evaluation criteria over time
        csv_file = (
            self.model.output_folder / "adaptation" / "adaptation_record_timeseries.csv"
        )
        df = pd.read_csv(csv_file)

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle("evaluation criteria over time")

        ax1.plot(df["year"], df["EAD"], label="EAD")
        ax1.set_ylabel("EAD (millions of €)")
        ax1.set_xlabel("Year")
        ax1.set_title("Expected Annual Damage (€) over time")

        ax2.plot(df["year"], df["exposure_inequality"], label="Exposure Inequality")
        ax2.set_ylabel("Exposure Inequality")
        ax2.set_xlabel("Year")
        ax2.set_title("Exposure Inequality over time")

        ax3.plot(df["year"], df["ecosystem_health"], label="Ecosystem Health")
        ax3.set_ylabel("Ecosystem Health")
        ax3.set_xlabel("Year")
        ax3.set_title("Ecosystem Health over time")

        fig.tight_layout()
        output_folder = self.model.output_folder / "adaptation"
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_folder / "adaptation_indicators_over_time.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def plot_adaptation_pathway(self) -> None:
        """plot the adaptation pathway, showing the adaptation measures that are implemented over time."""
        csv_file = (
            self.model.output_folder / "adaptation" / "adaptation_record_timeseries.csv"
        )

        df = pd.read_csv(csv_file)

        colour_map = {
            "floodproofing": "#5F5E5A",
            "reforestation": "#3B8B2E",
            "risk_communication": "#C0392B",
        }

        fig, ax = plt.subplots(figsize=(10, 5))

        # plot each segment between adjacent years, coloured by measure type
        for i in range(len(df) - 1):
            x_seg = [int(df["year"].iloc[i]), int(df["year"].iloc[i + 1])]
            y_seg = [float(df["IPV"].iloc[i]), float(df["IPV"].iloc[i + 1])]
            colour = colour_map.get(df["Implemented_measure"].iloc[i], "#aaaaaa")
            ax.plot(x_seg, y_seg, color=colour, linewidth=2.5)

        # add dots at each year coloured by measure
        for _, row in df.iterrows():
            colour = colour_map.get(row["Implemented_measure"], "#aaaaaa")
            ax.scatter(
                int(row["year"]), float(row["IPV"]), color=colour, s=40, zorder=5
            )

        # legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="#5F5E5A",
                linewidth=2.5,
                label="Floodproofing",
            ),
            Line2D([0], [0], color="#3B8B2E", linewidth=2.5, label="Reforestation"),
            Line2D(
                [0], [0], color="#C0392B", linewidth=2.5, label="Risk Communication"
            ),
        ]
        ax.legend(handles=legend_elements, fontsize=10, framealpha=0.3)

        ax.set_title("Adaptation pathway", fontsize=13)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Integrated Performance Value", fontsize=11)

        # Auto-scale y-axis to show actual variation in scores rather than fixed 0–1 range.
        # This makes small improvements visible without distorting the data.
        score_min = df["IPV"].min()
        score_max = df["IPV"].max()
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
