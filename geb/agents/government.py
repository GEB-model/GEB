"""This module contains the Government agent class for GEB."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import fftconvolve

from geb.hydrology.landcovers import FOREST
from geb.workflows.io import read_geom, read_zarr
from geb.workflows.raster import calculate_height_m, calculate_width_m

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

    def provide_risk_communication(self, seed: int | None = None) -> None:
        """Communicate risk to households based on the configuration.

        Raises:
            ValueError: If risk_communication.frequency is not "yearly" or "always".
            ValueError: If risk_communication.selected_households is not "all" or "random_share".
        """
        # Skip risk communication during spinup
        if self.model.in_spinup:
            return None

        # Skip if config is missing or disabled (for all timesteps)
        if "risk_communication" not in self.config or not self.config[
            "risk_communication"
        ].get("enabled", True):
            if self.model.current_timestep == 0:
                self.model.logger.warning(
                    "Risk communication is disabled or not configured; no risk communication will be provided."
                )
            return None

        risk_communication_config = self.config["risk_communication"]
        frequency = risk_communication_config.get("frequency", "yearly")
        if frequency == "yearly":
            self.model.logger.info("Providing yearly risk communication to households.")
            if not (
                self.model.current_time.day == 1 and self.model.current_time.month == 1
            ):  # provide risk communication on the first day of the year
                return None
        elif frequency != "always":
            raise ValueError(
                "risk_communication.frequency must be 'yearly' or 'always'"
            )

        selected_households = risk_communication_config.get(
            "selected_households", "all"
        )
        n_households = self.agents.households.n
        if selected_households == "all":
            self.model.logger.info("Providing risk communication to all households.")
            eligible_mask = np.ones(n_households, dtype=bool)
        elif selected_households == "random_share":
            self.model.logger.info(
                "Providing risk communication to a random share of households."
            )
            share = float(risk_communication_config.get("share", 1.0))
            share = min(max(share, 0.0), 1.0)
            actual_seed = (
                seed if seed is not None else risk_communication_config.get("seed", 42)
            )
            rng = np.random.default_rng(actual_seed)
            eligible_mask = rng.random(n_households) < share
        else:
            raise ValueError(
                "risk_communication.selected_households must be 'all' or 'random_share'"
            )
        percentage_increase_risk_perception = float(
            risk_communication_config.get("percentage_increase_risk_perception", 0.0)
        )
        self.agents.households.apply_risk_communication(
            percentage_increase=percentage_increase_risk_perception,
            household_mask=eligible_mask,
        )

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
        to identify suitable HRUs, blends soil property values from existing forest
        HRUs into suitable HRUs, saves a figure, and removes farmers from fully
        converted areas. The threshold is read from the config key
        ``forest_restoration_potential_threshold`` and defaults to 0.5.

        When adaptation is enabled, it is determined how much area to
        plant based on the available budget and reforestation costs. Since the area of a
        single HRU is often larger than what is affordable in one year, HRUs can also be partly converted.
        This is tracked by the self.forest_fraction. When reforestation is the chosen adaptation strategy, partly converted HRUs are first completed with forest
        before a new HRU is forested with the remaining budget. Once the HRU is fully converted, the land_use_type becomes
        FOREST and farmers are removed from it.

        When adaptation is disabled, all suitable HRUs are converted to forest in one go.

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
        potential_HRU: np.ndarray = hydrology.to_HRU(data=forest_potential)

        area_per_hru_m2: np.ndarray = hydrology.HRU.var.cell_area
        suitable_area_m2 = float(area_per_hru_m2[suitable_HRU].sum())

        adaptation_enabled = self.config.get("adaptation", {}).get("enabled", False)

        # Per-HRU conversion progress (0 = untouched, 1 = fully forested).
        if not hasattr(self, "forest_fraction"):
            self.forest_fraction = np.where(
                hydrology.HRU.var.land_use_type == FOREST, 1.0, 0.0
            ).astype(np.float64)
        forest_fraction = self.forest_fraction

        soil_properties = (
            "water_content_saturated_m",
            "water_content_field_capacity_m",
            "water_content_wilting_point_m",
            "water_content_residual_m",
            "saturated_hydraulic_conductivity_m_per_s",
            "bubbling_pressure_m_positive",
            "lambda_pore_size_distribution",
            "solid_heat_capacity_J_per_m2_K",
        )
        # save the original soil properties so we can calculate the change in soil properties due to reforestation, also when partial conversion of HRU happens.
        if not hasattr(self, "_reforestation_soil_baseline"):
            self._reforestation_soil_baseline = {
                prop: getattr(hydrology.HRU.var, prop).copy()
                for prop in soil_properties
            }
        baseline = self._reforestation_soil_baseline

        # incremental planting when adaptation is enabled.

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

            self.model.logger.info(
                "Suitable area for reforestation: %.2f m2", suitable_area_m2
            )
            self.model.logger.info(
                "Affordable area for reforestation: %.2f m2", affordable_area_m2
            )

            suitable_indices = np.where(suitable_HRU)[0]

            if len(suitable_indices) == 0:
                self.model.logger.warning(
                    "Incremental reforestation: no suitable HRUs found."
                )
                return 0.0

            remaining_capacity = 1.0 - forest_fraction[suitable_indices]
            remaining = suitable_indices[remaining_capacity > 1e-9]

            if len(remaining) == 0:
                self.model.logger.warning(
                    "Incremental reforestation: all suitable HRUs already forest."
                )
                return 0.0

            # Finish HRUs already partly converted in a previous year first, then spend any leftover budget on new HRUs,
            # highest restoration potential first (smallest area as a tiebreak among equally-suitable HRUs).
            already_started = forest_fraction[remaining] > 0.0
            order = np.lexsort(
                (
                    area_per_hru_m2[remaining],
                    -potential_HRU[remaining],
                    ~already_started,
                )
            )
            remaining = remaining[order]

            remaining_budget = affordable_area_m2
            chunk_indices = []
            newly_full_indices = []
            converted_area_m2 = 0.0

            for idx in remaining:
                if remaining_budget <= 0.0:
                    break
                area = float(area_per_hru_m2[idx])
                capacity_area = area * (1.0 - forest_fraction[idx])
                if capacity_area <= remaining_budget:
                    add_area = capacity_area
                    forest_fraction[idx] = 1.0
                    newly_full_indices.append(idx)
                else:
                    add_area = remaining_budget
                    forest_fraction[idx] += add_area / area
                remaining_budget -= add_area
                converted_area_m2 += add_area
                chunk_indices.append(idx)

            if len(chunk_indices) == 0:
                self.model.logger.info(
                    "Incremental reforestation: budget insufficient."
                )
                return 0.0

            final_HRU = np.zeros_like(suitable_HRU, dtype=bool)
            final_HRU[chunk_indices] = True

            fully_converted_HRU = np.zeros_like(suitable_HRU, dtype=bool)
            fully_converted_HRU[newly_full_indices] = True

            self.model.logger.info(
                "Incremental reforestation: converted %d HRUs (%.2f m2), %d fully forested.",
                len(chunk_indices),
                converted_area_m2,
                len(newly_full_indices),
            )

        else:
            # all at once reforestation when adaptation is disabled.
            final_HRU = suitable_HRU
            fully_converted_HRU = suitable_HRU
            forest_fraction[final_HRU] = 1.0
            converted_area_m2 = float(area_per_hru_m2[final_HRU].sum())

            self.model.logger.info(
                "Reforestation (all at once): planting %d HRUs (threshold %.2f).",
                int(final_HRU.sum()),
                threshold,
            )

        # modification for both incremental and all at once reforestation
        land_use_type_before = hydrology.HRU.var.land_use_type.copy()

        # forest_mean is computed from HRUs that were already fully FOREST
        forest_mask = hydrology.HRU.var.land_use_type == FOREST
        touched_fraction = forest_fraction[final_HRU]

        for prop in soil_properties:
            arr = getattr(hydrology.HRU.var, prop)
            forest_mean = arr[:, forest_mask].mean(axis=1)
            base = baseline[prop][:, final_HRU]
            arr[:, final_HRU] = (
                1.0 - touched_fraction
            ) * base + touched_fraction * forest_mean[:, np.newaxis]

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

        self.remove_farmers_from_converted_forest_areas(fully_converted_HRU)

        # Explicitly mark fully-converted HRUs as FOREST so that future calls to
        # prepare_modified_soil_maps_for_forest can detect them via forest_fraction
        # and remove_farmers
        hydrology.HRU.var.land_use_type[fully_converted_HRU] = FOREST

        output_folder = self.model.output_folder / "forest_planting"
        output_folder.mkdir(parents=True, exist_ok=True)
        self._save_forest_planting_figure(
            land_use_type_before, fully_converted_HRU, output_folder, threshold
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
            self.model.logger.info("No farmers found in suitable areas, none removed.")
            return

        unique_farmer_indices = np.unique(farmer_indices)
        farmers_before = crop_farmers.n
        crop_farmers.remove_agents(
            farmer_indices=unique_farmer_indices,
            new_land_use_type=FOREST,
        )
        self.model.logger.info(
            "Farmers removed: %d (%d → %d)",
            len(unique_farmer_indices),
            farmers_before,
            crop_farmers.n,
        )

    def adaptation(self) -> None:
        """From this function all steps for the adaptation implementation are called.

        Checks if adaptation is enabled and if it is January 1st, then calculates EAD,
        equity, and ecosystem indicators. Consequently, a mental simulation of all possible adaptation measures in an 'alternate universe' is done to see which one has the biggest improvement over X years
        based on the sum of the normalised indicators (the IPV). This measure is then implemented. As this is done yearly, an adaptation pathway forms that optimises this IPV.

        """
        if not self.config["adaptation"].get("enabled", True):
            return  # exits because adaptation is not enabled in the config file
        if getattr(self.model, "multiverse_name", None) is not None:
            return  # exits because we are inside an alternate universe hypothetical run
        if not (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            return  # exits because it is not the first of January
        if self.model.in_spinup:
            return  # exits because the model is in spinup, no adaptation during spinup

        budget = self.config["adaptation_costs"].get("initial_budget")

        EAD: float
        raw_EAD: float
        ead_per_household: np.ndarray
        EAD, raw_EAD, ead_per_household = self.calculate_risk_reduction_indicator()
        equity_indicator: float
        raw_flood_damage_burden: float
        raw_forest_access_low_income: float
        equity_indicator, raw_flood_damage_burden, raw_forest_access_low_income = (
            self.calculate_equity_indicator(ead_per_household)
        )
        ecosystem_health: float = self.calculate_ecosystem_indicator()
        # TEMP: capture now, before select_adaptation_measure_to_implement's hypothetical
        # evaluations below overwrite self._raw_ecosystem_health with their own values
        raw_ecosystem_health = self._raw_ecosystem_health

        # equal-weighted, for reporting/comparison across runs -- see calculate_IPV
        current_IPV = self.calculate_IPV(EAD, equity_indicator, ecosystem_health)
        # weighted by this run's priority_weights, to drive adaptation selection below
        current_weighted_IPV = self.calculate_weighted_IPV(
            EAD, equity_indicator, ecosystem_health
        )

        # do a mental simulationin an alternate universeof the improvement (based on indicators) the implementation of all measures would generate. This will help the government to decide which
        # adaptation_measure_to_implement = "floodproofing"
        adaptation_measure_to_implement = self.select_adaptation_measure_to_implement(
            budget, current_weighted_IPV
        )

        # implement the adaptation measure that is selected
        self.apply_adaptation(budget, adaptation_measure_to_implement)

        # we want to save these things yearly so we can plot them
        self.save_adaptation_record(
            EAD,
            equity_indicator,
            ecosystem_health,
            current_IPV,
            adaptation_measure_to_implement,
            raw_EAD,
            raw_ecosystem_health,
            raw_flood_damage_burden,
            raw_forest_access_low_income,
        )

    def calculate_IPV(
        self, EAD: float, equity_indicator: float, ecosystem_health: float
    ) -> float:
        """Calculate the Integrated Performance Value: the equal-weighted mean of the normalised indicators.

        Always uses equal (1/3) weights, regardless of this run's priority_weights,
        so the IPV is comparable across runs with different priorities. Adaptation
        *selection* is driven separately by calculate_weighted_IPV, which does
        use this run's own priority_weights.

        Returns:
            The IPV.
        """
        return (EAD + equity_indicator + ecosystem_health) / 3

    # made into a function so it can also be used in the hypothetical implementation
    def calculate_weighted_IPV(
        self, EAD: float, equity_indicator: float, ecosystem_health: float
    ) -> float:
        """Calculate the weighted sum of the normalised indicators, using this run's priority_weights.

        Used to drive which adaptation measure select_adaptation_measure_to_implement
        picks. Not used for the reported IPV -- see calculate_IPV.

        Returns:
            The weighted IPV.
        """
        risk_reduction_weight: float = self.config["priority_weights"].get(
            "risk_reduction"
        )
        equity_weight: float = self.config["priority_weights"].get("equity")
        ecosystem_weight: float = self.config["priority_weights"].get(
            "ecosystem_health"
        )

        weighted_IPV = sum(
            [
                (risk_reduction_weight * EAD),
                (equity_weight * equity_indicator),
                (ecosystem_weight * ecosystem_health),
            ]
        )
        return weighted_IPV

    def hypothetical_implementation(
        self, budget: float, measure: str, n_timesteps: int
    ) -> float:
        """Implement a hypothetical implementation of the adaptation measure to see what the improvement in priority score would be in an alternate universe.

        Args:
            budget: the available budget for adaptation.
            measure: the adaptation measure to implement.
            n_timesteps: the number of timesteps to simulate for the hypothetical implementation.

        Returns:
        the hypothetical weighted IPV after the implementation of the adaptation measure.
        """

        def do_function() -> None:
            self.apply_adaptation(budget, measure)

        def collect_function() -> dict[str, Any]:
            hypothetical_EAD, _hypothetical_raw_EAD, hyp_ead_per_household = (
                self.calculate_risk_reduction_indicator()
            )
            hypothetical_equity_indicator, _, _ = self.calculate_equity_indicator(
                hyp_ead_per_household
            )
            hypothetical_ecosystem_health = self.calculate_ecosystem_indicator()
            hypothetical_weighted_IPV = self.calculate_weighted_IPV(
                hypothetical_EAD,
                hypothetical_equity_indicator,
                hypothetical_ecosystem_health,
            )
            return {
                "EAD": hypothetical_EAD,
                "equity_indicator": hypothetical_equity_indicator,
                "ecosystem_health": hypothetical_ecosystem_health,
                "weighted_IPV": hypothetical_weighted_IPV,
            }

        self.model.logger.info("Starting hypothetical simulation for '%s'.", measure)

        # alternate_universe() saves/restores proper model state (e.g.
        # hydrology.HRU.var.land_use_type), but cumulative_reforested_area_m2
        # is not part of that state, so needs to be saved/restored manually to prevent
        # hypothetical reforestation to affect the running total.
        cumulative_reforested_area_m2_before_hypothetical = getattr(
            self, "cumulative_reforested_area_m2", 0.0
        )

        # same for forest_fraction (per-HRU reforestation progress, see
        # prepare_modified_soil_maps_for_forest)
        forest_fraction_before_hypothetical = getattr(self, "forest_fraction", None)
        if forest_fraction_before_hypothetical is not None:
            forest_fraction_before_hypothetical = (
                forest_fraction_before_hypothetical.copy()
            )

        # same for households.buildings (especially 'flood_proofed' column and the
        # two damage-curve tables).
        households = self.agents.households
        buildings_before_hypothetical = households.buildings.copy()
        buildings_structure_curve_before_hypothetical = (
            households.buildings_structure_curve.copy()
        )
        buildings_content_curve_before_hypothetical = (
            households.buildings_content_curve.copy()
        )

        result_hypothetical_implementation = self.model.alternate_universe(
            name="",
            n_timesteps=n_timesteps,
            do_function=do_function,
            collect_function=collect_function,
        )

        self.cumulative_reforested_area_m2 = (
            cumulative_reforested_area_m2_before_hypothetical
        )

        if forest_fraction_before_hypothetical is None:
            if hasattr(self, "forest_fraction"):
                del self.forest_fraction
        else:
            self.forest_fraction = forest_fraction_before_hypothetical

        households.buildings = buildings_before_hypothetical
        households.buildings_structure_curve = (
            buildings_structure_curve_before_hypothetical
        )
        households.buildings_content_curve = buildings_content_curve_before_hypothetical

        self.model.logger.info(
            "[%d] Hypothetical '%s': EAD=%.4f, equity=%.4f, ecosystem=%.4f, weighted_IPV=%.4f",
            self.model.current_time.year,
            measure,
            result_hypothetical_implementation["EAD"],
            result_hypothetical_implementation["equity_indicator"],
            result_hypothetical_implementation["ecosystem_health"],
            result_hypothetical_implementation["weighted_IPV"],
        )

        return result_hypothetical_implementation["weighted_IPV"]

    def select_adaptation_measure_to_implement(
        self, budget: float, current_weighted_IPV: float
    ) -> str | None:
        """Calculate the improvement in weighted IPV for each potential adaptation measure.

        The adaptation measure that generates the biggest improvement in weighted
        IPV (using this run's priority_weights) during the hypothetical
        implementation will actually be implemented.

        Returns:
            The adaptation measure to implement.
        """
        measures = ["floodproofing", "risk_communication", "reforestation"]

        # Clamp the look-ahead so the alternate-universe rollout never simulates
        # past the model's configured end time. Yearly-indexed market data (e.g.
        # inflation rates) is only prepared up to run_end, so a hypothetical
        # evaluation started in the final years of the run would otherwise index
        # past the end of that data.
        remaining_timesteps = (self.model.run_end - self.model.current_time).days
        n_timesteps = max(min(1095, remaining_timesteps), 0)

        deltas = {}
        for measure in measures:
            hypothetical_weighted_IPV = self.hypothetical_implementation(
                budget=budget, measure=measure, n_timesteps=n_timesteps
            )
            # select the adaptation measure with the biggest improvement and make that the adaptation that is to be implemented
            deltas[measure] = hypothetical_weighted_IPV - current_weighted_IPV

            self.model.logger.info(
                "[%d] Delta weighted_IPV for '%s': hypothetical=%.4f, current=%.4f, delta=%.4f",
                self.model.current_time.year,
                measure,
                hypothetical_weighted_IPV,
                current_weighted_IPV,
                deltas[measure],
            )

        adaptation_measure_to_implement = max(deltas, key=deltas.get)

        self.model.logger.info(
            "[%d] Selected adaptation measure: '%s' (delta=%.4f)",
            self.model.current_time.year,
            adaptation_measure_to_implement,
            deltas[adaptation_measure_to_implement],
        )

        return adaptation_measure_to_implement

    def normalise_indicator(self, indicator: str, raw_value: float) -> float:
        """Normalise an indicator value on a 0-1 scale, where 0 is the worst value and 1 is the best value.

        the best and the worst value are computed based on the reference runs for the catchment. Using
        a run with no adaptation measures implemented and a run with all adaptation measures implemented.
        The values are specified in the config file.

        Args:
            indicator: The indicator name to normalise.
            raw_value: The raw indicator value to normalise.

        Returns:
            The normalised indicator value on a 0-1 scale.
        """
        if indicator == "ead":
            best_value = self.config["normalisation_values"]["ead_best_value"]
            worst_value = self.config["normalisation_values"]["ead_worst_value"]
            normalised_ead = (raw_value - worst_value) / (best_value - worst_value)
            return normalised_ead

        if indicator == "flood_damage_burden":
            best_value = self.config["normalisation_values"][
                "flooddamageburden_best_value"
            ]
            worst_value = self.config["normalisation_values"][
                "flooddamageburden_worst_value"
            ]
            normalised_flooddamageburden = (raw_value - worst_value) / (
                best_value - worst_value
            )
            return normalised_flooddamageburden

        if indicator == "forest_access_low_income":
            # Higher raw forest fraction is better, so best_value is
            # expected to be the *larger* number and worst_value the
            # *smaller* one -- the formula below still works either way
            # since it's just a fractional position between the two
            # anchors, not an assumption about which direction is better.
            best_value = self.config["normalisation_values"][
                "forestaccesslowincome_best_value"
            ]
            worst_value = self.config["normalisation_values"][
                "forestaccesslowincome_worst_value"
            ]
            normalised_forestaccesslowincome = (raw_value - worst_value) / (
                best_value - worst_value
            )
            return normalised_forestaccesslowincome

        if indicator == "ecosystem_health":
            best_value = self.config["normalisation_values"][
                "ecosystemhealth_best_value"
            ]
            worst_value = self.config["normalisation_values"][
                "ecosystemhealth_worst_value"
            ]
            normalised_ecosystemhealth = (raw_value - worst_value) / (
                best_value - worst_value
            )
            return normalised_ecosystemhealth

    def calculate_risk_reduction_indicator(self) -> tuple[float, float, np.ndarray]:
        """Calculate the expected annual damage (EAD) for the current year.

        EAD is computed by integrating total flood damage over the exceedance
        probability curve (trapezoid rule across return periods). Uses the same
        method as the flood risk module, but with the reforestation-aware per-household damages, to account
        for reforestation effects on flood depths and damages, as it is impleemented.

        Returns:
           The normalised EAD, the raw EAD in EUR/year, and the per-household EAD array.
        """
        fr = self.agents.households.flood_risk_module

        ead_per_household = self.calculate_reforestation_aware_ead_per_household()
        fr.load_flood_maps()  # restore the static baseline maps for other code

        raw_EAD = float(ead_per_household.sum())
        normalised_EAD = self.normalise_indicator("ead", raw_EAD)

        return normalised_EAD, raw_EAD, ead_per_household

    def reforestation_scenario_bracket(self) -> tuple[str, str, float] | None:
        """Bracketing reforestation flood-map scenarios for the current cumulative reforested area.

        The equity indicator is built from per-household damages, which (unlike
        the aggregate EAD) need real spatially explicit flood depths to respond
        to reforestation at all -- a single scaling factor cancels out in the
        low-income/total EAD ratio. The reforestation extent sweep already
        produced real SFINCS flood maps per scenario (see
        compute_reforestation_ead.py); this identifies the two scenarios that
        bracket the current reforested area, and a weight between them, so
        the per-household damages under each can be interpolated (see
        :meth:`calculate_reforestation_aware_ead_per_household`).

        Returns:
            (lower_folder, upper_folder, weight): output subfolder names
            (e.g. "reforestation_50%") of the bracketing scenarios, and the
            fraction of the way from lower to upper (0 = exactly at lower,
            1 = exactly at upper). lower_folder == upper_folder with
            weight == 0.0 at or beyond either end of the sweep. None if no
            lookup table is configured (household damages then fall back to
            the static baseline maps).
        """
        lookup_table_path = self.config.get("reforestation_ead_lookup_table")
        if not lookup_table_path:
            return None

        lookup_table = pd.read_csv(lookup_table_path).sort_values("reforested_area_m2")
        areas = lookup_table["reforested_area_m2"].to_numpy(dtype=float)
        pcts = lookup_table["reforested_pct"].to_numpy(dtype=int)

        current_area_m2 = getattr(self, "cumulative_reforested_area_m2", 0.0)
        clipped_area_m2 = min(max(current_area_m2, areas[0]), areas[-1])

        upper_idx = int(np.searchsorted(areas, clipped_area_m2))
        upper_idx = min(max(upper_idx, 1), len(areas) - 1)
        lower_idx = upper_idx - 1

        area_lower, area_upper = areas[lower_idx], areas[upper_idx]
        weight = (
            0.0
            if area_upper == area_lower
            else (clipped_area_m2 - area_lower) / (area_upper - area_lower)
        )

        return (
            f"reforestation_{pcts[lower_idx]}%",
            f"reforestation_{pcts[upper_idx]}%",
            float(weight),
        )

    def load_household_flood_maps_for_scenario(self, scenario_folder: str) -> None:
        """Load per-household flood maps from a specific precomputed reforestation scenario.

        Points ``households.flood_maps`` at real flood depth maps for one of
        the precomputed reforestation-extent scenarios (see
        `reforestation_scenario_bracket`), instead of the static spinup
        baseline that ``FloodRiskModule.load_flood_maps()`` always loads.

        Callers must call ``fr.load_flood_maps()`` afterwards to restore the
        baseline maps once the scenario-conditioned damages have been used,
        so other code relying on ``households.flood_maps`` is not affected.

        Args:
            scenario_folder: Output subfolder to load flood maps from (e.g.
                "reforestation_50%").
        """
        households = self.agents.households
        flood_maps = {}
        for return_period in households.return_periods:
            file_path = (
                self.model.output_folder.parent
                / scenario_folder
                / "flood_maps"
                / f"{return_period}.zarr"
            )
            flood_maps[return_period] = read_zarr(file_path)
        households.flood_maps = flood_maps

    def calculate_reforestation_aware_ead_per_household(self) -> np.ndarray:
        """Per-household EAD, interpolated between the bracketing reforestation scenarios.

        This computes the real per-household EAD under each bracketing scenario's own flood maps, then linearly
        interpolates those two already-resolved damage vectors.

        Returns:
            1-D array of expected annual damage per household, reflecting the
            current cumulative reforested area.
        """
        households = self.agents.households
        fr = households.flood_risk_module

        if "flooded" not in households.buildings.columns:
            households.update_building_attributes()

        bracket = self.reforestation_scenario_bracket()
        if bracket is None:
            damages_do_not_adapt, damages_adapt = fr.calculate_building_flood_damages(
                dynamic=True
            )
            return fr.calculate_ead(
                damages_do_not_adapt, damages_adapt, households.var.adapted.data
            )

        lower_folder, upper_folder, weight = bracket

        self.load_household_flood_maps_for_scenario(lower_folder)
        damages_do_not_adapt, damages_adapt = fr.calculate_building_flood_damages(
            dynamic=True
        )
        ead_lower = fr.calculate_ead(
            damages_do_not_adapt, damages_adapt, households.var.adapted.data
        )

        if weight == 0.0 or upper_folder == lower_folder:
            return ead_lower

        self.load_household_flood_maps_for_scenario(upper_folder)
        damages_do_not_adapt, damages_adapt = fr.calculate_building_flood_damages(
            dynamic=True
        )
        ead_upper = fr.calculate_ead(
            damages_do_not_adapt, damages_adapt, households.var.adapted.data
        )

        return ead_lower + weight * (ead_upper - ead_lower)

    def calculate_equity_indicator(
        self, ead_per_household: np.ndarray
    ) -> tuple[float, float, float]:
        """Calculate the equity indicator for the current year.

        The equity indicator consists of 2 subindicators: the flood damage burden, the share of the ead borne by low-income households,
        and the forest access, the local forest fraction (within 1km) around low-income households. The two subindicators are normalised and averaged to get the final equity indicator.

        Args:
            ead_per_household: 1-D array of expected annual damage per household.

        Returns:
            the normalised equity indicator value, the raw flood damage
            burden, and the raw forest access (low-income) value.
        """
        households = self.agents.households

        # EU at-risk-of-poverty threshold: 60% of median income
        # https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:At-risk-of-poverty_threshold
        median_income = np.median(households.var.income.data)
        low_income_mask = households.var.income.data <= 0.6 * median_income

        # flood_damage_burden subindicator
        income = households.var.income.data
        relative_ead_low_income = (
            ead_per_household[low_income_mask] / income[low_income_mask]
        )
        relative_ead_all = ead_per_household / income

        flood_damage_burden = 1 - (
            relative_ead_low_income.sum() / relative_ead_all.sum()
        )

        normalised_flood_damage_burden = self.normalise_indicator(
            "flood_damage_burden", flood_damage_burden
        )

        # Forest-access sub-indicator: the raw average local forest
        # fraction (fraction of land within 1km that is forest low-income households have
        # around them

        forest_access_low_income = self.calculate_forest_access_indicator(
            low_income_mask=low_income_mask
        )
        normalised_forest_access = self.normalise_indicator(
            "forest_access_low_income", forest_access_low_income
        )

        # Combine the two sub-indicators into a single equity indicator.
        normalised_equity_indicator = (
            normalised_flood_damage_burden + normalised_forest_access
        ) / 2

        self.model.logger.info(
            "Calculated equity indicator: %s", normalised_equity_indicator
        )
        return (
            normalised_equity_indicator,
            flood_damage_burden,
            forest_access_low_income,
        )

    def calculate_household_forest_fraction(
        self, radius_m: float = 1000.0
    ) -> np.ndarray:
        """Calculate the fraction of forest within radius_m of each household.

        This tracks local forest cover around each household, which turns
        this into a reforestation-attributable signal by comparing against
        the baseline value captured before any reforestation happened.

        Within the disk, the fraction is computed over in-catchment cells
        only (cells outside the modelled domain are excluded from both the
        forest count and the denominator), so households near the
        catchment boundary aren't penalised for their neighbourhood disk
        spilling outside the domain.

        Args:
            radius_m: radius, in meters, of the neighbourhood around each
                household within which forest cover is averaged.

        Returns:
            1-D array (one value per household), each in [0, 1]: the
            fraction of the household's radius_m neighbourhood, among
            in-catchment cells only, that is forest.
        """
        hydrology = self.model.hydrology

        # Loaded only as a raster template (shape, coordinates, affine
        # transform) matching the HRU grid.
        zpath = self.model.files["subgrid"]["landcover/classification"]
        template_da = read_zarr(zpath)

        n_hrus = hydrology.HRU.var.land_use_type.size
        hru_index_1d = np.arange(n_hrus, dtype=np.int32)
        hru_index_2d = hydrology.HRU.decompress(hru_index_1d)
        valid_mask = hru_index_2d >= 0
        forest_hru_mask = hydrology.HRU.var.land_use_type == FOREST
        forest_mask = np.zeros(template_da.shape, dtype=np.float64)
        forest_mask[valid_mask] = forest_hru_mask[hru_index_2d[valid_mask]]

        height, width = forest_mask.shape
        transform = template_da.rio.transform()
        height_m = float(calculate_height_m(transform, height, width)[0, 0])
        width_m = float(calculate_width_m(transform, height, width)[height // 2, 0])

        # build a circle-shaped kernel (1s inside radius_m, 0s outside). we don't use
        # scipy's uniform_filter here since it only does square windows, and a square
        # reaching radius_m in every direction covers ~27% more area than the circle
        # we actually want.
        radius_px_y = max(1, int(round(radius_m / height_m)))
        radius_px_x = max(1, int(round(radius_m / width_m)))
        yy, xx = np.ogrid[
            -radius_px_y : radius_px_y + 1, -radius_px_x : radius_px_x + 1
        ]
        disk_kernel = (
            (yy * height_m) ** 2 + (xx * width_m) ** 2 <= radius_m**2
        ).astype(np.float64)

        forest_count = fftconvolve(forest_mask, disk_kernel, mode="same")
        valid_count = fftconvolve(
            valid_mask.astype(np.float64), disk_kernel, mode="same"
        )

        # build a circular kernal, where a cell is 1 if it is within 1km of the kernels centre, 0 otherwise
        forest_fraction = np.full(valid_count.shape, np.nan)
        has_coverage = valid_count > 1e-9
        forest_fraction[has_coverage] = (
            forest_count[has_coverage] / valid_count[has_coverage]
        )
        # only count in-catchment cells, so households near the catchment boundary aren't penalised for their neighbourhood disk spilling outside the domain.
        np.clip(forest_fraction, 0.0, 1.0, out=forest_fraction)

        fraction_da = xr.DataArray(
            forest_fraction, coords=template_da.coords, dims=template_da.dims
        )

        households = self.agents.households
        household_points = households.var.household_points.to_crs(template_da.rio.crs)
        x_coords = household_points.geometry.x.values
        y_coords = household_points.geometry.y.values

        x_dim = template_da.rio.x_dim
        y_dim = template_da.rio.y_dim
        sampled_fraction = fraction_da.interp(
            {x_dim: ("points", x_coords), y_dim: ("points", y_coords)},
            method="nearest",
        )

        return sampled_fraction.values

    def calculate_forest_access_indicator(
        self,
        forest_fraction: np.ndarray | None = None,
        low_income_mask: np.ndarray | None = None,
    ) -> float:
        """Calculate the forest-access indicator for the current year.

        the average amount of forest low-income households have within their
        local neighbourhood.

        Args:
            forest_fraction: 1-D array of each household's local forest
                fraction (fraction of land within radius_m that is forest,
                see calculate_household_forest_fraction()), in [0, 1].
                Computed via calculate_household_forest_fraction() if not
                given.
            low_income_mask: 1-D boolean array marking low-income households.
                Computed the same way as in calculate_equity_indicator if not
                given.

        Returns:
            the raw average local forest fraction for low-income
            households, in [0, 1] (higher = better).
        """
        households = self.agents.households
        if forest_fraction is None:
            forest_fraction = self.calculate_household_forest_fraction()

        if low_income_mask is None:
            # EU at-risk-of-poverty threshold: 60% of median income
            median_income = np.median(households.var.income.data)
            low_income_mask = households.var.income.data <= 0.6 * median_income

        n_outside_domain = np.isnan(forest_fraction).sum()
        if n_outside_domain:
            # A handful of households sit right at the edge of the landcover
            # raster's domain, nanmean so the edge cases
            # don't propagate to NaN for the whole group average.
            self.model.logger.debug(
                "Forest access: %d/%d households outside the landcover raster domain, excluded from the average.",
                n_outside_domain,
                forest_fraction.size,
            )

        avg_forest_fraction_low_income = np.nanmean(forest_fraction[low_income_mask])

        self.model.logger.info(
            "Forest access: avg local forest fraction low-income=%.4f",
            avg_forest_fraction_low_income,
        )
        return avg_forest_fraction_low_income

    def calculate_ecosystem_indicator(self) -> float:
        """Calculate the ecosystem health for the current year.

        Returns:
        the ecosystem indicator value.
        """
        hydrology = self.model.hydrology

        # load the dataset
        zpath = self.model.files["subgrid"]["landcover/classification"]

        values_per_esa_land_use_type = {
            10: 0.74,  # tree cover
            30: 0.31,  # grassland
            40: 0.26,  # cropland
            50: 0.01,  # built-up
            80: 0.26,  # permanent water bodies
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

        # Blend in the live reforestation state for HRUs converted (fully or
        # partially) via reforestation — the ESA zarr is static and won't reflect
        # those changes. Uses forest_fraction (continuous, 0-1) rather than a binary
        # land_use_type check so partially-converted HRUs get proportional credit,
        # consistent with the EAD/equity indicator which is also area-continuous.
        forest_score = values_per_esa_land_use_type[10]  # tree cover = 1.0
        forest_fraction = getattr(self, "forest_fraction", None)
        if forest_fraction is None:
            forest_fraction = (hydrology.HRU.var.land_use_type == FOREST).astype(float)
        mean_score_per_hru = (
            1.0 - forest_fraction
        ) * mean_score_per_hru + forest_fraction * forest_score

        # land use type and area per HRU
        area_per_HRU = self.model.hydrology.HRU.var.cell_area.astype(float)

        # compute the area-weighted sum of HRU mean scores across the catchment
        weighted_sum = float((mean_score_per_hru * area_per_HRU).sum())

        # divide by total area to get the area-weighted mean score as the ecosystem indicator
        ecosystem_health = weighted_sum / (area_per_HRU.sum())
        self.model.logger.debug("Calculated ecosystem health: %s", ecosystem_health)
        # TEMP: stash raw value for reporting, to help recalibrate normalisation bounds
        self._raw_ecosystem_health = ecosystem_health

        normalised_ecosystem_health = self.normalise_indicator(
            "ecosystem_health", ecosystem_health
        )

        return normalised_ecosystem_health

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

        communication_cost_per_household = self.config["adaptation_costs"].get(
            "communication_cost_per_household"
        )

        if adaptation_measure_to_implement == "floodproofing":
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
                self.model.logger.info(
                    "No eligible households for government floodproofing (all flooded households already adapted)."
                )
                return

            # the government decides who of those are adapting
            # the number to adapt should actually be based on the available budget, but if the eligible households are less than budget allows, we can only adapt that numnber.
            potential_to_adapt = int(budget / floodproofing_cost_per_household)
            n_to_adapt = min(potential_to_adapt, len(eligible_households))
            if n_to_adapt == 0:
                self.model.logger.info(
                    "No households selected for government floodproofing this year."
                )
                return
            # randomly select the households that are adapted based on the number of households that can be adapted.
            # Seeded (year-varying, like risk_communication's rng below) so hypothetical
            # evaluations of 'floodproofing' are reproducible instead of depending on
            # whatever the global numpy RNG state happens to be at this point in the run.
            base_seed = self.config.get("floodproofing", {}).get("seed", 42)
            rng = np.random.default_rng(base_seed + self.model.current_time.year)
            adapting_households_sample = rng.choice(
                eligible_households, size=n_to_adapt, replace=False
            )
            # update the households that are adapted so they are not eligible for adaptation again.
            households.var.adapted[adapting_households_sample] = 1

            # use the function to floodproof the buildings of the households who are selected to adapt.
            # pass the full accumulated adapted set (not just this year's
            # sample), since update_building_adaptation_status() overwrites
            # flood_proofed for every building based only on what is passed in
            households.update_building_adaptation_status(
                np.where(households.var.adapted.data == 1)[0]
            )
            fr = households.flood_risk_module
            fr.load_damage_curves()  # ensure damage curves are loaded before altering them
            fr.alter_damage_curves_for_flood_proofed_buildings()

            self.model.logger.info(
                "Government floodproofed %d of %d eligible households in the flood zone.",
                n_to_adapt,
                len(eligible_households),
            )

        if adaptation_measure_to_implement == "risk_communication":
            n_affordable = int(budget / communication_cost_per_household)
            share = min(n_affordable / self.agents.households.n, 1.0)
            base_seed = self.config.get("risk_communication", {}).get("seed", 42)

            rc_config = self.config.get("risk_communication", {})
            risk_perception_increase = float(
                rc_config.get("risk_perception_increase", 0.0)
            )

            rng = np.random.default_rng(base_seed + self.model.current_time.year)
            eligible_mask = rng.random(self.agents.households.n) < share

            # Uses the decay-aware variant so the boost survives until the
            # household's next decision moment and fades out over subsequent
            # years, instead of apply_risk_communication()'s direct overwrite
            # of risk_perception (which gets wiped by that year's
            # update_risk_perceptions() call before it is ever read).
            self.agents.households.apply_risk_communication_with_decay(
                percentage_increase=risk_perception_increase,
                household_mask=eligible_mask,
            )

            self.model.logger.info(
                "Government applied risk communication to %d of %d households (%.1f%%).",
                eligible_mask.sum(),
                self.agents.households.n,
                100.0 * eligible_mask.mean(),
            )

        if adaptation_measure_to_implement == "reforestation":
            # apply reforestation, the reforestation limited by budget is already implemented in the prepare_modified_soil_maps_for_forest_function.
            converted_area_m2 = self.prepare_modified_soil_maps_for_forest()
            self.model.logger.info(
                "Government planted %.2f m2 of forest in suitable areas.",
                converted_area_m2,
            )
            # track the running total so calculate_risk_reduction_indicator()
            # can pick the bracketing precomputed flood-map scenarios for the
            # current level of reforestation (see reforestation_scenario_bracket())
            if not hasattr(self, "cumulative_reforested_area_m2"):
                self.cumulative_reforested_area_m2 = 0.0
            self.cumulative_reforested_area_m2 += converted_area_m2

    def save_adaptation_record(
        self,
        EAD: float | None,
        equity_indicator: float | None,
        ecosystem_health: float | None,
        current_IPV: float | None = None,
        adaptation_measure_to_implement: str | None = None,
        raw_EAD: float | None = None,
        raw_ecosystem_health: float
        | None = None,  # TEMP: for recalibrating normalisation bounds
        raw_flood_damage_burden: float
        | None = None,  # TEMP: for recalibrating normalisation bounds
        raw_forest_access_low_income: float
        | None = None,  # TEMP: for recalibrating normalisation bounds
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
                    "EAD_raw_eur",
                    "equity_indicator",
                    "ecosystem_health",
                    "ecosystem_health_raw",  # for calibrating normalisation bounds
                    "flood_damage_burden_raw",  # for calibrating normalisation bounds
                    "forest_access_low_income_raw",  # for calibrating normalisation bounds
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
                    "EAD_raw_eur": raw_EAD,
                    "equity_indicator": equity_indicator,
                    "ecosystem_health": ecosystem_health,
                    "ecosystem_health_raw": raw_ecosystem_health,
                    "flood_damage_burden_raw": raw_flood_damage_burden,
                    "forest_access_low_income_raw": raw_forest_access_low_income,
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
        fig.suptitle("Normalised Evaluation Criteria Over Time")

        ax1.plot(df["year"], df["EAD"], label="EAD")
        ax1.set_ylabel("Normalised EAD")
        ax1.set_xlabel("Year")
        ax1.set_title("Normalised Expected Annual Damage over time")

        ax2.plot(df["year"], df["equity_indicator"], label="Exposure Inequality")
        ax2.set_ylabel("Normalised Equity Indicator")
        ax2.set_xlabel("Year")
        ax2.set_title("Normalised Equity Indicator over time")

        ax3.plot(df["year"], df["ecosystem_health"], label="Ecosystem Health")
        ax3.set_ylabel("Normalised Ecosystem Health")
        ax3.set_xlabel("Year")
        ax3.set_title("Normalised Ecosystem Health over time")

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
            "floodproofing": "#1F77B4",
            "reforestation": "#3B8B2E",
            "risk_communication": "#F1C40F",
        }

        fig, ax = plt.subplots(figsize=(10, 5))

        # plot each segment between adjacent years -- lines stay black, only
        # the markers (below) are coloured by the measure implemented that year
        for i in range(len(df) - 1):
            x_seg = [int(df["year"].iloc[i]), int(df["year"].iloc[i + 1])]
            y_seg = [float(df["IPV"].iloc[i]), float(df["IPV"].iloc[i + 1])]
            ax.plot(x_seg, y_seg, color="black", linewidth=2.5)

        # add dots at each year coloured by measure
        for _, row in df.iterrows():
            colour = colour_map.get(row["Implemented_measure"], "#aaaaaa")
            ax.scatter(
                int(row["year"]), float(row["IPV"]), color=colour, s=40, zorder=5
            )

        # legend -- markers now, not lines, since colour lives on the dots
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#1F77B4",
                markersize=8,
                label="Floodproofing",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#3B8B2E",
                markersize=8,
                label="Reforestation",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F1C40F",
                markersize=8,
                label="Risk Communication",
            ),
        ]
        ax.legend(handles=legend_elements, fontsize=10, framealpha=0.3)

        ax.set_title("Adaptation pathway", fontsize=13)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(
            "Integrated Performance Value",
            fontsize=11,
        )

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
