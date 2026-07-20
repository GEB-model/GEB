"""This module contains the Government agent class for GEB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

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

    def provide_subsidies(self) -> None:
        """Provide subsidies to households based on the configuration.

        Configuration (model.yml):
            agent_settings.government.subsidies:
                enabled (bool, default: True):
                    Whether to apply subsidies at all.
                frequency (str, default: "yearly"):
                    When to apply subsidies. Allowed values:
                    - "yearly": only on Jan 1.
                    - "always": every timestep.
                    - "after_flood": only year after a flood event.
                apply_to (str, default: "all"):
                    Which households are eligible. Allowed values:
                    - "all": all households.
                    - "random_share": random subset of households.
                share (float, default: 1.0):
                    Only used when apply_to == "random_share".
                    Fraction in [0, 1] of households to select.
                seed (int, default: 42):
                    RNG seed used when apply_to == "random_share".
                dryproofing_subsidy_value (float, default: 0.0):
                    Absolute subsidy amount for dry-proofing (currency units).
                wetproofing_subsidy_value (float, default: 0.0):
                    Absolute subsidy amount for wet-proofing (currency units).

        Raises:
            ValueError: If subsidies.frequency is not "yearly" or "always".
            ValueError: If subsidies.apply_to is not "all" or "random_share".
        """
        # Skip subsidies during spinup
        if self.model.in_spinup:
            return None

        # Skip if config is missing or disabled (for all timesteps)
        if "subsidies" not in self.config or not self.config["subsidies"].get(
            "enabled", True
        ):
            if self.model.current_timestep == 0:
                print(
                    "Warning: subsidies are disabled or not configured for government agent. No subsidies will be provided"
                )
            return None
        subsidies_config = self.config["subsidies"]
        frequency = subsidies_config.get("frequency", "yearly")

        if frequency == "yearly":
            print("Providing yearly subsidies to households.")
            if not (
                self.model.current_time.day == 1 and self.model.current_time.month == 1
            ):  # provide subsidies on the first day of the year
                return None
        elif frequency == "after_flood":
            print("Providing subsidies only in the year after a flood event.")
            if not (
                (
                    self.model.current_time.year == 2004
                    and self.model.current_time.day == 1
                    and self.model.current_time.month == 1
                )
                or (
                    self.model.current_time.year == 2011
                    and self.model.current_time.day == 1
                    and self.model.current_time.month == 1
                )
                or (
                    self.model.current_time.year == 2019
                    and self.model.current_time.day == 1
                    and self.model.current_time.month == 1
                )
                or (
                    self.model.current_time.year == 2022
                    and self.model.current_time.day == 1
                    and self.model.current_time.month == 1
                )
            ):
                return None
        elif frequency != "always":
            raise ValueError(
                "subsidies.frequency must be 'yearly', 'always' or 'after_flood'"
            )

        selected_households = subsidies_config.get("selected_households", "all")
        n_households = self.agents.households.n
        if selected_households == "all":
            print("Providing subsidies to all households.")
            eligible_mask = np.ones(n_households, dtype=bool)
        elif selected_households == "random_share":
            print("Providing subsidies to a random share of households.")
            share = float(subsidies_config.get("share", 1.0))
            share = min(max(share, 0.0), 1.0)
            rng = np.random.default_rng(subsidies_config.get("seed", 42))
            eligible_mask = rng.random(n_households) < share
        else:
            raise ValueError(
                "subsidies.selected_households must be 'all' or 'random_share'"
            )

        dry_value = float(subsidies_config.get("dryproofing_subsidy_value", 0.0))
        print(f"Dry-proofing subsidy value: {dry_value}")
        wet_value = float(subsidies_config.get("wetproofing_subsidy_value", 0.0))
        print(f"Wet-proofing subsidy value: {wet_value}")
        self.agents.households.apply_subsidy(
            dryproofing_subsidy_value=dry_value,
            wetproofing_subsidy_value=wet_value,
            household_mask=eligible_mask,
        )

    def provide_risk_communication(self) -> None:
        """Communicate risk to households based on the configuration.

        Raises:
            ValueError: If risk_communication.frequency is not "yearly", "after_flood", "always".
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
                print(
                    "Warning: risk communication is disabled or not configured for government agent. No risk communication will be provided"
                )
            return None

        risk_communication_config = self.config["risk_communication"]
        frequency = risk_communication_config.get("frequency", "yearly")
        if frequency == "yearly":
            print("Providing yearly risk communication to households.")
            if not (
                self.model.current_time.day == 1 and self.model.current_time.month == 1
            ):  # provide risk communication on the first day of the year
                return None

        elif frequency == "after_flood":
            print("Providing risk communication only 3 years after a flood event.")
            if not (
                (
                    self.model.current_time.year == 2006
                    and self.model.current_time.day == 1
                    and self.model.current_time.month == 1
                )
                or (
                    self.model.current_time.year == 2013
                    and self.model.current_time.day == 1
                    and self.model.current_time.month == 1
                )
                or (
                    self.model.current_time.year == 2021
                    and self.model.current_time.day == 1
                    and self.model.current_time.month == 1
                )
                or (
                    self.model.current_time.year == 2024
                    and self.model.current_time.day == 1
                    and self.model.current_time.month == 1
                )
            ):
                return None
        elif frequency != "always":
            raise ValueError(
                "risk_communication.frequency must be 'yearly', 'after_flood', or 'always'"
            )

        selected_households = risk_communication_config.get(
            "selected_households", "all"
        )
        n_households = self.agents.households.n
        if selected_households == "all":
            print("Providing risk communication to all households.")
            eligible_mask = np.ones(n_households, dtype=bool)
        elif selected_households == "random_share":
            print("Providing risk communication to a random share of households.")
            share = float(risk_communication_config.get("share", 1.0))
            share = min(max(share, 0.0), 1.0)
            rng = np.random.default_rng(risk_communication_config.get("seed", 42))
            eligible_mask = rng.random(n_households) < share
        else:
            raise ValueError(
                "risk_communication.selected_households must be 'all' or 'random_share'"
            )
        percentage_increase_risk_perception = float(
            risk_communication_config.get("percentage_increase_risk_perception", 0.0)
        )
        # Queue the communication on households so it is applied after households recompute
        # their base risk perceptions (otherwise update_risk_perceptions overwrites it).
        self.agents.households._pending_risk_communication = {
            "percentage_increase": percentage_increase_risk_perception,
            "household_mask": eligible_mask,
            "absolute": False,
        }

    def provide_subsidies_to_vulnerable_households(
        self,
        subsidy_pot: float = 1_000_000.0,
        subsidy_per_household: float = 5_000.0,
        overhead_cost_per_household: float = 800.0,
        conversion_rate: float = 0.05,
        random_seed: int | None = 42,
    ) -> None:
        """Neighborhood-loop subsidy allocator.

        Parameters
        ---------
        subsidy_pot
            Total budget in euros.
        subsidy_per_household
            Fixed subsidy per participating household (euros).
        overhead_cost_per_household
            Additional administrative cost per participating household (euros).
        conversion_rate
            Fraction of households per neighborhood to select (e.g. 0.05 = 5%).
        random_seed
            Seed for reproducible random selection.

        Returns:
        -------
        pd.DataFrame
            Allocation log with columns: `buurtcode`, `MCDA`, `n_households`,
            `n_selected`, `cost`, `remaining_pot`.

        Raises:
            RuntimeError: If household_points do not contain 'buurtcode' and the
                attempted spatial join to assign neighborhood codes fails.
        """
        # Skip subsidies during spinup
        if self.model.in_spinup:
            return None

        # Skip if config is missing or disabled (for all timesteps)
        if "subsidies_to_vulnerable_households" not in self.config or not self.config[
            "subsidies_to_vulnerable_households"
        ].get("enabled", True):
            if self.model.current_timestep == 0:
                print(
                    "Warning: subsidies to vulnerable households are disabled or not configured for government agent. No subsidies will be provided"
                )
            return None

        if (
            self.model.current_time.year == 2024
            and self.model.current_time.month == 1
            and self.model.current_time.day == 1
        ):
            print(
                "Debug: Starting subsidy allocation to vulnerable households with the following parameters:"
            )
            print(f"  subsidy_pot: {subsidy_pot}")
            print(f"  subsidy_per_household: {subsidy_per_household}")
            print(f"  overhead_cost_per_household: {overhead_cost_per_household}")
            print(f"  conversion_rate: {conversion_rate}")
            print(f"  random_seed: {random_seed}")

            rng = np.random.default_rng(random_seed)

            # Load MCDA and mask, keep only intersecting features
            mcda = gpd.read_file(
                "/net/sys/pscst201/BETA-IVM-HPC@ada-nodes/vbl220/paper2/MCDA_limburg.gpkg"
            )
            mask = gpd.read_parquet(
                "/net/sys/pscst201/BETA-IVM-HPC@ada-nodes/vbl220/paper2/models/geul/input/geom/mask.geoparquet"
            )

            # Reproject mask to MCDA CRS if needed
            if mask.crs != mcda.crs:
                mask = mask.to_crs(mcda.crs)

            # Spatial intersection: keep MCDA features that intersect the mask
            mcda_in_mask = gpd.sjoin(mcda, mask, how="inner", predicate="intersects")

            # Keep required columns and sort descending by MCDA
            keep_cols = [
                c
                for c in ("MCDA", "buurtcode", "buurtnaam", "gemeentenaam", "geometry")
                if c in mcda_in_mask.columns
            ]
            mcda_neigh = mcda_in_mask[keep_cols].copy()
            mcda_neigh = mcda_neigh.sort_values("MCDA", ascending=False).reset_index(
                drop=True
            )

            # Prepare household points (assumes households.var.household_points exists)
            hp = self.agents.households.var.household_points.copy()
            if hp.crs != mcda_neigh.crs:
                hp = hp.to_crs(mcda_neigh.crs)

            # Ensure `buurtcode` exists on household points (assumes earlier spatial join)
            if "buurtcode" not in hp.columns:
                # attempt a spatial join to assign buurtcode to household points
                try:
                    hp = gpd.sjoin(
                        hp,
                        mcda_neigh[["buurtcode", "geometry"]],
                        how="left",
                        predicate="within",
                    )
                    hp = hp.drop(
                        columns=[c for c in ("index_right",) if c in hp.columns]
                    )
                except Exception:
                    raise RuntimeError(
                        "household_points do not contain 'buurtcode' and spatial join failed"
                    )

            # Iterate neighborhoods in order, select 5% randomly, apply subsidy until pot exhausted
            allocations = []

            # Map household GeoDataFrame index to integer household index used by `Households` arrays
            # This dummy assumes hp.index aligns with households ordering or contains a 'household_id' column
            if "household_id" in hp.columns:
                hp_index_to_household = hp["household_id"].astype(int).values
            else:
                # best-effort: assume sequential alignment
                hp_index_to_household = np.arange(len(hp), dtype=int)

            all_eligible_households = np.zeros(self.agents.households.n, dtype=bool)

            for _, neigh in mcda_neigh.iterrows():
                if subsidy_pot <= 0:
                    break

                buurt = neigh.get("buurtcode")
                buurtnaam = neigh.get("buurtnaam")
                gemeentenaam = neigh.get("gemeentenaam")
                mcda_score = neigh.get("MCDA")

                # households in this neighborhood (local indices into hp)
                idxs = hp.index[hp["buurtcode"] == buurt].to_numpy()
                n_house = len(idxs)
                if n_house == 0:
                    continue

                k = max(1, int(np.ceil(n_house * conversion_rate)))
                chosen_local = rng.choice(idxs, size=min(k, n_house), replace=False)

                # map chosen_local indices to model household ids (requires stable mapping)
                chosen_household_ids = hp_index_to_household[chosen_local.astype(int)]

                # compute newly selected (not already subsidized and not already adapted)
                # exclude households that already took an adaptation measure
                adapted_households = (
                    self.agents.households.var.adaptation_type.data != 0
                )
                newly_selected = np.setdiff1d(
                    chosen_household_ids,
                    np.nonzero(all_eligible_households | adapted_households)[0],
                    assume_unique=False,
                )
                n_new = len(newly_selected)
                if n_new == 0:
                    continue

                incremental_cost = n_new * (
                    subsidy_per_household + overhead_cost_per_household
                )
                if incremental_cost > subsidy_pot:
                    affordable = int(
                        subsidy_pot
                        // (subsidy_per_household + overhead_cost_per_household)
                    )
                    if affordable <= 0:
                        break
                    # pick a random subset of the newly_selected to fit budget
                    newly_selected = rng.choice(
                        newly_selected, size=affordable, replace=False
                    )
                    n_new = len(newly_selected)
                    incremental_cost = n_new * (
                        subsidy_per_household + overhead_cost_per_household
                    )

                # update cumulative mask
                all_eligible_households[newly_selected.astype(int)] = True

                # Automatically force adaptation to dryproofing (1) for newly selected households
                hh = self.agents.households
                idxs = newly_selected.astype(int)
                hh.var.adaptation_type.data[idxs] = 1
                hh.var.adapted.data[idxs] = 1
                # set time adapted to 1 for newly adapted households
                hh.var.time_adapted.data[idxs] = 1
                # update buildings' adaptation status based on household choices
                try:
                    hh.update_building_adaptation_status(hh.var.adaptation_type.data)
                except Exception:
                    pass

                subsidy_pot -= float(incremental_cost)

                allocations.append(
                    {
                        "buurtcode": buurt,
                        "buurtnaam": buurtnaam,
                        "gemeentenaam": gemeentenaam,
                        "MCDA": mcda_score,
                        "n_households": n_house,
                        "n_selected": int(n_new),
                        "cost": float(incremental_cost),
                        "remaining_pot": float(subsidy_pot),
                    }
                )

                print(allocations[-1])  # log each allocation step

    def provide_ontzorgen_to_vulnerable_households(
        self,
        subsidy_pot: float = 1_000_000.0,
        total_costs_per_household: float = 8200.0,
        conversion_rate: float = 0.3,
        random_seed: Optional[int] = 42,
    ) -> None:
        """Neighborhood-loop allocator for the ontzorgen scenario.

        Parameters
        ---------
        subsidy_pot
            Total budget in euros.
        total_costs_per_household
            Total costs per participating household, including subsidy and administrative costs (euros).
        conversion_rate
            Fraction of households per neighborhood to select (e.g. 0.05 = 5%).
        random_seed
            Seed for reproducible random selection.

        Returns:
        -------
        pd.DataFrame
            Allocation log with columns: `buurtcode`, `MCDA`, `n_households`,
            `n_selected`, `cost`, `remaining_pot`.

        Raises:
            RuntimeError: If household_points do not contain 'buurtcode' and the
                attempted spatial join to assign neighborhood codes fails.
        """
        # Skip subsidies during spinup
        if self.model.in_spinup:
            return None

        # Skip if config is missing or disabled (for all timesteps)
        if "ontzorgen_to_vulnerable_households" not in self.config or not self.config[
            "ontzorgen_to_vulnerable_households"
        ].get("enabled", True):
            if self.model.current_timestep == 0:
                print(
                    "Warning: ontzorgen to vulnerable households are disabled or not configured for government agent. No subsidies will be provided"
                )
            return None

        if (
            self.model.current_time.year == 2024
            and self.model.current_time.month == 1
            and self.model.current_time.day == 1
        ):
            print(
                "Debug: Starting subsidy allocation to vulnerable households with the following parameters:"
            )
            print(f"  subsidy_pot: {subsidy_pot}")
            print(f"  total_costs_per_household: {total_costs_per_household}")
            print(f"  conversion_rate: {conversion_rate}")
            print(f"  random_seed: {random_seed}")

            rng = np.random.default_rng(random_seed)

            # Load MCDA and mask, keep only intersecting features
            mcda = gpd.read_file(
                "/net/sys/pscst201/BETA-IVM-HPC@ada-nodes/vbl220/paper2/MCDA_limburg.gpkg"
            )
            mask = gpd.read_parquet(
                "/net/sys/pscst201/BETA-IVM-HPC@ada-nodes/vbl220/paper2/models/geul/input/geom/mask.geoparquet"
            )

            # Reproject mask to MCDA CRS if needed
            if mask.crs != mcda.crs:
                mask = mask.to_crs(mcda.crs)

            # Spatial intersection: keep MCDA features that intersect the mask
            mcda_in_mask = gpd.sjoin(mcda, mask, how="inner", predicate="intersects")

            # Keep required columns and sort descending by MCDA
            keep_cols = [
                c
                for c in ("MCDA", "buurtcode", "buurtnaam", "gemeentenaam", "geometry")
                if c in mcda_in_mask.columns
            ]
            mcda_neigh = mcda_in_mask[keep_cols].copy()
            mcda_neigh = mcda_neigh.sort_values("MCDA", ascending=False).reset_index(
                drop=True
            )

            # Prepare household points (assumes households.var.household_points exists)
            hp = self.agents.households.var.household_points.copy()
            if hp.crs != mcda_neigh.crs:
                hp = hp.to_crs(mcda_neigh.crs)

            # Ensure `buurtcode` exists on household points (assumes earlier spatial join)
            if "buurtcode" not in hp.columns:
                # attempt a spatial join to assign buurtcode to household points
                try:
                    hp = gpd.sjoin(
                        hp,
                        mcda_neigh[["buurtcode", "geometry"]],
                        how="left",
                        predicate="within",
                    )
                    hp = hp.drop(
                        columns=[c for c in ("index_right",) if c in hp.columns]
                    )
                except Exception:
                    raise RuntimeError(
                        "household_points do not contain 'buurtcode' and spatial join failed"
                    )

            # Iterate neighborhoods in order, select 5% randomly, apply subsidy until pot exhausted
            allocations = []

            # Map household GeoDataFrame index to integer household index used by `Households` arrays
            # This dummy assumes hp.index aligns with households ordering or contains a 'household_id' column
            if "household_id" in hp.columns:
                hp_index_to_household = hp["household_id"].astype(int).values
            else:
                # best-effort: assume sequential alignment
                hp_index_to_household = np.arange(len(hp), dtype=int)

            all_eligible_households = np.zeros(self.agents.households.n, dtype=bool)

            for _, neigh in mcda_neigh.iterrows():
                if subsidy_pot <= 0:
                    break

                buurt = neigh.get("buurtcode")
                buurtnaam = neigh.get("buurtnaam")
                gemeentenaam = neigh.get("gemeentenaam")
                mcda_score = neigh.get("MCDA")

                # households in this neighborhood (local indices into hp)
                idxs = hp.index[hp["buurtcode"] == buurt].to_numpy()
                n_house = len(idxs)
                if n_house == 0:
                    continue

                k = max(1, int(np.ceil(n_house * conversion_rate)))
                chosen_local = rng.choice(idxs, size=min(k, n_house), replace=False)

                # map chosen_local indices to model household ids (requires stable mapping)
                chosen_household_ids = hp_index_to_household[chosen_local.astype(int)]

                # compute newly selected (not already subsidized and not already adapted)
                # exclude households that already took an adaptation measure
                adapted_households = (
                    self.agents.households.var.adaptation_type.data != 0
                )
                newly_selected = np.setdiff1d(
                    chosen_household_ids,
                    np.nonzero(all_eligible_households | adapted_households)[0],
                    assume_unique=False,
                )
                n_new = len(newly_selected)
                if n_new == 0:
                    continue

                incremental_cost = n_new * (total_costs_per_household)
                if incremental_cost > subsidy_pot:
                    affordable = int(subsidy_pot // total_costs_per_household)
                    if affordable <= 0:
                        break
                    # pick a random subset of the newly_selected to fit budget
                    newly_selected = rng.choice(
                        newly_selected, size=affordable, replace=False
                    )
                    n_new = len(newly_selected)
                    incremental_cost = n_new * total_costs_per_household

                # update cumulative mask
                all_eligible_households[newly_selected.astype(int)] = True

                # Automatically force adaptation to dryproofing (1) for newly selected households
                hh = self.agents.households
                idxs = newly_selected.astype(int)
                hh.var.adaptation_type.data[idxs] = 1
                hh.var.adapted.data[idxs] = 1
                hh.var.time_adapted.data[idxs] = 1
                try:
                    hh.update_building_adaptation_status(hh.var.adaptation_type.data)
                except Exception:
                    pass

                subsidy_pot -= float(incremental_cost)

                allocations.append(
                    {
                        "buurtcode": buurt,
                        "buurtnaam": buurtnaam,
                        "gemeentenaam": gemeentenaam,
                        "MCDA": mcda_score,
                        "n_households": n_house,
                        "n_selected": int(n_new),
                        "cost": float(incremental_cost),
                        "remaining_pot": float(subsidy_pot),
                    }
                )

                print(allocations[-1])  # log each allocation step

    def step(self) -> None:
        """This function is run each timestep."""
        if self.model.current_timestep == 0 and self.config.get("plant_forest", False):
            self.prepare_modified_soil_maps_for_forest()

        self.set_irrigation_limit()
        self.provide_subsidies()
        self.provide_risk_communication()
        self.provide_subsidies_to_vulnerable_households()
        self.provide_ontzorgen_to_vulnerable_households()

        self.report(locals())

    def prepare_modified_soil_maps_for_forest(self) -> None:
        """Plant forest: update soil properties in memory and remove displaced farmers.

        Loads the forest restoration potential at grid scale, applies a threshold
        to identify suitable HRUs, copies mean soil property values from existing forest
        HRUs to suitable HRUs, saves a figure, and removes farmers from converted areas.
        The threshold is read from the config key ``forest_restoration_potential_threshold``
        and defaults to 0.5.
        """
        hydrology = self.model.hydrology
        plant_forest_config = self.config.get("plant_forest", {})
        threshold = (
            plant_forest_config.get("forest_restoration_potential_threshold", 0.5)
            if isinstance(plant_forest_config, dict)
            else 0.5
        )

        forest_potential = hydrology.grid.load(
            self.model.files["grid"]["landsurface/forest_restoration_potential_ratio"]
        )
        suitability_grid = forest_potential >= threshold
        suitability_HRU = hydrology.to_HRU(data=suitability_grid).astype(bool)

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

        output_folder = self.model.output_folder / "forest_planting"
        output_folder.mkdir(parents=True, exist_ok=True)
        self._save_forest_planting_figure(
            land_use_type_before, suitability_HRU, output_folder
        )

    def _save_forest_planting_figure(
        self,
        land_use_type_before: np.ndarray,
        suitability_HRU: np.ndarray,
        output_folder: Path,
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
        axes[1, 0].set_title("Reforestation Suitability (50% threshold)")
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
