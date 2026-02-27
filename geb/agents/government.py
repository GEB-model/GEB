"""This module contains the Government agent class for GEB."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
        elif frequency != "always":
            raise ValueError("subsidies.frequency must be 'yearly' or 'always'")

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
        elif frequency != "always":
            raise ValueError(
                "risk_communication.frequency must be 'yearly' or 'always'"
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
        self.agents.households.apply_risk_communication(
            percentage_increase=percentage_increase_risk_perception,
            household_mask=eligible_mask,
        )

    def step(self) -> None:
        """This function is run each timestep."""
        self.set_irrigation_limit()
        self.provide_subsidies()
        self.provide_risk_communication()
        self.report(locals())
