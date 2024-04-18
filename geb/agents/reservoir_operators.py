# -*- coding: utf-8 -*-
from . import AgentBaseClass
import numpy as np
import pandas as pd
from .general import AgentArray


class ReservoirOperators(AgentBaseClass):
    """This class is used to simulate the government.

    Args:
        model: The GEB model.
        agents: The class that includes all agent types (allowing easier communication between agents).
    """

    def __init__(self, model, agents):
        self.model = model
        self.agents = agents
        self.config = (
            self.model.config["agent_settings"]["reservoir_operators"]
            if "reservoir_operators" in self.model.config["agent_settings"]
            else {}
        )
        AgentBaseClass.__init__(self)
        df = pd.read_csv(
            self.model.model_structure["table"][
                "routing/lakesreservoirs/basin_lakes_data"
            ],
            dtype={
                "waterbody_type": int,
                "volume_total": float,
                "average_discharge": float,
                "average_area": float,
                "volume_flood": float,
                "relative_area_in_region": float,
            },
        ).set_index("waterbody_id")

        self.reservoirs = df[df["waterbody_type"] == 2].copy()
        self.cons_limit_ratio = 0.02
        self.flood_limit_ratio = 1

        super().__init__()

    def initiate(self):
        assert (self.reservoirs["volume_total"] > 0).all()
        self.active_reservoirs = self.reservoirs[self.reservoirs["waterbody_type"] == 2]

        np.save(
            self.model.report_folder / "active_reservoirs_waterBodyIDs.npy",
            self.active_reservoirs.index.to_numpy(),
        )

        self.reservoir_release_factors = AgentArray(
            np.full(
                len(self.active_reservoirs),
                self.model.config["agent_settings"]["reservoir_operators"][
                    "max_reservoir_release_factor"
                ],
            )
        )

        self.reservoir_volume = AgentArray(
            self.active_reservoirs["volume_total"].values
        )
        self.flood_volume = AgentArray(self.active_reservoirs["volume_flood"].values)
        self.dis_avg = AgentArray(self.active_reservoirs["average_discharge"].values)

        self.norm_limit_ratio = AgentArray(self.flood_volume / self.reservoir_volume)
        self.norm_flood_limit_ratio = AgentArray(
            self.norm_limit_ratio
            + 0.5 * (self.flood_limit_ratio - self.norm_limit_ratio)
        )

        self.minQC = AgentArray(
            self.model.config["agent_settings"]["reservoir_operators"]["MinOutflowQ"]
            * self.dis_avg
        )
        self.normQC = AgentArray(
            self.model.config["agent_settings"]["reservoir_operators"]["NormalOutflowQ"]
            * self.dis_avg
        )
        self.nondmgQC = AgentArray(
            self.model.config["agent_settings"]["reservoir_operators"][
                "NonDamagingOutflowQ"
            ]
            * self.dis_avg
        )

    def regulate_reservoir_outflow(self, reservoirStorageM3C, inflowC, waterBodyIDs):
        assert reservoirStorageM3C.size == inflowC.size == waterBodyIDs.size

        reservoir_fill = reservoirStorageM3C / self.reservoir_volume
        reservoir_outflow1 = np.minimum(
            self.minQC, reservoirStorageM3C * self.model.InvDtSec
        )
        reservoir_outflow2 = self.minQC + (self.normQC - self.minQC) * (
            reservoir_fill - self.cons_limit_ratio
        ) / (self.norm_limit_ratio - self.cons_limit_ratio)
        reservoir_outflow3 = self.normQC
        temp = np.minimum(self.nondmgQC, np.maximum(inflowC * 1.2, self.normQC))
        reservoir_outflow4 = np.maximum(
            (reservoir_fill - self.flood_limit_ratio - 0.01)
            * self.reservoir_volume
            * self.model.InvDtSec,
            temp,
        )
        reservoir_outflow = reservoir_outflow1.copy()

        reservoir_outflow = np.where(
            reservoir_fill > self.cons_limit_ratio,
            reservoir_outflow2,
            reservoir_outflow,
        )
        reservoir_outflow = np.where(
            reservoir_fill > self.norm_limit_ratio, self.normQC, reservoir_outflow
        )
        reservoir_outflow = np.where(
            reservoir_fill > self.norm_flood_limit_ratio,
            reservoir_outflow3,
            reservoir_outflow,
        )
        reservoir_outflow = np.where(
            reservoir_fill > self.flood_limit_ratio,
            reservoir_outflow4,
            reservoir_outflow,
        )

        temp = np.minimum(reservoir_outflow, np.maximum(inflowC, self.normQC))

        reservoir_outflow = np.where(
            (reservoir_outflow > 1.2 * inflowC)
            & (reservoir_outflow > self.normQC)
            & (reservoir_fill < self.flood_limit_ratio),
            temp,
            reservoir_outflow,
        )

        # make outflow same as inflow for a setting without a reservoir
        if "ruleset" in self.config and self.config["ruleset"] == "no-human-influence":
            reservoir_outflow = inflowC

        return reservoir_outflow

    def get_available_water_reservoir_command_areas(
        self, reservoir_storage_m3, potential_irrigation_consumption_m
    ):
        potential_irrigation_consumption_m3 = (
            potential_irrigation_consumption_m * self.model.data.HRU.cellArea
        )
        potential_irrigation_consumption_m3[self.model.data.HRU.land_owners != -1]
        potential_irrigation_consumption_m3_per_farmer = np.bincount(
            self.model.data.HRU.land_owners[self.model.data.HRU.land_owners != -1],
            weights=potential_irrigation_consumption_m3[
                self.model.data.HRU.land_owners != -1
            ],
            minlength=self.model.agents.farmers.n,
        )
        return self.reservoir_release_factors * reservoir_storage_m3

    def step(self) -> None:
        return None
