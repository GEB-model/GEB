# -*- coding: utf-8 -*-
from . import AgentBaseClass
import numpy as np
import pandas as pd
from .general import AgentArray

from numba import njit


@njit(cache=True)
def regulate_reservoir_outflow(
    current_storage,
    volume,
    inflow,
    minQ,
    normQ,
    nondmgQ,
    conservative_limit_ratio,
    normal_limit_ratio,
    flood_limit_ratio,
):
    day_to_sec = 1 / (24 * 60 * 60)
    outflow = np.zeros_like(current_storage)
    for i in range(current_storage.size):
        fill = current_storage[i] / volume[i]
        if fill <= conservative_limit_ratio * 2:
            outflow[i] = min(minQ[i], current_storage[i] * day_to_sec)
        elif fill <= normal_limit_ratio[i]:
            outflow[i] = minQ[i] + (normQ[i] - minQ[i]) * (
                fill - 2 * conservative_limit_ratio
            ) / (normal_limit_ratio[i] - 2 * conservative_limit_ratio)
        elif fill <= flood_limit_ratio:
            outflow[i] = normQ[i] + (
                (fill - normal_limit_ratio[i])
                / (flood_limit_ratio - normal_limit_ratio[i])
            ) * (nondmgQ[i] - normQ[i])
        else:
            outflow[i] = max(
                max(
                    (fill - flood_limit_ratio - 0.01) * volume[i] * day_to_sec,
                    min(nondmgQ[i], np.maximum(inflow[i], normQ[i])),
                ),
                inflow[i],
            )
    return outflow


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
                "average_discharge": float,  # m3/s
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

    def regulate_reservoir_outflow(self, reservoirStorageM3, inflow, waterBodyIDs):
        """Regulate the outflow of the reservoirs.

        Args:
            reservoirStorageM3C: The current storage of the reservoirs in m3.
            inflowC: The inflow of the reservoirs (m/s).
            waterBodyIDs: The IDs of the water bodies.
            delta_t: The time step in seconds.

        Returns:
            The outflow of the reservoirs (m/s).

        """
        assert reservoirStorageM3.size == inflow.size == waterBodyIDs.size

        # make outflow same as inflow for a setting without a reservoir
        if "ruleset" in self.config and self.config["ruleset"] == "no-human-influence":
            return inflow.copy()

        reservoir_outflow = regulate_reservoir_outflow(
            reservoirStorageM3,
            self.reservoir_volume.data,
            inflow,
            self.minQC.data,
            self.normQC.data,
            self.nondmgQC.data,
            self.cons_limit_ratio,
            self.norm_limit_ratio.data,
            self.flood_limit_ratio,
        )

        return reservoir_outflow

    def get_available_water_reservoir_command_areas(self, reservoir_storage_m3):
        return self.reservoir_release_factors * reservoir_storage_m3

    def step(self) -> None:
        return None
