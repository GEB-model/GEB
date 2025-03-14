import numpy as np
from .general import AgentBaseClass
from ..store import DynamicArray

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
        if fill <= conservative_limit_ratio[i] * 2:
            outflow[i] = min(minQ[i], current_storage[i] * day_to_sec)
        elif fill <= normal_limit_ratio[i]:
            outflow[i] = minQ[i] + (normQ[i] - minQ[i]) * (
                fill - 2 * conservative_limit_ratio[i]
            ) / (normal_limit_ratio[i] - 2 * conservative_limit_ratio[i])
        elif fill <= flood_limit_ratio[i]:
            outflow[i] = normQ[i] + (
                (fill - normal_limit_ratio[i])
                / (flood_limit_ratio[i] - normal_limit_ratio[i])
            ) * (nondmgQ[i] - normQ[i])
        else:
            outflow[i] = max(
                max(
                    (fill - flood_limit_ratio[i] - 0.01) * volume[i] * day_to_sec,
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

        if self.model.in_spinup:
            self.spinup()

        AgentBaseClass.__init__(self)
        super().__init__()

    def spinup(self):
        self.var = self.model.store.create_bucket("agents.reservoir_operators.var")
        self.set_reservoir_data(
            self.model.hydrology.lakes_reservoirs.var.water_body_data
        )

    def set_reservoir_data(self, water_body_data):
        self.reservoirs = water_body_data[water_body_data["waterbody_type"] == 2].copy()
        assert (self.reservoirs["volume_total"] > 0).all()
        self.var.active_reservoirs = self.reservoirs[
            self.reservoirs["waterbody_type"] == 2
        ]

        self.var.reservoir_release_factors = DynamicArray(
            np.full(
                len(self.var.active_reservoirs),
                self.model.config["agent_settings"]["reservoir_operators"][
                    "max_reservoir_release_factor"
                ],
            )
        )

        self.var.reservoir_volume = DynamicArray(
            self.var.active_reservoirs["volume_total"].values
        )
        self.var.flood_volume = DynamicArray(
            self.var.active_reservoirs["volume_flood"].values
        )
        self.var.dis_avg = DynamicArray(
            self.var.active_reservoirs["average_discharge"].values
        )
        self.var.norm_limit_ratio = DynamicArray(
            self.var.flood_volume / self.var.reservoir_volume
        )
        self.var.cons_limit_ratio = DynamicArray(
            np.full(len(self.var.active_reservoirs), 0.02, dtype=np.float32)
        )
        self.var.flood_limit_ratio = DynamicArray(
            np.full(len(self.var.active_reservoirs), 1.0, dtype=np.float32)
        )

        self.var.minQC = DynamicArray(
            self.model.config["agent_settings"]["reservoir_operators"]["MinOutflowQ"]
            * self.var.dis_avg
        )
        self.var.normQC = DynamicArray(
            self.model.config["agent_settings"]["reservoir_operators"]["NormalOutflowQ"]
            * self.var.dis_avg
        )
        self.var.nondmgQC = DynamicArray(
            self.model.config["agent_settings"]["reservoir_operators"][
                "NonDamagingOutflowQ"
            ]
            * self.var.dis_avg
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
        # assert that the reservoir IDs match the active reservoirs
        assert np.array_equal(waterBodyIDs, self.var.active_reservoirs.index.to_numpy())

        # make outflow same as inflow for a setting without a reservoir
        if "ruleset" in self.config and self.config["ruleset"] == "no-human-influence":
            return inflow.copy()

        reservoir_outflow = regulate_reservoir_outflow(
            reservoirStorageM3,
            self.var.reservoir_volume.data,
            inflow,
            self.var.minQC.data,
            self.var.normQC.data,
            self.var.nondmgQC.data,
            self.var.cons_limit_ratio.data,
            self.var.norm_limit_ratio.data,
            self.var.flood_limit_ratio.data,
        )
        assert (reservoir_outflow >= 0).all()

        return reservoir_outflow

    def get_available_water_reservoir_command_areas(self, reservoir_storage_m3):
        return self.var.reservoir_release_factors * reservoir_storage_m3

    def step(self) -> None:
        return None

    @property
    def storage(self):
        return self.hydrology.grid.storage
