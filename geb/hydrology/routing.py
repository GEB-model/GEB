# --------------------------------------------------------------------------------
# This file contains code that has been adapted from an original source available
# in a public repository under the GNU General Public License. The original code
# has been modified to fit the specific needs of this project.
#
# Original source repository: https://github.com/iiasa/CWatM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------------


import numpy as np
import numpy.typing as npt
import pyflwdir
from numba import njit

from geb.module import Module
from geb.workflows import balance_check


def get_channel_ratio(river_width, river_length, cell_area):
    return np.minimum(
        1.0,
        river_width * river_length / cell_area,
    )


def create_river_network(
    ldd_uncompressed: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool]
) -> pyflwdir.FlwdirRaster:
    return pyflwdir.from_array(
        ldd_uncompressed,
        ftype="ldd",
        latlon=True,
        mask=mask,
    )


MAX_ITERS: int = 10


class Router:
    def __init__(
        self,
        dt: float | int,
        river_network: pyflwdir.FlwdirRaster,
        Q_initial: np.ndarray,
        waterbody_id: np.ndarray,
        is_waterbody_outflow: np.ndarray,
    ) -> None:
        """Prepare the routing for the model.

        Parameters
        ----------
        dt: float
            The time step in seconds, must be greater than 0.
        river_network: pyflwdir.pyflwdir.FlwdirRaster
            The river network as a FlwdirRaster object, which contains the flow
            direction and other information about the river network.
        Q_initial: np.ndarray
            The initial discharge array, which is a 1D array which is only valid for the masked.
        is_waterbody_outflow: np.ndarray
            A 1D array with the same shape as the grid, which is True for the outflow cells.
        waterbody_id: np.ndarray
            A 1D array with the same shape as the grid, which is the waterbody ID for each cell.

        Sets the following attributes:
        -------------------------------
        upstream_matrix_from_up_to_downstream: np.ndarray
            A 2-D array with the upstream matrix from the river network. The first
            dimension is the number of cells in the river network, and the second
            is the index of the upstream cell in the river network. The value is -1
            if there is no upstream cell. For example, if a cell has two
            upstream cells, the value may be [0, 1, -1, -1].
            Uses masked indices (see below).
        idxs_up_to_downstream: np.ndarray
            Indices of the cells in the river network, sorted from upstream to
            downstream. Of course many orderings are possible, but this is one of
            them with the up- to downstream property.
            Uses masked indices (see below).
        pits: np.ndarray
            The indices of the pits in the river network. These are the cells
            where the flow ends. The value is -1 if there is no pit.
            Uses masked indices (see below).

        Notes:
        -----
        The ldd is a 2D array with the same shape as the grid, where each cell
        contains the flow direction of the cell. The following keys are used:

        |7|8|9|
        |4|5|6|
        |1|2|3|

        - 1: Bottom-left
        - 2: Bottom
        - 3: Bottom-right
        - 4: Left
        - 5: Pit (end of flow)
        - 6: Right
        - 7: Top-left
        - 8: Top
        - 9: Top-right
        - 255: Not defined (no flow)

        All outputs are masked with the mask, so that only the cells that are
        selected in the mask are included. All indices also refer to the index
        in the mask rather than the original ldd.
        """
        assert dt > 0, "dt must be greater than 0"
        self.dt = dt

        # we create a mapper from the 2D ldd to the 1D river network
        # the mapper size is ldd.size + 1, because we need to map the
        # the "nan-value" of the ldd to -1 in the river network, thus
        # mapping -1 to -1.
        mapper: npt.NDArray[np.int32] = np.full(
            river_network.size + 1, -1, dtype=np.int32
        )
        indices: npt.NDArray[np.int64] = np.arange(river_network.size, dtype=np.int32)[
            river_network.mask
        ]
        mapper[indices] = np.arange(indices.size, dtype=np.int32)

        river_network.order_cells(method="walk")
        upstream_matrix: npt.NDArray[np.int32] = pyflwdir.core.upstream_matrix(
            river_network.idxs_ds,
        )

        self.idxs_up_to_downstream: npt.NDarray[np.int32] = river_network.idxs_seq[::-1]

        # make sure all non-selected cells are set to -1
        assert (
            upstream_matrix[
                ~np.isin(np.arange(river_network.size), self.idxs_up_to_downstream)
            ]
            == -1
        ).all()

        self.upstream_matrix_from_up_to_downstream = upstream_matrix[
            self.idxs_up_to_downstream
        ]
        self.upstream_matrix_from_up_to_downstream = mapper[
            self.upstream_matrix_from_up_to_downstream
        ]
        self.idxs_up_to_downstream = mapper[self.idxs_up_to_downstream]

        self.is_pit = np.zeros_like(self.idxs_up_to_downstream, dtype=bool)
        self.is_pit[mapper[river_network.idxs_pit]] = True

        assert is_waterbody_outflow is not None, (
            "is_waterbody_outflow must be provided if waterbody_id is provided"
        )
        assert waterbody_id.shape == self.idxs_up_to_downstream.shape
        self.waterbody_id = waterbody_id

        assert is_waterbody_outflow.shape == self.idxs_up_to_downstream.shape
        # ensurre each waterbody has one outflow (no more, no less)
        assert (
            np.bincount(
                self.waterbody_id[self.waterbody_id != -1],
                weights=is_waterbody_outflow[self.waterbody_id != -1],
            )
            == 1
        ).all()
        self.is_waterbody_outflow = is_waterbody_outflow

        assert Q_initial.shape == self.idxs_up_to_downstream.shape
        assert (Q_initial[self.waterbody_id != -1] == 0).all()
        self.Q_prev = Q_initial

    def get_total_storage(self) -> npt.NDArray[np.float32]:
        """Get the total storage of the river network, which is the sum of the available storage in each cell."""
        return self.get_available_storage(maximum_abstraction_ratio=1.0)


@njit(cache=True)
def update_node_kinematic(
    Qin, Qold, q, alpha, beta, deltaT, deltaX, epsilon=np.float32(0.0001)
) -> np.float32:
    # If there's no inflow, no previous flow, and no lateral inflow,
    # then the discharge at the new time step will be zero.
    if (Qin + Qold + q) < 1e-30:
        return np.float32(1e-30)

    Qin: np.float32 = max(Qin, np.float32(1e-30))

    # Common terms
    ab_pQ: np.float32 = alpha * beta * ((Qold + Qin) / 2) ** (beta - 1)
    deltaTX: np.float32 = deltaT / deltaX
    C: np.float32 = deltaTX * Qin + alpha * Qold**beta + deltaT * q

    # Initial guess for Qkx and iterative process
    Qkx: np.float32 = (deltaTX * Qin + Qold * ab_pQ + deltaT * q) / (deltaTX + ab_pQ)
    Qkx: np.float32 = max(Qkx, np.float32(1e-30))

    # Newton-Raphson method
    fQkx: np.float32 = deltaTX * Qkx + alpha * Qkx**beta - C

    # Get the derivative
    dfQkx: np.float32 = deltaTX + alpha * beta * Qkx ** (beta - 1)
    Qkx -= fQkx / dfQkx
    Qkx: np.float32 = max(Qkx, np.float32(1e-30))

    count: int = 0
    while np.abs(fQkx) > epsilon and count < MAX_ITERS:
        fQkx: np.float32 = deltaTX * Qkx + alpha * Qkx**beta - C
        dfQkx: np.float32 = deltaTX + alpha * beta * Qkx ** (beta - 1)
        Qkx -= fQkx / dfQkx
        Qkx: np.float32 = max(Qkx, np.float32(1e-30))

        count += 1

    assert not np.isnan(Qkx), "Qkx is NaN"

    return Qkx


class KinematicWave(Router):
    def __init__(
        self,
        dt,
        river_network,
        Q_initial,
        river_width,
        river_length,
        river_alpha,
        river_beta,
        waterbody_id,
        is_waterbody_outflow,
    ):
        super().__init__(
            dt, river_network, Q_initial, waterbody_id, is_waterbody_outflow
        )

        self.river_width = river_width.ravel()
        self.river_length = river_length.ravel()
        self.river_alpha = river_alpha.ravel()
        self.river_beta = river_beta

    def calculate_river_storage_from_discharge(
        self,
        discharge: npt.NDArray[np.float32],
        river_alpha: npt.NDArray[np.float32],
        river_length: npt.NDArray[np.float32],
        river_beta: float,
        waterbody_id: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float32]:
        # The momentum equation, see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
        cross_sectional_area_of_flow: npt.NDArray[np.float32] = (
            river_alpha * discharge**river_beta
        )
        river_storage: npt.NDArray[np.float32] = (
            cross_sectional_area_of_flow * river_length
        )
        river_storage[waterbody_id != -1] = 0.0
        return river_storage

    def calculate_discharge_from_storage(
        self, river_storage, river_alpha, river_length, river_beta
    ):
        # The momentum equation (solved for Q), see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
        return (river_storage / (river_length * river_alpha)) ** (1 / river_beta)

    def get_available_storage(
        self, maximum_abstraction_ratio: float = 0.9
    ) -> npt.NDArray[np.float32]:
        """Get the available storage of the river network, which is the sum of the available storage in each cell.

        Args:
            maximum_abstraction_ratio: he maximum abstraction ratio, default is 0.9.
                This is the ratio of the available storage that can be used for abstraction.

        Returns:
            The available storage of the river network.
        """
        assert not np.isnan(self.Q_prev).any()
        assert (self.Q_prev >= 0.0).all()

        river_storage: npt.NDArray[np.float32] = (
            self.calculate_river_storage_from_discharge(
                discharge=self.Q_prev,
                river_alpha=self.river_alpha,
                river_length=self.river_length,
                river_beta=self.river_beta,
                waterbody_id=self.waterbody_id,
            )
        )
        return river_storage * maximum_abstraction_ratio

    @staticmethod
    @njit(cache=True)
    def _step(
        Qold,
        sideflow_m3,
        waterbody_storage_m3,
        outflow_per_waterbody_m3,
        upstream_matrix_from_up_to_downstream,
        idxs_up_to_downstream,
        is_waterbody_outflow,
        waterbody_id,
        river_alpha,
        river_beta,
        river_length,
        dt,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Kinematic wave routing.

        Parameters
        ----------
        dt: float
            Time step, must be > 0
        river_length: np.ndarray
            Array of floats containing the channel length, must be > 0
        """
        Qnew: npt.NDArray[np.float32] = np.zeros_like(Qold)
        over_abstraction_m3: npt.NDArray[np.float32] = np.zeros_like(
            Qold, dtype=np.float32
        )

        for i in range(upstream_matrix_from_up_to_downstream.shape[0]):
            node: np.int32 = idxs_up_to_downstream[i]
            upstream_nodes: npt.NDArray[np.int32] = (
                upstream_matrix_from_up_to_downstream[i]
            )

            Qin: np.float32 = np.float32(0.0)
            sideflow_node_m3: np.float32 = sideflow_m3[node]

            for upstream_node in upstream_nodes:
                if upstream_node == -1:
                    break

                if is_waterbody_outflow[upstream_node]:
                    # if upstream node is an outflow add the outflow of the waterbody
                    # to the sideflow
                    upstream_node_waterbody_id = waterbody_id[upstream_node]

                    # make sure that the waterbody ID is valid
                    assert upstream_node_waterbody_id != -1
                    waterbody_outflow_m3 = outflow_per_waterbody_m3[
                        upstream_node_waterbody_id
                    ]

                    waterbody_storage_m3[upstream_node_waterbody_id] -= (
                        waterbody_outflow_m3
                    )

                    # make sure that the waterbody storage does not go below 0
                    assert waterbody_storage_m3[upstream_node_waterbody_id] >= 0

                    sideflow_node_m3 += waterbody_outflow_m3

                elif (
                    waterbody_id[upstream_node] != -1
                ):  # if upstream node is a waterbody, but not an outflow
                    assert sideflow_m3[upstream_node] == 0

                else:  # in normal case, just take the inflow from upstream
                    assert not np.isnan(Qnew[upstream_node])
                    Qin += Qnew[upstream_node]

            node_waterbody_id: np.int32 = waterbody_id[node]
            if node_waterbody_id != -1:
                waterbody_storage_m3[node_waterbody_id] += Qin * dt
                waterbody_storage_m3[node_waterbody_id] += sideflow_node_m3
            else:
                Qnew[node] = update_node_kinematic(
                    Qin,
                    Qold[node],
                    sideflow_node_m3 / dt / river_length[node],
                    river_alpha[node],
                    river_beta,
                    dt,
                    river_length[node],
                )
        return Qnew, over_abstraction_m3

    def step(
        self,
        sideflow_m3,
        waterbody_storage_m3,
        outflow_per_waterbody_m3,
    ):
        Q, over_abstraction_m3 = self._step(
            Qold=self.Q_prev,
            sideflow_m3=sideflow_m3,
            waterbody_storage_m3=waterbody_storage_m3,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            upstream_matrix_from_up_to_downstream=self.upstream_matrix_from_up_to_downstream,
            idxs_up_to_downstream=self.idxs_up_to_downstream,
            is_waterbody_outflow=self.is_waterbody_outflow,
            waterbody_id=self.waterbody_id,
            river_alpha=self.river_alpha,
            river_beta=self.river_beta,
            river_length=self.river_length,
            dt=self.dt,
        )
        assert (Q[self.waterbody_id != -1] == 0.0).all()

        self.Q_prev = Q

        outflow_at_pits_m3 = (Q[self.is_pit] * self.dt).sum()

        return Q, over_abstraction_m3, waterbody_storage_m3, outflow_at_pits_m3


class Accuflux(Router):
    def __init__(self, dt, river_network, *args, **kwargs):
        super().__init__(dt, river_network, *args, **kwargs)

    def get_available_storage(
        self, maximum_abstraction_ratio=0.9
    ) -> npt.NDArray[np.float32]:
        assert not np.isnan(self.Q_prev).any()
        assert (self.Q_prev >= 0.0).all()
        return self.Q_prev * self.dt * maximum_abstraction_ratio

    @staticmethod
    @njit(cache=True)
    def _step(
        dt,
        Qold,
        sideflow_m3,
        waterbody_storage_m3,
        outflow_per_waterbody_m3,
        upstream_matrix_from_up_to_downstream,
        idxs_up_to_downstream,
        is_waterbody_outflow,
        waterbody_id,
    ):
        Qold += sideflow_m3 / dt
        Qnew: np.ndarray = np.full_like(Qold, 0.0)
        over_abstraction_m3: np.ndarray = np.zeros_like(Qold, dtype=np.float32)
        for i in range(upstream_matrix_from_up_to_downstream.shape[0]):
            node = idxs_up_to_downstream[i]
            upstream_nodes = upstream_matrix_from_up_to_downstream[i]

            inflow_volume = np.float32(0.0)

            for upstream_node in upstream_nodes:
                if upstream_node == -1:
                    break

                if is_waterbody_outflow[upstream_node]:
                    # if upstream node is an outflow add the outflow of the waterbody
                    # to the sideflow
                    upstream_node_waterbody_id = waterbody_id[upstream_node]

                    # make sure that the waterbody ID is valid
                    assert upstream_node_waterbody_id != -1
                    waterbody_outflow_m3 = outflow_per_waterbody_m3[
                        upstream_node_waterbody_id
                    ]

                    waterbody_storage_m3[upstream_node_waterbody_id] -= (
                        waterbody_outflow_m3
                    )

                    # make sure that the waterbody storage does not go below 0
                    assert waterbody_storage_m3[upstream_node_waterbody_id] >= 0

                    inflow_volume += waterbody_outflow_m3

                elif (
                    waterbody_id[upstream_node] != -1
                ):  # if upstream node is a waterbody, but not an outflow
                    assert sideflow_m3[upstream_node] == 0

                else:  # in normal case, just take the inflow from upstream
                    inflow_volume += Qold[upstream_node] * dt

            node_waterbody_id = waterbody_id[node]
            if node_waterbody_id != -1:
                waterbody_storage_m3[node_waterbody_id] += inflow_volume
            else:
                Qnew_node = inflow_volume / dt
                if Qnew_node < 0.0:
                    # if the new discharge is negative, we have over-abstraction
                    over_abstraction_m3[node] = -Qnew_node * dt
                    Qnew_node = 0.0
                Qnew[node] = Qnew_node
                assert Qnew[node] >= 0.0, "Discharge cannot be negative"
        return Qnew, over_abstraction_m3

    def step(
        self,
        sideflow_m3,
        waterbody_storage_m3,
        outflow_per_waterbody_m3,
    ):
        outflow_at_pits_m3 = (
            self.get_total_storage()[self.is_pit].sum() + sideflow_m3[self.is_pit].sum()
        )
        Q, over_abstraction_m3 = self._step(
            dt=self.dt,
            Qold=self.Q_prev,
            sideflow_m3=sideflow_m3,
            waterbody_storage_m3=waterbody_storage_m3,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            upstream_matrix_from_up_to_downstream=self.upstream_matrix_from_up_to_downstream,
            idxs_up_to_downstream=self.idxs_up_to_downstream,
            is_waterbody_outflow=self.is_waterbody_outflow,
            waterbody_id=self.waterbody_id,
        )
        assert (Q[self.waterbody_id != -1] == 0.0).all()
        self.Q_prev = Q
        return Q, over_abstraction_m3, waterbody_storage_m3, outflow_at_pits_m3


class Routing(Module):
    def __init__(self, model, hydrology):
        super().__init__(model)

        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.ldd: npt.NDArray[np.uint8] = self.grid.load(
            self.model.files["grid"]["routing/ldd"],
        )

        if self.model.in_spinup:
            self.spinup()

        mask: npt.NDArray[np.bool] = ~self.grid.mask

        ldd_uncompressed: npt.NDArray[np.uint8] = np.full_like(
            mask, 255, dtype=self.ldd.dtype
        )
        ldd_uncompressed[mask] = self.ldd.ravel()

        self.river_network: pyflwdir.FlwdirRaster = create_river_network(
            ldd_uncompressed=ldd_uncompressed, mask=mask
        )

    def set_router(self):
        routing_algorithm: str = self.model.config["hydrology"]["routing"]["algorithm"]
        is_waterbody_outflow: npt.NDArray[bool] = (
            self.grid.var.waterbody_outflow_points != -1
        )
        if routing_algorithm == "kinematic_wave":
            self.router = KinematicWave(
                dt=self.var.routing_step_length_seconds,
                river_network=self.river_network,
                Q_initial=self.grid.var.discharge_m3_s,
                river_width=self.grid.var.river_width,
                river_length=self.grid.var.river_length,
                river_alpha=self.grid.var.river_alpha,
                river_beta=self.var.river_beta,
                waterbody_id=self.grid.var.waterBodyID,
                is_waterbody_outflow=is_waterbody_outflow,
            )
        elif routing_algorithm == "accuflux":
            self.router = Accuflux(
                dt=self.var.routing_step_length_seconds,
                river_network=self.river_network,
                Q_initial=self.grid.var.discharge_m3_s,
                waterbody_id=self.grid.var.waterBodyID,
                is_waterbody_outflow=is_waterbody_outflow,
            )
        else:
            raise ValueError(
                f"Unknown routing algorithm: {routing_algorithm}. "
                "Available algorithms are 'kinematic_wave' and 'accuflux'."
            )

    def spinup(self):
        self.grid.var.upstream_area = self.grid.load(
            self.model.files["grid"]["routing/upstream_area"]
        )

        # number of substep per day
        self.var.n_routing_substeps: int = 24
        # kinematic wave parameter: 0.6 is for broad sheet flow

        self.var.river_beta: float = 0.6  # TODO: Make this a parameter

        # Channel Manning's n
        self.grid.var.river_mannings = (
            self.grid.load(self.model.files["grid"]["routing/mannings"])
            * self.model.config["parameters"]["manningsN"]
        )
        assert (self.grid.var.river_mannings > 0).all()

        # Channel length [meters]
        self.grid.var.river_length = self.grid.load(
            self.model.files["grid"]["routing/river_length"]
        )

        # where there is a pit, the river length is set to distance to the center of the cell,
        # thus half of the sqrt of the cell area
        self.grid.var.river_length[self.ldd == 5] = (
            np.sqrt(self.grid.var.cell_area[self.ldd == 5]) / 2
        )
        assert (self.grid.var.river_length > 0).all(), (
            "Channel length must be greater than 0 for all cells"
        )

        # Channel bottom width [meters]
        self.grid.var.river_width = self.grid.load(
            self.model.files["grid"]["routing/river_width"]
        )

        # Corresponding sub-timestep (seconds)
        self.var.routing_step_length_seconds = (
            self.model.timestep_length.total_seconds() / self.var.n_routing_substeps
        )

        # for a river, the wetted perimeter can be approximated by the channel width
        river_wetted_perimeter = self.grid.var.river_width

        # Channel gradient (fraction, dy/dx)
        minimum_river_slope = 0.0001
        river_slope = np.maximum(
            self.grid.load(self.model.files["grid"]["routing/river_slope"]),
            minimum_river_slope,
        )

        # river_alpha for kinematic wave
        # source: https://gmd.copernicus.org/articles/13/3267/2020/ eq. 21
        self.grid.var.river_alpha = (
            self.grid.var.river_mannings
            * river_wetted_perimeter ** (2 / 3)
            / np.sqrt(river_slope)
        ) ** self.var.river_beta

        # Initialize discharge with zero
        self.grid.var.discharge_m3_s = self.grid.full_compressed(0, dtype=np.float32)
        self.grid.var.discharge_m3_s_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge_m3_s.size),
            0,
            dtype=self.grid.var.discharge_m3_s.dtype,
        )

    def step(
        self,
        total_runoff: np.ndarray,
        channel_abstraction_m3: np.ndarray,
        return_flow: np.ndarray,
    ):
        if __debug__:
            pre_storage: np.ndarray = self.hydrology.lakes_reservoirs.var.storage.copy()
            pre_river_storage_m3: npt.NDArray[np.float32] = (
                self.router.get_total_storage()
            )

        channel_abstraction_m3_per_routing_step: np.ndarray = (
            channel_abstraction_m3 / self.var.n_routing_substeps
        )
        assert (
            channel_abstraction_m3_per_routing_step[self.grid.var.waterBodyID != -1]
            == 0.0
        ).all(), (
            "Channel abstraction must be zero for water bodies, "
            "but found non-zero value."
        )

        return_flow_m3_per_routing_step: np.ndarray = (
            return_flow * self.grid.var.cell_area / self.var.n_routing_substeps
        )

        # add return flow to the water bodies
        return_flow_m3_to_water_bodies_per_routing_step: np.ndarray = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=return_flow_m3_per_routing_step[self.grid.var.waterBodyID != -1],
        )
        return_flow_m3_per_routing_step[self.grid.var.waterBodyID != -1] = 0.0

        runoff_m3_per_routing_step: np.ndarray = (
            total_runoff * self.grid.var.cell_area / self.var.n_routing_substeps
        )

        # add runoff to the water bodies
        runoff_m3_per_routing_step_water_bodies = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=runoff_m3_per_routing_step[self.grid.var.waterBodyID != -1],
        )

        runoff_m3_per_routing_step[self.grid.var.waterBodyID != -1] = 0.0

        self.grid.var.discharge_m3_s_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge_m3_s.size),
            np.nan,
            dtype=self.grid.var.discharge_m3_s.dtype,
        )

        potential_evaporation_per_water_body_m3_per_routing_step = (
            np.bincount(
                self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                weights=self.grid.var.EWRef[self.grid.var.waterBodyID != -1],
            )
            / np.bincount(self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1])
            * self.hydrology.lakes_reservoirs.var.lake_area
        ) / self.var.n_routing_substeps

        # the ratio of each grid cell that is currently covered by a river
        channel_ratio: np.ndarray = get_channel_ratio(
            river_length=self.grid.var.river_length,
            river_width=self.grid.var.river_width,
            cell_area=self.grid.var.cell_area,
        )

        # calculate evaporation from rivers per timestep usting the current channel ratio
        potential_evaporation_in_rivers_m3_per_routing_step = (
            self.grid.var.EWRef * channel_ratio * self.grid.var.cell_area
        ) / self.var.n_routing_substeps

        if __debug__:
            # these are for balance checks, the sum of all routing steps
            evaporation_in_rivers_m3 = 0
            waterbody_evaporation_m3 = 0
            outflow_at_pits_m3 = 0
            over_abstraction_m3 = 0
            command_area_release_m3 = 0

        for subrouting_step in range(self.var.n_routing_substeps):
            self.hydrology.lakes_reservoirs.var.storage += (
                runoff_m3_per_routing_step_water_bodies
            )

            self.hydrology.lakes_reservoirs.var.storage += (
                return_flow_m3_to_water_bodies_per_routing_step
            )

            actual_evaporation_from_water_bodies_per_routing_step_m3 = np.minimum(
                potential_evaporation_per_water_body_m3_per_routing_step,
                self.hydrology.lakes_reservoirs.var.storage,
            )

            self.hydrology.lakes_reservoirs.var.storage -= (
                actual_evaporation_from_water_bodies_per_routing_step_m3
            )

            outflow_per_waterbody_m3, command_area_release_m3_routing_step = (
                self.hydrology.lakes_reservoirs.substep(
                    current_substep=subrouting_step,
                    n_routing_substeps=self.var.n_routing_substeps,
                    routing_step_length_seconds=self.var.routing_step_length_seconds,
                )
            )

            self.hydrology.lakes_reservoirs.var.storage -= (
                command_area_release_m3_routing_step
            )

            assert (
                outflow_per_waterbody_m3 <= self.hydrology.lakes_reservoirs.var.storage
            ).all(), "outflow cannot be smaller or equal to storage"

            side_flow_channel_m3_per_routing_step = (
                runoff_m3_per_routing_step
                + return_flow_m3_per_routing_step
                - channel_abstraction_m3_per_routing_step
            )
            assert (
                side_flow_channel_m3_per_routing_step[self.grid.var.waterBodyID != -1]
                == 0
            ).all()

            evaporation_in_rivers_m3_per_routing_step: np.ndarray = np.minimum(
                self.router.get_total_storage() + side_flow_channel_m3_per_routing_step,
                potential_evaporation_in_rivers_m3_per_routing_step,
            )
            evaporation_in_rivers_m3_per_routing_step: np.ndarray = np.maximum(
                evaporation_in_rivers_m3_per_routing_step, 0
            )
            assert (
                evaporation_in_rivers_m3_per_routing_step[
                    self.grid.var.waterBodyID != -1
                ]
                == 0
            ).all()

            side_flow_channel_m3_per_routing_step -= (
                evaporation_in_rivers_m3_per_routing_step
            )

            (
                self.grid.var.discharge_m3_s,
                over_abstraction_m3_routing_step,
                self.hydrology.lakes_reservoirs.var.storage,
                outflow_at_pits_m3_routing_step,
            ) = self.router.step(
                sideflow_m3=side_flow_channel_m3_per_routing_step.astype(np.float32),
                waterbody_storage_m3=self.hydrology.lakes_reservoirs.var.storage,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            )

            assert (
                self.grid.var.discharge_m3_s[self.grid.var.waterBodyID != -1] == 0
            ).all()

            self.grid.var.discharge_m3_s_substep[subrouting_step, :] = (
                self.grid.var.discharge_m3_s.copy()
            )

            assert (self.router.get_available_storage() >= 0.0).all()

            if __debug__:
                # Discharge at outlets and lakes and reservoirs
                outflow_at_pits_m3 += outflow_at_pits_m3_routing_step
                waterbody_evaporation_m3 += (
                    actual_evaporation_from_water_bodies_per_routing_step_m3
                )
                evaporation_in_rivers_m3 += evaporation_in_rivers_m3_per_routing_step
                over_abstraction_m3 += over_abstraction_m3_routing_step
                command_area_release_m3 += command_area_release_m3_routing_step

        assert not np.isnan(self.grid.var.discharge_m3_s).any()

        if __debug__:
            # TODO: make dependent on routing step length
            river_storage_m3: npt.NDArray[np.float32] = self.router.get_total_storage()
            balance_check(
                how="sum",
                influxes=[
                    total_runoff * self.grid.var.cell_area,
                    return_flow * self.grid.var.cell_area,
                    over_abstraction_m3,
                ],
                outfluxes=[
                    channel_abstraction_m3,
                    outflow_at_pits_m3,
                    evaporation_in_rivers_m3,
                    waterbody_evaporation_m3,
                    command_area_release_m3,
                ],
                prestorages=[
                    pre_storage,
                    pre_river_storage_m3,
                ],
                poststorages=[
                    self.hydrology.lakes_reservoirs.var.storage,
                    river_storage_m3,
                ],
                name="routing_1",
                tollerance=100,
            )

            self.routing_loss: np.float64 = (
                evaporation_in_rivers_m3.sum()
                + waterbody_evaporation_m3.sum()
                + outflow_at_pits_m3.sum()
            )

            assert self.routing_loss >= 0, "Routing loss cannot be negative"

        self.report(self, locals())

    @property
    def name(self) -> str:
        return "hydrology.routing"
