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
import pyflwdir
from numba import njit

from geb.module import Module
from geb.workflows import balance_check


def get_channel_ratio(river_width, river_length, cell_area):
    return np.minimum(
        1.0,
        river_width * river_length / cell_area,
    )


MAX_ITERS = 10


class Router:
    def __init__(
        self, dt, ldd, mask, Q_initial, is_waterbody_outflow=None, waterbody_id=None
    ):
        """
        Prepare the routing for the model.

        Parameters
        ----------
        dt: float
            The time step in seconds, must be greater than 0.
        ldd: np.ndarray
            The ldd array, which is a 1D array which is only valid for the masked.
        mask: np.ndarray
            The mask array, which is a 2D array with the same shape as the grid.
        Q_initial: np.ndarray
            The initial discharge array, which is a 1D array which is only valid for the masked.
        is_waterbody_outflow: np.ndarray, optional
            A 1D array with the same shape as the grid, which is True for the outflow cells.
            If not provided, the outflow cells are set to False for all cells.
        waterbody_id: np.ndarray, optional
            A 1D array with the same shape as the grid, which is the waterbody ID for each cell.
            If not provided, the waterbody ID is set to -1 for all cells.

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
        upstream_area_n_cells: np.ndarray
            The upstream area in number of cells for each cell in the river
            network. The value is -1 if there is no upstream area.

        Notes
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

        ldd_uncompressed = np.full_like(mask, 255, dtype=ldd.dtype)
        ldd_uncompressed[mask] = ldd.ravel()

        river_network = pyflwdir.from_array(
            ldd_uncompressed,
            ftype="ldd",
            latlon=True,
            mask=mask,
        )

        # we create a mapper from the 2D ldd to the 1D river network
        # the mapper size is ldd.size + 1, because we need to map the
        # the "nan-value" of the ldd to -1 in the river network, thus
        # mapping -1 to -1.
        mapper = np.full(ldd_uncompressed.size + 1, -1, dtype=np.int32)
        indices = np.arange(ldd_uncompressed.size)[mask.ravel()]
        mapper[indices] = np.arange(indices.size)

        river_network.order_cells(method="walk")
        upstream_matrix = pyflwdir.core.upstream_matrix(
            river_network.idxs_ds,
        )

        self.idxs_up_to_downstream = river_network.idxs_seq[::-1]

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

        self.upstream_area_n_cells = river_network.upstream_area(unit="cell")
        self.upstream_area_n_cells[self.upstream_area_n_cells < 0] = 0
        self.upstream_area_n_cells = self.upstream_area_n_cells[mask]

        if waterbody_id is None:
            self.waterbody_id = np.full_like(
                self.idxs_up_to_downstream, -1, dtype=np.int32
            )
        else:
            assert is_waterbody_outflow is not None, (
                "is_waterbody_outflow must be provided if waterbody_id is provided"
            )
            assert waterbody_id.shape == self.idxs_up_to_downstream.shape
            self.waterbody_id = waterbody_id

        if is_waterbody_outflow is None:
            self.is_waterbody_outflow = np.zeros_like(
                self.idxs_up_to_downstream, dtype=bool
            )
        else:
            assert is_waterbody_outflow.shape == self.idxs_up_to_downstream.shape
            # ensure each waterbody has one outflow (no more, no less)
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

    def get_total_storage(self):
        """
        Get the total storage of the river network, which is the sum of the
        available storage in each cell.
        """
        return self.get_available_storage(maximum_abstraction_ratio=1.0)


@njit(cache=True)
def update_node_kinematic(
    Qin, Qold, q, alpha, beta, deltaT, deltaX, epsilon=np.float64(0.0001)
):
    # If there's no inflow, no previous flow, and no lateral inflow,
    # then the discharge at the new time step will be zero.
    if (Qin + Qold + q) < 1e-30:
        return 1e-30

    # Common terms
    ab_pQ = alpha * beta * ((Qold + Qin) / 2) ** (beta - 1)
    deltaTX = deltaT / deltaX
    C = deltaTX * Qin + alpha * Qold**beta + deltaT * q

    # Initial guess for Qkx and iterative process
    Qkx = (deltaTX * Qin + Qold * ab_pQ + deltaT * q) / (deltaTX + ab_pQ)
    Qkx = max(Qkx, 1e-30)

    # Newton-Raphson method
    fQkx = deltaTX * Qkx + alpha * Qkx**beta - C

    # Get the derivative
    dfQkx = deltaTX + alpha * beta * Qkx ** (beta - 1)
    Qkx -= fQkx / dfQkx
    Qkx = max(Qkx, 1e-30)

    count = 0
    while np.abs(fQkx) > epsilon and count < MAX_ITERS:
        fQkx = deltaTX * Qkx + alpha * Qkx**beta - C
        dfQkx = deltaTX + alpha * beta * Qkx ** (beta - 1)
        Qkx -= fQkx / dfQkx
        Qkx = max(Qkx, 1e-30)

        count += 1

    assert not np.isnan(Qkx), "Qkx is NaN"

    return Qkx


class KinematicWave(Router):
    def __init__(
        self,
        dt,
        ldd,
        mask,
        Q_initial,
        river_width,
        river_length,
        river_alpha,
        river_beta,
    ):
        super().__init__(dt, ldd, mask, Q_initial)

        self.river_width = river_width.ravel()
        self.river_length = river_length.ravel()
        self.river_alpha = river_alpha.ravel()
        self.river_beta = river_beta
        self.dt = dt

    def calculate_river_storage_from_discharge(
        self, discharge, river_alpha, river_length, river_beta, waterbody_id
    ):
        # The momentum equation, see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
        cross_sectional_area_of_flow = river_alpha * discharge**river_beta
        river_storage = cross_sectional_area_of_flow * river_length
        river_storage[waterbody_id != -1] = 0.0
        return river_storage

    def calculate_discharge_from_storage(
        self, river_storage, river_alpha, river_length, river_beta
    ):
        # The momentum equation (solved for Q), see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
        return (river_storage / (river_length * river_alpha)) ** (1 / river_beta)

    def get_available_storage(self, maximum_abstraction_ratio=0.9):
        """
        Get the available storage of the river network, which is the sum of the
        available storage in each cell.
        """
        assert not np.isnan(self.Q_prev).any()
        assert (self.Q_prev >= 0.0).all()

        river_storage = self.calculate_river_storage_from_discharge(
            discharge=self.Q_prev,
            river_alpha=self.river_alpha,
            river_length=self.river_length,
            river_beta=self.river_beta,
            waterbody_id=self.waterbody_id,
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
    ):
        """
        Kinematic wave routing

        Parameters
        ----------
        dt: float
            Time step, must be > 0
        river_length: np.ndarray
            Array of floats containing the channel length, must be > 0
        """
        Qnew = np.full_like(Qold, np.nan)
        over_abstraction_m3 = np.zeros_like(Qold, dtype=np.float32)

        for i in range(upstream_matrix_from_up_to_downstream.shape[0]):
            node = idxs_up_to_downstream[i]
            upstream_nodes = upstream_matrix_from_up_to_downstream[i]

            Qin = np.float32(0.0)
            sideflow_node_m3 = sideflow_m3[node]

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

            Qnew_node = update_node_kinematic(
                Qin,
                Qold[node],
                sideflow_node_m3 / dt / river_length[node],
                river_alpha[node],
                river_beta,
                dt,
                river_length[node],
            )

            node_waterbody_id = waterbody_id[node]
            if node_waterbody_id != -1:
                waterbody_storage_m3[node_waterbody_id] += Qnew_node * dt
            else:
                Qnew[node] = Qnew_node
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

        self.Q_prev = Q

        outflow_at_pits_m3 = (Q[self.is_pit] * self.dt).sum()

        return Q, over_abstraction_m3, waterbody_storage_m3, outflow_at_pits_m3


class Accuflux(Router):
    def __init__(self, dt, ldd, mask, *args, **kwargs):
        super().__init__(dt, ldd, mask, *args, **kwargs)

    def get_available_storage(self, maximum_abstraction_ratio=0.9):
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
        Qnew = np.full_like(Qold, 0.0)
        over_abstraction_m3 = np.zeros_like(Qold, dtype=np.float32)
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
        self.Q_prev = Q
        return Q, over_abstraction_m3, waterbody_storage_m3, outflow_at_pits_m3


class Routing(Module):
    def __init__(self, model, hydrology):
        super().__init__(model)

        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    def set_router(self):
        routing_algorithm = self.model.config["hydrology"]["routing"]["algorithm"]
        if routing_algorithm == "kinematic_wave":
            self.router = KinematicWave(
                dt=self.var.routing_step_length_seconds,
                ldd=self.grid.var.ldd,
                mask=~self.grid.mask,
                Q_initial=self.grid.var.discharge_m3_s,
                river_width=self.grid.var.river_width,
                river_length=self.grid.var.river_length,
                river_alpha=self.grid.var.river_alpha,
                river_beta=self.var.river_beta,
            )
        elif routing_algorithm == "accuflux":
            self.router = Accuflux(
                dt=self.var.routing_step_length_seconds,
                ldd=self.grid.var.ldd,
                mask=~self.grid.mask,
                Q_initial=self.grid.var.discharge_m3_s,
            )
        else:
            raise ValueError(
                f"Unknown routing algorithm: {routing_algorithm}. "
                "Available algorithms are 'kinematic_wave' and 'accuflux'."
            )

    def spinup(self):
        self.grid.var.ldd = self.grid.load(
            self.model.files["grid"]["routing/ldd"],
        )

        self.grid.var.upstream_area = self.grid.load(
            self.model.files["grid"]["routing/upstream_area"]
        )

        # number of substep per day
        self.var.n_routing_substeps = 24
        # kinematic wave parameter: 0.6 is for broad sheet flow

        self.var.river_beta = 0.6  # TODO: Make this a parameter

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
        self.grid.var.river_length[self.grid.var.ldd == 5] = (
            np.sqrt(self.grid.var.cell_area[self.grid.var.ldd == 5]) / 2
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
        self.grid.var.discharge_m3_s = self.grid.full_compressed(
            1e-30, dtype=np.float32
        )
        self.grid.var.discharge_m3_s_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge_m3_s.size),
            1e-30,
            dtype=self.grid.var.discharge_m3_s.dtype,
        )

        self.set_router()

    def step(self, total_runoff, channel_abstraction_m3, return_flow):
        if __debug__:
            pre_storage = self.hydrology.lakes_reservoirs.var.storage.copy()
            pre_river_storage_m3 = self.router.get_total_storage()

        channel_abstraction_m3_per_routing_step = (
            channel_abstraction_m3 / self.var.n_routing_substeps
        )
        assert (
            channel_abstraction_m3_per_routing_step[self.grid.var.waterBodyID != -1]
            == 0.0
        ).all(), (
            "Channel abstraction must be zero for water bodies, "
            "but found non-zero value."
        )

        return_flow_m3_per_routing_step = (
            return_flow * self.grid.var.cell_area / self.var.n_routing_substeps
        )

        # add return flow to the water bodies
        return_flow_m3_to_water_bodies_per_routing_step = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=return_flow_m3_per_routing_step[self.grid.var.waterBodyID != -1],
        )
        return_flow_m3_per_routing_step[self.grid.var.waterBodyID != -1] = 0.0

        runoff_m3_per_routing_step = (
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
        channel_ratio = get_channel_ratio(
            river_length=self.grid.var.river_length,
            river_width=self.grid.var.river_width,
            cell_area=self.grid.var.cell_area,
        )

        # calculate evaporation from rivers per timestep usting the current channel ratio
        potential_evaporation_in_rivers_m3_per_routing_step = (
            self.grid.var.EWRef * channel_ratio * self.grid.var.cell_area
        ) / self.var.n_routing_substeps
        potential_evaporation_in_rivers_m3_per_routing_step.fill(0)

        if __debug__:
            # these are for balance checks, the sum of all routing steps
            evaporation_in_rivers_m3 = 0
            waterbody_evaporation_m3 = 0
            outflow_at_pits_m3 = 0
            over_abstraction_m3 = 0

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

            outflow_per_waterbody_m3 = self.hydrology.lakes_reservoirs.substep(
                current_substep=subrouting_step,
                n_routing_substeps=self.var.n_routing_substeps,
                routing_step_length_seconds=self.var.routing_step_length_seconds,
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

        assert not np.isnan(self.grid.var.discharge_m3_s).any()

        if __debug__:
            # TODO: make dependent on routing step length
            river_storage_m3: np.ndarray = self.router.get_total_storage()
            assert balance_check(
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

            self.routing_loss = (
                evaporation_in_rivers_m3.sum()
                + waterbody_evaporation_m3.sum()
                + outflow_at_pits_m3.sum()
            )
            assert self.routing_loss >= 0, "Routing loss cannot be negative"

        self.report(self, locals())

    @property
    def name(self):
        return "hydrology.routing"
