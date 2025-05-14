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

import math

import numpy as np
import pyflwdir
from numba import njit

from geb.hydrology.lakes_reservoirs import OFF
from geb.module import Module
from geb.workflows import balance_check


def get_channel_ratio(river_width, river_length, cell_area):
    return np.minimum(
        1.0,
        river_width * river_length / cell_area,
    )


def calculate_river_storage_from_discharge(
    discharge, river_alpha, river_length, river_beta
):
    # The momentum equation, see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
    cross_sectional_area_of_flow = river_alpha * discharge**river_beta
    return cross_sectional_area_of_flow * river_length


def calculate_discharge_from_storage(
    river_storage, river_alpha, river_length, river_beta
):
    # The momentum equation (solved for Q), see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/
    return (river_storage / (river_length * river_alpha)) ** (1 / river_beta)


MAX_ITERS = 10


@njit(cache=True)
def IterateToQnew(Qin, Qold, sideflow, alpha, beta, deltaT, deltaX):
    epsilon = np.float64(0.0001)

    assert deltaX > 0, "channel length must be greater than 0"

    # If no input, then output = 0
    if (Qin + Qold + sideflow) == 0:
        return 0

    # Common terms
    ab_pQ = alpha * beta * ((Qold + Qin) / 2) ** (beta - 1)
    deltaTX = deltaT / deltaX
    C = deltaTX * Qin + alpha * Qold**beta + deltaT * sideflow

    # Initial guess for Qnew and iterative process
    Qnew = (deltaTX * Qin + Qold * ab_pQ + deltaT * sideflow) / (deltaTX + ab_pQ)
    fQnew = deltaTX * Qnew + alpha * Qnew**beta - C
    dfQnew = deltaTX + alpha * beta * Qnew ** (beta - 1)
    Qnew -= fQnew / dfQnew
    if np.isnan(Qnew):
        Qnew = 1e-30
    else:
        Qnew = max(Qnew, 1e-30)
    count = 0

    while np.abs(fQnew) > epsilon and count < MAX_ITERS:
        fQnew = deltaTX * Qnew + alpha * Qnew**beta - C
        dfQnew = deltaTX + alpha * beta * Qnew ** (beta - 1)
        Qnew -= fQnew / dfQnew
        count += 1

    return max(Qnew, 0)


@njit(cache=True)
def kinematic(
    Qold,
    sideflow,
    upstream_matrix_from_up_to_downstream,
    idxs_up_to_downstream,
    alpha,
    beta,
    deltaT,
    deltaX,
    is_waterbody,
    is_outflow,
    waterbody_id,
    waterbody_storage,
    outflow_per_waterbody_m3,
):
    """
    Kinematic wave routing

    Parameters
    ----------
    deltaT: float
        Time step, must be > 0
    deltaX: np.ndarray
        Array of floats containing the channel length, must be > 0
    """
    Qnew = np.full_like(Qold, np.nan)

    count = 0

    for i in range(upstream_matrix_from_up_to_downstream.shape[0]):
        node = idxs_up_to_downstream[i]
        upstream_nodes = upstream_matrix_from_up_to_downstream[i]

        Qin = np.float32(0.0)
        sideflow_node = sideflow[node]

        for upstream_node in upstream_nodes:
            if upstream_node == -1:
                break

            count += 1

            if is_outflow[upstream_node]:
                # if upstream node is an outflow add the outflow of the waterbody
                # to the sideflow
                node_waterbody_id = waterbody_id[upstream_node]

                # make sure that the waterbody ID is valid
                assert node_waterbody_id != -1
                waterbody_outflow_m3 = outflow_per_waterbody_m3[node_waterbody_id]

                waterbody_storage[node_waterbody_id] -= waterbody_outflow_m3

                # make sure that the waterbody storage does not go below 0
                assert waterbody_storage[node_waterbody_id] >= 0

                sideflow_node += waterbody_outflow_m3 / deltaT / deltaX[node]

            elif is_waterbody[
                upstream_node
            ]:  # if upstream node is a waterbody, but not an outflow
                assert sideflow[upstream_node] == 0

            else:  # in normal case, just take the inflow from upstream
                assert not np.isnan(Qnew[upstream_node])
                Qin += Qnew[upstream_node]

        Qnew[node] = IterateToQnew(
            Qin, Qold[node], sideflow_node, alpha[node], beta, deltaT, deltaX[node]
        )
    return Qnew


def get_outflow_at_outflows(discharge_m3_s, pits, routing_step_length_seconds):
    return (
        discharge_m3_s[pits].sum() * routing_step_length_seconds
    )  # m3, total outflow at outflows


class Routing(Module):
    def __init__(self, model, hydrology):
        super().__init__(model)

        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        if self.model.in_spinup:
            self.spinup()

    def spinup(self):
        ldd = self.grid.load(
            self.model.files["grid"]["routing/ldd"],
            compress=False,
        )

        river_network = pyflwdir.from_array(
            ldd,
            ftype="ldd",
            transform=self.grid.transform,
            latlon=True,
            mask=~self.grid.mask,
        )

        # we create a mapper from the 2D ldd to the 1D river network
        # the mapper size is ldd.size + 1, because we need to map the
        # the "nan-value" of the ldd to -1 in the river network, thus
        # mapping -1 to -1.
        mapper = np.full(ldd.size + 1, -1, dtype=np.int32)
        indices = np.arange(ldd.size)[~self.grid.mask.ravel()]
        mapper[indices] = np.arange(indices.size)

        river_network.order_cells(method="walk")
        upstream_matrix = pyflwdir.core.upstream_matrix(
            river_network.idxs_ds,
        )

        idxs_up_to_downstream = river_network.idxs_seq[::-1]

        upstream_matrix_from_up_to_downstream = upstream_matrix[idxs_up_to_downstream]
        self.var.upstream_matrix_from_up_to_downstream = mapper[
            upstream_matrix_from_up_to_downstream
        ]
        self.var.idxs_up_to_downstream = mapper[idxs_up_to_downstream]

        # make sure all non-selected cells are set to -1
        assert (
            upstream_matrix[
                ~np.isin(np.arange(river_network.size), idxs_up_to_downstream)
            ]
            == -1
        ).all()

        self.var.pits = mapper[river_network.idxs_pit]

        self.grid.var.upstream_area = river_network.upstream_area(unit="m2")
        self.grid.var.upstream_area[self.grid.var.upstream_area < 0] = np.nan
        self.grid.var.upstream_area = self.grid.var.upstream_area[~self.grid.mask]
        self.grid.var.upstream_area_n_cells = river_network.upstream_area(unit="cell")
        self.grid.var.upstream_area_n_cells[self.grid.var.upstream_area_n_cells < 0] = 0
        self.grid.var.upstream_area_n_cells = self.grid.var.upstream_area_n_cells[
            ~self.grid.mask
        ]

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
        self.grid.var.river_length[self.var.pits] = (
            np.sqrt(self.grid.var.cell_area[self.var.pits]) / 2
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

        # Initialise water volume and discharge in rivers, just set at 0 [m3]
        self.grid.var.river_storage_m3 = np.ones_like(
            self.grid.var.river_width, dtype=np.float64
        )

        self.grid.var.discharge_m3_s = calculate_discharge_from_storage(
            self.grid.var.river_storage_m3,
            self.grid.var.river_alpha,
            self.grid.var.river_length,
            self.var.river_beta,
        )

        self.grid.var.outflow_at_outflows_m3_substep = get_outflow_at_outflows(
            self.grid.var.discharge_m3_s,
            self.var.pits,
            self.var.routing_step_length_seconds,
        )

        self.grid.var.discharge_m3_s_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge_m3_s.size),
            0,
            dtype=self.grid.var.discharge_m3_s.dtype,
        )

    def step(self, total_runoff, channel_abstraction_m, return_flow):
        if __debug__:
            pre_river_storage_m3 = self.grid.var.river_storage_m3.copy()
            pre_storage = self.hydrology.lakes_reservoirs.var.storage.copy()

        return_flow_m3_per_routing_step = (
            return_flow * self.grid.var.cell_area / self.var.n_routing_substeps
        )
        channel_abstraction_m3_per_routing_step = (
            channel_abstraction_m
            * self.grid.var.cell_area
            / self.var.n_routing_substeps
        )

        return_flow_m3_to_water_bodies_per_routing_step = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=return_flow_m3_per_routing_step[self.grid.var.waterBodyID != -1],
        )
        return_flow_m3_per_routing_step[self.grid.var.waterBodyID != -1] = 0.0

        runoff_m3_per_routing_step = (
            total_runoff * self.grid.var.cell_area / self.var.n_routing_substeps
        )

        runoff_m3_per_routing_step_pre = runoff_m3_per_routing_step.sum()

        runoff_m3_per_routing_step_water_bodies = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=runoff_m3_per_routing_step[self.grid.var.waterBodyID != -1],
        )

        runoff_m3_per_routing_step[self.grid.var.waterBodyID != -1] = 0.0

        assert math.isclose(
            runoff_m3_per_routing_step_water_bodies.sum()
            + runoff_m3_per_routing_step.sum(),
            runoff_m3_per_routing_step_pre,
            rel_tol=1e-6,
            abs_tol=0,
        )

        self.grid.var.discharge_m3_s_substep = np.full(
            (self.var.n_routing_substeps, self.grid.var.discharge_m3_s.size),
            np.nan,
            dtype=self.grid.var.discharge_m3_s.dtype,
        )

        if self.model.current_timestep == 0 and self.model.in_spinup:
            self.var.discharge_out_of_water_bodies_into_other_water_bodies_m3 = (
                np.zeros(
                    self.hydrology.lakes_reservoirs.var.capacity.size, dtype=np.float32
                )
            )
            self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] = 0

        total_discharge_out_of_water_bodies_into_other_water_bodies_m3_pre = (
            self.var.discharge_out_of_water_bodies_into_other_water_bodies_m3.sum()
        )

        potential_evaporation_per_water_body_m3_per_routing_step = (
            np.bincount(
                self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                weights=self.grid.var.EWRef[self.grid.var.waterBodyID != -1],
            )
            / np.bincount(self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1])
            * self.hydrology.lakes_reservoirs.var.lake_area
        ) / self.var.n_routing_substeps
        potential_evaporation_per_water_body_m3_per_routing_step[
            self.hydrology.lakes_reservoirs.var.water_body_type == OFF
        ] = 0

        # the ratio of each grid cell that is currently covered by a river
        channel_ratio = get_channel_ratio(
            river_length=self.grid.var.river_length,
            river_width=self.grid.var.river_width,
            cell_area=self.grid.var.cell_area,
        )

        # calculate evaporation from rivers per timestep usting the current channel ratio
        evaporation_in_rivers_m3_per_routing_step = (
            self.grid.var.EWRef * channel_ratio * self.grid.var.cell_area
        ) / self.var.n_routing_substeps
        # set the evaporation in rivers to 0 for all water bodies
        evaporation_in_rivers_m3_per_routing_step[self.grid.var.waterBodyID != -1] = 0.0

        if __debug__:
            # these are for balance checks, the sum of all routing steps
            side_flow_channel_m3 = 0
            evaporation_in_rivers_m3 = 0
            waterbody_evaporation_m3 = 0
            outflow_at_outflows_m3 = 0

        for subrouting_step in range(self.var.n_routing_substeps):
            # ensure there is no river storage in the water bodies
            assert (
                self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] == 0
            ).all()

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
                - evaporation_in_rivers_m3_per_routing_step
            )
            assert (
                side_flow_channel_m3_per_routing_step[self.grid.var.waterBodyID != -1]
                == 0
            ).all()

            # m2 because this is per unit of channel length, see division
            side_flow_channel_m2_per_s = (
                side_flow_channel_m3_per_routing_step
                / self.grid.var.river_length
                / self.var.routing_step_length_seconds
            )

            assert (
                side_flow_channel_m2_per_s[self.grid.var.waterBodyID != -1] == 0.0
            ).all()

            self.grid.var.discharge_m3_s = kinematic(
                self.grid.var.discharge_m3_s,
                side_flow_channel_m2_per_s.astype(np.float32),
                self.var.upstream_matrix_from_up_to_downstream,
                self.var.idxs_up_to_downstream,
                self.grid.var.river_alpha,
                self.var.river_beta,
                self.var.routing_step_length_seconds,
                self.grid.var.river_length,
                is_waterbody=self.grid.var.waterBodyID != -1,
                is_outflow=self.grid.var.waterbody_outflow_points != -1,
                waterbody_id=self.grid.var.waterBodyID,
                waterbody_storage=self.hydrology.lakes_reservoirs.var.storage,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            )

            # update river storage
            self.grid.var.river_storage_m3 = calculate_river_storage_from_discharge(
                self.grid.var.discharge_m3_s,
                self.grid.var.river_alpha,
                self.grid.var.river_length,
                self.var.river_beta,
            )

            self.grid.var.outflow_at_outflows_m3_substep_new = get_outflow_at_outflows(
                self.grid.var.discharge_m3_s,
                self.var.pits,
                self.var.routing_step_length_seconds,
            )

            # get storage discharged into lakes and reservoirs
            discharge_into_water_bodies_m3 = (
                np.bincount(
                    self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                    weights=self.grid.var.discharge_m3_s[
                        self.grid.var.waterBodyID != -1
                    ],
                )
                * self.var.routing_step_length_seconds
                + self.var.discharge_out_of_water_bodies_into_other_water_bodies_m3
                + runoff_m3_per_routing_step_water_bodies
                + return_flow_m3_to_water_bodies_per_routing_step
            )

            # remove storage and discharge
            # that is discharged into lakes and reservoirs from river storage
            # and set discharge to 0 in those locations
            self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] = 0.0
            self.grid.var.discharge_m3_s[self.grid.var.waterBodyID != -1] = 0.0

            self.hydrology.lakes_reservoirs.var.storage += (
                discharge_into_water_bodies_m3
            )

            assert (
                self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] == 0.0
            ).all()

            self.grid.var.discharge_m3_s_substep[subrouting_step, :] = (
                self.grid.var.discharge_m3_s.copy()
            )

            if __debug__:
                # Discharge at outlets and lakes and reservoirs
                outflow_at_outflows_m3 += (
                    self.grid.var.outflow_at_outflows_m3_substep
                    + self.grid.var.outflow_at_outflows_m3_substep_new
                ) / 2
                side_flow_channel_m3 += side_flow_channel_m3_per_routing_step
                waterbody_evaporation_m3 += (
                    actual_evaporation_from_water_bodies_per_routing_step_m3
                )
                evaporation_in_rivers_m3 += evaporation_in_rivers_m3_per_routing_step

            self.grid.var.outflow_at_outflows_m3_substep = (
                self.grid.var.outflow_at_outflows_m3_substep_new
            )

        assert not np.isnan(self.grid.var.discharge_m3_s).any()

        if __debug__:
            # TODO: make dependent on routing step length
            balance_check(
                how="sum",
                influxes=[
                    total_runoff * self.grid.var.cell_area,
                    return_flow * self.grid.var.cell_area,
                ],
                outfluxes=[
                    channel_abstraction_m * self.grid.var.cell_area,
                    outflow_at_outflows_m3,
                    evaporation_in_rivers_m3,
                    waterbody_evaporation_m3,
                ],
                prestorages=[
                    pre_storage,
                    pre_river_storage_m3,
                    total_discharge_out_of_water_bodies_into_other_water_bodies_m3_pre,
                ],
                poststorages=[
                    self.hydrology.lakes_reservoirs.var.storage,
                    self.grid.var.river_storage_m3,
                    self.var.discharge_out_of_water_bodies_into_other_water_bodies_m3,
                ],
                name="routing_1",
                tollerance=1000,
            )

            self.routing_loss = (
                evaporation_in_rivers_m3.sum()
                + waterbody_evaporation_m3.sum()
                + outflow_at_outflows_m3.sum()
            )

        assert (
            self.grid.var.river_storage_m3[self.grid.var.waterBodyID != -1] == 0.0
        ).all()

        self.report(self, locals())

    @property
    def name(self):
        return "hydrology.routing"
