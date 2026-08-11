"""Kinematic wave routing algorithm for river networks."""

import numpy as np
import pyflwdir
from numba import njit

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
)

from .routerbase import Router, compute_retention_routing

MAX_ITERS: int = 10


@njit(cache=True)
def update_node_kinematic(
    Qin: np.float32,
    Qold: np.float32,
    Qside: np.float32,
    evaporation_m3_s: np.float32,
    alpha: np.float32,
    beta: np.float32,
    deltaT: np.float32,
    deltaX: np.float32,
    epsilon: np.float32 = np.float32(0.00001),
) -> tuple[np.float32, np.float32]:
    """Update the discharge for a single node using the kinematic wave equation.

    Args:
        Qin: Inflow to the node in m3/s.
        Qold: Discharge at the previous time step in m3/s.
        Qside: Sideflow to the node in m3/s.
        evaporation_m3_s: Evaporation from the node in m3/s.
        alpha: The alpha parameter for the kinematic wave equation.
        beta: The beta parameter for the kinematic wave equation.
        deltaT: The time step in seconds, must be > 0
        deltaX: The length of the river segment in meters, must be > 0
        epsilon: Convergence criterion for the Newton-Raphson method.

    Returns:
        A tuple containing:
            The new discharge in m3/s.
            The actual evaporation in m3/s.
    """
    evaporation_m3_s_l: np.float32 = (
        evaporation_m3_s / deltaX
    )  # Convert evaporation from m3/s to m3/s/m

    q: np.float32 = Qside / deltaX  # Convert sideflow from m3/s to m3/s/m

    # If evaporation is larger than the inflow and sideflow, we limit it to the sum of inflow and sideflow
    evaporation_m3_s_l: np.float32 = min(
        evaporation_m3_s_l, (Qin + Qold) / 2 + max(q, 0)
    )

    q -= evaporation_m3_s_l  # Adjust lateral inflow for evaporation

    actual_evaporation_m3_s: np.float32 = evaporation_m3_s_l * deltaX

    # If there's no inflow, no previous flow, and no lateral inflow,
    # then the discharge at the new time step will be zero.
    if (Qin + Qold + q) < 1e-30:
        return np.float32(1e-30), actual_evaporation_m3_s

    Qin: np.float32 = max(Qin, np.float32(1e-30))

    # Common terms
    ab_pQ: np.float32 = alpha * beta * ((Qold + Qin) / 2) ** (beta - 1)
    deltaTX: np.float32 = deltaT / deltaX
    C: np.float32 = deltaTX * Qin + alpha * Qold**beta + deltaT * q

    # Initial guess for Qnew and iterative process
    Qnew: np.float32 = (deltaTX * Qin + Qold * ab_pQ + deltaT * q) / (deltaTX + ab_pQ)
    Qnew: np.float32 = max(Qnew, np.float32(1e-30))

    # Newton-Raphson method
    count: int = 0
    fQkx: np.float32 = deltaTX * Qnew + alpha * Qnew**beta - C

    while np.abs(fQkx) > epsilon and count < MAX_ITERS:
        dfQkx: np.float32 = deltaTX + alpha * beta * Qnew ** (beta - 1)
        Qnew -= fQkx / dfQkx
        Qnew: np.float32 = max(Qnew, np.float32(1e-30))

        # Update fQkx for the next iteration check
        fQkx = deltaTX * Qnew + alpha * Qnew**beta - C
        count += 1

    return Qnew, actual_evaporation_m3_s


class KinematicWave(Router):
    """Kinematic wave routing algorithm.

    This class implements the kinematic wave routing algorithm for river networks.
    """

    def __init__(
        self,
        dt: float | int,
        river_network: pyflwdir.FlwdirRaster,
        river_length: ArrayFloat32,
        waterbody_id: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
        retention_max_storage_m3: ArrayFloat32,
        retention_node_id: ArrayInt32,
        controlled_retention: ArrayBool,
        retention_basin_release_threshold_factor: float,
    ) -> None:
        """Initializes the KinematicWave class.

        Args:
            dt: length of the time step in seconds.
            river_network: The river network as a FlwdirRaster object, which contains the flow
                direction and other information about the river network.
            river_length: The length of the river in each cell,.
            waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell.
            is_waterbody_outflow: A 1D array with the same shape as the grid, which is True for the outflow cells.
            retention_max_storage_m3: Array of floats containing the maximum storage in each retention basin
            retention_node_id: Array of integers containing the node ID for each retention basin
            controlled_retention: Array of booleans indicating whether each retention basin is controlled or uncontrolled
            retention_basin_release_threshold_factor: Factor to multiply the activation threshold by to get the release threshold.
        """
        super().__init__(
            dt,
            river_network,
            waterbody_id,
            is_waterbody_outflow,
            retention_basin_release_threshold_factor,
        )

        self.river_length = river_length.ravel()

        # retention basin parameters
        self.retention_max_storage_m3 = retention_max_storage_m3.ravel()
        self.retention_node_id = retention_node_id.ravel()
        self.controlled_retention = controlled_retention.ravel()

    def calculate_river_storage_from_discharge(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_length: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat64:
        """Calculate the river storage from the discharge using the kinematic wave equation.

        Uses the momentum equation, see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/

        Args:
            discharge: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_length: The length of the river in each cell, in meters.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell.

        Returns:
            A 1D array with the calculated river storage for each cell, in m3.
        """
        cross_sectional_area_of_flow: ArrayFloat64 = (
            river_storage_alpha * discharge.astype(np.float64) ** river_storage_beta
        )
        river_storage: ArrayFloat64 = cross_sectional_area_of_flow * river_length
        river_storage[waterbody_id != -1] = 0.0
        return river_storage

    def calculate_discharge_from_river_storage(
        self,
        river_storage: ArrayFloat64,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        river_length: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat32:
        """Calculate the discharge from the river storage using the kinematic wave equation.

        Inverts the momentum equation: Q = (Area / alpha) ** (1 / beta)

        Args:
            river_storage: The storage in each cell, in m3.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            river_length: The length of the river in each cell, in meters.
            waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell.

        Returns:
            A 1D array with the calculated discharge for each cell, in m3/s.
        """
        cross_sectional_area_of_flow: ArrayFloat32 = (
            river_storage / river_length
        ).astype(np.float32)
        discharge: ArrayFloat32 = (
            cross_sectional_area_of_flow / river_storage_alpha
        ) ** (1 / river_storage_beta)
        discharge[waterbody_id != -1] = np.nan
        return discharge

    def get_available_storage(
        self,
        Q: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        maximum_abstraction_ratio: float = 0.9,
    ) -> ArrayFloat64:
        """Get the available storage of the river network, which is the sum of the available storage in each cell.

        Args:
            Q: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            maximum_abstraction_ratio: he maximum abstraction ratio, default is 0.9.
                This is the ratio of the available storage that can be used for abstraction.

        Returns:
            The available storage of the river network [m3].
        """
        return (
            self.get_total_storage(Q, river_storage_alpha, river_storage_beta)
            * maximum_abstraction_ratio
        )

    def get_total_storage(
        self,
        Q: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
    ) -> ArrayFloat64:
        """Get the total storage of the river network, which is the sum of the available storage in each cell.

        Args:
            Q: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.

        Returns:
            The total storage of the river network [m3].

        """
        total_storage = self.calculate_river_storage_from_discharge(
            discharge=Q,
            river_storage_alpha=river_storage_alpha,
            river_length=self.river_length,
            river_storage_beta=river_storage_beta,
            waterbody_id=self.waterbody_id,
        )

        assert not np.isnan(total_storage).any()
        return total_storage

    @staticmethod
    @njit(cache=True)
    def _step(
        dt: float | int,
        Qold: ArrayFloat32,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat64,
        outflow_per_waterbody_m3: ArrayFloat32,
        upstream_matrix_from_up_to_downstream: ArrayInt32,
        idxs_up_to_downstream: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
        waterbody_id: ArrayInt32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        river_length: ArrayFloat32,
        retention_storage_m3: ArrayFloat32,
        retention_max_storage_m3: ArrayFloat32,
        retention_node_id: ArrayInt32,
        controlled_retention: ArrayBool,
        retention_activation_threshold_m3_s: ArrayFloat32,
        retention_basin_release_threshold_factor: np.float32,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
    ]:
        """Kinematic wave routing.

        Args:
            dt: Time step, must be > 0
            Qold: Old discharge array, which is a 1D array with dicharge for each grid cell in the river network.
            sideflow_m3: Sideflow in m3 for each grid cell in the river network.
            evaporation_m3: Evaporation in m3 for each grid cell in the river network.
            waterbody_storage_m3: Storage of each waterbody in m3.
            outflow_per_waterbody_m3: Outflow of each waterbody in m3.
            upstream_matrix_from_up_to_downstream: Upstream matrix from the river network, which is a 2D array. For each
                cell (first dimension) in the river network, it contains the indices of the upstream cells (second dimension).
                -1 indicates no upstream cell. There should never be any upstream cells after the first -1. The node associated with
                the row is specified by idxs_up_to_downstream.
            idxs_up_to_downstream: Indices of the cells in the river network, associated with the upstream_matrix_from_up_to_downstream.
            is_waterbody_outflow: A 1D array with the same shape as the grid, which is True for the outflow cells.
            waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell. -1 indicates no waterbody.
            river_storage_alpha: The alpha parameter for the kinematic wave equation, which is a 1D array with the same shape as the grid.
            river_storage_beta: The beta parameter for the kinematic wave equation, which is a 1D array.
            river_length: Array of floats containing the channel length, must be > 0
            retention_storage_m3: Array of floats containing the current storage in retention basins
            retention_max_storage_m3: Array of floats containing the maximum storage in each retention basin
            retention_node_id: Array of integers containing the node ID for each retention basin
            controlled_retention: Array of booleans indicating whether each retention basin is controlled or uncontrolled
            retention_activation_threshold_m3_s: Array of floats containing the activation threshold for each retention basin
            retention_basin_release_threshold_factor: Factor to multiply the activation threshold by to get the release threshold.

        Returns:
            Qnew: New discharge array, which is a 1D array with discharge for each grid cell in the river network.
            actual_evaporation_m3: Actual evaporation in m3 for each grid cell in the river network.
            over_abstraction_m3: Over abstraction in m3 for each grid cell in the river network.
            waterbody_inflow_m3: Inflow to each waterbody in m3.
            retention_inflow_m3: Inflow to each retention basin in m3.
            retention_outflow_m3: Outflow from each retention basin in m3.
        """
        Qnew: ArrayFloat32 = np.full_like(Qold, np.nan, dtype=np.float32)
        actual_evaporation_m3: ArrayFloat32 = np.zeros_like(Qold, dtype=np.float32)
        over_abstraction_m3: ArrayFloat32 = np.zeros_like(Qold, dtype=np.float32)
        waterbody_inflow_m3: ArrayFloat32 = np.zeros_like(
            waterbody_storage_m3, dtype=np.float32
        )

        # initialize retention in- and outflow arrays, which are updated in the routing step and can be used for tracking daily and total water fluxes per basin
        retention_inflow_m3 = np.zeros_like(retention_storage_m3, dtype=np.float32)
        retention_outflow_m3 = np.zeros_like(retention_storage_m3, dtype=np.float32)

        for i in range(upstream_matrix_from_up_to_downstream.shape[0]):
            node = idxs_up_to_downstream[i]
            upstream_nodes: ArrayInt32 = upstream_matrix_from_up_to_downstream[i]

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

            node_waterbody_id = waterbody_id[node]
            if node_waterbody_id != -1:
                waterbody_inflow_m3_node = Qin * dt + sideflow_node_m3
                waterbody_storage_m3[node_waterbody_id] += waterbody_inflow_m3_node
                waterbody_inflow_m3[node_waterbody_id] += waterbody_inflow_m3_node
                assert evaporation_m3[node] == np.float32(0.0)
            else:
                node_retention_id = retention_node_id[node]
                # if the node is associated with a retention basin (not -1), we apply the retention logic
                if node_retention_id != -1:
                    # Discharge before diversion, to compare against activation thresholds; convert to flow rate (m3/s)
                    # We use the average flow during the timestep, considering both inflow and existing storage.
                    discharge_before_diversion_m3_per_s: np.float32 = (
                        Qin + Qold[node]
                    ) / 2 + (sideflow_node_m3 / dt)

                    # total discharge entering the retention basin river cell during timestep, including sideflow
                    discharge_at_retention_basin_m3_per_timestep: np.float32 = (
                        Qin * dt + sideflow_node_m3
                    )

                    # limit inflow into basins (20% per timestep (hour))
                    inflow_limit_m3: np.float32 = (
                        np.float32(0.20) * retention_max_storage_m3[node_retention_id]
                    )

                    # Maximum volume of water that the basin can release during this step (5% of max storage)
                    max_outflow_limit_m3: np.float32 = (
                        np.float32(0.05) * retention_max_storage_m3[node_retention_id]
                    )

                    is_rising_limb: bool = (
                        discharge_before_diversion_m3_per_s > Qold[node]
                    )

                    (
                        diverted_volume_m3,
                        outflow_volume_m3,
                        retention_storage_m3[node_retention_id],
                        discharge_at_retention_basin_m3_per_timestep,
                    ) = compute_retention_routing(
                        dt=np.float32(dt),
                        river_volume_m3=discharge_at_retention_basin_m3_per_timestep,
                        discharge_before_diversion_m3_s=discharge_before_diversion_m3_per_s,
                        is_rising_limb=is_rising_limb,
                        retention_storage_m3=retention_storage_m3[node_retention_id],
                        retention_max_storage_m3=retention_max_storage_m3[
                            node_retention_id
                        ],
                        controlled_retention=controlled_retention[node_retention_id],
                        activation_threshold_m3_s=retention_activation_threshold_m3_s[
                            node_retention_id
                        ],
                        release_threshold_factor=retention_basin_release_threshold_factor,
                        inflow_limit_m3=inflow_limit_m3,
                        max_outflow_limit_m3=max_outflow_limit_m3,
                    )

                    retention_inflow_m3[node_retention_id] += diverted_volume_m3
                    retention_outflow_m3[node_retention_id] += outflow_volume_m3

                    # we subract retained water from sideflow (water added/subtracted locally) to not break upstream routing logic
                    # Qin cant be modified since it represents upstream routing
                    sideflow_node_m3: np.float32 = (
                        discharge_at_retention_basin_m3_per_timestep - Qin * dt
                    )

                Qnew[node], actual_evaporation_m3_dt = update_node_kinematic(
                    Qin,
                    Qold[node],
                    sideflow_node_m3 / dt,
                    evaporation_m3[node] / dt,
                    river_storage_alpha[node],
                    river_storage_beta[node],
                    dt,
                    river_length[node],
                )
                actual_evaporation_m3[node] = actual_evaporation_m3_dt * dt

        return (
            Qnew,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
        )

    def step(
        self,
        Q_prev_m3_s: ArrayFloat32,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat64,
        outflow_per_waterbody_m3: ArrayFloat32,
        retention_storage_m3: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        river_width: ArrayFloat32,
        retention_activation_threshold_m3_s: ArrayFloat32,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat64,
        ArrayFloat32,
        np.float32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
    ]:
        """Perform a single routing step.

        Discharge is updated based on the inflow from upstream cells, sideflow,
        evaporation, waterbody outflow, and retention basin diversion. Uses an implicit version of the kinematic wave equation.

        Args:
            Q_prev_m3_s: Old discharge array, which is a 1D array with dicharge for each grid cell in the river network.
            sideflow_m3: Sideflow in m3 for each grid cell in the river network.
            evaporation_m3: Evaporation in m3 for each grid cell in the river network.
            waterbody_storage_m3: Storage of each waterbody in m3.
            outflow_per_waterbody_m3: Outflow of each waterbody in m3.
            retention_storage_m3: Array of floats containing the current storage in retention basins
            river_storage_alpha: The alpha parameter for the kinematic wave equation, which is a 1D array with the same shape as the grid.
            river_storage_beta: The beta parameter for the kinematic wave equation, which is a 1D array.
            river_width: The width of the river in each cell (meters).
            retention_activation_threshold_m3_s: Array of floats containing the activation threshold for each retention basin

        Returns:
            Q: New discharge array, which is a 1D array with discharge for each grid cell in the river network.
            actual_evaporation_m3: Actual evaporation in m3 for each grid cell in the river network.
            over_abstraction_m3: Over abstraction in m3 for each grid cell in the river network.
            waterbody_storage_m3: Updated storage of each waterbody in m3.
            waterbody_inflow_m3: Inflow to each waterbody in m3.
            outflow_at_pits_m3: Outflow at pits in m3.
            retention_storage_m3: Updated storage in retention basins in m3.
            retention_inflow_m3: Inflow to each retention basin in m3.
            retention_outflow_m3: Outflow from each retention basin in m3.
        """
        (
            Q,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
        ) = self._step(
            dt=self.dt,
            Qold=Q_prev_m3_s,
            sideflow_m3=sideflow_m3,
            evaporation_m3=evaporation_m3,
            waterbody_storage_m3=waterbody_storage_m3,
            outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            upstream_matrix_from_up_to_downstream=self.upstream_matrix_from_up_to_downstream,
            idxs_up_to_downstream=self.idxs_up_to_downstream,
            is_waterbody_outflow=self.is_waterbody_outflow,
            waterbody_id=self.waterbody_id,
            river_storage_alpha=river_storage_alpha,
            river_storage_beta=river_storage_beta,
            river_length=self.river_length,
            retention_storage_m3=retention_storage_m3,
            retention_max_storage_m3=self.retention_max_storage_m3,
            retention_node_id=self.retention_node_id,
            controlled_retention=self.controlled_retention,
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
            retention_basin_release_threshold_factor=self.retention_basin_release_threshold_factor,
        )

        # Because some pits may also be waterbodies (where Q is NaN), we use nansum
        outflow_at_pits_m3 = np.nansum(Q[self.is_pit] * self.dt)

        return (
            Q,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_storage_m3,
            waterbody_inflow_m3,
            outflow_at_pits_m3,
            retention_storage_m3,
            retention_inflow_m3,
            retention_outflow_m3,
        )
