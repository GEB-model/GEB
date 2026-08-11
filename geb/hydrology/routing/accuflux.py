"""Accuflux routing algorithm for river networks."""

import numpy as np
import pyflwdir
from numba import njit

from geb.geb_types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    TwoDArrayInt32,
)

from .routerbase import Router, compute_retention_routing


class Accuflux(Router):
    """Accuflux routing algorithm.

    In each step, the algorithm calculates the new discharge for each cell
    based on the inflow from upstream cells, sideflow, waterbody outflow, and retention basin diversion.

    The algorithm works as follows:

    1. For each cell, it calculates the inflow from upstream cells.
    2. It adds the sideflow and waterbody outflow to the inflow and subtracts water volumes stored per retention basin.
    3. It calculates the new discharge for each cell based on the inflow.
    4. It updates the waterbody storage based on the outflow.
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
        """Initializes the Accuflux class.

        Args:
            dt: Number of seconds in the time step, must be > 0
            river_network: The river network as a FlwdirRaster object, which contains the flow
                direction and other information about the river network.
            river_length: The length of the river in each cell, in meters.
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
        self.retention_max_storage_m3 = retention_max_storage_m3
        self.retention_node_id = retention_node_id
        self.controlled_retention = controlled_retention

    def get_available_storage(
        self,
        Q: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        maximum_abstraction_ratio: float = 0.9,
    ) -> ArrayFloat64:
        """Get the available storage of the river network, which is the sum of the available storage in each cell.

        Available storage in lakes and reservoirs is set to 0.

        Args:
            Q: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter. Unused. Only added so that the interface is the same for all routers.
            river_storage_beta: The beta parameter. Unused. Only added so that the interface is the same for all routers.
            maximum_abstraction_ratio: The maximum abstraction ratio, default is 0.9.
                This is the ratio of the available storage that can be used for abstraction.

        Returns:
            The available storage of the river network.
        """
        available_storage = Q.astype(np.float64) * self.dt * maximum_abstraction_ratio
        available_storage[self.waterbody_id != -1] = 0.0
        assert not np.isnan(available_storage).any()
        return available_storage

    def calculate_river_storage_from_discharge(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_length: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat64:
        """Calculate the river storage from the discharge for the accuflux router.

        Note: for accuflux we just assume all water stored is discharged in the next
        timestep, as it is a single-step linear reservoir with k=dt.

        Args:
            discharge: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter. Unused for accuflux, but required for the interface.
            river_length: The length of the river in each cell, in meters. Unused for accuflux, but required for the interface.
            river_storage_beta: The beta parameter. Unused for accuflux, but required for the interface.
            waterbody_id: A 1D array with same shape as the grid, which is the waterbody ID for each cell.

        Returns:
            A 1D array with the calculated river storage for each cell, in m3.
        """
        river_storage: ArrayFloat64 = discharge.astype(np.float64) * self.dt
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
        """Calculate the discharge from the river storage for the accuflux router.

        Inverse of storage calculation: Q = S / dt.

        Args:
            river_storage: The storage in each cell, in m3.
            river_storage_alpha: The alpha parameter. Unused for accuflux, but required for the interface.
            river_storage_beta: The beta parameter. Unused for accuflux, but required for the interface.
            river_length: The length of the river in each cell, in meters. Unused for accuflux, but required for the interface.
            waterbody_id: A 1D array with same shape as the grid, which is the waterbody ID for each cell.

        Returns:
            A 1D array with the calculated discharge for each cell, in m3/s.
        """
        discharge: ArrayFloat32 = (river_storage.astype(np.float32) / self.dt).astype(
            np.float32
        )
        discharge[waterbody_id != -1] = np.nan
        return discharge

    @staticmethod
    @njit(cache=True)
    def _step(
        dt: float | int,
        Qold: ArrayFloat32,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat64,
        outflow_per_waterbody_m3: ArrayFloat32,
        upstream_matrix_from_up_to_downstream: TwoDArrayInt32,
        idxs_up_to_downstream: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
        waterbody_id: ArrayInt32,
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
        ArrayFloat32,  # retention inflow
        ArrayFloat32,  # retention outflow
    ]:
        """Accuflux routing.

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
            retention_storage_m3: Storage of each retention node in m3.
            retention_max_storage_m3: Maximum storage of each retention node in m3.
            retention_node_id: A 1D array with the same shape as the grid, which is the retention node ID for each cell. -1 indicates no retention node.
            controlled_retention: A 1D boolean array with the same shape as the grid, which is True for retention nodes with controlled operation.
            retention_activation_threshold_m3_s: A 1D array with the same shape as the grid, which is the activation threshold for each retention node in m3/s.
                If river discharge at the retention node exceeds this threshold, it starts to fill until it reaches the maximum storage.
            retention_basin_release_threshold_factor: Factor to multiply the activation threshold by to get the release threshold.


        Returns:
            Qnew: New discharge array, which is a 1D array with discharge for each grid cell in the river network.
            actual_evaporation_m3: Actual evaporation in m3 for each grid cell in the river network.
            over_abstraction_m3: Over abstraction in m3 for each grid cell in the river network.
            waterbody_inflow_m3: Inflow to each waterbody in m3.
            retention_outflow_m3: Outflow from retention nodes in m3.
            retention_inflow_m3: Inflow to retention nodes in m3.
        """
        Qold += sideflow_m3 / dt
        # initialize over abstraction array, which keeps track of the amount of water that is abstracted beyond the available storage (i.e. negative discharge)
        over_abstraction_m3: ArrayFloat32 = np.zeros_like(Qold, dtype=np.float32)
        # Prevent negative discharge before evaporation, but keep track of the negative discharge as over abstraction
        neg_mask = Qold < 0.0
        over_abstraction_m3[neg_mask] += -Qold[neg_mask] * dt
        Qold[neg_mask] = 0.0

        evaporation_m3_s: ArrayFloat32 = evaporation_m3 * np.float32(1 / dt)
        actual_evaporation_m3_s: ArrayFloat32 = np.minimum(evaporation_m3_s, Qold)
        actual_evaporation_m3: ArrayFloat32 = actual_evaporation_m3_s * dt
        actual_evaporation_m3[waterbody_id != -1] = 0.0

        Qold -= actual_evaporation_m3_s

        Qnew: ArrayFloat32 = np.full_like(Qold, np.nan, dtype=np.float32)
        waterbody_inflow_m3: ArrayFloat32 = np.zeros_like(
            waterbody_storage_m3, dtype=np.float32
        )
        # Initialize retention inflow/outflow arrays
        retention_inflow_m3 = np.zeros_like(retention_storage_m3, dtype=np.float32)
        retention_outflow_m3 = np.zeros_like(retention_storage_m3, dtype=np.float32)

        for i in range(upstream_matrix_from_up_to_downstream.shape[0]):
            node = idxs_up_to_downstream[i]
            upstream_nodes = upstream_matrix_from_up_to_downstream[i]

            inflow_volume_m3 = np.float32(0.0)

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

                    inflow_volume_m3 += waterbody_outflow_m3

                elif (
                    waterbody_id[upstream_node] != -1
                ):  # if upstream node is a waterbody, but not an outflow
                    assert sideflow_m3[upstream_node] == 0

                else:  # in normal case, just take the inflow from upstream
                    inflow_volume_m3 += Qold[upstream_node] * dt

            node_waterbody_id = waterbody_id[node]
            node_retention_id = retention_node_id[node]

            # if the node is associated with a retention basin (not -1), we apply the retention logic
            if node_retention_id != -1:
                # Compute discharge before diversion into ret. basins to check against activation threshold
                # We use the total flow passing through the node during the timestep.
                Q_before_diversion_m3_per_s = inflow_volume_m3 / dt + Qold[node]

                # define inflow limit of 20% per ts of max capacity per timestep
                inflow_limit_m3: np.float32 = (
                    np.float32(0.2) * retention_max_storage_m3[node_retention_id]
                )

                # Maximum volume of water that the basin can release during this step (5% of max storage)
                max_outflow_limit_m3: np.float32 = (
                    np.float32(0.05) * retention_max_storage_m3[node_retention_id]
                )

                is_rising_limb: bool = Q_before_diversion_m3_per_s > Qold[node]

                (
                    diverted_volume_m3,
                    outflow_volume,
                    retention_storage_m3[node_retention_id],
                    inflow_volume_m3,
                ) = compute_retention_routing(
                    dt=np.float32(dt),
                    river_volume_m3=inflow_volume_m3,
                    discharge_before_diversion_m3_s=Q_before_diversion_m3_per_s,
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
                retention_outflow_m3[node_retention_id] += outflow_volume

            if node_waterbody_id != -1:
                waterbody_storage_m3[node_waterbody_id] += inflow_volume_m3
                waterbody_inflow_m3[node_waterbody_id] += inflow_volume_m3
            else:
                Qnew_node = inflow_volume_m3 / dt
                if Qnew_node < 0.0:
                    # if the new discharge is negative, we have over-abstraction
                    over_abstraction_m3[node] = -Qnew_node * dt
                    Qnew_node = 0.0
                Qnew[node] = Qnew_node
                assert Qnew[node] >= 0.0, "Discharge cannot be negative"
        return (
            Qnew,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_inflow_m3,
            retention_inflow_m3,
            retention_outflow_m3,
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
            river_storage_alpha: The alpha parameter. Unused.
            river_storage_beta: The beta parameter. Unused.

        Returns:
            The total storage of the river network [m3].

        """
        return self.get_available_storage(
            Q,
            river_storage_alpha=river_storage_alpha,
            river_storage_beta=river_storage_beta,
            maximum_abstraction_ratio=1.0,
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
        """Perform a routing step using the simple accumulation algorithm.

        All discharge from all upstream cells is simply summed to get the discharge
        for each cell. Sideflow and waterbody outflow is added to the discharge.

        Args:
            Q_prev_m3_s: Previous discharge array, which is a 1D array with discharge for each grid cell in the river network.
            sideflow_m3: Sideflow in m3 for each grid cell in the river network.
            evaporation_m3: Evaporation in m3 for each grid cell in the river network.
            waterbody_storage_m3: Storage of each waterbody in m3.
            outflow_per_waterbody_m3: Outflow of each waterbody in m3.
            retention_storage_m3: Storage of each retention node in m3.
            river_storage_alpha: The alpha parameter for the kinematic wave equation, which is a 1D array with the same shape as the grid. Not used in this method, but included for consistency with the KinematicWave class.
            river_storage_beta: The beta parameter for the kinematic wave equation, which is a 1D array. Not used in this method, but included for consistency with the KinematicWave class.
            river_width: The width of the river in each cell (meters).
            retention_activation_threshold_m3_s: Array of floats containing the activation threshold for each retention node in m3/s.

        Returns:
            A tuple containing:
                Q: New discharge array, which is a 1D array with discharge for each grid cell in the river network.
                actual_evaporation_m3: Actual evaporation in m3 for each grid cell in the river network.
                over_abstraction_m3: Over abstraction in m3 for each grid cell in the river network.
                waterbody_storage_m3: Updated storage of each waterbody in m3.
                waterbody_inflow_m3: Inflow to each waterbody in m3.
                outflow_at_pits_m3: Outflow at pits in m3.
                retention_storage_m3: Updated storage of each retention node in m3.
                retention_inflow_m3: Inflow to each retention node in m3.
                retention_outflow_m3: Outflow from each retention node in m3.
        """
        outflow_at_pits_m3 = (
            self.get_total_storage(
                Q_prev_m3_s, river_storage_alpha, river_storage_beta
            )[self.is_pit].sum()
            + sideflow_m3[self.is_pit].sum()
            - evaporation_m3[self.is_pit].sum()
        )
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
            retention_storage_m3=retention_storage_m3,
            retention_max_storage_m3=self.retention_max_storage_m3,
            retention_node_id=self.retention_node_id,
            controlled_retention=self.controlled_retention,
            retention_activation_threshold_m3_s=retention_activation_threshold_m3_s,
            retention_basin_release_threshold_factor=self.retention_basin_release_threshold_factor,
        )

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
