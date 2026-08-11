"""Routing base class for the GEB model."""

import numpy as np
import pyflwdir
import pyflwdir.core
from numba import njit

from geb.geb_types import (
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    ArrayInt64,
)


class Router:
    """Generic routing class.

    This class is the base class for all routing algorithms. It provides the
    basic functionality for routing, such as the upstream matrix and the
    indices of the cells in the river network.

    Args:
        dt: The time step in seconds, must be greater than 0.
        river_network: The river network as a FlwdirRaster object, which contains the flow
            direction and other information about the river network.
        is_waterbody_outflow: A 1D array with the same shape as the grid, which is True for the outflow cells.
        waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell.

    Notes:
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

        Sets the following attributes:
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
    """

    def __init__(
        self,
        dt: float | int,
        river_network: pyflwdir.FlwdirRaster,
        waterbody_id: np.ndarray,
        is_waterbody_outflow: np.ndarray,
        retention_basin_release_threshold_factor: float,
    ) -> None:
        """Initializes the Router class.

        Args:
            dt: Number of seconds in the time step, must be > 0
            river_network: The river network as a FlwdirRaster object
            waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell.
                -1 indicates no waterbody.
            is_waterbody_outflow: A 1D array with the same shape as the grid, which is True for the outflow cells.
            retention_basin_release_threshold_factor: Factor to multiply the activation threshold by to get the release threshold.
        """
        assert dt > 0, "dt must be greater than 0"
        self.dt = dt
        self.retention_basin_release_threshold_factor = np.float32(
            retention_basin_release_threshold_factor
        )

        # we create a mapper from the 2D ldd to the 1D river network
        # the mapper size is ldd.size + 1, because we need to map the
        # the "nan-value" of the ldd to -1 in the river network, thus
        # mapping -1 to -1.
        mapper: ArrayInt32 = np.full(river_network.size + 1, -1, dtype=np.int32)
        indices: ArrayInt64 = np.arange(river_network.size, dtype=np.int32)[
            river_network.mask
        ]
        mapper[indices] = np.arange(indices.size, dtype=np.int32)

        river_network.order_cells(method="walk")
        upstream_matrix: ArrayInt32 = pyflwdir.core.upstream_matrix(
            river_network.idxs_ds,
        )

        self.idxs_up_to_downstream: ArrayInt32 = river_network.idxs_seq[::-1]

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

    def get_total_storage(
        self,
        Q: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
    ) -> ArrayFloat64:
        """Get the total storage of the river network.

        Args:
            Q: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.

        Returns:
            The total storage of the river network [m3].
        """
        raise NotImplementedError

    def get_available_storage(
        self,
        Q: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        maximum_abstraction_ratio: float = 0.9,
    ) -> ArrayFloat64:
        """Get the available storage of the river network.

        Args:
            Q: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            maximum_abstraction_ratio: The maximum abstraction ratio.

        Returns:
            The available storage of the river network [m3].
        """
        raise NotImplementedError

    def calculate_river_storage_from_discharge(
        self,
        discharge: ArrayFloat32,
        river_storage_alpha: ArrayFloat32,
        river_length: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat64:
        """Calculate the river storage from the discharge.

        Args:
            discharge: The discharge in each cell, in m3/s.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_length: The length of the river in each cell, in meters.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            waterbody_id: A 1D array with the same shape as the grid.

        Returns:
            A 1D array with the calculated river storage for each cell, in m3.
        """
        raise NotImplementedError

    def calculate_discharge_from_river_storage(
        self,
        river_storage: ArrayFloat64,
        river_storage_alpha: ArrayFloat32,
        river_storage_beta: ArrayFloat32,
        river_length: ArrayFloat32,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat32:
        """Calculate the discharge from the river storage.

        Args:
            river_storage: The storage in each cell, in m3.
            river_storage_alpha: The alpha parameter for the kinematic wave equation.
            river_storage_beta: The beta parameter for the kinematic wave equation.
            river_length: The length of the river in each cell, in meters.
            waterbody_id: A 1D array with the same shape as the grid.

        Returns:
            A 1D array with the calculated discharge for each cell, in m3/s.
        """
        raise NotImplementedError


@njit(cache=True)
def compute_retention_routing(
    dt: np.float32,
    river_volume_m3: np.float32,
    discharge_before_diversion_m3_s: np.float32,
    is_rising_limb: bool,
    retention_storage_m3: np.float32,
    retention_max_storage_m3: np.float32,
    controlled_retention: bool,
    activation_threshold_m3_s: np.float32,
    release_threshold_factor: np.float32,
    inflow_limit_m3: np.float32,
    max_outflow_limit_m3: np.float32,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Calculate the water diversion and release for a single retention basin node.

    This function computes how much water is diverted from the river into a
    retention basin (during high flows) or released from the basin back into
    the river (during low flows) based on predefined thresholds.

    Notes:
        Diversions only occur when the river's discharge rate exceeds the activation threshold.
        For controlled retention basins, diversion can only occur on a rising limb of the hydrograph.
        Releases only occur when the storage is non-empty and the initial discharge rate is below
        the release threshold. High flows and low flows are handled as mutually exclusive states.

    Args:
        dt: The length of the routing time step (seconds).
        river_volume_m3: The current available volume of water in the river cell (m3).
        discharge_before_diversion_m3_s: The initial discharge rate in the river cell before diversion (m3/s).
        is_rising_limb: Flag indicating if the discharge is currently rising compared to the previous time step.
        retention_storage_m3: The current amount of water stored in the retention basin (m3).
        retention_max_storage_m3: The maximum storage capacity of the retention basin (m3).
        controlled_retention: Flag indicating whether the retention basin is controlled.
        activation_threshold_m3_s: The discharge threshold above which water is diverted into the basin (m3/s).
        release_threshold_factor: The multiplier applied to the activation threshold to determine the release threshold.
        inflow_limit_m3: The physical limit of water volume that can enter the basin during this step (m3).
        max_outflow_limit_m3: The maximum volume of water that the basin can release during this step (m3).

    Returns:
        A tuple containing:
            diverted_volume_m3: The volume of water diverted into the basin (m3).
            outflow_volume_m3: The volume of water released from the basin (m3).
            updated_retention_storage_m3: The updated retention storage (m3).
            final_river_volume_m3: The updated available water volume in the river cell (m3).

    Raises:
        ValueError: If `dt` is <= 0 or if `retention_max_storage_m3` is negative.
    """
    if dt <= np.float32(0.0):
        raise ValueError("Time step dt must be positive.")
    if retention_max_storage_m3 < np.float32(0.0):
        raise ValueError("Maximum retention storage must be non-negative.")

    # Determine available storage in the retention basin. Can not be negative.
    available_storage_m3: np.float32 = max(
        np.float32(0.0),
        retention_max_storage_m3 - retention_storage_m3,
    )

    diverted_volume_m3: np.float32 = np.float32(0.0)
    outflow_volume_m3: np.float32 = np.float32(0.0)

    # Determine the release threshold
    release_threshold_m3_s: np.float32 = (
        activation_threshold_m3_s * release_threshold_factor
    )

    # High flow branch (diversion)
    if discharge_before_diversion_m3_s > activation_threshold_m3_s:
        if controlled_retention:
            if not is_rising_limb:
                # If the discharge is not rising, we do not divert any water.
                diverted_volume_m3: np.float32 = np.float32(0.0)
            else:
                discharge_above_activation_threshold_m3_s: np.float32 = (
                    discharge_before_diversion_m3_s - activation_threshold_m3_s
                )
                diverted_volume_m3: np.float32 = min(
                    river_volume_m3,
                    available_storage_m3,
                    inflow_limit_m3,
                    discharge_above_activation_threshold_m3_s * dt,
                )
        else:
            # Uncontrolled basin: divert any discharge above the activation threshold.
            discharge_above_activation_threshold_m3_s: np.float32 = (
                discharge_before_diversion_m3_s - activation_threshold_m3_s
            )
            diverted_volume_m3: np.float32 = min(
                river_volume_m3,
                available_storage_m3,
                inflow_limit_m3,
                discharge_above_activation_threshold_m3_s * dt,
            )

    # Low flow branch (release)
    elif discharge_before_diversion_m3_s <= release_threshold_m3_s:
        if retention_storage_m3 > np.float32(0.0):
            # We ensure the release does not cause the river flow rate to exceed the release threshold.
            allowed_extra_outflow_m3: np.float32 = (
                release_threshold_m3_s - discharge_before_diversion_m3_s
            ) * dt
            outflow_volume_m3: np.float32 = min(
                allowed_extra_outflow_m3,
                max_outflow_limit_m3,
            )
            outflow_volume_m3 = max(np.float32(0.0), outflow_volume_m3)
            outflow_volume_m3 = min(outflow_volume_m3, retention_storage_m3)

    # Update retention storage and remaining river volume.
    updated_retention_storage_m3: np.float32 = (
        retention_storage_m3 + diverted_volume_m3 - outflow_volume_m3
    )
    final_river_volume_m3: np.float32 = (
        river_volume_m3 - diverted_volume_m3 + outflow_volume_m3
    )

    return (
        diverted_volume_m3,
        outflow_volume_m3,
        updated_retention_storage_m3,
        final_river_volume_m3,
    )
