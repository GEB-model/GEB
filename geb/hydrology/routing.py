"""Routing algorithms for river networks."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import pyflwdir.core
from numba import njit
from pyflwdir import core, core_d8, core_ldd

from geb.module import Module
from geb.types import (
    ArrayBool,
    ArrayFloat32,
    ArrayFloat64,
    ArrayInt32,
    ArrayInt64,
    ArrayUint8,
    TwoDArrayBool,
    TwoDArrayFloat32,
    TwoDArrayInt32,
    TwoDArrayUint8,
)
from geb.workflows import balance_check
from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.model import GEBModel, Hydrology

# Wrap pyflwdir core functions with @njit(cache=True) to enable Numba caching.
# This significantly speeds up model initialization by caching the compiled versions
# of these frequently-called functions. The original functions are already JIT-compiled
# but don't have caching enabled.

_upstream_matrix_orig = core.upstream_matrix
_idxs_seq_orig = core.idxs_seq
_from_array_ldd_orig = core_ldd.from_array
_check_values_d8_orig = core_d8.check_values


@njit(cache=True)
def wrap_upstream_matrix(
    idxs_ds: ArrayInt32, mv: np.int64 = core._mv
) -> TwoDArrayInt32:
    """Returns a 2D array with upstream cell indices for each cell.

    The shape of the array is (idxs_ds.size, max number of upstream cells per cell).

    Args:
        idxs_ds: Linear index of next downstream cell.
        mv: Missing value, default is -1.

    Returns:
        2D array with upstream cell indices for each cell.
    """
    return _upstream_matrix_orig(idxs_ds, mv=mv)


@njit(cache=True)
def wrap_idxs_seq(
    idxs_ds: ArrayInt32, idxs_pit: ArrayInt32, mv: np.int64 = core._mv
) -> ArrayInt32:
    """Returns indices ordered from down- to upstream.

    Args:
        idxs_ds: Linear index of next downstream cell.
        idxs_pit: Linear index of pit cells.
        mv: Missing value, default is -1.

    Returns:
        Linear indices of valid cells ordered from down- to upstream.
    """
    return _idxs_seq_orig(idxs_ds, idxs_pit, mv=mv)


@njit(cache=True)
def wrap_from_array_ldd(
    flwdir: TwoDArrayUint8, _mv: np.uint8 = core_ldd._mv, dtype: type = np.intp
) -> tuple[ArrayInt32, ArrayInt32, int]:
    """Convert 2D LDD data to 1D next downstream indices.

    Args:
        flwdir: 2D array with LDD data.
        _mv: Missing value in LDD data.
        dtype: Data type of the output indices.

    Returns:
        Tuple containing:
            - Linear index of next downstream cell.
            - Linear index of pit cells.
            - Number of valid cells.
    """
    return _from_array_ldd_orig(flwdir, _mv=_mv, dtype=dtype)


@njit(cache=True)
def wrap_check_values_d8(
    flwdir: TwoDArrayUint8, _all: ArrayUint8 = core_d8._all
) -> bool:
    """Check if values in D8 flow direction array are valid.

    Args:
        flwdir: 2D array with D8 flow direction data.
        _all: Array with all valid D8 values.

    Returns:
        True if all values are valid, False otherwise.
    """
    return _check_values_d8_orig(flwdir, _all=_all)


core.upstream_matrix = wrap_upstream_matrix
core.idxs_seq = wrap_idxs_seq
core_ldd.from_array = wrap_from_array_ldd
core_d8.check_values = wrap_check_values_d8


def get_river_width(
    alpha: ArrayFloat32,
    beta: ArrayFloat32,
    discharge_m3_s: ArrayFloat32,
) -> ArrayFloat32:
    """Calculate the river width based on the alpha and beta parameters and the discharge.

    Args:
        alpha: The alpha parameter for the river width calculation.
        beta: The beta parameter for the river width calculation.
        discharge_m3_s: The discharge in cubic meters per second.

    Returns:
        A 1D array with the calculated river width for each cell.
    """
    return alpha * discharge_m3_s**beta


def get_channel_ratio(
    river_width: ArrayFloat32,
    river_length: ArrayFloat32,
    cell_area: ArrayFloat32,
) -> ArrayFloat32:
    """Calculate the ratio of the river channel area to the cell area.

    Args:
        river_width: The width of the river in each cell, in meters.
        river_length: The length of the river in each cell, in meters.
        cell_area: The area of each cell, in square meters.

    Returns:
        A 1D array with the ratio of the river channel area to the cell area.
    """
    channel_ratio: ArrayFloat32 = np.minimum(
        1.0,
        river_width * river_length / cell_area,
    )

    assert not np.isnan(channel_ratio).any()
    return channel_ratio


def create_river_network(
    ldd_uncompressed: TwoDArrayUint8, mask: TwoDArrayBool
) -> pyflwdir.FlwdirRaster:
    """Create a river network from a local drain direction (LDD) array.

    Args:
        ldd_uncompressed: A 2D array with the local drain direction (LDD) values.
        mask: A 2D boolean array with the same shape as the LDD array, where True indicates
            that the cell is part of the river network.

    Returns:
        A FlwdirRaster object representing the river network.
    """
    return pyflwdir.from_array(
        ldd_uncompressed,
        ftype="ldd",
        latlon=True,
        mask=mask,
    )


MAX_ITERS: int = 10


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
    ) -> None:
        """Initializes the Router class.

        Args:
            dt: Number of seconds in the time step, must be > 0
            river_network: The river network as a FlwdirRaster object
            waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell.
                -1 indicates no waterbody.
            is_waterbody_outflow: A 1D array with the same shape as the grid, which is True for the outflow cells.
        """
        assert dt > 0, "dt must be greater than 0"
        self.dt = dt

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
    epsilon: np.float32 = np.float32(0.0001),
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
    fQkx: np.float32 = deltaTX * Qnew + alpha * Qnew**beta - C

    # Get the derivative
    dfQkx: np.float32 = deltaTX + alpha * beta * Qnew ** (beta - 1)
    Qnew -= fQkx / dfQkx
    Qnew: np.float32 = max(Qnew, np.float32(1e-30))

    count: int = 0
    while np.abs(fQkx) > epsilon and count < MAX_ITERS:
        fQkx: np.float32 = deltaTX * Qnew + alpha * Qnew**beta - C
        dfQkx: np.float32 = deltaTX + alpha * beta * Qnew ** (beta - 1)
        Qnew -= fQkx / dfQkx
        Qnew: np.float32 = max(Qnew, np.float32(1e-30))

        count += 1

    assert not np.isnan(Qnew), "Qkx is NaN"

    return Qnew, actual_evaporation_m3_s


class KinematicWave(Router):
    """Kinematic wave routing algorithm.

    This class implements the kinematic wave routing algorithm for river networks.
    """

    def __init__(
        self,
        dt: float | int,
        river_network: pyflwdir.FlwdirRaster,
        river_width: ArrayFloat32,
        river_length: ArrayFloat32,
        river_alpha: ArrayFloat32,
        river_beta: float,
        waterbody_id: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
    ) -> None:
        """Initializes the KinematicWave class.

        Args:
            dt: length of the time step in seconds.
            river_network: The river network as a FlwdirRaster object, which contains the flow
                direction and other information about the river network.
            river_width: The width of the river in each cell.
            river_length: The length of the river in each cell,.
            river_alpha: The alpha parameter for the kinematic wave equation.
            river_beta: The beta parameter for the kinematic wave equation.
            waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell.
            is_waterbody_outflow: A 1D array with the same shape as the grid, which is True for the outflow cells.
        """
        super().__init__(dt, river_network, waterbody_id, is_waterbody_outflow)

        self.river_width = river_width.ravel()
        self.river_length = river_length.ravel()
        self.river_alpha = river_alpha.ravel()
        self.river_beta = river_beta

    def calculate_river_storage_from_discharge(
        self,
        discharge: ArrayFloat32,
        river_alpha: ArrayFloat32,
        river_length: ArrayFloat32,
        river_beta: float,
        waterbody_id: ArrayInt32,
    ) -> ArrayFloat32:
        """Calculate the river storage from the discharge using the kinematic wave equation.

        Uses the momentum equation, see eq. 18 in https://gmd.copernicus.org/articles/13/3267/2020/

        Args:
            discharge: The discharge in each cell, in m3/s.
            river_alpha: The alpha parameter for the kinematic wave equation.
            river_length: The length of the river in each cell, in meters.
            river_beta: The beta parameter for the kinematic wave equation.
            waterbody_id: A 1D array with the same shape as the grid, which is the waterbody ID for each cell.

        Returns:
            A 1D array with the calculated river storage for each cell, in m3.
        """
        cross_sectional_area_of_flow: ArrayFloat32 = river_alpha * discharge**river_beta
        river_storage: ArrayFloat32 = cross_sectional_area_of_flow * river_length
        river_storage[waterbody_id != -1] = 0.0
        return river_storage

    def get_available_storage(
        self, Q: ArrayFloat32, maximum_abstraction_ratio: float = 0.9
    ) -> ArrayFloat32:
        """Get the available storage of the river network, which is the sum of the available storage in each cell.

        Args:
            Q: The discharge in each cell, in m3/s.
            maximum_abstraction_ratio: he maximum abstraction ratio, default is 0.9.
                This is the ratio of the available storage that can be used for abstraction.

        Returns:
            The available storage of the river network [m3].
        """
        return self.get_total_storage(Q) * maximum_abstraction_ratio

    def get_total_storage(self, Q: ArrayFloat32) -> ArrayFloat32:
        """Get the total storage of the river network, which is the sum of the available storage in each cell.

        Args:
            Q: The discharge in each cell, in m3/s.

        Returns:
            The total storage of the river network [m3].

        """
        total_storage = self.calculate_river_storage_from_discharge(
            discharge=Q,
            river_alpha=self.river_alpha,
            river_length=self.river_length,
            river_beta=self.river_beta,
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
        waterbody_storage_m3: ArrayFloat32,
        outflow_per_waterbody_m3: ArrayFloat32,
        upstream_matrix_from_up_to_downstream: ArrayInt32,
        idxs_up_to_downstream: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
        waterbody_id: ArrayInt32,
        river_alpha: ArrayFloat32,
        river_beta: np.float32,
        river_length: ArrayFloat32,
    ) -> tuple[ArrayFloat32, ArrayFloat32, ArrayFloat32, ArrayFloat32]:
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
            river_alpha: The alpha parameter for the kinematic wave equation, which is a 1D array with the same shape as the grid.
            river_beta: The beta parameter for the kinematic wave equation, which is a float.
            river_length: Array of floats containing the channel length, must be > 0

        Returns:
            Qnew: New discharge array, which is a 1D array with discharge for each grid cell in the river network.
            actual_evaporation_m3: Actual evaporation in m3 for each grid cell in the river network.
            over_abstraction_m3: Over abstraction in m3 for each grid cell in the river network.
            waterbody_inflow_m3: Inflow to each waterbody in m3.
        """
        Qnew: ArrayFloat32 = np.full_like(Qold, np.nan, dtype=np.float32)
        actual_evaporation_m3: ArrayFloat32 = np.zeros_like(Qold, dtype=np.float32)
        over_abstraction_m3: ArrayFloat32 = np.zeros_like(Qold, dtype=np.float32)
        waterbody_inflow_m3: ArrayFloat32 = np.zeros_like(
            waterbody_storage_m3, dtype=np.float32
        )

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
                assert evaporation_m3[node] == 0.0
            else:
                Qnew[node], actual_evaporation_m3_dt = update_node_kinematic(
                    Qin,
                    Qold[node],
                    sideflow_node_m3 / dt,
                    evaporation_m3[node] / dt,
                    river_alpha[node],
                    river_beta,
                    dt,
                    river_length[node],
                )
                actual_evaporation_m3[node] = actual_evaporation_m3_dt * dt

        return (
            Qnew,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_inflow_m3,
        )

    def step(
        self,
        Q_prev_m3_s: ArrayFloat32,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat32,
        outflow_per_waterbody_m3: ArrayFloat32,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        np.float32,
    ]:
        """Perform a single routing step.

        Discharge is updated based on the inflow from upstream cells, sideflow,
        evaporation, and waterbody outflow. Uses an implicit version of the kinematic wave equation.

        Args:
            Q_prev_m3_s: Old discharge array, which is a 1D array with dicharge for each grid cell in the river network.
            sideflow_m3: Sideflow in m3 for each grid cell in the river network.
            evaporation_m3: Evaporation in m3 for each grid cell in the river network.
            waterbody_storage_m3: Storage of each waterbody in m3.
            outflow_per_waterbody_m3: Outflow of each waterbody in m3.

        Returns:
            Q: New discharge array, which is a 1D array with discharge for each grid cell in the river network.
            actual_evaporation_m3: Actual evaporation in m3 for each grid cell in the river network.
            over_abstraction_m3: Over abstraction in m3 for each grid cell in the river network.
            waterbody_storage_m3: Updated storage of each waterbody in m3.
            waterbody_inflow_m3: Inflow to each waterbody in m3.
            outflow_at_pits_m3: Outflow at pits in m3.
        """
        Q, actual_evaporation_m3, over_abstraction_m3, waterbody_inflow_m3 = self._step(
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
            river_alpha=self.river_alpha,
            river_beta=self.river_beta,
            river_length=self.river_length,
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
        )


def fill_discharge_gaps(
    discharge_m3_s: ArrayFloat32,
    rivers: gpd.GeoDataFrame | pd.DataFrame,
    waterbody_ids: ArrayInt32,
    outflow_per_waterbody_m3_s: ArrayFloat32,
) -> ArrayFloat32:
    """Fill gaps with NaN values in discharge data with valid discharge data.

    First, discharge values are filled:

    1) with discharges from waterbodies if available
    2) by propagating discharge values from up to downstream
    3) by propagating discharge values from downstream to upstream

    Todo:
        In some cases when a river is entirely in a waterbody, the discharge
        cannot be filled. In these cases, we currently leave the discharge as NaN.

    Args:
        discharge_m3_s: 1D array of discharge values with possible NaNs.
        rivers: GeoDataFrame containing river network geometries.
        waterbody_ids: 1D array with waterbody IDs for each cell. This must be the same
            size as discharge_m3_s. -1 indicates no waterbody, and non-negative values indicate
            waterbody cells.
        outflow_per_waterbody_m3_s: 1D array with outflow values for each waterbody. This
            must be the same size as the number of unique waterbody IDs.

    Returns:
        1D array of discharge values with NaNs in rivers filled.
    """
    filled_discharge_m3_s: ArrayFloat32 = discharge_m3_s.copy()
    for river_id, river in rivers.iterrows():
        # iterate from upstream to downstream
        valid_discharge: np.float32 = np.float32(np.nan)
        for idx in river["hydrography_linear"]:
            if not np.isnan(discharge_m3_s[idx]):
                valid_discharge = discharge_m3_s[idx]

            elif np.isnan(filled_discharge_m3_s[idx]):
                if waterbody_ids[idx] != -1:
                    waterbody_id = waterbody_ids[idx]
                    valid_discharge = outflow_per_waterbody_m3_s[waterbody_id]

                filled_discharge_m3_s[idx] = valid_discharge

        if np.isnan(valid_discharge):
            warnings.warn(
                f"WARNING: No valid discharge found for river: {river_id}, please let Jens know."
            )
            continue  # skip if no valid discharge found

        down_stream_discharge: np.float32 = filled_discharge_m3_s[
            river["hydrography_linear"][-1]
        ]
        for idx in reversed(river["hydrography_linear"][:-1]):
            current_discharge: np.float32 = filled_discharge_m3_s[idx]
            if np.isnan(current_discharge):
                filled_discharge_m3_s[idx] = down_stream_discharge
            else:
                down_stream_discharge = current_discharge

    return filled_discharge_m3_s


class Accuflux(Router):
    """Accuflux routing algorithm.

    In each step, the algorithm calculates the new discharge for each cell
    based on the inflow from upstream cells, sideflow, and waterbody outflow.

    The algorithm works as follows:

    1. For each cell, it calculates the inflow from upstream cells.
    2. It adds the sideflow and waterbody outflow to the inflow.
    3. It calculates the new discharge for each cell based on the inflow.
    4. It updates the waterbody storage based on the outflow.
    """

    def __init__(
        self,
        dt: float | int,
        river_network: pyflwdir.FlwdirRaster,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the Accuflux class.

        Args:
            dt: Number of seconds in the time step, must be > 0
            river_network: The river network as a FlwdirRaster object
            *args: Additional arguments to pass to the Router class.
            **kwargs: Additional keyword arguments to pass to the Router class.
        """
        super().__init__(dt, river_network, *args, **kwargs)

    def get_available_storage(
        self, Q: ArrayFloat32, maximum_abstraction_ratio: float = 0.9
    ) -> ArrayFloat32:
        """Get the available storage of the river network, which is the sum of the available storage in each cell.

        Available storage in lakes and reservoirs is set to 0.

        Args:
            Q: The discharge in each cell, in m3/s.
            maximum_abstraction_ratio: The maximum abstraction ratio, default is 0.9.
                This is the ratio of the available storage that can be used for abstraction.

        Returns:
            The available storage of the river network.
        """
        available_storage = Q * self.dt * maximum_abstraction_ratio
        available_storage[self.waterbody_id != -1] = 0.0
        assert not np.isnan(available_storage).any()
        return available_storage

    @staticmethod
    @njit(cache=True)
    def _step(
        dt: int,
        Qold: ArrayFloat32,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat32,
        outflow_per_waterbody_m3: ArrayFloat32,
        upstream_matrix_from_up_to_downstream: TwoDArrayInt32,
        idxs_up_to_downstream: ArrayInt32,
        is_waterbody_outflow: ArrayBool,
        waterbody_id: ArrayInt32,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
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

        Returns:
            Qnew: New discharge array, which is a 1D array with discharge for each grid cell in the river network.
            actual_evaporation_m3: Actual evaporation in m3 for each grid cell in the river network.
            over_abstraction_m3: Over abstraction in m3 for each grid cell in the river network.
            waterbody_inflow_m3: Inflow to each waterbody in m3.
        """
        Qold += sideflow_m3 / dt

        evaporation_m3_s: ArrayFloat32 = evaporation_m3 / np.float32(dt)
        actual_evaporation_m3_s: ArrayFloat32 = np.minimum(evaporation_m3_s, Qold)
        actual_evaporation_m3: ArrayFloat32 = actual_evaporation_m3_s * dt
        actual_evaporation_m3[waterbody_id != -1] = 0.0

        Qold -= actual_evaporation_m3_s

        Qnew: ArrayFloat32 = np.full_like(Qold, np.nan, dtype=np.float32)
        over_abstraction_m3: ArrayFloat32 = np.zeros_like(Qold, dtype=np.float32)
        waterbody_inflow_m3: ArrayFloat32 = np.zeros_like(
            waterbody_storage_m3, dtype=np.float32
        )
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
                waterbody_inflow_m3[node_waterbody_id] += inflow_volume
            else:
                Qnew_node = inflow_volume / dt
                if Qnew_node < 0.0:
                    # if the new discharge is negative, we have over-abstraction
                    over_abstraction_m3[node] = -Qnew_node * dt
                    Qnew_node = 0.0
                Qnew[node] = Qnew_node
                assert Qnew[node] >= 0.0, "Discharge cannot be negative"
        return Qnew, actual_evaporation_m3, over_abstraction_m3, waterbody_inflow_m3

    def step(
        self,
        Q_prev_m3_s: ArrayFloat32,
        sideflow_m3: ArrayFloat32,
        evaporation_m3: ArrayFloat32,
        waterbody_storage_m3: ArrayFloat32,
        outflow_per_waterbody_m3: ArrayFloat32,
    ) -> tuple[
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        ArrayFloat32,
        np.float32,
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

        Returns:
            A tuple containing:
                Q: New discharge array, which is a 1D array with discharge for each grid cell in the river network.
                actual_evaporation_m3: Actual evaporation in m3 for each grid cell in the river network.
                over_abstraction_m3: Over abstraction in m3 for each grid cell in the river network.
                waterbody_storage_m3: Updated storage of each waterbody in m3.
                waterbody_inflow_m3: Inflow to each waterbody in m3.
                outflow_at_pits_m3: Outflow at pits in m3.
        """
        outflow_at_pits_m3 = (
            self.get_total_storage(Q_prev_m3_s)[self.is_pit].sum()
            + sideflow_m3[self.is_pit].sum()
            - evaporation_m3[self.is_pit].sum()
        )
        Q, actual_evaporation_m3, over_abstraction_m3, waterbody_inflow_m3 = self._step(
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
        )

        return (
            Q,
            actual_evaporation_m3,
            over_abstraction_m3,
            waterbody_storage_m3,
            waterbody_inflow_m3,
            outflow_at_pits_m3,
        )

    def get_total_storage(self, Q: ArrayFloat32) -> ArrayFloat32:
        """Get the total storage of the river network, which is the sum of the available storage in each cell.

        Args:
            Q: The discharge in each cell, in m3/s.

        Returns:
            The total storage of the river network [m3].

        """
        return self.get_available_storage(Q, maximum_abstraction_ratio=1.0)


class Routing(Module):
    """Routing module of the hydrological model.

    Args:
        model: The GEB model instance.
        hydrology: The hydrology submodel instance.
    """

    def __init__(self, model: GEBModel, hydrology: Hydrology) -> None:
        """Initialize the Routing module.

        Args:
            model: The GEB model instance.
            hydrology: The hydrology submodel instance.

        """
        super().__init__(model)

        self.config = model.config["hydrology"]["routing"]

        self.default_missing_channel_width: float = (
            3.0  # Default width for missing values
        )

        self.hydrology = hydrology

        self.HRU = hydrology.HRU
        self.grid = hydrology.grid

        self.ldd: ArrayUint8 = self.grid.load(
            self.model.files["grid"]["routing/ldd"],
        )

        mask: TwoDArrayBool = ~self.grid.mask

        ldd_uncompressed: TwoDArrayUint8 = np.full_like(mask, 255, dtype=self.ldd.dtype)
        ldd_uncompressed[mask] = self.ldd.ravel()

        self.river_network: pyflwdir.FlwdirRaster = create_river_network(
            ldd_uncompressed=ldd_uncompressed, mask=mask
        )

        self.rivers: gpd.GeoDataFrame = self.load_rivers(
            grid_linear_mapping=self.grid.linear_mapping
        )

        self.river_ids = self.grid.load(
            self.model.files["grid"]["routing/river_ids"],
        )

        if self.model.in_spinup:
            self.spinup()

    def load_rivers(self, grid_linear_mapping: TwoDArrayInt32) -> gpd.GeoDataFrame:
        """Load the river network geometries.

        Args:
            grid_linear_mapping: A 2D array mapping grid cells to linear indices.

        Returns:
            A GeoDataFrame containing the river network geometries.
        """
        rivers: gpd.GeoDataFrame = read_geom(self.model.files["geom"]["routing/rivers"])
        rivers["hydrography_linear"] = rivers["hydrography_xy"].apply(
            lambda xys: np.array(
                [grid_linear_mapping[xy[1], xy[0]] for xy in xys], dtype=np.int32
            )
        )
        return rivers

    def set_router(self) -> None:
        """Initialize the routing algorithm based on the configuration.

        Options in the configuration are 'kinematic_wave' and 'accuflux'.

        Raises:
            ValueError: If an unknown routing algorithm is specified in the configuration.
        """
        routing_algorithm: str = self.config["algorithm"]
        is_waterbody_outflow: ArrayBool = self.grid.var.waterbody_outflow_points != -1
        if routing_algorithm == "kinematic_wave":
            river_width: ArrayFloat32 = np.where(
                ~np.isnan(self.grid.var.average_river_width),
                self.grid.var.average_river_width,
                self.default_missing_channel_width,
            )
            self.router = KinematicWave(
                dt=3600,
                river_network=self.river_network,
                river_width=river_width,
                river_length=self.grid.var.river_length,
                river_alpha=self.grid.var.river_alpha,
                river_beta=self.var.river_beta,
                waterbody_id=self.grid.var.waterBodyID,
                is_waterbody_outflow=is_waterbody_outflow,
            )
        elif routing_algorithm == "accuflux":
            self.router = Accuflux(
                dt=3600,
                river_network=self.river_network,
                waterbody_id=self.grid.var.waterBodyID,
                is_waterbody_outflow=is_waterbody_outflow,
            )
        else:
            raise ValueError(
                f"Unknown routing algorithm: {routing_algorithm}. "
                "Available algorithms are 'kinematic_wave' and 'accuflux'."
            )

    def spinup(self) -> None:
        """Initialize routing variables during model spinup.

        Steps:
        1. Load upstream area, Manning's n, river length, and river width from grid files.
        2. Set number of routing substeps per day and kinematic wave parameter.
        3. Calculate routing step length in seconds.
        4. Compute river alpha parameter for kinematic wave routing.
        5. Initialize discharge variables and counters.

        """
        self.grid.var.upstream_area = self.grid.load(
            self.model.files["grid"]["routing/upstream_area"]
        )
        if "routing/upstream_area_n_cells" in self.model.files["grid"]:
            self.grid.var.upstream_area_n_cells = self.grid.load(
                self.model.files["grid"]["routing/upstream_area_n_cells"]
            )
        else:
            # TODO: Remove this in feb 2026
            self.grid.var.upstream_area_n_cells = self.river_network.upstream_area(
                unit="cell"
            )[~self.grid.mask]

        # kinematic wave parameter: 0.6 is for broad sheet flow
        self.var.river_beta = 0.6  # TODO: Make this a parameter

        # Channel Manning's n
        self.grid.var.river_mannings = (
            self.grid.load(self.model.files["grid"]["routing/mannings"])
            * self.model.config["parameters"]["mannings_n_multiplier"]
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
        self.grid.var.average_river_width = self.grid.load(
            self.model.files["grid"]["routing/river_width"]
        )

        # for a river, the wetted perimeter can be approximated by the channel width
        river_wetted_perimeter = np.where(
            ~np.isnan(self.grid.var.average_river_width),
            self.grid.var.average_river_width,
            self.default_missing_channel_width,  # Default value for missing values
        )

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
        self.grid.var.discharge_in_rivers_m3_s_substep: ArrayFloat32 = (
            self.grid.full_compressed(1e-30, dtype=np.float32)
        )
        self.grid.var.discharge_m3_s_substep: ArrayFloat32 = self.grid.full_compressed(
            1e-30, dtype=np.float32
        )
        self.grid.var.discharge_m3_s_per_substep: TwoDArrayFloat32 = np.full(
            (24, self.grid.var.discharge_m3_s_substep.size),
            0,
            dtype=self.grid.var.discharge_m3_s_substep.dtype,
        )

        self.var.sum_of_all_discharge_steps: ArrayFloat64 = self.grid.full_compressed(
            0, dtype=np.float64
        )
        self.var.discharge_step_count: int = 0

    def get_river_width_alpha_and_beta(
        self,
        beta: float,
        default_alpha: float,
    ) -> tuple[ArrayFloat32, ArrayFloat32]:
        """Calculate the river alpha parameter for the kinematic wave routing.

        For river widths where we have an observed average river width, we use the default
        values for the first year of simulation, and then calculate the river width
        based on the average river width and the discharge using the formula

        river_width = alpha * discharge^beta

        for alpha a global value of 7.2 is used, and beta is set to 0.50
        based on https://doi.org/10.1002/esp.403

        for rivers where we don't have an observed average river width, we use the default values
        throughout the simulation.

        Args:
            beta: The beta parameter for the kinematic wave routing, default is 0.50.
            default_alpha: The default alpha value to use for rivers without an observed average river width,
                default is 7.2.

        Returns:
            A tuple containing:
            - alpha: The alpha parameter for the kinematic wave routing, which is a 1D array with the same shape as the grid.
            - beta_array: The beta parameter for the kinematic wave routing, which is a 1D array with the same shape as the grid.
        """
        beta_array: ArrayFloat32 = np.full_like(
            self.grid.var.average_river_width, beta, dtype=np.float32
        )
        # for the first year of simulation, we use the default alpha value for all rivers
        if self.var.discharge_step_count < 365 * 24:
            alpha: ArrayFloat32 = np.full_like(
                self.grid.var.average_river_width,
                default_alpha,
                dtype=np.float32,
            )
        else:
            average_discharge: ArrayFloat32 = self.var.sum_of_all_discharge_steps / (
                self.var.discharge_step_count
            )

            alpha: ArrayFloat32 = np.where(
                ~np.isnan(self.grid.var.average_river_width),
                self.grid.var.average_river_width / (average_discharge**beta_array),
                default_alpha,
            )

        return alpha, beta_array

    def step(
        self,
        total_runoff_m: ArrayFloat32,
        channel_abstraction_m3: ArrayFloat32,
        return_flow: ArrayFloat32,
        reference_evapotranspiration_water_m: TwoDArrayFloat32,
    ) -> tuple[
        np.float64,
        np.float64,
    ]:
        """Perform a daily routing step with multiple substeps.

        Args:
            total_runoff_m: Total runoff in meters for each grid cell for each hour.
                Shape is (24, n_cells).
            channel_abstraction_m3: Channel abstraction in m3 for each grid cell over the whole day.
            return_flow: Return flow in meters for each grid cell over the whole day.
            reference_evapotranspiration_water_m: Reference evapotranspiration from water in meters for for each grid cell for each hour.

        Returns:
            A tuple containing:
            - Total routing loss, including outflow at pits, evaporation in rivers and water bodies,
            - Total over abstraction in m3. This should be zero if the abstraction is within the available storage.
                Otherwise, it indicates the amount of abstraction that could not be met and indicates an error
                in the model.

        """
        if __debug__:
            pre_storage: np.ndarray = self.hydrology.lakes_reservoirs.var.storage.copy()
            pre_river_storage_m3: ArrayFloat32 = self.router.get_total_storage(
                self.grid.var.discharge_in_rivers_m3_s_substep
            )

        channel_abstraction_m3_per_hour: np.ndarray = channel_abstraction_m3 / 24
        assert (
            channel_abstraction_m3_per_hour[self.grid.var.waterBodyID != -1] == 0.0
        ).all(), (
            "Channel abstraction must be zero for water bodies, "
            "but found non-zero value."
        )

        return_flow_m3_per_hour: np.ndarray = return_flow * self.grid.var.cell_area / 24

        # add return flow to the water bodies
        return_flow_m3_to_water_bodies_per_hour: np.ndarray = np.bincount(
            self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
            weights=return_flow_m3_per_hour[self.grid.var.waterBodyID != -1],
        )
        return_flow_m3_per_hour[self.grid.var.waterBodyID != -1] = 0.0

        self.grid.var.discharge_m3_s_per_substep: TwoDArrayFloat32 = np.full_like(
            self.grid.var.discharge_m3_s_per_substep,
            fill_value=np.nan,
        )

        if __debug__:
            # these are for balance checks, the sum of all routing steps
            evaporation_in_rivers_m3: ArrayFloat32 = self.grid.full_compressed(
                0, dtype=np.float32
            )
            waterbody_evaporation_m3: ArrayFloat32 = np.zeros(
                self.hydrology.lakes_reservoirs.n, dtype=np.float32
            )
            outflow_at_pits_m3 = np.float32(0)
            command_area_release_m3 = np.float32(0)

        over_abstraction_m3: ArrayFloat32 = self.grid.full_compressed(
            0, dtype=np.float32
        )

        for hour in range(24):
            total_runoff_m3: np.ndarray = (
                total_runoff_m[hour, :] * self.grid.var.cell_area
            )

            # then split the runoff into runoff directly to water bodies
            # and runoff to the channel network
            self.hydrology.lakes_reservoirs.var.storage += np.bincount(
                self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                weights=total_runoff_m3[self.grid.var.waterBodyID != -1],
            )

            # after adding the runoff to the water bodies, we set the runoff to zero
            # in those grid cells
            total_runoff_m3[self.grid.var.waterBodyID != -1] = 0.0

            self.hydrology.lakes_reservoirs.var.storage += (
                return_flow_m3_to_water_bodies_per_hour
            )

            # TODO: This calculation can be optimized by pre-calculating some parts
            potential_evaporation_per_water_body_m3 = (
                np.bincount(
                    self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1],
                    weights=reference_evapotranspiration_water_m[
                        hour, self.grid.var.waterBodyID != -1
                    ],
                )
                / np.bincount(
                    self.grid.var.waterBodyID[self.grid.var.waterBodyID != -1]
                )
                * self.hydrology.lakes_reservoirs.var.lake_area
            )

            actual_evaporation_from_water_bodies_per_hour_m3 = np.minimum(
                potential_evaporation_per_water_body_m3,
                self.hydrology.lakes_reservoirs.var.storage,
            )

            self.hydrology.lakes_reservoirs.var.storage -= (
                actual_evaporation_from_water_bodies_per_hour_m3
            )

            outflow_per_waterbody_m3, command_area_release_m3_routing_step = (
                self.hydrology.lakes_reservoirs.substep(
                    current_substep=hour,
                    n_routing_substeps=24,
                    routing_step_length_seconds=3600,
                )
            )

            self.hydrology.lakes_reservoirs.var.storage -= (
                command_area_release_m3_routing_step
            )

            assert (
                outflow_per_waterbody_m3 <= self.hydrology.lakes_reservoirs.var.storage
            ).all(), "outflow cannot be smaller or equal to storage"

            side_flow_channel_m3_per_hour = (
                total_runoff_m3
                + return_flow_m3_per_hour
                - channel_abstraction_m3_per_hour
            )
            assert (
                side_flow_channel_m3_per_hour[self.grid.var.waterBodyID != -1] == 0
            ).all()

            if self.model.in_spinup:
                self.model.var.river_width_alpha, self.model.var.river_width_beta = (
                    self.get_river_width_alpha_and_beta(
                        default_alpha=self.config["river_width"]["parameters"][
                            "default_alpha"
                        ],
                        beta=self.config["river_width"]["parameters"]["beta"],
                    )
                )

            assert (
                self.grid.var.discharge_in_rivers_m3_s_substep[
                    self.grid.var.waterBodyID == -1
                ]
                >= 0.0
            ).all()

            river_width: ArrayFloat32 = get_river_width(
                self.model.var.river_width_alpha,
                self.model.var.river_width_beta,
                self.grid.var.discharge_in_rivers_m3_s_substep,
            )
            # the ratio of each grid cell that is currently covered by a river
            channel_ratio: ArrayFloat32 = get_channel_ratio(
                river_length=self.grid.var.river_length,
                river_width=np.where(self.grid.var.waterBodyID == -1, river_width, 0),
                cell_area=self.grid.var.cell_area,
            )

            # calculate evaporation from rivers per timestep usting the current channel ratio
            potential_evaporation_in_rivers_m3_per_hour = (
                reference_evapotranspiration_water_m[hour]
                * channel_ratio
                * self.grid.var.cell_area
            )

            (
                self.grid.var.discharge_in_rivers_m3_s_substep,
                actual_evaporation_in_rivers_m3_per_hour,
                over_abstraction_m3_routing_step,
                self.hydrology.lakes_reservoirs.var.storage,
                waterbody_inflow_m3,
                outflow_at_pits_m3_routing_step,
            ) = self.router.step(
                Q_prev_m3_s=self.grid.var.discharge_in_rivers_m3_s_substep,
                sideflow_m3=side_flow_channel_m3_per_hour.astype(np.float32),
                evaporation_m3=potential_evaporation_in_rivers_m3_per_hour,
                waterbody_storage_m3=self.hydrology.lakes_reservoirs.var.storage,
                outflow_per_waterbody_m3=outflow_per_waterbody_m3,
            )

            assert (actual_evaporation_in_rivers_m3_per_hour >= 0.0).all()

            # the reservoir operators need to track the inflow to the reservoirs
            self.model.agents.reservoir_operators.track_inflow(
                waterbody_inflow_m3[self.model.hydrology.lakes_reservoirs.is_reservoir]
            )

            # ensure that discharge is nan for water bodies
            assert np.isnan(
                self.grid.var.discharge_in_rivers_m3_s_substep[
                    self.grid.var.waterBodyID != -1
                ]
            ).all()

            discharge_m3_s_substep: ArrayFloat32 = (
                self.grid.var.discharge_in_rivers_m3_s_substep.copy()
            )

            discharge_m3_s_substep_filled: ArrayFloat32 = fill_discharge_gaps(
                self.grid.var.discharge_in_rivers_m3_s_substep,
                rivers=self.rivers,
                waterbody_ids=self.grid.var.waterBodyID,
                outflow_per_waterbody_m3_s=outflow_per_waterbody_m3 / np.float32(3600),
            )

            self.grid.var.discharge_m3_s_per_substep[hour, :] = (
                discharge_m3_s_substep_filled
            )

            self.var.sum_of_all_discharge_steps += discharge_m3_s_substep_filled
            self.var.discharge_step_count += 1

            assert (
                self.router.get_available_storage(
                    self.grid.var.discharge_in_rivers_m3_s_substep
                )
                >= 0.0
            ).all()

            if __debug__:
                # Discharge at outlets and lakes and reservoirs
                outflow_at_pits_m3 += outflow_at_pits_m3_routing_step
                waterbody_evaporation_m3 += (
                    actual_evaporation_from_water_bodies_per_hour_m3
                )
                evaporation_in_rivers_m3 += actual_evaporation_in_rivers_m3_per_hour
                over_abstraction_m3 += over_abstraction_m3_routing_step
                command_area_release_m3 += command_area_release_m3_routing_step

        self.grid.var.discharge_m3_s: ArrayFloat32 = (
            self.grid.var.discharge_m3_s_per_substep.mean(axis=0)
        )

        if __debug__:
            # TODO: make dependent on routing step length
            river_storage_m3: ArrayFloat32 = self.router.get_total_storage(
                self.grid.var.discharge_in_rivers_m3_s_substep
            )
            balance_check(
                how="sum",
                influxes=[
                    total_runoff_m.sum(axis=0) * self.grid.var.cell_area,
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
                tolerance=100,
            )

            total_evaporation_in_rivers_m3: np.float64 = (
                evaporation_in_rivers_m3.astype(np.float64).sum()
            )
            total_waterbody_evaporation_m3: np.float64 = (
                waterbody_evaporation_m3.astype(np.float64).sum()
            )
            total_outflow_at_pits_m3: np.float64 = outflow_at_pits_m3.astype(
                np.float64
            ).sum()

            assert total_evaporation_in_rivers_m3 >= 0
            assert total_waterbody_evaporation_m3 >= 0
            assert total_outflow_at_pits_m3 >= 0

            routing_loss: np.float64 = (
                total_evaporation_in_rivers_m3
                + total_waterbody_evaporation_m3
                + total_outflow_at_pits_m3
            )

            assert routing_loss >= 0, "Routing loss cannot be negative"

        # outside debug, we return NaN for routing loss
        else:
            routing_loss: np.float64 = np.float64(np.nan)

        self.report(locals())

        total_over_abstraction_m3: np.float64 = over_abstraction_m3.astype(
            np.float64
        ).sum()
        if over_abstraction_m3.sum() > 100:
            print(
                f"Total over-abstraction in routing step is {total_over_abstraction_m3:.2f} m³"
            )

        return routing_loss, total_over_abstraction_m3

    @property
    def name(self) -> str:
        """Name of the module."""
        return "hydrology.routing"

    @property
    def outflow_rivers(self) -> gpd.GeoDataFrame:
        """Get the outflow rivers.

        Returns:
            A GeoDataFrame containing the outflow rivers.
        """
        rivers: gpd.GeoDataFrame = self.rivers
        rivers = rivers[~rivers["is_downstream_outflow_subbasin"]]
        outflow_rivers: gpd.GeoDataFrame = rivers[
            ~rivers["downstream_ID"].isin(rivers.index)
        ]
        return outflow_rivers
