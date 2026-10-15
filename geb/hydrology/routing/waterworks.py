"""Sample river-barrier diagnostics without changing hydraulic calculations."""

import numpy as np

from geb.geb_types import ArrayFloat32, ArrayFloat64, ArrayInt32, TwoDArrayFloat32

from .inertial_substeps import (
    _compute_reach_stage_from_vol,
    compute_instream_dam_open_fraction,
)
from .local_inertial import LocalInertial


def sample_waterworks(
    router: LocalInertial,
    river_storage_m3: ArrayFloat64,
    discharge_m3_s: ArrayFloat32,
    barrier_indices: ArrayInt32,
) -> TwoDArrayFloat32:
    """Sample opening, total barrier outflow, and upstream river inflow.

    Notes:
        Opening is an end-of-hour snapshot of the modeled opening rule, not
        an observed gate setting or an hourly mean. Outflow includes both
        the opening and overflow. Inflow sums immediate upstream river links;
        lateral runoff, abstractions and return flows are excluded.
        Fixed barriers have an opening of -1 (no modeled gate).

    Args:
        router: Local-inertial solver with current channel geometry.
        river_storage_m3: Storage at the end of the hour (m³), in grid order.
        discharge_m3_s: Reported hourly river discharge (m³/s), in grid order.
        barrier_indices: Barrier indices within the inertial solver domain.

    Returns:
        Array shaped (3, barriers): opening fraction, outflow (m³/s), and
        upstream river inflow (m³/s).

    Raises:
        ValueError: If barrier indices are outside the inertial domain.
    """
    if np.any((barrier_indices < 0) | (barrier_indices >= router.n_inertial)):
        raise ValueError("Barrier indices must lie within the inertial domain.")
    diagnostics: TwoDArrayFloat32 = np.empty(
        (3, len(barrier_indices)), dtype=np.float32
    )
    position: int
    barrier_index: np.int32
    for position, barrier_index in enumerate(barrier_indices):
        grid_cell: int = int(router._inertial_cells[barrier_index])
        opening: np.float32 = np.float32(-1)
        if router._instream_dam_inertial[barrier_index]:
            stage_m: np.float32 = _compute_reach_stage_from_vol(
                int(barrier_index),
                np.float32(river_storage_m3[grid_cell]),
                router._geom_inbank,
                router._geom_overbank,
            )
            depth_m: np.float32 = (
                stage_m - router._bed_elevation_inertial[barrier_index]
            )
            opening = compute_instream_dam_open_fraction(
                depth_m, router._bankfull_depth_inertial[barrier_index]
            )
        # The adjacency matrix uses sorted nodes; convert back to grid order.
        upstream_nodes: ArrayInt32 = router._upstream_matrix[
            router.n_kinematic + barrier_index
        ]
        upstream_cells: ArrayInt32 = router.sorted_idxs[
            upstream_nodes[upstream_nodes >= 0]
        ]
        diagnostics[:, position] = (
            opening,
            discharge_m3_s[grid_cell],
            np.sum(discharge_m3_s[upstream_cells]),
        )
    return diagnostics
