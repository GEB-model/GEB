"""Water-level profiles around AMBER weirs for inspecting real model runs."""

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from geb.geb_types import ArrayInt32

from .local_inertial import LocalInertial
from .topology import InertialTopology

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_weir_water_levels(
    router: LocalInertial,
    river_ids: ArrayInt32,
    output_folder: Path,
    simulation_date: date,
) -> tuple[Path, Path] | None:
    """Save end-of-day weir diagnostics with elevation and water-depth views.

    Export every represented weir to CSV. Plot up to three local profiles, each
    with an elevation view and a depth-above-bed view. Start with the tallest
    crests, skipping focal weirs already covered by an earlier profile. Follow
    the widest upstream branch and up to two reaches on either side.

    Bed and water elevations use the same cell edges; interpolating the bed
    between cell centers can misleadingly draw water below the bed. In the
    depth view, each cell's own bed is subtracted to reveal shallow water;
    that view does not show the longitudinal water-surface gradient.

    Disabled crests remain visible for comparison runs. A separate profiles CSV
    preserves the selected reaches' levels and distances for detailed replotting.

    Args:
        router: Local inertial router with current water elevations (meters).
        river_ids: River identifiers in model grid order.
        output_folder: Directory for dated PNG and CSV files.
        simulation_date: Model day whose final water levels are shown.

    Returns:
        PNG and CSV paths, or None when no supported weirs were built.
    """
    cells: np.ndarray = router.sorted_idxs[
        router.n_kinematic : router.n_kinematic + router.n_inertial
    ]
    heights: np.ndarray = router.weir_height_m[cells]
    weirs: np.ndarray = np.flatnonzero(heights > 0)
    if not weirs.size:
        return None

    # Import plotting only when needed; most model runs have no AMBER weirs.
    import matplotlib.pyplot as plt

    topology: InertialTopology = router._topology
    downstream: np.ndarray = topology.ds_inertial_k[weirs]
    stages: np.ndarray = router.get_water_stage()[cells]
    beds: np.ndarray = router.bed_elevation[cells]
    widths: np.ndarray = router.river_width[cells]
    lengths: np.ndarray = router.river_length[cells]
    bankfull: np.ndarray = router.bankfull_depth[cells]
    crests: np.ndarray = beds[weirs] + heights[weirs]
    output_folder.mkdir(parents=True, exist_ok=True)
    basename: str = f"water_levels_{simulation_date:%Y-%m-%d}"
    csv_path: Path = output_folder / f"{basename}.csv"
    png_path: Path = output_folder / f"{basename}.png"
    pd.DataFrame(
        {
            "grid_cell": cells[weirs],
            "river_id": river_ids[cells[weirs]],
            "downstream_grid_cell": cells[downstream],
            "bed_elevation_m": beds[weirs],
            "bankfull_depth_m": bankfull[weirs],
            "weir_height_m": heights[weirs],
            "crest_elevation_m": crests,
            "crest_width_m": widths[weirs],
            "upstream_water_level_m": stages[weirs],
            "downstream_water_level_m": stages[downstream],
            "upstream_head_m": np.maximum(stages[weirs] - crests, 0.0),
            "water_level_drop_m": stages[weirs] - stages[downstream],
            "weirs_enabled": router.weirs_enabled,
        }
    ).to_csv(csv_path, index=False)

    selected: list[int] = []
    profiles: list[np.ndarray] = []
    covered: set[int] = set()
    candidate: np.int64
    for candidate in weirs[np.argsort(-heights[weirs], kind="stable")]:
        if int(candidate) in covered:
            continue
        profile: list[int] = [int(candidate)]
        current: int = int(candidate)
        step: int
        for step in range(2):
            tributaries: np.ndarray = topology.inertial_up_indices[
                topology.inertial_up_offsets[current] : topology.inertial_up_offsets[
                    current + 1
                ]
            ]
            if not tributaries.size:
                break
            current = int(tributaries[np.argmax(widths[tributaries])])
            profile.insert(0, current)
        current = int(candidate)
        for step in range(2):
            if topology.ds_boundary_type[current] != 0:
                break
            current = int(topology.ds_inertial_k[current])
            profile.append(current)
        selected.append(int(candidate))
        profiles.append(np.asarray(profile, dtype=np.int32))
        covered.update(profile)
        if len(selected) == 3:
            break

    figure: Figure
    axes: np.ndarray
    figure, axes = plt.subplots(
        len(selected), 2, figsize=(15, 4.2 * len(selected)), squeeze=False
    )
    profile_tables: list[pd.DataFrame] = []
    row: int
    weir: int
    indices: np.ndarray
    for row, (weir, indices) in enumerate(zip(selected, profiles)):
        distance: np.ndarray = np.cumsum(lengths[indices]) - lengths[indices] / 2
        focal_position: int = int(np.flatnonzero(indices == weir)[0])
        distance -= distance[focal_position] + lengths[weir] / 2
        edges: np.ndarray = np.concatenate(
            (distance - lengths[indices] / 2, [distance[-1] + lengths[indices[-1]] / 2])
        )
        profile_beds: np.ndarray = beds[indices]
        depth_m: np.ndarray = np.maximum(stages[indices] - profile_beds, 0.0)
        profile_tables.append(
            pd.DataFrame(
                {
                    "focal_grid_cell": cells[weir],
                    "grid_cell": cells[indices],
                    "river_id": river_ids[cells[indices]],
                    "distance_from_weir_m": distance,
                    "reach_length_m": lengths[indices],
                    "bed_elevation_m": profile_beds,
                    "water_level_m": stages[indices],
                    "water_depth_m": depth_m,
                    "bankfull_depth_m": bankfull[indices],
                    "weir_height_m": heights[indices],
                }
            )
        )
        elevation_axis: Axes = axes[row, 0]
        depth_axis: Axes = axes[row, 1]
        axis: Axes
        reference: np.ndarray
        for axis, reference in zip(
            (elevation_axis, depth_axis), (np.zeros_like(profile_beds), profile_beds)
        ):
            bed_values: np.ndarray = profile_beds - reference
            water_values: np.ndarray = stages[indices] - reference
            axis.fill_between(
                edges,
                np.append(bed_values, bed_values[-1]),
                np.append(water_values, water_values[-1]),
                step="post",
                color="tab:blue",
                alpha=0.25,
            )
            axis.stairs(
                bed_values,
                edges,
                baseline=None,
                color="saddlebrown",
                linewidth=2,
                label="River bed",
            )
            axis.stairs(
                water_values,
                edges,
                baseline=None,
                color="tab:blue",
                linewidth=2,
                label="Water",
            )
            axis.stairs(
                profile_beds + bankfull[indices] - reference,
                edges,
                baseline=None,
                color="gray",
                linestyle=":",
                label="Bankfull",
            )
            position: int
            cell: np.int32
            for position, cell in enumerate(indices):
                if heights[cell] <= 0:
                    continue
                is_focal: bool = cell == weir
                location_m: float = float(distance[position] + lengths[cell] / 2)
                axis.vlines(
                    location_m,
                    bed_values[position],
                    bed_values[position] + heights[cell],
                    color="tab:red" if is_focal else "0.4",
                    linewidth=4 if is_focal else 2,
                    linestyles="solid" if router.weirs_enabled else "dashed",
                    label="Focal weir" if is_focal else "Other weir",
                )
            axis.axvline(0, color="tab:red", alpha=0.3, linewidth=1, linestyle="--")
            axis.set(
                xlabel="Distance from focal weir (m; downstream →)",
                xlim=(edges[0], edges[-1]),
            )
            axis.grid(alpha=0.2)
        elevation_axis.set(
            ylabel="Elevation (m)",
            title=f"River {river_ids[cells[weir]]}, cell {cells[weir]} — river profile",
        )
        depth_axis.set(
            ylabel="Depth above local bed (m)",
            title="Water depth — bed slope removed",
            ylim=(
                0,
                1.3
                * max(
                    float(depth_m.max()),
                    float(bankfull[indices].max()),
                    float(heights[indices].max()),
                    0.1,
                ),
            ),
        )
        head_m: float = max(float(stages[weir] - beds[weir] - heights[weir]), 0.0)
        depth_axis.text(
            0.02,
            0.97,
            f"At focal weir: depth {depth_m[focal_position]:.2f} m  |  crest {heights[weir]:.2f} m\n"
            f"Head above crest {head_m:.2f} m",
            transform=depth_axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    pd.concat(profile_tables, ignore_index=True).to_csv(
        output_folder / f"{basename}_profiles.csv", index=False
    )
    # A shared legend leaves room for the shallow water and crest annotations.
    handles, labels = elevation_axis.get_legend_handles_labels()
    legend_entries: dict = dict(zip(labels, handles))
    figure.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=5,
    )
    state: str = "enabled" if router.weirs_enabled else "disabled (comparison run)"
    figure.suptitle(
        f"{simulation_date:%Y-%m-%d}: end-of-day water levels; weirs {state}\n"
        f"{len(selected)} local views of {weirs.size} weirs — depth uses each cell's own bed",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return png_path, csv_path
