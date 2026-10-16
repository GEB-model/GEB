"""Readable spatial snapshots and daily records around selected AMBER weirs."""

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
    from matplotlib.gridspec import GridSpec


def plot_weir_water_levels(
    router: LocalInertial,
    river_ids: ArrayInt32,
    output_folder: Path,
    simulation_date: date,
    *,
    render: bool = True,
    reset: bool = False,
) -> tuple[Path, Path] | None:
    """Record daily weir levels and update a fixed set of spatial snapshots.

    Select up to three tall weirs, skipping immediate neighbors, on the first
    call. Keep these cells throughout the simulation so geometry changes do not
    switch the plotted locations. Append daily values to one CSV and redraw the
    same snapshot PNGs monthly and on the final day. Selection is restricted to
    three weirs to keep diagnostics small for continental simulations.

    Args:
        router: Router containing current stages and channel geometry (meters).
        river_ids: River identifiers in model grid order.
        output_folder: Directory for the shared CSV, selection and PNG files.
        simulation_date: Day whose final water levels are recorded.
        render: Whether to redraw the figures after recording today's values.
        reset: Start a new history, used on the first timestep of a simulation.

    Returns:
        First snapshot PNG and shared daily CSV paths when rendered, otherwise None.
        Returns None without writing files when there are no supported weirs.

    Raises:
        ValueError: If a saved selection no longer matches the routing grid.
    """
    cells: np.ndarray = router.sorted_idxs[
        router.n_kinematic : router.n_kinematic + router.n_inertial
    ]
    heights: np.ndarray = router.weir_height_m[cells]
    weirs: np.ndarray = np.flatnonzero(heights > 0)
    if not weirs.size:
        return None
    output_folder.mkdir(parents=True, exist_ok=True)
    selection_path: Path = output_folder / "selected_weirs.csv"
    selected: list[int] = []
    if reset or not selection_path.exists():
        covered: set[int] = set()
        candidate: np.int64
        for candidate in weirs[np.argsort(-heights[weirs], kind="stable")]:
            if int(candidate) in covered:
                continue
            selected.append(int(candidate))
            covered.add(int(candidate))
            covered.add(int(router._topology.ds_inertial_k[candidate]))
            covered.update(
                router._topology.inertial_up_indices[
                    router._topology.inertial_up_offsets[
                        candidate
                    ] : router._topology.inertial_up_offsets[candidate + 1]
                ].tolist()
            )
            if len(selected) == 3:
                break
        pd.DataFrame({"grid_cell": cells[selected]}).to_csv(selection_path, index=False)
    else:
        saved_cells: np.ndarray = pd.read_csv(selection_path).grid_cell.to_numpy()
        for cell in saved_cells:
            matches: np.ndarray = np.flatnonzero(cells == cell)
            if matches.size != 1 or heights[matches[0]] <= 0:
                raise ValueError(
                    "Saved weir plot selection does not match this routing grid."
                )
            selected.append(int(matches[0]))
    upstream: np.ndarray = np.asarray(selected, dtype=np.int32)
    downstream: np.ndarray = router._topology.ds_inertial_k[upstream]
    stages: np.ndarray = router.get_water_stage()[cells]
    beds: np.ndarray = router.bed_elevation[cells]
    bankfull: np.ndarray = router.bankfull_depth[cells]
    widths: np.ndarray = router.river_width[cells]
    records: pd.DataFrame = pd.DataFrame(
        {
            "date": simulation_date.isoformat(),
            "grid_cell": cells[upstream],
            "river_id": river_ids[cells[upstream]],
            "downstream_grid_cell": cells[downstream],
            "bed_elevation_m": beds[upstream],
            "downstream_bed_elevation_m": beds[downstream],
            "bankfull_depth_m": bankfull[upstream],
            "downstream_bankfull_depth_m": bankfull[downstream],
            "bankfull_width_m": widths[upstream],
            "downstream_bankfull_width_m": widths[downstream],
            "weir_height_m": heights[upstream],
            "crest_elevation_m": beds[upstream] + heights[upstream],
            "crest_width_m": widths[upstream]
            * (heights[upstream] / bankfull[upstream])
            ** router._shape_exponent_inertial[upstream],
            "upstream_water_level_m": stages[upstream],
            "downstream_water_level_m": stages[downstream],
            "upstream_water_depth_m": np.maximum(stages[upstream] - beds[upstream], 0),
            "downstream_water_depth_m": np.maximum(
                stages[downstream] - beds[downstream], 0
            ),
            "upstream_head_m": np.maximum(
                stages[upstream] - beds[upstream] - heights[upstream], 0
            ),
            "water_level_drop_m": stages[upstream] - stages[downstream],
            "weirs_enabled": router.weirs_enabled,
        }
    )
    csv_path: Path = output_folder / "water_levels_daily.csv"
    fresh: bool = reset or not csv_path.exists()
    records.to_csv(csv_path, mode="w" if fresh else "a", header=fresh, index=False)
    if not render:
        return None
    paths: list[Path] = []
    cell: int
    reach: np.int32
    for cell, reach in zip(records.grid_cell, upstream):
        path: Path = output_folder / f"water_levels_weir_{cell}.png"
        _plot_weir_detail(
            router,
            int(reach),
            int(router._topology.ds_inertial_k[reach]),
            cells,
            int(river_ids[cell]),
            path,
            simulation_date,
        )
        paths.append(path)
    return paths[0], csv_path


def _plot_weir_detail(
    router: LocalInertial,
    upstream: int,
    downstream: int,
    cells: np.ndarray,
    river_id: int,
    output_path: Path,
    simulation_date: date,
) -> None:
    """Plot a multi-cell river profile and two dimensioned cross-sections.

    The side view follows up to five cells on each side of the weir, taking
    the widest upstream branch at junctions. Each
    cross-section uses its own bed so shallow water remains visible despite
    bed slope. Cross-sections show the channel only, not the floodplain.

    Args:
        router: Router containing current stages and channel geometry (meters).
        upstream: Focal weir's upstream index in the inertial reach order.
        downstream: Adjacent downstream index in the inertial reach order.
        cells: Grid cell identifiers in inertial reach order.
        river_id: Focal river identifier.
        output_path: Destination PNG file.
        simulation_date: Date of the end-of-day snapshot.

    Returns:
        None. Saves and closes the detail figure.
    """  # noqa: DOC202
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    indices: np.ndarray = np.asarray([upstream, downstream])
    grid_cells: np.ndarray = cells[indices]
    beds: np.ndarray = router.bed_elevation[grid_cells]
    stages: np.ndarray = router.get_water_stage()[grid_cells]
    depths: np.ndarray = np.maximum(stages - beds, 0.0)
    bankfull: np.ndarray = router.bankfull_depth[grid_cells]
    widths: np.ndarray = router.river_width[grid_cells]
    lengths: np.ndarray = router.river_length[grid_cells]
    shapes: np.ndarray = router._shape_exponent_inertial[indices]
    height: float = float(router.weir_height_m[grid_cells[0]])
    crest: float = float(beds[0]) + height
    crest_width: float = float(widths[0] * (height / bankfull[0]) ** shapes[0])
    drop: float = float(stages[0] - stages[1])
    head: float = max(float(stages[0]) - crest, 0.0)
    blue: str = "#187ca8"
    orange: str = "#cf641e"
    brown: str = "#88634c"
    ink: str = "#253849"
    muted: str = "#637481"
    style: str = "solid" if router.weirs_enabled else "dashed"
    figure: Figure = plt.figure(figsize=(13, 10), facecolor="#f6f8fa")
    figure.text(
        0.07, 0.965, "Water around the weir", fontsize=24, weight="bold", color=ink
    )
    figure.text(
        0.07,
        0.935,
        "A weir is a low barrier that water can flow over.",
        fontsize=12,
        color=muted,
    )
    label: str
    value: float
    position: int
    # Put the quantities most readers need first; exact model identifiers stay below.
    for position, (label, value) in enumerate(
        zip(
            (
                "Water depth before",
                "Water depth after",
                "Barrier height",
                "Water above barrier top",
            ),
            (float(depths[0]), float(depths[1]), height, head),
        )
    ):
        left: float = 0.07 + position * 0.225
        card: FancyBboxPatch = FancyBboxPatch(
            (left, 0.828),
            0.208,
            0.081,
            boxstyle="round,pad=0.006,rounding_size=0.009",
            transform=figure.transFigure,
            facecolor="white",
            edgecolor="#e0e6eb",
        )
        figure.add_artist(card)
        figure.text(left + 0.012, 0.884, label, fontsize=10, color=muted)
        figure.text(
            left + 0.012,
            0.846,
            f"{value:.2f} m",
            fontsize=21,
            weight="bold",
            color=blue if position < 2 else orange,
        )
    figure.legend(
        handles=[
            Line2D([], [], color=blue, lw=3, label="Water"),
            Line2D([], [], color=brown, lw=2, label="River bed"),
            Line2D(
                [],
                [],
                color=muted,
                lw=1.5,
                ls=":",
                label="Top of river banks (bankfull)",
            ),
            Line2D([], [], color=orange, lw=4, ls=style, label="Weir / barrier"),
        ],
        loc="center",
        bbox_to_anchor=(0.51, 0.797),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    grid: GridSpec = figure.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.97,
        top=0.735,
        bottom=0.15,
        hspace=0.65,
        wspace=0.23,
        height_ratios=(1.1, 1),
    )
    profile_axis: Axes = figure.add_subplot(grid[0, :])
    # The weir is at the downstream edge of the focal cell, not its center.
    profile: np.ndarray = _weir_profile(router, upstream)
    profile_cells: np.ndarray = cells[profile]
    profile_lengths: np.ndarray = router.river_length[profile_cells]
    focal_position: int = int(np.flatnonzero(profile == upstream)[0])
    edges: np.ndarray = np.concatenate(([0.0], np.cumsum(profile_lengths)))
    edges -= edges[focal_position + 1]
    profile_beds: np.ndarray = router.bed_elevation[profile_cells]
    profile_stages: np.ndarray = router.get_water_stage()[profile_cells]
    profile_bankfull: np.ndarray = router.bankfull_depth[profile_cells]
    profile_heights: np.ndarray = router.weir_height_m[profile_cells]
    pd.DataFrame(
        {
            "date": simulation_date.isoformat(),
            "focal_grid_cell": cells[upstream],
            "grid_cell": profile_cells,
            "distance_upstream_edge_m": edges[:-1],
            "distance_downstream_edge_m": edges[1:],
            "bed_elevation_m": profile_beds,
            "water_level_m": profile_stages,
            "water_depth_m": np.maximum(profile_stages - profile_beds, 0),
            "bankfull_depth_m": profile_bankfull,
            "bankfull_width_m": router.river_width[profile_cells],
            "weir_height_m": profile_heights,
        }
    ).to_csv(output_path.with_suffix(".csv"), index=False)
    # One shared elevation reference preserves the real water-surface difference.
    profile_axis.fill_between(
        edges,
        np.append(profile_beds, profile_beds[-1]),
        np.append(profile_stages, profile_stages[-1]),
        step="post",
        color=blue,
        alpha=0.2,
    )
    profile_axis.stairs(profile_beds, edges, color=brown, linewidth=2)
    profile_axis.stairs(profile_stages, edges, color=blue, linewidth=2.5)
    profile_axis.stairs(
        profile_beds + profile_bankfull, edges, color=muted, linestyle=":"
    )
    profile_axis.vlines(
        0, float(beds[0]), crest, color=orange, linewidth=6, linestyles=style
    )
    span: float = max(
        float((profile_beds + profile_bankfull).max() - profile_beds.min()),
        float(profile_stages.max() - profile_beds.min()),
        0.1,
    )
    cell: np.int32
    for position, cell in enumerate(profile_cells):
        profile_axis.axvline(edges[position], color="#b9c7d1", lw=0.7, ls=":")
        profile_axis.text(
            float((edges[position] + edges[position + 1]) / 2),
            0.025,
            f"cell {cell}",
            transform=profile_axis.get_xaxis_transform(),
            ha="center",
            fontsize=7,
            color=muted,
        )
        if profile[position] != upstream and profile_heights[position] > 0:
            profile_axis.vlines(
                edges[position + 1],
                profile_beds[position],
                profile_beds[position] + profile_heights[position],
                color=orange,
                linewidth=3,
                alpha=0.4,
                linestyles=style,
            )
    profile_axis.set(
        xlabel="Distance from the barrier (m)     → downstream",
        ylabel="Elevation (m)",
        xlim=(edges[0], edges[-1]),
        ylim=(
            float(profile_beds.min()) - 0.18 * span,
            max(
                float(profile_stages.max()),
                float((profile_beds + profile_bankfull).max()),
            )
            + 0.27 * span,
        ),
    )
    profile_axis.set_title(
        f"1  Along the river — {focal_position + 1} cells before, {len(profile) - focal_position - 1} after",
        loc="left",
        fontsize=13,
        weight="bold",
        color=ink,
    )
    profile_axis.text(
        0.21,
        0.95,
        "BEFORE · upstream",
        transform=profile_axis.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=muted,
    )
    profile_axis.text(
        0.77,
        0.95,
        "AFTER · downstream",
        transform=profile_axis.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=muted,
    )
    profile_axis.annotate(
        "Top of barrier",
        xy=(0, crest),
        xytext=(32, 22),
        textcoords="offset points",
        fontsize=10,
        color=orange,
        arrowprops={"arrowstyle": "->", "color": orange},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    if abs(drop) > 0.001:
        arrow_x: float = float(lengths[1]) * 0.18
        profile_axis.annotate(
            "",
            xy=(arrow_x, stages[0]),
            xytext=(arrow_x, stages[1]),
            arrowprops={"arrowstyle": "<->", "color": ink},
        )
        profile_axis.annotate(
            f"Level difference\n{abs(drop):.2f} m",
            xy=(arrow_x, float(stages.mean())),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=ink,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
    axis: Axes
    for position, label in enumerate(("Before the barrier", "After the barrier")):
        axis = figure.add_subplot(grid[1, position])
        half_width: float = float(widths[position]) / 2
        across: np.ndarray = np.linspace(-half_width, half_width, 301)
        channel_bed: np.ndarray = bankfull[position] * (
            np.abs(across) / half_width
        ) ** (1 / shapes[position])
        water_depth: float = float(depths[position])
        axis.fill_between(
            across,
            channel_bed,
            water_depth,
            where=channel_bed <= water_depth,
            interpolate=True,
            color=blue,
            alpha=0.25,
        )
        axis.plot(across, channel_bed, color=brown, linewidth=2)
        wet_half_width: float = half_width * min(
            water_depth / float(bankfull[position]), 1.0
        ) ** float(shapes[position])
        axis.hlines(
            water_depth, -wet_half_width, wet_half_width, color=blue, linewidth=2.5
        )
        axis.axhline(bankfull[position], color=muted, linestyle=":")
        if position == 0:
            axis.hlines(
                height,
                -crest_width / 2,
                crest_width / 2,
                color=orange,
                linewidth=4,
                linestyles=style,
            )
            axis.text(
                0,
                height,
                f"Barrier top · {crest_width:.2f} m wide",
                ha="center",
                va="top",
                fontsize=9,
                color=orange,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
        axis.set_title(
            f"{position + 2}  Looking across the river · {label.lower()}\n"
            f"At bankfull: {widths[position]:.2f} m wide × {bankfull[position]:.2f} m deep",
            loc="left",
            fontsize=11,
            color=ink,
            pad=12,
        )
        axis.set(
            xlabel="Distance across the river (m)",
            ylabel="Height above river bed (m)",
            xlim=(-half_width * 1.08, half_width * 1.08),
            ylim=(
                -0.04 * float(bankfull[position]),
                1.22 * max(float(bankfull[position]), water_depth, 0.1),
            ),
        )
    # Light framing keeps the water, bed and barrier visually dominant.
    for axis in figure.axes:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#e7edf1", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9, colors=muted)
        axis.xaxis.label.set_color(muted)
        axis.yaxis.label.set_color(muted)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#d4dde4")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    state: str = (
        "weir enabled"
        if router.weirs_enabled
        else "weir disabled; orange marks the proposed barrier"
    )
    figure.text(
        0.08,
        0.092,
        "Cross-sections show the two cells next to the main barrier. Bankfull means water reaches the river banks.",
        fontsize=10,
        color=ink,
    )
    figure.text(
        0.08,
        0.068,
        "The level difference also includes the bed slope. Compare with a no-weir run to measure the weir's effect.",
        fontsize=9,
        color=muted,
    )
    figure.text(
        0.08,
        0.045,
        f"{simulation_date:%Y-%m-%d} · river {river_id} · cell {grid_cells[0]} · {state}  |  Cell-average model values; channel only.",
        fontsize=9,
        color=muted,
    )
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _weir_profile(router: LocalInertial, upstream: int) -> np.ndarray:
    """Select up to five cells on each side of the focal weir.

    The focal upstream cell counts toward the five upstream cells. Follow the
    widest upstream tributary at junctions; stop at routing boundaries. Use
    reach indices so the profile follows connectivity rather than map distance.

    Args:
        router: Local inertial router containing channel widths and topology.
        upstream: Focal weir's upstream index in inertial reach order.

    Returns:
        Profile reach indices, ordered from upstream to downstream.
    """
    cells: np.ndarray = router.sorted_idxs[
        router.n_kinematic : router.n_kinematic + router.n_inertial
    ]
    topology: InertialTopology = router._topology
    profile: list[int] = [upstream]
    current: int = upstream
    step: int
    for step in range(4):
        tributaries: np.ndarray = topology.inertial_up_indices[
            topology.inertial_up_offsets[current] : topology.inertial_up_offsets[
                current + 1
            ]
        ]
        if not tributaries.size:
            break
        current = int(tributaries[np.argmax(router.river_width[cells[tributaries]])])
        profile.insert(0, current)
    current = upstream
    for step in range(5):
        if topology.ds_boundary_type[current] != 0:
            break
        current = int(topology.ds_inertial_k[current])
        profile.append(current)
    return np.asarray(profile, dtype=np.int32)
