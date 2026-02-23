"""Script to generate figures for paired basins and flood maps.

This script produces two SVG figures:
1. `paired_basins.svg`: Shows subbasins of interest with forcing points and hydrograph insets.
2. `paired_basins_floods.svg`: Shows flood maps for the same subbasins.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from shapely.geometry import Point

MODEL_PATH: Path = Path("tests") / "tmp" / "model"

basins_of_interest: list[int] = [23011154, 23011325, 23011646, 23011632]
outflow_basin: int = 23011154

COLOR_BY_KIND: dict[str, str] = {
    "Off": "white",
    "Outflow subbasin": "grey",
    "Subbasin of interest": "#009485",
}

ID_TO_COLOR: dict[int, str] = {
    23011325: "red",
    23011646: "purple",
    23011632: "orange",
}

FLOOD_BUFFER_DISTANCE_M: float = 500.0

subbasins = (
    gpd.read_parquet(MODEL_PATH / "input" / "geom" / "routing" / "subbasins.geoparquet")
    .to_crs(32631)
    .loc[basins_of_interest]
)


rivers = (
    gpd.read_parquet(MODEL_PATH / "input" / "geom" / "routing" / "rivers.geoparquet")
    .to_crs(32631)
    .loc[basins_of_interest]
)


def plot_basins(
    subbasins_gdf: gpd.GeoDataFrame,
    basins_of_interest_ids: list[int] | int,
    ax: Axes,
    title: str | None = None,
) -> None:
    """Plot subbasins and rivers with a fixed kind→color mapping.

    Notes:
        We assign a per-row color column to avoid geopandas errors when plotting an
        empty subset (which can happen when a category is absent in a given panel).

    Args:
        subbasins_gdf: Subbasin geometries indexed by subbasin ID.
        basins_of_interest_ids: One ID or a list of IDs to highlight.
        ax: Matplotlib axes to plot into.
        title: Optional title for the subplot.
    """
    if isinstance(basins_of_interest_ids, int):
        basins_of_interest_ids = [basins_of_interest_ids]

    subbasins_plot: gpd.GeoDataFrame = subbasins_gdf.copy()

    outflow_subbasins: list[int] = rivers.loc[basins_of_interest_ids][
        "downstream_ID"
    ].to_list()
    # Remove duplicates and avoid marking the basin(s) of interest as outflow.
    outflow_subbasins = list(set(outflow_subbasins) - set(basins_of_interest_ids))

    subbasins_plot["kind"] = "Off"
    subbasins_plot.loc[outflow_subbasins, "kind"] = "Outflow subbasin"
    subbasins_plot.loc[basins_of_interest_ids, "kind"] = "Subbasin of interest"

    subbasins_plot["color"] = subbasins_plot["kind"].map(COLOR_BY_KIND)
    off_subbasins: gpd.GeoDataFrame = subbasins_plot[subbasins_plot["kind"] == "Off"]
    visible_subbasins: gpd.GeoDataFrame = subbasins_plot[
        subbasins_plot["kind"] != "Off"
    ]
    if not off_subbasins.empty:
        off_subbasins.plot(
            color=COLOR_BY_KIND["Off"],
            ax=ax,
            lw=1,
            edgecolor="grey",
            linestyle=":",
        )
    if not visible_subbasins.empty:
        visible_subbasins.plot(
            color=visible_subbasins["color"], ax=ax, lw=1, edgecolor="white", alpha=0.6
        )
    rivers[
        rivers.index.isin(subbasins_plot[subbasins_plot["kind"] != "Off"].index)
    ].plot(color="blue", ax=ax)

    forcing_points: gpd.GeoDataFrame = rivers.loc[basins_of_interest_ids]
    forcing_points["geometry"] = forcing_points["geometry"].apply(
        lambda geom: Point(geom.coords[0])
    )
    forcing_colors: list[str] = [
        ID_TO_COLOR.get(idx, "red") for idx in forcing_points.index
    ]
    forcing_points.plot(
        ax=ax,
        color=forcing_colors,
        markersize=50,
        marker="o",
        edgecolor="white",
        lw=1,
        zorder=3,
    )

    # outflow_points = rivers.loc[outflow_subbasins]
    # outflow_points["geometry"] = outflow_points["geometry"].apply(
    #     lambda geom: Point(geom.coords[-1])
    # )

    # if len(basins_of_interest_ids) == 1:
    #     outflow_points.plot(
    #         ax=ax,
    #         color="black",
    #         markersize=50,
    #         marker="o",
    #         edgecolor="white",
    #         lw=1,
    #         zorder=3,
    #     )

    if title is not None:
        ax.set_title(title, fontsize="small", fontweight="bold", pad=-4)

    ax.set_aspect("equal")
    ax.margins(0)
    ax.axis("off")


def plot_flood_map(
    subbasins_gdf: gpd.GeoDataFrame,
    basins_of_interest_ids: list[int] | int,
    ax: Axes,
    title: str | None = None,
    flood_buffer_distance_m: float = FLOOD_BUFFER_DISTANCE_M,
) -> None:
    """Plot a flood map by buffering rivers within the basin(s) of interest.

    Notes:
        The flood extent is approximated as a fixed-width buffer around river
        geometries, clipped to the basin(s) of interest to avoid spillover.

    Args:
        subbasins_gdf: Subbasin geometries indexed by subbasin ID.
        basins_of_interest_ids: One ID or a list of IDs to highlight.
        ax: Matplotlib axes to plot into.
        title: Optional title for the subplot.
        flood_buffer_distance_m: Buffer distance around rivers (meters).
    """
    if isinstance(basins_of_interest_ids, int):
        basins_of_interest_ids = [basins_of_interest_ids]

    subbasins_plot: gpd.GeoDataFrame = subbasins_gdf.copy()
    subbasins_plot["kind"] = "Off"
    subbasins_plot.loc[basins_of_interest_ids, "kind"] = "Subbasin of interest"

    subbasins_plot["color"] = subbasins_plot["kind"].map(COLOR_BY_KIND)
    off_subbasins: gpd.GeoDataFrame = subbasins_plot[subbasins_plot["kind"] == "Off"]
    visible_subbasins: gpd.GeoDataFrame = subbasins_plot[
        subbasins_plot["kind"] != "Off"
    ]
    if not off_subbasins.empty:
        off_subbasins.plot(
            color=COLOR_BY_KIND["Off"],
            ax=ax,
            lw=1,
            edgecolor="grey",
            linestyle=":",
        )
    if not visible_subbasins.empty:
        visible_subbasins.plot(
            color=visible_subbasins["color"],
            ax=ax,
            lw=1,
            edgecolor="white",
            alpha=0.6,
        )

    basin_geometries: gpd.GeoSeries = subbasins_plot.loc[
        basins_of_interest_ids
    ].geometry
    flood_geometries: list[Any] = []
    for basin_geometry in basin_geometries:
        basin_rivers: gpd.GeoDataFrame = rivers[rivers.intersects(basin_geometry)]
        if len(subbasins_plot) == 1:
            flood_buffer_distance_m *= 3
        basin_buffer: gpd.GeoSeries = basin_rivers.buffer(flood_buffer_distance_m)
        basin_flood: gpd.GeoSeries = basin_buffer.intersection(basin_geometry)
        flood_geometries.extend(basin_flood.geometry)

    flood_union: Any = gpd.GeoSeries(flood_geometries, crs=rivers.crs).union_all()
    gpd.GeoSeries([flood_union], crs=rivers.crs).plot(
        ax=ax,
        color="#0088E2",
        alpha=1.0,
        edgecolor="none",
        zorder=2,
    )

    rivers[
        rivers.index.isin(subbasins_plot[subbasins_plot["kind"] != "Off"].index)
    ].plot(color="blue", ax=ax, zorder=3)

    if title is not None:
        ax.set_title(title, fontsize="small", fontweight="bold", pad=-4)

    ax.set_aspect("equal")
    ax.margins(0)
    ax.axis("off")


def add_hydrograph_inset(
    ax: Axes,
    color: str,
    position: tuple[float, float, float, float] = (0.7, 0.7, 0.25, 0.2),
    show_spines: bool = True,
    reduce: bool = False,
) -> Axes:
    """Add a small hydrograph inset to the given axes.

    Args:
        ax: Parent axes to add the inset to.
        color: Line color for the hydrograph.
        position: Inset position (x0, y0, width, height) in axes coordinates.
        show_spines: Whether to show axes spines and background.
        reduce: Whether to reduce the peak height visually (doubles y-limit).

    Returns:
        The created inset axes.
    """
    ax_ins: Axes = ax.inset_axes(position)
    t: np.ndarray = np.linspace(0, 1, 50)
    q: np.ndarray = np.exp(-(((t - 0.4) / 0.15) ** 2))
    q_max: float = float(q.max())

    ax_ins.plot(t, q, color=color, lw=1)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    if not show_spines:
        for spine in ax_ins.spines.values():
            spine.set_visible(False)
        ax_ins.patch.set_alpha(0.0)
    else:
        for spine in ax_ins.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)
        ax_ins.patch.set_alpha(0.8)

    y_limit: float = 1.1 * q_max
    if reduce:
        y_limit *= 2.0  # Make the curve look half-height in the same size box
    ax_ins.set_ylim(0, y_limit)
    return ax_ins


def add_split_connectors(
    fig: plt.Figure,
    main_ax: Axes,
    stacked_axes: list[Axes],
    side: str,
    arrow_mode: str,
    color: str = "black",
    line_width: float = 1.5,
) -> None:
    """Draw bracket-like connectors between a main axis and stacked axes.

    Args:
        fig: Figure to draw onto.
        main_ax: The large axis.
        stacked_axes: Smaller stacked axes.
        side: "right" if stacked axes are right of main, "left" otherwise.
        arrow_mode: "main" to draw one arrow from main to spine, "stacked" for
            arrows from the stacked axes toward the spine.
        color: Line color.
        line_width: Line width for connectors.
    """
    main_pos: Any = main_ax.get_position()
    stacked_positions: list[Any] = [ax.get_position() for ax in stacked_axes]
    centers_y: list[float] = [pos.y0 + pos.height / 2.0 for pos in stacked_positions]
    min_y: float = min(centers_y)
    max_y: float = max(centers_y)

    if side == "right":
        stacked_edge: float = min(pos.x0 for pos in stacked_positions)
        vertical_x: float = main_pos.x1 + (stacked_edge - main_pos.x1) * 0.5
        main_anchor_x: float = main_pos.x1 + 0.005
        small_anchor_x: float = stacked_edge - 0.005
        main_y: float = main_pos.y0 + main_pos.height / 2.0
    else:
        stacked_edge: float = max(pos.x1 for pos in stacked_positions)
        vertical_x: float = stacked_edge + (main_pos.x0 - stacked_edge) * 0.5
        main_anchor_x: float = main_pos.x0 - 0.005
        small_anchor_x: float = stacked_edge + 0.005
        main_y: float = main_pos.y0 + main_pos.height / 2.0

    fig.add_artist(
        Line2D([vertical_x, vertical_x], [min_y, max_y], color=color, lw=line_width)
    )

    if arrow_mode == "main":
        fig.add_artist(
            FancyArrowPatch(
                (vertical_x, main_y),
                (main_anchor_x, main_y),
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                lw=line_width,
            )
        )
    else:
        fig.add_artist(
            Line2D(
                [main_anchor_x, vertical_x],
                [main_y, main_y],
                color=color,
                lw=line_width,
            )
        )
    for center_y in centers_y:
        if arrow_mode == "stacked":
            fig.add_artist(
                FancyArrowPatch(
                    (vertical_x, center_y),
                    (small_anchor_x, center_y),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=color,
                    lw=line_width,
                )
            )
        else:
            fig.add_artist(
                Line2D(
                    [vertical_x, small_anchor_x],
                    [center_y, center_y],
                    color=color,
                    lw=line_width,
                )
            )


fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(
    3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1], hspace=0.02, wspace=0.02
)
ax_main = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, 1])

# map subbasin of interest to blue and outflow subbasin to grey
plot_basins(
    subbasins,
    [23011325, 23011646, 23011632],
    ax_main,
    title=None,
)

# add hydrograph insets for each forcing point
forcing_ids = [23011325, 23011646, 23011632]
forcing_colors = ["red", "purple", "orange"]
offsets = [
    (-0.25, -0.05),
    (-0.07, 0.03),
    (0.0, -0.15),
]  # relative offsets in axes units

for i, (idx, color) in enumerate(zip(forcing_ids, forcing_colors)):
    point_data = rivers.loc[idx].geometry.coords[0]
    axis_to_data = ax_main.transData + ax_main.transAxes.inverted()
    point_axes = axis_to_data.transform(point_data)

    # create inset axes near the point in main axis
    add_hydrograph_inset(
        ax_main,
        color,
        (point_axes[0] + offsets[i][0], point_axes[1] + offsets[i][1], 0.15, 0.1),
        reduce=(color == "purple" or color == "orange"),
    )

plot_basins(subbasins, 23011646, ax1, title=None)
add_hydrograph_inset(
    ax1, "purple", (0.55, 0.75, 0.3, 0.2), show_spines=False, reduce=True
)

plot_basins(subbasins, 23011632, ax2, title=None)
add_hydrograph_inset(
    ax2, "orange", (0.55, 0.75, 0.3, 0.2), show_spines=False, reduce=True
)

plot_basins(subbasins, 23011325, ax3, title=None)
add_hydrograph_inset(ax3, "red", (0.55, 0.75, 0.3, 0.2), show_spines=False)

add_split_connectors(fig, ax_main, [ax1, ax2, ax3], side="right", arrow_mode="stacked")

# move legend below all axes
legend_handles = [
    Patch(facecolor=color, edgecolor="white", label=kind)
    for kind, color in COLOR_BY_KIND.items()
    if kind != "Off"
]

# Add river line to legend
legend_handles.append(
    plt.Line2D(
        [0],
        [0],
        color="blue",
        lw=2,
        label="River",
    )
)

# add forcing point markers to legend
forcing_legend_dots = tuple(
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markeredgecolor="white",
        markersize=8,
        markeredgewidth=1,
    )
    for color in ["red", "purple", "orange"]
)
legend_handles.append(forcing_legend_dots)
legend_labels = [
    h.get_label() if hasattr(h, "get_label") else "" for h in legend_handles
]
legend_labels[-1] = "Forcing points"

fig.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="lower center",
    ncol=len(legend_handles),
    frameon=False,
    bbox_to_anchor=(0.5, 0.04),
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
)

plt.tight_layout(pad=0, rect=(0.0, 0.05, 1.0, 1.0))
plt.savefig(Path(__file__).parent.parent / "paired_basins.svg", bbox_inches="tight")

flood_fig = plt.figure(figsize=(10, 6))
flood_gs = gridspec.GridSpec(
    3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1], hspace=0.02, wspace=0.02
)
flood_ax_left_1 = flood_fig.add_subplot(flood_gs[0, 0])
flood_ax_left_2 = flood_fig.add_subplot(flood_gs[1, 0])
flood_ax_left_3 = flood_fig.add_subplot(flood_gs[2, 0])
flood_ax_right = flood_fig.add_subplot(flood_gs[:, 1])

plot_flood_map(subbasins, 23011646, flood_ax_left_1, title=None)
plot_flood_map(subbasins, 23011632, flood_ax_left_2, title=None)
plot_flood_map(subbasins, 23011325, flood_ax_left_3, title=None)
plot_flood_map(
    subbasins,
    [23011325, 23011646, 23011632],
    flood_ax_right,
    title=None,
)

add_split_connectors(
    flood_fig,
    flood_ax_right,
    [flood_ax_left_1, flood_ax_left_2, flood_ax_left_3],
    side="left",
    arrow_mode="main",
)

flood_legend_handles = [
    Patch(
        facecolor=COLOR_BY_KIND["Subbasin of interest"],
        edgecolor="white",
        label="Subbasin of interest",
    ),
    Patch(facecolor="#0088E2", edgecolor="none", alpha=1.0, label="Flooded area"),
    plt.Line2D(
        [0],
        [0],
        color="blue",
        lw=2,
        label="River",
    ),
]

flood_fig.legend(
    handles=flood_legend_handles,
    loc="lower center",
    ncol=len(flood_legend_handles),
    frameon=False,
    bbox_to_anchor=(0.5, 0.04),
)

plt.tight_layout(pad=0, rect=(0.0, 0.05, 1.0, 1.0))
plt.savefig(
    Path(__file__).parent.parent / "paired_basins_floods.svg", bbox_inches="tight"
)
