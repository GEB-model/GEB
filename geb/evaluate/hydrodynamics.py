"""Module implementing hydrodynamics evaluation functions for the GEB model."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import contextily as ctx
import geopandas as gpd
import matplotlib.animation as manimation
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
from matplotlib.lines import Line2D
from pyproj import Transformer
from rasterio.features import geometry_mask
from rioxarray.exceptions import NoDataInBounds

from geb.hazards.floods.workflows.utils import FLOOD_DEPTH_COLORS
from geb.workflows.io import read_geom, read_zarr

if TYPE_CHECKING:
    from geb.evaluate import Evaluate
    from geb.model import GEBModel


def calculate_critical_success_index(
    simulations: xr.DataArray, observations: xr.DataArray
) -> float:
    """Calculate the critical success index (CSI) metric.

    Input can be three values:
        -1: invalid
        0: no flood
        1: flood

    Args:
        simulations: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        Critical success index as a float.
    """
    assert set(np.unique(simulations.values)).issubset({-1, 0, 1}), (
        "Simulations array contains values other than -1, 0, or 1."
    )
    assert set(np.unique(observations.values)).issubset({-1, 0, 1}), (
        "Observations array contains values other than -1, 0, or 1."
    )
    hit = np.sum(((simulations == 1) & (observations == 1)).values)
    false_alarm = np.sum(((simulations == 1) & (observations == 0)).values)
    miss = np.sum(((simulations == 0) & (observations == 1)).values)
    csi = hit / (hit + false_alarm + miss)
    return float(csi)


def calculate_false_alarm_ratio(
    simulations: xr.DataArray, observations: xr.DataArray
) -> float:
    """Calculate the false alarm ratio metric.

    Input can be three values:
        -1: invalid
        0: no flood
        1: flood

    Args:
        simulations: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        False alarm ratio as a float.
    """
    assert set(np.unique(simulations.values)).issubset({-1, 0, 1}), (
        "Simulations array contains values other than -1, 0, or 1."
    )
    assert set(np.unique(observations.values)).issubset({-1, 0, 1}), (
        "Observations array contains values other than -1, 0, or 1."
    )

    false_alarm = np.sum(((simulations == 1) & (observations == 0)).values)
    hit = np.sum(((simulations == 1) & (observations == 1)).values)
    if false_alarm + hit == 0:
        return 0.0  # Avoid division by zero; if no hits or false alarms, FAR is defined as 0
    else:
        false_alarm_ratio = false_alarm / (false_alarm + hit)
        return float(false_alarm_ratio)


def calculate_hit_rate(simulations: xr.DataArray, observations: xr.DataArray) -> float:
    """Calculate the hit rate metric.

    Input can be three values:
        -1: invalid
        0: no flood
        1: flood

    Args:
        simulations: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        Hit rate as a float.
    """
    assert set(np.unique(simulations.values)).issubset({-1, 0, 1}), (
        "Simulations array contains values other than -1, 0, or 1."
    )
    assert set(np.unique(observations.values)).issubset({-1, 0, 1}), (
        "Observations array contains values other than -1, 0, or 1."
    )
    miss = np.sum(((simulations == 0) & (observations == 1)).values)
    hit = np.sum(((simulations == 1) & (observations == 1)).values)
    hit_rate = hit / (hit + miss)
    return float(hit_rate)


def parse_flood_forecast_initialisation(
    filename: str,
) -> tuple[str | None, str | None, str, str, str]:
    """Parse flood map filename to extract components.

    Expected format: YYYYMMDDTHHMMSS - MEMBER - EVENT_START - EVENT_END.zarr

    Args:
        filename: Name of the flood map file.

    Returns:
        Tuple containing (forecast_init, member, event_start, event_end, event_name)

    Raises:
        ValueError: If the filename does not match the expected format.

    """
    # Remove .zarr extension
    name_without_ext = filename.replace(".zarr", "")

    # Split by ' - ' to get components
    parts = name_without_ext.split(" - ")

    if len(parts) >= 4:
        # Handle case with forecasts included
        forecast_init = parts[0]  # First 17 characters: YYYYMMDDTHHMMSS
        member = parts[1]  # Member number
        event_start = parts[2]  # Event start time
        event_end = parts[3]  # Event end time

        # Create event name from start and end times
        event_name = f"{event_start} - {event_end}"

    elif len(parts) == 2:
        # Handle case with no forecasts included
        forecast_init = None
        member = None
        event_start = parts[0]  # Event start time
        event_end = parts[1]  # Event end time

        # Create event name from start and end times
        event_name = f"{event_start} - {event_end}"

    else:
        raise ValueError(
            f"Filename '{filename}' does not match expected flood map format."
        )

    return forecast_init, member, event_start, event_end, event_name


def calculate_performance_metrics(
    observation: xr.DataArray,
    simulated: xr.DataArray,
    output_folder: Path,
    run_name: str,
    elevation_data: xr.DataArray,
    minimum_flood_depth: float,
    visualization_type: Literal["Hillshade", "OSM"] = "Hillshade",
    name: str = "simulation",
    dpi: int = 300,
) -> dict[str, float | int] | None:
    """Calculate performance metrics for flood maps against observations and generate visualizations.

    Args:
        observation: Observed flood extent data as an xarray DataArray.
        simulated: Model-generated flood map data as an xarray DataArray.
        output_folder: Path to the folder where results will be saved.
        run_name: Name of the simulation run.
        minimum_flood_depth: Minimum water depth (in meters) to consider as flooded.
        elevation_data: Elevation data as an xarray DataArray for hillshade visualization.
        visualization_type: Type of visualization for plotting (default is "Hillshade").
        name: Name of the simulation for filename purposes.
        dpi: Resolution of the output figure in dots per inch (default is 300).

    Returns:
        Dictionary containing performance metrics:
            - hit_rate_pct: Percentage of correctly predicted flooded areas.
            - false_alarm_ratio_pct: Percentage of falsely predicted flooded areas.
            - csi_pct: Percentage of accurately predicted flood areas (CSI).
            - flooded_area_km2: Total flooded area in square kilometers.

    Raises:
        ValueError: If visualization_type is unknown.
    """
    simulated = simulated.rio.reproject_match(observation)

    rivers: gpd.GeoDataFrame = read_geom(
        Path("simulation_root") / run_name / "SFINCS" / "group_0" / "rivers.geoparquet"
    )

    subbasins: gpd.GeoDataFrame = read_geom(
        Path("simulation_root")
        / run_name
        / "SFINCS"
        / "group_0"
        / "subbasins.geoparquet"
    ).to_crs(observation.rio.crs)
    subbasins = subbasins[
        ~subbasins.index.isin(rivers[rivers["is_downstream_outflow"]].index)
    ]
    subbasins = gpd.GeoDataFrame(geometry=[subbasins.union_all()], crs=subbasins.crs)

    rivers = rivers.to_crs(3857)
    rivers = rivers[
        ~rivers["width"].isnull()
    ]  # Filter out rivers with non-positive width to avoid invalid geometries
    rivers["geometry"] = rivers.buffer(rivers["width"] / 2)

    river_mask = ~geometry_mask(
        rivers.to_crs(simulated.rio.crs).geometry,
        out_shape=simulated.rio.shape,
        transform=simulated.rio.transform(),
        all_touched=True,
        invert=False,
    )

    # remove river areas from both simulated and observed flood maps to focus on overland flooding performance
    simulated = simulated.where(~river_mask).fillna(0)
    observation: xr.DataArray = observation.where(~river_mask).fillna(0)

    # Clip out region from observations
    try:
        observation = observation.rio.clip(
            subbasins.geometry.values, subbasins.crs
        ).compute()
    except NoDataInBounds:
        print(
            f"No observation data found within the subbasin bounds for event {name}. Skipping this event."
        )
        return None
    try:
        simulated: xr.DataArray = simulated.rio.clip(
            subbasins.geometry.values, subbasins.crs
        ).compute()
    except NoDataInBounds:
        print(
            f"No simulated data found within the subbasin bounds for event {name}. Skipping this event."
        )

    # Create masks for remapping and plotting
    # 'encoding': {'invalid': -2, 'cloud': -1, 'land': 0, 'flood': 2, 'permanent_water': 1}
    # Map to -1, 0, 1 for consistent use in downstream metric functions
    flooded_mask_obs = (observation == 1) | (observation == 2)
    land_mask_obs = observation == 0

    # Distinguish masks for plotting
    cloud_mask = (observation == -1).squeeze()
    invalid_mask = (observation == -2).squeeze()

    # Mask areas with no observation data (e.g. from clipping) as also invalid
    invalid_mask = xr.where(observation.isnull().squeeze(), 1, invalid_mask)
    invalid_mask = invalid_mask.rio.write_crs(observation.rio.crs)

    observation_crs = observation.rio.crs
    observation = xr.where(flooded_mask_obs, 1, xr.where(land_mask_obs, 0, -1))
    observation = observation.rio.write_crs(observation_crs)

    # Convert simulation to binary
    simulated: xr.DataArray = simulated > minimum_flood_depth

    # Remap simulation to match observation valid mask (-1 means no data)
    # This ensures areas with no observation data are ignored in metrics
    simulated = xr.where(observation == -1, -1, simulated)
    simulated = simulated.rio.write_crs(observation_crs)

    hit_rate_pct: float = calculate_hit_rate(simulated, observation) * 100
    false_alarm_ratio_pct: float = (
        calculate_false_alarm_ratio(simulated, observation) * 100
    )
    csi_pct: float = calculate_critical_success_index(simulated, observation) * 100

    flooded_pixels = float((simulated == 1).sum())

    # Calculate resolution in meters from coordinate spacing
    x_res = float(np.abs(simulated.x[1] - simulated.x[0]))
    pixel_size = x_res  # meters
    flooded_area_km2: float = flooded_pixels * (pixel_size * pixel_size) / 1_000_000

    # Catchment borders
    target_crs = observation.rio.crs

    # Fix extent for imshow: (left, right, bottom, top)
    bounds = observation.rio.bounds()
    flood_extent = (bounds[0], bounds[2], bounds[1], bounds[3])

    if (simulated == 1).any():
        misses = (observation == 1) & (simulated == 0)
        simulation_masked = simulated.where(simulated == 1)
        hits = simulation_masked.where(observation == 1)
        misses_masked = misses.where(misses == 1)

        if visualization_type == "OSM":
            margin = 3000
            fig, ax = plt.subplots(figsize=(10, 10))

            # Set plot extent including margin
            ax.set_xlim(flood_extent[0] - margin, flood_extent[1] + margin)
            ax.set_ylim(flood_extent[2] - margin, flood_extent[3] + margin)

            # Pad invalid mask to cover the entire plot extent to avoid gray square artifacts
            # from projection margins.
            invalid_mask_plotting = invalid_mask.rio.pad_box(
                minx=flood_extent[0] - margin,
                miny=flood_extent[2] - margin,
                maxx=flood_extent[1] + margin,
                maxy=flood_extent[3] + margin,
                constant_values=1,
            )

            # Clouds: white, opaque
            cloud_cmap = mcolors.ListedColormap(["white"])
            cloud_mask.where(cloud_mask == 1).plot(
                ax=ax, cmap=cloud_cmap, add_colorbar=False, add_labels=False, zorder=1.1
            )

            # Invalid: grey, half transparent
            invalid_cmap = mcolors.ListedColormap(["grey"])
            invalid_mask_plotting.where(invalid_mask_plotting == 1).plot(
                ax=ax,
                cmap=invalid_cmap,
                add_colorbar=False,
                add_labels=False,
                alpha=0.5,
                zorder=1.2,
            )

            ax.imshow(
                simulation_masked,  # False alarms (where obs is 0)
                extent=flood_extent,
                # origin="lower",
                cmap="Wistia",
                vmin=0,
                vmax=1,
                interpolation="none",
                zorder=2,
            )
            ax.imshow(
                misses_masked,  # Misses
                extent=flood_extent,
                # origin="lower",
                cmap="autumn_r",
                vmin=0,
                vmax=1,
                interpolation="none",
                zorder=3,
            )

            ax.imshow(
                hits,
                extent=flood_extent,
                # origin="lower",
                cmap="brg",
                vmin=0,
                vmax=1,
                interpolation="none",
                zorder=4,
            )

            subbasins.boundary.plot(
                ax=ax,
                color="black",
                linestyle="--",
                linewidth=1.5,
                zorder=5,
                alpha=0.5,
            )
            # Add a base map
            ctx.add_basemap(
                ax,
                crs=target_crs,
                source="OpenStreetMap.Mapnik",
                zoom=12,
                zorder=0,
                alpha=0.9,
            )

            ax.set_title("Validation of the Predicted Flood Areas", fontsize=14)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal", adjustable="box")

            green_patch = mpatches.Patch(color="#94f944", label="Hits")
            orange_patch = mpatches.Patch(color="orange", label="False Alarms")
            red_patch = mpatches.Patch(color="red", label="Misses")

            catchment_patch = Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Catchment Border",
            )
            # Add no observation data patch
            cloud_patch = mpatches.Patch(color="white", label="Clouds")
            invalid_patch = mpatches.Patch(color="grey", alpha=0.5, label="No data")

            handles = [
                green_patch,
                orange_patch,
                red_patch,
                cloud_patch,
                invalid_patch,
                catchment_patch,
            ]
            # Assuming you have a list of corresponding labels, e.g.:
            labels = ["Green", "Orange", "Red", "Cloud", "Invalid", "Catchment"]

            import matplotlib.text as mtext

            # Add a separator and the metrics using empty patches
            text_marker = mtext.Text()
            handles.extend([text_marker] * 4)  # 1 for spacer, 3 for metrics
            labels.extend(
                [
                    "",  # Empty line separator
                    f"HR  = {hit_rate_pct:.2f} %",
                    f"FAR = {false_alarm_ratio_pct:.2f} %",
                    f"CSI = {csi_pct:.2f} %",
                ]
            )
            from matplotlib.legend_handler import HandlerEmpty

            # Create a single legend
            legend = ax.legend(
                handles=handles,
                labels=labels,
                facecolor=(1.0, 1.0, 1.0, 0.8),
                edgecolor="grey",
                prop={"family": "monospace", "size": 10},
                handler_map={
                    text_marker: HandlerEmpty()
                },  # Strips handle width for text rows
            )

            ax.annotate(
                target_crs.to_string(),
                xy=(0.99, 0.02),  # Bottom right corner in axes coordinates
                xycoords="axes fraction",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    edgecolor="grey",
                    boxstyle="round,pad=0.2",
                    alpha=0.8,
                ),
                verticalalignment="bottom",
                horizontalalignment="right",
                zorder=7,
            )

            fig.savefig(
                output_folder / f"{name}_validation_floodextent_plot.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            print(
                f"Figure with {visualization_type} saved as: {output_folder / f'{name}_validation_floodextent_plot.png'}"
            )

        elif visualization_type == "Hillshade":
            green_cmap = mcolors.ListedColormap(["green"])  # Hits
            orange_cmap = mcolors.ListedColormap(["orange"])  # False alarms
            red_cmap = mcolors.ListedColormap(["red"])  # Misses

            fig, ax = plt.subplots(figsize=(10, 10))

            # Pad invalid mask to cover the entire plot extent
            invalid_mask_plotting = invalid_mask.rio.pad_box(
                minx=flood_extent[0],
                miny=flood_extent[2],
                maxx=flood_extent[1],
                maxy=flood_extent[3],
                constant_values=1,
            )

            # Save results to file and plot the results
            elevation_data = elevation_data.rio.reproject_match(observation)

            elevation_array = (
                elevation_data.squeeze().astype("float32").compute().values
            )

            hillshade = LightSource(azdeg=315, altdeg=45).hillshade(
                elevation_array, vert_exag=1, dx=1, dy=1
            )

            ax.imshow(
                hillshade,
                cmap="gray",
                extent=flood_extent,
            )

            subbasins.boundary.plot(
                ax=ax, edgecolor="black", linewidth=2, label="Region Boundary"
            )

            # Clouds: white, opaque
            cloud_cmap = mcolors.ListedColormap(["white"])
            cloud_mask.where(cloud_mask == 1).plot(
                ax=ax, cmap=cloud_cmap, add_colorbar=False, add_labels=False, zorder=1.1
            )

            # Invalid: grey, half transparent
            invalid_cmap = mcolors.ListedColormap(["grey"])
            invalid_mask_plotting.where(invalid_mask_plotting == 1).plot(
                ax=ax,
                cmap=invalid_cmap,
                add_colorbar=False,
                add_labels=False,
                alpha=0.5,
                zorder=1.2,
            )
            ax.imshow(
                simulation_masked,
                extent=flood_extent,
                cmap=orange_cmap,
                interpolation="none",
                zorder=2,
            )
            ax.imshow(
                hits,
                extent=flood_extent,
                cmap=green_cmap,
                interpolation="none",
                zorder=3,
            )
            ax.imshow(
                misses_masked,
                extent=flood_extent,
                cmap=red_cmap,
                interpolation="none",
                zorder=4,
            )

            ax.set_aspect("equal")
            ax.axis("off")
            handles = [
                mpatches.Patch(color=green_cmap(0.5)),
                mpatches.Patch(color=red_cmap(0.5)),
                mpatches.Patch(color=orange_cmap(0.5)),
                mpatches.Patch(color="white"),
                mpatches.Patch(color="grey", alpha=0.5),
                mlines.Line2D([], [], color="black", linewidth=2),
            ]

            labels: list[str] = [
                "Hits",
                "Misses",
                "False alarms",
                "Clouds",
                "No data",
                "Region",
            ]

            plt.legend(handles=handles, labels=labels, loc="upper right", fontsize=16)

            plt.savefig(
                output_folder / f"{name}_validation_floodextent_plot.png", dpi=dpi
            )
            print(
                f"Figure with {visualization_type} saved as: {output_folder / f'{name}_validation_floodextent_plot.png'}"
            )

        else:
            raise ValueError(
                f"Unknown visualization type: {visualization_type}, choose either 'OSM' or 'Hillshade'."
            )

        results: dict[str, int | float] = {
            "hit_rate_pct": hit_rate_pct,
            "false_alarm_ratio_pct": false_alarm_ratio_pct,
            "csi_pct": csi_pct,
            "flooded_area_km2": flooded_area_km2,
        }

        with open(output_folder / f"{name}_performance_metrics.json", "w") as f:
            json.dump(
                results,
                f,
                indent=4,
            )

        return results


def create_forecast_performance_plots(
    performance_df: pd.DataFrame, event_name: str, output_folder: Path
) -> None:
    """Create performance metric plots showing spread and mean across forecast initializations.

    Generates line plots showing how performance metrics vary across different
    forecast initialization times for a given flood event. Shows both the spread
    of the ensemble (min-max range) with transparent fill and the ensemble mean as a black line.

    Notes:
        The spread is calculated using the minimum and maximum values across ensemble
        members for each forecast initialization. If only one member exists for a
        forecast initialization, no spread is shown for that time point.

    Args:
        performance_df: DataFrame containing performance metrics with forecast
            initialization times and ensemble members. Must include columns:
            'forecast_init', 'hit_rate_pct', 'false_alarm_ratio_pct', 'csi_pct', 'flooded_area_km2'.
        event_name: Name of the flood event being analyzed.
        output_folder: Directory to save the performance plots.

    Raises:
        ValueError: If required columns are missing from performance_df.
    """
    # Validate required columns
    required_columns = [
        "forecast_init",
        "hit_rate_pct",
        "false_alarm_ratio_pct",
        "csi_pct",
        "flooded_area_km2",
    ]
    missing_columns = [
        col for col in required_columns if col not in performance_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in performance_df: {missing_columns}"
        )

    # Group by forecast initialization and calculate statistics
    grouped_stats = (
        performance_df.groupby("forecast_init")[
            ["hit_rate_pct", "false_alarm_ratio_pct", "csi_pct", "flooded_area_km2"]
        ]
        .agg(["mean", "min", "max", "std", "count"])
        .reset_index()
    )

    # Convert forecast_init to datetime for better plotting
    grouped_stats["forecast_init_dt"] = pd.to_datetime(
        grouped_stats["forecast_init"], format="%Y%m%dT%H%M%S"
    )

    # Sort by datetime for proper line plotting
    grouped_stats = grouped_stats.sort_values("forecast_init_dt")

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Forecast Performance Metrics with Ensemble Spread - {event_name}",
        fontsize=16,
        fontweight="bold",
    )

    # Define colors for spread (transparent) and metric-specific colors
    spread_colors = {
        "hit_rate_pct": "#2E86AB",
        "false_alarm_ratio_pct": "#A23B72",
        "csi_pct": "#F18F01",
        "flooded_area_km2": "#636EFA",
    }

    # Hit Rate
    ax = axes[0, 0]
    metric = "hit_rate_pct"

    # Plot spread (min-max range) with transparent fill
    ax.fill_between(
        grouped_stats["forecast_init_dt"],
        grouped_stats[(metric, "min")],
        grouped_stats[(metric, "max")],
        color=spread_colors[metric],
        alpha=0.3,
        label="Ensemble member range (Min-Max)",
    )

    # Plot mean as black line
    ax.plot(
        grouped_stats["forecast_init_dt"],
        grouped_stats[(metric, "mean")],
        color="black",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Ensemble Mean",
    )

    ax.set_title("Hit Rate (%)", fontweight="bold")
    ax.set_ylabel("Hit Rate (%)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend()

    # False Alarm Ratio
    ax = axes[0, 1]
    metric = "false_alarm_ratio_pct"

    ax.fill_between(
        grouped_stats["forecast_init_dt"],
        grouped_stats[(metric, "min")],
        grouped_stats[(metric, "max")],
        color=spread_colors[metric],
        alpha=0.3,
        label="Ensemble member range (Min-Max)",
    )

    ax.plot(
        grouped_stats["forecast_init_dt"],
        grouped_stats[(metric, "mean")],
        color="black",
        linewidth=2,
        marker="s",
        markersize=6,
        label="Ensemble Mean",
    )

    ax.set_title("False Alarm Ratio (%)", fontweight="bold")
    ax.set_ylabel("False Alarm Ratio (%)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend()

    # Critical Success Index
    ax = axes[1, 0]
    metric = "csi_pct"

    ax.fill_between(
        grouped_stats["forecast_init_dt"],
        grouped_stats[(metric, "min")],
        grouped_stats[(metric, "max")],
        color=spread_colors[metric],
        alpha=0.3,
        label="Ensemble member range (Min-Max)",
    )

    ax.plot(
        grouped_stats["forecast_init_dt"],
        grouped_stats[(metric, "mean")],
        color="black",
        linewidth=2,
        marker="^",
        markersize=6,
        label="Ensemble Mean",
    )

    ax.set_title("Critical Success Index (%)", fontweight="bold")
    ax.set_ylabel("CSI (%)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend()

    # Flooded Area
    ax = axes[1, 1]
    metric = "flooded_area_km2"

    ax.fill_between(
        grouped_stats["forecast_init_dt"],
        grouped_stats[(metric, "min")],
        grouped_stats[(metric, "max")],
        color=spread_colors[metric],
        alpha=0.3,
        label="Ensemble member range (Min-Max)",
    )

    ax.plot(
        grouped_stats["forecast_init_dt"],
        grouped_stats[(metric, "mean")],
        color="black",
        linewidth=2,
        marker="d",
        markersize=6,
        label="Ensemble Mean",
    )

    ax.set_title("Flooded Area (km²)", fontweight="bold")
    ax.set_ylabel("Area (km²)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format x-axis for all subplots
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Forecast Initialization Time")

    plt.tight_layout()

    # Save the plot
    plot_filename = f"{event_name.replace(':', '_')}_forecast_performance_spread.png"
    fig.savefig(output_folder / plot_filename, dpi=300, bbox_inches="tight")
    print(f"Forecast performance spread plot saved as: {output_folder / plot_filename}")


def find_exact_observation_file(event_name: str, files: list[Path]) -> Path | None:
    """Find the matching observation file for a flood event.

    The observation files must be named exactly using the event's
    start and end times (e.g., `20210712T090000 - 20210720T090000.zarr`).
    Matching is done by comparing the filename stem (without extension)
    to the event_name.

    Args:
        event_name: The event identifier in the format
            "YYYYMMDDTHHMMSS - YYYYMMDDTHHMMSS".
        files: List of file paths to available observation files.

    Returns:
        Path | None: The matching observation file if found, otherwise None.
    """
    for f in files:
        if f.stem == event_name:
            return f
    return None


def create_flood_animation(
    simulation_dir: Path,
    output_path_stem: Path,
    minimum_flood_depth: float,
    logger: Any,
    step: int = 1,
    fps: int = 5,
    dpi: int = 100,
    background: str = "sat",
    zoom: int = 12,
    fmt: str | None = None,
    vmax: float | None = None,
) -> Path | None:
    """Create a flood depth animation from a single SFINCS simulation output.

    Reads the per-time-step water depth (``h``) from ``sfincs_map.nc`` in the
    simulation directory and renders it as a video (MP4 if ffmpeg is available,
    otherwise an animated GIF), optionally on top of a satellite or OSM
    background map.

    Args:
        simulation_dir: SFINCS simulation directory containing ``sfincs_map.nc``.
        output_path_stem: Output file path without extension; the extension is
            derived from the selected format.
        minimum_flood_depth: Water depths below this value (in m) are masked.
        logger: Logger for progress and warning messages.
        step: Animate every nth time step.
        fps: Frames per second of the video.
        dpi: Resolution of the video frames.
        background: Background map: "sat" (satellite), "osm", or "none".
        zoom: Zoom level of the background map tiles.
        fmt: Output format, "mp4" or "gif". If None, MP4 is used when ffmpeg is
            available and GIF otherwise.
        vmax: Upper limit of the color scale in m. If None, the 98th percentile
            of the wet-cell depths is used, so that a few deep (river channel)
            cells do not compress the rest of the flood into the lightest color
            bin; deeper cells saturate into the darkest bin.

    Returns:
        Path to the created animation, or None if the simulation has no
        per-time-step water depth output or no flooding above the threshold.

    Raises:
        RuntimeError: If fmt is "mp4" but no ffmpeg is available.
        ValueError: If fmt or background has an unknown value.
    """
    if fmt not in (None, "mp4", "gif"):
        raise ValueError(f"Unknown format: {fmt}. Use 'mp4' or 'gif'.")
    if background not in ("sat", "osm", "none"):
        raise ValueError(
            f"Unknown background: {background}. Use 'sat', 'osm' or 'none'."
        )

    sfincs_model = SfincsModel(str(simulation_dir), mode="r")
    sfincs_model.read()
    results = sfincs_model.output.data

    if "h" in results:
        water_depth_m = results["h"]
        assert isinstance(water_depth_m, xr.DataArray)
    elif "zs" in results and "zb" in results:
        water_surface_m, bed_elevation_m = results["zs"], results["zb"]
        assert isinstance(water_surface_m, xr.DataArray)
        assert isinstance(bed_elevation_m, xr.DataArray)
        water_depth_m = water_surface_m - bed_elevation_m
    else:
        logger.warning(
            f"No per-time-step water depth (h or zs/zb) found in {simulation_dir / 'sfincs_map.nc'}. "
            "Set hazards.floods.flood_map_output_interval_seconds in the model config "
            "(e.g. 3600 for hourly maps) and re-run the model to store intermediate flood maps."
        )
        return None

    water_depth_m = water_depth_m.where(water_depth_m > minimum_flood_depth)
    if water_depth_m.time.size <= 1:
        logger.warning(
            f"Only {water_depth_m.time.size} time step(s) found in {simulation_dir / 'sfincs_map.nc'}; "
            "cannot animate. Set hazards.floods.flood_map_output_interval_seconds in the "
            "model config (e.g. 3600 for hourly maps) and re-run the model."
        )
        return None

    max_depth_m = float(water_depth_m.max().values)
    if not np.isfinite(max_depth_m):
        logger.warning(
            f"No flooding above {minimum_flood_depth} m in {simulation_dir.name}, skipping animation."
        )
        return None

    # select the animated frames and load them into memory
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}.")
    frame_indices = np.arange(0, water_depth_m.time.size, step)
    frames: np.ndarray = water_depth_m.isel(time=frame_indices).values
    frame_times = water_depth_m.time.values[frame_indices]

    # scale the colorbar to the bulk of the wet cells rather than the absolute
    # maximum, which is typically reached only inside the river channel
    if vmax is None:
        vmax = float(np.nanquantile(frames, 0.98))
        if vmax <= minimum_flood_depth:
            vmax = max_depth_m

    # discrete flood depth colormap (color scheme shared with the flood map plots)
    cmap = ListedColormap(FLOOD_DEPTH_COLORS)
    bounds = np.linspace(minimum_flood_depth, vmax, cmap.N + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    colorbar_extend = "max" if max_depth_m > vmax else "neither"

    # 2D cell-center coordinates: xc/yc for rotated grids, else regular x/y
    if "xc" in water_depth_m.coords and "yc" in water_depth_m.coords:
        x_2d = water_depth_m["xc"].values
        y_2d = water_depth_m["yc"].values
    else:
        x_2d, y_2d = np.meshgrid(water_depth_m["x"].values, water_depth_m["y"].values)

    crs = water_depth_m.rio.crs

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    mesh_kwargs: dict[str, Any] = {}
    if background != "none":
        # fetch the background tiles once as a static image (in Web Mercator)
        # instead of a cartopy tile layer, which would be re-rendered for
        # every frame of the animation
        transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
        x_2d, y_2d = transformer.transform(x_2d, y_2d)
        tile_source = (
            ctx.providers.Esri.WorldImagery  # ty:ignore[unresolved-attribute]
            if background == "sat"
            else ctx.providers.OpenStreetMap.Mapnik  # ty:ignore[unresolved-attribute]
        )
        tiles, tiles_extent = ctx.bounds2img(
            x_2d.min(),
            y_2d.min(),
            x_2d.max(),
            y_2d.max(),
            zoom=zoom,
            source=tile_source,
        )
        ax.imshow(tiles, extent=tiles_extent, zorder=0)
        ax.set_xlim(x_2d.min(), x_2d.max())
        ax.set_ylim(y_2d.min(), y_2d.max())
        mesh_kwargs["alpha"] = 0.8
    ax.set_xticks([])
    ax.set_yticks([])

    mesh = ax.pcolormesh(
        x_2d,
        y_2d,
        np.ma.masked_invalid(frames[0]),
        cmap=cmap,
        norm=norm,
        shading="auto",
        zorder=1,
        **mesh_kwargs,
    )
    colorbar = fig.colorbar(
        mesh, ax=ax, shrink=0.6, format="%.2f", extend=colorbar_extend
    )
    colorbar.set_label("Water depth (m)")
    title = ax.set_title("")

    def update_frame(frame_index: int) -> list:
        mesh.set_array(np.ma.masked_invalid(frames[frame_index]).ravel())
        timestamp = np.datetime_as_string(frame_times[frame_index], unit="s")
        title.set_text(f"Water depth: {timestamp}")
        return [mesh, title]

    animation = manimation.FuncAnimation(
        fig, update_frame, frames=len(frames), blit=False
    )

    # use the ffmpeg bundled with imageio-ffmpeg, so MP4 export works without
    # a system-wide ffmpeg installation
    try:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    ffmpeg_available: bool = manimation.writers.is_available("ffmpeg")
    if fmt is None:
        fmt = "mp4" if ffmpeg_available else "gif"
    if fmt == "mp4":
        if not ffmpeg_available:
            raise RuntimeError(
                "MP4 export requires ffmpeg, which was not found. "
                "Install imageio-ffmpeg or use fmt='gif'."
            )
        # crop to even pixel dimensions, required by the yuv420p pixel format
        writer = manimation.FFMpegWriter(
            fps=fps,
            extra_args=[
                "-vf",
                "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
            ],
        )
    else:
        writer = manimation.PillowWriter(fps=fps)

    output_path = output_path_stem.with_suffix(f".{fmt}")
    logger.info(
        f"Rendering {len(frames)} frames "
        f"({np.datetime_as_string(frame_times[0], unit='s')} - "
        f"{np.datetime_as_string(frame_times[-1], unit='s')}) to {output_path}"
    )
    animation.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


class Hydrodynamics:
    """Implements several functions to evaluate the hydrodynamics module of GEB."""

    def __init__(self, model: GEBModel, evaluator: Evaluate) -> None:
        """Initialize the Hydrodynamics evaluation module."""
        self.model = model
        self.evaluator = evaluator

    def animate_flood(
        self,
        run_name: str = "default",
        event: str | None = None,
        step: int = 1,
        fps: int = 5,
        dpi: int = 100,
        background: str = "sat",
        zoom: int = 12,
        fmt: str | None = None,
        vmax: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[Path]:
        """Create videos of the flood depth over time from SFINCS simulation outputs.

        One animation is created per SFINCS model and event found in the
        simulation root of the run. The videos are written to
        `output/<run_name>/evaluate/hydrodynamics/<event>/`.

        By default GEB does not store intermediate flood maps. Set
        `hazards.floods.flood_map_output_interval_seconds` in the model config
        (e.g. 3600 for hourly maps) and re-run the model first.

        Args:
            run_name: Name of the simulation run to animate.
            event: Only animate simulations whose event name contains this
                string. If None, all events are animated.
            step: Animate every nth time step.
            fps: Frames per second of the video.
            dpi: Resolution of the video frames.
            background: Background map: "sat" (satellite), "osm", or "none".
            zoom: Zoom level of the background map tiles.
            fmt: Output format, "mp4" or "gif". If None, MP4 is used when
                ffmpeg is available and GIF otherwise.
            vmax: Upper limit of the color scale in m. If None, the 98th
                percentile of the wet-cell depths is used; deeper (river
                channel) cells saturate into the darkest bin. Set explicitly
                to compare animations of different runs on the same scale.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of paths to the created animations.

        Raises:
            FileNotFoundError: If no SFINCS simulations are found for the run.
        """
        minimum_flood_depth: float = self.model.config["hazards"]["floods"][
            "minimum_flood_depth"
        ]

        sfincs_root = (
            Path(self.model.config["general"]["simulation_root"]) / run_name / "SFINCS"
        )
        if not sfincs_root.exists():
            raise FileNotFoundError(
                f"SFINCS simulation folder {sfincs_root} does not exist. "
                "Did you run the hydrodynamic model?"
            )

        animations_root = Path(self.evaluator.output_folder_evaluate) / "hydrodynamics"

        created_animations: list[Path] = []
        for model_dir in sorted(sfincs_root.iterdir()):
            simulations_folder = model_dir / "simulations"
            if not simulations_folder.is_dir():
                continue
            for simulation_dir in sorted(simulations_folder.iterdir()):
                if not (simulation_dir / "sfincs_map.nc").exists():
                    continue
                if event is not None and event not in simulation_dir.name:
                    continue

                event_folder = animations_root / simulation_dir.name
                event_folder.mkdir(parents=True, exist_ok=True)

                self.model.logger.info(
                    f"Creating flood animation for {model_dir.name} / {simulation_dir.name}"
                )
                animation_path = create_flood_animation(
                    simulation_dir=simulation_dir,
                    output_path_stem=event_folder / f"flood_animation_{model_dir.name}",
                    minimum_flood_depth=minimum_flood_depth,
                    logger=self.model.logger,
                    step=step,
                    fps=fps,
                    dpi=dpi,
                    background=background,
                    zoom=zoom,
                    fmt=fmt,
                    vmax=vmax,
                )
                if animation_path is not None:
                    self.model.logger.info(f"Saved {animation_path}")
                    created_animations.append(animation_path)

        if not created_animations:
            raise FileNotFoundError(
                f"No SFINCS simulations with per-time-step output found in {sfincs_root}"
                + (f" for event filter '{event}'" if event is not None else "")
                + ". Set hazards.floods.flood_map_output_interval_seconds in the model "
                "config (e.g. 3600 for hourly maps) and re-run the model."
            )
        return created_animations

    def evaluate_flood(
        self, run_name: str = "default", *args: Any, **kwargs: Any
    ) -> dict[str, float]:
        """Evaluate flood model performance against validation events.

        This method loads modelled flood maps and corresponding observations for
        pre-set validation events, computes spatial performance metrics (e.g.,
        hit rate, false alarm ratio, critical success index), and generates
        diagnostic visualisations and summary outputs for the specified
        simulation run.

        Args:
            run_name: Name of the simulation run to evaluate.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Dictionary containing mean performance metrics across all validation events:
                - CSI_mean: Mean critical success index across events.
                - FAR_mean: Mean false alarm ratio across events.
                - HR_mean: Mean hit rate across events.

        Raises:
            FileNotFoundError: If the flood map folder or observation file does not exist.
        """
        self.config = self.model.config["hazards"]

        eval_flood_folders = (
            Path(self.evaluator.output_folder_evaluate) / "hydrodynamics" / "validation"
        )
        eval_flood_folders.mkdir(parents=True, exist_ok=True)

        flood_maps_folder = self.model.output_folder / "flood_maps"
        if not flood_maps_folder.exists():
            raise FileNotFoundError(
                "Flood map folder does not exist in the output directory. Did you run GEB correctly?"
            )

        # Read validation events
        validation_events_file = self.model.files["geom"]["observations/floods"]
        validation_events = read_geom(validation_events_file)

        if validation_events.empty:
            self.model.logger.warning(
                "No validation events found in the observation file. Skipping evaluation."
            )
            return {
                "CSI_mean": np.nan,
                "FAR_mean": np.nan,
                "HR_mean": np.nan,
            }
        else:
            performance_metrics_list: list[dict[str, float | int]] = []
            for _, event_row in validation_events.iterrows():
                event_name = event_row["name"]
                self.model.logger.info(f"Evaluating event: {event_name}")

                # Create event-specific folder
                event_folder = eval_flood_folders / event_name
                event_folder.mkdir(parents=True, exist_ok=True)

                # Find observation file from model files (setup via setup_flood_observations)
                obs_file = self.model.files["other"][
                    f"observations/floods/{event_name}"
                ]

                if not obs_file.exists():
                    raise FileNotFoundError(
                        f"Observation file for event {event_name} not found at {obs_file}. Please check the path in the config file and ensure setup_flood_observations was run correctly."
                    )

                flood_map_path = flood_maps_folder / f"{event_name}_final.zarr"

                if not flood_map_path.exists():
                    raise FileNotFoundError(
                        f"Flood map for event {event_name} not found at {flood_map_path}. Please check the path in the config file and ensure the simulation was run correctly."
                    )

                performance = calculate_performance_metrics(
                    observation=read_zarr(obs_file),
                    simulated=read_zarr(flood_map_path),
                    output_folder=event_folder,
                    run_name=run_name,
                    minimum_flood_depth=self.config["floods"]["minimum_flood_depth"],
                    elevation_data=read_zarr(self.model.files["other"]["DEM/fabdem"]),
                    visualization_type="OSM",
                    name=flood_map_path.stem,
                )
                if performance is None:
                    self.model.logger.warning(
                        f"Performance metrics calculation failed for event {event_name}. Skipping."
                    )
                    continue

                self.model.logger.info(f"Successfully evaluated: {flood_map_path.name}")
                performance_metrics_list.append(performance)

            overall_performance_df = pd.DataFrame(performance_metrics_list)
            overall_performance_df.to_csv(
                eval_flood_folders / "overall_performance_metrics.csv", index=False
            )
            self.model.logger.info("Finished evaluating validation flood events.")

            return {
                "CSI_mean": overall_performance_df["csi_pct"].mean(),
                "FAR_mean": overall_performance_df["false_alarm_ratio_pct"].mean(),
                "HR_mean": overall_performance_df["hit_rate_pct"].mean(),
            }

    def evaluate_hydrodynamics(
        self, run_name: str = "default", *args: Any, **kwargs: Any
    ) -> None:
        """Evaluate hydrodynamic model performance against flood observations.

        This method loads modelled flood maps and corresponding observations,
        computes spatial performance metrics (e.g., hit rate, false alarm ratio,
        critical success index), and generates diagnostic visualisations and
        summary outputs for the specified simulation run.

        Args:
            run_name: Name of the simulation run to evaluate.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            FileNotFoundError: If the flood map folder does not exist in the output directory.
            ValueError: If the flood observation file is not in .zarr format.
        """
        self.config = self.model.config["hazards"]

        eval_hydrodynamics_folders = (
            Path(self.evaluator.output_folder_evaluate) / "hydrodynamics"
        )

        eval_hydrodynamics_folders.mkdir(parents=True, exist_ok=True)

        # Calculate performance metrics for every event in config file
        for event in self.config["floods"]["events"]:
            event_name = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}"
            self.model.logger.info(f"event: {event_name}")

            # Create event-specific folder
            if self.model.config["general"]["forecasts"]["use"]:
                event_folder = eval_hydrodynamics_folders / "forecasts" / event_name
                event_folder.mkdir(parents=True, exist_ok=True)
                flood_maps_folder = (
                    self.model.output_folder / "flood_maps" / "forecasts"
                )
            else:
                event_folder = eval_hydrodynamics_folders / event_name
                event_folder.mkdir(parents=True, exist_ok=True)
                flood_maps_folder = self.model.output_folder / "flood_maps"

            # check if run file exists, if not, raise an error
            if not flood_maps_folder.exists():
                raise FileNotFoundError(
                    "Flood map folder does not exist in the output directory. Did you run the hydrodynamic model?"
                )

            # Extract the observation files, find the match with the flood event
            obs_raw = self.config["floods"]["observation_files"]
            if isinstance(obs_raw, str):
                observation_files = [Path(obs_raw)]
            else:
                observation_files = [Path(p) for p in obs_raw]
            obs_file = find_exact_observation_file(event_name, observation_files)

            # check if observation file exists, if not, raise an error
            if obs_file is None:
                print(
                    f"No observation file for this event: '{event_name}'. Skipping event."
                )
                continue
            if not obs_file.exists():
                raise FileNotFoundError(
                    "Flood observation file is not found in the given path in the model.yml Please check the path in the config file."
                )
            if obs_file.suffix != ".zarr":
                raise ValueError(
                    "Flood observation file is not in the correct format. Please provide a .zarr file."
                )

            # Find all flood maps corresponding to the event
            all_flood_map_files = list(flood_maps_folder.glob("*.zarr"))

            # Filter flood_map_files for the current event only
            flood_map_files = []
            for flood_map_path in all_flood_map_files:
                parsed = parse_flood_forecast_initialisation(flood_map_path.name)

                # Skip files that do not match the expected format
                if parsed is None:
                    continue

                file_forecast_init, _, _, _, parsed_event_name = parsed
                # Check if file matches current event
                if parsed_event_name == event_name:
                    flood_map_files.append(flood_map_path)

            self.model.logger.info(
                f"Found {len(flood_map_files)} flood map files for event {event_name}"
            )

            if len(flood_map_files) == 1:
                self.model.logger.info(
                    "Only one flood map found, assuming no forecasts were included in the simulation."
                )
                flood_map_name = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
                flood_map_path = (
                    Path(self.model.output_folder) / "flood_maps" / flood_map_name
                )
                calculate_performance_metrics(
                    observation=read_zarr(obs_file),
                    simulated=read_zarr(flood_map_path),
                    output_folder=event_folder,
                    run_name=run_name,
                    minimum_flood_depth=self.config["floods"]["minimum_flood_depth"],
                    elevation_data=read_zarr(self.model.files["other"]["DEM/fabdem"]),
                    visualization_type="OSM",
                    name=flood_map_path.stem,
                )
                print(f"Successfully evaluated: {flood_map_path.name}")

            elif len(flood_map_files) == 0:
                raise FileNotFoundError(
                    "No flood map files found for this event. Did you run the hydrodynamic model?"
                )

            else:
                print(
                    f"Multiple flood maps found ({len(flood_map_files)}), processing each."
                )
                unique_forecast_inits = set()
                performance_metrics_list = []

                # Identify unique forecast initializations
                for flood_map_name in flood_map_files:
                    # Parse the flood map filename to extract components
                    print(f"flood_map_name: {flood_map_name}")
                    forecast_init, member, event_start, event_end, parsed_event_name = (
                        parse_flood_forecast_initialisation(flood_map_name.name)
                    )
                    unique_forecast_inits.add(forecast_init)

                # Convert to sorted list for consistent processing order
                unique_forecast_inits_list = sorted(
                    [init for init in unique_forecast_inits if init is not None]
                )
                print(
                    f"Found {len(unique_forecast_inits_list)} unique forecast initializations: {unique_forecast_inits_list}"
                )

                # Process each unique forecast initialization
                for forecast_init in unique_forecast_inits_list:
                    print(f"Processing forecast initialization: {forecast_init}")

                    # Create forecast initialization folder
                    forecast_folder = event_folder / forecast_init
                    forecast_folder.mkdir(parents=True, exist_ok=True)

                    matching_flood_maps = []
                    for flood_map_path in flood_map_files:
                        parsed = parse_flood_forecast_initialisation(
                            flood_map_path.name
                        )

                        # Skip files that do not match the expected format
                        if parsed is None:
                            continue

                        file_forecast_init, _, _, _, parsed_event_name = parsed

                        # Only include files that match current forecast init and event
                        if (
                            file_forecast_init == forecast_init
                            and parsed_event_name == event_name
                        ):
                            matching_flood_maps.append(flood_map_path)

                    print(
                        f"Found {len(matching_flood_maps)} flood maps for forecast initialization {forecast_init}"
                    )
                    # Evaluate each matching flood map
                    forecast_metrics_list = []

                    for flood_map_path in matching_flood_maps:
                        print(f"   Evaluating: {flood_map_path.name}")

                        elevation_path = self.config["floods"].get(
                            "elevation_data", self.model.files["other"]["DEM/fabdem"]
                        )
                        metrics = calculate_performance_metrics(
                            observation=read_zarr(obs_file),
                            simulated=read_zarr(flood_map_path),
                            visualization_type="OSM",
                            output_folder=forecast_folder,
                            run_name=run_name,
                            minimum_flood_depth=self.config["floods"][
                                "minimum_flood_depth"
                            ],
                            elevation_data=read_zarr(elevation_path),
                            name=flood_map_path.stem,
                        )
                        if metrics is None:
                            continue
                        print("   Flood map evaluation complete.")
                        # Add metadata to metrics
                        forecast_init_parsed, member, _, _, _ = (
                            parse_flood_forecast_initialisation(flood_map_path.name)
                        )
                        metrics_with_metadata = {
                            "forecast_init": forecast_init_parsed,
                            "member": member,
                            "filename": flood_map_path.name,
                            **metrics,
                        }

                        performance_metrics_list.append(metrics_with_metadata)
                        forecast_metrics_list.append(metrics)
                        print(f"   Successfully evaluated: {flood_map_path.name}")

                if performance_metrics_list:
                    performance_df = pd.DataFrame(performance_metrics_list)

                    # Create forecast performance plots
                    create_forecast_performance_plots(
                        performance_df, event_name, event_folder
                    )

                    # Save detailed performance metrics
                    detailed_filename = f"{event_name.replace(':', '_')}_detailed_performance_metrics.csv"
                    performance_df.to_csv(event_folder / detailed_filename, index=False)
                    print(
                        f"Detailed performance metrics saved as: {event_folder / detailed_filename}"
                    )

            print(f"Completed processing event: {event_name}\n")

        print("Flood map performance metrics calculated for all events.")
