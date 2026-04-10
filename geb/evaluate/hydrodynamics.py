"""Module implementing hydrodynamics evaluation functions for the GEB model."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from rasterio.crs import CRS  # ty:ignore[unresolved-import]
from rasterio.features import geometry_mask

from geb.workflows.io import read_geom, read_zarr

if TYPE_CHECKING:
    from geb.evaluate import Evaluate
    from geb.model import GEBModel


def calculate_critical_success_index(
    model: xr.DataArray, observations: xr.DataArray
) -> float:
    """Calculate the critical success index (CSI) metric.

    Args:
        model: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        Critical success index as a float.
    """
    hit = np.sum(((model == 1) & (observations == 1)).values)
    false_alarm = np.sum(((model == 1) & (observations == 0)).values)
    miss = np.sum(((model == 0) & (observations == 1)).values)
    csi = hit / (hit + false_alarm + miss)
    return float(csi)


def calculate_false_alarm_ratio(
    model: xr.DataArray, observations: xr.DataArray
) -> float:
    """Calculate the false alarm ratio metric.

    Args:
        model: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        False alarm ratio as a float.
    """
    false_alarm = np.sum(((model == 1) & (observations == 0)).values)
    hit = np.sum(((model == 1) & (observations == 1)).values)
    false_alarm_ratio = false_alarm / (false_alarm + hit)
    return float(false_alarm_ratio)


def calculate_hit_rate(model: xr.DataArray, observations: xr.DataArray) -> float:
    """Calculate the hit rate metric.

    Args:
        model: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        Hit rate as a float.
    """
    miss = np.sum(((model == 0) & (observations == 1)).values)
    hit = np.sum(((model == 1) & (observations == 1)).values)
    hit_rate = hit / (hit + miss)
    return float(hit_rate)


class Hydrodynamics:
    """Implements several functions to evaluate the hydrodynamics module of GEB."""

    def __init__(self, model: GEBModel, evaluator: Evaluate) -> None:
        """Initialize the Hydrodynamics evaluation module."""
        self.model = model
        self.evaluator = evaluator

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

        # Main function for the performance metrics
        def calculate_performance_metrics(
            observation: Path | str,
            flood_map_path: Path | str,
            output_folder: Path,
            visualization_type: str = "Hillshade",
        ) -> dict[str, float | int] | None:
            """Calculate performance metrics for flood maps against observations.

            Args:
                observation: Path to the observed flood extent data (.zarr format).
                flood_map_path: Path to the model-generated flood map data (.zarr format).
                visualization_type: Type of visualization for plotting (default is "Hillshade").
                output_folder: Path to the folder where results will be saved.

            Returns:
                Dictionary containing performance metrics:
                    - hit_rate: Percentage of correctly predicted flooded areas.
                    - false_alarm_ratio: Percentage of falsely predicted flooded areas.
                    - critical_success_index: Overall accuracy of flood predictions.
                    - flooded_area_km2: Total flooded area in square kilometers.
                or None if an error occurs.

            Raises:
                ValueError: If the observation file is not in .zarr format.
            """
            # Step 1: Open needed datasets
            flood_map = read_zarr(flood_map_path)
            obs = read_zarr(observation)
            print("obs CRS", obs.rio.crs)
            sim = flood_map.rio.reproject_match(obs)
            rivers = read_geom(
                Path("simulation_root")
                / run_name
                / "SFINCS"
                / "group_0"
                / "rivers.geoparquet"
            ).to_crs(obs.rio.crs)
            region = read_geom(
                Path("simulation_root")
                / run_name
                / "SFINCS"
                / "group_0"
                / "region.geoparquet"
            ).to_crs(obs.rio.crs)

            crs_mercator = CRS.from_epsg(3857)
            gdf_mercator = rivers.to_crs(crs_mercator)
            gdf_mercator["geometry"] = gdf_mercator.buffer(gdf_mercator["width"] / 2)

            # Create river mask for simulation data
            gdf_buffered_sim = gdf_mercator.to_crs(sim.rio.crs)
            rivers_mask_sim = ~geometry_mask(
                gdf_buffered_sim.geometry,
                out_shape=sim.rio.shape,
                transform=sim.rio.transform(),
                all_touched=True,
                invert=False,
            )
            sim_no_rivers = sim.where(~rivers_mask_sim).fillna(0)

            # Create river mask for observation data
            gdf_buffered_obs = gdf_mercator.to_crs(obs.rio.crs)
            rivers_mask_obs = ~geometry_mask(
                gdf_buffered_obs.geometry,
                out_shape=obs.rio.shape,
                transform=obs.rio.transform(),
                all_touched=True,
                invert=False,
            )
            obs_no_rivers = obs.where(~rivers_mask_obs).fillna(0)

            # Clip out region from observations
            obs_region = obs_no_rivers.rio.clip(region.geometry.values, region.crs)

            # Optionally clip using extra validation region from config yml
            extra_validation_path = self.config["floods"].get(
                "extra_validation_region", None
            )

            # Mask water depth values
            hmin: float = self.config["floods"]["minimum_flood_depth"]

            if extra_validation_path and Path(extra_validation_path).exists():
                extra_clip_region = gpd.read_file(extra_validation_path).set_crs(28992)
                extra_clip_region = extra_clip_region.to_crs(region.crs)
                extra_clip_region_buffer = extra_clip_region.buffer(160)

                sim_extra_clipped = sim_no_rivers.rio.clip(
                    extra_clip_region_buffer.geometry.values,
                    extra_clip_region_buffer.crs,
                )
                clipped_out = (sim_no_rivers > hmin) & (sim_extra_clipped.isnull())
                clipped_out_raster = sim_no_rivers.where(clipped_out)
            else:
                # If no extra validation region, skip clipping
                sim_extra_clipped = sim_no_rivers
                clipped_out_raster = xr.full_like(sim_no_rivers, np.nan)

            sim_extra_clipped = sim_extra_clipped.rio.reproject_like(obs_region)
            simulation_final = sim_extra_clipped > hmin
            observation_final = obs_region > 0

            xmin, ymin, xmax, ymax = region.total_bounds
            catchment_extent = [xmin, xmax, ymin, ymax]

            xmin, ymin, xmax, ymax = observation_final.rio.bounds()
            flood_extent: tuple[float, float, float, float] = (xmin, xmax, ymin, ymax)

            # Calculate performance metrics
            # Compute the arrays first to get concrete values
            sim_final_computed = simulation_final.compute()
            obs_final_computed = observation_final.compute()

            hit_rate = calculate_hit_rate(sim_final_computed, obs_final_computed) * 100
            false_rate = (
                calculate_false_alarm_ratio(sim_final_computed, obs_final_computed)
                * 100
            )
            csi = (
                calculate_critical_success_index(sim_final_computed, obs_final_computed)
                * 100
            )

            flooded_pixels = float(sim_final_computed.sum().item())

            # Calculate resolution in meters from coordinate spacing
            x_res = float(np.abs(flood_map.x[1] - flood_map.x[0]))
            pixel_size = x_res  # meters
            flooded_area_km2 = flooded_pixels * (pixel_size * pixel_size) / 1_000_000

            # Step 7: Save results to file and plot the results
            elevation_data = read_zarr(self.model.files["other"]["DEM/fabdem"])
            elevation_data = elevation_data.rio.reproject_match(obs)

            elevation_array = (
                elevation_data.squeeze().astype("float32").compute().values
            )

            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(elevation_array, vert_exag=1, dx=1, dy=1)

            # Catchment borders
            target_crs = obs_region.rio.crs
            catchment_borders = region.boundary

            if simulation_final.sum() > 0:
                misses = (observation_final == 1) & (simulation_final == 0)
                simulation_masked = simulation_final.where(simulation_final == 1)
                hits = simulation_masked.where(observation_final)
                misses_masked = misses.where(misses == 1)

                if visualization_type == "OSM":
                    # Ensure all arrays have the same shape by squeezing extra dimensions
                    simulation_final = simulation_final.squeeze()
                    observation_final = observation_final.squeeze()
                    misses = misses.squeeze()
                    hits = hits.squeeze()
                    simulation_masked = simulation_masked.squeeze()
                    misses_masked = misses_masked.squeeze()

                    margin = 3000
                    fig, ax = plt.subplots(figsize=(10, 10))

                    # clipped_out_raster.plot(
                    #     ax=ax, cmap=blue_cmap, add_colorbar=False, add_labels=False
                    # )

                    ax.imshow(
                        simulation_masked,  # False alarms
                        extent=flood_extent,
                        # origin="lower",
                        cmap="Wistia",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=1,
                    )
                    ax.imshow(
                        misses_masked,  # Misses
                        extent=flood_extent,
                        # origin="lower",
                        cmap="autumn_r",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=2,
                    )

                    ax.imshow(
                        hits,
                        extent=flood_extent,
                        # origin="lower",
                        cmap="brg",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=3,
                    )

                    catchment_borders.plot(
                        ax=ax,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        zorder=4,
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

                    # Set the extent based on the raster bounds
                    ax.set_xlim(
                        catchment_extent[0] - margin, catchment_extent[1] + margin
                    )
                    ax.set_ylim(
                        catchment_extent[2] - margin, catchment_extent[3] + margin
                    )
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
                    legend = ax.legend(
                        handles=[
                            green_patch,
                            orange_patch,
                            red_patch,
                            catchment_patch,
                        ]
                    )

                    # Add a comment about the metrics in the plot
                    legend_bbox = legend.get_window_extent(
                        renderer=fig.canvas.get_renderer()  # ty:ignore[unresolved-attribute]
                    )
                    legend_bbox_ax = legend_bbox.transformed(ax.transAxes.inverted())

                    # Add text below legend using axes coordinates
                    ax.annotate(
                        f"Validation Metrics:\n"
                        f"HR    = {hit_rate:.2f} %\n"
                        f"FAR   = {false_rate:.2f} %\n"
                        f"CSI   = {csi:.2f} %",
                        xy=(legend_bbox_ax.x0 + 0.055, legend_bbox_ax.y0 + 0.002),
                        xycoords="axes fraction",
                        fontsize=10,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="grey",
                            boxstyle="round,pad=0.2",
                            alpha=0.8,
                        ),
                        verticalalignment="top",
                        horizontalalignment="left",
                        zorder=5,
                    )

                    crs_text = f"CRS: {target_crs.to_string()}"
                    ax.annotate(
                        crs_text,
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
                        zorder=6,
                    )

                    simulation_filename = os.path.splitext(
                        os.path.basename(flood_map_path)
                    )[0]
                    fig.savefig(
                        output_folder
                        / f"{simulation_filename}_validation_floodextent_plot.png",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    print(
                        f"Figure with {visualization_type} saved as: {output_folder / f'{simulation_filename}_validation_floodextent_plot.png'}"
                    )

                elif visualization_type == "Hillshade":
                    green_cmap = mcolors.ListedColormap(["green"])  # Hits
                    orange_cmap = mcolors.ListedColormap(["orange"])  # False alarms
                    red_cmap = mcolors.ListedColormap(["red"])  # Misses
                    blue_cmap = mcolors.ListedColormap(
                        ["#72c1db"]
                    )  # No observation data

                    fig, ax = plt.subplots(figsize=(10, 10))

                    ax.imshow(
                        hillshade,
                        cmap="gray",
                        extent=(
                            elevation_data.x.min(),
                            elevation_data.x.max(),
                            elevation_data.y.min(),
                            elevation_data.y.max(),
                        ),
                    )

                    region.boundary.plot(
                        ax=ax, edgecolor="black", linewidth=2, label="Region Boundary"
                    )

                    clipped_out_raster.plot(
                        ax=ax, cmap=blue_cmap, add_colorbar=False, add_labels=False
                    )
                    simulation_masked.plot(
                        ax=ax, cmap=orange_cmap, add_colorbar=False, add_labels=False
                    )
                    hits.plot(
                        ax=ax, cmap=green_cmap, add_colorbar=False, add_labels=False
                    )
                    misses_masked.plot(
                        ax=ax, cmap=red_cmap, add_colorbar=False, add_labels=False
                    )

                    ax.set_aspect("equal")
                    ax.axis("off")
                    handles = [
                        mpatches.Patch(color=green_cmap(0.5)),
                        mpatches.Patch(color=red_cmap(0.5)),
                        mpatches.Patch(color=orange_cmap(0.5)),
                        mpatches.Patch(color=blue_cmap(0.5)),
                        mlines.Line2D([], [], color="black", linewidth=2),
                    ]

                    labels = [
                        "Hits",
                        "Misses",
                        "False alarms",
                        "No observation data",
                        "Region",
                    ]

                    plt.legend(
                        handles=handles, labels=labels, loc="upper right", fontsize=16
                    )

                    simulation_filename = os.path.splitext(
                        os.path.basename(flood_map_path)
                    )[0]
                    plt.savefig(
                        output_folder
                        / f"{simulation_filename}_validation_floodextent_plot.png"
                    )
                    print(
                        f"Figure with {visualization_type} saved as: {output_folder / f'{simulation_filename}_validation_floodextent_plot.png'}"
                    )

                else:
                    raise ValueError(
                        f"Unknown visualization type: {visualization_type}, choose either 'OSM' or 'Hillshade'."
                    )

                performance_numbers = (
                    output_folder / f"{simulation_filename}_performance_metrics.txt"
                )

                with open(performance_numbers, "w") as f:
                    f.write(f"Hit rate (H): {hit_rate}\n")
                    f.write(f"False alarm rate (F): {false_rate}\n")
                    f.write(f"Critical Success Index (CSI) (C): {csi}\n")
                    f.write(f"Number of flooded pixels: {flooded_pixels}\n")
                    f.write(f"Flooded area (km2): {flooded_area_km2}")

                # Return metrics for further analysis
                return {
                    "hit_rate": hit_rate,
                    "false_alarm_rate": false_rate,
                    "csi": csi,
                    "flooded_area_km2": flooded_area_km2,
                }

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
                    'forecast_init', 'hit_rate', 'false_alarm_rate', 'csi', 'flooded_area_km2'.
                event_name: Name of the flood event being analyzed.
                output_folder: Directory to save the performance plots.

            Raises:
                ValueError: If required columns are missing from performance_df.
            """
            # Validate required columns
            required_columns = [
                "forecast_init",
                "hit_rate",
                "false_alarm_rate",
                "csi",
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
                    ["hit_rate", "false_alarm_rate", "csi", "flooded_area_km2"]
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
                "hit_rate": "#2E86AB",
                "false_alarm_rate": "#A23B72",
                "csi": "#F18F01",
                "flooded_area_km2": "#636EFA",
            }

            # Hit Rate
            ax = axes[0, 0]
            metric = "hit_rate"

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

            # False Alarm Rate
            ax = axes[0, 1]
            metric = "false_alarm_rate"

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

            ax.set_title("False Alarm Rate (%)", fontweight="bold")
            ax.set_ylabel("False Alarm Rate (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # Critical Success Index
            ax = axes[1, 0]
            metric = "csi"

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
            plot_filename = (
                f"{event_name.replace(':', '_')}_forecast_performance_spread.png"
            )
            fig.savefig(output_folder / plot_filename, dpi=300, bbox_inches="tight")
            print(
                f"Forecast performance spread plot saved as: {output_folder / plot_filename}"
            )

        def find_exact_observation_file(
            event_name: str, files: list[Path]
        ) -> Path | None:
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

        self.config = self.model.config["hazards"]

        eval_hydrodynamics_folders = (
            Path(self.evaluator.output_folder_evaluate) / "hydrodynamics"
        )

        eval_hydrodynamics_folders.mkdir(parents=True, exist_ok=True)

        # Calculate performance metrics for every event in config file
        for event in self.config["floods"]["events"]:
            event_name = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}"
            print(f"event: {event_name}")

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

            print(
                f"Found {len(flood_map_files)} flood map files for event {event_name}"
            )

            if len(flood_map_files) == 1:
                print(
                    "Only one flood map found, assuming no forecasts were included in the simulation."
                )
                flood_map_name = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
                flood_map_path = (
                    Path(self.model.output_folder) / "flood_maps" / flood_map_name
                )
                calculate_performance_metrics(
                    observation=str(obs_file),
                    flood_map_path=flood_map_path,
                    output_folder=event_folder,
                    visualization_type="OSM",
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

                        metrics = calculate_performance_metrics(
                            observation=str(obs_file),
                            flood_map_path=flood_map_path,
                            visualization_type="OSM",
                            output_folder=forecast_folder,
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
