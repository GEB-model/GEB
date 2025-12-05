"""Evaluation of meteorological forecasts in GEB.

This module provides functionality to evaluate and visualize meteorological forecasts and especially rainfall data by
comparing ECMWF ensemble forecasts against ERA5 reanalysis data for precipitation analysis.
Supports both intensity and cumulative precipitation plotting.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

# from matplotlib_scalebar.scalebar import ScaleBar
from geb.workflows.io import open_zarr


class MeteorologicalForecasts:
    """Implements several functions to evaluate the meteorological forecasts inside GEB."""

    def __init__(self) -> None:
        """Initialize MeteorologicalForecasts."""
        pass

    def evaluate_forecasts(
        self,
        spinup_name: str = "spinup",
        run_name: str = "default",
        include_spinup: bool = False,
        include_yearly_plots: bool = False,
        correct_Q_obs: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate meteorological forecasts by comparing different forecast types at different initialisation times (00:00 (not yet vs 12:00) UTC).

        It compares ERA5 reanalysis data against forecast types such as ensemble forecasts and deterministic forecasts.

        Args:
            spinup_name: Name of the spinup run (not used for meteorological evaluation).
            run_name: Name of the simulation run (not used for meteorological evaluation).
            include_spinup: Whether to include spinup run (not used for meteorological evaluation).
            include_yearly_plots: Whether to create yearly plots (not used for meteorological evaluation).
            correct_Q_obs: Whether to correct Q_obs data (not used for meteorological evaluation).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the forecast model path does not exist.
        """
        print("Evaluating meteorological forecasts...")
        # Create forecast output folder within the evaluate folder
        forecast_folder: Path = self.output_folder_evaluate / "forecasts"
        forecast_folder.mkdir(parents=True, exist_ok=True)

        processing_forecasts = (
            self.model.config.get("general", {})
            .get("forecasts", {})
            .get("processing", [])
        )

        # Base path for forecast data
        forecast_base_path: Path = (
            self.model.input_folder
            / "other"
            / "forecasts"
            / "ECMWF"
            / processing_forecasts  # or self.config["general"]["forecasts"]["processing"]
        )

        if not forecast_base_path.exists():
            raise ValueError(
                f"Forecast model path does not exist: {forecast_base_path}"
            )

        # Extract event datetime from model configuration
        flood_events = (
            self.model.config.get("hazards", {}).get("floods", {}).get("events", [])
        )
        moment_of_flooding: pd.Timestamp | None = None

        # Check for optional moment_of_flooding setting
        if flood_events and "moment_of_flooding" in flood_events[0]:
            moment_of_flooding = pd.to_datetime(
                str(flood_events[0]["moment_of_flooding"])
            )

        def evaluate_precipitation_forecasts(
            plot_type: str = "intensity",
            moment_of_flooding: pd.Timestamp | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            """Evaluate precipitation forecasts by plotting rainfall data over time for different forecast initialisations.

            This function creates multi-panel plots comparing ERA5 reanalysis data with ECMWF ensemble
            forecasts. The plot type determines whether intensity or cumulative precipitation is shown.

            Args:
                plot_type: Type of plot to generate. Either "intensity" for mm/h or "cumulative" for mm.
                moment_of_flooding: Optional moment when flooding actually occurred. If None, no flooding line is shown.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.

            Raises:
                ValueError: If plot_type is not "intensity" or "cumulative", or if forecast path does not exist.
            """
            # Validate plot type
            if plot_type not in ["intensity", "cumulative"]:
                raise ValueError(
                    f"plot_type must be 'intensity' or 'cumulative', got '{plot_type}'"
                )

            def load_forecast_data(
                init_folder: Path, plot_type: str = "intensity"
            ) -> tuple[
                xarray.DataArray, xarray.DataArray, np.ndarray, xarray.DataArray
            ]:
                """Import ERA5, control and ensemble data for one forecast initialisation.

                Processes precipitation data either as maximum intensity per time step or
                cumulative precipitation over time, depending on the plot_type parameter.

                Args:
                    init_folder: Path to forecast initialisation folder containing zarr files.
                    plot_type: If "cumulative", calculate cumulative precipitation. If "intensity", use intensity.

                Returns:
                    Tuple containing ERA5 clipped data, control forecast, control time coordinates,
                    and ensemble data. Units are mm/h for intensity or mm for cumulative.

                Raises:
                    FileNotFoundError: If no precipitation zarr file found in init_folder.
                """
                zarr_files: list[Path] = list(init_folder.glob("*.zarr"))

                # Look for files containing 'pr' (precipitation)
                pr_files: list[Path] = [f for f in zarr_files if "pr" in f.name.lower()]

                if not pr_files:
                    raise FileNotFoundError(
                        f"No precipitation zarr file found in {init_folder}"
                    )

                if len(pr_files) > 1:
                    print(
                        f"Warning: Multiple precipitation files found in {init_folder}, using first: {pr_files[0]}"
                    )

                era5_path = (
                    self.model.input_folder
                    / "other"
                    / "climate"
                    / "pr_kg_per_m2_per_s.zarr"
                )

                # ERA5
                era5_ds = open_zarr(era5_path)
                era5_mm_per_h: xarray.DataArray = (
                    era5_ds * 3600
                )  # Convert from m/s to mm/h
                era5_max_mm_per_h: xarray.DataArray = era5_mm_per_h.max(dim=["y", "x"])

                # Load ensemble data (including control)
                ens_ds = open_zarr(init_folder / pr_files[0].name)
                ens_time = ens_ds["time"].values
                ens_mm_per_h: xarray.DataArray = (
                    ens_ds * 3600
                )  # Convert from m/s to mm/h
                ens_max_mm_per_h: xarray.DataArray = ens_mm_per_h.max(dim=["y", "x"])

                # ERA5 clip on time range control
                era5_clipped = era5_max_mm_per_h.sel(
                    time=slice(ens_time[0], ens_time[-1])
                )

                # Apply cumulative sum if requested
                if plot_type == "cumulative":
                    # For cumulative: first cumulative sum, then spatial maximum (same as spatial plot)
                    era5_cumulative_spatial = era5_mm_per_h.sel(
                        time=slice(ens_time[0], ens_time[-1])
                    ).cumsum(dim="time")
                    era5_processed: xarray.DataArray = era5_cumulative_spatial.max(
                        dim=["y", "x"]
                    )

                    # For ensemble data: first cumulative sum, then spatial maximum
                    ens_cumulative_spatial = ens_mm_per_h.cumsum(dim="time")
                    ens_processed: xarray.DataArray = ens_cumulative_spatial.max(
                        dim=["y", "x"]
                    )
                else:
                    era5_processed: xarray.DataArray = era5_clipped
                    ens_processed: xarray.DataArray = ens_max_mm_per_h

                # Control is member 0
                control_processed: xarray.DataArray = ens_processed.isel(member=0)

                # Ensemble data (all members)
                ensemble_processed: xarray.DataArray = ens_processed.isel(
                    member=slice(1, None)
                )

                return era5_processed, control_processed, ens_time, ensemble_processed

            def format_time_axis(
                ax: plt.Axes,
                x_start: pd.Timestamp,
                x_end: pd.Timestamp,
                x_ticks: list[pd.Timestamp],
            ) -> None:
                """Format the time axis for the plots."""
                ax.set_xlim(x_start, x_end)
                ax.set_xticks(x_ticks)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True)

            def plot_rainfall_data(
                ax: plt.Axes,
                era5_data: xarray.DataArray,
                control_data: xarray.DataArray,
                ensemble_data: xarray.DataArray,
                control_time: np.ndarray,
                init_time_str: str,
                plot_type: str = "intensity",
                show_legend: bool = False,
                moment_of_flooding: pd.Timestamp | None = None,
                x_start: pd.Timestamp = pd.Timestamp("2024-04-26 00:00"),
                x_end: pd.Timestamp = pd.Timestamp("2024-05-12 00:00"),
                spatial_ax: plt.Axes | None = None,
                spatial_data: dict | None = None,
            ) -> None:
                """Plot precipitation data (intensity or cumulative) for a single subplot.

                Args:
                    ax: Matplotlib axes object to plot on.
                    era5_data: ERA5 precipitation data (mm/h for intensity, mm for cumulative).
                    control_data: Control forecast precipitation data (mm/h for intensity, mm for cumulative).
                    ensemble_data: Ensemble forecast precipitation data (mm/h for intensity, mm for cumulative).
                    control_time: Time array for the forecast data (e.g. timesteps corresponding to the data).
                    init_time_str: Initialization time identifier string (e.g., '20240429T000000').
                    moment_of_flooding: Optional moment when flooding actually occurred. If None, no flooding line is shown.
                    plot_type: If "cumulative", format for cumulative data. If "intensity", format for intensity.
                    show_legend: Whether to show the legend on this subplot.
                    x_start: Start time for x-axis.
                    x_end: End time for x-axis.
                    spatial_ax: Optional matplotlib axes object for spatial plot.
                    spatial_data: Optional dictionary with spatial data for plotting.
                """
                # Parse initialization time string to readable format
                init_datetime: pd.Timestamp = pd.to_datetime(
                    init_time_str, format="%Y%m%dT%H%M%S"
                )
                init_time_readable: str = init_datetime.strftime("%d %B %Y, %H:%M UTC")

                # Calculate lead time in days if moment of flooding is available
                if moment_of_flooding is not None:
                    lead_time_days: float = (
                        moment_of_flooding - init_datetime
                    ).total_seconds() / (24 * 3600)

                # Plot ERA5 and control
                ax.plot(
                    control_time,
                    era5_data,
                    color="black",
                    label="ERA5",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.plot(
                    control_time,
                    control_data,
                    color="green",
                    label="Control",
                    linewidth=2,
                    alpha=0.8,
                )

                # Calculate min and max across all ensemble members for each time step
                ensemble_min: xarray.DataArray = ensemble_data.min(dim="member")
                ensemble_max: xarray.DataArray = ensemble_data.max(dim="member")

                # Plot ensemble spread as filled area between min and max
                ax.fill_between(
                    control_time,
                    ensemble_min,
                    ensemble_max,
                    color="blue",
                    alpha=0.2,
                    label="Ensemble spread of all the members",
                )

                # Add moment of flooding line only if specified in configuration
                if moment_of_flooding is not None:
                    ax.axvline(
                        moment_of_flooding,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label="Moment of Flooding",
                    )

                # Set title and formatting
                ax.set_title(
                    f"{lead_time_days:.1f} day lead time\n(Initialized: {init_time_readable})",
                    fontsize=18,
                )

                if plot_type == "cumulative":
                    ax.set_yticks(
                        range(0, 801, 100)
                    )  # Adjusted for cumulative mm values
                # else:
                #    ax.set_yticks(range(0, 62, 5))  # For intensity mm/h values

                # Use static x-axis bounds
                x_ticks: list[pd.Timestamp] = pd.date_range(
                    start=x_start, end=x_end, freq="24h"
                )
                format_time_axis(ax, x_start, x_end, x_ticks)

                if show_legend:
                    ax.legend(fontsize=14, loc="upper right")

                # Create spatial plot if requested and data is provided
                if (
                    spatial_ax is not None
                    and spatial_data is not None
                    and plot_type == "cumulative"
                ):
                    # Load region boundary for spatial plot
                    region = gpd.read_parquet(
                        Path(self.model.input_folder / "geom" / "mask.geoparquet")
                    )

                    # Use ensemble cumulative data for spatial plot
                    data_to_plot = spatial_data["ensemble_cumulative"]

                    # Create the spatial plot
                    im = data_to_plot.plot(
                        ax=spatial_ax,
                        cmap="viridis",
                        add_colorbar=False,
                        alpha=0.45,
                        zorder=1,
                    )

                    # Add region boundary
                    region.boundary.plot(
                        ax=spatial_ax, color="black", linestyle="-", linewidth=2
                    )
                    target_crs = region.crs
                    ctx.add_basemap(
                        spatial_ax,
                        crs=target_crs,
                        source="OpenStreetMap.Mapnik",
                        zoom=12,
                        zorder=0,
                    )

                    # Format the spatial plot
                    init_datetime = pd.to_datetime(
                        init_time_str, format="%Y%m%dT%H%M%S"
                    )
                    init_readable = init_datetime.strftime("%d %B %Y, %H:%M")

                    # Calculate lead time for spatial plot
                    if moment_of_flooding is not None:
                        spatial_lead_time_days: float = (
                            moment_of_flooding - init_datetime
                        ).total_seconds() / (24 * 3600)

                    spatial_ax.set_title(
                        f"Ensemble Spatial Mean Cumulative Rainfall\n{spatial_lead_time_days:.1f} day lead time\n(Initialized: {init_readable})",
                        fontsize=16,
                    )
                    spatial_ax.set_xlabel("Longitude", fontsize=14)
                    spatial_ax.set_ylabel("Latitude", fontsize=14)

                    # Add colorbar
                    cbar = plt.colorbar(im, ax=spatial_ax, shrink=0.8)
                    cbar.set_label("Rainfall [mm]", fontsize=16)

            # Main evaluation logic
            forecast_initialisations: list[str] = sorted(
                [item.name for item in forecast_base_path.iterdir() if item.is_dir()]
            )
            print(
                f"Found forecast initialisations (chronologically ordered): {forecast_initialisations}"
            )

            num_forecasts: int = len(forecast_initialisations)

            # Calculate grid dimensions - for cumulative plots: 2 columns (timeline + ensemble spatial), for intensity: 2 columns
            if plot_type == "cumulative":
                num_cols: int = 2  # Timeline grid: 2 columns
                timeline_rows: int = (
                    num_forecasts + num_cols - 1
                ) // num_cols  # Standard 2x2 grid for timelines
                num_rows: int = timeline_rows + 1  # Add one extra row for spatial plots
                fig_width = 20
            else:
                num_cols: int = 2  # Just timelines
                num_rows: int = (
                    num_forecasts + num_cols - 1
                ) // num_cols  # Standard grid calculation
                fig_width = 20

            # Create subplots
            if plot_type == "cumulative":
                # Create figure and subplots normally, then adjust spacing
                fig, axes = plt.subplots(
                    num_rows, num_cols, figsize=(fig_width, 6 * num_rows)
                )
                if num_rows == 1:
                    axes = axes.reshape(1, -1)
                elif num_cols == 1:
                    axes = axes.reshape(-1, 1)

                # Add extra spacing between timeline plots and spatial plots
                if num_rows > 1:  # Only if we have both timeline and spatial plots
                    timeline_rows = (num_forecasts + num_cols - 1) // num_cols
                    extra_gap = 3  # Extra gap between timeline and spatial plots

                    # Move spatial plots down to create extra gap
                    for col in range(num_cols):
                        if timeline_rows < num_rows:  # If spatial plots exist
                            spatial_ax = axes[-1, col]  # Bottom row (spatial plots)
                            pos = spatial_ax.get_position()
                            spatial_ax.set_position(
                                [
                                    pos.x0,
                                    pos.y0 - extra_gap,  # Move down
                                    pos.width,
                                    pos.height,
                                ]
                            )
            else:
                fig, axes = plt.subplots(
                    num_rows,
                    num_cols,
                    figsize=(fig_width, 4 * num_rows),
                    sharex=True,
                    sharey=True,
                )
            plt.rcParams.update(
                {
                    #'font.family': 'Times New Roman',
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "xtick.labelsize": 16,
                    "ytick.labelsize": 16,
                }
            )

            # Handle single subplot case
            if num_forecasts == 1:
                axes = [axes]

            for idx, init_time in enumerate(forecast_initialisations):
                print(
                    f"Processing forecast initialisation: {init_time}"
                )  # Log the iteration of forecasts

                init_folder: Path = forecast_base_path / init_time

                # Load data for this initialisation
                era5_data, control_data, time_data, ensemble_data = load_forecast_data(
                    init_folder, plot_type=plot_type
                )

                # Load spatial data for cumulative plots
                spatial_data = None
                if plot_type == "cumulative":
                    # Load full spatial data for ensemble cumulative plots
                    zarr_files = list(init_folder.glob("*.zarr"))
                    pr_files = [f for f in zarr_files if "pr" in f.name.lower()]

                    if pr_files:
                        # Ensemble spatial data
                        ens_ds = open_zarr(init_folder / pr_files[0].name)
                        ens_mm_per_h_spatial = ens_ds * 3600  # Convert to mm/h

                        # Calculate spatial cumulative rainfall (excluding control member 0)
                        ensemble_members = ens_mm_per_h_spatial.isel(
                            member=slice(1, None)
                        )
                        # Cumulative over time per gridcel, then final timestep, then max over members
                        ensemble_cumulative = (
                            ensemble_members.cumsum(dim="time")
                            .isel(time=-1)
                            .mean(dim="member")
                        )

                        spatial_data = {"ensemble_cumulative": ensemble_cumulative}

                if plot_type == "cumulative":
                    # For cumulative: 2x2 grid for timeline plots, spatial plots in extra row
                    row_idx = idx // 2
                    col_idx = idx % 2
                    timeline_ax = axes[row_idx, col_idx]
                    ensemble_spatial_ax = None
                    show_legend_flag = (
                        row_idx == 0 and col_idx == 1
                    )  # Top-right subplot

                    # Spatial plots only for first and last forecast (in bottom row)
                    if idx == 0:  # First forecast
                        ensemble_spatial_ax = axes[-1, 0]  # Bottom left
                    elif idx == num_forecasts - 1:  # Last forecast
                        ensemble_spatial_ax = axes[-1, 1]  # Bottom right
                else:
                    # For intensity: original 2-column layout
                    row_idx = idx // 2
                    col_idx = idx % 2
                    timeline_ax = (
                        axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]
                    )
                    ensemble_spatial_ax = None
                    show_legend_flag = row_idx == 0 and col_idx == 1
                # Plot timeline data
                plot_rainfall_data(
                    ax=timeline_ax,
                    era5_data=era5_data,
                    control_data=control_data,
                    ensemble_data=ensemble_data,
                    control_time=time_data,
                    init_time_str=init_time,
                    moment_of_flooding=moment_of_flooding,
                    plot_type=plot_type,
                    show_legend=show_legend_flag,
                    spatial_ax=ensemble_spatial_ax,
                    spatial_data=spatial_data,
                )

            # Hide unused subplots if odd number of forecasts
            for row in range(num_rows):
                for col in range(num_cols):
                    idx: int = row * num_cols + col

                    # Get current axis using same logic as above
                    if plot_type == "cumulative":
                        # For cumulative plots: 2x2 grid for timelines, spatial plots in last row
                        timeline_rows = (num_forecasts + num_cols - 1) // num_cols
                        if row < timeline_rows:
                            # Timeline plots in 2x2 grid
                            current_ax: plt.Axes = axes[row, col]
                        elif row == timeline_rows:
                            # Spatial plots in last row (only first two positions)
                            if col < 2:
                                current_ax: plt.Axes = axes[row, col]
                            else:
                                continue  # Skip extra columns in spatial row
                        else:
                            continue  # Skip if indices are out of bounds
                    else:
                        # For intensity plots: handle different axis configurations
                        if num_rows == 1 and num_cols == 1:
                            current_ax: plt.Axes = axes
                        elif num_rows == 1:
                            current_ax: plt.Axes = axes[col]
                        elif num_cols == 1:
                            current_ax: plt.Axes = axes[row]
                        else:
                            current_ax: plt.Axes = axes[row, col]

                    if plot_type == "cumulative":
                        # For cumulative plots: handle 2x2 timeline grid + spatial row
                        timeline_rows = (num_forecasts + num_cols - 1) // num_cols
                        timeline_idx = row * num_cols + col

                        if row < timeline_rows:
                            # Timeline plots in 2x2 grid
                            if timeline_idx >= num_forecasts:
                                # Hide unused timeline subplot
                                current_ax.set_visible(False)
                            else:
                                # Show x-axis labels only on bottom timeline row
                                is_bottom_timeline_row = row == timeline_rows - 1
                                subplot_below_is_empty = (
                                    row + 1
                                ) * num_cols + col >= num_forecasts

                                if is_bottom_timeline_row or subplot_below_is_empty:
                                    current_ax.tick_params(labelbottom=True)
                                    # Don't add individual x-axis labels - will use central label

                                # Show y-axis labels and label only on left middle subplot
                                if col == 0:
                                    current_ax.tick_params(labelleft=True)
                                    # Don't add individual y-axis labels - will use central label
                        elif row == timeline_rows:
                            # Spatial plots row
                            if col < 2:  # Only first two columns can have spatial plots
                                # Show labels for spatial plots
                                current_ax.tick_params(labelbottom=True, labelleft=True)
                    else:
                        # Original logic for intensity plots
                        if idx >= num_forecasts:
                            # Hide unused subplot
                            current_ax.set_visible(False)
                        else:
                            # Show x-axis labels on bottom row OR if the subplot below is empty
                            subplot_below_is_empty: bool = (
                                row + 1
                            ) * num_cols + col >= num_forecasts
                            is_bottom_row: bool = row == num_rows - 1

                            if is_bottom_row or subplot_below_is_empty:
                                current_ax.tick_params(labelbottom=True)

                            # Show y-axis labels on leftmost column
                            if col == 0:
                                current_ax.tick_params(labelleft=True)
            # Final plot formatting
            if plot_type == "cumulative":
                plot_title: str = "ERA5 vs ECMWF Control & Probabilistic Cumulative Precipitation Forecasts"
                output_filename: str = "ERA5_vs_ECMWF_Control_&_Probabilistic_Cumulative_Precipitation_Forecasts.png"

                # Add central axis labels for timeline plots only
                timeline_rows = (num_forecasts + num_cols - 1) // num_cols

                # Calculate position for timeline plots (excluding spatial row)
                timeline_height = timeline_rows * 6  # Each timeline row is 6 units high
                total_height = num_rows * 6  # Total figure height units
                spatial_height = 6  # Spatial row height

                # Position labels for timeline plots
                timeline_center_x = 0.5  # Center horizontally
                timeline_center_y = (
                    spatial_height + timeline_height / 2
                ) / total_height  # Center of timeline area vertically

                # Y-position for x-axis label: at bottom of timeline area (above spatial row)
                timeline_bottom_y = spatial_height / total_height

                fig.text(
                    timeline_center_x,
                    timeline_bottom_y - 0.03,
                    "Date (UTC)",
                    ha="center",
                    va="top",
                    fontsize=16,
                    fontweight="bold",
                )
                fig.text(
                    0.02,
                    timeline_center_y,
                    "Cumulative Rainfall [mm]",
                    va="center",
                    rotation="vertical",
                    fontsize=16,
                    fontweight="bold",
                )

            else:
                plot_title: str = "ERA5 vs ECMWF Control & Probabilistic Maximum Intensity Precipitation Forecasts"
                output_filename: str = "ERA5_vs_ECMWF_Control_&_Probabilistic_Precipitation_Forecasts_Maximum_Intensity.png"
                ylabel: str = "Rainfall intensity [mm/h]"
                # Set global axis labels for intensity plots (original behavior)
                fig.text(0.5, 0.02, "Date (UTC)", ha="center", fontsize=18)
                fig.text(
                    0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=18
                )

            fig.suptitle(plot_title, fontsize=22, y=0.95)

            if plot_type == "cumulative":
                plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
            else:
                plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])

            # Save plot
            plt.savefig(
                forecast_folder / output_filename, dpi=1000, bbox_inches="tight"
            )
            plt.close()

        def issue_rainfall_warning(
            processing_forecasts: str,
            moment_of_flooding: pd.Timestamp | None = None,
            threshold_time_unit: str = "daily",
            daily_cumulative_thresholds_mm_per_day: dict[str, float] = {
                "yellow": 50.0,  # 50 mm/day threshold
                "orange": 100.0,  # 100 mm/day threshold
                "red": 150.0,  # High cumulative threshold
            },
            ensemble_probability_thresholds: dict[str, float] = {
                "yellow": 0.15,  # 15% of ensemble members exceed threshold
                "orange": 0.30,  # 30% of ensemble members exceed threshold
                "red": 0.45,  # 45% of ensemble members exceed threshold
                "purple": 0.60,  # 60% of ensemble members exceed threshold
            },
        ) -> None:
            """Generate rainfall warning timeline showing warning evolution across forecast initializations.

            Creates timeline visualizations showing how warnings evolve from forecast initializations
            to the actual event, comparing ERA5 reanalysis, deterministic control forecasts, and
            probabilistic ensemble forecasts. The event datetime is automatically extracted from the
            first event in hazards.floods.events configuration.

            Args:
                processing_forecasts: The type of forecasts being processed.
                moment_of_flooding: Optional moment when flooding actually occurred. If None, no flooding line is shown.
                threshold_time_unit: Time unit for thresholds ("daily" or "hourly").
                daily_cumulative_thresholds_mm_per_day: Warning thresholds for rainfall per time unit.
                ensemble_probability_thresholds: Probability thresholds for ensemble warnings.

            Raises:
                ValueError: If no flood events are configured or event end_time format is invalid.
                FileNotFoundError: If required forecast or ERA5 data files are missing.
                ValueError: If no precipitation forecast file is found for a given initialization.
                ValueError: If threshold_time_unit is not "daily" or "hourly".
            """
            print("Issuing rainfall warnings based on forecasts...")

            # Validate threshold time unit
            if threshold_time_unit not in ["daily", "hourly"]:
                raise ValueError(
                    f"threshold_time_unit must be 'daily' or 'hourly', got '{threshold_time_unit}'"
                )

            def create_binary_threshold_arrays(
                data_array: xarray.DataArray,
                thresholds: dict[str, float],
                time_unit: str,
            ) -> dict[str, xarray.DataArray]:
                """Create binary arrays where 1 indicates threshold exceedance.

                Args:
                    data_array: Precipitation data (mm/h for input, processed based on time_unit).
                    thresholds: Dictionary with threshold values.
                    time_unit: Time unit for thresholds ("daily" or "hourly").

                Returns:
                    Dictionary of binary arrays for each threshold level.

                Raises:
                    ValueError: If time_unit is not "daily" or "hourly".
                """
                if time_unit == "daily":
                    # For daily thresholds, resample hourly data to daily cumulative precipitation (mm/day)
                    processed_data = data_array.resample(time="1D").sum()
                elif time_unit == "hourly":
                    # For hourly thresholds, use hourly intensity directly (mm/h)
                    processed_data = data_array.copy()
                else:
                    raise ValueError(f"Unsupported time_unit: {time_unit}")

                binary_arrays = {}
                # Create independent binary threshold arrays
                # Each threshold is evaluated separately - a grid cell can exceed multiple thresholds
                for level_name, threshold_value in thresholds.items():
                    # Create binary array: 1 where threshold exceeded, 0 otherwise
                    # Create boolean mask and convert to int, ensuring no NaN values remain
                    threshold_mask = processed_data >= threshold_value
                    binary_arrays[level_name] = threshold_mask.fillna(False).astype(int)

                return binary_arrays

            def create_deterministic_warning_timeseries(
                binary_arrays: dict[str, xarray.DataArray],
                data_array: xarray.DataArray,
            ) -> xarray.DataArray:
                """Create warning level time series for deterministic forecasts (ERA5, control).

                Uses binary threshold exceedance arrays to assign warning levels to each grid cell
                and timestep. Higher warning levels override lower ones when multiple thresholds
                are exceeded.

                Args:
                    binary_arrays: Binary threshold exceedance arrays for each warning level.
                    data_array: Precipitation data for spatial structure reference (processed based on time unit).

                Returns:
                    DataArray with warning levels (0-3) for each grid cell and timestep.
                        0 = No warning, 1 = Yellow, 2 = Orange, 3 = Red.

                Raises:
                    ValueError: If binary_arrays is empty or data_array contains only NaN values.
                """
                if not binary_arrays:
                    raise ValueError("binary_arrays cannot be empty")

                # Initialize warnings array with zeros, preserving spatial and temporal structure
                clean_data = data_array.fillna(0)
                if clean_data.size == 0 or clean_data.isnull().all():
                    raise ValueError(
                        "data_array contains only NaN values after cleaning"
                    )

                warnings_array = xarray.zeros_like(clean_data, dtype=int)

                # Convert binary arrays to final warning levels using "highest wins" logic
                # Unlike the binary arrays, each grid cell gets exactly ONE warning level (0-3)
                # If multiple thresholds exceeded, highest level overwrites lower ones
                # Example: yellow=1, orange=1, red=0 → final warning = 2 (orange)
                level_mapping = {"yellow": 1, "orange": 2, "red": 3}

                for level_name, level_value in level_mapping.items():
                    if level_name in binary_arrays:
                        # Set warning level where threshold exceeded - higher levels overwrite lower ones
                        exceeded_mask = binary_arrays[level_name] == 1
                        warnings_array = warnings_array.where(
                            ~exceeded_mask, level_value
                        )

                # Ensure no NaN values remain before final conversion
                warnings_array = warnings_array.fillna(0)
                return warnings_array.astype(int)

            def create_ensemble_warning_timeseries(
                ensemble_data: xarray.DataArray,
                thresholds: dict[str, float],
                probability_thresholds: dict[str, float],
                time_unit: str,
            ) -> xarray.DataArray:
                """Create warning level time series for ensemble forecasts with probability thresholds.

                Uses probabilistic approach where warnings are issued when a certain percentage
                of ensemble members exceed precipitation thresholds. Purple level uses red threshold
                but with higher probability requirement.

                Args:
                    ensemble_data: Precipitation ensemble data with member dimension (mm/h input).
                        Will be processed based on time_unit: resampled to mm/day for daily, kept as mm/h for hourly.
                    thresholds: Dictionary with precipitation threshold values.
                        Expected keys: yellow, orange, red.
                    probability_thresholds: Probability thresholds for ensemble warnings.
                        Expected keys: yellow, orange, red, purple.
                    time_unit: Time unit for thresholds ("daily" or "hourly").

                Returns:
                    DataArray with warning levels (0-4) for each grid cell and timestep.
                        0 = No warning, 1 = Yellow, 2 = Orange, 3 = Red, 4 = Purple.

                Raises:
                    ValueError: If ensemble_data lacks member dimension or contains only NaN values.
                    ValueError: If thresholds or probability_thresholds are empty.
                    ValueError: If time_unit is not "daily" or "hourly".
                """
                # Input validation
                if not thresholds:
                    raise ValueError("thresholds dictionary cannot be empty")
                if not probability_thresholds:
                    raise ValueError(
                        "probability_thresholds dictionary cannot be empty"
                    )
                if "member" not in ensemble_data.dims:
                    raise ValueError("ensemble_data must have 'member' dimension")
                if time_unit not in ["daily", "hourly"]:
                    raise ValueError(f"Unsupported time_unit: {time_unit}")

                # Process data based on time unit
                if time_unit == "daily":
                    # For daily thresholds, resample hourly ensemble data to daily cumulative precipitation (mm/day)
                    processed_data = ensemble_data.resample(time="1D").sum()
                elif time_unit == "hourly":
                    # For hourly thresholds, use hourly intensity directly (mm/h)
                    processed_data = ensemble_data.copy()

                # Initialize warnings array (no member dimension in output)
                clean_ensemble_data = processed_data.fillna(0)
                if clean_ensemble_data.size == 0 or clean_ensemble_data.isnull().all():
                    raise ValueError(
                        "ensemble_data contains only NaN values after cleaning"
                    )

                # Create warnings array efficiently
                warnings_array = xarray.zeros_like(
                    clean_ensemble_data.isel(member=0), dtype=int
                )

                # Process warning levels with probability thresholds
                # Define hierarchy: yellow (1) < orange (2) < red (3) for "highest wins" logic
                level_mapping = {"yellow": 1, "orange": 2, "red": 3}

                # Iterate through each warning level in hierarchical order
                for level_name, level_value in level_mapping.items():
                    # Check if this warning level is configured in both dictionaries
                    if (
                        level_name in thresholds
                        and level_name in probability_thresholds
                    ):
                        # Create binary threshold exceedance array for this warning level
                        # Compare ensemble data against precipitation threshold (e.g., 50mm for yellow)
                        # Result: 4D boolean array [time, y, x, member] where True = threshold exceeded
                        threshold_exceeded = (
                            (clean_ensemble_data >= thresholds[level_name])
                            .fillna(False)
                            .astype(int)
                        )  # Handle NaN and convert True/False to 1/0

                        # Calculate exceedance probability across ensemble members
                        # Mean over 'member' dimension gives probability (0.0 to 1.0) per grid cell per timestep
                        # Example: 5 of 20 members exceed threshold → 0.25 (25% probability)
                        exceedance_probability = threshold_exceeded.mean(dim="member")

                        # Get minimum required probability for this warning level
                        # Example: yellow needs 15%, orange needs 30%, red needs 45% of members
                        prob_threshold = probability_thresholds[level_name]

                        # Create mask where probability threshold is exceeded
                        # True where enough ensemble members exceed the precipitation threshold
                        prob_exceeded_mask = exceedance_probability >= prob_threshold

                        # Apply "highest wins" logic - set warning level where criteria met
                        # Higher warning levels (processed later) will overwrite lower ones
                        # Example: grid cell with yellow=True, orange=True → final result = orange (2)
                        warnings_array = warnings_array.where(
                            ~prob_exceeded_mask, level_value
                        )

                # Handle purple level separately (uses red threshold with higher probability)
                if "purple" in probability_thresholds and "red" in thresholds:
                    threshold_exceeded = (
                        (clean_ensemble_data >= thresholds["red"])
                        .fillna(False)
                        .astype(int)
                    )
                    exceedance_probability = threshold_exceeded.mean(dim="member")

                    purple_prob_threshold = probability_thresholds["purple"]
                    purple_exceeded_mask = (
                        exceedance_probability >= purple_prob_threshold
                    )

                    # Purple (level 4) overwrites all other warning levels
                    warnings_array = warnings_array.where(~purple_exceeded_mask, 4)
                return warnings_array.astype(int)

            def save_warning_timeseries(
                output_folder: Path,
                forecast_init: str,
                era5_warnings: xarray.DataArray,
                control_warnings: xarray.DataArray,
                ensemble_warnings: xarray.DataArray,
            ) -> None:
                """Save warning time series as zarr files for further analysis.

                Args:
                    output_folder: Path to output folder.
                    forecast_init: Forecast initialization time string.
                    era5_warnings: ERA5 warning time series.
                    control_warnings: Control forecast warning time series.
                    ensemble_warnings: Ensemble forecast warning time series.
                """
                # Create subfolder for this forecast initialization
                init_folder = output_folder / f"timeseries_{forecast_init}"
                init_folder.mkdir(exist_ok=True)

                # Define base clean attributes to avoid NaN conversion issues
                base_attrs = {
                    "description": "Rainfall warning levels per grid cell and timestep",
                    "forecast_init": forecast_init,
                    "warning_levels": "0=No Warning, 1=Yellow, 2=Orange, 3=Red, 4=Purple",
                    "units": "warning_level",
                    "created_at": str(pd.Timestamp.now()),
                }

                # Save ERA5 warnings with clean attributes
                era5_clean = era5_warnings.copy()
                era5_clean.attrs.clear()
                era5_clean.attrs.update(base_attrs)
                era5_clean.attrs["source"] = "ERA5 reanalysis"
                era5_clean.attrs["method"] = "deterministic"
                # Force computation to avoid chunk inheritance from previous runs
                era5_clean = era5_clean.compute()
                era5_clean.to_zarr(init_folder / "era5_warnings.zarr", mode="w")

                # Save control warnings with clean attributes
                control_clean = control_warnings.copy()
                control_clean.attrs.clear()
                control_clean.attrs.update(base_attrs)
                control_clean.attrs["source"] = "ECMWF control forecast"
                control_clean.attrs["method"] = "deterministic"
                # Force computation to avoid chunk inheritance from previous runs
                control_clean = control_clean.compute()
                control_clean.to_zarr(init_folder / "control_warnings.zarr", mode="w")

                # Save ensemble warnings with clean attributes
                ensemble_clean = ensemble_warnings.copy()
                ensemble_clean.attrs.clear()
                ensemble_clean.attrs.update(base_attrs)
                ensemble_clean.attrs["source"] = "ECMWF ensemble forecast"
                ensemble_clean.attrs["method"] = "probabilistic"
                # Force computation to avoid chunk inheritance from previous runs
                ensemble_clean = ensemble_clean.compute()
                ensemble_clean.to_zarr(init_folder / "ensemble_warnings.zarr", mode="w")

                print(f"Saved warning time series for {forecast_init} to {init_folder}")

            def create_spatial_warning_maps(
                forecast_lead_times_days: list[int], moment_of_flooding: pd.Timestamp
            ) -> None:
                """Create comprehensive spatial warning maps including maximum warning levels.

                Creates a single plot with lead times as columns and warning types (ERA5, Control, Ensemble) as rows,
                showing the highest warning level reached during the 10-day forecast period for different lead times.

                Args:
                    forecast_lead_times_days: Lead times (days before event) to analyze.
                    moment_of_flooding: The moment when flooding actually occurred.
                Raises:
                    ValueError: If no suitable forecast is found for a given lead time.
                """
                # Define warning colormap
                warning_colors: list[str] = [
                    "white",
                    "yellow",
                    "orange",
                    "red",
                    "purple",
                ]
                warning_cmap = mcolors.ListedColormap(warning_colors)
                warning_labels: list[str] = [
                    "No Warning",
                    "Yellow",
                    "Orange",
                    "Red",
                    "Purple",
                ]

                region = gpd.read_parquet(
                    Path(self.model.input_folder / "geom" / "mask.geoparquet")
                )

                print("Creating Maximum Warning Level Maps for all lead times...")

                # Collect data for all lead times
                lead_time_data = {}

                for target_lead in forecast_lead_times_days:
                    print(f"Processing {target_lead}-day lead time...")

                    # Find forecast closest to target lead time
                    best_forecast = None
                    best_diff = float("inf")
                    actual_lead = None

                    for forecast_init in available_forecasts:
                        init_datetime = datetime.strptime(
                            forecast_init, "%Y%m%dT%H%M%S"
                        )

                        lead_time = (event_datetime - init_datetime).total_seconds() / (
                            24 * 3600
                        )
                        diff = abs(lead_time - target_lead)

                        if diff < best_diff:
                            best_diff = diff
                            best_forecast = forecast_init
                            actual_lead = lead_time

                    if best_forecast is None:
                        raise ValueError(
                            f"No suitable forecast found for {target_lead}-day lead time."
                        )

                    init_time_str = best_forecast
                    timeseries_folder = warnings_folder / f"timeseries_{init_time_str}"

                    if not timeseries_folder.exists():
                        raise ValueError(
                            f"Warning time series not found for {init_time_str}"
                        )

                    # Load warning time series data
                    era5_warnings = open_zarr(timeseries_folder / "era5_warnings.zarr")
                    control_warnings = open_zarr(
                        timeseries_folder / "control_warnings.zarr"
                    )
                    ensemble_warnings = open_zarr(
                        timeseries_folder / "ensemble_warnings.zarr"
                    )
                    target_crs = era5_warnings.rio.crs
                    # Calculate maximum warning level over 10-day forecast period for each grid cell
                    era5_max_warning = era5_warnings.max(dim="time")
                    control_max_warning = control_warnings.max(dim="time")
                    ensemble_max_warning = ensemble_warnings.max(dim="time")

                    # Store data for this lead time
                    lead_time_data[target_lead] = {
                        "era5": era5_max_warning,
                        "control": control_max_warning,
                        "ensemble": ensemble_max_warning,
                        "actual_lead": actual_lead,
                        "init_time": init_time_str,
                    }

                # Create combined plot: rows = warning types, columns = lead times
                num_rows = 3  # ERA5, Control, Ensemble
                num_cols = len(forecast_lead_times_days)

                fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 15))

                # Handle case of single column
                if num_cols == 1:
                    axes = axes.reshape(-1, 1)

                warning_type_names = ["ERA5", "Control Forecast", "Ensemble Forecast"]
                warning_type_keys = ["era5", "control", "ensemble"]

                # Plot each combination of warning type and lead time
                for row_idx, (type_name, type_key) in enumerate(
                    zip(warning_type_names, warning_type_keys)
                ):
                    for col_idx, target_lead in enumerate(forecast_lead_times_days):
                        ax = axes[row_idx, col_idx]

                        # Get data for this lead time and warning type
                        data = lead_time_data[target_lead][type_key]
                        actual_lead = lead_time_data[target_lead]["actual_lead"]
                        init_time = lead_time_data[target_lead]["init_time"]

                        # Plot the maximum warning levels
                        im = data.plot(
                            ax=ax,
                            cmap=warning_cmap,
                            vmin=0,
                            vmax=4,
                            add_colorbar=False,
                            alpha=0.5,
                            zorder=1,
                        )
                        region.boundary.plot(
                            ax=ax,
                            color="black",
                            linestyle="--",
                            linewidth=1.5,
                            zorder=2,
                            alpha=0.5,
                        )
                        ctx.add_basemap(
                            ax,
                            crs=target_crs,
                            source="OpenStreetMap.Mapnik",
                            zoom=12,
                            zorder=0,
                        )

                        # Set titles
                        if row_idx == 0:  # Top row gets lead time info
                            ax.set_title(
                                f"{target_lead}-day lead time\n{type_name}\n({init_time} forecast)",
                                fontsize=12,
                            )
                        else:
                            ax.set_title(f"{type_name}", fontsize=12)

                        # Set axis labels
                        if row_idx == num_rows - 1:  # Bottom row
                            ax.set_xlabel("Longitude")
                        else:
                            ax.set_xlabel("")

                        if col_idx == 0:  # Left column
                            ax.set_ylabel("Latitude")
                        else:
                            ax.set_ylabel("")

                # Add single colorbar for all subplots positioned on the right side
                cbar = fig.colorbar(
                    im,
                    ax=axes.ravel().tolist(),
                    shrink=0.8,
                    aspect=30,
                    pad=0.02,
                    fraction=0.05,
                    location="right",
                )
                cbar.set_ticks(range(5))
                cbar.set_ticklabels(warning_labels)
                cbar.set_label("Maximum Warning Level", fontsize=14)

                # Overall title
                plt.suptitle(
                    f"Maximum Meteorological Warning Levels Over 10-Day Period\nComparison Across Lead Times and Forecast Types",
                    fontsize=16,
                    y=0.98,
                )

                # plt.tight_layout()

                # Save the combined plot
                output_filename = "maximum_warnings_all_lead_times.png"
                plt.savefig(
                    warnings_folder / output_filename, dpi=300, bbox_inches="tight"
                )
                plt.close()
                print(f"Saved combined maximum warning level map: {output_filename}")

                # Create comprehensive consolidated timeline showing all information in one figure
                print("Creating comprehensive consolidated warning timeline...")

                # Prepare dictionaries with full time series data for timeline plotting
                era5_warnings_dict = {}
                control_warnings_dict = {}
                ensemble_warnings_dict = {}

                for target_lead in forecast_lead_times_days:
                    if target_lead in lead_time_data:
                        init_time_str = lead_time_data[target_lead]["init_time"]
                        timeseries_folder = (
                            warnings_folder / f"timeseries_{init_time_str}"
                        )

                        # Load full time series (not just maximum)
                        era5_warnings_full = open_zarr(
                            timeseries_folder / "era5_warnings.zarr"
                        )
                        control_warnings_full = open_zarr(
                            timeseries_folder / "control_warnings.zarr"
                        )
                        ensemble_warnings_full = open_zarr(
                            timeseries_folder / "ensemble_warnings.zarr"
                        )

                        # Store in dictionaries with lead time as key
                        lead_time_key = f"{target_lead}day"
                        era5_warnings_dict[lead_time_key] = era5_warnings_full
                        control_warnings_dict[lead_time_key] = control_warnings_full
                        ensemble_warnings_dict[lead_time_key] = ensemble_warnings_full

                # Enhanced color mapping for warning levels and forecast types
                warning_level_colors = {
                    0: "#90EE90",  # Light green (No warning)
                    1: "#FFD700",  # Gold (Yellow warning)
                    2: "#FF8C00",  # Dark orange (Orange warning)
                    3: "#FF0000",  # Red (Red warning)
                    4: "#8B008B",  # Dark magenta (Purple warning)
                }

                warning_labels = {
                    0: "No Warning",
                    1: "Yellow Warning",
                    2: "Orange Warning",
                    3: "Red Warning",
                    4: "Purple Warning",
                }

                # Enhanced forecast type styling with line styles for lead time distinction
                base_forecast_colors = {
                    "ERA5": "#2E2E2E",  # Dark gray base
                    "Control": "#1E90FF",  # DodgerBlue base
                    "Ensemble": "#9400D3",  # DarkViolet base
                }

                # Different line styles for each lead time (professional distinction)
                lead_time_styles = {
                    0: {
                        "linestyle": "-",
                        "linewidth": 3.0,
                        "alpha": 0.9,
                    },  # Solid line (longest lead)
                    1: {
                        "linestyle": "--",
                        "linewidth": 2.8,
                        "alpha": 0.85,
                    },  # Dashed line
                    2: {
                        "linestyle": "-.",
                        "linewidth": 2.6,
                        "alpha": 0.8,
                    },  # Dash-dot line
                    3: {
                        "linestyle": ":",
                        "linewidth": 2.4,
                        "alpha": 0.75,
                    },  # Dotted line (shortest lead)
                }

                forecast_styles = {
                    "ERA5": {
                        "marker": "s",
                        "markersize": 7,
                    },
                    "Control": {
                        "marker": "o",
                        "markersize": 6,
                    },
                    "Ensemble": {
                        "marker": "^",
                        "markersize": 6,
                    },
                }

                # Create single comprehensive timeline plot with simplified layout (no colorbar)
                fig, ax_main = plt.subplots(figsize=(20, 10))

                # Define small y-offset for each forecast initialization (0.05 units per lead time)
                y_offset_per_lead = 0.05

                plt.rcParams.update(
                    {
                        "axes.titlesize": 16,
                        "axes.labelsize": 14,
                        "xtick.labelsize": 12,
                        "ytick.labelsize": 12,
                        "legend.fontsize": 11,
                    }
                )

                # Validate that we have data for at least one lead time
                valid_lead_times = []
                for lead_time in forecast_lead_times_days:
                    lead_time_str = f"{lead_time}day"
                    if lead_time_str in era5_warnings_dict:
                        valid_lead_times.append(lead_time)

                if not valid_lead_times:
                    print("Error: No valid lead time data found for plotting")
                    return

                print(f"Creating timeline for lead times: {valid_lead_times}")
                total_lead_times = len(valid_lead_times)

                # Process and plot all lead times and forecast types on main plot
                legend_handles = []
                legend_labels = []

                # Store forecast initialization dates for legend
                lead_time_init_dates = {}
                for lead_time in valid_lead_times:
                    lead_time_str = f"{lead_time}day"
                    if lead_time_str in era5_warnings_dict:
                        # Get initialization time from the first available forecast for this lead time
                        for target_lead in forecast_lead_times_days:
                            if (
                                target_lead == lead_time
                                and target_lead in lead_time_data
                            ):
                                init_time_str = lead_time_data[target_lead]["init_time"]
                                init_datetime = datetime.strptime(
                                    init_time_str, "%Y%m%dT%H%M%S"
                                )
                                formatted_date = init_datetime.strftime(
                                    "%d-%m-%Y %H:%M"
                                )
                                lead_time_init_dates[lead_time] = formatted_date
                                break

                # Create legend handles for forecast types (one per type, showing base color)
                for fc_type, base_color in base_forecast_colors.items():
                    style = forecast_styles[fc_type]
                    # Create dummy scatter for legend using base color
                    dummy_scatter = ax_main.scatter(
                        [],
                        [],
                        color=base_color,
                        marker=style["marker"],
                        s=120,
                        alpha=0.9,
                        edgecolor="white",
                        linewidth=2,
                        label=f"{fc_type} Forecast",
                    )
                    legend_handles.append(dummy_scatter)
                    legend_labels.append(f"{fc_type} Forecast")

                # Add separator line in legend
                separator_line = plt.Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=0,
                    label="Lead Times & Init Dates:",
                )
                legend_handles.append(separator_line)
                legend_labels.append("Lead Times & Init Dates:")

                # Add lead time examples to legend with line styles and initialization dates
                for lead_idx, lead_time in enumerate(valid_lead_times):
                    if lead_idx < len(lead_time_styles):
                        line_style = lead_time_styles[lead_idx]
                        init_date = lead_time_init_dates.get(lead_time, "Unknown")

                        # Create line example for legend
                        lead_line = plt.Line2D(
                            [0],
                            [0],
                            color=base_forecast_colors[
                                "ERA5"
                            ],  # Use ERA5 color as neutral example
                            linestyle=line_style["linestyle"],
                            linewidth=line_style["linewidth"],
                            alpha=line_style["alpha"],
                            label=f"{lead_time}d lead ({init_date})",
                        )
                        legend_handles.append(lead_line)
                        legend_labels.append(f"{lead_time}d lead ({init_date})")

                # Main plotting loop for all lead times
                for lead_idx, lead_time in enumerate(valid_lead_times):
                    lead_time_str = f"{lead_time}day"

                    # Load warning data for this lead time
                    era5_warnings = era5_warnings_dict[lead_time_str]
                    control_warnings = control_warnings_dict[lead_time_str]
                    ensemble_warnings = ensemble_warnings_dict[lead_time_str]

                    # Calculate domain-maximum warning level at each timestep
                    era5_max_warning = era5_warnings.max(dim=["y", "x"])
                    control_max_warning = control_warnings.max(dim=["y", "x"])
                    ensemble_max_warning = ensemble_warnings.max(dim=["y", "x"])

                    # Get time coordinates
                    time_coords = era5_warnings.time.values

                    # Validate data before plotting
                    if len(time_coords) == 0:
                        print(
                            f"Warning: No time coordinates found for lead time {lead_time}"
                        )
                        continue

                    # Calculate y-offset for this lead time (small offset to distinguish forecasts)
                    lead_y_offset = lead_idx * y_offset_per_lead

                    # Plot each forecast type with gradient colors and y-offset
                    forecast_data = {
                        "ERA5": era5_max_warning.values,
                        "Control": control_max_warning.values,
                        "Ensemble": ensemble_max_warning.values,
                    }

                    # Validate each forecast data array
                    valid_forecast_data = {}
                    for fc_type, values in forecast_data.items():
                        if (
                            values is not None
                            and len(values) > 0
                            and not np.all(np.isnan(values))
                        ):
                            valid_forecast_data[fc_type] = values
                        else:
                            print(
                                f"Warning: Invalid data for {fc_type} at lead time {lead_time}"
                            )

                    forecast_data = valid_forecast_data

                    for fc_type, warning_values in forecast_data.items():
                        style = forecast_styles[fc_type]
                        base_color = base_forecast_colors[fc_type]

                        # Get line style for this lead time
                        if lead_idx < len(lead_time_styles):
                            line_style = lead_time_styles[lead_idx]
                        else:
                            line_style = lead_time_styles[0]  # Fallback to solid line

                        # Apply y-offset to warning values (small shift to distinguish different forecasts)
                        warning_values_offset = warning_values + lead_y_offset

                        # Plot connected line with markers
                        ax_main.plot(
                            time_coords,
                            warning_values_offset,
                            color=base_color,
                            linestyle=line_style["linestyle"],
                            linewidth=line_style["linewidth"],
                            alpha=line_style["alpha"],
                            marker=style["marker"],
                            markersize=style["markersize"],
                            markerfacecolor=base_color,
                            markeredgecolor="white",
                            markeredgewidth=1.5,
                            zorder=15,
                        )

                    # Add lead time labels on the right side with line style indicators
                    if lead_idx < len(lead_time_styles):
                        line_style = lead_time_styles[lead_idx]
                        # Create line style indicator text
                        style_indicators = {
                            "-": "─────",
                            "--": "─ ─ ─",
                            "-.": "─ · ─",
                            ":": "· · · ·",
                        }
                        line_indicator = style_indicators.get(
                            line_style["linestyle"], "─────"
                        )

                        # Use transAxes for reliable positioning
                        ax_main.text(
                            0.98,  # Stay within plot bounds
                            0.85 - (lead_idx * 0.1),  # Stagger labels vertically
                            f"{lead_time}d {line_indicator}",
                            transform=ax_main.transAxes,
                            fontsize=11,
                            fontweight="bold",
                            ha="right",
                            va="center",
                            color="black",
                            bbox=dict(
                                boxstyle="round,pad=0.4",
                                facecolor="lightblue",
                                alpha=0.8,
                                edgecolor="navy",
                                linewidth=1.5,
                            ),
                        )

                # Add lead time information as text annotation
                lead_times_text = "Lead Times: " + ", ".join(
                    [f"{lt}d" for lt in forecast_lead_times_days]
                )
                ax_main.text(
                    0.02,
                    0.98,
                    lead_times_text,
                    transform=ax_main.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    ha="left",
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="lightyellow",
                        alpha=0.9,
                        edgecolor="orange",
                    ),
                )

                # Add flooding moment to main plot if present
                if moment_of_flooding is not None:
                    flood_line = ax_main.axvline(
                        moment_of_flooding,
                        color="#FF0000",
                        linestyle="--",
                        linewidth=3,
                        alpha=0.8,
                        zorder=20,
                        label="Moment of Flooding",
                    )
                    legend_handles.append(flood_line)
                    legend_labels.append("Moment of Flooding")

                # Add legend directly to main plot
                legend = ax_main.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper left",
                    fontsize=11,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    ncol=1,
                    title="Forecast Types & Lead Times",
                    title_fontsize=12,
                    bbox_to_anchor=(0.02, 0.85),
                )
                legend.get_frame().set_facecolor("white")
                legend.get_frame().set_alpha(0.95)
                legend.get_title().set_fontweight("bold")

                # Enhanced y-axis formatting for main plot
                ax_main.set_ylim(-0.5, 4.5)

                # Set y-ticks at warning levels
                y_ticks = []
                y_tick_labels = []
                for level in range(5):
                    y_ticks.append(level)
                    y_tick_labels.append(warning_labels[level])

                ax_main.set_yticks(y_ticks)
                ax_main.set_yticklabels(y_tick_labels)

                # Color the y-axis tick labels according to warning colors
                for tick, level in zip(ax_main.get_yticklabels(), range(5)):
                    tick.set_color(warning_level_colors[level])
                    tick.set_fontweight("bold")
                    tick.set_fontsize(12)

                # Add subtle background colors for warning levels (removed colorbar blocks)
                for level in range(5):
                    ax_main.axhspan(
                        level - 0.4,
                        level + 0.4,
                        color=warning_level_colors[level],
                        alpha=0.1,
                        zorder=0,
                    )
                # Enhanced grid and formatting for main plot
                ax_main.grid(True, alpha=0.3, zorder=1)
                ax_main.set_title(
                    "Meteorological Warning Timeline: Domain Maximum Warning Levels\n"
                    "Different Colors Show Different Lead Times for Each Forecast Type",
                    fontsize=18,
                    fontweight="bold",
                    pad=20,
                )
                ax_main.set_ylabel("Warning Level", fontsize=14, fontweight="bold")

                # Enhanced time axis formatting with daily ticks
                ax_main.xaxis.set_major_locator(mdates.DayLocator())
                ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
                ax_main.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
                ax_main.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))

                # Rotate labels for better readability
                ax_main.tick_params(axis="x", which="major", rotation=45, labelsize=11)
                ax_main.tick_params(axis="x", which="minor", rotation=45, labelsize=9)
                ax_main.set_xlabel("Date (UTC)", fontsize=14, fontweight="bold")

                # Final layout adjustments with simplified layout
                try:
                    plt.tight_layout(pad=2.0)

                    # Save the comprehensive plot with error handling
                    comprehensive_filename = (
                        "Comprehensive_Warning_Timeline_All_Data.png"
                    )
                    output_path = warnings_folder / comprehensive_filename

                    # Ensure output directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    plt.savefig(
                        output_path,
                        dpi=300,
                        bbox_inches="tight",
                        facecolor="white",
                        edgecolor="none",
                        format="png",
                        pad_inches=0.1,
                    )
                    print(f"Successfully saved plot to: {output_path}")

                except Exception as e:
                    print(f"Error saving plot: {e}")

                plt.close()

                print(
                    f"Comprehensive warning timeline saved as: {comprehensive_filename}"
                )
                print("Comprehensive warning timeline plot completed successfully!")

            # Step 1: Initialize output directory for warning results
            warnings_folder: Path = (
                self.output_folder_evaluate / "forecasts" / "rainfall_warnings"
            )
            warnings_folder.mkdir(parents=True, exist_ok=True)

            # Step 2: Extract flood event datetime from model configuration
            flood_events = (
                self.model.config.get("hazards", {}).get("floods", {}).get("events", [])
            )
            if not flood_events:
                raise ValueError(
                    "No flood events found in model configuration. "
                    "Please define events under hazards.floods.events with end_time."
                )

            # Use the end_time from the first event
            first_event = flood_events[0]
            if "end_time" not in first_event:
                raise ValueError(
                    "Flood event must have an 'end_time' field in the configuration."
                )

            try:
                # Parse the end_time which should be in ISO format like '2021-07-20 09:00:00'
                event_datetime: datetime = datetime.fromisoformat(
                    str(first_event["end_time"]).replace(" ", "T")
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid event end_time format: {first_event['end_time']}. "
                    f"Expected ISO format like '2021-07-20 09:00:00'. Error: {e}"
                )

            # Step 3: Get all available forecast initializations and sort chronologically
            forecast_base_path: Path = (
                self.model.input_folder
                / "other"
                / "forecasts"
                / "ECMWF"
                / processing_forecasts
            )

            if not forecast_base_path.exists():
                raise FileNotFoundError(
                    f"Forecast path does not exist: {forecast_base_path}"
                )

            available_forecasts: list[str] = [
                item.name
                for item in forecast_base_path.iterdir()
                if item.is_dir() and len(item.name) == 15  # YYYYMMDDTHHMMSS format
            ]
            available_forecasts.sort()  # Chronological order for lead time analysis

            print("Available forecasts:", available_forecasts)
            # Load ERA5 data once
            era5_path: Path = (
                self.model.input_folder
                / "other"
                / "climate"
                / "pr_kg_per_m2_per_s.zarr"
            )
            if not era5_path.exists():
                raise FileNotFoundError(f"ERA5 data not found: {era5_path}")

            era5_ds = open_zarr(era5_path)
            era5_mm_per_h: xarray.DataArray = era5_ds * 3600  # Convert from m/s to mm/h

            # Main processing loop for all forecasts
            for forecast_init in available_forecasts:
                print(f"Processing forecast initialization: {forecast_init}")

                # Parse initialization time
                init_datetime: datetime = datetime.strptime(
                    forecast_init, "%Y%m%dT%H%M%S"
                )

                forecast_folder: Path = forecast_base_path / forecast_init
                pr_files: list[Path] = list(forecast_folder.glob("*pr*.zarr"))

                if not pr_files:
                    raise ValueError(
                        f"No precipitation forecast file found for initialization {forecast_init}"
                    )

                # Load forecast data
                forecast_ds = open_zarr(pr_files[0])
                forecast_time = forecast_ds["time"].values
                forecast_mm_per_h: xarray.DataArray = (
                    forecast_ds * 3600
                )  # Convert to mm/h

                # Resample to daily data
                # ERA5 data for the same period
                era5_period: xarray.DataArray = era5_mm_per_h.sel(
                    time=slice(forecast_time[0], forecast_time[-1])
                )

                # Create warning time series for different data sources
                era5_binary_arrays = create_binary_threshold_arrays(
                    era5_period,
                    daily_cumulative_thresholds_mm_per_day,
                    threshold_time_unit,
                )
                # Use appropriate data for warning timeseries based on time unit
                era5_data_for_timeseries = (
                    era5_period
                    if threshold_time_unit == "hourly"
                    else era5_period.resample(time="1D").sum()
                )
                era5_warning_timeseries = create_deterministic_warning_timeseries(
                    era5_binary_arrays, era5_data_for_timeseries
                )

                # Create warning time series for control forecast (deterministic)
                control_binary_arrays = create_binary_threshold_arrays(
                    forecast_mm_per_h.isel(member=0),
                    daily_cumulative_thresholds_mm_per_day,
                    threshold_time_unit,
                )
                # Use appropriate data for warning timeseries based on time unit
                control_data_for_timeseries = (
                    forecast_mm_per_h.isel(member=0)
                    if threshold_time_unit == "hourly"
                    else forecast_mm_per_h.isel(member=0).resample(time="1D").sum()
                )
                control_warning_timeseries = create_deterministic_warning_timeseries(
                    control_binary_arrays, control_data_for_timeseries
                )

                ensemble_warning_timeseries = create_ensemble_warning_timeseries(
                    forecast_mm_per_h.isel(member=slice(1, None)),
                    daily_cumulative_thresholds_mm_per_day,
                    ensemble_probability_thresholds,
                    threshold_time_unit,
                )

                # Save detailed warning time series for spatial analysis
                save_warning_timeseries(
                    warnings_folder,
                    forecast_init,
                    era5_warning_timeseries,
                    control_warning_timeseries,
                    ensemble_warning_timeseries,
                )

            # Create visualization outputs
            print("\nCreating Warning Visualizations")

            # Define standard lead times for spatial analysis
            standard_lead_times = [1, 2, 5, 10]  # days before event

            # Create maximum warning level maps for key lead times
            create_spatial_warning_maps(
                standard_lead_times, moment_of_flooding=moment_of_flooding
            )

            print("Rainfall warning analysis completed successfully!")

        # # Call the rainfall evaluation functions
        evaluate_precipitation_forecasts(
            plot_type="intensity", moment_of_flooding=moment_of_flooding
        )
        evaluate_precipitation_forecasts(
            plot_type="cumulative", moment_of_flooding=moment_of_flooding
        )

        # Call the rainfall warning analysis
        # issue_rainfall_warning(
        #    processing_forecasts, moment_of_flooding=moment_of_flooding
        # )
