"""Evaluation of meteorological forecasts in GEB.

This module provides functionality to evaluate and visualize meteorological forecasts and especially rainfall data by
comparing ECMWF ensemble forecasts against ERA5 reanalysis data for precipitation analysis.
Supports both intensity and cumulative precipitation plotting.
"""

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

# from matplotlib_scalebar.scalebar import ScaleBar
from geb.workflows.io import read_zarr


class MeteorologicalForecasts:
    """Implements several functions to evaluate the meteorological forecasts inside GEB."""

    def __init__(self) -> None:
        """Initialize MeteorologicalForecasts."""
        pass

    def evaluate_forecasts(
        self, model: Any, output_folder: Path, *args: Any, **kwargs: Any
    ) -> None:
        """Evaluate meteorological forecasts by comparing different forecast types at different initialisation times (00:00 (not yet vs 12:00) UTC).

        It compares ERA5 reanalysis data against forecast types such as ensemble forecasts and deterministic forecasts.

        Args:
            model: The forecast model to evaluate.
            output_folder: The folder to save evaluation outputs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the forecast model path does not exist.
        """
        # Create forecast output folder within the evaluate folder
        forecast_folder: Path = output_folder / "forecasts"
        forecast_folder.mkdir(parents=True, exist_ok=True)

        # Base path for forecast data
        forecast_base_path: Path = (
            model.input_folder
            / "other"
            / "forecasts"
            / "ECMWF"
            / "merged_control_ensemble"
        )

        if not forecast_base_path.exists():
            raise ValueError(
                f"Forecast model path does not exist: {forecast_base_path}"
            )

        def evaluate_precipitation_forecasts(
            plot_type: str = "intensity", *args: Any, **kwargs: Any
        ) -> None:
            """Evaluate precipitation forecasts by plotting rainfall data over time for different forecast initialisations.

            This function creates multi-panel plots comparing ERA5 reanalysis data with ECMWF ensemble
            forecasts. The plot type determines whether intensity or cumulative precipitation is shown.

            Args:
                plot_type: Type of plot to generate. Either "intensity" for mm/h or "cumulative" for mm.
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
                    model.input_folder / "other" / "climate" / "pr_kg_per_m2_per_s.zarr"
                )

                # ERA5
                era5_ds = read_zarr(era5_path)
                era5_mm_per_h: xarray.DataArray = (
                    era5_ds * 3600
                )  # Convert from m/s to mm/h
                era5_max_mm_per_h: xarray.DataArray = era5_mm_per_h.max(dim=["y", "x"])

                # Load ensemble data (including control)
                ens_ds = read_zarr(init_folder / pr_files[0].name)
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
                    era5_processed: xarray.DataArray = era5_clipped.cumsum(dim="time")
                    ens_processed: xarray.DataArray = ens_max_mm_per_h.cumsum(
                        dim="time"
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
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m %H:%M"))
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
                x_start: pd.Timestamp = pd.Timestamp("2024-04-26 00:00"),
                x_end: pd.Timestamp = pd.Timestamp("2024-05-16 00:00"),
            ) -> None:
                """Plot precipitation data (intensity or cumulative) for a single subplot.

                Args:
                    ax: Matplotlib axes object to plot on.
                    era5_data: ERA5 precipitation data (mm/h for intensity, mm for cumulative).
                    control_data: Control forecast precipitation data (mm/h for intensity, mm for cumulative).
                    ensemble_data: Ensemble forecast precipitation data (mm/h for intensity, mm for cumulative).
                    control_time: Time array for the forecast data (e.g. timesteps corresponding to the data).
                    init_time_str: Initialization time identifier string (e.g., '20240429T000000').
                    plot_type: If "cumulative", format for cumulative data. If "intensity", format for intensity.
                    show_legend: Whether to show the legend on this subplot.
                    x_start: Start time for x-axis.
                    x_end: End time for x-axis.
                """
                # Parse initialization time string to readable format
                init_datetime: pd.Timestamp = pd.to_datetime(
                    init_time_str, format="%Y%m%dT%H%M%S"
                )
                init_time_readable: str = init_datetime.strftime("%d %B %Y, %H:%M UTC")
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

                # Add moment of flooding line
                moment_of_inundation: pd.Timestamp = pd.Timestamp(
                    "2024-05-06T00:00:00.000000000"
                )
                ax.axvline(
                    moment_of_inundation,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Moment of Flooding Guaíba Lake",
                )

                # Set title and formatting
                ax.set_title(
                    f"Forecast Initialization: {init_time_readable}", fontsize=18
                )

                if plot_type == "cumulative":
                    ax.set_yticks(
                        range(0, 1501, 250)
                    )  # Adjusted for cumulative mm values
                else:
                    ax.set_yticks(range(0, 47, 5))  # For intensity mm/h values

                x_ticks: list[pd.Timestamp] = pd.date_range(
                    start=x_start, end=x_end, freq="12h"
                )
                format_time_axis(ax, x_start, x_end, x_ticks)

                if show_legend:
                    ax.legend(fontsize=12, loc="upper right")

            # Main evaluation logic
            forecast_initialisations: list[str] = [
                item.name for item in forecast_base_path.iterdir() if item.is_dir()
            ]
            print(f"Found forecast initialisations: {forecast_initialisations}")

            num_forecasts: int = len(forecast_initialisations)

            # Calculate grid dimensions - 2 columns, enough rows for all forecasts
            num_cols: int = 2
            num_rows: int = (
                num_forecasts + num_cols - 1
            ) // num_cols  # Ceiling division

            # Create subplots in 2-column grid
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(20, 4 * num_rows), sharex=True, sharey=True
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

                # Calculate subplot position
                row_idx: int = idx // num_cols
                col_idx: int = idx % num_cols

                init_folder: Path = forecast_base_path / init_time

                # Load data for this initialisation
                era5_data, control_data, time_data, ensemble_data = load_forecast_data(
                    init_folder, plot_type=plot_type
                )

                show_legend_flag: bool = row_idx == 0 and col_idx == 1
                # Get the correct axis
                if num_rows == 1:
                    current_ax: plt.Axes = axes[col_idx] if num_cols > 1 else axes
                else:
                    current_ax: plt.Axes = (
                        axes[row_idx, col_idx] if num_cols > 1 else axes[row_idx]
                    )

                # Plot data
                plot_rainfall_data(
                    ax=current_ax,
                    era5_data=era5_data,
                    control_data=control_data,
                    ensemble_data=ensemble_data,
                    control_time=time_data,
                    init_time_str=init_time,
                    plot_type=plot_type,
                    show_legend=show_legend_flag,  # Only show legend on first right subplot
                )

            # Hide unused subplots if odd number of forecasts
            for row in range(num_rows):
                for col in range(num_cols):
                    idx: int = row * num_cols + col

                    # Get current axis using same logic as above
                    if num_rows == 1:
                        current_ax: plt.Axes = axes[col] if num_cols > 1 else axes
                    else:
                        current_ax: plt.Axes = (
                            axes[row, col] if num_cols > 1 else axes[row]
                        )

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
                ylabel: str = "Cumulative Rainfall [mm]"
            else:
                plot_title: str = "ERA5 vs ECMWF Control & Probabilistic Maximum Intensity Precipitation Forecasts"
                output_filename: str = "ERA5_vs_ECMWF_Control_&_Probabilistic_Precipitation_Forecasts_Maximum_Intensity.png"
                ylabel: str = "Rainfall intensity [mm/h]"
            fig.suptitle(plot_title, fontsize=22, y=0.95)
            fig.text(0.5, 0.02, "Date (UTC)", ha="center", fontsize=18)
            fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=18)

            plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])

            # Save plot
            plt.savefig(
                forecast_folder / output_filename, dpi=1000, bbox_inches="tight"
            )
            plt.close()

        # Call the rainfall evaluation functions
        evaluate_precipitation_forecasts(plot_type="intensity")
        evaluate_precipitation_forecasts(plot_type="cumulative")
