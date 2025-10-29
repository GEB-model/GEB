from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from matplotlib_scalebar.scalebar import ScaleBar
from geb.workflows.io import open_zarr

# from matplotlib_scalebar.scalebar import ScaleBar


class MeteorologicalForecasts:
    """Implements several functions to evaluate the meteorological forecasts inside GEB."""

    def __init__(self) -> None:
        pass

    def evaluate_forecasts(
        self, run_name: str = "default", *args: Any, **kwargs: Any
    ) -> None:
        """Evaluate meteorological forecasts by comparing precipitation forecasts from different initialisation times (00:00 (not yet vs 12:00) UTC) against ERA5 reanalysis data and deterministic forecasts.

        Args:
            run_name: Name of the GEB run to evaluate. Default is "default".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        forecast_folder = self.output_folder_evaluate / "forecasts"
        forecast_folder.mkdir(parents=True, exist_ok=True)

        flood_maps_folder = self.model.input_folder / "other" / "forecasts" / "ECMWF"

        forecast_initialisations = [
            item.name for item in flood_maps_folder.iterdir() if item.is_dir()
        ]
        print(f"Found forecast initialisations: {forecast_initialisations}")
        num_lead_times = len(forecast_initialisations)

        def load_max_forecast_data(
            folder: Path, lead_time_str: str, percentiles=[25, 50, 75, 90, 95]
        ):
            """
            Import ERA5, control and ensemble data for one specific lead time and take the maximum value spatially
            """
            era5_path = self.model.input_folder / "other" / "climate" / "pr_hourly.zarr"

            # ERA5
            era5_ds = open_zarr(era5_path)
            era5 = era5_ds["Precipitation"].max(dim=["y", "x"])
            era5_time = era5_ds["time"].values

            # Control
            control_ds = open_zarr(folder)
            control = control_ds["Precipitation"].max(dim=["y", "x"])
            control_time = control_ds["time"].values

            # ERA5 clip on time range control
            era5_clipped = era5.sel(time=slice(control_time[0], control_time[-1]))

            # Ensemble percentiles per map
            # ensemble = {}
            ens_ds = open_zarr(folder)
            ens_data = ens_ds["Precipitation"].max(dim=["y", "x"])

            print(ens_data)
            ensemble = ens_data
            return era5_clipped, control, control_time, ensemble

        