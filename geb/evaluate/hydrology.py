import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from geb.workflows.io import to_zarr


class Hydrology:
    def __init__(self):
        pass

    def evaluate_discharge_grid(self):
        discharge = xr.open_dataarray(
            self.model.output_folder
            / "report"
            / "spinup"
            / "hydrology.routing"
            / "discharge_daily.zarr"
        )
        mean_discharge = discharge.mean(dim="time")
        mean_discharge.attrs["_FillValue"] = np.nan

        to_zarr(
            mean_discharge,
            self.output_folder_evaluate / "mean_discharge_m3_per_s.zarr",
            crs=4326,
        )

        fig, ax = plt.subplots(figsize=(10, 10))

        mean_discharge.plot(ax=ax, cmap="Blues")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.savefig(
            self.output_folder_evaluate / "mean_discharge_m3_per_s.png", dpi=300
        )
