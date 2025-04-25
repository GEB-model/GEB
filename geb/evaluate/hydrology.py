import matplotlib.pyplot as plt
import xarray as xr


class Hydrology:
    def __init__(self):
        pass

    def plot_discharge(self):
        discharge = xr.open_dataarray(
            self.model.output_folder
            / "report"
            / "spinup"
            / "hydrology.routing"
            / "discharge_daily.zarr"
        )
        mean_discharge = discharge.mean(dim="time")

        fig, ax = plt.subplots(figsize=(10, 10))

        mean_discharge.plot(ax=ax, cmap="Blues")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.savefig(
            self.output_folder_evaluate / "mean_discharge_m3_per_s.tif", dpi=300
        )
