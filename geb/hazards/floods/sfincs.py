from pathlib import Path
from collections import deque
from datetime import datetime
import xarray as xr
import zarr
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import CRS

from ...HRUs import load_geom

try:
    from geb_hydrodynamics.build_model import build_sfincs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'GEB-hydrodynamics' package is not installed. Please install it by running 'pip install geb-hydrodynamics'."
    )
from geb_hydrodynamics.sfincs_utils import run_sfincs_simulation
from geb_hydrodynamics.update_model_forcing import update_sfincs_model_forcing
from geb_hydrodynamics.run_sfincs_for_return_periods import (
    run_sfincs_for_return_periods,
)
from geb_hydrodynamics.postprocess_model import read_flood_map
from geb_hydrodynamics.estimate_discharge_for_return_periods import (
    estimate_discharge_for_return_periods,
)


class SFINCS:
    def __init__(self, model, config, n_timesteps=10):
        self.model = model
        self.hydrology = model.hydrology
        self.config = (
            self.model.config["hazards"]["floods"]
            if "floods" in self.model.config["hazards"]
            else {}
        )
        self.n_timesteps = n_timesteps

        self.discharge_per_timestep = deque(maxlen=self.n_timesteps)

    def sfincs_model_root(self, basin_id):
        folder = self.model.simulation_root / "SFINCS" / str(basin_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def sfincs_simulation_root(self, basin_id):
        folder = (
            self.sfincs_model_root(basin_id)
            / "simulations"
            / f"{self.model.current_time.strftime('%Y%m%dT%H%M%S')}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_event_name(self, event):
        if "basin_id" in event:
            return event["basin_id"]
        elif "region" in event:
            return "region"
        else:
            return "all"

    @property
    def data_catalogs(self):
        return [
            str(
                Path(self.model.config["general"]["input_folder"])
                / "hydrodynamics"
                / "data_catalog.yml"
            )
        ]

    def get_utm_zone(self, region_file):
        region = load_geom(region_file)
        # Calculate the central longitude of the dataset
        centroid = region.geometry.centroid
        central_lon = centroid.x.mean()  # Mean longitude of the dataset

        # Determine the UTM zone based on the longitude
        utm_zone = int((central_lon + 180) // 6) + 1

        # Determine if the data is in the Northern or Southern Hemisphere
        # The EPSG code for UTM in the northern hemisphere is EPSG:326xx (xx = zone)
        # The EPSG code for UTM in the southern hemisphere is EPSG:327xx (xx = zone)
        if centroid.y.mean() > 0:
            utm_crs = f"EPSG:326{utm_zone}"  # Northern hemisphere
        else:
            utm_crs = f"EPSG:327{utm_zone}"  # Southern hemisphere
        return utm_crs

    def setup(self, event):
        build_parameters = {}

        event_name = self.get_event_name(event)
        if "region" in event:
            if event["region"] is None:
                model_bbox = self.hydrology.data.grid.bounds
                build_parameters["bbox"] = (
                    model_bbox[0] + 0.1,
                    model_bbox[1] + 0.1,
                    model_bbox[2] - 0.1,
                    model_bbox[3] - 0.1,
                )
            else:
                raise NotImplementedError

        if (
            "simulate_coastal_floods" in self.model.config["general"]
            and self.model.config["general"]["simulate_coastal_floods"]
        ):
            build_parameters["simulate_coastal_floods"] = True

        model_root = self.sfincs_model_root(event_name)

        if (
            self.config["force_overwrite"]
            or not (self.sfincs_model_root(event_name) / "sfincs.inp").exists()
        ):
            build_parameters = self.get_build_parameters(model_root)
            build_sfincs(**build_parameters)

    def to_sfincs_datetime(self, dt: datetime):
        return dt.strftime("%Y%m%d %H%M%S")

    def set_forcing(self, event, start_time):
        if self.model.simulate_hydrology:
            n_timesteps = min(self.n_timesteps, len(self.discharge_per_timestep))
            substeps = self.discharge_per_timestep[0].shape[0]
            discharge_grid = self.hydrology.data.grid.decompress(
                np.vstack(self.discharge_per_timestep)
            )

            # when SFINCS starts with high values, this leads to numerical instabilities. Therefore, we first start with very low discharge and then build up slowly to timestep 0
            # TODO: Check if this is a right approach
            discharge_grid = np.vstack(
                [
                    np.full_like(discharge_grid[:substeps, :, :], fill_value=np.nan),
                    discharge_grid,
                ]
            )  # prepend zeros
            for i in range(substeps - 1, -1, -1):
                discharge_grid[i] = discharge_grid[i + 1] * 0.3
            # convert the discharge grid to an xarray DataArray
            discharge_grid = xr.DataArray(
                data=discharge_grid,
                coords={
                    "time": pd.date_range(
                        end=self.model.current_time
                        - self.model.timestep_length / substeps,
                        periods=(n_timesteps + 1)
                        * substeps,  # +1 because we prepend the discharge above
                        freq=self.model.timestep_length / substeps,
                        inclusive="right",
                    ),
                    "y": self.hydrology.data.grid.lat,
                    "x": self.hydrology.data.grid.lon,
                },
                dims=["time", "y", "x"],
                name="discharge",
            )
        else:
            substeps = 24  # when setting 0 it doesn't matter so much how many substeps. 24 is a reasonable default.
            n_timesteps = (event["end_time"] - event["start_time"]).days
            time = pd.date_range(
                end=self.model.current_time - self.model.timestep_length / substeps,
                periods=(n_timesteps + 1)
                * substeps,  # +1 because we prepend the discharge above
                freq=self.model.timestep_length / substeps,
                inclusive="right",
            )
            discharge_grid = np.zeros(
                shape=(len(time), *self.model.hydrology.data.grid.mask.shape)
            )
            discharge_grid = xr.DataArray(
                data=discharge_grid,
                coords={
                    "time": time,
                    "y": self.hydrology.data.grid.lat,
                    "x": self.hydrology.data.grid.lon,
                },
                dims=["time", "y", "x"],
                name="discharge",
            )

        discharge_grid = xr.Dataset({"discharge": discharge_grid})

        # Convert the WKT string to a pyproj CRS object
        crs_obj = CRS.from_wkt(self.hydrology.data.grid.crs)

        # Now you can safely call to_proj4() on the CRS object
        discharge_grid.raster.set_crs(crs_obj.to_proj4())
        tstart = start_time
        tend = discharge_grid.time[-1] + pd.Timedelta(
            self.model.timestep_length / substeps
        )
        discharge_grid = discharge_grid.sel(time=slice(tstart, tend))

        sfincs_precipitation = event.get("precipitation", None)

        if sfincs_precipitation is None:
            sfincs_precipitation = (
                xr.open_dataset(
                    self.model.files["forcing"]["climate/pr_hourly"], engine="zarr"
                ).rename(pr_hourly="precip")["precip"]
                * 3600
            )  # convert from kg/m2/s to mm/h for
            sfincs_precipitation.raster.set_crs(
                4326
            )  # TODO: Remove when this is added to hydromt_sfincs
            sfincs_precipitation = sfincs_precipitation.rio.set_crs(4326)

        event_name = self.get_event_name(event)

        update_sfincs_model_forcing(
            model_root=self.sfincs_model_root(event_name),
            simulation_root=self.sfincs_simulation_root(event_name),
            current_event={
                "tstart": self.to_sfincs_datetime(tstart),
                "tend": self.to_sfincs_datetime(tend.dt).item(),
            },
            forcing_method="precipitation",
            discharge_grid=discharge_grid,
            precipitation_grid=sfincs_precipitation,
            data_catalogs=self.data_catalogs,
            uparea_discharge_grid=xr.open_dataset(
                self.model.files["grid"]["routing/upstream_area"],
                engine="zarr",
            ),  # .isel(band=0),
        )
        return None

    def run_single_event(self, event, start_time, return_period=None):
        self.setup(event)
        self.set_forcing(event, start_time)
        self.model.logger.info(f"Running SFINCS for {self.model.current_time}...")
        event_name = self.get_event_name(event)
        run_sfincs_simulation(
            simulation_root=self.sfincs_simulation_root(event_name),
            model_root=self.sfincs_model_root(event_name),
            gpu=False,
        )
        flood_map = read_flood_map(
            model_root=self.sfincs_model_root(event_name),
            simulation_root=self.sfincs_simulation_root(event_name),
            return_period=return_period,
        )  # xc, yc is for x and y in rotated grid`DD`
        damages = self.flood(
            flood_map=flood_map,
            simulation_root=self.sfincs_simulation_root(event_name),
            return_period=return_period,
        )
        return damages

    def scale_event(self, event, scale_factor):
        scaled_event = event.copy()
        sfincs_precipitation = (
            xr.open_dataset(
                self.model.files["forcing"]["climate/pr_hourly"], engine="zarr"
            ).rename(pr_hourly="precip")["precip"]
            * 3600
            * scale_factor
        )  # convert from kg/m2/s to mm/h for
        sfincs_precipitation.raster.set_crs(
            4326
        )  # TODO: Remove when this is added to hydromt_sfincs
        sfincs_precipitation = sfincs_precipitation.rio.set_crs(4326)

        scaled_event["precipitation"] = sfincs_precipitation
        return scaled_event

    def get_return_period_maps(self):
        # close the zarr store
        if hasattr(self.model, "reporter"):
            self.model.reporter.variables["discharge_daily"].close()

        model_root = self.sfincs_model_root("entire_region")
        if self.config["force_overwrite"] or not (model_root / "sfincs.inp").exists():
            build_sfincs(
                **self.get_build_parameters(model_root),
            )
        estimate_discharge_for_return_periods(
            model_root,
            discharge=self.discharge_spinup_ds,
            rivers=self.rivers,
            return_periods=self.config["return_periods"],
        )
        run_sfincs_for_return_periods(
            model_root=model_root,
            return_periods=self.config["return_periods"],
            gpu=self.config["gpu"],
            export_dir=self.model.report_folder / "flood_maps",
            clean_working_dir=True,
        )

        if hasattr(self.model, "reporter"):
            # and re-open afterwards
            self.model.reporter.variables["discharge_daily"] = zarr.ZipStore(
                self.model.config["report_hydrology"]["discharge_daily"]["path"],
                mode="a",
            )

    def run(self, event):
        start_time = event["start_time"]

        if self.model.config["hazards"]["floods"]["flood_risk"]:
            print("config settings are read")
            scale_factors = pd.read_parquet(
                self.model.files["table"]["hydrodynamics/risk_scaling_factors"]
            )
            scale_factors["return_period"] = 1 / scale_factors["exceedance_probability"]
            damages_list = []
            return_periods_list = []
            exceedence_probabilities_list = []
            for index, row in scale_factors.iterrows():
                return_period = row["return_period"]
                exceedence_probability = row["exceedance_probability"]
                scale_factor = row["scaling_factor"]
                scaled_event = self.scale_event(event, scale_factor)
                damages = self.run_single_event(scaled_event, start_time, return_period)

                damages_list.append(damages)
                return_periods_list.append(return_period)
                exceedence_probabilities_list.append(exceedence_probability)

            print(damages_list)
            print(return_periods_list)
            print(exceedence_probabilities_list)

            plt.plot(return_periods_list, damages_list)
            plt.xlabel("Return period")
            plt.ylabel("Flood damages [euro]")
            plt.title("Damages per return period")
            plt.show()

            inverted_damage_list = damages_list[::-1]
            inverted_exceedence_probabilities_list = exceedence_probabilities_list[::-1]

            expected_annual_damage = np.trapz(
                y=inverted_damage_list, x=inverted_exceedence_probabilities_list
            )  # np.trapezoid or np.trapz -> depends on np version
            print(f"exptected annual damage is: {expected_annual_damage}")

        else:
            self.run_single_event(event, start_time)

    def flood(self, simulation_root, flood_map, return_period=None):
        damages = self.model.agents.households.flood(
            simulation_root=simulation_root,
            flood_map=flood_map,
            return_period=return_period,
        )
        return damages

    def save_discharge(self):
        self.discharge_per_timestep.append(
            self.hydrology.data.grid.var.discharge_substep
        )  # this is a deque, so it will automatically remove the oldest discharge

    @property
    def discharge_spinup_ds(self):
        ds = xr.open_dataset(
            Path("report") / "spinup" / "discharge_daily.zarr.zip", engine="zarr"
        )["discharge_daily"]
        # start_time = pd.to_datetime(ds.time[0].item()) + pd.DateOffset(years=10)
        # ds = ds.sel(time=slice(start_time, ds.time[-1]))

        # # make sure there is at least 20 years of data
        # if not len(ds.time.groupby(ds.time.dt.year).groups) >= 20:
        #     raise ValueError(
        #         """Not enough data available for reliable spinup, should be at least 20 years of data left.
        #         Please run the model for at least 30 years (10 years of data is discarded)."""
        #     )

        return ds

    @property
    def rivers(self):
        return load_geom(self.model.files["geoms"]["routing/rivers"])

    @property
    def crs(self):
        crs = self.config["crs"]
        if crs == "auto":
            crs = self.get_utm_zone(self.model.files["geoms"]["region"])
        return crs

    def get_build_parameters(self, model_root):
        with open(self.model.files["dict"]["hydrodynamics/DEM_config"]) as f:
            DEM_config = json.load(f)
        return {
            "model_root": model_root,
            "data_catalogs": self.data_catalogs,
            "region": load_geom(self.model.files["geoms"]["routing/subbasins"]),
            "DEM_config": DEM_config,
            "rivers": self.rivers,
            "discharge": self.discharge_spinup_ds,
            "resolution": self.config["resolution"],
            "nr_subgrid_pixels": self.config["nr_subgrid_pixels"],
            "crs": self.crs,
            "depth_calculation": "power_law",
            "mask_flood_plains": False,  # setting this to True sometimes leads to errors
        }
