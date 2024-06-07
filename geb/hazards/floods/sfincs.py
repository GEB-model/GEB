import os
from pathlib import Path
from collections import deque
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from sfincs_river_flood_simulator import (
    build_sfincs,
    update_sfincs_model_forcing,
    run_sfincs_simulation,
    read_flood_map,
)


class SFINCS:
    def __init__(self, model, config, n_timesteps=10):
        self.model = model
        self.config = config
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
            raise ValueError(
                "Either 'basin_id' or 'region' must be specified in the event."
            )

    @property
    def data_catalogs(self):
        return [
            str(
                Path(self.model.config["general"]["input_folder"])
                / "SFINCS"
                / "sfincs_data_catalog.yml"
            )
        ]

    def setup(self, event, config_fn="sfincs.yml", force_overwrite=False):
        build_parameters = {}

        force_overwrite = True

        if "basin_id" in event:
            build_parameters["basin_id"] = event["basin_id"]
            event_name = self.get_event_name(event)
        elif "region" in event:
            if event["region"] is None:
                model_bbox = self.model.data.grid.bounds
                build_parameters["bbox"] = (
                    model_bbox[0] + 0.1,
                    model_bbox[1] + 0.1,
                    model_bbox[2] - 0.1,
                    model_bbox[3] - 0.1,
                )
            else:
                raise NotImplementedError
            event_name = self.get_event_name(event)
        else:
            raise ValueError(
                "Either 'basin_id' or 'region' must be specified in the event."
            )

        if (
            "simulate_coastal_floods" in self.model.config["general"]
            and self.model.config["general"]["simulate_coastal_floods"]
        ):
            build_parameters["simulate_coastal_floods"] = True

        build_parameters.update(
            {
                "config_fn": str(config_fn),
                "model_root": self.sfincs_model_root(event_name),
                "data_catalogs": self.data_catalogs,
                "mask": gpd.read_file(
                    self.model.model_structure["geoms"]["areamaps/region"]
                ),
                "method": "precipitation",
            }
        )
        if (
            force_overwrite
            or not (self.sfincs_model_root(event_name) / "sfincs.inp").exists()
        ):
            build_sfincs(**build_parameters)

    def to_sfincs_datetime(self, dt: datetime):
        return dt.strftime("%Y%m%d %H%M%S")

    def set_forcing(self, event, start_time):
        if self.model.config["general"]["simulate_hydrology"]:
            n_timesteps = min(self.n_timesteps, len(self.discharge_per_timestep))
            substeps = self.discharge_per_timestep[0].shape[0]
            discharge_grid = self.model.data.grid.decompress(
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
                    "y": self.model.data.grid.lat,
                    "x": self.model.data.grid.lon,
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
                shape=(len(time), *self.model.data.grid.mask.shape)
            )
            discharge_grid = xr.DataArray(
                data=discharge_grid,
                coords={
                    "time": time,
                    "y": self.model.data.grid.lat,
                    "x": self.model.data.grid.lon,
                },
                dims=["time", "y", "x"],
                name="discharge",
            )

        discharge_grid = xr.Dataset({"discharge": discharge_grid})
        discharge_grid.raster.set_crs(self.model.data.grid.crs.to_proj4())
        tstart = start_time
        tend = discharge_grid.time[-1] + pd.Timedelta(
            self.model.timestep_length / substeps
        )
        discharge_grid = discharge_grid.sel(time=slice(tstart, tend))
        sfincs_precipitation = (
            xr.open_dataset(
                self.model.model_structure["forcing"]["climate/pr_hourly"]
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
            discharge_grid=discharge_grid,
            precipitation_grid=sfincs_precipitation,
            data_catalogs=self.data_catalogs,
            uparea_discharge_grid=xr.open_dataset(
                self.model.model_structure["grid"]["routing/kinematic/upstream_area"]
            ).isel(band=0),
        )
        return None

    def run(self, event):
        start_time = event["start_time"]
        self.setup(event, force_overwrite=False)
        self.set_forcing(event, start_time)
        self.model.logger.info(f"Running SFINCS for {self.model.current_time}...")

        event_name = self.get_event_name(event)
        run_sfincs_simulation(simulation_root=self.sfincs_simulation_root(event_name))
        flood_map = read_flood_map(
            model_root=self.sfincs_model_root(event_name),
            simulation_root=self.sfincs_simulation_root(event_name),
        )  # xc, yc is for x and y in rotated grid
        self.flood(flood_map)

    def flood(self, flood_map):
        self.model.agents.households.flood(flood_map)

    def save_discharge(self):
        self.discharge_per_timestep.append(
            self.model.data.grid.discharge_substep
        )  # this is a deque, so it will automatically remove the oldest discharge
