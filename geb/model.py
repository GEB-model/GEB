import copy
import datetime
import os
from pathlib import Path
from time import time
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from honeybees.model import Model as ABM_Model

from geb.agents import Agents
from geb.artists import Artists
from geb.hazards.driver import HazardDriver
from geb.module import Module
from geb.reporter import Reporter
from geb.store import Store
from geb.workflows.dt import round_up_to_start_of_next_day_unless_midnight
from geb.workflows.io import open_zarr

from .evaluate import Evaluate
from .forcing import Forcing
from .hydrology import Hydrology
from .hydrology.HRUs import load_geom


class GEBModel(Module, HazardDriver, ABM_Model):
    """GEB parent class.

    Args:
        config: Filepath of the YAML-configuration file (e.g. model.yml).
        files: Dictionary with the paths of the input files.
        mode: Mode of the model. Either `w` (write) or `r` (read).
        timing: Boolean indicating if the model steps should be timed.
    """

    def __init__(
        self,
        config: dict,
        files: dict,
        mode: str = "w",
        timing: bool = False,
    ):
        self.timing = timing
        self.mode = mode

        Module.__init__(self, self, create_var=False)

        self._multiverse_name = None

        self.config = config

        # make a deep copy to avoid issues when the model is initialized multiple times
        self.files = copy.deepcopy(files)
        for data in self.files.values():
            for key, value in data.items():
                data[key] = self.input_folder / value

        self.mask = load_geom(self.files["geoms"]["mask"])

        self.store = Store(self)
        self.artists = Artists(self)

        self.forcing = Forcing(self)

        # Empty list to hold plantFATE models. If forests are not used, this will be empty
        self.plantFATE = []

    @property
    def name(self) -> str:
        return ""

    def restore(
        self, store_location: str | Path, timestep: int, n_timesteps: int
    ) -> None:
        """Restore the model state to the original state given by the function input.

        Args:
            store_location: Location of the store to restore the model state from.
            timestep: timestep to restore the model state to.
            n_timesteps: number of timesteps (i.e., the final timestep) to restore the model state to.
        """
        self.store.load(store_location)

        # restore the heads of the groundwater model
        self.hydrology.groundwater.modflow.restore(self.hydrology.grid.var.heads)

        # restore the discharge from the store
        self.hydrology.routing.router.Q_prev = (
            self.hydrology.routing.grid.var.discharge_m3_s.copy()
        )
        self.current_timestep = timestep
        self.n_timesteps = n_timesteps

    @overload
    def multiverse(
        self, forecast_dt: datetime.datetime, return_mean_discharge: Literal[True]
    ) -> dict[Any, float]: ...

    @overload
    def multiverse(
        self, forecast_dt: datetime.datetime, return_mean_discharge: Literal[False]
    ) -> None: ...

    def multiverse(
        self, forecast_dt: datetime.datetime, return_mean_discharge: bool = False
    ) -> None | dict[Any, float]:
        # copy current state of timestep and time
        store_timestep: int = copy.copy(self.current_timestep)
        store_n_timesteps: int = copy.copy(self.n_timesteps)

        # set a folder to store the initial state of the multiverse
        store_location: Path = self.simulation_root / "multiverse" / "forecast"
        self.store.save(store_location)

        original_pr_hourly: xr.DataArray = self.forcing["pr_hourly"]

        forecasts: xr.DataArray = open_zarr(
            Path("data") / "forecasts" / f"{forecast_dt.strftime('%Y%m%dT%H%M%S')}.zarr"
        )

        if return_mean_discharge:
            mean_discharge: dict[Any, float] = {}

        for member in forecasts.member:
            self.multiverse_name = member.item()

            pr_hourly_forecast: xr.DataArray = forecasts.sel(member=member)

            foecast_end_date = round_up_to_start_of_next_day_unless_midnight(
                pd.to_datetime(pr_hourly_forecast.time[-1].item()).to_pydatetime()
            ).date()
            self.n_timesteps = (foecast_end_date - self.start_time.date()).days

            # Clip the original precipitation data to the start of the forecast
            # Therefore we take the start of the forecast and subtract one second
            # to ensure that the original precipitation data does not overlap with the forecast
            original_pr_hourly_clipped_to_start_of_forecast: xr.DataArray = (
                original_pr_hourly.sel(
                    time=slice(
                        None, (pr_hourly_forecast.time[0] - pd.Timedelta(seconds=1))
                    )
                )
            )

            # Concatenate the original precipitation data with the forecast data
            pr_hourly_observed_and_forecasted_combined: list[xr.DataArray] = [
                original_pr_hourly_clipped_to_start_of_forecast,
                pr_hourly_forecast,
            ]

            # Set the pr_hourly forcing data to the combined data
            self.model.forcing["pr_hourly"] = pr_hourly_observed_and_forecasted_combined

            print(f"Running forecast member {member.item()}...")
            self.step_to_end()

            if return_mean_discharge:
                mean_discharge[member.item()] = (
                    self.hydrology.routing.grid.var.discharge_m3_s.mean()
                ).item()

            # restore the initial state of the multiverse
            self.restore(
                store_location=store_location,
                timestep=store_timestep,
                n_timesteps=store_n_timesteps,
            )

        print("Forecast finished, restoring all conditions...")

        # restore the precipitation dataarray, step out of the multiverse
        self.forcing["pr_hourly"] = original_pr_hourly
        self.multiverse_name: None = None

        if return_mean_discharge:
            return mean_discharge
        else:
            return None

    def step(self) -> None:
        """Forward the model by the given the number of steps.

        Args:
            step_size: Number of steps the model should take. Can be integer or string `day`, `week`, `month`, `year`, `decade` or `century`.
        """
        # only if forecasts is used, and if we are not already in multiverse (avoiding infinite recursion)
        # and if the current date is in the list of forecast days
        if (
            self.config["general"]["forecasts"]["use"]
            and self.multiverse_name
            is None  # only start multiverse if not already in one
            and self.current_time.date()
        ):
            for dt in self.config["general"]["forecasts"]["times"]:
                if dt.date() == self.current_time.date():
                    self.multiverse(
                        forecast_dt=dt,
                        return_mean_discharge=True,
                    )

            # If the multiverse function is called and we want to simulate a warning response afterwards,
            if self.config["agent_settings"]["households"]["warning_response"]:
                self.agents.households.warning_strategy_1()
                # self.agents.households.infrastructure_warning_strategy()

        t0 = time()
        self.agents.step()
        if self.simulate_hydrology:
            self.hydrology.step()

        HazardDriver.step(self)

        self.report(self, locals())

        t1 = time()
        print(
            f"{self.multiverse_name + ' - ' if self.multiverse_name is not None else ''}finished {self.current_time} ({round(t1 - t0, 4)}s)",
            flush=True,
        )

        self.current_timestep += 1

    def create_datetime(self, date: datetime.date) -> datetime.datetime:
        """Create a datetime object from a date with time set to midnight.

        Args:
            date:  Date object to convert to datetime.

        Returns:
            Datetime object with time set to midnight.
        """
        return datetime.datetime.combine(date, datetime.time(0))

    def _initialize(
        self,
        create_reporter,
        current_time,
        n_timesteps,
        timestep_length,
        in_spinup=False,
        simulate_hydrology=True,
        clean_report_folder=False,
        load_data_from_store=False,
    ) -> None:
        """Initializes the model."""
        self.in_spinup = in_spinup
        self.simulate_hydrology = simulate_hydrology

        self.regions = load_geom(self.files["geoms"]["regions"])

        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.spinup_start = datetime.datetime.combine(
            self.config["general"]["spinup_time"], datetime.time(0)
        )
        self.timestep_length = timestep_length

        self.hydrology = Hydrology(self)

        HazardDriver.__init__(self)
        ABM_Model.__init__(
            self,
            current_time,
            self.timestep_length,
            n_timesteps=n_timesteps,
        )

        self.agents = Agents(self)

        if load_data_from_store:
            self.store.load()

        if self.simulate_hydrology:
            self.hydrology.routing.set_router()
            self.hydrology.groundwater.initalize_modflow_model()
            self.hydrology.soil.set_global_variables()

        if create_reporter:
            self.reporter = Reporter(self, clean=clean_report_folder)

    def step_to_end(self) -> None:
        """Run the model to the end of the simulation period."""
        for _ in range(self.n_timesteps - self.current_timestep):
            self.step()

    def run(self, initialize_only: bool = False) -> None:
        """Run the model for the entire period, and export water table in case of spinup scenario."""
        if not self.store.path.exists():
            raise FileNotFoundError(
                f"The initial conditions folder ({self.store.path.resolve()}) does not exist. Spinup is required before running the model. Please run the spinup first."
            )

        current_time: datetime.datetime = self.create_datetime(
            self.config["general"]["start_time"]
        )
        end_time: datetime.datetime = self.create_datetime(
            self.config["general"]["end_time"]
        )

        timestep_length: datetime.timedelta = datetime.timedelta(days=1)
        n_timesteps: float | int = (
            end_time + timestep_length - current_time
        ) / timestep_length
        assert n_timesteps.is_integer()
        n_timesteps: int = int(n_timesteps)
        assert n_timesteps > 0, "End time is before or identical to start time"

        self._initialize(
            create_reporter=True,
            current_time=current_time,
            n_timesteps=n_timesteps,
            timestep_length=timestep_length,
            clean_report_folder=True,
            load_data_from_store=True,
        )

        if initialize_only:
            return

        self.step_to_end()

        print("Model run finished, finalizing report...")
        self.reporter.finalize()

    def run_yearly(self) -> None:
        current_time: datetime.datetime = self.create_datetime(
            self.config["general"]["start_time"]
        )
        end_time: datetime.datetime = self.create_datetime(
            self.config["general"]["end_time"]
        )

        assert self.config["hazards"]["floods"]["simulate"] is False, (
            "Yearly mode is not compatible with flood simulation. Please set 'simulate' to False in the config."
        )

        assert current_time.month == 1 and current_time.day == 1, (
            "In yearly mode start time should be the first day of the year"
        )
        assert end_time.month == 12 and end_time.day == 31, (
            "In yearly mode end time should be the last day of the year"
        )

        n_timesteps = end_time.year - current_time.year + 1

        self._initialize(
            create_reporter=True,
            current_time=current_time,
            n_timesteps=n_timesteps,
            timestep_length=relativedelta(years=1),
            simulate_hydrology=False,
            clean_report_folder=True,
            load_data_from_store=True,
        )

        self.step_to_end()

        print("Model run finished, finalizing report...")
        self.reporter.finalize()

    def spinup(self, initialize_only: bool = False) -> None:
        """Run the model for the spinup period."""
        # set the start and end time for the spinup. The end of the spinup is the start of the actual model run
        current_time = self.create_datetime(self.config["general"]["spinup_time"])
        end_time_exclusive = self.create_datetime(self.config["general"]["start_time"])

        if end_time_exclusive.year - current_time.year < 10:
            print(
                "Spinup time is less than 10 years. This is not recommended and may lead to issues later."
            )

        timestep_length = datetime.timedelta(days=1)
        n_timesteps = (end_time_exclusive - current_time) / timestep_length
        assert n_timesteps.is_integer()
        n_timesteps = int(n_timesteps)
        assert n_timesteps > 0, "End time is before or identical to start time"

        # turn off any reporting for the ABM
        # self.config["report"] = {
        #     "hydrology.routing": {
        #         "discharge_daily": {
        #             "varname": "grid.var.discharge_m3_s",
        #             "type": "grid",
        #             "function": None,
        #             "format": "zarr",
        #             "single_file": True,
        #         }
        #     }
        # }

        self.var = self.store.create_bucket("var")

        self._initialize(
            create_reporter=True,
            current_time=current_time,
            n_timesteps=n_timesteps,
            timestep_length=datetime.timedelta(days=1),
            clean_report_folder=True,
            in_spinup=True,
        )

        if initialize_only:
            return

        self.step_to_end()

        print("Spinup finished, saving conditions at end of spinup...")
        self.store.save()

        self.reporter.finalize()

    def estimate_return_periods(self) -> None:
        """Estimate the risk of the model."""
        current_time: datetime.datetime = self.create_datetime(
            self.config["general"]["start_time"]
        )
        self.config["general"]["name"] = "estimate_return_periods"

        self._initialize(
            create_reporter=False,
            current_time=current_time,
            n_timesteps=0,
            timestep_length=relativedelta(years=1),
            load_data_from_store=True,
            simulate_hydrology=False,
            clean_report_folder=False,
        )

        HazardDriver.initialize(self, longest_flood_event=30)
        self.sfincs.get_return_period_maps()

    def evaluate(self, *args, **kwargs) -> None:
        print("Evaluating model...")
        self.evaluate = Evaluate(self)
        self.evaluate.run(*args, **kwargs)

    @property
    def current_day_of_year(self) -> int:
        """Gets the current day of the year.

        Returns:
            day: current day of the year.
        """
        return self.current_time.timetuple().tm_yday

    @property
    def current_time_unix_s(self) -> int:
        """Gets the current time in unix seconds.

        Returns:
            time: current time in unix seconds.
        """
        return np.datetime64(self.current_time, "s").astype(np.int64).item()

    @property
    def simulation_root(self) -> Path:
        """Gets the simulation root.

        Returns:
            simulation_root: Path of the simulation root.
        """
        folder = Path("simulation_root") / self.run_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def simulation_root_spinup(self) -> Path:
        """Gets the simulation root of the spinup.

        Returns:
            simulation_root: Path of the simulation root.
        """
        folder = Path("simulation_root") / "spinup"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def run_name(self):
        if self.mode == "w" and self.in_spinup:
            return self.config["general"]["spinup_name"]
        else:
            if "name" in self.config["general"]:
                return self.config["general"]["name"]
            else:
                print(
                    'No "name" specified in config file under general. Using "default".'
                )
                return "default"

    @property
    def multiverse_name(self):
        return self._multiverse_name

    @multiverse_name.setter
    def multiverse_name(self, value):
        self._multiverse_name = str(value) if value is not None else None

    @property
    def output_folder(self):
        return Path(self.config["general"]["output_folder"])

    @property
    def input_folder(self):
        return Path(self.config["general"]["input_folder"])

    @property
    def bin_folder(self) -> Path:
        return Path(os.environ.get("GEB_PACKAGE_DIR")) / "bin"

    @property
    def crs(self):
        return 4326

    @property
    def bounds(self):
        return self.mask.total_bounds

    @property
    def xmin(self):
        return self.bounds[0]

    @property
    def xmax(self):
        return self.bounds[2]

    @property
    def ymin(self):
        return self.bounds[1]

    @property
    def ymax(self):
        return self.bounds[3]

    def close(self) -> None:
        """Finalizes the model."""
        if (
            self.mode == "w"
            and hasattr(self, "simulate_hydrology")
            and self.simulate_hydrology
        ):
            Hydrology.finalize(self.hydrology)

            from geb.workflows.io import all_async_readers

            for reader in all_async_readers:
                reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
