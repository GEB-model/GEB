"""The main GEB model class. This class is used to initialize and run the model."""

import copy
import datetime
import os
from pathlib import Path
from time import time
from types import TracebackType
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from honeybees.model import Model as ABM_Model

from geb.agents import Agents
from geb.hazards.driver import HazardDriver
from geb.hazards.floods.workflows.construct_storm_surge_hydrographs import (
    generate_storm_surge_hydrographs,
)
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
    ) -> None:
        """Initialize the GEB model.

        Args:
            config: A dictionary containing the model configuration.
            files: A dictionary containing the paths to the input files.
            mode: Run model writing mode "w", or reading mode "r". Defaults to "w" for writing.
                If "r", the model will not write any output files, but will read from existing files.
            timing: Whether to log timing of modules. Defaults to False.

        Raises:
            ValueError: If the mode is not 'r' or 'w'.
        """
        self.timing = timing  # whether to log timing of modules
        self.mode = mode  # mode of the model, either 'r' (read) or 'w' (write)
        if self.mode not in ["r", "w"]:
            raise ValueError(
                "Mode must be either 'r' (read) or 'w' (write)"
            )  # validate mode

        Module.__init__(self, self, create_var=False)  # initialize the Module class

        self._multiverse_name = None  # name of the multiverse, if any

        self.config = config  # model configuration

        self.files = copy.deepcopy(
            files
        )  # make a deep copy to avoid issues when the model is initialized multiple times
        if "geoms" in self.files:
            # geoms was renamed to geom in the file library. To upgrade old models,
            # we check if "geoms" is in the files and rename it to "geom"
            # this line can be removed in august 2026 (also in geb/build/__init__.py)
            self.files["geom"] = self.files.pop("geoms")  # upgrade old models

        for data in self.files.values():
            for key, value in data.items():
                data[key] = self.input_folder / value  # make paths absolute

        self.mask = load_geom(self.files["geom"]["mask"])  # load the model mask

        self.store = Store(self)
        self.forcing = Forcing(self)

        self.evaluator = Evaluate(self)  # initialize the evaluator

        self.plantFATE = []  # Empty list to hold plantFATE models. If forests are not used, this will be empty

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
        self,
        variables: list[str],
        forecast_issue_datetime: datetime.datetime,
        return_mean_discharge: Literal[True],
    ) -> dict[Any, float]: ...

    @overload
    def multiverse(
        self,
        variables: list[str],
        forecast_issue_datetime: datetime.datetime,
        return_mean_discharge: Literal[False],
    ) -> None: ...

    def multiverse(
        self,
        variables: list[str],
        forecast_issue_datetime: datetime.datetime,
        return_mean_discharge: bool = False,
    ) -> None | dict[Any, float]:
        """Run the model in a multiverse mode, where different forecast members are used to run the model.

        This function first saves the current state of the model to a temporary location.
        Then, for each forecast member, it sets the precipitation forcing to the forecast data,
        runs the model to the end of the forecast period, and optionally calculates the mean discharge.
        After all forecast members have been processed, the model state is restored to the original state
        before the function was called.

        The file where the forecast data is stored should have the following format:
        `data/forecasts/YYYYMMDDTHHMMSS.zarr`, where `YYYYMMDDTHHMMSS` is the datetime of the forecast.
        The forecast data should be a zarr file with a `member` dimension, where each member is a different forecast.

        All other dimensions should be the same as the original forcing data. Units are also expected to be the same.

        Args:
            variables: List of variables to use in the multiverse. Currently, only `pr_hourly` is supported.
            forecast_issue_datetime: Datetime that the forecast was issued.
            return_mean_discharge: Whether to return the mean discharge for each forecast member. This is
                mostly useful for testing purposes.

        Returns:
            If `return_mean_discharge` is True, a dictionary with the mean discharge for each forecast member is returned.
            Otherwise, None is returned.

        Raises:
            ValueError: If the x and y dimensions of the forecast data are not the same as the original data.

        """
        # copy current state of timestep and time
        store_timestep: int = copy.copy(self.current_timestep)  # store current timestep
        store_n_timesteps: int = copy.copy(self.n_timesteps)  # store n_timesteps

        # set a folder to store the initial state of the multiverse
        store_location: Path = (
            self.simulation_root / "multiverse" / "forecast_initial_state"
        )  # create a temporary folder for the multiverse
        self.store.save(store_location)  # save the current state of the model

        forecasts: xr.DataArray = open_zarr(  # open the forecast data
            self.input_folder
            / "other"
            / "forecasts"
            / "ECMWF"
            / f"{'pr_hourly'}_{forecast_issue_datetime.strftime('%Y%m%dT%H%M%S')}.zarr"
        )
        forecast_lead_time = pd.to_datetime(forecasts.time[-1].item()) - pd.to_datetime(
            forecasts.time[0].item()
        )  # calculate the lead time of the forecast

        forecast_end_date = round_up_to_start_of_next_day_unless_midnight(
            pd.to_datetime(forecasts.time[-1].item()).to_pydatetime()
        ).date()  # calculate the end date of the forecast
        self.n_timesteps = (
            forecast_end_date - self.start_time.date()
        ).days  # set the number of timesteps to the end of the forecast
        print(
            f"Hydrological model run starts at {self.start_time}. SFINCS and forecasts will be active from {forecast_issue_datetime} with max. lead time of {forecast_lead_time} days"
        )

        original_data: dict[
            str, xr.DataArray
        ] = {}  # Save original data arrays for all variables to restore later
        for var in variables:
            original_data[var] = self.forcing[var]  # store original forcing data

        if return_mean_discharge:
            mean_discharge: dict[
                Any, float
            ] = {}  # dictionary to store mean discharge for each member

        self.forecast_issue_date = forecast_issue_datetime.strftime(
            "%Y%m%dT%H%M%S"
        )  # set the forecast issue date

        for member in forecasts.member:  # loop over all forecast members
            print(member)
            self.multiverse_name = (
                member.item()
            )  # set the multiverse name to the member name
            for var in variables:
                print(
                    f"Entering the multiverse space for member {member.item()} and variable {var}"
                )  # debugging print
                forecasts: xr.DataArray = open_zarr(
                    self.input_folder
                    / "other"
                    / "forecasts"
                    / "ECMWF"
                    / f"{var}_{forecast_issue_datetime.strftime('%Y%m%dT%H%M%S')}.zarr"
                )  # open the forecast data for the variable

                forecast_member: xr.DataArray = forecasts.sel(
                    member=member
                )  # select the forecast member

                # check if the x and y dimensions of the forecast data is exactly the same as the original data
                if not np.array_equal(
                    forecasts.x, original_data[var].x
                ) or not np.array_equal(forecasts.y, original_data[var].y):
                    raise ValueError(
                        f"The x and y dimensions of the forecast data for variable {var} are not the same as the original data. Cannot run multiverse."
                    )  # raise an error if the dimensions are not the same

                # Clip the original precipitation data to the start of the forecast
                # Therefore we take the start of the forecast and subtract one second
                # to ensure that the original precipitation data does not overlap with the forecast
                original_data_clipped_to_start_of_forecast: xr.DataArray = (
                    original_data[var].sel(
                        time=slice(
                            None, (forecast_member.time[0] - pd.Timedelta(seconds=1))
                        )
                    )
                )  # clip the original data to the start of the forecast

                observed_and_forecasted_combined: xr.DataArray = xr.concat(
                    [original_data_clipped_to_start_of_forecast, forecast_member],
                    dim="time",
                )  # Concatenate the original forcing data with the forecast data along time dimension

                self.model.forcing[var] = (
                    observed_and_forecasted_combined  # set the forcing data to the combined data
                )

            print(f"Running forecast member {member.item()}")  # debugging print
            self.step_to_end()  # steps to end of forecast period as defined in self.n_timesteps

            if return_mean_discharge:
                mean_discharge[member.item()] = (
                    self.hydrology.routing.grid.var.discharge_m3_s.mean()
                ).item()  # calculate the mean discharge for the member

            self.restore(
                store_location=store_location,
                timestep=store_timestep,
                n_timesteps=store_n_timesteps,
            )  # restore the initial state of the multiverse

        print("Forecast finished, restoring all conditions...")  # debugging print

        for var in variables:
            self.forcing[var] = original_data[
                var
            ]  # restore the forcing data arrays, step out of the multiverse
        self.multiverse_name: None = None  # reset the multiverse name

        if return_mean_discharge:
            return mean_discharge  # return the mean discharge for each member
        else:
            return None  # nothing to return

    def step(self) -> None:
        """Forward the model by one timestep.

        If configured, this function will also run the model in multiverse mode
        for the current timestep, using forecast data if available.

        """
        # only if forecasts is used, and if we are not already in multiverse (avoiding infinite recursion)
        # and if the current date is in the list of forecast days
        if (
            self.config["general"]["forecasts"]["use"]
            and self.multiverse_name
            is None  # only start multiverse if not already in one
            and self.current_time.date()
        ):
            forecast_files: list[Path] = list(
                (self.input_folder / "other" / "forecasts" / "ECMWF").glob("*.zarr")
            )  # get all forecast files in the input folder
            forecast_issue_dates: list[
                datetime.date
            ] = []  # list to store forecast issue dates
            for f in forecast_files:
                datetime_str = f.stem.split("_")[
                    -1
                ]  # extract the datetime string from the filename
                if (
                    datetime_str.replace("T", "").replace(":", "").isdigit()
                ):  # Check if datetime string contains only digits, T, and colons (valid format)
                    dt = datetime.datetime.strptime(
                        datetime_str, "%Y%m%dT%H%M%S"
                    )  # convert the string to a datetime object
                    forecast_issue_dates.append(dt)  # append the date to the list
                else:
                    print(
                        f"Warning: Forecast file {f.name} does not have a valid datetime format. Expected format: 'YYYYMMDDTHHMMSS'. Skipping this file."
                    )  # print a warning if the format is invalid

            forecast_issue_dates = list(
                set(forecast_issue_dates)
            )  # only keep unique dates

            if self.config["general"]["forecasts"]["only_rainfall"]:
                variables = ["pr_hourly"]  # only rainfall is currently implemented
            else:
                print("Other variables than rainfall not yet implemented.")

            for dt in forecast_issue_dates:
                if (
                    dt == self.current_time
                ):  # change to include hours (for when we move to hourly)
                    forecast_datetime = datetime.datetime.combine(
                        dt, datetime.time(0)
                    )  # Convert date back to datetime for the multiverse method

                    self.multiverse(
                        variables=variables,
                        forecast_issue_datetime=forecast_datetime,
                        return_mean_discharge=True,
                    )  # run the multiverse for the current timestep

            if self.config["agent_settings"]["households"]["warning_response"]:
                self.agents.households.warning_strategy_1()
                # simulate household response to warning
                # self.agents.households.infrastructure_warning_strategy()

        t0 = time()  # start timing
        self.agents.step()  # step the agents
        if self.simulate_hydrology:
            self.hydrology.step()  # step the hydrology

        HazardDriver.step(self)  # step the hazards

        self.report(locals())  # report the current state of the model

        t1 = time()  # end timing
        print(
            f"{self.multiverse_name + ' - ' if self.multiverse_name is not None else ''}finished {self.current_time} ({round(t1 - t0, 4)}s)",
            flush=True,
        )  # print the time taken for the step

        self.current_timestep += 1  # increment the timestep

    def _initialize(
        self,
        create_reporter: bool,
        current_time: datetime.datetime,
        n_timesteps: int,
        timestep_length: datetime.timedelta | relativedelta,
        in_spinup: bool = False,
        simulate_hydrology: bool = True,
        clean_report_folder: bool = False,
        load_data_from_store: bool = False,
    ) -> None:
        """Initializes the model.

        Args:
            create_reporter: Whether to create a reporter instance.
            current_time: Current time of the model.
            n_timesteps: Number of timesteps to run the model for.
            timestep_length: Length of each timestep.
            in_spinup: Whether the model is in spinup mode.
            simulate_hydrology: Whether to simulate hydrology.
            clean_report_folder: Whether to clean the report folder before creating a new reporter.
            load_data_from_store: Whether to load data from the store.

        """
        self.in_spinup = in_spinup
        self.simulate_hydrology = simulate_hydrology

        self.regions = load_geom(self.files["geom"]["regions"])

        self.output_folder.mkdir(parents=True, exist_ok=True)

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

        # in spinup mode, save the spinup time range to the store for later verification
        # in run mode, verify that the spinup time range matches the stored time range
        if in_spinup:
            self._store_spinup_time_range()
        else:
            self._verify_spinup_time_range()

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
        """Run the model for the entire period, and export water table in case of spinup scenario.

        Args:
            initialize_only: If True, only initialize the model without running it.

        Raises:
            FileNotFoundError: If the initial conditions folder does not exist. Spinup is required before running the model.
        """
        if not self.store.path.exists():
            raise FileNotFoundError(
                f"The initial conditions folder ({self.store.path.resolve()}) does not exist. Spinup is required before running the model. Please run the spinup first."
            )

        current_time: datetime.datetime = self.run_start
        end_time: datetime.datetime = self.run_end

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
        """Run the model in yearly mode, where timesteps are yearly rather than daily.

        This depends on a spinup run that was run in daily mode.

        Notes:
            Cannot be run in combination with hydrology simulation.
            This mode is experimential and is not fully tested.

        Raises:
            ValueError: If the start or end time is not at the beginning or end of a year, respectively.
            ValueError: If flood simulation is enabled in the config, as this is not compatible with yearly mode.
        """
        current_time: datetime.datetime = self.run_start
        end_time: datetime.datetime = self.run_end

        if self.config["hazards"]["floods"]["simulate"] is True:
            raise ValueError(
                "Yearly mode is not compatible with flood simulation. Please set 'simulate' to False in the config."
            )

        if not (current_time.month == 1 and current_time.day == 1):
            raise ValueError(
                "In yearly mode start time should be the first day of the year"
            )

        if not (end_time.month == 12 and end_time.day == 31):
            raise ValueError(
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
        """Run the model for the spinup period.

        Also reports all data at the end of the spinup period, and saves the model state to the store,
        so that it can be used as initial conditions for the actual model run.

        Args:
            initialize_only: If True, only initialize the model without running it.
        """
        # set the start and end time for the spinup. The end of the spinup is the start of the actual model run
        current_time = self.spinup_start
        end_time_exclusive = self.run_start

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

    def _store_spinup_time_range(self) -> None:
        """Store the spinup time range in the variable store.

        This is used in the run and estimate_return_periods methods to verify that the spinup time
        range matches the stored time range. If they do not match, an error is raised, because this
        indicates that the model configuration has changed since the spinup was run,
        and can lead to undefined or unexpected behavior.
        """
        self.var._spinup_start = self.spinup_start
        self.var._run_start = self.run_start

    def _verify_spinup_time_range(self) -> None:
        """Verify that the spinup time range matches the stored time range.

        If they do not match, an error is raised, because this indicates that the model configuration
        has changed since the spinup was run, and can lead to undefined or unexpected behavior.

        Raises:
            ValueError: If the spinup start or run start time does not match the stored time range.
        """
        if self.var._spinup_start != self.spinup_start:
            raise ValueError(
                f"Spinup start time does not match the stored time range. Stored: {self.var._spinup_start}, Configured: {self.spinup_start}"
            )

        if self.var._run_start != self.run_start:
            raise ValueError(
                f"Run start time does not match the stored time range. Stored: {self.var._run_start}, Configured: {self.run_start}"
            )

    def estimate_return_periods(self) -> None:
        """Estimate flood maps for different return periods."""
        current_time: datetime.datetime = self.run_start
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

        HazardDriver.initialize(self, longest_flood_event_in_days=30)
        # ugly switch to determine whether model has coastal basins
        subbasins = load_geom(self.model.files["geom"]["routing/subbasins"])
        if subbasins["is_coastal_basin"].any():
            generate_storm_surge_hydrographs(self)
            rp_maps_coastal = self.sfincs.get_coastal_return_period_maps()
        else:
            rp_maps_coastal = None
        rp_maps_riverine = self.sfincs.get_riverine_return_period_maps()
        self.sfincs.merge_return_period_maps(rp_maps_coastal, rp_maps_riverine)

    def evaluate(self, *args: Any, **kwargs: Any) -> None:
        """Call the evaluator to evaluate the model results."""
        print("Evaluating model...")
        self.evaluator.run(*args, **kwargs)

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
    def run_name(self) -> str:
        """Get the name of the current model spinup or run.

        Returns:
            Name of the current model run. If in spinup mode, the spinup name is returned.
        """
        if self.in_spinup:
            return self.config["general"]["spinup_name"]
        else:
            return self.config["general"]["name"]

    @property
    def multiverse_name(self) -> str | None:
        """To explore different model futures, GEB can be run in a multiverse mode.

        In this mode, a number of timesteps can be run with different input data (e.g. different precipitation forecasts).
        The multiverse_name is used to identify the different model futures. It is typically set to the forecast member name.

        Returns:
            Name of the multiverse. If None, the model is not in multiverse mode.
        """
        return self._multiverse_name

    @multiverse_name.setter
    def multiverse_name(self, value: str | None) -> None:
        """To explore different model futures, GEB can be run in a multiverse mode.

        In this mode, a number of timesteps can be run with different input data (e.g. different precipitation forecasts).
        The multiverse_name is used to identify the different model futures. It is typically set to the forecast member name.

        Args:
            value: Name of the multiverse. If None, the model is not in multiverse mode.
        """
        self._multiverse_name = str(value) if value is not None else None

    @property
    def forecast_issue_date(self) -> str | None:
        """Get the forecast issue date as a string in the format YYYYMMDD.

        This is used to identify the forecast data used in the multiverse mode.

        Returns:
            Forecast issue date as a string in the format YYYYMMDD. If None, the model
            is not in multiverse mode.

        """
        return self._forecast_issue_date

    @forecast_issue_date.setter
    def forecast_issue_date(self, value: str | None) -> None:
        """To explore different model futures, GEB can be run in a multiverse mode.

        In this mode, a number of timesteps can be run with different input data (e.g. different precipitation forecasts).
        The forecast_issue_date is used to identify the forecast data used in the multiverse mode.

        Args:
            value: Forecast issue date as a string in the format YYYYMMDD. If None,
            the model is not in multiverse mode.
        """
        self._forecast_issue_date = str(value) if value is not None else None

    @property
    def output_folder(self) -> Path:
        """Get the folder where the output files will be saved.

        Returns:
            Path to the folder where output files will be saved.
        """
        return Path(self.config["general"]["output_folder"])

    @property
    def input_folder(self) -> Path:
        """Get the folder where the input files are located.

        Returns:
            Path to the folder containing input files.
        """
        return Path(self.config["general"]["input_folder"])

    @property
    def bin_folder(self) -> Path:
        """Get the folder where the GEB binaries, such as MODFLOW and TBB, are located.

        Returns:
            Path to the folder containing GEB binaries.
        """
        return Path(os.environ.get("GEB_PACKAGE_DIR")) / "bin"

    @property
    def crs(self) -> int:
        """Get the coordinate reference system (CRS) of the model."""
        return 4326

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get the bounding box of the model's mask.

        Returns:
            A tuple representing the bounding box in the format (minx, miny, maxx, maxy).
        """
        total_bounds = self.mask.total_bounds
        return (total_bounds[0], total_bounds[1], total_bounds[2], total_bounds[3])

    @property
    def xmin(self) -> float:
        """Get the minimum x-coordinate of the model's bounding box.

        Returns:
            Minimum x-coordinate of the bounding box.
        """
        return self.bounds[0]

    @property
    def xmax(self) -> float:
        """Get the maximum x-coordinate of the model's bounding box.

        Returns:
            Maximum x-coordinate of the bounding box.
        """
        return self.bounds[2]

    @property
    def ymin(self) -> float:
        """Get the minimum y-coordinate of the model's bounding box.

        Returns:
            Minimum y-coordinate of the bounding box.
        """
        return self.bounds[1]

    @property
    def ymax(self) -> float:
        """Get the maximum y-coordinate of the model's bounding box.

        Returns:
            Maximum y-coordinate of the bounding box.
        """
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

    def __enter__(self) -> "GEBModel":
        """Enters the context of the model.

        Returns:
            The model instance itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the context of the model, ensuring proper cleanup.

        Args:
            exc_type: The type of exception raised (if any).
            exc_val: The exception instance raised (if any).
            exc_tb: The traceback of the exception raised (if any).
        """
        self.close()

    def create_datetime(self, date: datetime.date) -> datetime.datetime:
        """Create a datetime object from a date with time set to midnight.

        Args:
            date:  Date object to convert to datetime.

        Returns:
            Datetime object with time set to midnight.
        """
        return datetime.datetime.combine(date, datetime.time(0))

    @property
    def spinup_start(self) -> datetime.datetime:
        """Get the start time of the spinup period.

        Returns:
            Datetime object representing the start of the spinup period.
        """
        return self.create_datetime(self.config["general"]["spinup_time"])

    @property
    def run_start(self) -> datetime.datetime:
        """Get the start time of the model run.

        Returns:
            Datetime object representing the start of the model run.
        """
        return self.create_datetime(self.config["general"]["start_time"])

    @property
    def run_end(self) -> datetime.datetime:
        """Get the end time of the model run.

        Returns:
            Datetime object representing the end of the model run.
        """
        return self.create_datetime(self.config["general"]["end_time"])

    @property
    def name(self) -> str:
        """This is the name of this module, NOT the model or model run.

        Used to store variables in the store.

        Returns:
            Name of the module.
        """
        return ""
