"""The main GEB model class. This class is used to initialize and run the model."""

import copy
import datetime
import logging
from pathlib import Path
from time import time
from types import TracebackType
from typing import Any, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta

from geb import GEB_PACKAGE_DIR
from geb.agents import Agents
from geb.hazards.driver import HazardDriver
from geb.hazards.floods.workflows.construct_storm_surge_hydrographs import (
    generate_storm_surge_hydrographs,
)
from geb.module import Module
from geb.reporter import Reporter
from geb.store import Store
from geb.workflows.io import read_dict, read_geom, read_zarr

from .evaluate import Evaluate
from .forcing import Forcing
from .hydrology import Hydrology


class GEBModel(Module, HazardDriver):
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
        self.config: dict[str, Any] = config  # model configuration
        self.logger: logging.Logger = self.create_logger()

        self.timing = timing  # whether to log timing of modules
        self.mode = mode  # mode of the model, either 'r' (read) or 'w' (write)
        if self.mode not in ["r", "w"]:
            raise ValueError(
                "Mode must be either 'r' (read) or 'w' (write)"
            )  # validate mode

        Module.__init__(self, self, create_var=False)  # initialize the Module class

        self._multiverse_name = None  # name of the multiverse, if any

        self.files = copy.deepcopy(
            files
        )  # make a deep copy to avoid issues when the model is initialized multiple times
        for data in self.files.values():
            for key, value in data.items():
                data[key] = self.input_folder / value  # make paths absolute

        self.mask = read_geom(self.files["geom"]["mask"])  # load the model mask

        self.store = Store(self)

        self.evaluator = Evaluate(self)  # initialize the evaluator

        self.plantFATE = []  # Empty list to hold plantFATE models. If forests are not used, this will be empty

    def restore(self, store_location: Path, timestep: int, n_timesteps: int) -> None:
        """Restore the model state to the original state given by the function input.

        Args:
            store_location: Location of the store to restore the model state from.
            timestep: timestep to restore the model state to.
            n_timesteps: number of timesteps (i.e., the final timestep) to restore the model state to.
        """
        self.store.load(store_location)

        # restore the heads of the groundwater model
        self.hydrology.groundwater.modflow.restore(self.hydrology.grid.var.heads)

        self.current_timestep = timestep
        self.n_timesteps = n_timesteps

    @overload
    def multiverse(
        self,
        forecast_issue_datetime: datetime.datetime,
        return_mean_discharge: bool = True,
    ) -> dict[Any, float]: ...

    @overload
    def multiverse(
        self,
        forecast_issue_datetime: datetime.datetime,
        return_mean_discharge: bool = False,
    ) -> None: ...

    def multiverse(
        self,
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
            forecast_issue_datetime: Datetime that the forecast was issued.
            return_mean_discharge: Whether to return the mean discharge for each forecast member. This is
                mostly useful for testing purposes.

        Returns:
            If `return_mean_discharge` is True, a dictionary with the mean discharge for each forecast member is returned.
            Otherwise, None is returned.

        Raises:
            ValueError: If forecast members or datetimes do not match between variables.

        """
        # copy current state of timestep and time
        store_timestep: int = copy.copy(self.current_timestep)  # store current timestep
        store_n_timesteps: int = copy.copy(self.n_timesteps)  # store n_timesteps

        # set a folder to store the initial state of the multiverse
        store_location: Path = (
            self.simulation_root / "multiverse" / "forecast_initial_state"
        )  # create a temporary folder for the multiverse
        self.store.save(store_location)  # save the current state of the model

        original_is_activated: bool = (
            self.reporter.is_activated
        )  # store original reporter state
        self.reporter.is_activated = False  # disable reporting during multiverse runs

        if return_mean_discharge:
            mean_discharge: dict[
                Any, float
            ] = {}  # dictionary to store mean discharge for each member

        # load all zarr files for all forecast variables for the given issue date
        forecast_members: list[str] | None = None
        forecast_end_dt: datetime.datetime | None = None
        forecast_data: dict[str, xr.DataArray] = {}
        for loader_name, loader in self.forcing.loaders.items():
            if loader.supports_forecast:
                # open one forecast to see the number of members
                forecast_data[loader_name] = read_zarr(
                    self.input_folder
                    / "other"
                    / "forecasts"
                    / self.config["general"]["forecasts"]["provider"]
                    / f"{loader_name}_{forecast_issue_datetime.strftime('%Y%m%dT%H%M%S')}.zarr"
                )  # open the forecast data for the variable
                # these are the forecast members to loop over
                variable_forecast_members: list[str] = [
                    i.item() for i in forecast_data[loader_name].member.values
                ]
                variable_forecast_end_dt = (
                    forecast_data[loader_name].time.values[-1]
                ).item()  # get the end datetime of the forecast
                if forecast_members is None:
                    forecast_members: list[str] = variable_forecast_members
                    forecast_end_dt = variable_forecast_end_dt
                else:
                    if forecast_members != variable_forecast_members:
                        raise ValueError(
                            "Forecast members do not match between variables."
                        )
                    if forecast_end_dt != variable_forecast_end_dt:
                        raise ValueError(
                            "Forecast end datetimes do not match between variables."
                        )

        assert len(forecast_data) > 0, (
            "No forecast data found for any variable. Please check the forecast files."
        )  # ensure that forecast data was found
        assert forecast_members is not None, (
            "Forecast members could not be determined. Please check the forecast files."
        )  # ensure that forecast members were found
        assert forecast_end_dt is not None, (
            "Forecast end datetime could not be determined. Please check the forecast files."
        )  # ensure that forecast end datetime was found

        forecast_end_dt = pd.to_datetime(forecast_end_dt).to_pydatetime()
        forecast_end_day = forecast_end_dt.date()

        self.n_timesteps = (
            forecast_end_day - self.simulation_start.date()
        ).days  # set the number of timesteps to the end of the forecast

        for member in forecast_members:  # loop over all forecast members
            self.multiverse_name: str = f"forecast_{forecast_issue_datetime.strftime('%Y%m%dT%H%M%S')}/member_{member}"  # set the multiverse name to the member name

            for loader_name, loader in self.forcing.loaders.items():
                if loader.supports_forecast:
                    loader.set_forecast(
                        forecast_issue_datetime=forecast_issue_datetime,
                        da=forecast_data[loader_name].sel(member=member),
                    )

            print(f"Running forecast member {member}")  # debugging print
            self.step_to_end()  # steps to end of forecast period as defined in self.n_timesteps

            if return_mean_discharge:
                mean_discharge[member] = (
                    self.hydrology.routing.grid.var.discharge_m3_s.mean()
                ).item()  # calculate the mean discharge for the member

            # restore the model to the state before the forecast for the next member
            # so the n_timesteps is restored to the number of timesteps at
            # the end of the forecast period
            self.restore(
                store_location=store_location,
                timestep=store_timestep,
                n_timesteps=self.n_timesteps,
            )  # restore the initial state of the multiverse

        print("Forecast finished, restoring all conditions...")  # debugging print

        # after ALL forecast members have been processed, restore the model to the state before the multiverse
        # so the n_timesteps is restored to the number of the full model run
        self.restore(
            store_location=store_location,
            timestep=store_timestep,
            n_timesteps=store_n_timesteps,
        )  # restore the initial state of the multiverse

        self.reporter.is_activated = (
            original_is_activated  # restore original reporter state
        )

        # after all forecast members have been processed, restore the original forcing data
        for loader in self.forcing.loaders.values():
            if loader.supports_forecast:
                loader.unset_forecast()  # unset forecast mode

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
                (
                    self.input_folder
                    / "other"
                    / "forecasts"
                    / self.config["general"]["forecasts"]["provider"]
                ).glob("*.zarr")
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

            for dt in forecast_issue_dates:
                if (
                    dt == self.current_time
                ):  # change to include hours (for when we move to hourly)
                    forecast_datetime = datetime.datetime.combine(
                        dt, datetime.time(0)
                    )  # Convert date back to datetime for the multiverse method

                    self.multiverse(
                        forecast_issue_datetime=forecast_datetime,
                        return_mean_discharge=True,
                    )  # run the multiverse for the current timestep

                    # after the multiverse has run all members for one day, if warning response is enabled, run the warning system
                    if self.config["agent_settings"]["households"]["warning_response"]:
                        print(
                            f"Running flood early warning system for date time {self.current_time.isoformat()}..."
                        )
                        self.agents.households.create_flood_probability_maps(
                            date_time=self.current_time, strategy=1, exceedance=True
                        )
                        self.agents.households.water_level_warning_strategy(
                            date_time=self.current_time
                        )
                        self.agents.households.critical_infrastructure_warning_strategy(
                            date_time=self.current_time
                        )
                        self.agents.households.household_decision_making(
                            date_time=self.current_time
                        )
                        self.agents.households.update_households_geodataframe_w_warning_variables(
                            date_time=self.current_time
                        )

        t0 = time()
        self.agents.step()
        if self.simulate_hydrology:
            self.hydrology.step()

        HazardDriver.step(self)

        self.report(locals())

        t1 = time()
        print(
            f"{self.multiverse_name + ' - ' if self.multiverse_name is not None else ''}finished {self.current_time} ({round(t1 - t0, 4)}s)",
            flush=True,
        )

        self.current_timestep += 1

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
        omit: None | str = None,
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
            omit: Name of the bucket to omit when loading data from the store.

        """
        self.in_spinup = in_spinup
        self.simulate_hydrology = simulate_hydrology

        self.timestep_length = timestep_length
        self.n_timesteps = n_timesteps
        self.current_timestep = 0

        self.regions: gpd.GeoDataFrame = read_geom(self.files["geom"]["regions"])

        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.hydrology: Hydrology = Hydrology(self)

        HazardDriver.__init__(self)

        self.agents = Agents(self)

        if load_data_from_store:
            self.store.load(omit=omit)

        # in spinup mode, save the spinup time range to the store for later verification
        # in run mode, verify that the spinup time range matches the stored time range
        if in_spinup:
            self._store_spinup_time_range()
        elif load_data_from_store:
            self._verify_spinup_time_range()

        if self.simulate_hydrology:
            self.forcing: Forcing = Forcing(self)
            self.hydrology.routing.set_router()
            self.hydrology.groundwater.initalize_modflow_model()
            self.hydrology.landsurface.set_global_variables()

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

        self.check_time_range()
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

    def refresh_agent_attributes(self, agent_type: str = "households") -> None:
        """Initiate the model to update household adaptation attributes to pre-spinup state after an updated build or adding/ renaming of agent variables.

        This function is only included for development purposes.

        Args:
            agent_type: Type of agent to refresh attributes for. Examples: "households", "crop_farmers", etc.

        """
        # set the start and end time for the spinup. The end of the spinup is the start of the actual model run
        current_time = self.spinup_start
        end_time_exclusive = self.run_start

        timestep_length = datetime.timedelta(days=1)
        n_timesteps = (end_time_exclusive - current_time) / timestep_length
        assert n_timesteps.is_integer()
        n_timesteps = int(n_timesteps)
        assert n_timesteps > 0, "End time is before or identical to start time"

        # create var bucket
        self.var = self.store.create_bucket("var")

        # initialize the model
        self._initialize(
            create_reporter=True,
            current_time=current_time,
            n_timesteps=n_timesteps,
            timestep_length=datetime.timedelta(days=1),
            load_data_from_store=False,
            clean_report_folder=False,
            in_spinup=True,
        )

        # save initial household attributes
        print(f"Refreshing household attributes for {agent_type}...")
        path: Path = self.store.path
        name = getattr(self.agents, agent_type).name
        self.logger.debug(f"Saving {name}.var")
        bucket = self.store.buckets[f"{name}.var"]
        bucket.save(path / f"{name}.var")

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

        self.check_time_range()
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
            # omit="agents",
            simulate_hydrology=True,
            clean_report_folder=False,
        )

        # ugly switch to determine whether model has coastal basins
        subbasins = read_geom(self.model.files["geom"]["routing/subbasins"])
        if subbasins["is_coastal_basin"].any():
            generate_storm_surge_hydrographs(self)

        self.floods.get_return_period_maps()

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
        folder = Path(self.config["general"]["simulation_root"]) / self.run_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def simulation_root_spinup(self) -> Path:
        """Gets the simulation root of the spinup.

        Returns:
            simulation_root: Path of the simulation root.
        """
        folder = Path(self.config["general"]["simulation_root"]) / "spinup"
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
        return GEB_PACKAGE_DIR / "bin"

    @property
    def diagnostics_folder(self) -> Path:
        """Get the folder where diagnostic output files will be saved.

        Returns:
            Path to the folder where diagnostic output files will be saved.
        """
        folder = self.output_folder / "diagnostics"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

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

            # Close all async forcing readers
            if hasattr(self, "forcing"):
                for forcing_loader in self.forcing._loaders.values():
                    if hasattr(forcing_loader, "reader"):
                        forcing_loader.reader.close()

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

    def check_time_range(self) -> None:
        """Check that the model's spinup and run time ranges are within the model build time range.

        Raises:
            ValueError: If the spinup start date is before the model build start date.
            ValueError: If the run end date is after the model build end date.
        """
        model_build_time_range: dict[str, str] = read_dict(
            self.files["dict"]["model_time_range"]
        )

        model_build_start_date = model_build_time_range["start_date"]

        # TODO: Remove in 2026
        if isinstance(model_build_start_date, str):
            model_build_start_date: datetime.datetime = datetime.datetime.fromisoformat(
                model_build_start_date
            )
        model_build_end_date = model_build_time_range["end_date"]

        # TODO: Remove in 2026
        if isinstance(model_build_end_date, str):
            model_build_end_date: datetime.datetime = datetime.datetime.fromisoformat(
                model_build_end_date
            )

        if self.spinup_start.date() < model_build_start_date:
            raise ValueError(
                "Spinup start date cannot be before model build start date. Adjust the time range in your build configuration and rebuild the model or adjust the spinup time of the model."
            )

        if self.run_end.date() > model_build_end_date:
            raise ValueError(
                "Run end date cannot be after model build end date. Adjust the time range in your build configuration and rebuild the model or adjust the simulation end time of the model."
            )

    @property
    def spinup_start(self) -> datetime.datetime:
        """Get the start time of the spinup period.

        Returns:
            Datetime object representing the start of the spinup period.
        """
        return self.create_datetime(self.config["general"]["spinup_time"])

    @property
    def spinup_end(self) -> datetime.datetime:
        """Get the end time of the spinup period.

        Returns:
            Datetime object representing the end of the spinup period.
        """
        return (
            self.create_datetime(self.config["general"]["start_time"])
            - self.timestep_length
        )

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
    def simulation_start(self) -> datetime.datetime:
        """Get the start of the current simulation, depending on whether in spinup or run mode.

        Returns:
            Datetime object representing the start of the current simulation.
        """
        if self.in_spinup:
            return self.spinup_start
        else:
            return self.run_start

    @property
    def simulation_end(self) -> datetime.datetime:
        """Get the end of the current simulation, depending on whether in spinup or run mode.

        Returns:
            Datetime object representing the end of the current simulation.
        """
        return self.simulation_start + (self.n_timesteps - 1) * self.timestep_length

    @property
    def current_timestep(self) -> int:
        """The current model timestep.

        Returns:
            current model timestep
        """
        return self._current_timestep

    @current_timestep.setter
    def current_timestep(self, timestep: int) -> None:
        """Set the current model timestep.

        Args:
            timestep: current model timestep
        """
        self._current_timestep = timestep

    @property
    def current_time(self) -> datetime.datetime:
        """Get the current model time.

        Returns:
            Current model time

        Raises:
            AttributeError: If `timestep_length` or `simulation_start` are not initialized.
        """
        # Defensive check: ensure required attributes are initialized
        if not hasattr(self, "timestep_length") or not hasattr(
            self, "simulation_start"
        ):
            raise AttributeError(
                "Cannot compute current_time: 'timestep_length' and/or 'simulation_start' are not initialized. "
                "Ensure the model is fully initialized before accessing current_time."
            )
        return self.simulation_start + self.current_timestep * self.timestep_length

    @property
    def name(self) -> str:
        """This is the name of this module, NOT the model or model run.

        Used to store variables in the store.

        Returns:
            Name of the module.
        """
        return ""

    def create_logger(self) -> logging.Logger:
        """Create a logger for the model.

        Returns:
            Logger instance for the model.
        """
        logger: logging.Logger = logging.getLogger("GEB")

        if (
            self.config
            and "logging" in self.config
            and "loglevel" in self.config["logging"]
        ):
            loglevel = self.config["logging"]["loglevel"]
        else:
            loglevel = "INFO"
        logger.setLevel(logging.getLevelName(loglevel))

        if (
            self.config
            and "logging" in self.config
            and "logfile" in self.config["logging"]
        ):
            logfile = self.config["logging"]["logfile"]
        else:
            logfile = "GEB.log"

        formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")

        file_handler = logging.FileHandler(logfile, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger
