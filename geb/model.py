"""The main GEB model class. This class is used to initialize and run the model."""

import copy
import datetime
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
from types import TracebackType
from typing import Any, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from packaging.version import Version

from geb import GEB_PACKAGE_DIR, __version__
from geb.agents import Agents
from geb.build.version_updates import get_and_maybe_do_version_updates
from geb.hazards.driver import HazardDriver
from geb.hazards.floods.workflows.construct_storm_surge_hydrographs import (
    generate_storm_surge_hydrographs,
)
from geb.module import Module
from geb.reporter import Reporter
from geb.store import Bucket, Store
from geb.agents.
from geb.workflows.io import read_geom, read_params, read_zarr

from .evaluate import Evaluate
from .forcing import Forcing
from .hydrology import Hydrology


class GEBModelVariables(Bucket):
    """Class to hold GEB model variables."""

    _spinup_start: datetime.datetime
    _run_start: datetime.datetime


class GEBModel(Module):
    """GEB parent class.

    Args:
        config: Filepath of the YAML-configuration file (e.g. model.yml).
        files: Dictionary with the paths of the input files.
        mode: Mode of the model. Either `w` (write) or `r` (read).
        timing: Boolean indicating if the model steps should be timed.
    """

    var: GEBModelVariables
    plantFATE: list[Any]

    def __init__(
        self,
        config: dict,
        files: dict,
        logger: logging.Logger | None = None,
        mode: str = "w",
        timing: bool = False,
    ) -> None:
        """Initialize the GEB model.

        Args:
            config: A dictionary containing the model configuration.
            files: A dictionary containing the paths to the input files.
            logger: A logging.Logger instance to use for logging. If None, a default logger will be created.
            mode: Run model writing mode "w", or reading mode "r". Defaults to "w" for writing.
                If "r", the model will not write any output files, but will read from existing files.
            timing: Whether to log timing of modules. Defaults to False.

        Raises:
            ValueError: If the mode is not 'r' or 'w'.
        """
        self.config: dict[str, Any] = copy.deepcopy(config)  # model configuration
        self.logger = logger or logging.getLogger(__name__)  # model logger
        self.timing = timing  # whether to log timing of modules
        self.mode = mode  # mode of the model, either 'r' (read) or 'w' (write)
        if self.mode not in ["r", "w"]:
            raise ValueError(
                "Mode must be either 'r' (read) or 'w' (write)"
            )  # validate mode

        self.verify_build_complete()
        self.check_data_version()

        Module.__init__(self, self, create_var=False)  # initialize the Module class

        self._multiverse_name = None  # name of the multiverse, if any

        self.files = copy.deepcopy(
            files
        )  # make a deep copy to avoid issues when the model is initialized multiple times
        for data in self.files.values():
            for key, value in data.items():
                data[key] = self.input_folder / value  # make paths absolute

        self.store = Store(self)

        self.evaluator = Evaluate(self)  # initialize the evaluator

        self.plantFATE = []  # Empty list to hold plantFATE models. If forests are not used, this will be empty

    def verify_build_complete(self) -> None:
        """Verify that the build completed.

        Raises:
            RuntimeError: If the file 'build_complete.txt' is not found in the input folder, indicating that the build is not complete.
        """
        build_complete_path = self.input_folder / "build_complete.txt"
        if not build_complete_path.exists():
            raise RuntimeError(
                (
                    f"Build not complete. The file 'build_complete.txt' was not found in the input folder "
                    f"({self.input_folder.resolve()}). If you created the model with an older version, make "
                    f"a new file named 'build_complete.txt' in {self.input_folder.resolve()} to indicate that "
                    "the build is complete, or run a new build with the current version of GEB."
                )
            )

    def check_data_version(self) -> None:
        """Check if the model version of the data matches the current model version.

        If the version file does not exist, it will ignore the check.
        If the version file exists, but there are no updates between the data version
        and the current model version, it will also ignore the check.

        Raises:
            RuntimeError: If the version file exists and there are updates between the data version and the current model version.
        """
        version_path = self.input_folder / "version.txt"
        if not version_path.exists():
            return

        version_info = version_path.read_text()
        if Version(version_info) == Version(__version__):
            return

        updates: list[str] = get_and_maybe_do_version_updates(
            version_info, logger=self.logger
        )
        if updates:
            error = f"Version mismatch and updating is required: input data version is {version_info}, but current model version is {__version__}. Please run 'geb update-version' to update the model to the current version."
            self.logger.error(error)
            raise RuntimeError(error)

        else:
            self.logger.info(
                "Version mismatch but no specific updates found for this version. Updated version file."
            )
            version_path.write_text(__version__)

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

        # load all zarr files for forecast data for all supported variables
        forecast_members: list[str] | None = None
        forecast_end_dt: datetime.datetime | None = None
        forecast_data: dict[str, xr.DataArray] = {}
        print(
            f"DEBUG: Starting to load forecast data for {len(self.forcing.loaders)} loaders",
            flush=True,
        )
        for loader_name, loader in self.forcing.loaders.items():
            if loader.supports_forecast:
                forecast_file_path = (
                    self.input_folder
                    / "other"
                    / "forecasts"
                    / self.config["general"]["forecasts"]["provider"]
                    / self.config["general"]["forecasts"]["processing"]
                    / forecast_issue_datetime.strftime("%Y%m%dT%H%M%S")
                    / f"{loader_name}_{forecast_issue_datetime.strftime('%Y%m%dT%H%M%S')}.zarr"
                )

                # Check if forecast file exists for this variable
                if forecast_file_path.exists():
                    print(
                        f"DEBUG: Forecast file exists for {loader_name}, loading...",
                        flush=True,
                    )
                    # open one forecast to see the number of members
                    forecast_data[loader_name] = read_zarr(forecast_file_path)
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
                else:
                    print(
                        f"DEBUG: Forecast file does NOT exist for {loader_name}",
                        flush=True,
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
                if loader.supports_forecast and loader_name in forecast_data:
                    loader.set_forecast(
                        forecast_issue_datetime=forecast_issue_datetime,
                        da=forecast_data[loader_name].sel(member=member),
                    )

            self.logger.info(f"Running forecast member {member}")
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

        self.logger.info("Forecast finished, restoring all conditions...")

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
        for loader_name, loader in self.forcing.loaders.items():
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

        Raises:
            ValueError: If forecast directories do not have the expected datetime format.
            RuntimeError: If forecast file for the current timestep is not found when forecasts are enabled in the config.
        """
        # only if forecasts is used, and if we are not already in multiverse (avoiding infinite recursion)
        # and if the current date is in the list of forecast days
        if (
            self.config["general"]["forecasts"]["use"]
            and self.multiverse_name
            is None  # only start multiverse if not already in one
            and self.current_time.date()
        ):
            # Discover all available forecast initialization directories
            forecast_base_path = (
                self.input_folder
                / "other"
                / "forecasts"
                / self.config["general"]["forecasts"]["provider"]
                / self.config["general"]["forecasts"]["processing"]
            )

            # extract unique forecast issue datetimes from files keys
            forecast_issue_dates: list[datetime.datetime] = sorted(
                {
                    datetime.datetime.strptime(Path(file_key).parts[3], "%Y%m%dT%H%M%S")
                    for file_key in other_files
                    if file_key.startswith(forecast_prefix)
                    and len(Path(file_key).parts) >= 5
                    and Path(file_key).parts[3].replace("T", "").isdigit()
                }
            )

            if forecast_base_path.exists():
                forecast_dirs: list[Path] = [
                    d
                    for d in forecast_base_path.iterdir()
                    if d.is_dir()
                    and d.name.startswith(
                        "2024"
                    )  # assuming forecasts start with year 2024
                ]  # get all forecast initialization directories
            else:
                forecast_dirs = []
            forecast_issue_dates: list[
                datetime.datetime
            ] = []  # list to store forecast issue dates

            for forecast_dir in forecast_dirs:
                if (
                    len(forecast_dir.name) == 15 and forecast_dir.name[8] == "T"
                ):  # YYYYMMDDTHHMMSS format
                    dt = datetime.datetime.strptime(forecast_dir.name, "%Y%m%dT%H%M%S")

                    # Check if this forecast directory contains precipitation data
                    forecast_files = list(forecast_dir.glob("**/*.zarr"))

                    if (
                        forecast_files
                    ):  # Only include forecasts that have precipitation data
                        forecast_issue_dates.append(dt)
                else:
                    raise RuntimeError(
                        f"Forecast file {f.name} does not have a valid datetime format. Expected format: 'YYYYMMDDTHHMMSS'."
                    )

            forecast_issue_dates = list(
                set(forecast_issue_dates)
            )  # only keep unique dates
            # Get warning system config settings
            warning_config = self.model.config["agent_settings"]["households"][
                "warning_system"
            ]

            prob_threshold = warning_config["probability_threshold"]
            area_threshold = warning_config["area_threshold"]
            building_threshold = warning_config["building_threshold"]
            warning_type = warning_config["strategies"]["warning_type"]
            communication_efficiency = warning_config["communication_efficiency"]
            evacuation_lead_time_threshold = warning_config[
                "evacuation_lead_time_threshold"
            ]
            weight_by_socioeconomic_factors = warning_config[
                "weight_by_socioeconomic_factors"
            ]
            # Determine response rate based on warning type
            if warning_type == "building_based":
                responsive_ratio = warning_config["response_rates"][
                    "building_based_warnings"
                ]

            elif warning_type == "area_based":
                responsive_ratio = warning_config["response_rates"][
                    "area_based_warnings"
                ]
            else:
                raise ValueError(
                    f"Unknown warning type: {warning_type} selected in config, choose 'building_based' or 'area_based'."
                )
            for dt in forecast_issue_dates:
                self.logger.debug(
                    "Checking forecast issue datetime: %s vs current model time: %s",
                    dt,
                    self.current_time,
                )

                if (
                    dt == self.current_time
                ):  # change to include hours (for when we move to hourly)
                    self.logger.debug(
                        "Forecast issue datetime matched current model time: %s",
                        dt.isoformat(),
                    )
                    forecast_datetime = datetime.datetime.combine(
                        dt, datetime.time(0)
                    )  # Convert date back to datetime for the multiverse method

                    self.multiverse(
                        forecast_issue_datetime=dt,
                        return_mean_discharge=True,
                    )  # run the multiverse for the current timestep

                    # after the multiverse has run all members for one day, if warning response is enabled, run the warning system
                    if self.config["agent_settings"]["households"]["warning_response"]:
                        self.logger.info(
                            f"Running flood early warning system for date time {self.current_time.isoformat()}..."
                        )
                        # Run warning strategies based on config settings
                        # Check whether water level warnings are enabled
                        if warning_config["strategies"]["water_level_warnings"]:
                            self.logger.info(
                                f"Running water level based warning strategy with {warning_type} warnings..."
                            )
                            self.agents.households.early_warning_module.water_level_warning_strategy(
                                date_time=self.current_time,
                                warning_type=warning_type,
                                prob_threshold=prob_threshold,
                                buildings_hit_threshold=building_threshold,
                                area_hit_threshold=area_threshold,
                                communication_efficiency=communication_efficiency,
                                evacuation_lead_time_threshold=evacuation_lead_time_threshold,
                                weight_by_socioeconomic_factors=weight_by_socioeconomic_factors,
                                exceedance=True,
                            )
                        if warning_config["strategies"][
                            "critical_infrastructure_warnings"
                        ]:
                            config_asset_type = warning_config["strategies"][
                                "critical_infrastructure_warnings"
                            ]["asset_type"]

                            self.agents.households.early_warning_module.critical_infrastructure_warning_strategy(
                                date_time=self.current_time,
                                config_asset_type=config_asset_type,
                                prob_threshold=prob_threshold,
                                exceedance=True,
                            )

                        # Run household decision-making to convert warnings into actions
                        self.agents.households.early_warning_module.household_decision_making(
                            date_time=self.current_time,
                            warning_type=warning_type,
                            responsive_ratio=responsive_ratio,
                        )

                        # Update household geodataframe with warning parameters
                        self.agents.households.early_warning_module.update_households_geodataframe_w_warning_variables(
                            date_time=self.current_time
                        )
                        print()

        t0 = time()
        self.agents.step()
        if self.simulate_hydrology:
            self.hydrology.step()

        self.hazard_driver.step()

        self.report(locals())

        t1 = time()
        self.logger.info(
            f"{self.multiverse_name + ' - ' if self.multiverse_name is not None else ''}step {self.current_time.date()} took {round(t1 - t0, 4)}s",
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

        self.hazard_driver = HazardDriver(self)

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

        self.report_folder = self.model.output_folder / "report"

        if create_reporter:
            self.reporter = Reporter(
                self, self.report_folder, clean=clean_report_folder
            )

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

        self.logger.info("Model run finished, finalizing report...")
        self.reporter.finalize()
        self.create_done_file()

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
        # only report household attributes (for now)
        self.config["report"] = {
            key: value
            for key, value in self.config["report"].items()
            if key.startswith("agents.households")
        }

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
            clean_report_folder=False,
            load_data_from_store=True,
        )

        self.step_to_end()

        self.logger.info("Model run finished, finalizing report...")
        self.reporter.finalize()
        self.create_done_file()

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
        self.var = cast(GEBModelVariables, self.store.create_bucket("var"))

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
        self.logger.info(f"Refreshing household attributes for {agent_type}...")
        path: Path = self.store.path
        name = getattr(self.agents, agent_type).name
        self.logger.debug(f"Saving {name}.var")
        bucket = self.store.buckets[f"{name}.var"]
        with ThreadPoolExecutor() as executor:
            futures = bucket.save(path / f"{name}.var", executor)
            for future in futures:
                future.result()

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
            warnings.warn(
                "Spinup time is less than 10 years. This is not recommended and may lead to issues later.",
                UserWarning,
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

        self.var: GEBModelVariables = cast(
            GEBModelVariables, self.store.create_bucket("var")
        )

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

        self.logger.info("Spinup finished, saving conditions at end of spinup...")
        self.store.save()

        self.reporter.finalize()
        self.create_done_file()

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

        # if self.var._run_start != self.run_start:
        #    raise ValueError(
        #        f"Run start time does not match the stored time range. Stored: {self.var._run_start}, Configured: {self.run_start}"
        #    )

    def estimate_return_periods(self, run_name: str = "spinup") -> None:
        """Estimate flood maps for different return periods."""
        current_time: datetime.datetime = self.run_start

        self._initialize(
            create_reporter=False,
            in_spinup=run_name == self.model.config["general"]["spinup_name"],
            current_time=current_time,
            n_timesteps=0,
            timestep_length=relativedelta(years=1),
            load_data_from_store=True,
            simulate_hydrology=True,
            clean_report_folder=False,
        )

        # ugly switch to determine whether model has coastal basins
        subbasins = read_geom(self.model.files["geom"]["routing/subbasins"])
        if subbasins["is_coastal"].any():
            generate_storm_surge_hydrographs(self)

        self.hazard_driver.floods.get_return_period_maps(run_name)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Call the evaluator to evaluate the model results.

        Returns:
            The result of the evaluation method.
        """
        self.logger.info("Evaluating model...")
        return self.evaluator.run(*args, **kwargs)

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
        return Path(self.config["general"]["output_folder"]) / self.model.run_name

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

    def close(self) -> None:
        """Finalizes the model."""
        if (
            self.mode == "w"
            and hasattr(self, "simulate_hydrology")
            and self.simulate_hydrology
        ):
            Hydrology.finalize(self.hydrology)

            # Close all forcing readers
            if hasattr(self, "forcing"):
                for forcing_loader in self.forcing.forcing_loaders.values():
                    if hasattr(forcing_loader, "reader") and hasattr(
                        forcing_loader.reader, "close"
                    ):
                        forcing_loader.reader.close()

    def __enter__(self) -> GEBModel:
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
        model_build_time_range: dict[str, str] = read_params(
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

        # if self.spinup_start.date() < model_build_start_date:
        #    raise ValueError(
        #        "Spinup start date cannot be before model build start date. Adjust the time range in your build configuration and rebuild the model or adjust the spinup time of the model."
        #    )

        # if self.run_end.date() > model_build_end_date:
        #    raise ValueError(
        #        "Run end date cannot be after model build end date. Adjust the time range in your build configuration and rebuild the model or adjust the simulation end time of the model."
        #    )

    def create_done_file(self) -> None:
        """Create a file to indicate that the model run or spinup is done."""
        (self.output_folder / "done.txt").touch()

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
        self._current_time = (
            self.simulation_start + self.current_timestep * self.timestep_length
        )

    @property
    def current_time(self) -> datetime.datetime:
        """Get the current model time.

        Returns:
            Current model time

        Raises:
            AttributeError: If `timestep_length` or `simulation_start` are not initialized.
        """
        try:
            return self._current_time
        except AttributeError as e:
            raise AttributeError(
                "Error computing current_time: " + str(e) + ". "
                "This may be due to 'simulation_start' or 'timestep_length' not being properly initialized."
            ) from e

    @property
    def name(self) -> str:
        """This is the name of this module, NOT the model or model run.

        Used to store variables in the store.

        Returns:
            Name of the module.
        """
        return ""
