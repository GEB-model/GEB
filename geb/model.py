import datetime
import shutil
from pathlib import Path
import geopandas as gpd
from typing import Union
from time import time
import copy
import numpy as np

from honeybees.library.helpers import timeprint
from honeybees.area import Area
from honeybees.model import Model as ABM_Model

from geb.store import Store
from geb.reporter import Reporter
from geb.agents import Agents
from geb.artists import Artists
from geb.HRUs import Data
from .hydrology import Hydrology
from geb.hazards.driver import HazardDriver


class ABM(ABM_Model):
    def __init__(
        self,
        current_time,
        timestep_length,
        n_timesteps,
    ) -> None:
        """Initializes the agent-based model.

        Args:
            config_path: Filepath of the YAML-configuration file.
            args: Run arguments.
            coordinate_system: Coordinate system that should be used. Currently only accepts WGS84.
        """

        ABM_Model.__init__(
            self,
            current_time,
            timestep_length,
            args=None,
            n_timesteps=n_timesteps,
        )

        study_area = {
            "xmin": self.data.grid.bounds[0],
            "xmax": self.data.grid.bounds[2],
            "ymin": self.data.grid.bounds[1],
            "ymax": self.data.grid.bounds[3],
        }

        self.area = Area(self, study_area)
        self.agents = Agents(self)

        # This variable is required for the batch runner. To stop the model
        # if some condition is met set running to False.
        timeprint("Finished setup")

    def step(self):
        self.agents.step()


class GEBModel(HazardDriver, ABM, Hydrology):
    """GEB parent class.

    Args:
        config: Filepath of the YAML-configuration file (e.g. model.yml).
        name: Name of model.
        xmin: Minimum x coordinate.
        xmax: Maximum x coordinate.
        ymin: Minimum y coordinate.
        ymax: Maximum y coordinate.
        args: Run arguments.
        coordinate_system: Coordinate system that should be used. Currently only accepts WGS84.
    """

    description = """GEB stands for Geographic Environmental and Behavioural model and is named after Geb, the personification of Earth in Egyptian mythology.\nGEB aims to simulate both environment, for now the hydrological system, the behaviour of people and their interactions at large scale without sacrificing too much detail. The model does so by coupling an agent-based model which simulates millions individual people or households and a hydrological model. While the model can be expanded to other agents and environmental interactions, we focus on farmers, high-level agents, irrigation behaviour and land management for now."""

    def __init__(
        self,
        config: dict,
        files: dict,
        crs=4326,
        mode="w",
        timing=False,
    ):
        self.crs = crs
        self.timing = timing
        self.mode = mode

        self.config = self.setup_config(config)

        # make a deep copy to avoid issues when the model is initialized multiple times
        self.files = copy.deepcopy(files)
        for data in self.files.values():
            for key, value in data.items():
                data[key] = Path(config["general"]["input_folder"]) / value

        self.regions = gpd.read_file(self.files["geoms"]["areamaps/regions"])
        self.store = Store(self)

    def restore(self, store_location, timestep):
        self.store.load(store_location)
        self.groundwater.modflow.restore(self.data.grid.var.heads)
        self.current_timestep = timestep

    def multiverse(self):
        # copy current state of timestep and time
        store_timestep = copy.copy(self.current_timestep)

        # set a folder to store the initial state of the multiverse
        store_location = self.simulation_root / "multiverse" / "forecast"
        self.store.save(store_location)

        # perform one run of the multiverse
        discharges_before_restore = []
        for _ in range(10):
            discharges_before_restore.append(self.data.grid.var.discharge.copy())
            self.step()

        # restore the initial state of the multiverse
        self.restore(store_location=store_location, timestep=store_timestep)

        # again perform one run of the multiverse
        discharges_after_restore = []
        for _ in range(10):
            discharges_after_restore.append(self.data.grid.var.discharge.copy())
            self.step()

        # restore the initial state of the multiverse
        self.restore(store_location=store_location, timestep=store_timestep)

        # check if the discharges are the same in both multiverses
        assert np.array_equal(discharges_before_restore, discharges_after_restore)

    def step(self, step_size: Union[int, str] = 1, report=False) -> None:
        """
        Forward the model by the given the number of steps.

        Args:
            step_size: Number of steps the model should take. Can be integer or string `day`, `week`, `month`, `year`, `decade` or `century`.
        """
        if isinstance(step_size, str):
            n = self.parse_step_str(step_size)
        else:
            n = step_size
        for _ in range(n):
            t0 = time()
            HazardDriver.step(self, 1)
            ABM.step(self)
            if self.config["general"]["simulate_hydrology"]:
                Hydrology.step(self)

            t1 = time()
            print(
                f"{self.current_time} ({round(t1 - t0, 4)}s)",
                flush=True,
            )

            if report:
                self.reporter.step()

            # if self.current_timestep == 5:
            #     self.multiverse()
            self.current_timestep += 1

    def create_datetime(self, date):
        return datetime.datetime.combine(date, datetime.time(0))

    def _initialize(
        self,
        run_name,
        report,
        current_time,
        end_time,
        in_spinup=False,
        clean_report_folder=False,
        load_data_from_store=False,
    ) -> None:
        """Initializes the model."""
        self.run_name = run_name
        self.in_spinup = in_spinup

        # optionally clean report model at start of run
        if clean_report_folder:
            shutil.rmtree(self.report_folder, ignore_errors=True)

        self.data = Data(self)
        self.report_folder.mkdir(parents=True, exist_ok=True)

        self.spinup_start = datetime.datetime.combine(
            self.config["general"]["spinup_time"], datetime.time(0)
        )

        timestep_length = datetime.timedelta(days=1)
        self.seconds_per_timestep = timestep_length.total_seconds()
        n_timesteps = (end_time - current_time) / timestep_length
        assert n_timesteps.is_integer()
        n_timesteps = int(n_timesteps)
        assert n_timesteps > 0, "End time is before or identical to start time"

        HazardDriver.__init__(self)

        ABM.__init__(
            self,
            current_time,
            timestep_length,
            n_timesteps,
        )

        if self.config["general"]["simulate_hydrology"]:
            Hydrology.__init__(
                self,
            )

        if load_data_from_store:
            self.store.load()

        self.groundwater.initalize_modflow_model()
        self.soil.set_global_variables()

        if report:
            self.reporter = Reporter(self)
        self.artists = Artists(self)

    def run(self) -> None:
        """Run the model for the entire period, and export water table in case of spinup scenario."""
        if not self.store.path.exists():
            raise FileNotFoundError(
                f"The initial conditions folder ({self.store.path.resolve()}) does not exist. Spinup is required before running the model. Please run the spinup first."
            )

        if "name" in self.config["general"]:
            run_name = self.config["general"]["name"]
        else:
            print('No "name" specified in config file under general. Using "default".')
            run_name = "default"

        current_time = self.create_datetime(self.config["general"]["start_time"])
        end_time = self.create_datetime(self.config["general"]["end_time"])

        self._initialize(
            run_name=run_name,
            report=True,
            current_time=current_time,
            end_time=end_time,
            clean_report_folder=True,
            load_data_from_store=True,
        )

        for _ in range(self.n_timesteps):
            self.step(report=True)

        print("Model run finished, finalizing report...")
        self.reporter.finalize()

    def spinup(self) -> None:
        """Run the model for the spinup period."""
        run_name = "spinup"

        # set the start and end time for the spinup. The end of the spinup is the start of the actual model run
        current_time = self.create_datetime(self.config["general"]["spinup_time"])
        end_time = self.create_datetime(self.config["general"]["start_time"])

        if end_time.year - current_time.year < 10:
            print(
                "Spinup time is less than 10 years. This is not recommended and may lead to issues later."
            )

        self._initialize(
            run_name=run_name,
            report=False,
            current_time=current_time,
            end_time=end_time,
            clean_report_folder=True,
            in_spinup=True,
        )

        for _ in range(self.n_timesteps):
            self.step(report=False)

        print("Spinup finished, saving conditions at end of spinup...")

        self.store.save()

    def estimate_risk(self) -> None:
        """Estimate the risk of the model."""
        current_time = self.create_datetime(self.config["general"]["start_time"])
        end_time = self.create_datetime(self.config["general"]["end_time"])

        self._initialize(
            run_name="estimate_risk",
            report=False,
            current_time=current_time,
            end_time=end_time,
            load_data_from_store=True,
        )

        ...

    @property
    def current_day_of_year(self) -> int:
        """Gets the current day of the year.

        Returns:
            day: current day of the year.
        """
        return self.current_time.timetuple().tm_yday

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
    def report_folder(self):
        return Path(self.config["general"]["report_folder"]) / self.run_name

    def close(self) -> None:
        """Finalizes the model."""
        if self.mode == "w" and self.config["general"]["simulate_hydrology"]:
            Hydrology.finalize(self)

            from geb.workflows import all_async_readers

            for reader in all_async_readers:
                reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
