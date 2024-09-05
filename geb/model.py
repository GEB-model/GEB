import datetime
from pathlib import Path
import geopandas as gpd
from typing import Union
from time import time
import copy
import numpy as np
import warnings

try:
    import cupy as cp
except ImportError:
    pass

from honeybees.library.helpers import timeprint
from honeybees.area import Area
from honeybees.model import Model as ABM_Model

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
        self.artists = Artists(self)

        # This variable is required for the batch runner. To stop the model
        # if some condition is met set running to False.
        timeprint("Finished setup")


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
        spinup: bool = False,
        use_gpu: bool = False,
        gpu_device=0,
        timing=False,
        crs=4326,
        mode="w",
    ):
        self.timing = timing
        assert mode in ("w", "r")
        self.mode = mode

        self.spinup = spinup
        self.use_gpu = use_gpu
        if self.use_gpu:
            cp.cuda.Device(gpu_device).use()

        self.config = self.setup_config(config)

        if "simulate_hydrology" not in self.config["general"]:
            self.config["general"]["simulate_hydrology"] = True
            warnings.warn(
                "Please add 'simulate_hydrology' to the general section of the config file. For most cases this should be set to 'true'.",
                DeprecationWarning,
            )

        if self.spinup:
            self.config["report"] = {}

        # make a deep copy to avoid issues when the model is initialized multiple times
        self.files = copy.deepcopy(files)
        for data in self.files.values():
            for key, value in data.items():
                data[key] = Path(config["general"]["input_folder"]) / value

        if spinup is True:
            self.run_name = "spinup"
        elif "name" in self.config["general"]:
            self.run_name = self.config["general"]["name"]
        else:
            print('No "name" specified in config file under general. Using "default".')
            self.run_name = "default"

        self.report_folder = (
            Path(self.config["general"]["report_folder"]) / self.run_name
        )
        self.report_folder.mkdir(parents=True, exist_ok=True)

        self.initial_conditions_folder = Path(
            self.config["general"]["initial_conditions_folder"]
        )

        self.spinup_start = datetime.datetime.combine(
            self.config["general"]["spinup_time"], datetime.time(0)
        )

        if self.spinup is True:
            end_time = datetime.datetime.combine(
                self.config["general"]["start_time"], datetime.time(0)
            )
            current_time = datetime.datetime.combine(
                self.config["general"]["spinup_time"], datetime.time(0)
            )
            if end_time.year - current_time.year < 10:
                print(
                    "Spinup time is less than 10 years. This is not recommended and may lead to issues later."
                )

            self.load_initial_data = False
            self.save_initial_data = self.config["general"]["export_inital_on_spinup"]
        else:
            # check if spinup has been executed before
            if not self.initial_conditions_folder.exists():
                raise FileNotFoundError(
                    f"The initial conditions folder ({self.initial_conditions_folder.resolve()}) does not exist. Spinup is required before running the model, and will make the 'initial' folder. Please run the spinup first."
                )

            current_time = datetime.datetime.combine(
                self.config["general"]["start_time"], datetime.time(0)
            )
            end_time = datetime.datetime.combine(
                self.config["general"]["end_time"], datetime.time(0)
            )
            self.load_initial_data = True
            self.save_initial_data = False

        assert isinstance(end_time, datetime.datetime)
        assert isinstance(current_time, datetime.datetime)

        timestep_length = datetime.timedelta(days=1)
        self.seconds_per_timestep = timestep_length.total_seconds()
        n_timesteps = (end_time - current_time) / timestep_length
        assert n_timesteps.is_integer()
        n_timesteps = int(n_timesteps)
        assert n_timesteps > 0, "End time is before or identical to start time"

        self.regions = gpd.read_file(self.files["geoms"]["areamaps/regions"])
        self.data = Data(self)

        if self.mode == "w":
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

            self.reporter = Reporter(self)

            np.savez_compressed(
                Path(self.reporter.abm_reporter.export_folder, "land_owners.npz"),
                data=self.data.HRU.land_owners,
            )
            np.savez_compressed(
                Path(
                    self.reporter.abm_reporter.export_folder, "unmerged_HRU_indices.npz"
                ),
                data=self.data.HRU.unmerged_HRU_indices,
            )
            np.savez_compressed(
                Path(self.reporter.abm_reporter.export_folder, "scaling.npz"),
                data=self.data.HRU.scaling,
            )
            np.savez_compressed(
                Path(self.reporter.abm_reporter.export_folder, "activation_order.npz"),
                data=self.agents.crop_farmers.activation_order_by_elevation,
            )

    def step(self, step_size: Union[int, str] = 1) -> None:
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
            self.data.step()
            HazardDriver.step(self, 1)
            ABM_Model.step(self, 1, report=False)
            if self.config["general"]["simulate_hydrology"]:
                Hydrology.step(self)

            self.reporter.step()
            t1 = time()
            print(
                f"{self.current_time} ({round(t1 - t0, 4)}s)",
                flush=True,
            )

    def run(self) -> None:
        """Run the model for the entire period, and export water table in case of spinup scenario."""
        for _ in range(self.n_timesteps):
            self.step()

        if self.save_initial_data:
            self.data.save_state()
            self.agents.save_state()

        print("Model run finished")

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
