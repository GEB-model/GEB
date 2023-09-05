# -*- coding: utf-8 -*-
import datetime
from pathlib import Path
from operator import attrgetter
import argparse
from typing import Union
from time import time

import pandas as pd
import numpy as np
try:
    import cupy as cp
except ImportError:
    pass

from honeybees.library.helpers import timeprint
from honeybees.area import Area
from honeybees.model import Model as ABM_Model

from .reporter import Reporter
from .agents import Agents
from .artists import Artists
from .HRUs import Data
from .cwatm_model import CWatM_Model

from .sfincs import SFINCS

class GEBModel(ABM_Model, CWatM_Model):
    """GEB parent class.
    
    Args:
        GEB_config_path: Filepath of the YAML-configuration file.
        CwatM_settings: Path of CWatM settings file.
        name: Name of model.
        xmin: Minimum x coordinate.
        xmax: Maximum x coordinate.
        ymin: Minimum y coordinate.
        ymax: Maximum y coordinate.
        args: Run arguments.
        coordinate_system: Coordinate system that should be used. Currently only accepts WGS84.
    """

    description = """GEB stands for Geographic Environmental and Behavioural model and is named after Geb, the personification of Earth in Egyptian mythology.\nGEB aims to simulate both environment, for now the hydrological system, the behaviour of people and their interactions at large scale without sacrificing too much detail. The model does so by coupling an agent-based model which simulates millions individual people or households and a hydrological model. While the model can be expanded to other agents and environmental interactions, we focus on farmers, high-level agents, irrigation behaviour and land management for now."""
    def __init__(self, GEB_config_path: str, study_area: dict, scenario: str, use_gpu: bool=False, coordinate_system: str='WGS84'):
        self.use_gpu = use_gpu
        if self.use_gpu:
            cp.cuda.Device(self.args.gpu_device).use()
        
        self.config = self.setup_config(GEB_config_path)

        self.initial_conditions_folder = Path(self.config['general']['initial_conditions_folder'])
        if scenario == 'spinup':
            end_time = datetime.datetime.combine(self.config['general']['start_time'], datetime.time(0))
            current_time = datetime.datetime.combine(self.config['general']['spinup_time'], datetime.time(0))
            self.load_initial_data = False
            self.save_initial_data = self.config['general']['export_inital_on_spinup']
            self.initial_conditions = []
        else:
            current_time = datetime.datetime.combine(self.config['general']['start_time'], datetime.time(0))
            end_time = datetime.datetime.combine(self.config['general']['end_time'], datetime.time(0))
            self.load_initial_data = True
            self.save_initial_data = False

        assert isinstance(end_time, datetime.datetime)
        assert isinstance(current_time, datetime.datetime)
        
        timestep_length = datetime.timedelta(days=1)
        n_timesteps = (end_time - current_time) / timestep_length
        assert n_timesteps.is_integer()
        n_timesteps = int(n_timesteps)
        assert n_timesteps > 0
        
        self.data = Data(self)

        self.__init_ABM__(GEB_config_path, study_area, current_time, timestep_length, n_timesteps, coordinate_system)
        self.__init_hydromodel__(self.config['general']['CWatM_settings'])
        if self.config['general']['simulate_floods']:
            bbox = [73.86941, 18.99371, 73.94790, 19.05860]
            self.sfincs = SFINCS(self, self.config, bbox=bbox)
        self.reporter = Reporter(self)

        np.savez_compressed(Path(self.reporter.abm_reporter.export_folder, 'land_owners.npz'), data=self.data.HRU.land_owners)
        np.savez_compressed(Path(self.reporter.abm_reporter.export_folder, 'unmerged_HRU_indices.npz'), data=self.data.HRU.unmerged_HRU_indices)
        np.savez_compressed(Path(self.reporter.abm_reporter.export_folder, 'scaling.npz'), data=self.data.HRU.scaling)
        np.savez_compressed(Path(self.reporter.abm_reporter.export_folder, 'activation_order.npz'), data=self.agents.farmers.activation_order_by_elevation)

        self.running = True

    def __init_ABM__(self, config_path: str, study_area: dict, current_time, timestep_length, n_timesteps, coordinate_system: str) -> None:
        """Initializes the agent-based model.
        
        Args:
            config_path: Filepath of the YAML-configuration file.
            study_area: Dictionary with study area name, xmin, xmax, ymin and ymax.
            args: Run arguments.
            coordinate_system: Coordinate system that should be used. Currently only accepts WGS84.
        """

        ABM_Model.__init__(self, current_time, timestep_length, config_path, args=None, n_timesteps=n_timesteps)
        
        study_area.update({
            'xmin': self.data.grid.bounds.left,
            'xmax': self.data.grid.bounds.right,
            'ymin': self.data.grid.bounds.bottom,
            'ymax': self.data.grid.bounds.top,
        })

        self.area = Area(self, study_area)
        self.agents = Agents(self)
        self.artists = Artists(self)

        assert coordinate_system == 'WGS84'  # coordinate system must be WGS84. If not, all code needs to be reviewed

        # This variable is required for the batch runner. To stop the model
        # if some condition is met set running to False.
        timeprint("Finished setup")

    def __init_hydromodel__(self, settings: str) -> None:
        """Function to initialize CWatM.
        
        Args:
            settings: Filepath of CWatM settingsfile
        """
        CWatM_Model.__init__(self, self.current_time + self.timestep_length, self.n_timesteps, settings)

    def step(self, step_size: Union[int, str]=1) -> None:
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
            # print model information 
            print(self.current_time)
            #print('random_test_message')
            t0 = time()
            self.data.step()
            ABM_Model.step(self, 1, report=False)
            CWatM_Model.step(self, 1)

            if self.config['general']['simulate_floods']:
                n_routing_steps = self.data.grid.noRoutingSteps
                n_days = 2
                previous_discharges = pd.DataFrame(self.data.grid.previous_discharges).set_index('time').tail(n_days * n_routing_steps)
                print('multiplying discharge by 100 to create a flood')
                previous_discharges *= 100
                flood, crs, gt = self.sfincs.run(previous_discharges, lons=[73.87007], lats=[19.05390], plot=())  # plot can be basemap, forcing, max_flood_depth
                self.agents.farmers.flood(flood, crs, gt)
         
            self.reporter.step()
            t1 = time()
            # print(t1-t0, 'seconds')

    def run(self) -> None:
        """Run the model for the entire period, and export water table in case of spinup scenario."""
        for _ in range(self.n_timesteps):
            self.step()

        CWatM_Model.finalize(self)

        if self.config['general']['couple_plantFATE']:
            self.data.HRU.plant_fate_df.to_csv('plantFATE.csv')

        if self.save_initial_data:
            self.initial_conditions_folder.mkdir(parents=True, exist_ok=True)
            with open(Path(self.initial_conditions_folder, 'initial_conditions.txt'), 'w') as f:
                for var in self.initial_conditions:
                    f.write(f"{var}\n")
            
                    fp = self.initial_conditions_folder / f"{var}.npz"
                    values = attrgetter(var)(self.data)
                    np.savez_compressed(fp, data=values)

            for attribute in self.agents.farmers.agent_attributes:
                fp = Path(self.initial_conditions_folder, f"farmers.{attribute}.npz")
                values = attrgetter(attribute)(self.agents.farmers)
                np.savez_compressed(fp, data=values)

    @property
    def current_day_of_year(self) -> int:
        """Gets the current day of the year.
        
        Returns:
            day: current day of the year.
        """
        return self.current_time.timetuple().tm_yday