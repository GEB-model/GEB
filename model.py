# -*- coding: utf-8 -*-
from datetime import timedelta, date
from hyve.library.helpers import timeprint
from hyve.area import Area
from reporter import Reporter
from hyve.model import Model as ABM_Model
from agents import Agents
from artists import Artists
from landunits import Data
import argparse
from cwatm_model import CWatM_Model
from typing import Union
import yaml
import os
from operator import attrgetter
import numpy as np
from time import time

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
    def __init__(self, GEB_config_path: str, CwatM_settings: str, name: str, xmin: float, xmax: float, ymin: float, ymax: float, args: argparse.Namespace, coordinate_system: str='WGS84'):
        study_area = {
            "name": name,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,            
        }
        self.args = args


        self.data = Data(self)
        self.config = self.setup_config(GEB_config_path)
        self.initial_conditions_folder = os.path.join(self.config['general']['initial_conditions_folder'])
        self.__init_ABM__(GEB_config_path, study_area, args, coordinate_system)
        self.__init_hydromodel__(CwatM_settings)

        self.reporter = Reporter(self)

    def __init_ABM__(self, config_path: str, study_area: dict[str, float], args: argparse.Namespace, coordinate_system: str) -> None:
        """Initializes the agent-based model.
        
        Args:
            config_path: Filepath of the YAML-configuration file.
            study_area: Dictionary with study area name, xmin, xmax, ymin and ymax.
            args: Run arguments.
            coordinate_system: Coordinate system that should be used. Currently only accepts WGS84.
        """

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        timestep_length = timedelta(days=1)
        if args.scenario == 'spinup':
            end_time = config['general']['start_time']
            current_time = config['general']['spinup_start']
        else:
            current_time = config['general']['start_time']
            end_time = config['general']['end_time']

        assert isinstance(end_time, date)
        assert isinstance(current_time, date)
        
        n_timesteps = (end_time - current_time) / timestep_length
        assert n_timesteps.is_integer()
        n_timesteps = int(n_timesteps)

        ABM_Model.__init__(self, current_time, timestep_length, config_path, args=args, n_timesteps=n_timesteps)
        
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
        CWatM_Model.__init__(self, self.current_time, self.n_timesteps, settings)

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
            print(self.current_time)
            t0 = time()
            ABM_Model.step(self, 1, report=False)
            CWatM_Model.step(self, 1)
            self.reporter.step()
            t1 = time()
            print(t1-t0, 'seconds')

    def run(self) -> None:
        """Run the model for the entire period, and export water table in case of spinup scenario."""
        for _ in range(self.n_timesteps):
            self.step()

        if self.save_initial:
            
            initCondVar = ['landunit.w1', 'landunit.w2', 'landunit.w3', 'landunit.topwater', 'landunit.interceptStor', 'landunit.SnowCoverS', 'landunit.FrostIndex', 'grid.channelStorageM3', 'grid.discharge', 'grid.lakeInflow', 'grid.lakeStorage', 'grid.reservoirStorage', 'grid.lakeVolume', 'grid.outLake', 'grid.lakeOutflow', 'modflow.head']
            # self.initCondVar.extend(['grid.smalllakeInflow', 'grid.smalllakeStorage', 'grid.smalllakeOutflow', 'grid.smalllakeInflowOld', 'grid.smalllakeVolumeM3'])

            if not os.path.exists(self.initial_conditions_folder):
                os.makedirs(self.initial_conditions_folder)
            for initvar in initCondVar:
                fp = os.path.join(self.initial_conditions_folder, f"{initvar}.npy")
                values = attrgetter(initvar)(self.data)
                np.save(fp, values)

