from datetime import timedelta
from hyve.library.helpers import timeprint
from hyve.area import Area
from reporter import Reporter
from hyve.model import Model as ABM_Model
from agents import Agents
from artists import Artists
from hydrounits import Data
import argparse
from cwatm_model import CWatM_Model


class GEBModel(ABM_Model, CWatM_Model):
    def __init__(self, ABM_config_path, CwatM_settings, name, xmin, xmax, ymin, ymax, args, coordinate_system='WGS84'):
        study_area = {
            "name": name,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,            
        }
        
        self.__init_ABM__(ABM_config_path, study_area, args, coordinate_system)
        self.data = Data(self)
        self.__init_hydromodel__(CwatM_settings)

        self.reporter = Reporter(self)

    def __init_ABM__(self, config_path: str, study_area: dict[str, float], args: argparse.Namespace, coordinate_system: str):
        """Initializes the agent-based model.
        
        Args:
            config_path: Filepath of the YAML-configuration file.
            study_area: Dictionary with study area name, xmin, xmax, ymin and ymax.
        """
        ABM_Model.__init__(self, config_path, args)

        self.timestep_length = timedelta(days=1)
        if args.scenario == 'spinup':
            self.end_time = self.config['general']['start_time']
            self.current_time = self.config['general']['spinup_start']
        else:
            self.current_time = self.config['general']['start_time']
            self.end_time = self.config['general']['end_time']
        
        self.n_timesteps = (self.end_time - self.current_time) / self.timestep_length
        assert self.n_timesteps.is_integer()
        self.n_timesteps = int(self.n_timesteps)
        
        self.artists = Artists(self)
        self.area = Area(self, study_area)
        self.agents = Agents(self)

        assert coordinate_system == 'WGS84'  # coordinate system must be WGS84. If not, all code needs to be reviewed

        # This variable is required for the batch runner. To stop the model
        # if some condition is met set running to False.
        timeprint("Finished setup")

    def __init_hydromodel__(self, settings):
        CWatM_Model.__init__(self, self.current_time, self.n_timesteps, settings, self.config['general']['use_gpu'])

    def step(self, step_size=1):
        if isinstance(step_size, str):
            n = self.parse_step_str(step_size)
        else:
            n = step_size
        for _ in range(n):
            ABM_Model.step(self, 1, report=False)
            CWatM_Model.step(self, 1)
            self.reporter.step()

    def run(self):
        self.step(self.n_timesteps)
        CWatM_Model.report(self)