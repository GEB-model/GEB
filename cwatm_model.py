# -*- coding: utf-8 -*-
import numpy as np  
import os
from cwatm.cwatm_model import CWATModel
from cwatm.management_modules.dynamicModel import ModelFrame
from cwatm.management_modules.configuration import parse_configuration, read_metanetcdf
from cwatm.management_modules.globals import dateVar, settingsfile, binding, option, outDir
from cwatm.management_modules.data_handling import cbinding
from cwatm.management_modules.timestep import checkifDate
from cwatm.run_cwatm import headerinfo
import datetime

class CWatM_Model(CWATModel):
    """
    This class is used to initalize the CWatM model from GEB. Several static configuration files are read first, then several dynamic parameters are set based on the configuration of GEB. Then, the model frame is created that can then later be used to iteratate.

    Args:
        start_date: Start date of the model.
        n_steps: Number of steps that the model will run for.
        settings: Filepath of the CWatM settingsfile. For full configuration options please refer to the `CWatM documentation <https://cwatm.iiasa.ac.at/>`.
        use_gpu: Whether the model can use a GPU.
    """
    def __init__(self, start_date: datetime.datetime, n_steps: int, settings: str) -> None:
        self.init_water_table_file = os.path.join(self.config['general']['init_water_table'])

        settingsfile.append(settings)
        parse_configuration(settings)

        outDir['OUTPUT'] = os.path.join(self.config['general']['report_folder'], self.args.scenario)

        # calibration
        for parameter, value in self.config['parameters'].items():
            binding[parameter] = value

        if self.args.scenario == 'spinup':
            self.load_initial = False
            self.save_initial = self.config['general']['export_inital_on_spinup'] 
        else:
            self.load_initial = True
            self.save_initial = False
        
        # read_metanetcdf(cbinding('metaNetcdfFile'), 'metaNetcdfFile')
        binding['StepStart'] = start_date.strftime('%d/%m/%Y')
        binding['SpinUp'] = '0'
        binding['StepEnd'] = str(n_steps)
        checkifDate('StepStart', 'StepEnd', 'SpinUp', cbinding('PrecipitationMaps'))
        headerinfo()

        CWATModel.__init__(self)
        self.stCWATM = ModelFrame(self, firstTimestep=dateVar["intStart"], lastTimeStep=dateVar["intEnd"])
        self.stCWATM.initialize_run()

    def step(self, n: int) -> None:
        """Performs n number of (daily) steps in CWatM.
        
        Args:
            n: Number of timesteps to perform.
        """
        for _ in range(n):
            self.stCWATM.step()

    def export_water_table(self) -> None:
        """Function to save required water table output to file."""
        dirname = os.path.dirname(self.init_water_table_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        np.save(self.init_water_table_file, self.groundwater_modflow_module.modflow.decompress(self.groundwater_modflow_module.modflow.head))