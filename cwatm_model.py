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
    def __init__(self, start_date: datetime.datetime, n_steps: int, settings: str, use_gpu: bool) -> None:
        self.init_water_table_file = os.path.join('report', 'init', 'water_table.npy')

        settingsfile.append(settings)
        parse_configuration(settings)

        if self.args and hasattr(self.args, "export_folder") and self.args.export_folder is not None:
            outDir['OUTPUT'] = os.path.join(outDir['OUTPUT'], self.args.export_folder)
            if not os.path.exists(outDir['OUTPUT']):
                os.makedirs(outDir['OUTPUT'])

        option['useGPU'] = use_gpu
        binding['max_groundwater_abstraction_depth'] = 50
        if self.args.scenario == 'spinup':
            binding['load_init_water_table'] = 'false'
            binding['initial_water_table_depth'] = 2
        else:
            binding['load_init_water_table'] = 'true'
            binding['init_water_table'] = self.init_water_table_file
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

    def report(self) -> None:
        """Function to save required CWatM output to file. Right now only the water table from the initial run is saved to a npy-file, which can then be used to initalize the water table in other scenarios.
        """
        dirname = os.path.dirname(self.init_water_table_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if self.args.scenario == 'spinup':
            np.save(self.init_water_table_file, self.groundwater_modflow_module.modflow.decompress(self.groundwater_modflow_module.modflow.head))