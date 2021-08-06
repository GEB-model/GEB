import numpy as np  
import os
from cwatm.cwatm_model import CWATModel
from cwatm.management_modules.dynamicModel import ModelFrame
from cwatm.management_modules.configuration import parse_configuration, read_metanetcdf
from cwatm.management_modules.globals import dateVar, settingsfile, binding, option, outDir
from cwatm.management_modules.data_handling import Flags, cbinding
from cwatm.management_modules.timestep import checkifDate
from cwatm.run_cwatm import headerinfo

class CWatM_Model(CWATModel):
    def __init__(self, start_date, n_steps, settings, use_gpu):
        self.init_water_table_file = os.path.join('report', 'init', 'water_table.npy')
        self.set_paramaters(start_date, n_steps, settings, use_gpu)
        CWATModel.__init__(self)
        self.initialize_model_frame()

    def set_paramaters(self, start_date, n_steps, settings, use_gpu):
        settingsfile.append(settings)
        parse_configuration(settings)

        if self.args and hasattr(self.args, "export_folder") and self.args.export_folder is not None:
            outDir['OUTPUT'] = os.path.join(outDir['OUTPUT'], self.args.export_folder)
            if not os.path.exists(outDir['OUTPUT']):
                os.makedirs(outDir['OUTPUT'])

        option['useGPU'] = use_gpu
        binding['max_groundwater_abstraction_depth'] = 50
        if self.args.scenario == 'initial':
            binding['load_init_water_table'] = 'false'
            binding['initial_water_table_depth'] = 2
        else:
            binding['load_init_water_table'] = 'true'
            binding['init_water_table'] = self.init_water_table_file
        # read_metanetcdf(cbinding('metaNetcdfFile'), 'metaNetcdfFile')
        assert start_date.hour == 0 and start_date.minute == 0 and start_date.second == 0 and start_date.microsecond == 0
        binding['StepStart'] = start_date.strftime('%d/%m/%Y')
        binding['SpinUp'] = '0'
        binding['StepEnd'] = str(n_steps)
        checkifDate('StepStart', 'StepEnd', 'SpinUp', cbinding('PrecipitationMaps'))
        headerinfo()

    def initialize_model_frame(self):
        self.stCWATM = ModelFrame(self, firstTimestep=dateVar["intStart"], lastTimeStep=dateVar["intEnd"])
        self.stCWATM.initialize_run()

    def step(self, n):
        for _ in range(n):
            self.stCWATM.step()

    def report(self):
        dirname = os.path.dirname(self.init_water_table_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if self.args.scenario == 'initial':
            np.save(self.init_water_table_file, self.groundwater_modflow_module.modflow.decompress(self.groundwater_modflow_module.modflow.head))