import os
import pandas as pd
from collections.abc import Iterable
import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
from operator import attrgetter


from hyve.reporter import Reporter as ABMReporter

class CWatMReporter(ABMReporter):
    def __init__(self, model):
        self.model = model
        self.set_variables()

        self.export_folder = (
            os.path.join('report', self.model.args.export_folder)
            if self.model.args and hasattr(self.model.args, "export_folder") and self.model.args.export_folder is not None
            else "report"
        )
        self.maybe_create_export_folder()

        self.variables = {}
        self.timesteps = []
        self._initialize()  # initalize

    def _initialize(self):
        for name in self.model.config['report_cwatm']:
            self.variables[name] = []
        self.step()  # report on inital state

    def set_variables(self):
        self.variables_dict = {}

        def add_var(attr):
            attributes = attrgetter(attr)(self.model)
            compressed_size = attributes.compressed_size
            for varname, variable in vars(attributes).items():
                if isinstance(variable, (np.ndarray, cp.ndarray)):
                    if variable.ndim == 1 and variable.size == compressed_size:
                        name = f'{attr}.{varname}'
                        self.variables_dict[name] = variable
                    if variable.ndim == 2 and variable.shape[1] == compressed_size:
                        for i in range(variable.shape[0]):
                            name = f'{attr}.{varname}[{i}]'
                            self.variables_dict[name] = variable[i]
                    else:
                        continue
        
        add_var('data.var')
        add_var('data.landunit')

    def get_array(self, attr, decompress=False):
        array = attrgetter(attr)(self.model)
        if decompress:
            array = attrgetter('.'.join(attr.split('.')[:-1]))(self.model).decompress(array)

        assert isinstance(array, (np.ndarray, cp.ndarray))

        return array

    def step(self):
        self.timesteps.append(self.model.current_time)
        self.set_variables()
        for name, conf in self.model.config['report_cwatm'].items():
            array = self.get_array(conf['varname'])
            if array is None:
                print(f"variable {name} not found at timestep {self.model.current_time}")
                self.report_value(name, None, conf)
            else:
                if conf['varname'].endswith("crop"):
                    crop_map = self.get_array('landunit.crop_map')
                    array = array[crop_map == conf['crop']]
                if array.size == 0:
                    value = None
                else:
                    if conf['function'] == 'mean':
                        value = np.mean(array)
                        if np.isnan(value):
                            value = None
                    elif conf['function'] == 'nanmean':
                        value = np.nanmean(array)
                        if np.isnan(value):
                            value = None
                    elif conf['function'] == 'sum':
                        value = np.sum(array)
                        if np.isnan(value):
                            value = None
                    elif conf['function'] == 'nansum':
                        value = np.nansum(array)
                        if np.isnan(value):
                            value = None
                    else:
                        raise ValueError()
                self.report_value(name, value, conf)

    def report(self):
        report_dict = {}
        for name, values in self.variables.items():
            if isinstance(values[0], Iterable):
                df = pd.DataFrame.from_dict(
                    {
                        k: v
                        for k, v in zip(self.timesteps, values)
                    }
                )
            else:
                df = pd.DataFrame(values, index=self.timesteps, columns=[name])
            export_format = self.model.config['report_cwatm'][name]['format']
            if export_format == 'csv':
                df.to_csv(os.path.join(self.export_folder, name + '.' + export_format))
            elif export_format == 'xlsx':
                df.to_excel(os.path.join(self.export_folder, name + '.' + export_format))
            else:
                raise ValueError(f'save_to format {export_format} unknown')
        return report_dict


class Reporter:
    def __init__(self, model):
        self.model = model
        self.abm_reporter = ABMReporter(model)
        self.cwatmreporter = CWatMReporter(model)

    @property
    def variables(self):
        return {**self.abm_reporter.variables, **self.cwatmreporter.variables}

    @property
    def timesteps(self):
        return self.abm_reporter.timesteps

    def step(self):
        self.abm_reporter.step()
        self.cwatmreporter.step()

    def report(self):
        np.save('report/fields.npy', self.model.agents.farmers.fields)
        np.save('report/mask.npy', self.model.data.var.mask)
        np.save('report/subcell_locations.npy', self.model.data.landunit.subcell_locations)
        np.save('report/scaling.npy', self.model.data.landunit.scaling)

        self.abm_reporter.report()
        self.cwatmreporter.report()