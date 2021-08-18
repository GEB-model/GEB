# -*- coding: utf-8 -*-
"""This module is used to report data to the disk. After initialization, the :meth:`reporter.Report.step` method is called every timestep, which in turn calls the equivalent methods in Hyve's reporter (to report data from the agents) and the CWatM reporter, to report data from CWatM. The variables to report can be configured in `GEB.yml` (see :doc:`configuration`). All data is saved in a subfolder (see `--export_folder` in :doc:`running`) of a folder called "report". 

"""

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
    """This class is used to report CWatM data to disk. On initialization the export folder is created if it does not yet exist. Then all variables to report are on read from the configuration folder, and the datastructures to save the data are created.
    
    Args:
        model: The GEB model.
    """
    def __init__(self, model) -> None:
        self.model = model

        self.export_folder = (
            os.path.join('report', self.model.args.export_folder)
            if self.model.args and hasattr(self.model.args, "export_folder") and self.model.args.export_folder is not None
            else "report"
        )
        self.maybe_create_export_folder()

        self.variables = {}
        self.timesteps = []

        for name in self.model.config['report_cwatm']:
            self.variables[name] = []
        self.step()  # report on inital state

    def get_array(self, attr: str, decompress: bool=False) -> np.ndarray:
        """This function retrieves a NumPy array from the model based the name of the variable. Optionally decompresses the array.

        Args:
            attr: Name of the variable to retrieve. Name can contain "." to specify variables are a "deeper" level.
            decompress: Boolean value whether to decompress the array. If True, the class to which the top variable name belongs to must have an equivalent function called `decompress`.

        Returns:
            array: The requested array.

        Example:
            Read discharge from `data.grid`. Because :code:`decompress=True`, `data.grid` must have a `decompress` method.
            ::
        
                >>> get_array(data.grid.discharge, decompress=True)
        """
        array = attrgetter(attr)(self.model)
        if decompress:
            array = attrgetter('.'.join(attr.split('.')[:-1]))(self.model).decompress(array)

        assert isinstance(array, (np.ndarray, cp.ndarray))

        return array

    def step(self) -> None:
        """This method is called after every timestep, to collect data for reporting from the model."""
        self.timesteps.append(self.model.current_time)
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

    def report(self) -> None:
        """At the end of the model run, all previously collected data is reported to disk."""
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

class Reporter:
    """This is the main reporter class for the GEB model. On initialization the ABMReporter and CWatMReporter classes are initalized.
    
    Args:
        model: The GEB model.
    """
    def __init__(self, model):
        self.model = model
        self.abm_reporter = ABMReporter(model)
        self.cwatmreporter = CWatMReporter(model)

    def step(self) -> None:
        """This function is called at the end of every timestep. This function only forwards the step function to the reporter for the ABM model and CWatM."""
        self.abm_reporter.step()
        self.cwatmreporter.step()

    def report(self):
        """At the end of the model run, all previously collected data is reported to disk. This function only forwards the report function to the reporter for the ABM model and CWatM. """
        self.abm_reporter.report()
        self.cwatmreporter.report()