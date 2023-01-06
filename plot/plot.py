import os
import numpy as np

def read_npy(report_folder, name, dt, scenario=None):
    if scenario:
        report_folder = os.path.join(report_folder, scenario)
    fn = os.path.join(report_folder, name, dt.isoformat().replace(':', '').replace('-', '') + '.npy')
    return np.load(fn)
