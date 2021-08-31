import os
import numpy as np
import yaml

with open('GEB.yml', 'r') as f:
    config = yaml.load(f)

def read_npy(name, dt, scenario=None):
    if scenario:
        report_folder = os.path.join('report', scenario)
    else:
        report_folder = 'report'
    fn = os.path.join(report_folder, name, dt.isoformat().replace(':', '').replace('-', '') + '.npy')
    return np.load(fn)
