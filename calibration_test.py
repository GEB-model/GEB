import yaml
import os
import pandas as pd
from pathlib import Path
from datetime import timedelta
from model import GEBModel
from honeybees.argparse import parser  
import geopandas as gpd
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from calibration import hydroStats

import faulthandler
faulthandler.enable()


parser.description = "GEB aims to simulate both environment, for now the hydrological system, the behaviour of people and their interactions at large scale without sacrificing too much detail."
parser.add_argument('--scenario', dest='scenario', type=str, default='base', help="""Here you can specify which scenario you would like to run. Currently 4 scenarios (base, self_investement, ngo_training, government_subsidies) are implemented, and model spinup are implemented.""")
parser.add_argument('--switch_crops', dest='switch_crops', default=False, action='store_true', required=False, help="""Whether agents should switch crops or not.   """)
parser.add_argument('--gpu_device', dest='gpu_device', type=int, default=0, required=False, help="""Specify the GPU to use (zero-indexed).""")
parser.add_argument('--profiling', dest='profiling', default=False, action='store_true', help="Run GEB with with profiling. If this option is used a file `profiling_stats.cprof` is saved in the working directory.")
parser.add_argument('--GPU', dest='use_gpu', default=False, action='store_true', help="Whether a GPU can be used to run the model. This requires CuPy to be installed.")
parser.add_argument('--config', dest='config', default='sandbox.yml', help="Path of the model configuration file.")
args = parser.parse_args()

config = yaml.load(open(Path(os.path.dirname(__file__), args.config), 'r'), Loader=yaml.FullLoader)

observations = pd.read_csv(Path(config['general']['original_data'], 'calibration', 'dev', 'discharge_sample.csv'), index_col=0, parse_dates=True)

def get_study_area(input_folder):
    study_area = {
        "name": "GEB"
    }
    gdf = gpd.read_file(os.path.join(input_folder, 'areamaps', 'region.geojson')).to_crs(epsg=4326)
    assert len(gdf) == 1, "There should be only one region in the region.geojson file."
    study_area['region'] = gdf.geometry[0]
    return study_area


study_area = get_study_area(config['general']['input_folder'])

def step(values):
    print()
    N = 28
    return_fraction, = values
    print('return_fraction', return_fraction)

    model = GEBModel(GEB_config_path=args.config, args=args, study_area=study_area)
    model.config['agent_settings']['farmers']['return_fraction'] = return_fraction
    model.step(N)
    model.groundwater_modflow_module.modflow.finalize()

    

    simulated_discharge = model.reporter.variables['discharge_sample'][-N:]
    end_date = model.current_time
    start_date = end_date - (N - 1) * model.timestep_length
    assert len(simulated_discharge) == N

    # get observation data between start and end date
    discharge = observations.loc[start_date:end_date].rename(columns={'discharge_sample': 'observed'})
    discharge['simulated'] = simulated_discharge

    discharge['delta'] = discharge['observed'] - discharge['simulated']
    discharge['delta_abs'] = discharge['delta'].abs()

    KGE = hydroStats.KGE(s=discharge['simulated'], o=discharge['observed'])

    total_error = discharge['delta_abs'].sum()

    # print('total_error', total_error)
    # print('KGE', KGE)

    return -KGE

# optimizer = forest_minimize
# optimizer = gbrt_minimize
optimizer = gp_minimize

res = optimizer(step, [(0., 1., "uniform")], n_calls=100, n_initial_points=10, n_points=10, initial_point_generator='random', noise='gaussian', n_jobs=1, verbose=True)

print(res)