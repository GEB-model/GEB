import rasterio
import cProfile
from pstats import Stats

from hyve.visualization.ModularVisualization import ModularServer
from hyve.visualization.modules import ChartModule
from hyve.visualization.canvas import Canvas
from hyve.argparse import parser  
from model import GEBModel

import faulthandler
faulthandler.enable()

def krishna():
    with rasterio.open('DataDrive/GEB/input/areamaps/mask.tif') as src:
        bounds = src.bounds

    return 'krishna', bounds.left, bounds.right, bounds.bottom, bounds.top

parser.description = "GEB aims to simulate both environment, for now the hydrological system, the behaviour of people and their interactions at large scale without sacrificing too much detail."
parser.add_argument('--scenario', dest='scenario', type=str, default='base', required=True, help="""Here you can specify which scenario you would like to run. Currently 4 scenarios (base, self_investement, ngo_training, government_subsidies) are implemented, and model spinup are implemented.
""")
parser.add_argument('--export_folder', dest='export_folder', type=str, default=None, help="The folder to export model results to. If not specified the name of the scenario is used.")
parser.add_argument('--profiling', dest='profiling', default=False, action='store_true', help="Run GEB with with profiling. If this option is used a file `profiling_stats.cprof` is saved in the working directory.")
parser.add_argument('--GPU', dest='use_gpu', default=False, action='store_true', help="Whether a GPU can be used to run the model. This requires CuPy to be installed.")
if __name__ == '__main__':
    args = parser.parse_args()
    if args.export_folder is None and args.scenario is not None:
        args.export_folder = args.scenario
    import sys; sys.setrecursionlimit(2000)

    import faulthandler
    faulthandler.enable()

    study_area_name, xmin, xmax, ymin, ymax = krishna()

    CWATM_SETTINGS = 'CWatM_GEB.ini'
    ABM_CONFIG_PATH = 'GEB.yml'

    MODEL_NAME = 'GEB'

    # crops_colors = pd.read_excel('DataDrive/Bhima/crops/crop_data.xlsx', index_col=1)['Color'].to_dict()

    series_to_plot = [
        # crop_series,
        # [
        #     {"name": "channel abstraction M3", "color": "#FF0000"},
        #     {"name": "reservoir abstraction M3", "color": "#FFFF00"},
        #     {"name": "groundwater abstraction M3", "color": "#000000"},
        #     # {"name": "total potential irrigation consumption M3", "color": "#FF00FF"},
        # ],
        # [
        #     {"name": "channel irrigation", "color": "#FF0000"},
        #     {"name": "reservoir irrigation", "color": "#FFFF00"},
        #     {"name": "groundwater irrigation", "color": "#000000"},
        # ],
        # [
        #     {"name": "reservoir storage", "color": "#FF0000"},
        # ],
        # [
        #     {"name": "w1", "color": "#FF0000"},
        #     {"name": "w2", "color": "#FFAA00"},
        #     {"name": "w3", "color": "#FF00AA"},
        #     {"name": "potevap", "color": "#FFFF00"},
        #     {"name": "actevap", "color": "#000000"},
        # ],
        [
            {"name": "hydraulic head", "color": "#FF0000"},
        ],
        # [
        #     {"name": "precipitation", "color": "#FF0000"},
        # ],
    ]

    model_params = {
        "CwatM_settings": CWATM_SETTINGS,
        "GEB_config_path": ABM_CONFIG_PATH,
        "name": study_area_name,
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax, 
        "args": args,
    }

    if args.headless:
        model = GEBModel(**model_params)
        if args.profiling:
            with cProfile.Profile() as pr:
                model.run()
            with open('profiling_stats.cprof', 'w') as stream:
                stats = Stats(pr, stream=stream)
                stats.strip_dirs()
                stats.sort_stats('time')
                stats.dump_stats('.prof_stats')
                stats.print_stats()
        else:
            model.run()
        report = model.report()
    else:
        if args.profiling:
            print("Profiling not available for browser version")
        server_elements = [
            Canvas(xmin, xmax, ymin, ymax, max_canvas_height=800, max_canvas_width=1200)
        ] + [ChartModule(series) for series in series_to_plot]

        DISPLAY_TIMESTEPS = [
            'day',
            'week',
            'month',
            'year'
        ]

        server = ModularServer(MODEL_NAME, GEBModel, server_elements, DISPLAY_TIMESTEPS, model_params=model_params, port=None)
        server.launch(port=args.port, browser=args.browser)