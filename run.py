# -*- coding: utf-8 -*-
import cProfile
from pstats import Stats
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from honeybees.visualization.ModularVisualization import ModularServer
from honeybees.visualization.modules import ChartModule
from honeybees.visualization.canvas import Canvas
from honeybees.argparse import parser  
from model import GEBModel

import faulthandler
faulthandler.enable()

parser.description = "GEB aims to simulate both environment, for now the hydrological system, the behaviour of people and their interactions at large scale without sacrificing too much detail."
parser.add_argument('--scenario', dest='scenario', type=str, default='base', required=True, help="""Here you can specify which scenario you would like to run. Currently 4 scenarios (base, self_investement, ngo_training, government_subsidies) are implemented, and model spinup are implemented.""")
parser.add_argument('--switch_crops', dest='switch_crops', default=False, action='store_true', required=False, help="""Whether agents should switch crops or not.   """)
parser.add_argument('--gpu_device', dest='gpu_device', type=int, default=0, required=False, help="""Specify the GPU to use (zero-indexed).""")
parser.add_argument('--profiling', dest='profiling', default=False, action='store_true', help="Run GEB with with profiling. If this option is used a file `profiling_stats.cprof` is saved in the working directory.")
parser.add_argument('--GPU', dest='use_gpu', default=False, action='store_true', help="Whether a GPU can be used to run the model. This requires CuPy to be installed.")
parser.add_argument('--config', dest='config', default='GEB.yml', help="Path of the model configuration file.")

TEHSILS = [26, 30, 34, 35]

def get_study_area():
    study_area = {
        "name": "Bhima basin"
    }
    gdf = gpd.read_file(os.path.join('DataDrive', 'GEB_Bhima', 'input', 'areamaps', 'subdistricts_bhima.shp')).to_crs(epsg=4326)
    gdf = gdf[gdf['ID'].isin(TEHSILS)]
    tehsils = []
    color_map = plt.get_cmap('gist_rainbow')
    colors = {}
    for i, (_, tehsil) in enumerate(gdf.iterrows()):
        color = mcolors.rgb2hex(color_map(i / len(gdf)))
        tehsils.append({
            'geometry': tehsil.geometry.__geo_interface__,
            'properties': {
                'id': tehsil['ID'],
                'color': color
            }
        })
        colors[tehsil['ID']] = color
    study_area['tehsil'] = tehsils
    return study_area, colors


if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_gpu:
        import cupy
    import sys; sys.setrecursionlimit(2000)

    import faulthandler
    faulthandler.enable()

    MODEL_NAME = 'GEB'
    study_area, colors = get_study_area()

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
        #     {"name": "crop_sample", "size": 3, "color": ["#ff0000", "#00ff00", "#0000ff"]},
        # ],
        # [
        #     {"name": "surface_irrigated_sample", "size": 3, "color": ["#ff0000", "#00ff00", "#0000ff"]},
        # ],
        # [
        #     {"name": "groundwater_irrigated_sample", "size": 3, "color": ["#ff0000", "#00ff00", "#0000ff"]},
        # ],
        # [
        #     {
        #         "name": "groundwater_irrigated_tehsil",
        #         "ID": f"{admin['properties']['id']}"
        #     }
        #     for admin in study_area['admin']
        # ],
        # [
        #     {"name": "surface_irrigated_per_district", "IDs": TEHSILS, "color": [colors[tehsil] for tehsil in TEHSILS]},
        # ],
        # [
        #     {"name": "wealth_sample", "size": 3, "color": ["#ff0000", "#00ff00", "#0000ff"]},
        # ],
        [
            {"name": "discharge", "color": "#FF0000"},
        ],
        [
            {"name": "reservoir storage", "color": "#FF0000"},
        ],
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
        "GEB_config_path": args.config,
        "args": args,
        "study_area": study_area
    }

    if args.headless:
        model = GEBModel(**model_params)
        if args.profiling:
            with cProfile.Profile() as pr:
                model.run()
            with open('profiling_stats.cprof', 'w') as stream:
                stats = Stats(pr, stream=stream)
                stats.strip_dirs()
                stats.sort_stats('cumtime')
                stats.dump_stats('.prof_stats')
                stats.print_stats()
            pr.dump_stats('profile.prof')
        else:
            model.run()
        report = model.report()
    else:
        if args.profiling:
            print("Profiling not available for browser version")
        server_elements = [
            Canvas(max_canvas_height=800, max_canvas_width=1200)
        ] + [ChartModule(series) for series in series_to_plot]

        DISPLAY_TIMESTEPS = [
            'day',
            'week',
            'month',
            'year'
        ]

        server = ModularServer(MODEL_NAME, GEBModel, server_elements, DISPLAY_TIMESTEPS, model_params=model_params, port=None)
        server.launch(port=args.port, browser=args.browser)