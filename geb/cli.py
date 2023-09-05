import click
import os
import cProfile
from pstats import Stats
import geopandas as gpd
import yaml
import faulthandler

from honeybees.visualization.ModularVisualization import ModularServer
from honeybees.visualization.modules import ChartModule
from honeybees.visualization.canvas import Canvas

from .model import GEBModel
from . import __version__

faulthandler.enable()

@click.group()
@click.version_option(__version__, message="GEB version: %(version)s")
@click.pass_context
def main(ctx):  # , quiet, verbose):
    """Command line interface for hydromt models."""
    if ctx.obj is None:
        ctx.obj = {}

@main.command()
@click.option('--scenario', type=str, default='base', required=True, help="""Here you can specify which scenario you would like to run. Currently 4 scenarios (base, self_investement, ngo_training, government_subsidies) are implemented, and model spinup are implemented.""")
@click.option('--switch_crops', is_flag=True, help="""Whether agents should switch crops or not.""")
@click.option('--gpu_device', type=int, default=0, help="""Specify the GPU to use (zero-indexed).""")
@click.option('--profiling', is_flag=True, help="Run GEB with with profiling. If this option is used a file `profiling_stats.cprof` is saved in the working directory.")
@click.option('--use_gpu', is_flag=True, help="Whether a GPU can be used to run the model. This requires CuPy to be installed.")
@click.option('--config', default='sandbox.yml', help="Path of the model configuration file.")
@click.option('--gui', is_flag=True, help="""The model can be run with or without a visual interface. The visual interface is useful to display the results in real-time while the model is running and to better understand what is going on. You can simply start or stop the model with the click of a buttion, or advance the model by an `x` number of timesteps. However, the visual interface is much slower than running the model without it.""")
@click.option('--no-browser', is_flag=True, help="""Do not open browser when running the model. This option is, for example, useful when running the model on a server, and you would like to remotely access the model.""")
@click.option('--port', type=int, default=8521, help="""Port used for display environment (default: 8521)""")
def run(scenario, switch_crops, gpu_device, profiling, use_gpu, config, gui, no_browser, port):
    """Run model."""
    if use_gpu:
        import cupy

    def get_study_area(input_folder):
        study_area = {
            "name": "GEB"
        }
        gdf = gpd.read_file(os.path.join(input_folder, 'areamaps', 'region.geojson')).to_crs(epsg=4326)
        assert len(gdf) == 1, "There should be only one region in the region.geojson file."
        study_area['region'] = gdf.geometry[0]
        return study_area


    MODEL_NAME = 'GEB'
    config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
    study_area = get_study_area(config['general']['input_folder'])

    series_to_plot = [
        # crop_series,
        # [
        #     {"name": "channel abstraction M3", "color": "#FF0000"},co
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
        # [
        #     {"name": "discharge", "color": "#FF0000"},
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
        [
            {"name": "precipitation", "color": "#FF0000"},
        ],
        [
            {"name": "disposable_income sample", "color": "#FF0000"},
        ],
        [
            {"name": "wealth sample", "color": "#FF0000"},
         ],
        [
            {"name": "mean_risk_perception", "color": "#FF0000"},
        ],

    ]

    model_params = {
        "GEB_config_path": config,
        "use_gpu": use_gpu,
        "scenario": scenario,
        "study_area": study_area
    }

    if not gui:
        model = GEBModel(**model_params)
        if profiling:
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
        if profiling:
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
        server.launch(port=port, browser=no_browser)

    if use_gpu:
        from numba import cuda 
        device = cuda.get_current_device()
        device.reset()