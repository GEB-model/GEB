import click
import os
import cProfile
from pstats import Stats
import geopandas as gpd
import yaml
import logging
import functools
import faulthandler
from pathlib import Path

from honeybees.visualization.ModularVisualization import ModularServer
from honeybees.visualization.modules import ChartModule
from honeybees.visualization.canvas import Canvas

from hydromt.config import configread
import hydromt_geb
import geb
from geb.model import GEBModel
from geb.calibrate import calibrate as geb_calibrate

faulthandler.enable()


def multi_level_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            multi_level_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def parse_config(config):
    """Parse config."""
    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    if 'inherits' in config:
        inherited_config = yaml.load(open(config['inherits'], "r"), Loader=yaml.FullLoader)
        del config['inherits']
        config = multi_level_merge(inherited_config, config)
    return config


def create_logger(fp):
    logger = logging.getLogger(__name__)
    # set log level to debug
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    # add file handler
    Path(fp).parent.mkdir(exist_ok=True, parents=True)
    fh = logging.FileHandler(fp)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


@click.group()
@click.version_option(geb.__version__, message="GEB version: %(version)s")
@click.pass_context
def main(ctx):  # , quiet, verbose):
    """Command line interface for hydromt models."""
    if ctx.obj is None:
        ctx.obj = {}


def click_config(func):
    @click.option(
        "--config",
        "-c",
        default="model.yml",
        help="Path of the model configuration file.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@main.command()
@click_config
@click.option(
    "--scenario",
    "-s",
    type=str,
    default="spinup",
    required=True,
    help="""Here you can specify which scenario you would like to run. Currently 4 scenarios (base, self_investement, ngo_training, government_subsidies) are implemented, and model spinup are implemented.""",
)
@click.option(
    "--switch_crops",
    is_flag=True,
    help="""Whether agents should switch crops or not.""",
)
@click.option(
    "--gpu_device",
    type=int,
    default=0,
    help="""Specify the GPU to use (zero-indexed).""",
)
@click.option(
    "--profiling",
    is_flag=True,
    help="Run GEB with with profiling. If this option is used a file `profiling_stats.cprof` is saved in the working directory.",
)
@click.option(
    "--use_gpu",
    is_flag=True,
    help="Whether a GPU can be used to run the model. This requires CuPy to be installed.",
)
@click.option(
    "--working-directory", "-wd", default=".", help="Working directory for model."
)
@click.option(
    "--gui",
    is_flag=True,
    help="""The model can be run with or without a visual interface. The visual interface is useful to display the results in real-time while the model is running and to better understand what is going on. You can simply start or stop the model with the click of a buttion, or advance the model by an `x` number of timesteps. However, the visual interface is much slower than running the model without it.""",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="""Do not open browser when running the model. This option is, for example, useful when running the model on a server, and you would like to remotely access the model.""",
)
@click.option(
    "--port",
    type=int,
    default=8521,
    help="""Port used for display environment (default: 8521)""",
)
def run(
    scenario,
    switch_crops,
    gpu_device,
    profiling,
    use_gpu,
    config,
    working_directory,
    gui,
    no_browser,
    port,
):
    """Run model."""

    # set the working directory
    os.chdir(working_directory)

    if use_gpu:
        import cupy

    def get_study_area(model_structure):
        study_area = {"name": "GEB"}
        gdf = gpd.read_file(model_structure["geoms"]["areamaps/region"]).to_crs(
            epsg=4326
        )
        assert (
            len(gdf) == 1
        ), "There should be only one region in the region.geojson file."
        study_area["region"] = gdf.geometry[0]
        return study_area

    MODEL_NAME = "GEB"
    config = parse_config(config)

    model_structure = parse_config(
        "input/model_structure.json"
        if not "model_stucture" in config["general"]
        else config["general"]["model_stucture"]
    )
    for data in model_structure.values():
        for key, value in data.items():
            data[key] = Path(config["general"]["input_folder"]) / value
    study_area = get_study_area(model_structure)

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
        "model_structure": model_structure,
        "use_gpu": use_gpu,
        "gpu_device": gpu_device,
        "scenario": scenario,
        "study_area": study_area,
        "switch_crops": switch_crops,
    }

    if not gui:
        model = GEBModel(**model_params)
        if profiling:
            with cProfile.Profile() as pr:
                model.run()
            with open("profiling_stats.cprof", "w") as stream:
                stats = Stats(pr, stream=stream)
                stats.strip_dirs()
                stats.sort_stats("cumtime")
                stats.dump_stats(".prof_stats")
                stats.print_stats()
            pr.dump_stats("profile.prof")
        else:
            model.run()
        report = model.report()
    else:
        if profiling:
            print("Profiling not available for browser version")
        server_elements = [Canvas(max_canvas_height=800, max_canvas_width=1200)] + [
            ChartModule(series) for series in series_to_plot
        ]

        DISPLAY_TIMESTEPS = ["day", "week", "month", "year"]

        server = ModularServer(
            MODEL_NAME,
            GEBModel,
            server_elements,
            DISPLAY_TIMESTEPS,
            model_params=model_params,
            port=None,
        )
        server.launch(port=port, browser=no_browser)

    if use_gpu:
        from numba import cuda

        device = cuda.get_current_device()
        device.reset()


@main.command()
@click_config
@click.option(
    "--working-directory", "-wd", default=".", help="Working directory for model."
)
def calibrate(config, working_directory):
    os.chdir(working_directory)

    config = parse_config(config)
    geb_calibrate(config, working_directory)


def click_build_options(func):
    @click_config
    @click.option(
        "--data-catalog",
        "-d",
        type=str,
        multiple=True,
        default=[os.environ.get("GEB_DATA_CATALOG", "data_catalog.yml")],
        help="""A list of paths to the data library YAML files. By default the GEB_DATA_CATALOG environment variable is used. If this is not set, defaults to data_catalog.yml""",
    )
    @click.option(
        "--build-config",
        "-b",
        default="build.yml",
        help="Path of the model build configuration file.",
    )
    @click.option(
        "--working-directory", "-wd", default=".", help="Working directory for model."
    )
    @click.option(
        "--data-provider",
        "-p",
        default=os.environ.get("GEB_DATA_PROVIDER", None),
        help="Data variant to use from data catalog (see hydroMT documentation).",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@main.command()
@click_build_options
def build(data_catalog, config, build_config, working_directory, data_provider):
    """Build model."""

    # set the working directory
    os.chdir(working_directory)

    config = parse_config(config)
    input_folder = Path(config["general"]["input_folder"])

    geb_model = hydromt_geb.GEBModel(
        root=input_folder,
        mode="w+",
        data_libs=data_catalog,
        logger=create_logger("build.log"),
        data_provider=data_provider,
    )

    pour_point = config["general"]["pour_point"]
    geb_model.build(
        opt=configread(build_config),
        region={
            "subbasin": [[pour_point[0]], [pour_point[1]]],
        },
    )


@main.command()
@click_build_options
@click.option("--model", "-m", default="../base", help="Folder for base model.")
def alter(data_catalog, config, build_config, working_directory, model, data_provider):
    """Build model."""

    # set the working directory
    os.chdir(working_directory)

    config = parse_config(config)
    reference_model_folder = Path(model) / Path(config["general"]["input_folder"])

    geb_model = hydromt_geb.GEBModel(
        root=reference_model_folder,
        mode="w+",
        data_libs=data_catalog,
        logger=create_logger("build.log"),
        data_provider=data_provider,
    )

    geb_model.read()
    geb_model.set_alternate_root(
        Path(".") / Path(config["general"]["input_folder"]), mode="w+"
    )
    geb_model.update(
        opt=configread(build_config),
        model_out=Path(".") / Path(config["general"]["input_folder"]),
    )


@main.command()
@click_build_options
def update(data_catalog, config, build_config, working_directory, data_provider):
    """Update model."""

    # set the working directory
    os.chdir(working_directory)

    config = parse_config(config)
    input_folder = Path(config["general"]["input_folder"])

    geb_model = hydromt_geb.GEBModel(
        root=input_folder,
        mode="r+",
        data_libs=data_catalog,
        logger=create_logger("build_update.log"),
        # data_provider=data_provider,
    )
    geb_model.read()
    geb_model.update(opt=configread(build_config))


if __name__ == "__main__":
    main()
