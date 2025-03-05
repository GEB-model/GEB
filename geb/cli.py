import click
import os
import platform
import subprocess
import tempfile
import sys
import cProfile
from pstats import Stats
from operator import attrgetter
import yaml
import logging
import functools
from pathlib import Path
import warnings
import importlib

from honeybees.visualization.ModularVisualization import ModularServer
from honeybees.visualization.modules.ChartVisualization import ChartModule
from honeybees.visualization.canvas import Canvas

from hydromt.config import configread
from geb import __version__
from geb import setup
from geb.model import GEBModel
from geb.calibrate import calibrate as geb_calibrate
from geb.sensitivity import sensitivity_analysis as geb_sensitivity_analysis
from geb.multirun import multi_run as geb_multi_run


def multi_level_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            multi_level_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


class DetectDuplicateKeysYamlLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key found: {key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def parse_config(config_path, current_directory=None):
    """Parse config."""
    if current_directory is None:
        current_directory = Path.cwd()

    if isinstance(config_path, dict):
        config = config_path
    else:
        config = yaml.load(
            open(current_directory / config_path, "r"),
            Loader=DetectDuplicateKeysYamlLoader,
        )
        current_directory = current_directory / Path(config_path).parent

    if "inherits" in config:
        inherit_config_path = config["inherits"]
        inherit_config_path = inherit_config_path.format(**os.environ)
        # replace {VAR} with environment variable VAR if it exists
        inherit_config_path = os.path.expandvars(inherit_config_path)
        # if inherits is not an absolute path, we assume it is relative to the config file
        if not Path(inherit_config_path).is_absolute():
            inherit_config_path = current_directory / config["inherits"]
        inherited_config = yaml.load(
            open(inherit_config_path, "r"),
            Loader=yaml.FullLoader,
        )
        current_directory = current_directory / Path(inherit_config_path).parent
        del config[
            "inherits"
        ]  # remove inherits key from config to avoid infinite recursion
        config = multi_level_merge(inherited_config, config)
        config = parse_config(config, current_directory=current_directory)
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
@click.version_option(__version__, message="GEB version: %(version)s")
@click.pass_context
def cli(ctx):  # , quiet, verbose):
    """Command line interface for GEB."""
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


def click_run_options():
    def decorator(func):
        @click_config
        @click.option(
            "--working-directory",
            "-wd",
            default=".",
            help="Working directory for model.",
        )
        @click.option(
            "--gui",
            is_flag=True,
            help="""The model can be run with a graphical user interface in a browser. The visual interface is useful to display the results in real-time while the model is running and to better understand what is going on. You can simply start or stop the model with the click of a buttion, or advance the model by an `x` number of timesteps. However, the visual interface is much slower than running the model without it.""",
        )
        @click.option(
            "--no-browser",
            is_flag=True,
            help="""Run graphical user interface, but serve interface through the server but do not open the browser. You may connect to the server from a browswer. This option is only works in combination with the graphical user interface.""",
        )
        @click.option(
            "--port",
            type=int,
            default=8521,
            help="""Port used for graphical user interface (default: 8521)""",
        )
        @click.option(
            "--profiling",
            is_flag=True,
            help="Run GEB with profiling. If this option is used a file `profiling_stats.cprof` is saved in the working directory.",
        )
        @click.option(
            "--optimize",
            is_flag=True,
            help="Run GEB in optimized mode, skipping asserts and water balance checks.",
        )
        @click.option("--timing", is_flag=True, help="Run GEB with timing.")
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def run_model_with_method(
    method,
    profiling,
    config,
    working_directory,
    gui,
    no_browser,
    port,
    timing,
    optimize,
):
    """Run model."""
    # check if we need to run the model in optimized mode
    # if the model is already running in optimized mode, we don't need to restart it
    # or else we start an infinite loop
    if optimize and sys.flags.optimize == 0:
        if platform.system() == "Windows":
            # If the script is not a .py file, we need to add the .exe extension
            if not sys.argv[0].endswith(".py"):
                sys.argv[0] = sys.argv[0] + ".exe"
            subprocess.run([sys.executable, "-O"] + sys.argv)
        else:
            os.execv(sys.executable, ["-O"] + sys.argv)

    # set the working directory
    os.chdir(working_directory)

    config = parse_config(config)

    files = parse_config(
        "input/files.json"
        if "files" not in config["general"]
        else config["general"]["files"]
    )

    model_params = {
        "config": config,
        "files": files,
        "timing": timing,
    }

    if not gui:
        if profiling:
            profile = cProfile.Profile()
            profile.enable()

        with GEBModel(**model_params) as model:
            getattr(model, method)()

        if profiling:
            profile.disable()
            with open("profiling_stats.cprof", "w") as stream:
                stats = Stats(profile, stream=stream)
                stats.strip_dirs()
                stats.sort_stats("cumtime")
                stats.dump_stats(".prof_stats")
                stats.print_stats()
            profile.dump_stats("profile.prof")

    else:
        # Using the GUI, GEB runs in an asyncio event loop. This is not compatible with
        # the event loop started for reading data, unless we use nest_asyncio.
        # so that's what we do here.
        import nest_asyncio

        nest_asyncio.apply()

        if profiling:
            print("Profiling not available for browser version")
        server_elements = [Canvas(max_canvas_height=800, max_canvas_width=1200)]
        if "draw" in config and "plot" in config["draw"] and config["draw"]["plot"]:
            server_elements = server_elements + [
                ChartModule(series) for series in config["draw"]["plot"]
            ]

        DISPLAY_TIMESTEPS = ["day", "week", "month", "year"]

        server = ModularServer(
            "GEB",
            GEBModel,
            server_elements,
            DISPLAY_TIMESTEPS,
            model_params=model_params,
            port=None,
            initialization_method=method,
        )
        server.launch(port=port, browser=no_browser)


@cli.command()
@click_run_options()
def run(*args, **kwargs):
    run_model_with_method(method="run", *args, **kwargs)


@cli.command()
@click_run_options()
def spinup(*args, **kwargs):
    run_model_with_method(method="spinup", *args, **kwargs)


@cli.command()
@click.argument("method", required=True)
@click_run_options()
def exec(method, *args, **kwargs):
    run_model_with_method(method=method, *args, **kwargs)


@cli.command()
@click_config
@click.option(
    "--working-directory", "-wd", default=".", help="Working directory for model."
)
def calibrate(config, working_directory):
    os.chdir(working_directory)

    config = parse_config(config)
    geb_calibrate(config, working_directory)


@cli.command()
@click_config
@click.option(
    "--working-directory", "-wd", default=".", help="Working directory for model."
)
def sensitivity(config, working_directory):
    os.chdir(working_directory)

    config = parse_config(config)
    geb_sensitivity_analysis(config, working_directory)


@cli.command()
@click_config
@click.option(
    "--working-directory", "-wd", default=".", help="Working directory for model."
)
def multirun(config, working_directory):
    os.chdir(working_directory)

    config = parse_config(config)
    geb_multi_run(config, working_directory)


def click_build_options(build_config="build.yml"):
    def decorator(func):
        @click_config
        @click.option(
            "--data-catalog",
            "-d",
            type=str,
            multiple=True,
            default=[Path(os.environ.get("GEB_PACKAGE_DIR")) / "data_catalog.yml"],
            help="""A list of paths to the data library YAML files. By default the data_catalog in the examples is used. If this is not set, defaults to data_catalog.yml""",
        )
        @click.option(
            "--build-config",
            "-b",
            default=build_config,
            help="Path of the model build configuration file.",
        )
        @click.option(
            "--custom-model",
            default=None,
            type=str,
            help="name of custom preprocessing model",
        )
        @click.option(
            "--working-directory",
            "-wd",
            default=".",
            help="Working directory for model.",
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

    return decorator


def get_model(custom_model):
    if custom_model is None:
        return setup.GEBModel
    else:
        importlib.import_module(
            "." + custom_model.split(".")[0], package="geb.setup.custom_models"
        )
        return attrgetter(custom_model)(setup.custom_models)


def customize_data_catalog(data_catalogs):
    """This functions adds the GEB_DATA_ROOT to the data catalog if it is set as an environment variable.
    This enables reading the data catalog from a different location than the location of the yml-file
    without the need to specify root in the meta of the data catalog."""
    geb_data_root = os.environ.get("GEB_DATA_ROOT", None)

    if geb_data_root:
        customized_data_catalogs = []
        for data_catalog in data_catalogs:
            with open(data_catalog, "r") as stream:
                data_catalog_yml = yaml.load(stream, Loader=yaml.FullLoader)

                if "meta" not in data_catalog_yml:
                    data_catalog_yml["meta"] = {}
                data_catalog_yml["meta"]["root"] = geb_data_root

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
                yaml.dump(data_catalog_yml, tmp, default_flow_style=False)
                customized_data_catalogs.append(tmp.name)
        return customized_data_catalogs
    else:
        return data_catalogs


def build_fn(
    data_catalog, config, build_config, custom_model, working_directory, data_provider
):
    """Build model."""

    # set the working directory
    os.chdir(working_directory)

    config = parse_config(config)
    input_folder = Path(config["general"]["input_folder"])

    arguments = {
        "root": input_folder,
        "mode": "w+",
        "data_libs": customize_data_catalog(data_catalog),
        "logger": create_logger("build.log"),
        "data_provider": data_provider,
    }

    geb_model = get_model(custom_model)(**arguments)

    # TODO: remove pour_point option in future versions
    if "pour_point" in config["general"]:
        assert "region" not in config
        warnings.warn(
            "The `pour_point` option is deprecated and will be removed in future versions. Please use `region.pour_point` instead.",
            DeprecationWarning,
        )
        config["general"]["region"] = {}
        config["general"]["region"]["pour_point"] = config["general"]["pour_point"]

    geb_model.build(
        opt=configread(build_config),
        region=config["general"]["region"],
    )


@cli.command()
@click_build_options()
def build(*args, **kwargs):
    build_fn(*args, **kwargs)


@cli.command()
@click_build_options()
@click.option("--model", "-m", default="../base", help="Folder for base model.")
def alter(
    data_catalog,
    config,
    build_config,
    custom_model,
    working_directory,
    model,
    data_provider,
):
    """Build model."""

    # set the working directory
    os.chdir(working_directory)

    config = parse_config(config)
    reference_model_folder = Path(model) / Path(config["general"]["input_folder"])

    arguments = {
        "root": reference_model_folder,
        "mode": "r+",
        "data_libs": customize_data_catalog(data_catalog),
        "logger": create_logger("build.log"),
        "data_provider": data_provider,
    }

    geb_model = get_model(custom_model)(**arguments)
    geb_model.read()
    geb_model.set_alternate_root(
        Path(".") / Path(config["general"]["input_folder"]), mode="w+"
    )
    geb_model.update(
        opt=configread(build_config),
        model_out=Path(".") / Path(config["general"]["input_folder"]),
    )


@cli.command()
@click_build_options(build_config="update.yml")
def update(
    data_catalog, config, build_config, custom_model, working_directory, data_provider
):
    """Update model."""

    # set the working directory
    os.chdir(working_directory)

    config = parse_config(config)
    input_folder = Path(config["general"]["input_folder"])

    arguments = {
        "root": input_folder,
        "mode": "r+",
        "data_libs": customize_data_catalog(data_catalog),
        "logger": create_logger("build_update.log"),
        "data_provider": data_provider,
    }

    geb_model = get_model(custom_model)(**arguments)
    geb_model.read()
    geb_model.update(opt=configread(build_config))


@cli.command()
def evaluate():
    """Evaluate model."""
    raise NotImplementedError


@click.option(
    "--working-directory",
    "-wd",
    default=".",
    help="Working directory for model.",
)
@click.option(
    "--name",
    "-n",
    default="model",
    help="Working directory for model.",
)
@cli.command()
def share(working_directory, name):
    """Share model."""

    os.chdir(working_directory)

    # create a zip file called model.zip with the folders input, and model files
    # in the working directory
    import zipfile

    folders = ["input"]
    files = ["model.yml", "build.yml"]
    optional_files = ["update.yml", "data_catalog.yml"]
    zip_filename = f"{name}.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        total_files = (
            sum(
                [
                    sum(len(files) for _, _, files in os.walk(folder))
                    for folder in folders
                ]
            )
            + len(files)
            + len(optional_files)
        )  # Count total number of files
        progress = 0  # Initialize progress counter
        for folder in folders:
            for root, _, filenames in os.walk(folder):
                for filename in filenames:
                    zipf.write(os.path.join(root, filename))
                    progress += 1  # Increment progress counter
                    print(
                        f"Exporting file {progress}/{total_files} to {zip_filename}",
                        end="\r",
                    )  # Print progress
        for file in files:
            zipf.write(file)
            progress += 1  # Increment progress counter
            print(
                f"Exporting file {progress}/{total_files} to {zip_filename}", end="\r"
            )  # Print progress
        for file in optional_files:
            if os.path.exists(file):
                zipf.write(file)
            progress += 1  # Increment progress counter
            print(
                f"Exporting file {progress}/{total_files} to {zip_filename}", end="\r"
            )  # Print progress
        print(f"Exporting file {progress}/{total_files} to {zip_filename}")
        print("Done!")


if __name__ == "__main__":
    cli()
