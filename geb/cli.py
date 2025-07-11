import cProfile
import functools
import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from operator import attrgetter
from pathlib import Path
from pstats import Stats
from typing import Any

import click
import yaml
from honeybees.visualization.canvas import Canvas
from honeybees.visualization.ModularVisualization import ModularServer
from honeybees.visualization.modules.ChartVisualization import ChartModule

from geb import __version__
from geb.build import GEBModel as GEBModelBuild
from geb.calibrate import calibrate as geb_calibrate
from geb.model import GEBModel
from geb.multirun import multi_run as geb_multi_run
from geb.sensitivity import sensitivity_analysis as geb_sensitivity_analysis
from geb.workflows.io import WorkingDirectory


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


def parse_config(
    config_path: dict | Path | str, current_directory: Path | None = None
) -> dict[str, Any]:
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
    default: str = "model.yml"

    @click.option(
        "--config",
        "-c",
        default=default,
        help=f"Path of the model configuration file. Defaults to '{default}'.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def working_directory_option(func):
    @click.option(
        "--working-directory",
        "-wd",
        default=".",
        help="Working directory for model. Default is the current directory.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def click_run_options():
    def decorator(func):
        @click_config
        @working_directory_option
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
    method: str | None,
    profiling: bool,
    config,
    working_directory,
    gui,
    no_browser,
    port,
    timing,
    optimize,
    method_args: dict = {},
    close_after_run=True,
) -> GEBModel | None:
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

    with WorkingDirectory(working_directory):
        config: dict[str, Any] = parse_config(config)

        files: dict[str, Any] = parse_config(
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

            geb = GEBModel(**model_params)
            if method is not None:
                getattr(geb, method)(**method_args)
            if close_after_run:
                geb.close()

            if profiling:
                profile.disable()
                with open("profiling_stats.cprof", "w") as stream:
                    stats = Stats(profile, stream=stream)
                    stats.strip_dirs()
                    stats.sort_stats("cumtime")
                    stats.dump_stats(".prof_stats")
                    stats.print_stats()
                profile.dump_stats("profile.prof")

            return geb

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

            DISPLAY_TIMESTEPS = [1, 10, 100, 1000]

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
@working_directory_option
def calibrate(config, working_directory):
    with WorkingDirectory(working_directory):
        config = parse_config(config)
        geb_calibrate(config, working_directory)


@cli.command()
@click_config
@working_directory_option
def sensitivity(config, working_directory):
    with WorkingDirectory(working_directory):
        config = parse_config(config)
        geb_sensitivity_analysis(config, working_directory)


@cli.command()
@click_config
@working_directory_option
def multirun(config, working_directory):
    with WorkingDirectory(working_directory):
        config = parse_config(config)
        geb_multi_run(config, working_directory)


def click_build_options(build_config="build.yml", build_config_help_extra=None):
    def decorator(func):
        build_config_help = (
            f"Path of the model build configuration file. Defaults to '{build_config}'."
        )
        if build_config_help_extra:
            build_config_help += f"""

            {build_config_help_extra}"""

        @click_config
        @click.option(
            "--build-config",
            "-b",
            default=build_config,
            help=build_config_help,
        )
        @click.option(
            "--custom-model",
            default=None,
            type=str,
            help="name of custom preprocessing model",
        )
        @working_directory_option
        @click.option(
            "--data-catalog",
            "-d",
            type=str,
            multiple=True,
            default=[Path(os.environ.get("GEB_PACKAGE_DIR")) / "data_catalog.yml"],
            help="""A list of paths to the data library YAML files. By default the data_catalog in the examples is used. If this is not set, defaults to data_catalog.yml""",
        )
        @click.option(
            "--data-provider",
            "-p",
            default=os.environ.get("GEB_DATA_PROVIDER", "default"),
            help="Data variant to use from data catalog (see hydroMT documentation).",
        )
        @click.option(
            "--data-root",
            "-r",
            default=os.environ.get(
                "GEB_DATA_ROOT",
                Path(os.environ.get("GEB_PACKAGE_DIR")) / ".." / ".." / "data_catalog",
            ),
            help="Root folder where the data is located. When the environment variable GEB_DATA_ROOT is set, this is used as the root folder for the data catalog. If not set, defaults to the data_catalog folder in parent of the GEB source code directory.",
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_model_builder_class(custom_model) -> type:
    if custom_model is None:
        return GEBModelBuild
    else:
        importlib.import_module(
            "." + custom_model.split(".")[0], package="geb.build.custom_models"
        )
        return attrgetter(custom_model)(build.custom_models)


def customize_data_catalog(data_catalogs, data_root=None):
    """This functions adds the GEB_DATA_ROOT to the data catalog if it is set as an environment variable.

    This enables reading the data catalog from a different location than the location of the yml-file
    without the need to specify root in the meta of the data catalog.
    """
    if data_root:
        customized_data_catalogs = []
        for data_catalog in data_catalogs:
            with open(data_catalog, "r") as stream:
                data_catalog_yml = yaml.load(stream, Loader=yaml.FullLoader)

                if "meta" not in data_catalog_yml:
                    data_catalog_yml["meta"] = {}
                data_catalog_yml["meta"]["root"] = str(data_root)

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
                yaml.dump(data_catalog_yml, tmp, default_flow_style=False)
                customized_data_catalogs.append(tmp.name)
        return customized_data_catalogs
    else:
        return data_catalogs


def get_builder(config, data_catalog, custom_model, data_provider, data_root):
    config = parse_config(config)
    input_folder = Path(config["general"]["input_folder"])

    arguments = {
        "root": input_folder,
        "data_catalogs": customize_data_catalog(data_catalog, data_root=data_root),
        "logger": create_logger("build.log"),
        "data_provider": data_provider,
    }

    return get_model_builder_class(custom_model)(**arguments)


def init_fn(
    config: str | Path,
    build_config: str | Path,
    update_config: str | Path,
    working_directory: str | Path,
    from_example: str,
    basin_id: str | None = None,
    overwrite: bool = False,
) -> None:
    """Create a new model.

    Args:
        config: Path to the model configuration file to create.
        build_config: Path to the model build configuration file to create.
        update_config: Path to the model update configuration file to create.
        working_directory: Working directory for the model.
        from_example: Name of the example to use as a base for the model.
        basin_id:Basin ID(s) to use for the model. Can be a comma-separated list of integers.
            If not set, the basin ID is taken from the config file.
        overwrite: If True, overwrite existing config and build config files. Defaults to False.

    """
    config: Path = Path(config)
    build_config: Path = Path(build_config)
    update_config: Path = Path(update_config)
    working_directory: Path = Path(working_directory)

    if not working_directory.exists():
        working_directory.mkdir(parents=True, exist_ok=True)

    with WorkingDirectory(working_directory):
        if config.exists() and not overwrite:
            raise FileExistsError(
                f"Config file {config} already exists. Please remove it or use a different name, or use --overwrite."
            )

        if build_config.exists() and not overwrite:
            raise FileExistsError(
                f"Build config file {build_config} already exists. Please remove it or use a different name, or use --overwrite."
            )

        if update_config.exists() and not overwrite:
            raise FileExistsError(
                f"Update config file {update_config} already exists. Please remove it or use a different name, or use --overwrite."
            )

        example_folder: Path = (
            Path(os.environ.get("GEB_PACKAGE_DIR")) / ".." / "examples" / from_example
        )
        if not example_folder.exists():
            raise FileNotFoundError(
                f"Example folder {example_folder} does not exist. Did you use the right --from-example option?"
            )

        config_dict: dict = yaml.load(
            open(example_folder / "model.yml", "r"),
            Loader=DetectDuplicateKeysYamlLoader,
        )

        if basin_id is not None:
            # Allow passing a comma-separated list of integers
            if "," in basin_id:
                basin_ids: list[int] = [
                    int(x) for x in basin_id.split(",") if x.strip()
                ]
            else:
                basin_ids: int = int(basin_id)

            config_dict["general"]["region"]["subbasin"] = basin_ids

        with open(config, "w") as f:
            # do not sort keys, to keep the order of the config file
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        shutil.copy(example_folder / "build.yml", build_config)
        shutil.copy(example_folder / "update.yml", update_config)


@cli.command()
@click_config
@click.option(
    "--build-config",
    "-b",
    default="build.yml",
    help="Path of the model build configuration file. Defaults to 'build.yml'.",
)
@click.option(
    "--update-config",
    "-u",
    default="update.yml",
    help="Path of the model update configuration file.",
)
@click.option(
    "--from-example",
    default="geul",
    help="Name of the example to use as a base for the model. Defaults to 'geul'.",
)
@click.option(
    "--basin-id",
    default=None,
    type=str,
    help="Basin ID(s) to use for the model. Comma-separated list of integers. If not set, the basin ID is taken from the config file.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="If set, overwrite existing config and build config files.",
)
@working_directory_option
def init(*args, **kwargs):
    """Initialize a new model."""
    # Initialize the model with the given config and build config
    init_fn(*args, **kwargs)


def build_fn(
    data_catalog,
    config,
    build_config,
    custom_model,
    working_directory,
    data_provider,
    data_root,
):
    """Build model."""
    with WorkingDirectory(working_directory):
        model = get_builder(
            config,
            data_catalog,
            custom_model,
            data_provider,
            data_root,
        )
        model.build(
            methods=parse_config(build_config),
            region=parse_config(config)["general"]["region"],
        )


@cli.command()
@click_build_options()
def build(*args, **kwargs):
    build_fn(*args, **kwargs)


def alter_fn(
    data_catalog,
    config: dict,
    build_config: dict,
    custom_model,
    working_directory: Path | str,
    from_model: str,
    data_provider,
    data_root,
) -> None:
    from_model: Path = Path(from_model)

    with WorkingDirectory(working_directory):
        original_config: Path = from_model / config

        config_dict: dict[str, str] = {"inherits": str(original_config)}
        with open(config, "w") as f:
            # do not sort keys, to keep the order of the config file
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        config_from_original_model = parse_config(from_model / config)
        input_folder: Path = Path(config_from_original_model["general"]["input_folder"])

        original_input_path: Path = from_model / input_folder

        with open(original_input_path / "files.json", "r") as f:
            original_files = json.load(f)

            for file_class, files in original_files.items():
                for file_name, file_path in files.items():
                    if not file_path.startswith("/"):
                        original_files[file_class][file_name] = str(
                            from_model / original_input_path / file_path
                        )

        input_folder.mkdir(parents=True, exist_ok=True)
        with open(input_folder / "files.json", "w") as f:
            json.dump(original_files, f, indent=4)

        model = get_builder(
            config,
            data_catalog,
            custom_model,
            data_provider,
            data_root,
        )
        model.read()

        model.update(
            methods=parse_config(build_config),
        )


@cli.command()
@click_build_options()
@click.option("--from-model", default="../base", help="Folder for the existing model.")
def alter(*args, **kwargs):
    """Create alternative version from base model with only changed files.

    This command is useful to create a new model based on an existing one, but with
    only a few changes. It will copy the base model and overwrite the files that are
    specified in the config and build config files. The rest of the files will be
    linked to the original model to reduce disk space.
    """
    alter_fn(*args, **kwargs)


def update_fn(
    data_catalog,
    config,
    build_config,
    custom_model,
    working_directory,
    data_provider,
    data_root,
):
    """Update model."""
    with WorkingDirectory(working_directory):
        model = get_builder(
            config,
            data_catalog,
            custom_model,
            data_provider,
            data_root,
        )

        model.read()

        if isinstance(build_config, str):
            build_config_list: list[str] = build_config.split("::")
            build_config_file: str = build_config_list[0]

            try:
                methods: dict[Any] = parse_config(build_config_file)
            except FileNotFoundError:
                if ":" in build_config_file and "::" not in build_config_file:
                    raise FileNotFoundError(
                        f"Build config file '{build_config_file}' not found. Did you mean '{build_config_file.replace(':', '::')}'?"
                    )
                raise

            if len(build_config_list) > 1:
                assert len(build_config_list) == 2
                build_config_function: str = build_config_list[1]

                # Check if the method is specified with a trailing '+', and if so
                # we set a flag to keep all subsequent methods
                if build_config_function.endswith("+"):
                    build_config_function: str = build_config_function[:-1]
                    keep_subsequent_methods: bool = True
                else:
                    keep_subsequent_methods: bool = False

                if build_config_function not in methods:
                    raise KeyError(
                        f"Method '{build_config_function}' not found in build config file '{build_config_file}'. "
                        "Available methods: "
                        f"{', '.join(methods.keys())}"
                    )

                keys_to_remove: list[str] = []

                for key in methods.keys():
                    if key == build_config_function:
                        if keep_subsequent_methods:
                            # keep this method and all subsequent methods
                            break
                    else:
                        keys_to_remove.append(key)

                # remove all functions from the methods dict except the one we want to run
                for key in keys_to_remove:
                    del methods[key]

        elif isinstance(build_config, dict):
            methods = build_config

        else:
            raise ValueError

        model.update(methods=methods)


@cli.command()
@click_build_options(
    build_config="update.yml",
    build_config_help_extra="Optionally, you can specify a specific method within the update file using :: syntax, e.g., 'update.yml::setup_economic_data' to only run the setup_economic_data method. If the method ends with a '+', all subsequent methods are run as well.",
)
def update(*args, **kwargs):
    update_fn(*args, **kwargs)


@cli.command()
@click_run_options()
@click.option(
    "--methods", default=None, help="Comma-seperated list of methods to evaluate."
)
def evaluate(methods: list | None, *args, **kwargs) -> None:
    # If no methods are provided, pass None to run_model_with_method
    methods: list | None = None if not methods else methods.split(",")
    run_model_with_method(
        method="evaluate", method_args={"methods": methods}, *args, **kwargs
    )


def share_fn(working_directory, name, include_preprocessing, include_output):
    """Share model."""
    with WorkingDirectory(working_directory):
        # create a zip file called model.zip with the folders input, and model files
        # in the working directory
        folders: list = ["input"]
        if include_preprocessing:
            folders.append("preprocessing")
        if include_output:
            folders.append("output")
        files: list = ["model.yml", "build.yml"]
        optional_files: list = ["update.yml", "data_catalog.yml"]
        optional_folders: list = ["data"]
        zip_filename: str = f"{name}.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            total_files: int = (
                sum(
                    [
                        sum(len(files) for _, _, files in os.walk(folder))
                        for folder in folders
                    ]
                )
                + sum(
                    [
                        sum(len(files) for _, _, files in os.walk(folder))
                        for folder in optional_folders
                        if os.path.exists(folder)
                    ]
                )
                + len(files)
                + len(optional_files)
            )  # Count total number of files
            progress: int = 0  # Initialize progress counter
            for folder in folders:
                for root, _, filenames in os.walk(folder):
                    for filename in filenames:
                        zipf.write(os.path.join(root, filename))
                        progress += 1  # Increment progress counter
                        print(
                            f"Exporting file {progress}/{total_files} to {zip_filename}",
                            end="\r",
                        )  # Print progress
            for folder in optional_folders:
                if os.path.exists(folder):
                    for root, _, filenames in os.walk(folder):
                        for filename in filenames:
                            zipf.write(os.path.join(root, filename))
                            progress += 1
                            print(
                                f"Exporting file {progress}/{total_files} to {zip_filename}",
                                end="\r",
                            )
            for file in files:
                zipf.write(file)
                progress += 1  # Increment progress counter
                print(
                    f"Exporting file {progress}/{total_files} to {zip_filename}",
                    end="\r",
                )  # Print progress
            for file in optional_files:
                if os.path.exists(file):
                    zipf.write(file)
                progress += 1  # Increment progress counter
                print(
                    f"Exporting file {progress}/{total_files} to {zip_filename}",
                    end="\r",
                )  # Print progress
            print(f"Exporting file {progress}/{total_files} to {zip_filename}")
            print("Done!")


@cli.command()
@working_directory_option
@click.option(
    "--name",
    "-n",
    default="model",
    help="Name used for the zip file.",
)
@click.option(
    "--include-preprocessing",
    is_flag=True,
    default=False,
    help="Include preprocessing files in the zip file.",
)
@click.option(
    "--include-output",
    is_flag=True,
    default=False,
    help="Include output files in the zip file.",
)
def share(*args, **kwargs):
    """Share model as a zip file."""
    share_fn(*args, **kwargs)


if __name__ == "__main__":
    cli()
