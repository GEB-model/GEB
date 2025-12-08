"""Command line interface for GEB."""

import cProfile
import functools
import importlib
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
from typing import Any, Callable

import click
import geopandas as gpd
import yaml
from shapely.geometry import box

from geb import GEB_PACKAGE_DIR, __version__
from geb.build import GEBModel as GEBModelBuild
from geb.build.data_catalog import NewDataCatalog
from geb.build.methods import build_method
from geb.model import GEBModel
from geb.workflows.io import WorkingDirectory, read_dict, write_dict
from geb.workflows.methods import multi_level_merge

PROFILING_DEFAULT: bool = False
OPTIMIZE_DEFAULT: bool = False
TIMING_DEFAULT: bool = False
WORKING_DIRECTORY_DEFAULT: Path = Path(".")
CONFIG_DEFAULT: Path = Path("model.yml")
UPDATE_DEFAULT: Path = Path("update.yml")
BUILD_DEFAULT: Path = Path("build.yml")

DATA_CATALOG_DEFAULT: Path = GEB_PACKAGE_DIR / "data_catalog.yml"
DATA_PROVIDER_DEFAULT: str = os.environ.get("GEB_DATA_PROVIDER", "default")
DATA_ROOT_DEFAULT: Path = Path(
    os.environ.get(
        "GEB_DATA_ROOT",
        GEB_PACKAGE_DIR.parent.parent / "data_catalog",
    )
)
ALTER_FROM_MODEL_DEFAULT: Path = Path("../base")


class DetectDuplicateKeysYamlLoader(yaml.SafeLoader):
    """Custom YAML loader that detects duplicate keys in mappings.

    Raises:
        ValueError: If a duplicate key is found in the YAML mapping.
    """

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict:
        """Construct a mapping from a YAML node, checking for duplicate keys.

        Args:
            node: The YAML node to construct the mapping from.
            deep: Whether to perform a deep construction of the mapping. Defaults to False.

        Raises:
            ValueError: If a duplicate key is found in the YAML mapping.

        Returns:
            dict: The constructed mapping.
        """
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
    """Parse config.

    This method recursively parses the config file and resolves any 'inherits' keys.

    Args:
        config_path: Path to the config file or a dict with the config.
        current_directory: Current directory to resolve relative paths.
            If None, the current working directory is used.

    Returns:
        Full model configuation of the model without any remaining 'inherits' keys.
    """
    if current_directory is None:
        current_directory = Path.cwd()

    if isinstance(config_path, dict):
        config = config_path
    else:
        config: dict | None = yaml.load(
            open(current_directory / config_path, "r"),
            Loader=DetectDuplicateKeysYamlLoader,
        )
        if config is None:
            config = {}
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


def create_logger(fp: Path) -> logging.Logger:
    """Create logger with console and file handler.

    Args:
        fp: Path to the log file.
    Returns:
        Logger instance.
    """
    logger = logging.getLogger("GEB")
    # remove any previous handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
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
def cli(context: click.core.Context) -> None:
    """Command line interface for GEB.

    Args:
        context: Click context. (Auto-filled by click)
    """
    if context.obj is None:
        context.obj = {}


def click_config(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add config option to a click command.

    Useful to add the same option to multiple commands.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """

    @click.option(
        "--config",
        "-c",
        type=click.Path(path_type=Path),
        default=Path(CONFIG_DEFAULT),
        help=f"Path of the model configuration file. Defaults to '{CONFIG_DEFAULT}'.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function for config option.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The result of the wrapped function.
        """
        return func(*args, **kwargs)

    return wrapper


def working_directory_option(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add working directory option to a click command.

    Useful to add the same option to multiple commands.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """

    @click.option(
        "--working-directory",
        "-wd",
        type=click.Path(path_type=Path),
        default=Path(WORKING_DIRECTORY_DEFAULT),
        help="Working directory for model. Default is the current directory.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function for working directory option.

        Returns:
            The result of the wrapped function.
        """
        return func(*args, **kwargs)

    return wrapper


def click_run_options() -> Any:
    """Decorator to add run options to a click command.

    Useful to add the same options to multiple commands.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator function to add run options to a click command.

        Args:
            func: Function to decorate.

        Returns:
            Decorated function.
        """

        @click_config
        @working_directory_option
        @click.option(
            "--profiling",
            is_flag=True,
            default=PROFILING_DEFAULT,
            help="Run GEB with profiling. If this option is used a file `profiling_stats.cprof` is saved in the working directory.",
        )
        @click.option(
            "--optimize",
            is_flag=True,
            default=OPTIMIZE_DEFAULT,
            help="Run GEB in optimized mode, skipping asserts and water balance checks.",
        )
        @click.option(
            "--timing",
            is_flag=True,
            default=TIMING_DEFAULT,
            help="Run GEB with timing.",
        )
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper function for run options.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                The result of the wrapped function.
            """
            return func(*args, **kwargs)

        return wrapper

    return decorator


def run_model_with_method(
    method: str | None,
    config: dict | str | Path = CONFIG_DEFAULT,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    timing: bool = TIMING_DEFAULT,
    profiling: bool = PROFILING_DEFAULT,
    optimize: bool = OPTIMIZE_DEFAULT,
    method_args: dict = {},
    close_after_run: bool = True,
) -> GEBModel:
    """Run model with a specific method.

    Args:
        method: Method to run on the model. If None, the model is created but no method is run.
        config: Path to the model configuration file or a dict with the config.
        working_directory: Working directory for the model.
        timing: If True, run the model with timing, printing the time taken for specific methods.
        profiling: If True, run the model with profiling.
        optimize: If True, run the model in optimized mode, skipping asserts and water balance checks.
        method_args: Optional arguments to pass to the method.
        close_after_run: If True, close the model after running the method. Defaults to True.

    Returns:
        Instance of GEBModel

    Raises:
        SystemExit: If the model is restarted in optimized mode.
    """
    # check if we need to run the model in optimized mode
    # if the model is already running in optimized mode, we don't need to restart it
    # or else we start an infinite loop
    if optimize and sys.flags.optimize == 0:
        # If the script is not a .py file, we need to add the .exe extension
        if platform.system() == "Windows" and not sys.argv[0].endswith(".py"):
            sys.argv[0] = sys.argv[0] + ".exe"
        command: list[str] = [sys.executable, "-O"] + sys.argv
        raise SystemExit(subprocess.run(command).returncode)

    with WorkingDirectory(working_directory):
        config: dict[str, Any] = parse_config(config)

        # TODO: This can be removed in 2026
        if not Path("input/files.yml").exists() and Path("input/files.json").exists():
            # convert input/files.json to input/files.yml
            json_files: dict[str, Any] = read_dict(
                Path("input/files.json"),
            )
            write_dict(json_files, Path("input/files.yml"))

        files: dict[str, Any] = parse_config(
            read_dict(Path("input/files.yml"))
            if "files" not in config["general"]
            else config["general"]["files"]
        )

        if profiling:
            profile = cProfile.Profile()
            profile.enable()

        geb = GEBModel(config=config, files=files, timing=timing)
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


@cli.command()
@click_run_options()
def run(**kwargs: Any) -> None:
    """Run model.

    Can be run after model spinup.

    Args:
        **kwargs: Keyword arguments to pass to the run function.

    """
    run_model_with_method(method="run", **kwargs)


@cli.command()
@click_run_options()
def spinup(**kwargs: Any) -> None:
    """Run model spinup.

    Can be run after model build.

    Args:
        **kwargs: Keyword arguments to pass to the spinup function.

    """
    run_model_with_method(method="spinup", **kwargs)


@cli.command()
@click.argument("method", required=True)
@click_run_options()
def exec(method: str, **kwargs: Any) -> None:
    """Execute a specific method on the model.

    Args:
        method: Method to run on the model.
        **kwargs: Keyword arguments to pass to the method.
    """
    run_model_with_method(method=method, **kwargs)


def click_build_options(
    build_config: Path = BUILD_DEFAULT, build_config_help_extra: str | None = None
) -> Any:
    """Decorator to add build options to a click command.

    Args:
        build_config: Default path to the build config file.
        build_config_help_extra: Extra help text for the build config option.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator function to add build options to a click command.

        Args:
            func: Function to decorate.

        Returns:
            Decorated function.
        """
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
            type=click.Path(path_type=Path),
            default=Path(build_config),
            help=build_config_help,
        )
        @working_directory_option
        @click.option(
            "--data-catalog",
            "-d",
            type=click.Path(path_type=Path),
            default=Path(DATA_CATALOG_DEFAULT),
            help=f"""Path to data catalog YAML files. By default the data_catalog in the examples is used. If this is not set, defaults to {DATA_CATALOG_DEFAULT}""",
        )
        @click.option(
            "--data-provider",
            "-p",
            type=str,
            default=DATA_PROVIDER_DEFAULT,
            help="Data variant to use from data catalog (see hydroMT documentation).",
        )
        @click.option(
            "--data-root",
            "-r",
            type=click.Path(path_type=Path),
            default=Path(DATA_ROOT_DEFAULT),
            help="Root folder where the data is located. When the environment variable GEB_DATA_ROOT is set, this is used as the root folder for the data catalog. If not set, defaults to the data_catalog folder in parent of the GEB source code directory.",
        )
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper function for build options.

            Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

            Returns:
            The result of the wrapped function.
            """
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_model_builder_class(custom_model: None | str) -> type:
    """Get model builder class.

    This is usually the GEBModelBuild class, but can be a custom model builder class
    from the geb.build.custom_models module. This class would usually
    specify some custom build methods, but largely re-use the existing GEBModelBuild methods.

    Args:
        custom_model: Name of the custom model to use. If None, the default GEBModelBuild is used.
            custom_models are available in the geb.build.custom_models module.

    Returns:
        Model builder class.
    """
    if custom_model is None:
        return GEBModelBuild
    else:
        from geb import build as geb_build

        importlib.import_module(
            "." + custom_model.split(".")[0], package="geb.build.custom_models"
        )
        return attrgetter(custom_model)(geb_build.custom_models)


def customize_data_catalog(data_catalog: Path, data_root: None | Path = None) -> Path:
    """This functions adds the GEB_DATA_ROOT to the data catalog if it is set as an environment variable.

    This enables reading the data catalog from a different location than the location of the yml-file
    without the need to specify root in the meta of the data catalog.

    Args:
        data_catalog: List of paths to data catalog yml files.
        data_root: Root folder where the data is located. If None, the data catalog is not modified.

    Returns:
        List of paths to data catalog yml files, possibly modified to include the data_root.
    """
    if data_root:
        with open(data_catalog, "r") as stream:
            data_catalog_yml = yaml.load(stream, Loader=yaml.FullLoader)

            if "meta" not in data_catalog_yml:
                data_catalog_yml["meta"] = {}
            data_catalog_yml["meta"]["root"] = str(data_root)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml") as tmp:
            yaml.dump(data_catalog_yml, tmp, default_flow_style=False)
        return Path(tmp.name)
    else:
        return data_catalog


def get_builder(
    config: Path | dict[str, Any],
    data_catalog: Path,
    custom_model: str | None,
    data_provider: str | None,
    data_root: Path | None,
) -> GEBModelBuild:
    """Get model builder.

    Args:
        config: Path to the model configuration file.
        data_catalog: Path to the data catalog file.
        custom_model: Name of the custom model to use. If None, the default GEBModelBuild is used.
            custom_models are available in the geb.build.custom_models module.
        data_provider: Data variant to use from data catalog (see hydroMT documentation).
        data_root: Root folder where the data is located. If None, the data catalog is not modified.

    Returns:
        Instance of the model builder.
    """
    config = parse_config(config)
    input_folder = Path(config["general"]["input_folder"])

    data_catalog = customize_data_catalog(data_catalog, data_root=data_root)

    arguments = {
        "root": input_folder,
        "data_catalog": data_catalog,
        "logger": create_logger(Path("build.log")),
        "data_provider": data_provider,
    }

    builder_class = get_model_builder_class(custom_model)(**arguments)
    build_method.validate_tree()
    return builder_class


def init_fn(
    config: str | Path,
    build_config: str | Path,
    update_config: str | Path,
    working_directory: str | Path,
    from_example: str,
    basin_id: str | None = None,
    ISO3: str | None = None,
    overwrite: bool = False,
) -> None:
    """Create a new model.

    Args:
        config: Path to the model configuration file to create.
        build_config: Path to the model build configuration file to create.
        update_config: Path to the model update configuration file to create.
        working_directory: Working directory for the model.
        from_example: Name of the example to use as a base for the model.
        basin_id: Basin ID(s) to use for the model. Can be a comma-separated list of integers.
            If not set, the basin ID is taken from the config file.
        ISO3: ISO3 country code to use for the model. Cannot be used together with --basin-id.
        overwrite: If True, overwrite existing config and build config files. Defaults to False.

    Raises:
        FileExistsError: If the config or build config file already exists and overwrite is False.
        FileNotFoundError: If the example folder does not exist.
        ValueError: If both basin_id and ISO3 are set.

    """
    if basin_id is not None and ISO3 is not None:
        raise ValueError("Cannot use --basin-id and --ISO3 together.")

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

        example_folder: Path = GEB_PACKAGE_DIR / "examples" / from_example
        if not example_folder.exists():
            raise FileNotFoundError(
                f"Example folder {example_folder} does not exist. Did you use the right --from-example option?"
            )

        config_dict: dict = yaml.load(
            open(example_folder / CONFIG_DEFAULT, "r"),
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
        elif ISO3 is not None:
            del config_dict["general"]["region"]["subbasin"]
            config_dict["general"]["region"] = {
                "geom": {
                    "source": "GADM_level1",
                    "key": ISO3,
                    "column": "GID_0",
                }
            }

        with open(config, "w") as f:
            # do not sort keys, to keep the order of the config file
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        shutil.copy(example_folder / BUILD_DEFAULT, build_config)
        shutil.copy(example_folder / UPDATE_DEFAULT, update_config)


@cli.command()
@click_config
@click.option(
    "--build-config",
    "-b",
    type=click.Path(path_type=Path),
    default=Path(BUILD_DEFAULT),
    help=f"Path of the model build configuration file. Defaults to '{BUILD_DEFAULT}'.",
)
@click.option(
    "--update-config",
    "-u",
    type=click.Path(path_type=Path),
    default=Path(UPDATE_DEFAULT),
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
    help="Basin ID(s) to use for the model. Comma-separated list of integers. If not set, the basin ID is taken from the config file. Cannot be used together with --ISO3.",
)
@click.option(
    "--ISO3",
    "ISO3",
    default=None,
    type=str,
    help="ISO3 country code to use for the model. Cannot be used together with --basin-id.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="If set, overwrite existing config and build config files.",
)
@working_directory_option
def init(*args: Any, **kwargs: Any) -> None:
    """Initialize a new model."""
    # Initialize the model with the given config and build config
    init_fn(*args, **kwargs)


def set_fn(
    config: Path,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    **kwargs: Any,
) -> None:
    """Set model configuration values by updating a YAML configuration file.

    This function loads the existing configuration from the specified file,
    updates it with the provided keyword arguments (supporting nested keys
    using dot notation, e.g., 'section.subsection.key'), and saves the
    modified configuration back to the file.

        config: Path to the model configuration file (as a string or Path object).
        working_directory: Working directory for the model.
        **kwargs: Keyword arguments representing keys and values to set in the config.
                  Keys can be nested using dots (e.g., 'model.lr' sets 'lr' under 'model').

    Note:
        If a nested key does not exist, intermediate dictionaries are created automatically.
        The file is overwritten with the updated configuration in YAML format.

    Args:
        config: Path to the model configuration file.
        working_directory: Working directory for the model.
        **kwargs: Keyword arguments to set in the config file.

    Raises:
        KeyError: If a specified key does not exist in the config and cannot be created.
    """
    with WorkingDirectory(working_directory):
        config_dict: dict[str, Any] = parse_config(config)
        for key, value in kwargs.items():
            if key.endswith("+"):
                key: str = key[:-1]
                create: bool = True
            else:
                create: bool = False

            keys: list[str] = key.split(".")
            d = config_dict

            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    if create:
                        d[k] = {}
                    else:
                        raise KeyError(
                            f"Key '{k}' not found in config. If you want to create it, use the '+' suffix for the KEY."
                        )
                d: dict[str, Any] = d[k]

            if value == "null":
                value = None
            elif value == "true":
                value = True
            elif value == "false":
                value = False

            if not create and keys[-1] not in d:
                raise KeyError(
                    f"Key '{keys[-1]}' not found in config. If you want to create it, use the '+' suffix for the KEY."
                )
            d[keys[-1]] = value

        with open(config, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click_config
@working_directory_option
@click.pass_context
def set(ctx: click.Context, config: Path, working_directory: Path) -> None:
    """Set model configuration values.

    Accepts parameter assignments in the form key=value, where keys can use
    dot notation for nested values (e.g., model.param1=0.5).

    By default, only existing keys can be updated. To create new keys,
    append a '+' to the key (e.g., model.new_param+=10).

    Args:
        ctx: Click context containing extra arguments.
        config: Path to the model configuration file.
        working_directory: Working directory for the model.

    """
    # Parse extra arguments as key=value pairs
    params = {}
    for arg in ctx.args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to convert value to appropriate type
            try:
                # Try int first
                value = int(value)
            except ValueError:
                try:
                    # Try float
                    value = float(value)
                except ValueError:
                    # Keep as string
                    pass
            params[key] = value
        else:
            click.echo(
                f"Warning: Ignoring invalid argument '{arg}'. Expected format: key=value",
                err=True,
            )

    set_fn(config=config, working_directory=working_directory, **params)


def build_fn(
    data_catalog: Path = DATA_CATALOG_DEFAULT,
    config: Path | dict[str, Any] = CONFIG_DEFAULT,
    build_config: Path | dict[str, Any] = BUILD_DEFAULT,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    data_provider: str = DATA_PROVIDER_DEFAULT,
    data_root: Path = DATA_ROOT_DEFAULT,
    continue_: bool = False,
) -> None:
    """Build model.

    Args:
        data_catalog: Path to the data catalog file.
        config: Path to the model configuration file.
        build_config: Path to the model build configuration file.
        working_directory: Working directory for the model.
        data_provider: Data variant to use from data catalog (see hydroMT documentation).
        data_root: Root folder where the data is located. If None, the data catalog is not modified.
        continue_: Continue previous build if it was interrupted or failed.
    """
    with WorkingDirectory(working_directory):
        build_config = parse_config(build_config)
        model = get_builder(
            config,
            data_catalog,
            build_config["_custom_model"] if "_custom_model" in build_config else None,
            data_provider,
            data_root,
        )
        methods = {
            method: args
            for method, args in build_config.items()
            if not method.startswith("_")
        }
        model.build(
            methods=methods,
            region=parse_config(config)["general"]["region"],
            continue_=continue_,
        )


@cli.command()
@click_build_options()
@click.option(
    "--continue",
    "-c",
    "continue_",
    is_flag=True,
    default=False,
    help="Continue previous build if it was interrupted or failed.",
)
def build(*args: Any, **kwargs: Any) -> None:
    """Build model with configuration file.

    This command reads the model configuration file and the build configuration file
    and executes the build methods specified in the build configuration file.

    Args:
        *args: Positional arguments to pass to the build function.
        **kwargs: Keyword arguments to pass to the build function.
    """
    build_fn(*args, **kwargs)


def alter_fn(
    data_catalog: Path = DATA_CATALOG_DEFAULT,
    config: Path = CONFIG_DEFAULT,
    build_config: Path | dict[str, Any] = BUILD_DEFAULT,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    from_model: Path = ALTER_FROM_MODEL_DEFAULT,
    data_provider: str = DATA_PROVIDER_DEFAULT,
    data_root: Path = DATA_ROOT_DEFAULT,
) -> None:
    """Create alternative version from base model with only changed files.

    This function is useful to create a new model based on an existing one, but with
    only a few changes. It will copy the base model and overwrite the files that are
    specified in the config and build config files. The rest of the files will be
    linked to the original model to reduce disk space.

    Args:
        data_catalog: Path to the data catalog file.
        config: Path to the model configuration file.
        build_config: Path to the model build configuration file.
        working_directory: Working directory for the model.
        from_model: Folder for the existing model.
        data_provider: Data variant to use from data catalog (see hydroMT documentation).
        data_root: Root folder where the data is located. If None, the data catalog is not modified.
    """
    from_model: Path = Path(from_model)

    with WorkingDirectory(working_directory):
        original_config: Path = from_model / config

        # if config does not exist, create a new config that inherits from the original model
        if not config.exists():
            original_config: Path = from_model / config

            config_dict: dict[str, str] = {"inherits": str(original_config)}
            with open(config, "w") as f:
                # do not sort keys, to keep the order of the config file
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # if config exists, we just make sure it inherits from the original model
        # if this is set already, we just leave it as is and assume the user knows what they are doing
        else:
            # Read existing config
            with open(config, "r") as f:
                raw_config = yaml.load(f, Loader=DetectDuplicateKeysYamlLoader)
            if "inherits" not in raw_config:
                raw_config["inherits"] = str(from_model / config)
            with open(config, "w") as f:
                yaml.dump(raw_config, f, default_flow_style=False, sort_keys=False)

        config_from_original_model = parse_config(from_model / config)
        input_folder: Path = Path(config_from_original_model["general"]["input_folder"])

        original_input_path: Path = from_model / input_folder

        # TODO: This can be removed in 2026
        if (
            not (original_input_path / "files.yml").exists()
            and (original_input_path / "files.json").exists()
        ):
            # convert input/files.json to input/files.yml
            json_files: dict[str, Any] = read_dict(
                (original_input_path / "files.json"),
            )
            write_dict(json_files, original_input_path / "files.yml")
            # remove the original json file
            (original_input_path / "files.json").unlink()

        original_files = read_dict(original_input_path / "files.yml")

        for file_class, files in original_files.items():
            for file_name, file_path in files.items():
                if not file_path.startswith("/"):
                    original_files[file_class][file_name] = str(
                        Path("..") / original_input_path / file_path
                    )

        input_folder.mkdir(parents=True, exist_ok=True)
        with open(input_folder / "files.yml", "w") as f:
            yaml.dump(original_files, f, default_flow_style=False)

        build_config = parse_config(build_config)
        model = get_builder(
            config,
            data_catalog,
            build_config["_custom_model"] if "_custom_model" in build_config else None,
            data_provider,
            data_root,
        )
        methods = {
            method: args
            for method, args in build_config.items()
            if not method.startswith("_")
        }

        model.update(
            methods=methods,
        )


@cli.command()
@click_build_options()
@click.option(
    "--from-model",
    type=click.Path(path_type=Path),
    default=ALTER_FROM_MODEL_DEFAULT,
    help="Folder for the existing model.",
)
def alter(*args: Any, **kwargs: Any) -> None:
    """Create alternative version from base model with only changed files.

    This command is useful to create a new model based on an existing one, but with
    only a few changes. It will copy the base model and overwrite the files that are
    specified in the config and build config files. The rest of the files will be
    linked to the original model to reduce disk space.
    """
    alter_fn(*args, **kwargs)


def update_fn(
    data_catalog: Path = DATA_CATALOG_DEFAULT,
    config: Path | dict[str, Any] = CONFIG_DEFAULT,
    build_config: Path = BUILD_DEFAULT,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    data_provider: str = DATA_PROVIDER_DEFAULT,
    data_root: Path = DATA_ROOT_DEFAULT,
) -> None:
    """Update model.

    Args:
        data_catalog: Path to the data catalog file.
        config: Path to the model configuration file.
        build_config: Path to the model build configuration file or a specific method within the build file using :: syntax, e.g., 'build.yml::setup_economic_data' to only run the setup_economic_data method. If the method ends with a '+', all subsequent methods are run as well.
        working_directory: Working directory for the model.
        data_provider: Data variant to use from data catalog (see hydroMT documentation).
        data_root: Root folder where the data is located. If None, the data catalog is not modified.

    Raises:
        FileNotFoundError: if the build config file is not found.
        KeyError: if the specified method is not found in the build config file.
        ValueError: if build_config is not a str or dict.
    """
    with WorkingDirectory(working_directory):
        if isinstance(build_config, Path):
            build_config_list: list[str] = str(build_config).split("::")
            build_config_file: Path = Path(build_config_list[0])

            try:
                build_config: dict[str, Any] = parse_config(build_config_file)
            except FileNotFoundError:
                if ":" in str(build_config_file) and "::" not in str(build_config_file):
                    raise FileNotFoundError(
                        f"Build config file '{str(build_config_file)}' not found. Did you mean '{str(build_config_file).replace(':', '::')}'?"
                    )
                raise

            methods = {
                method: args
                for method, args in build_config.items()
                if not method.startswith("_")
            }

            if len(build_config_list) > 1:
                assert len(build_config_list) == 2
                build_config_function: str = build_config_list[1]

                # Check if the method is specified with a trailing '+' or '#'.
                # If + we set a flag to keep all subsequent methods
                # If # we set a flag to keep all dependent methods
                if build_config_function.endswith("+"):
                    build_config_function: str = build_config_function[:-1]
                    keys_to_remove: list[str] = []

                    for key in methods.keys():
                        if key == build_config_function:
                            break
                        else:
                            keys_to_remove.append(key)
                    else:
                        raise KeyError(
                            f"Method '{build_config_function}' not found in build config file '{build_config_file}'. "
                            "Available methods: "
                            f"{', '.join(methods.keys())}"
                        )

                    # remove all functions from the methods dict except the one we want to run
                    for key in keys_to_remove:
                        del methods[key]

                elif build_config_function.endswith("#"):
                    build_config_function: str = build_config_function[:-1]
                    dependents: list[str] = build_method.get_dependents(
                        build_config_function
                    )
                    dependents.append(build_config_function)
                    methods = {
                        dependent: methods[dependent]
                        for dependent in dependents
                        if dependent in methods
                    }

                else:
                    methods = {build_config_function: methods[build_config_function]}

        elif isinstance(build_config, dict):
            methods = {
                method: args
                for method, args in build_config.items()
                if not method.startswith("_")
            }

        else:
            raise ValueError("build_config must be a str or dict.")

        model = get_builder(
            config,
            data_catalog,
            build_config["_custom_model"] if "_custom_model" in build_config else None,
            data_provider,
            data_root,
        )

        model.update(methods=methods)


@cli.command()
@click_build_options(
    build_config=UPDATE_DEFAULT,
    build_config_help_extra="Optionally, you can specify a specific method within the update file using :: syntax, e.g., 'update.yml::setup_economic_data' to only run the setup_economic_data method. If the method ends with a '+', all subsequent methods are run as well.",
)
def update(*args: Any, **kwargs: Any) -> None:
    """Update model with configuration file.

    Args:
        *args: Positional arguments to pass to the update function.
        **kwargs: Keyword arguments to pass to the update function.
    """
    update_fn(*args, **kwargs)


@cli.command()
@click_run_options()
@click.option(
    "--methods",
    default="plot_discharge,evaluate_discharge,evaluate_hydrodynamics,evaluate_forecasts",
    help="Comma-seperated list of methods to evaluate. Currently supported methods: 'water-circle', 'evaluate-discharge' and 'plot-discharge'. Default is 'plot_discharge,evaluate_discharge'.",
)
@click.option("--spinup-name", default="spinup", help="Name of the evaluation run.")
@click.option("--run-name", default="default", help="Name of the run to evaluate.")
@click.option(
    "--include-spinup",
    is_flag=True,
    default=False,
    help="Include spinup in evaluation.",
)
@click.option(
    "--include-yearly-plots",
    is_flag=True,
    default=False,
    help="Create yearly plots in evaluation.",
)
@click.option(
    "--correct-q-obs",
    is_flag=True,
    default=False,
    help="correct_Q_obs can be flagged to correct the Q_obs discharge timeseries for the difference in upstream area between the Q_obs station and the simulated discharge",
)
def evaluate(
    methods: str,
    spinup_name: str,
    run_name: str,
    include_spinup: bool,
    include_yearly_plots: bool,
    correct_q_obs: bool,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    config: dict[str, Any] | Path = CONFIG_DEFAULT,
    profiling: bool = PROFILING_DEFAULT,
    optimize: bool = OPTIMIZE_DEFAULT,
    timing: bool = TIMING_DEFAULT,
) -> None:
    """Evaluate model, for example by comparing observed and simulated discharge.

    Args:
        methods: Comma-seperated list of methods to evaluate. Currently supported methods: '
            'water-circle', 'evaluate-discharge' and 'plot-discharge'. Default is 'plot_discharge,evaluate_discharge'.
        spinup_name: Name of the evaluation run.
        run_name: Name of the run to evaluate.
        include_spinup: Include spinup in evaluation.
        include_yearly_plots: Create yearly plots in evaluation.
        correct_q_obs: correct_Q_obs can be flagged to correct the Q_obs discharge timeseries
            for the difference in upstream area between the Q_obs station and the simulated discharge.
        working_directory: Working directory for the model.
        config: Path to the model configuration file or a dict with the config.
        profiling: If True, run the model with profiling.
        optimize: If True, run the model in optimized mode, skipping asserts and water balance checks.
        timing: If True, run the model with timing, printing the time taken for specific methods
    """
    # If no methods are provided, pass None to run_model_with_method
    methods_list: list[str] = methods.split(",")
    methods_list: list[str] = [
        method.replace("-", "_").strip() for method in methods_list
    ]
    run_model_with_method(
        method="evaluate",
        method_args={
            "methods": methods_list,
            "spinup_name": spinup_name,
            "run_name": run_name,
            "include_spinup": include_spinup,
            "include_yearly_plots": include_yearly_plots,
            "correct_Q_obs": correct_q_obs,
        },
        working_directory=working_directory,
        config=config,
        profiling=profiling,
        optimize=optimize,
        timing=timing,
    )


def share_fn(
    working_directory: Path,
    name: str,
    include_cache: bool,
    include_output: bool,
) -> None:
    """Share model."""
    with WorkingDirectory(working_directory):
        # create a zip file called model.zip with the folders input, and model files
        # in the working directory
        folders: list = ["input"]
        if include_cache:
            folders.append("cache")
        if include_output:
            folders.append("output")
        files: list = [CONFIG_DEFAULT, BUILD_DEFAULT]
        optional_files: list = [UPDATE_DEFAULT, DATA_CATALOG_DEFAULT]
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
    "--include-cache",
    is_flag=True,
    default=False,
    help="Include cache files in the zip file.",
)
@click.option(
    "--include-output",
    is_flag=True,
    default=False,
    help="Include output files in the zip file.",
)
def share(*args: Any, **kwargs: Any) -> None:
    """Share model as a zip file."""
    share_fn(*args, **kwargs)


@cli.command()
@click.argument(
    "method",
    required=True,
    type=click.Choice(["size", "license", "fetch"], case_sensitive=True),
)
def data_catalog(method: str) -> None:
    """Method to interact directly with the data catalog.

    Raises:
        ValueError: If the method is not recognized.
    """
    data_catalog = NewDataCatalog()
    if method == "size":
        print("Total size of data catalog:", data_catalog.size())
    elif method == "license":
        data_catalog.print_licenses()
    elif method == "fetch":
        data_catalog.fetch_global()
    else:
        raise ValueError(f"Unknown method '{method}'.")


@cli.command(
    context_settings=dict(ignore_unknown_options=True, allow_interspersed_args=False)
)
@click.argument(
    "workflow_name",
    required=True,
    type=click.Choice(
        ["calibrate", "sensitivity", "multirun", "benchmark"], case_sensitive=True
    ),
)
@click.option(
    "--cores",
    "-c",
    default="all",
    help="Number of cores to use. Default is 'all'.",
)
@click.option(
    "--profile",
    "-p",
    type=click.Path(path_type=Path),
    default=None,
    help="Snakemake profile to use. If not specified, uses profiles/{workflow_name}.",
)
@click.option(
    "--dryrun",
    "-n",
    is_flag=True,
    default=False,
    help="Perform a dry run without executing jobs.",
)
@click.option(
    "--config-override",
    "-co",
    multiple=True,
    help="Override config values (e.g., -co REGION=geul -co NGEN=10).",
)
@working_directory_option
@click.argument("snakemake_args", nargs=-1, type=click.UNPROCESSED)
def workflow(
    workflow_name: str,
    cores: str,
    profile: Path | None,
    dryrun: bool,
    config_override: tuple[str, ...],
    working_directory: Path,
    snakemake_args: tuple[str, ...],
) -> None:
    """Run a Snakemake workflow for GEB.

    Available workflows:
    - calibrate: Evolutionary algorithm calibration
    - sensitivity: Sensitivity analysis
    - multirun: Multiple scenario runs

    Examples:
        geb workflow calibrate --cores 8
        geb workflow calibrate --profile profiles/cluster
        geb workflow calibrate -co REGION=geul -co NGEN=10
        geb workflow sensitivity --dryrun

    Args:
        workflow_name: Name of the workflow to run.
        cores: Number of cores to use.
        profile: Snakemake profile directory.
        dryrun: Whether to perform a dry run.
        config_override: Config values to override.
        working_directory: Working directory for the workflow.
        snakemake_args: Additional arguments to pass to snakemake.
    """
    # Get GEB package directory for workflow files
    geb_dir: Path = GEB_PACKAGE_DIR.parent

    with WorkingDirectory(working_directory):
        # Build snakemake command
        cmd: list[str] = ["snakemake", "--directory", str(working_directory)]

        # Determine and add snakefile and configfile from GEB package
        snakefile = geb_dir / "workflow" / f"Snakefile_{workflow_name}"
        if not snakefile.exists():
            click.echo(f"Error: Workflow file {snakefile} not found.", err=True)
            sys.exit(1)
        cmd.extend(["-s", str(snakefile)])

        # Add workflow config file
        configfile = geb_dir / "workflow" / "config" / f"{workflow_name}.yml"
        if configfile.exists():
            cmd.extend(["--configfile", str(configfile)])

        # Add profile if specified and exists
        if profile is not None:
            if not profile.is_absolute():
                # Try relative to GEB package directory first
                profile_candidate = geb_dir / profile
                if profile_candidate.exists():
                    profile = profile_candidate
            if Path(profile).exists():
                cmd.extend(["--profile", str(profile)])
        else:
            # No profile specified, use default settings
            cmd.extend(["--cores", cores])

        # Add dry run flag
        if dryrun:
            cmd.append("-n")

        # Process config overrides
        config_overrides_dict = {}
        if config_override:
            for override in config_override:
                if "=" in override:
                    key, value = override.split("=", 1)
                    config_overrides_dict[key] = value
                else:
                    click.echo(
                        f"Warning: Invalid config override '{override}'. Expected format: KEY=VALUE",
                        err=True,
                    )

        if config_overrides_dict:
            cmd.append("--config")
            for key, value in config_overrides_dict.items():
                cmd.append(f"{key}={value}")

        # Add additional snakemake arguments
        cmd.extend(snakemake_args)

        # Print command
        click.echo(f"Running: {' '.join(cmd)}")

        # Execute snakemake in the working directory
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


def init_multiple_fn(
    config: str | Path,
    build_config: str | Path,
    update_config: str | Path,
    working_directory: str | Path,
    from_example: str,
    geometry_bounds: str,
    target_area_km2: float,
    area_tolerance: float,
    cluster_prefix: str,
    overwrite: bool,
    save_geoparquet: Path | None,
    save_map: str | Path | None,
) -> None:
    """Create multiple models from a geometry by clustering downstream subbasins.

    Args:
        config: Path to the base model configuration file.
        build_config: Path to the base model build configuration file.
        update_config: Path to the base model update configuration file.
        working_directory: Working directory for the models.
        from_example: Name of the example to use as a base for the models.
        geometry_bounds: Bounding box as "xmin,ymin,xmax,ymax" to select subbasins.
        target_area_km2: Target cumulative upstream area per cluster (default: Danube basin ~817,000 km2).
        area_tolerance: Tolerance for target area (0.3 = 30% tolerance).
        cluster_prefix: Prefix for cluster directory names.
        overwrite: If True, overwrite existing directories and files.
        save_geoparquet: Path to save clusters as geoparquet file. If None, no file is saved.
        save_map: Path to save visualization map as PNG file. If None, no map is created.

    Raises:
        FileExistsError: If directories already exist and overwrite is False.
        FileNotFoundError: If the example folder does not exist.
        ValueError: If geometry_bounds format is invalid.
    """
    from geb.build import (
        cluster_subbasins_by_area_and_proximity,
        create_cluster_visualization_map,
        create_multi_basin_configs,
        get_all_downstream_subbasins_in_geom,
        get_river_graph,
        save_clusters_as_merged_geometries,
        save_clusters_to_geoparquet,
    )
    from geb.build.data_catalog import NewDataCatalog

    config: Path = Path(config)
    build_config: Path = Path(build_config)
    update_config: Path = Path(update_config)
    working_directory: Path = Path(working_directory)

    # Create the models/large_scale directory structure
    models_dir = Path.cwd().parent / "models"
    large_scale_dir = models_dir / "large_scale"
    if not large_scale_dir.exists():
        large_scale_dir.mkdir(parents=True, exist_ok=True)

    # Always create geoparquet and map files in large_scale directory if not specified
    if save_geoparquet is None:
        save_geoparquet = large_scale_dir / f"{cluster_prefix}_clusters.geoparquet"
    if save_map is None:
        save_map = large_scale_dir / f"{cluster_prefix}_clusters_map.png"

    # Parse geometry bounds
    try:
        bounds = [float(x.strip()) for x in geometry_bounds.split(",")]
        if len(bounds) != 4:
            raise ValueError
        xmin, ymin, xmax, ymax = bounds
    except ValueError:
        raise ValueError(
            "geometry_bounds must be in format 'xmin,ymin,xmax,ymax' with numeric values"
        )

    bbox_geom = gpd.GeoDataFrame(
        geometry=[box(xmin, ymin, xmax, ymax)], crs="EPSG:4326"
    )

    # Initialize data catalog and logger
    data_catalog_instance = NewDataCatalog()
    logger = create_logger(working_directory / "init_multiple.log")

    logger.info("Starting multiple model initialization")
    logger.info(f"Using geometry bounds: {geometry_bounds}")
    logger.info(f"Target area: {target_area_km2:,.0f} km²")
    logger.info(f"Area tolerance: {area_tolerance:.1%}")

    logger.info("Loading river network...")
    river_graph = get_river_graph(data_catalog_instance)

    logger.info("Finding downstream subbasins in geometry...")
    downstream_subbasins = get_all_downstream_subbasins_in_geom(
        data_catalog_instance, bbox_geom, logger
    )

    if not downstream_subbasins:
        raise ValueError("No downstream subbasins found in the specified geometry")

    logger.info(f"Found {len(downstream_subbasins)} downstream subbasins")

    logger.info("Clustering subbasins by area and proximity...")
    clusters = cluster_subbasins_by_area_and_proximity(
        data_catalog_instance,
        downstream_subbasins,
        target_area_km2=target_area_km2,
        area_tolerance=area_tolerance,
        logger=logger,
    )

    logger.info(f"Created {len(clusters)} clusters")

    # Check for existing directories if not overwriting
    if not overwrite:
        for i in range(len(clusters)):
            cluster_dir = large_scale_dir / f"{cluster_prefix}_{i:03d}"
            if cluster_dir.exists():
                raise FileExistsError(
                    f"Cluster directory {cluster_dir} already exists. Remove --no-overwrite flag to overwrite."
                )

    # Verify example folder exists
    example_folder: Path = GEB_PACKAGE_DIR / "examples" / from_example
    if not example_folder.exists():
        raise FileNotFoundError(
            f"Example folder {example_folder} does not exist. Did you use the right --from-example option?"
        )

    logger.info(f"Creating cluster configurations using example: {from_example}")
    # Create cluster configurations
    cluster_directories = create_multi_basin_configs(
        clusters=clusters,
        working_directory=large_scale_dir,
        cluster_prefix=cluster_prefix,
    )

    logger.info(f"Saving clusters to geoparquet: {save_geoparquet}")
    # Save clusters to geoparquet (always create)
    save_clusters_to_geoparquet(
        clusters=clusters,
        data_catalog=data_catalog_instance,
        output_path=save_geoparquet,
        cluster_prefix=cluster_prefix,
    )

    # Save clusters as merged geometries (complete basins as single polygons)
    merged_basins_path = (
        save_geoparquet.parent / f"complete_basins_{save_geoparquet.stem}.geoparquet"
    )
    logger.info(f"Saving complete basins as merged geometries: {merged_basins_path}")
    save_clusters_as_merged_geometries(
        clusters=clusters,
        data_catalog=data_catalog_instance,
        river_graph=river_graph,
        output_path=merged_basins_path,
        cluster_prefix=cluster_prefix,
        include_upstream=True,  # Include all upstream subbasins in merged geometry
    )

    logger.info(f"Creating visualization map: {save_map}")
    # Create visualization map (always create)
    create_cluster_visualization_map(
        clusters=clusters,
        data_catalog=data_catalog_instance,
        output_path=save_map,
        cluster_prefix=cluster_prefix,
    )

    logger.info(
        f"Successfully created {len(cluster_directories)} model configurations:"
    )
    for cluster_dir in cluster_directories:
        logger.info(f"  {cluster_dir.relative_to(large_scale_dir)}")

    logger.info("To build all models, run:")
    logger.info(f"  cd {large_scale_dir}")
    logger.info(f"  for dir in {cluster_prefix}_*/; do")
    logger.info(f"    echo 'Building model in $dir'")
    logger.info(f"    cd $dir && geb build && cd ..")
    logger.info(f"  done")

    logger.info("Multiple model initialization completed successfully")


@cli.command()
@click_config
@click.option(
    "--build-config",
    "-b",
    default=BUILD_DEFAULT,
    help=f"Path of the model build configuration file. Defaults to '{BUILD_DEFAULT}'.",
)
@click.option(
    "--update-config",
    "-u",
    default=UPDATE_DEFAULT,
    help="Path of the model update configuration file.",
)
@click.option(
    "--from-example",
    default="geul",
    help="Name of the example to use as a base for the models. Defaults to 'geul'.",
)
@click.option(
    "--geometry-bounds",
    default="-10.0,35.0,40.0,70.0",  # World: "-180.0,-90.0,180.0,90.0" Western Europe: "-10.0,35.0,20.0,70.0" Europe: "-10.0,35.0,40.0,70.0"
    required=True,
    type=str,
    help="Bounding box as 'xmin,ymin,xmax,ymax' to select subbasins (e.g., '5.0,50.0,15.0,55.0' for parts of Europe). Defaults to Europe coverage.",
)
@click.option(
    "--target-area-km2",
    default=34000.0,
    type=float,
    help="Target cumulative upstream area per cluster in km². Defaults to 34,000 km².",
)
@click.option(
    "--area-tolerance",
    default=0.3,
    type=float,
    help="Tolerance for target area as fraction (0.3 = 30% tolerance).",
)
@click.option(
    "--cluster-prefix",
    default="cluster",
    help="Prefix for cluster directory names. Defaults to 'cluster'.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=True,
    help="If set, overwrite existing cluster directories and files.",
)
@click.option(
    "--save-geoparquet",
    type=click.Path(),
    help="Save clusters to geoparquet file at this path. If not specified, saves to 'models/clusters.geoparquet'.",
)
@click.option(
    "--save-map",
    type=click.Path(),
    help="Save visualization map to PNG file at this path. If not specified, saves to 'models/clusters_map.png'.",
)
@working_directory_option
def init_multiple(
    config: str,
    build_config: str,
    update_config: str,
    working_directory: Path,
    from_example: str,
    geometry_bounds: str,
    target_area_km2: float,
    area_tolerance: float,
    cluster_prefix: str,
    overwrite: bool,
    save_geoparquet: Path | None,
    save_map: str | None,
) -> None:
    """Initialize multiple models by clustering downstream subbasins in a geometry.

    This command identifies all downstream subbasins (outlets) within a specified
    bounding box, clusters them by proximity and cumulative upstream area, and
    creates separate model configurations for each cluster.

    Example for parts of Europe:
        geb init_multiple --geometry-bounds="5.0,50.0,15.0,55.0"

    By default, a region covering Europe is used. Use --geometry-bounds to specify a different region.
    """
    init_multiple_fn(
        config=config,
        build_config=build_config,
        update_config=update_config,
        working_directory=working_directory,
        from_example=from_example,
        geometry_bounds=geometry_bounds,
        target_area_km2=target_area_km2,
        area_tolerance=area_tolerance,
        cluster_prefix=cluster_prefix,
        overwrite=overwrite,
        save_geoparquet=save_geoparquet,
        save_map=save_map,
    )


if __name__ == "__main__":
    cli()
