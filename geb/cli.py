"""Command line interface for GEB."""

import datetime
import functools
import json
import subprocess
import sys
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, cast

import click

from geb import GEB_PACKAGE_DIR, __version__
from geb.build.data_catalog import DataCatalog
from geb.evaluate import Evaluate
from geb.runner import (
    ALTER_FROM_MODEL_DEFAULT,
    BUILD_DEFAULT,
    CONFIG_DEFAULT,
    CORES_DEFAULT,
    OPTIMIZE_DEFAULT,
    PROFILE_RAM_DEFAULT,
    PROFILE_SPEED_DEFAULT,
    TIMING_DEFAULT,
    UPDATE_DEFAULT,
    WORKING_DIRECTORY_DEFAULT,
    alter_fn,
    build_fn,
    init_fn,
    init_multiple_fn,
    run_model_with_method,
    set_fn,
    share_fn,
    update_fn,
    update_version_fn,
)
from geb.workflows.io import WorkingDirectory
from geb.workflows.raster import rechunk_zarr_file

IS_WINDOWS = sys.platform == "win32"


def get_available_evaluation_methods() -> list[str]:
    """Return the public evaluation methods available through ``geb evaluate``.

    Returns:
        Sorted list of fully-qualified evaluation method names.
    """
    evaluator = Evaluate(cast(Any, None))
    available_methods: list[str] = []

    for sub_evaluator_name in evaluator.sub_evaluators:
        sub_evaluator = getattr(evaluator, sub_evaluator_name)
        for attribute_name in dir(sub_evaluator):
            if attribute_name.startswith("_"):
                continue

            attribute = getattr(sub_evaluator, attribute_name)
            if callable(attribute):
                available_methods.append(f"{sub_evaluator_name}.{attribute_name}")

    return sorted(available_methods)


def format_available_evaluation_methods(methods: list[str]) -> str:
    """Format evaluation methods for CLI help text.

    Args:
        methods: Fully-qualified evaluation method names.

    Returns:
        Multi-line bullet list for CLI help output.
    """
    if not methods:
        return "  - No evaluation methods available."

    return "\n".join(f"  - {method_name}" for method_name in methods)


AVAILABLE_EVALUATION_METHODS: list[str] = get_available_evaluation_methods()
AVAILABLE_EVALUATION_METHODS_HELP: str = format_available_evaluation_methods(
    AVAILABLE_EVALUATION_METHODS
)
EVALUATE_HELP = (
    "Evaluate model, for example by comparing observed and simulated discharge.\n\n"
    "Accepts additional parameter assignments in the form `--key value` or "
    "`--key=value` to pass to the evaluation method. Strings `true` and "
    "`false` (case-insensitive) are converted to booleans.\n\n"
    "\b\n"
    "Available methods:\n"
    f"{AVAILABLE_EVALUATION_METHODS_HELP}\n\n"
    "Use `geb evaluate [METHOD] --help` for method-specific documentation."
)


@click.group(help="Command line interface for GEB.")
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


def universal_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to add universal options to a click command.

    Useful to add the same options to multiple commands.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """

    @click_config
    @working_directory_option
    @click.option(
        "--profile-speed",
        is_flag=True,
        default=PROFILE_SPEED_DEFAULT,
        help="Run GEB with speed profiling. Stats are saved in the profiling directory.",
    )
    @click.option(
        "--profile-ram",
        is_flag=True,
        default=PROFILE_RAM_DEFAULT,
        help="Run GEB with RAM profiling (using memray). A .bin file is saved in the profiling directory. Not supported on Windows."
        if not IS_WINDOWS
        else "RAM profiling is not supported on Windows.",
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
    @click.option(
        "--cores",
        "-n",
        type=int,
        default=CORES_DEFAULT,
        help="Restrict the number of CPU cores used by the process. Not supported on Windows.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function for universal options.

        Returns:
            The result of the wrapped function.

        Raises:
            click.ClickException: If RAM profiling is requested on Windows.
        """
        if kwargs.get("profile_ram") and IS_WINDOWS:
            raise click.ClickException(
                "RAM profiling with memray is not supported on Windows."
            )
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

        @universal_options
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

        @universal_options
        @click.option(
            "--build-config",
            "-b",
            type=click.Path(path_type=Path),
            default=Path(build_config),
            help=build_config_help,
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


# Implementation functions moved to geb.runner


@cli.command()
@universal_options
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
def init(*args: Any, **kwargs: Any) -> None:
    """Initialize a new model."""
    # Initialize the model with the given config and build config
    init_fn(*args, **kwargs)


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@universal_options
@click.pass_context
def set(
    ctx: click.Context, config: Path, working_directory: Path, **kwargs: Any
) -> None:
    """Set model configuration values.

    Accepts parameter assignments in the form key=value, where keys can use
    dot notation for nested values (e.g., model.param1=0.5).

    By default, only existing keys can be updated. To create new keys,
    append a '+' to the key (e.g., model.new_param+=10).

    Args:
        ctx: Click context containing extra arguments.
        config: Path to the model configuration file.
        working_directory: Working directory for the model.
        **kwargs: Universal options.

    """
    # Parse extra arguments as key=value pairs
    params = {}
    for arg in ctx.args:
        if "=" in arg:
            key, value = arg.split("=", 1)

            # Try to convert value to appropriate type
            def is_float(element: str) -> bool:
                try:
                    float(element)
                    return True
                except ValueError:
                    return False

            def is_date(element: str) -> bool:
                try:
                    datetime.date.fromisoformat(element)
                    return True
                except ValueError:
                    return False

            def is_datetime(element: str) -> bool:
                try:
                    datetime.datetime.fromisoformat(element)
                    return True
                except ValueError:
                    return False

            if value.isdigit():
                value = int(value)
            elif is_float(value):
                value = float(value)
            elif is_date(value):
                value = datetime.date.fromisoformat(value)
            elif is_datetime(value):
                value = datetime.datetime.fromisoformat(value)
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                value: str = value  # Keep as string if it cannot be converted to a more specific type

            params[key] = value
        else:
            click.echo(
                f"Warning: Ignoring invalid argument '{arg}'. Expected format: key=value",
                err=True,
            )

    set_fn(config=config, working_directory=working_directory, **params)


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


@cli.command()
@click_build_options()
@click.option(
    "--from-model",
    type=click.Path(path_type=Path),
    default=ALTER_FROM_MODEL_DEFAULT,
    help=f"Folder for the existing model. Defaults to {ALTER_FROM_MODEL_DEFAULT}.",
)
def alter(*args: Any, **kwargs: Any) -> None:
    """Create alternative version from base model with only changed files.

    This command is useful to create a new model based on an existing one, but with
    only a few changes. It will copy the base model and overwrite the files that are
    specified in the config and build config files. The rest of the files will be
    linked to the original model to reduce disk space.
    """
    alter_fn(*args, **kwargs)


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
@click_build_options()
def update_version(*args: Any, **kwargs: Any) -> None:
    """Update the model version file to the current model version.

    This command initializes the GEBModel, which automatically checks and updates
    the version file if it is outdated, printing any necessary update instructions.
    """
    update_version_fn(*args, **kwargs)


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True, allow_extra_args=True, help_option_names=[]
    ),
    help=EVALUATE_HELP,
)
@click.argument("method", default="hydrology.evaluate_discharge")
@universal_options
@click.option("--spinup-name", default="spinup", help="Name of the evaluation run.")
@click.option("--run-name", default="default", help="Name of the run to evaluate.")
@click.option("--help", is_flag=True, help="Show this message and exit.")
@click.pass_context
def evaluate(
    ctx: click.Context,
    method: str,
    spinup_name: str,
    run_name: str,
    help: bool,
    working_directory: Path = WORKING_DIRECTORY_DEFAULT,
    config: dict[str, Any] | Path = CONFIG_DEFAULT,
    profile_speed: bool = PROFILE_SPEED_DEFAULT,
    profile_ram: bool = PROFILE_RAM_DEFAULT,
    optimize: bool = OPTIMIZE_DEFAULT,
    timing: bool = TIMING_DEFAULT,
    cores: int | None = CORES_DEFAULT,
) -> None:
    """Evaluate model, for example by comparing observed and simulated discharge.

    Accepts additional parameter assignments in the form key=value to pass to the evaluation method.
    Strings "true" and "false" (case-insensitive) are converted to True and False.

    Additional help can be retrieved for a specific method like this:
    `geb evaluate [METHOD] --help`

    Args:
        ctx: Click context containing extra arguments.
        method: Single evaluation method to run, e.g. `hydrology.evaluate_discharge`.
        spinup_name: Name of the evaluation run.
        run_name: Name of the run to evaluate.
        help: Show this message and exit.
        working_directory: Working directory for the model.
        config: Path to the model configuration file or a dict with the config.
        profile_speed: If True, run the model with speed profiling.
        profile_ram: If True, run the model with RAM profiling.
        optimize: If True, run the model in optimized mode, skipping asserts and water balance checks.
        timing: If True, run the model with timing, printing the time taken for specific methods.
        cores: Number of cores to restrict the model to using taskset.

    Raises:
        click.ClickException: If RAM profiling is requested on Windows.
    """
    if help:
        # Check if the user specifically requested help for the command or a method.
        # If 'method' was explicitly provided by the user (and is not just the default),
        # we show the help for that specific method.
        is_method_help = (
            ctx.get_parameter_source("method") != click.core.ParameterSource.DEFAULT
        )

        if not is_method_help:
            click.echo(ctx.get_help())
            ctx.exit()

        # If it's method help, show method docstring

        try:
            evaluator = Evaluate(cast(Any, None))
            attr = attrgetter(method)(evaluator)
            click.echo(f"\nHelp for method '{method}':\n")
            if attr.__doc__:
                click.echo(attr.__doc__)
            else:
                click.echo("No documentation found for this method.")
        except Exception:
            click.echo(f"Error: Method '{method}' not found.")
            if AVAILABLE_EVALUATION_METHODS:
                click.echo("\nAvailable methods are:")
                for m in AVAILABLE_EVALUATION_METHODS:
                    click.echo(f"  - {m}")
        ctx.exit()

    method_name = method.replace("-", "_").strip()
    # Parse extra arguments from ctx.args
    # Supports --key value or --key=value formats
    extra_args = {}
    i = 0
    while i < len(ctx.args):
        arg = ctx.args[i]
        key = None
        value = None

        if arg.startswith("--"):
            if "=" in arg:
                # Handle --key=value
                parts = arg[2:].split("=", 1)
                key = parts[0]
                value = parts[1]
                i += 1
            else:
                # Handle --key value
                key = arg[2:]
                if i + 1 < len(ctx.args) and not ctx.args[i + 1].startswith("--"):
                    value = ctx.args[i + 1]
                    i += 2
                else:
                    # Flag case: --key without value
                    value = "true"
                    i += 1
        else:
            click.echo(
                f"Warning: Ignoring invalid argument '{arg}'. Expected format: --key value or --key=value",
                err=True,
            )
            i += 1
            continue

        if key:
            # Try to convert value to appropriate type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
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
            extra_args[key] = value

    if profile_ram and IS_WINDOWS:
        raise click.ClickException(
            "RAM profiling with memray is not supported on Windows."
        )

    result = run_model_with_method(
        method="evaluate",
        method_args={
            "method": method_name,
            "spinup_name": spinup_name,
            "run_name": run_name,
            **extra_args,
        },
        working_directory=working_directory,
        config=config,
        profile_speed=profile_speed,
        profile_ram=profile_ram,
        optimize=optimize,
        timing=timing,
        cores=cores,
    )

    # If the result is a dictionary, print it as JSON to stdout
    # This allows piping metrics to other tools or Snakemake
    if isinstance(result, dict):
        click.echo(json.dumps(result))


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
    data_catalog = DataCatalog()
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
@click.argument(
    "track",
    required=False,
    type=str,
)
@click.option(
    "--cores",
    "-c",
    "workflow_cores",
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
    track: str | None,
    workflow_cores: str,
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
        geb workflow calibrate hydrology --cores 8
        geb workflow calibrate --profile profiles/cluster
        geb workflow calibrate -co REGION=geul -co NGEN=10
        geb workflow sensitivity --dryrun

    Args:
        workflow_name: Name of the workflow to run.
        track: Optional calibration track (e.g., 'hydrology').
        workflow_cores: Number of cores to use.
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

        # Add track as config override if provided
        config_overrides_dict = {}
        if track is not None:
            config_overrides_dict["TRACK"] = track

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
            cmd.extend(["--cores", workflow_cores])

        # Add dry run flag
        if dryrun:
            cmd.append("-n")

        # Process config overrides
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

        # Snakemake uses a lock file to prevent multiple runs in the same directory.
        # However, often when a process is interrupted (e.g., by Ctrl+C) the lock file is not removed,
        # which prevents running the workflow again until the lock file is manually removed.
        # However, we leave it to the user to ensure they don't run multiple workflows
        # so we unlock any existing lock file at the start of the workflow.
        result = subprocess.run(cmd + ["--unlock"], check=True)
        if result.returncode != 0:
            click.echo("Error: Failed to unlock Snakemake lock file.", err=True)
            sys.exit(result.returncode)

        # Print command
        click.echo(f"Running: {' '.join(cmd)}")

        # Execute snakemake in the working directory
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


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
    "--region-shapefile",
    type=str,
    help="Optional path to region shape file (in a format supported by geopandas and relative to the working directory). Defaults to geometry bounds if not specified.",
)
@click.option(
    "--target-area-km2",
    default=817e3,
    type=float,
    help="Target cumulative upstream area per cluster in km². Defaults to 817,000 km².",
)
@click.option(
    "--cluster-prefix",
    default="cluster",
    help="Prefix for cluster directory names. Defaults to 'cluster'.",
)
@click.option(
    "--skip-merged-geometries",
    is_flag=True,
    default=False,
    help="Skip creating merged geometry file (faster, but no dissolved basin polygons).",
)
@click.option(
    "--skip-visualization",
    is_flag=True,
    default=False,
    help="Skip creating visualization map (faster).",
)
@click.option(
    "--min-bbox-efficiency",
    default=0.99,
    type=float,
    help="Minimum bbox efficiency (0-1) for cluster merging. Higher values create more compact/square clusters. Default: 0.97 (97% land fill ratio, allows only ~3% wasted land). Use 0.85 for slightly less compact (85%), 0.70 for moderate compactness, or 0.60 for more elongated shapes.",
)
@click.option(
    "--ocean-outlets-only",
    is_flag=True,
    default=False,
    help="If set, only include clusters that flow to the ocean (exclude endorheic basins).",
)
@click.option(
    "--init-multiple-dir",
    required=True,
    help="Name of the subdirectory in models/ where the large scale model directories will be created (e.g. 'large_scale' or 'large_scale2').",
)
@working_directory_option
def init_multiple(*args: Any, **kwargs: Any) -> None:
    """Initialize a new model for multiple subbasins."""
    # Initialize the model with the given config and build config
    init_multiple_fn(*args, **kwargs)


@cli.command()
def server() -> None:
    """Run the GEB MCP server."""
    from geb.mcp_server import mcp

    mcp.run()


@cli.group()
def tool() -> None:
    """Useful tools for GEB."""
    pass


@tool.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--how",
    type=click.Choice(
        ["time-optimized", "space-optimized", "balanced"], case_sensitive=False
    ),
    required=True,
    help="How to optimize the chunks.",
)
@click.option(
    "--no-intermediate",
    is_flag=False,
    default=True,
    help="Use intermediate rechunking step (recommended for large files).",
)
def rechunk(
    input_path: Path, output_path: Path, how: str, no_intermediate: bool
) -> None:
    """Rechunk a Zarr file."""
    rechunk_zarr_file(input_path, output_path, how, not no_intermediate)  # type: ignore


if __name__ == "__main__":
    cli()
