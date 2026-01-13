"""Command line interface for GEB."""

import functools
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import click

from geb import GEB_PACKAGE_DIR, __version__
from geb.build.data_catalog import NewDataCatalog
from geb.runner import (
    ALTER_FROM_MODEL_DEFAULT,
    BUILD_DEFAULT,
    CONFIG_DEFAULT,
    DATA_CATALOG_DEFAULT,
    DATA_PROVIDER_DEFAULT,
    DATA_ROOT_DEFAULT,
    OPTIMIZE_DEFAULT,
    PROFILING_DEFAULT,
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
)
from geb.workflows.io import WorkingDirectory


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


# Implementation functions moved to geb.runner


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
    default="plot_discharge,evaluate_discharge,evaluate_hydrodynamics,water_balance",
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
    cluster_prefix: str,
    skip_merged_geometries: bool = False,
    skip_visualization: bool = False,
    min_bbox_efficiency: float = 0.99,
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
        cluster_prefix: Prefix for cluster directory names.
        skip_merged_geometries: If True, skip creating dissolved basin polygon file (much faster).
        skip_visualization: If True, skip creating visualization map (faster).
        min_bbox_efficiency: Minimum bbox efficiency (0-1) for cluster merging. Lower values allow more elongated clusters.

    Raises:
        FileNotFoundError: If the example folder does not exist.
        ValueError: If geometry_bounds format is invalid.
    """
    from geb.build import (
        cluster_subbasins_following_coastline,
        create_cluster_visualization_map,
        create_multi_basin_configs,
        get_all_downstream_subbasins_in_geom,
        get_river_graph,
        save_clusters_as_merged_geometries,
        save_clusters_to_geoparquet,
    )
    from geb.build.data_catalog import NewDataCatalog

    # set paths
    config: Path = Path(config)
    build_config: Path = Path(build_config)
    update_config: Path = Path(update_config)
    working_directory: Path = Path(working_directory)

    # Initialize data catalog and logger
    data_catalog_instance = NewDataCatalog()
    logger = create_logger(working_directory / "init_multiple.log")

    # Create the models/large_scale directory structure
    models_dir = Path.cwd().parent / "models"
    large_scale_dir = models_dir / "large_scale"

    # Clean the large_scale directory to start fresh
    if large_scale_dir.exists():
        logger.info(f"Removing existing large_scale directory: {large_scale_dir}")
        shutil.rmtree(large_scale_dir)

    # Create fresh directory
    large_scale_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created fresh large_scale directory: {large_scale_dir}")

    # create river
    logger.info("Starting multiple model initialization")
    logger.info(f"Using geometry bounds: {geometry_bounds}")
    logger.info(f"Target area: {target_area_km2:,.0f} km²")

    logger.info("Loading river network...")
    river_graph = get_river_graph(data_catalog_instance)

    # Parse geometry bounds and convert to geodataframe
    bounds = [float(x.strip()) for x in geometry_bounds.split(",")]
    if len(bounds) != 4:
        raise ValueError(
            "Invalid geometry_bounds format. Expected 'xmin,ymin,xmax,ymax'."
        )
    xmin, ymin, xmax, ymax = bounds

    bbox_geom = gpd.GeoDataFrame(
        geometry=[box(xmin, ymin, xmax, ymax)], crs="EPSG:4326"
    )

    downstream_subbasins = get_all_downstream_subbasins_in_geom(
        data_catalog_instance, bbox_geom, logger
    )  # get all downstream subbasins in the bounding box geometry

    if not downstream_subbasins:
        raise ValueError("No downstream subbasins found in the specified geometry")

    logger.info(f"Found {len(downstream_subbasins)} downstream subbasins")

    logger.info("Clustering subbasins by area and proximity...")
    clusters = cluster_subbasins_following_coastline(
        data_catalog_instance,
        downstream_subbasins,
        target_area_km2=target_area_km2,
        logger=logger,
        river_graph=river_graph,
        min_bbox_efficiency=min_bbox_efficiency,
    )

    logger.info(f"Created {len(clusters)} clusters")

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
        data_catalog=data_catalog_instance,
        river_graph=river_graph,
    )

    # save geoparquet and maps to default location
    save_geoparquet = large_scale_dir / f"{cluster_prefix}_outlets.geoparquet"
    save_map = large_scale_dir / f"{cluster_prefix}_clusters_map.png"

    logger.info(f"Saving outlet basins to geoparquet: {save_geoparquet}")
    # Save outlet-only clusters to geoparquet (simplified geometries)
    save_clusters_to_geoparquet(
        clusters=clusters,
        data_catalog=data_catalog_instance,
        output_path=save_geoparquet,
        cluster_prefix=cluster_prefix,
    )

    # Save complete basins as merged geometries (full upstream basins as single polygons)
    # This is slow for large datasets, so allow skipping
    if not skip_merged_geometries:
        merged_basins_path = (
            large_scale_dir / f"{cluster_prefix}_complete_basins.geoparquet"
        )
        logger.info(
            f"Saving complete basins as merged geometries: {merged_basins_path}"
        )
        save_clusters_as_merged_geometries(
            clusters=clusters,
            data_catalog=data_catalog_instance,
            river_graph=river_graph,
            output_path=merged_basins_path,
            cluster_prefix=cluster_prefix,
            buffer_distance_km=5.0,  # 5km buffer to merge nearby polygons (reduces MultiPolygon complexity)
        )
    else:
        logger.info("Skipping merged geometries (--skip-merged-geometries flag set)")

    # Create visualization map (optional)
    if not skip_visualization:
        logger.info(f"Creating visualization map: {save_map}")
        create_cluster_visualization_map(
            clusters=clusters,
            data_catalog=data_catalog_instance,
            river_graph=river_graph,
            output_path=save_map,
            cluster_prefix=cluster_prefix,
        )
    else:
        logger.info("Skipping visualization map (--skip-visualization flag set)")

    logger.info(
        f"Successfully created {len(cluster_directories)} model configurations:"
    )


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
@working_directory_option
def init_multiple(
    config: str,
    build_config: str,
    update_config: str,
    working_directory: Path,
    from_example: str,
    geometry_bounds: str,
    target_area_km2: float,
    cluster_prefix: str,
    skip_merged_geometries: bool,
    skip_visualization: bool,
    min_bbox_efficiency: float,
) -> None:
    """Initialize multiple models by clustering downstream subbasins in a geometry.

    This command identifies all downstream subbasins (outlets) within a specified
    bounding box, clusters them by proximity and cumulative upstream area, and
    creates separate model configurations for each cluster.

    Example for parts of Europe:
        geb init_multiple --geometry-bounds="5.0,50.0,15.0,55.0"

    By default, a region covering Europe is used. Use --geometry-bounds to specify a different region.

    Args:
        config: Path to the base model configuration file.
        build_config: Path to the base model build configuration file.
        update_config: Path to the base model update configuration file.
        working_directory: Working directory for the models.
        from_example: Name of the example to use as a base for the models.
        geometry_bounds: Bounding box as "xmin,ymin,xmax,ymax" to select sub-basins
        target_area_km2: Target cumulative upstream area per cluster
        cluster_prefix: Prefix used for created cluster directory names and output files.
        skip_merged_geometries: If True, skip creating dissolved basin polygon file (faster).
        skip_visualization: If True, skip creating visualization map (faster).
        min_bbox_efficiency: Minimum bbox efficiency (0-1) for cluster merging. Lower values allow more elongated clusters and fewer total clusters.
        overwrite: If True, existing cluster directories and files will be overwritten.

    """
    init_multiple_fn(
        config=config,
        build_config=build_config,
        update_config=update_config,
        working_directory=working_directory,
        from_example=from_example,
        geometry_bounds=geometry_bounds,
        target_area_km2=target_area_km2,
        cluster_prefix=cluster_prefix,
        skip_merged_geometries=skip_merged_geometries,
        skip_visualization=skip_visualization,
        min_bbox_efficiency=min_bbox_efficiency,
    )


@cli.command()
def server() -> None:
    """Run the GEB MCP server."""
    from geb.mcp_server import mcp

    mcp.run()


@cli.command()
def server() -> None:
    """Run the GEB MCP server."""
    from geb.mcp_server import mcp

    mcp.run()


if __name__ == "__main__":
    cli()
