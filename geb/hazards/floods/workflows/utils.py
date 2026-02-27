"""Utility functions for flood hazard workflows."""

import os
import platform
import shutil
import subprocess
from datetime import datetime
from os.path import isfile, join
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel, utils
from matplotlib.cm import viridis  # ty: ignore[unresolved-import]
from shapely import line_locate_point
from shapely.geometry import GeometryCollection, LineString, Point

from geb.geb_types import ArrayFloat32, ArrayInt64


def export_rivers(
    model_root: Path, rivers: gpd.GeoDataFrame, postfix: str = ""
) -> None:
    """Export river segments to a GeoParquet file.

    Args:
        model_root: The root path of the model directory.
        rivers: The GeoDataFrame containing the river segments.
        postfix: A postfix to add to the filename. Defaults to "".
    """
    rivers.to_parquet(model_root / f"rivers{postfix}.geoparquet")


def import_rivers(model_root: Path, postfix: str = "") -> gpd.GeoDataFrame:
    """Import river segments from a GeoParquet file.

    Args:
        model_root: The root path of the model directory.
        postfix: A postfix to add to the filename. Defaults to "".

    Returns:
        A GeoDataFrame containing the river segments.
    """
    return gpd.read_parquet(model_root / f"rivers{postfix}.geoparquet")


def read_flood_depth(
    model_root: Path,
    simulation_root: Path,
    method: str,
    minimum_flood_depth: float,
    end_time: datetime,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Read the maximum flood depth from the SFINCS model results.

    If SFINCS was run with subgrid, the flood depth is downscaled to subgrid resolution.

    Notes:
        If SFINCS was run as a fluvial model, a minimum flood depth of 0.15 m is applied.
        If SFINCS was run as a pluvial/coastal model, a minimum flood depth of 0.05 m is applied.
        There are some fundamental issues with the current generation of subgrid maps,
            especially in steep terrain, see conclusion here: https://doi.org/10.5194/gmd-18-843-2025
            Here, we use bilinear interpolation as a goodish solution. However, we keep our
            eyes out for better solutions.

    Args:
        model_root: The root path of the SFINCS model directory. This should contain the subgrid directory with the depth file.
        simulation_root: The root path of the SFINCS simulation directory. This should contain the simulation results files.
        minimum_flood_depth: The minimum flood depth to apply during downscaling (in meters).
        method: The method to use for calculating flood depth. Options are 'max' for maximum flood depth
            and 'final' for flood depth at the final time step.
        end_time: The end time of the simulation.
        mask: Optional xarray DataArray mask to apply to the flood depth. Defaults to None.

    Returns:
        The maximum flood depth downscaled to subgrid resolution.

    Raises:
        ValueError: If an unknown method is provided.
        KeyError: If the specified end_time is not in the results time dimension.
    """
    # Read SFINCS model, config and results
    model: SfincsModel = SfincsModel(
        str(simulation_root),
        mode="r",
    )
    model.read_config()

    # For unknown reasons, sometimes reading the results fails the first time
    # but succeeds the second time. Therefore, we try twice here.
    try:
        model.read_results()
    except OSError:
        model.read_results()

    # to detect whether SFINCS was run with subgrid, we check if the 'sbgfile' key exists in the config
    # to be extra safe, we also check if the value is not None or has has length > 0
    if (
        "sbgfile" in model.config
        and model.config["sbgfile"] is not None
        and len(model.config["sbgfile"]) > 0
    ):
        if method == "max":
            # get maximum water surface elevation (with respect to sea level)
            water_surface_elevation = model.results["zsmax"].max(dim="timemax")
            assert isinstance(water_surface_elevation, xr.DataArray)
        elif method == "final":
            # get water surface elevation at the final time step (with respect to sea level)
            all_water_surface_elevation: xr.Dataset | xr.DataArray = model.results["zs"]
            assert isinstance(all_water_surface_elevation, xr.DataArray)
            try:
                water_surface_elevation: xr.DataArray = all_water_surface_elevation.sel(
                    time=end_time
                )
            except KeyError:
                raise KeyError(
                    f"The specified end_time {end_time} is not in the results time dimension. \
                    Consider whether the reporting interval of SFINCS is expected \
                    to include the end_time."
                )
            assert isinstance(water_surface_elevation, xr.DataArray)
        else:
            raise ValueError(f"Unknown method: {method}")

        if mask is not None:
            mask = mask.compute()
            water_surface_elevation = water_surface_elevation.compute()
            water_surface_elevation = water_surface_elevation.where(mask, np.nan)
        # read subgrid elevation
        surface_elevation: xr.DataArray = (
            xr.open_dataarray(model_root / "subgrid" / "dep_subgrid.tif")
            .sel(band=1)
            .drop_vars(["band"])
        )

        flood_depth_m: xr.DataArray = utils.downscale_floodmap(
            zsmax=water_surface_elevation,
            dep=surface_elevation,
            hmin=minimum_flood_depth,
            reproj_method="bilinear",  # maybe use "nearest" for coastal
        )
        flood_depth_m: xr.DataArray = flood_depth_m.rio.write_crs(model.crs)

    else:
        if method == "max":
            flood_depth_m: xr.Dataset | xr.DataArray = model.results["hmax"].max(
                dim="timemax"
            )
            assert isinstance(flood_depth_m, xr.DataArray)
        elif method == "final":
            flood_depth_m_all_steps: xr.Dataset | xr.DataArray = model.results["h"]
            assert isinstance(flood_depth_m_all_steps, xr.DataArray)
            try:
                flood_depth_m: xr.DataArray = flood_depth_m_all_steps.sel(time=end_time)
            except KeyError:
                raise KeyError(
                    f"The specified end_time {end_time} is not in the results time dimension. \
                    Consider whether the reporting interval of SFINCS is expected \
                    to include the end_time."
                )

        else:
            raise ValueError(f"Unknown method: {method}")

        flood_depth_m: xr.DataArray = xr.where(
            flood_depth_m >= minimum_flood_depth, flood_depth_m, np.nan, keep_attrs=True
        )
        if mask is not None:
            flood_depth_m = flood_depth_m.where(mask.values, np.nan)

        flood_depth_m.attrs["_FillValue"] = np.nan

    print(
        f"Maximum flood depth: {float(flood_depth_m.max().values):.2f} m, "
        f"Mean flood depth: {float(flood_depth_m.mean().values):.2f} m"
    )
    # Create basemap plot
    fig, ax = model.plot_basemap(
        fn_out=None,  # ty: ignore[invalid-argument-type]
        variable="",  # No variable to plot, only basemap
        plot_geoms=False,
        zoomlevel=12,
        figsize=(11, 7),  # ty: ignore[invalid-argument-type]
    )

    # Plot flood depth with colorbar
    cbar_kwargs: dict[str, float | tuple[int, int]] = {"shrink": 0.6, "anchor": (0, 0)}
    flood_depth_m.plot(
        x="x",
        y="y",
        ax=ax,
        vmin=0,
        vmax=float(flood_depth_m.max().values),
        cmap=viridis,
        cbar_kwargs=cbar_kwargs,
    )

    ax.set_title("Maximum Water Depth over all time steps")

    output_path: Path = model_root / "flood_depth_all_time_steps.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return flood_depth_m


def to_sfincs_datetime(dt: datetime) -> str:
    """Convert a datetime object to a string in the format required by SFINCS.

    Args:
        dt: datetime object to convert.

    Returns:
        String representation of the datetime in the format "YYYYMMDD HHMMSS".
    """
    return dt.strftime("%Y%m%d %H%M%S")


def make_relative_paths(
    config: dict[str, Any],
    model_root: Path,
    new_root: Path,
) -> dict[str, Any]:
    """Return dict with paths to the new model new root.

    Always use POSIX format for paths, to ensure compatibility with Docker.

    Args:
        config: The configuration dictionary with paths to make relative.
        model_root: The original model root directory.
        new_root: The new model root directory.

    Returns:
        A dictionary with the updated paths relative to the new model root directory.

    Raises:
        ValueError: If model_root and new_root do not have a common path.
    """
    commonpath = ""
    if os.path.splitdrive(model_root)[0] == os.path.splitdrive(new_root)[0]:
        commonpath = os.path.commonpath([model_root, new_root])
    if os.path.basename(commonpath) == "":
        raise ValueError("model_root and new_root must have a common path")
    relpath = Path(os.path.relpath(commonpath, new_root))

    config_kwargs = dict()
    for k, v in config.items():
        if (
            isinstance(v, str)
            and isfile(join(model_root, v))
            and not isfile((join(new_root, v)))
        ):
            # Ensure paths are consistent across platforms
            path = Path(relpath) / Path(v)  # Ensure both are Path objects
            config_kwargs[k] = (
                path.as_posix()
            )  # Always use POSIX format for Docker compatibility

    return config_kwargs


def run_sfincs_subprocess(
    working_directory: Path, cmd: list[str], log_file: Path
) -> int:
    """Runs SFINCS model in a subprocess.

    Args:
        working_directory: The working directory for the subprocess.
        cmd: The command to run as a list of strings.
        log_file: The file to write the log output to.

    Returns:
        The return code of the subprocess.

    """
    print(f"Running SFINCS with: {cmd}")
    with open(file=log_file, mode="w") as log:
        with subprocess.Popen(
            args=cmd,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            stdout = process.stdout
            stderr = process.stderr
            assert stdout is not None
            assert stderr is not None

            # Continuously read lines from stdout and stderr
            for line in iter(lambda: stdout.readline() or stderr.readline(), ""):
                print(line.rstrip())
                log.write(line)
                log.flush()

            process.wait()

    return process.returncode


def get_start_point(geom: LineString) -> Point:
    """Extract the start point from a LineString geometry.

    Args:
        geom: LineString geometry.

    Returns:
        The start point as a Shapely Point object.
    """
    return Point(geom.coords[0])


def get_end_point(geom: LineString) -> Point:
    """Extract the end point from a LineString geometry.

    Args:
        geom: LineString geometry.

    Returns:
        The end point as a Shapely Point object.
    """
    return Point(geom.coords[-1])


def create_hourly_hydrograph(
    peak_discharge: float,
    rising_limb_hours: int,
    recession_limb_hours: int,
) -> pd.DataFrame:
    """Create a triangular hydrograph time series.

    Args:
        peak_discharge: The peak discharge of the hydrograph.
        rising_limb_hours: The duration of the rising limb in hours.
        recession_limb_hours: The duration of the recession limb in hours.

    Returns:
        A pandas DataFrame with the time series of the hydrograph.
    """
    total_duration_hours: int | float = rising_limb_hours + recession_limb_hours
    # Create a time array with hourly intervals
    time_hours: ArrayInt64 = np.arange(0, total_duration_hours + 1)
    # Create discharge array for triangular shape
    discharge: ArrayFloat32 = np.zeros_like(time_hours, dtype=np.float32)
    # Rising limb: linear increase to peak
    discharge[: rising_limb_hours + 1] = np.linspace(
        0, peak_discharge, rising_limb_hours + 1
    )
    # Recession limb: linear decrease from peak
    discharge[rising_limb_hours:] = np.linspace(
        peak_discharge, 0, recession_limb_hours + 1
    )
    # Create a pandas DataFrame for the time series
    time_index: pd.DatetimeIndex = pd.date_range(
        start="2024-01-01 00:00", periods=len(time_hours), freq="h"
    )
    hydrograph_df: pd.DataFrame = pd.DataFrame(
        {"time": time_index, "discharge": discharge}
    )
    hydrograph_df = hydrograph_df.set_index("time")
    return hydrograph_df


def create_hydrograph_from_discharge_shape(
    discharge_series: pd.Series,
    peak_discharge: float,
    anchor_discharge: float,
    window_hours: int = 84,
    tolerance: float = 0.1,
    min_events: int = 3,
    output_path: Path | None = None,
    river_idx: int | None = None,
    return_period: int | float | None = None,
) -> pd.DataFrame:
    """Create a hydrograph by extracting the mean shape from historical discharge events.

    Finds historical events where discharge is within ``tolerance`` of
    ``anchor_discharge``, extracts windows of ±``window_hours`` around each
    peak, averages them to obtain a normalised mean shape, then scales the
    shape so that its peak equals ``peak_discharge``.

    For the ``direct`` shape method ``anchor_discharge`` equals
    ``peak_discharge``.  For the ``anchor`` method ``anchor_discharge`` is
    the 2-year return period discharge (Q_2), which typically has enough
    historical occurrences to build a reliable mean shape even when the
    target return period is rare.

    Args:
        discharge_series: Hourly discharge time series for the river reach.
        peak_discharge: Target peak discharge (Q_RP from GPD-POT).
        anchor_discharge: Discharge value used to identify reference events.
        window_hours: Number of hours extracted before AND after the peak (total window = 2 × window_hours + 1). Default 84 (3.5 days either side → 7 days total).
        tolerance: Fractional tolerance around anchor_discharge. Default 0.1 (±10%).
        min_events: Minimum number of events required to build the shape. Default 3.
        output_path: Directory to save diagnostic figures. If None no figure is saved.
        river_idx: River index used in the figure title and filename.
        return_period: Return period used in the figure title and filename.

    Returns:
        pd.DataFrame with a DatetimeIndex and a ``discharge`` column.

    Raises:
        ValueError: If fewer than ``min_events`` events are found near
            ``anchor_discharge``.
    """
    lower = anchor_discharge * (1 - tolerance)
    upper = anchor_discharge * (1 + tolerance)

    # Find events where discharge rises above `lower`, then keep only those
    # whose TRUE event peak falls within [lower, upper].  Using `lower` (not
    # `upper`) as the entry threshold ensures we capture the full event even
    # when discharge briefly exceeds `upper` mid-event; the tolerance filter
    # on the peak value is what guarantees that every accepted event peak —
    # and therefore every extracted window peak — lies inside the yellow band.
    above_lower_mask = discharge_series >= lower

    peak_times: list[pd.Timestamp] = []
    in_event = False
    event_start: pd.Timestamp | None = None

    for time, above in above_lower_mask.items():
        if above and not in_event:
            in_event = True
            event_start = time
        elif not above and in_event:
            in_event = False
            event_slice = discharge_series[event_start:time]
            peak_val = float(event_slice.max())
            if lower <= peak_val <= upper:
                peak_times.append(event_slice.idxmax())

    if in_event and event_start is not None:
        event_slice = discharge_series[event_start:]
        peak_val = float(event_slice.max())
        if lower <= peak_val <= upper:
            peak_times.append(event_slice.idxmax())

    # Extract fixed-length windows around each peak
    windows: list[npt.NDArray[np.float64]] = []
    for peak_time in peak_times:
        start = peak_time - pd.Timedelta(hours=window_hours)
        end = peak_time + pd.Timedelta(hours=window_hours)
        window = discharge_series[start:end]
        if len(window) == 2 * window_hours + 1:
            windows.append(window.values)

    if len(windows) < min_events:
        raise ValueError(
            f"Only {len(windows)} event(s) found near discharge "
            f"{anchor_discharge:.2f} m³/s (±{tolerance * 100:.0f}%) for river "
            f"{river_idx}, return period {return_period}. "
            f"Need at least {min_events}. "
            f"Consider using 'triangular' shape or the 'anchor' method with a lower anchor RP."
        )

    mean_shape_at_anchor = np.nanmean(windows, axis=0)

    # Scale so peak equals peak_discharge
    peak_of_mean = float(mean_shape_at_anchor.max())
    scaled_shape = (
        (mean_shape_at_anchor * (peak_discharge / peak_of_mean)).astype(np.float32)
        if peak_of_mean > 0
        else mean_shape_at_anchor.astype(np.float32)
    )

    n_steps = 2 * window_hours + 1
    time_index = pd.date_range(start="2024-01-01 00:00", periods=n_steps, freq="h")
    hydrograph_df = pd.DataFrame({"discharge": scaled_shape}, index=time_index)
    hydrograph_df.index.name = "time"

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        time_axis = np.arange(-window_hours, window_hours + 1)

        fig, ax = plt.subplots(figsize=(12, 7))

        for w in windows:
            ax.plot(time_axis, w, color="gray", alpha=0.3)

        ax.plot(
            time_axis,
            mean_shape_at_anchor,
            color="blue",
            linewidth=2,
            label=f"Mean shape at anchor ({anchor_discharge:.1f} m³/s)",
        )
        ax.plot(
            time_axis,
            scaled_shape,
            color="purple",
            linewidth=2,
            linestyle="--",
            label=f"Scaled hydrograph (RP={return_period}, {peak_discharge:.1f} m³/s)",
        )
        ax.axhline(
            anchor_discharge,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Anchor discharge ({anchor_discharge:.1f} m³/s)",
        )
        ax.fill_between(
            time_axis,
            lower,
            upper,
            color="yellow",
            alpha=0.2,
            label=f"±{tolerance * 100:.0f}% anchor range",
        )
        ax.axhline(
            peak_discharge,
            color="red",
            linestyle="--",
            label=f"Q_RP={return_period} ({peak_discharge:.1f} m³/s)",
        )
        ax.axvline(0, color="black", linestyle=":", label="Peak hour")

        ax.set_title(
            f"Hydrograph shape — River {river_idx}, RP={return_period} "
            f"({len(windows)} events)"
        )
        ax.set_xlabel("Hours relative to peak")
        ax.set_ylabel("Discharge (m³/s)")
        ax.legend(loc="best")
        ax.grid(True)

        fig.savefig(
            output_path / f"river_{river_idx}_rp_{return_period}_shape.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    return hydrograph_df


def check_docker_running() -> bool | None:
    """Check if Docker is installed and running.

    Returns:
        True if Docker is installed and running, False otherwise.

    """
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Docker not installed properly or not running properly.")
        return False


def run_sfincs_simulation(
    model_root: Path,
    simulation_root: Path,
    ncpus: int | str = "auto",
    gpu: bool | str = "auto",
) -> int:
    """Run SFINCS simulation using either Apptainer or Docker.

    Args:
        model_root: Path to the model root directory.
        simulation_root: Path to the simulation root directory.
            Some paths in the configuration will be made relative to this path.
            The simulation directory must be a subdirectory of the model root directory.
        ncpus: Number of CPUs to use. Can be an integer or 'auto' to automatically detect the number of CPUs.
        gpu: Whether to use GPU support. Can be True, False, or 'auto'. In auto mode,
            the presence of an NVIDIA GPU is checked using `nvidia-smi`. Defaults to auto.

    Raises:
        ValueError: If gpu is not True, False, or 'auto'.
        RuntimeError: If there is an error running the SFINCS simulation.

    Returns:
        The return code of the SFINCS simulation subprocess.
    """
    if gpu not in [True, False, "auto"]:
        raise ValueError("gpu must be True, False, or 'auto'")

    if gpu == "auto":
        # check if nvidia-smi is available, if so, check if a GPU is present
        if shutil.which(cmd="nvidia-smi") is not None:
            result: subprocess.CompletedProcess[bytes] = subprocess.run(
                args=["nvidia-smi"], capture_output=True
            )
            maybe_gpu: bool = result.returncode == 0

            # if no GPU is found, there is no need to check further, set gpu to False
            if not maybe_gpu:
                gpu: bool = False
            else:
                # otherwise, check if we are in a SLURM job. We may be on a cluster
                # where a GPU is physically present, but not allocated to the job
                # and therefore not available for use.
                in_SLURM_job: bool = "SLURM_JOB_ID" in os.environ
                if in_SLURM_job:
                    # in a SLURM job, if the job has access to the GPU
                    gpu_ids: str | None = os.getenv(key="CUDA_VISIBLE_DEVICES")
                    if gpu_ids is not None and len(gpu_ids) > 0:
                        gpu: bool = True
                    else:
                        gpu: bool = False
                else:
                    gpu: bool = True
        else:
            gpu: bool = False

        if gpu:
            print("GPU detected, running SFINCS with GPU support.")
        else:
            print("No GPU detected, running SFINCS without GPU support.")

    if gpu:
        version: str | None = os.getenv(key="SFINCS_CONTAINER_GPU")
        assert version is not None, (
            "SFINCS_CONTAINER_GPU environment variable is not set"
        )
    else:
        version: str | None = os.getenv(key="SFINCS_CONTAINER")
        assert version is not None, "SFINCS_CONTAINER environment variable is not set"

    if platform.system() == "Linux":
        # If not a apptainer image, add docker:// prefix
        # to the version string
        if not version.endswith(".sif"):
            version: str = "docker://" + version

        c = (
            int(
                os.getenv("SLURM_CPUS_PER_TASK", None)
                or os.getenv("SLURM_CPUS_ON_NODE", None)
                or os.cpu_count()  # returns none if cannot be determined
                or 1
            )
            if ncpus == "auto"
            else int(ncpus)
        )
        ncpus_str = "0" if c == 1 else f"0-{c - 1}"

        cmd: list[str] = [
            "taskset",
            "-c",
            ncpus_str,  # get user defined or automatically detected number of CPUs
            "apptainer",
            "run",
            "-B",  ## Bind mount
            f"{model_root.resolve()}:/data",
            "--pwd",  ## Set working directory inside container
            f"/data/{simulation_root.relative_to(model_root)}",
            "--nv",
            version,
        ]

    else:
        assert check_docker_running(), "Docker is not running"
        # in the docker image of SFINCS thet working directory is /data
        # but since we also want to load files that are from the model
        # root, we mount the model root to /data and then change the
        # working directory within the Docker image to the simulation root
        cmd: list[str] = [
            "docker",
            "run",
            "-v",
            f"{model_root.resolve()}:/data",
            "-w",
            f"/data/{simulation_root.relative_to(model_root).as_posix()}",
            version,
        ]

    log_file: Path = simulation_root / "sfincs.log"
    return_code: int = run_sfincs_subprocess(simulation_root, cmd, log_file=log_file)
    if return_code != 0:
        raise RuntimeError(
            f"Error running SFINCS simulation: {return_code}. The used command was: {' '.join(cmd)}. The log file is located at {log_file}."
        )
    else:
        return return_code


def _get_xy(
    river: pd.Series,
    up_to_downstream: bool = True,
) -> tuple[int, int]:
    """Get the first valid xy coordinate from a river's hydrography_xy list.

    Starts upstream or downstream based on the up_to_downstream flag.

    Args:
        river: Series containing river information, including 'hydrography_xy'.
            This is a list of (x, y) tuples, representing the location of the river
            in the low-resolution hydrological grid.
        up_to_downstream: Whether to search from upstream to downstream.
            Defaults to True (starting upstream).

    Returns:
        A tuple (x, y) of the first valid coordinate.
    """
    xys = river["hydrography_xy"]
    if up_to_downstream:
        idx: int = 0
    else:
        idx: int = -1
    return (xys[idx][0], xys[idx][1])


def get_representative_river_points(
    river_ID: set, rivers: pd.DataFrame
) -> list[tuple[int, int]]:
    """Get representative river points for a given river ID.

    If the river is represented in the low-resolution hydrological grid, its
    representative point is used. If not, the function traverses upstream to find
    rivers that are represented in the grid and uses their representative points.

    Args:
        river_ID: The ID of the river for which to find representative points.
        rivers: DataFrame containing river information, including 'represented_in_grid' and 'hydrography_xy'.

    Returns:
        A list of tuples (x, y) representing the coordinates of the representative points.
        If no valid points are found, an empty list is returned.

    Raises:
        ValueError: If no valid xy coordinates are found for rivers.
    """
    river = rivers.loc[river_ID]
    if river["represented_in_grid"]:
        xy = _get_xy(river, up_to_downstream=True)
        if xy is not None:
            return [xy]
        else:
            raise ValueError(
                f"Error: No valid xy found for river {river_ID} which is represented in the grid."
            )

    else:
        river_IDs = set([river_ID])
        representitative_rivers = set()
        while river_IDs:
            river_ID = river_IDs.pop()
            river = rivers.loc[river_ID]
            if not river["represented_in_grid"]:
                upstream_rivers = rivers[rivers["downstream_ID"] == river_ID]
                river_IDs.update(upstream_rivers.index)
            else:
                representitative_rivers.add(river_ID)

        representitative_rivers = rivers[rivers.index.isin(representitative_rivers)]
        xys = []
        for river_ID, river in representitative_rivers.iterrows():
            xy = _get_xy(river, up_to_downstream=False)
            if xy is not None:
                xys.append(xy)
            else:
                raise ValueError(
                    f"Error: No valid xy found for river {river_ID} which is represented not in the grid. Likely because upstream rivers could not be found."
                )

        return xys


def get_discharge_and_river_parameters_by_river(
    river_IDs: list[int],
    points_per_river: list[list[tuple[int, int]]],
    discharge: xr.DataArray,
    river_width_alpha: npt.NDArray[np.float32] | None = None,
    river_width_beta: npt.NDArray[np.float32] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract discharge time series and river parameters for each river.

    When rivers are represented in the low-resolution hydrological grid, rivers
    should have only one point in the points_per_river list. When rivers are not
    represented in the low-resolution hydrological grid, the river is represented
    by its upstream rivers, which can be multiple. In that case, the discharge
    time series for the river is the sum of the discharge time series of all its
    representative points.

    Args:
        river_IDs: List of river IDs.
        points_per_river: List of lists of (x, y) tuples representing the points for each river.
        discharge: xarray DataArray containing discharge values with dimensions (time, y, x).
        river_width_alpha: 2D array of river width alpha parameters, same shape as discharge y and x dimensions.
            If None, river width alpha will not be extracted. Defaults to None.
        river_width_beta: 2D array of river width beta parameters, same shape as discharge y and x dimensions.
            If None, river width beta will not be extracted. Defaults to None.

    Returns:
        A tuple containing:
            - A pandas DataFrame with discharge time series for each river (columns are river IDs).
            - A pandas DataFrame with river parameters (index is river IDs, columns are 'river_width_alpha' and 'river_width_beta').

    Raises:
        ValueError: If no points are found for rivers or if discharge values contain NaNs.
    """
    xs: list[int] = []
    ys: list[int] = []
    for points in points_per_river:
        xs.extend([p[0] for p in points])
        ys.extend([p[1] for p in points])

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No points found for rivers.")

    x_points: xr.DataArray = xr.DataArray(
        xs,
        dims="points",
    )
    y_points: xr.DataArray = xr.DataArray(
        ys,
        dims="points",
    )

    discharge_per_point: xr.DataArray = discharge.isel(
        x=x_points,
        y=y_points,
    ).compute()
    assert not np.isnan(discharge_per_point.values).any(), (
        "Discharge values contain NaNs"
    )

    if river_width_alpha is not None:
        river_width_alpha_per_point = river_width_alpha[y_points, x_points]
    else:
        river_width_alpha_per_point = None
    if river_width_beta is not None:
        river_width_beta_per_point = river_width_beta[y_points, x_points]
    else:
        river_width_beta_per_point = None

    discharge_df: pd.DataFrame = pd.DataFrame(index=discharge.time)
    discharge_df.index.freq = "H"  # ty:ignore[invalid-assignment]
    river_parameters: pd.DataFrame = pd.DataFrame(
        index=np.array(river_IDs),
        columns=np.array(
            ["river_width_alpha", "river_width_beta"],
        ),
    )

    i: int = 0
    for river_ID, points in zip(river_IDs, points_per_river, strict=True):
        discharge_per_river = discharge_per_point.isel(
            points=slice(i, i + len(points))
        ).sum(dim="points")

        discharge_df[river_ID] = discharge_per_river

        if river_width_alpha_per_point is not None:
            river_width_alpha_per_river = river_width_alpha_per_point[
                i : i + len(points)
            ].mean()
            if np.isnan(river_width_alpha_per_river):
                print(
                    f"Warning: River width alpha for river {river_ID} is NaN. Setting to 0."
                )
                river_width_alpha_per_river = 0.0
            river_parameters.loc[river_ID, "river_width_alpha"] = (
                river_width_alpha_per_river
            )

        if river_width_beta_per_point is not None:
            river_width_beta_per_river = river_width_beta_per_point[
                i : i + len(points)
            ].mean()
            if np.isnan(river_width_beta_per_river):
                print(
                    f"Warning: River width beta for river {river_ID} is NaN. Setting to 1."
                )
                river_width_beta_per_river = 1.0
            river_parameters.loc[river_ID, "river_width_beta"] = (
                river_width_beta_per_river
            )

        i += len(points)

    assert i == len(xs), "Discharge values do not match the number of points"
    assert i == len(ys), "Discharge values do not match the number of points"
    # make sure no NaN values are present in the discharge DataFrame
    assert not discharge_df.isnull().values.any(), "Discharge DataFrame contains NaNs"

    if river_width_alpha_per_point is not None:
        # make sure no NaN values are present in the river parameters DataFrame
        assert not river_parameters["river_width_alpha"].isnull().values.any(), (
            "River width alpha DataFrame contains NaNs"
        )
    if river_width_beta_per_point is not None:
        # make sure no NaN values are present in the river parameters DataFrame
        assert not river_parameters["river_width_beta"].isnull().values.any(), (
            "River width beta DataFrame contains NaNs"
        )
    return discharge_df, river_parameters


def select_most_downstream_point(
    river: LineString, outflow_points: GeometryCollection
) -> Point:
    """Select the most downstream point from a collection of outflow points.

    Args:
        river: LineString of the river geometry.
        outflow_points: GeometryCollection of outflow points (can contain Points and LineStrings).

    Returns:
        The most downstream Point from the outflow_points.

    Raises:
        TypeError: If an unsupported geometry type is found in outflow_points.
    """
    points: list[Point] = []
    for geom in outflow_points.geoms:
        if isinstance(geom, Point):
            points.append(geom)
        elif isinstance(geom, LineString):
            for coord in geom.coords:
                points.append(Point(coord))
        else:
            raise TypeError(
                f"Unsupported geometry type in outflow_points: {type(geom)}"
            )

    most_downstream_point: Point = points[0]
    most_downstream_point_loc: float = line_locate_point(river, most_downstream_point)
    for point in points[1:]:
        loc = line_locate_point(river, point)
        if loc > most_downstream_point_loc:
            most_downstream_point = point
            most_downstream_point_loc = loc

    outflow_point: Point = most_downstream_point

    return outflow_point
