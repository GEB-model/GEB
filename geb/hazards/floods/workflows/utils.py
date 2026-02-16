"""Utility functions for flood hazard workflows."""

import os
import platform
import shutil
import subprocess
import warnings
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
from scipy import stats
from scipy.stats import genpareto, kstest
from shapely import line_locate_point
from shapely.geometry import GeometryCollection, LineString, Point
from tqdm import tqdm

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


# helper functions for new return period distribution fitting
def calculate_lmoments(data: np.ndarray) -> tuple[float, float, float]:
    """Calculate L-moments (L1, L2, L3) from sorted data.

    Computes the first three L-moments using probability-weighted moments.
    L-moments are linear combinations of order statistics that are robust
    to outliers and exist even when conventional moments do not.

    Args:
        data: Array of observed values (dimensionless).

    Returns:
        Tuple of (L1, L2, L3) where L1 is L-location, L2 is L-scale,
        and L3 is the third L-moment (dimensionless).

    Raises:
        ValueError: If fewer than 6 data points provided.
    """
    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    if n < 6:
        raise ValueError("Too few data points for L-moment estimation")

    # Calculate probability-weighted moments (PWMs)
    j = np.arange(n)
    b0 = np.mean(x)
    b1 = np.sum(x * j) / (n * (n - 1))
    b2 = np.sum(x * j * (j - 1)) / (n * (n - 1) * (n - 2))

    # Convert PWMs to L-moments
    L1 = b0
    L2 = 2.0 * b1 - b0
    L3 = 6.0 * b2 - 6.0 * b1 + b0

    return float(L1), float(L2), float(L3)


def fit_gpd_lmoments(exceedances: np.ndarray) -> tuple[float, float]:
    """Fit GPD to positive exceedances using L-moments method.

    Fits a Generalized Pareto Distribution to exceedances (y = x - u) using
    the method of L-moments. This approach is more robust to outliers than
    Maximum Likelihood Estimation and does not require iterative optimization.

    The GPD parameters are estimated from L-moments using:
    - tau3 = L3 / L2 (L-skewness)
    - xi = (1 - 3*tau3) / (1 + tau3)
    - sigma = L2 * (1 - xi) * (2 - xi)

    Notes:
        For GPD validity, requires xi > -1 and sigma > 0.
        Shape parameter xi is constrained to [-0.5, 0.5] for stability.

    Args:
        exceedances: Positive exceedance values above threshold (dimensionless).

    Returns:
        Tuple of (sigma, xi) where sigma is the scale parameter and xi is
        the shape parameter of the fitted GPD (dimensionless).

    Raises:
        ValueError: If fewer than 6 exceedances provided for reliable fit.
        ValueError: If estimated parameters are invalid (sigma <= 0).
    """
    y = np.asarray(exceedances, dtype=float)
    if len(y) < 6:
        raise ValueError("Too few exceedances for reliable fit")

    # Calculate L-moments
    L1, L2, L3 = calculate_lmoments(y)

    # Calculate L-skewness (tau3)
    if L2 <= 0:
        raise ValueError("Invalid L2: must be positive")
    tau3 = L3 / L2

    # Estimate shape parameter xi from L-skewness
    # xi = (1 - 3*tau3) / (1 + tau3)
    # Constrain xi to reasonable range for numerical stability
    xi = (1.0 - 3.0 * tau3) / (1.0 + tau3)
    xi = np.clip(xi, -0.5, 0.5)

    # Estimate scale parameter sigma from L2 and xi
    # sigma = L2 * (1 - xi) * (2 - xi)
    sigma = L2 * (1.0 - xi) * (2.0 - xi)

    if sigma <= 0:
        raise ValueError(f"Invalid sigma={sigma}: must be positive")

    return float(sigma), float(xi)


def gpd_cdf(y: np.ndarray | float, sigma: float, xi: float) -> np.ndarray | float:
    """GPD CDF for exceedance y>=0 with loc=0.

    Args:
        y: Exceedance values (y = x - u), must be >= 0 (dimensionless)
        sigma: Scale parameter (>0) (dimensionless)
        xi: Shape parameter (can be any real number) (dimensionless)

    Returns:
        CDF values at y (dimensionless)
    """
    y = np.asarray(y, float)
    if abs(xi) < 1e-12:
        return 1 - np.exp(-y / sigma)
    return 1 - (1 + xi * y / sigma) ** (-1.0 / xi)


def right_tail_ad_from_uniforms(u_sorted: np.ndarray) -> float:
    """Right-tail weighted Anderson-Darling statistic for uniform data.

    Computes the right-tail weighted AD statistic:
    A_R^2 = -n - (1/n) * sum_{i=1}^n (2i-1) * ln(1 - u_{n+1-i})
    where u_sorted are uniforms in ascending order.
    This emphasizes misfit in the right tail (large exceedances).

    Args:
        u_sorted: Uniform random variables sorted in ascending order (dimensionless).

    Returns:
        Right-tail Anderson-Darling statistic (dimensionless).
    """
    u = np.asarray(u_sorted, dtype=float)
    n = u.size
    if n < 1:
        return np.nan
    i = np.arange(1, n + 1)
    s_tail = np.sum((2 * i - 1) * np.log(1.0 - u[::-1]))
    a_r2 = -n - s_tail / n
    return float(a_r2)


def bootstrap_pvalue_for_ad(
    observed_stat: float,
    n: int,
    sigma_hat: float,
    xi_hat: float,
    nboot: int = 2000,
    random_seed: int = 123,
) -> float:
    """Parametric bootstrap p-value for right-tail Anderson-Darling statistic.

    Simulates n exceedances from GPD(sigma_hat, xi_hat) nboot times,
    refits using L-moments, computes AD statistic on transformed uniforms
    for each simulation, and returns the proportion of simulated statistics
    >= observed statistic.

    Args:
        observed_stat: Observed Anderson-Darling statistic (dimensionless).
        n: Number of exceedances to simulate (dimensionless).
        sigma_hat: Fitted GPD scale parameter (dimensionless).
        xi_hat: Fitted GPD shape parameter (dimensionless).
        nboot: Number of bootstrap samples (dimensionless).
        random_seed: Random seed for reproducibility (dimensionless).

    Returns:
        Bootstrap p-value as proportion of simulated stats >= observed (dimensionless).
    """
    rng = np.random.default_rng(random_seed)
    sim_stats = np.empty(nboot, dtype=float)
    for k in range(nboot):
        # Generate exceedances from the fitted GPD
        ysim = genpareto.rvs(c=xi_hat, scale=sigma_hat, size=n, random_state=rng)

        # Refit using L-moments (consistent with main fitting approach)
        try:
            sigma_boot, xi_boot = fit_gpd_lmoments(ysim)
            # Transform using refitted parameters
            u_sim = gpd_cdf(ysim, sigma_boot, xi_boot)
            sim_stats[k] = right_tail_ad_from_uniforms(np.sort(u_sim))
        except (ValueError, RuntimeError):
            # If refit fails, use original parameters (fallback)
            u_sim = gpd_cdf(ysim, sigma_hat, xi_hat)
            sim_stats[k] = right_tail_ad_from_uniforms(np.sort(u_sim))

    pval = np.mean(sim_stats >= observed_stat)
    return float(pval)


def mean_residual_life(data: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Calculate mean residual life for threshold values.

    Computes the mean excess over threshold u for each threshold in u_grid.
    This is used for mean residual life plots to assess threshold selection.

    Args:
        data: Array of observed values (dimensionless).
        u_grid: Array of threshold values to evaluate (dimensionless).

    Returns:
        Array of mean residual life values, one for each threshold (dimensionless).
        Returns NaN for thresholds with no exceedances.
    """
    x = np.asarray(data, dtype=float)
    return np.array(
        [(x[x > u] - u).mean() if np.sum(x > u) > 0 else np.nan for u in u_grid]
    )


def gpd_return_level(
    u: float, sigma: float, xi: float, lambda_per_year: float, T: np.ndarray | float
) -> np.ndarray | float:
    """Calculate return levels for given return periods using GPD parameters.

    Computes return levels using the Generalized Pareto Distribution fitted above
    threshold u. For shape parameter xi ≈ 0, uses exponential limit formula.

    Args:
        u: Threshold value (dimensionless).
        sigma: GPD scale parameter (>0) (dimensionless).
        xi: GPD shape parameter (dimensionless).
        lambda_per_year: Average number of exceedances per year (1/year).
        T: Return period(s) in years (years).

    Returns:
        Return level(s) corresponding to the given return period(s) (dimensionless).
    """
    T = np.asarray(T, float)
    LT = lambda_per_year * T
    if abs(xi) < 1e-8:
        return u + sigma * np.log(LT)
    return u + (sigma / xi) * (LT**xi - 1.0)


def gpd_pot_ad_auto(
    series: pd.Series,
    quantile_start: float = 0.80,
    quantile_end: float = 0.99,
    quantile_step: float = 0.01,
    min_exceed: int = 30,
    nboot: int = 2000,
    return_periods: np.ndarray | None = None,
    mrl_grid_q: np.ndarray | None = None,
    mrl_top_fraction: float = 0.75,
    random_seed: int = 123,
) -> dict:
    """Automated GPD-POT analysis with threshold selection using Anderson-Darling test.

    Performs automated Generalized Pareto Distribution Peaks-Over-Threshold analysis
    by scanning multiple threshold candidates and selecting the optimal threshold based
    on Anderson-Darling goodness-of-fit testing with bootstrap p-values. GPD parameters
    are estimated using the method of L-moments for robustness.

    Args:
        series: Time series data with DatetimeIndex (dimensionless).
        quantile_start: Starting quantile for threshold scan (dimensionless).
        quantile_end: Ending quantile for threshold scan (dimensionless).
        quantile_step: Step size between quantiles (dimensionless).
        min_exceed: Minimum number of exceedances required for fitting (dimensionless).
        nboot: Number of bootstrap samples for p-value calculation (dimensionless).
        return_periods: Array of return periods in years for return level calculation (years).
        mrl_grid_q: Quantiles for mean residual life plot baseline (dimensionless).
        mrl_top_fraction: Fraction of top quantiles to use for linear fit in mean residual life plot (dimensionless).
        random_seed: Random seed for reproducibility (dimensionless).

    Returns:
        Dictionary containing daily maxima, diagnostics DataFrame, chosen parameters,
        and return level table.

    Raises:
        TypeError: If series does not have a DatetimeIndex.
    """
    if return_periods is None:
        return_periods = np.array(
            [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000, 10000], float
        )

    if mrl_grid_q is None:
        mrl_grid_q = np.linspace(0.70, 0.995, 80)

    # ---- Prepare data ----
    s = series.dropna().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex.")

    daily_max = s.resample("D").max().dropna()
    total_days = (daily_max.index.max() - daily_max.index.min()).days + 1
    years = total_days / 365.25

    # ---- Threshold candidates ----
    q_grid = np.arange(quantile_start, quantile_end + 1e-9, quantile_step)
    u_candidates = np.quantile(daily_max.values, q_grid)

    # ---- MRL baseline ----
    mrl_grid_u = np.quantile(daily_max.values, mrl_grid_q)
    mrl_vals = mean_residual_life(daily_max.values, mrl_grid_u)

    top_idx = int(len(mrl_vals) * mrl_top_fraction)
    if np.sum(~np.isnan(mrl_vals[top_idx:])) >= 3:
        a_lin, b_lin = np.polyfit(mrl_grid_u[top_idx:], mrl_vals[top_idx:], 1)
    else:
        a_lin, b_lin = 0.0, 0.0

    diagnostics = []

    # ---- Main scan ----
    for u in u_candidates:
        exceed = daily_max[daily_max > u]
        n_exc = exceed.size

        if n_exc < min_exceed:
            diagnostics.append(
                (u, np.nan, np.nan, np.nan, n_exc, np.nan, np.nan, np.nan, np.nan)
            )
            continue

        y = exceed.values - u

        try:
            # Calculate L-moments for diagnostics
            L1, L2, L3 = calculate_lmoments(y)
            tau3 = L3 / L2  # L-skewness

            sigma, xi = fit_gpd_lmoments(y)
        except Exception as e:
            print(
                f"Exception in fit_gpd_lmoments(y) for threshold u={u}: {type(e).__name__}: {e}"
            )
            diagnostics.append(
                (u, np.nan, np.nan, np.nan, n_exc, np.nan, np.nan, np.nan, np.nan)
            )
            continue

        u_vals = gpd_cdf(y, sigma, xi)
        A_R2 = right_tail_ad_from_uniforms(np.sort(u_vals))

        p_ad = bootstrap_pvalue_for_ad(A_R2, n_exc, sigma, xi, nboot, random_seed)

        ks_p = np.nan
        if n_exc >= 20:
            _, ks_p = kstest(y, "genpareto", args=(xi, 0.0, sigma))

        idx = np.argmin(np.abs(mrl_grid_u - u))
        mrl_err = (
            np.nan
            if np.isnan(mrl_vals[idx])
            else abs(mrl_vals[idx] - (a_lin * mrl_grid_u[idx] + b_lin))
        )

        diagnostics.append((u, sigma, xi, tau3, n_exc, p_ad, ks_p, mrl_err, A_R2))

    # ---- Diagnostics to DataFrame ----
    diag_df = (
        pd.DataFrame(
            diagnostics,
            columns=np.array(
                ["u", "sigma", "xi", "tau3", "n_exc", "p_ad", "ks_p", "mrl_err", "A_R2"]
            ),
        )
        .sort_values("u")
        .reset_index(drop=True)
    )

    diag_df["xi_step"] = diag_df["xi"].diff().abs().bfill()
    diag_df["sigma_step"] = diag_df["sigma"].diff().abs().bfill()
    # ---- Pick threshold = highest AD p-value ----
    valid = diag_df.dropna(subset=["p_ad"])

    # If no valid thresholds (e.g., too few exceedances / bootstrap failed),
    # return a safe structure with NaN RLs so upstream code can handle it.
    if valid.empty:
        rl_table = pd.DataFrame(
            {
                "T_years": return_periods.astype(int),
                "GPD_POT_RL": np.full(len(return_periods), np.nan),
            }
        )
        chosen = {
            "u": np.nan,
            "sigma": np.nan,
            "xi": np.nan,
            "p_ad": np.nan,
            "n_exc": 0,
            "lambda_per_year": np.nan,
            "pct": np.nan,
        }
        # optional informative print
        print(
            "No valid thresholds found in gpd_pot_ad_auto — returning NaN return levels."
        )
        return {
            "daily_max": daily_max,
            "years": years,
            "diag_df": diag_df,
            "chosen": chosen,
            "rl_table": rl_table,
        }

    # normal path: pick threshold with largest AD p-value
    best = valid.loc[valid["p_ad"].idxmax()]

    u_star = float(best["u"])
    sigma_star = float(best["sigma"])
    xi_star = float(best["xi"])
    p_star = float(best["p_ad"])
    n_star = int(best["n_exc"])

    lambda_per_year = n_star / years
    pct = (daily_max < u_star).mean() * 100

    # ---- Return levels ----
    RLs = gpd_return_level(u_star, sigma_star, xi_star, lambda_per_year, return_periods)
    rl_table = pd.DataFrame({"T_years": return_periods.astype(int), "GPD_POT_RL": RLs})
    print(
        "Automatic threshold selection for exceedances done (checked by Anderson-Darling goodness-of-fit parameter test, L-moments estimation)."
    )
    print(
        f"Chosen threshold corresponds to approximately the {pct:.2f}th percentile of daily maxima of discharge."
    )

    return {
        "daily_max": daily_max,
        "years": years,
        "diag_df": diag_df,
        "chosen": {
            "u": u_star,
            "sigma": sigma_star,
            "xi": xi_star,
            "p_ad": p_star,
            "n_exc": n_star,
            "lambda_per_year": lambda_per_year,
            "pct": pct,
        },
        "rl_table": rl_table,
        "mrl_grid_u": mrl_grid_u,
        "mrl_vals": mrl_vals,
    }


def plot_gpd_diagnostics(res: dict, figsize: tuple[int, int] = (16, 12)) -> plt.Figure:
    """Plot comprehensive GPD-POT diagnostics.

    Creates a multi-panel diagnostic plot showing:
    - L-skewness vs threshold
    - Shape parameter (xi) stability
    - Scale parameter (sigma) stability
    - Anderson-Darling p-value vs threshold
    - Number of exceedances vs threshold
    - Mean Residual Life plot
    - AD statistic vs threshold
    - KS test p-value vs threshold

    Args:
        res: Dictionary returned by gpd_pot_ad_auto containing diagnostics.
        figsize: Figure size as (width, height) in inches.

    Returns:
        Matplotlib Figure object with diagnostic plots.
    """
    diag = res["diag_df"]
    chosen = res["chosen"]
    u_star = chosen["u"]

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(
        "GPD-POT Diagnostic Plots (L-moments Estimation)",
        fontsize=16,
        fontweight="bold",
    )

    # 1. L-skewness vs threshold
    ax = axes[0, 0]
    ax.plot(diag["u"], diag["tau3"], "b.-", linewidth=1.5, markersize=3)
    ax.axvline(
        u_star,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Chosen: u={u_star:.2f}",
    )
    ax.set_xlabel("Threshold (u)", fontsize=10)
    ax.set_ylabel("L-skewness (τ₃)", fontsize=10)
    ax.set_title("L-skewness vs Threshold", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. Shape parameter (xi) vs threshold
    ax = axes[0, 1]
    ax.plot(diag["u"], diag["xi"], "g.-", linewidth=1.5, markersize=3)
    ax.axvline(u_star, color="red", linestyle="--", linewidth=2)
    ax.axhline(chosen["xi"], color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Threshold (u)", fontsize=10)
    ax.set_ylabel("Shape parameter (ξ)", fontsize=10)
    ax.set_title(
        f"Shape Parameter (ξ={chosen['xi']:.3f})", fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # 3. Scale parameter (sigma) vs threshold
    ax = axes[0, 2]
    ax.plot(diag["u"], diag["sigma"], "m.-", linewidth=1.5, markersize=3)
    ax.axvline(u_star, color="red", linestyle="--", linewidth=2)
    ax.axhline(chosen["sigma"], color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Threshold (u)", fontsize=10)
    ax.set_ylabel("Scale parameter (σ)", fontsize=10)
    ax.set_title(
        f"Scale Parameter (σ={chosen['sigma']:.3f})", fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # 4. AD p-value vs threshold
    ax = axes[1, 0]
    ax.plot(diag["u"], diag["p_ad"], "b.-", linewidth=1.5, markersize=3)
    ax.axvline(u_star, color="red", linestyle="--", linewidth=2)
    ax.axhline(
        0.05, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="α=0.05"
    )
    ax.set_xlabel("Threshold (u)", fontsize=10)
    ax.set_ylabel("AD p-value", fontsize=10)
    ax.set_title(
        f"Anderson-Darling p-value (p={chosen['p_ad']:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 5. Number of exceedances vs threshold
    ax = axes[1, 1]
    ax.plot(diag["u"], diag["n_exc"], "r.-", linewidth=1.5, markersize=3)
    ax.axvline(u_star, color="red", linestyle="--", linewidth=2)
    ax.axhline(chosen["n_exc"], color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Threshold (u)", fontsize=10)
    ax.set_ylabel("Number of Exceedances", fontsize=10)
    ax.set_title(f"Exceedances (n={chosen['n_exc']})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 6. Mean Residual Life plot
    ax = axes[1, 2]
    if "mrl_grid_u" in res and "mrl_vals" in res:
        ax.plot(res["mrl_grid_u"], res["mrl_vals"], "k.-", linewidth=1.5, markersize=3)
        ax.axvline(
            u_star,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Chosen: u={u_star:.2f}",
        )
        ax.set_xlabel("Threshold (u)", fontsize=10)
        ax.set_ylabel("Mean Excess", fontsize=10)
        ax.set_title("Mean Residual Life Plot", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # 7. AD statistic vs threshold
    ax = axes[2, 0]
    ax.plot(diag["u"], diag["A_R2"], "c.-", linewidth=1.5, markersize=3)
    ax.axvline(u_star, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Threshold (u)", fontsize=10)
    ax.set_ylabel("AD Statistic (A²ᴿ)", fontsize=10)
    ax.set_title("Anderson-Darling Statistic", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 8. KS test p-value vs threshold
    ax = axes[2, 1]
    valid_ks = diag[diag["ks_p"].notna()]
    if len(valid_ks) > 0:
        ax.plot(valid_ks["u"], valid_ks["ks_p"], "y.-", linewidth=1.5, markersize=3)
        ax.axvline(u_star, color="red", linestyle="--", linewidth=2)
        ax.axhline(
            0.05, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="α=0.05"
        )
        ax.set_xlabel("Threshold (u)", fontsize=10)
        ax.set_ylabel("KS p-value", fontsize=10)
        ax.set_title("Kolmogorov-Smirnov p-value", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # 9. Parameter stability (xi and sigma step changes)
    ax = axes[2, 2]
    ax2 = ax.twinx()
    l1 = ax.plot(
        diag["u"], diag["xi_step"], "g.-", linewidth=1.5, markersize=3, label="|Δξ|"
    )
    l2 = ax2.plot(
        diag["u"], diag["sigma_step"], "m.-", linewidth=1.5, markersize=3, label="|Δσ|"
    )
    ax.axvline(u_star, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Threshold (u)", fontsize=10)
    ax.set_ylabel("|Δξ| (step change)", fontsize=10, color="g")
    ax2.set_ylabel("|Δσ| (step change)", fontsize=10, color="m")
    ax.tick_params(axis="y", labelcolor="g")
    ax2.tick_params(axis="y", labelcolor="m")
    ax.set_title("Parameter Stability", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc="upper left")

    plt.tight_layout()
    return fig


def plot_qq_pp_density(res: dict, figsize: tuple[int, int] = (15, 5)) -> plt.Figure:
    """Plot QQ, PP, and Density diagnostic plots for chosen threshold.

    Args:
        res: Dictionary returned by gpd_pot_ad_auto containing diagnostics.
        figsize: Figure size as (width, height) in inches.

    Returns:
        Matplotlib Figure object with QQ, PP, and density plots.
    """
    chosen = res["chosen"]
    u_star = chosen["u"]
    sigma = chosen["sigma"]
    xi = chosen["xi"]

    # Get exceedances at chosen threshold
    daily_max = res["daily_max"]
    exceed = daily_max[daily_max > u_star]
    y = exceed.values - u_star
    y_sorted = np.sort(y)
    n = len(y)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"Goodness-of-Fit Diagnostics at Chosen Threshold (u={u_star:.2f})",
        fontsize=14,
        fontweight="bold",
    )

    # 1. QQ Plot (Quantile-Quantile)
    ax = axes[0]
    # Empirical quantiles
    empirical_probs = (np.arange(1, n + 1) - 0.5) / n
    empirical_quantiles = y_sorted

    # Theoretical quantiles from fitted GPD
    theoretical_quantiles = genpareto.ppf(empirical_probs, c=xi, scale=sigma)

    ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6, s=20)

    # 45-degree reference line
    min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
    max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 line"
    )

    ax.set_xlabel("Theoretical Quantiles (GPD)", fontsize=10)
    ax.set_ylabel("Empirical Quantiles", fontsize=10)
    ax.set_title("Quantile-Quantile (QQ) Plot", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. PP Plot (Probability-Probability)
    ax = axes[1]
    # Empirical CDF
    empirical_cdf = empirical_probs

    # Theoretical CDF from fitted GPD
    theoretical_cdf = gpd_cdf(y_sorted, sigma, xi)

    ax.scatter(theoretical_cdf, empirical_cdf, alpha=0.6, s=20, color="green")
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="1:1 line")

    ax.set_xlabel("Theoretical CDF (GPD)", fontsize=10)
    ax.set_ylabel("Empirical CDF", fontsize=10)
    ax.set_title("Probability-Probability (PP) Plot", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3. Density Plot (Histogram vs fitted density)
    ax = axes[2]
    # Histogram
    ax.hist(
        y,
        bins=30,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        label="Empirical",
    )

    # Fitted GPD density
    y_range = np.linspace(0, y.max(), 200)
    fitted_density = genpareto.pdf(y_range, c=xi, scale=sigma)
    ax.plot(
        y_range,
        fitted_density,
        "r-",
        linewidth=2,
        label=f"GPD(σ={sigma:.2f}, ξ={xi:.3f})",
    )

    ax.set_xlabel("Exceedance (y = x - u)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Density Plot", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_return_level(
    res: dict, confidence_level: float = 0.95, figsize: tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot return level with confidence intervals using delta method.

    Args:
        res: Dictionary returned by gpd_pot_ad_auto containing diagnostics.
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95%).
        figsize: Figure size as (width, height) in inches.

    Returns:
        Matplotlib Figure object with return level plots.
    """
    chosen = res["chosen"]
    u_star = chosen["u"]
    sigma = chosen["sigma"]
    xi = chosen["xi"]
    lambda_per_year = chosen["lambda_per_year"]
    n_exc = chosen["n_exc"]

    # Extended return periods for plotting
    T_plot = np.logspace(np.log10(2), np.log10(10000), 100)

    # Return levels
    RLs = gpd_return_level(u_star, sigma, xi, lambda_per_year, T_plot)

    # Simple confidence intervals using delta method approximation
    # Standard errors (simplified, assumes large sample)
    z_alpha = stats.norm.ppf((1 + confidence_level) / 2)

    # Approximate standard error (simplified)
    # More rigorous would use bootstrap or profile likelihood
    se_multiplier = (
        z_alpha * np.sqrt(1 / n_exc) * (1 + 0.5 * xi * np.log(lambda_per_year * T_plot))
    )
    RL_lower = RLs * (1 - se_multiplier)
    RL_upper = RLs * (1 + se_multiplier)

    # Observed data for plotting
    daily_max = res["daily_max"]
    exceed = daily_max[daily_max > u_star]
    y = exceed.values - u_star

    # Empirical return periods
    y_sorted = np.sort(y)[::-1]  # Descending order
    empirical_RP = len(daily_max) / (365.25 * np.arange(1, len(y_sorted) + 1))
    empirical_levels = y_sorted + u_star

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Return Level Analysis", fontsize=14, fontweight="bold")

    # 1. Return level plot (log scale)
    ax = axes[0]
    ax.plot(T_plot, RLs, "b-", linewidth=2, label="Fitted GPD")
    ax.fill_between(
        T_plot,
        RL_lower,
        RL_upper,
        alpha=0.3,
        color="blue",
        label=f"{int(confidence_level * 100)}% CI",
    )
    ax.scatter(
        empirical_RP,
        empirical_levels,
        s=30,
        color="red",
        alpha=0.6,
        label="Empirical",
        zorder=5,
    )
    ax.axhline(
        u_star,
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Threshold u={u_star:.2f}",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Return Period (years)", fontsize=10)
    ax.set_ylabel("Return Level", fontsize=10)
    ax.set_title("Return Level Plot (Log Scale)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # 2. Return level plot (linear scale for shorter periods)
    ax = axes[1]
    T_short = T_plot[T_plot <= 200]
    RLs_short = RLs[T_plot <= 200]
    RL_lower_short = RL_lower[T_plot <= 200]
    RL_upper_short = RL_upper[T_plot <= 200]

    ax.plot(T_short, RLs_short, "b-", linewidth=2, label="Fitted GPD")
    ax.fill_between(
        T_short,
        RL_lower_short,
        RL_upper_short,
        alpha=0.3,
        color="blue",
        label=f"{int(confidence_level * 100)}% CI",
    )

    # Plot empirical points in range
    mask = empirical_RP <= 200
    ax.scatter(
        empirical_RP[mask],
        empirical_levels[mask],
        s=30,
        color="red",
        alpha=0.6,
        label="Empirical",
        zorder=5,
    )
    ax.axhline(
        u_star,
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Threshold u={u_star:.2f}",
    )

    ax.set_xlabel("Return Period (years)", fontsize=10)
    ax.set_ylabel("Return Level", fontsize=10)
    ax.set_title(
        "Return Level Plot (Linear Scale, T≤200)", fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_lmoment_ratio_diagram(
    res: dict, figsize: tuple[int, int] = (8, 8)
) -> plt.Figure:
    """Plot L-moment ratio diagram showing L-skewness vs L-kurtosis.

    Args:
        res: Dictionary returned by gpd_pot_ad_auto containing diagnostics.
        figsize: Figure size as (width, height) in inches.

    Returns:
        Matplotlib Figure object with L-moment ratio diagram.
    """
    diag = res["diag_df"]
    chosen = res["chosen"]
    u_star = chosen["u"]

    # Calculate L-kurtosis (tau4) for each threshold
    tau4_list = []
    for _, row in diag.iterrows():
        u = row["u"]
        daily_max = res["daily_max"]
        exceed = daily_max[daily_max > u]
        if len(exceed) >= 6:
            y = exceed.values - u
            try:
                # Calculate L4 for L-kurtosis
                x = np.sort(y)
                n = len(x)
                j = np.arange(n)
                b0 = np.mean(x)
                b1 = np.sum(x * j) / (n * (n - 1))
                b2 = np.sum(x * j * (j - 1)) / (n * (n - 1) * (n - 2))
                b3 = np.sum(x * j * (j - 1) * (j - 2)) / (
                    n * (n - 1) * (n - 2) * (n - 3)
                )

                L2 = 2.0 * b1 - b0
                L4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0
                tau4 = L4 / L2 if L2 > 0 else np.nan
                tau4_list.append(tau4)
            except:
                tau4_list.append(np.nan)
        else:
            tau4_list.append(np.nan)

    diag["tau4"] = tau4_list

    # GPD theoretical curve
    xi_range = np.linspace(-0.4, 0.4, 100)
    tau3_theory = (1.0 - xi_range) / (3.0 + xi_range)
    tau4_theory = (1.0 + 5.0 * xi_range) / (5.0 + xi_range)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot theoretical GPD curve
    ax.plot(
        tau3_theory,
        tau4_theory,
        "b-",
        linewidth=2,
        label="GPD Theoretical Curve",
        zorder=1,
    )

    # Plot observed L-moment ratios
    valid = diag.dropna(subset=["tau3", "tau4"])
    scatter = ax.scatter(
        valid["tau3"],
        valid["tau4"],
        c=valid["u"],
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        label="Observed (colored by threshold)",
        zorder=2,
    )

    # Highlight chosen threshold
    chosen_row = diag[diag["u"] == u_star].iloc[0]
    if not np.isnan(chosen_row["tau4"]):
        ax.scatter(
            chosen_row["tau3"],
            chosen_row["tau4"],
            s=200,
            color="red",
            marker="*",
            edgecolors="black",
            linewidth=2,
            label=f"Chosen (u={u_star:.2f})",
            zorder=3,
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Threshold (u)", fontsize=10)

    ax.set_xlabel("L-skewness (τ₃)", fontsize=11)
    ax.set_ylabel("L-kurtosis (τ₄)", fontsize=11)
    ax.set_title("L-moment Ratio Diagram", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_threshold_stability(
    res: dict,
    return_periods: list[int] = [10, 50, 100, 500],
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot return level stability across different thresholds.

    Args:
        res: Dictionary returned by gpd_pot_ad_auto containing diagnostics.
        return_periods: List of return periods to plot.
        figsize: Figure size as (width, height) in inches.

    Returns:
        Matplotlib Figure object showing threshold stability.
    """
    diag = res["diag_df"]
    chosen = res["chosen"]
    u_star = chosen["u"]
    years = res["years"]
    daily_max = res["daily_max"]

    # Calculate return levels for each threshold
    return_levels = {T: [] for T in return_periods}

    for _, row in diag.iterrows():
        if not np.isnan(row["sigma"]) and not np.isnan(row["xi"]):
            u = row["u"]
            sigma = row["sigma"]
            xi = row["xi"]
            n_exc = row["n_exc"]
            lambda_py = n_exc / years

            for T in return_periods:
                rl = gpd_return_level(u, sigma, xi, lambda_py, float(T))
                return_levels[T].append(rl)
        else:
            for T in return_periods:
                return_levels[T].append(np.nan)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Threshold Stability: Return Levels vs Threshold",
        fontsize=14,
        fontweight="bold",
    )

    colors = ["blue", "green", "red", "purple"]

    for idx, (T, color) in enumerate(zip(return_periods, colors)):
        ax = axes[idx // 2, idx % 2]

        diag[f"RL_{T}"] = return_levels[T]
        valid = diag.dropna(subset=[f"RL_{T}"])

        ax.plot(
            valid["u"], valid[f"RL_{T}"], ".-", color=color, linewidth=1.5, markersize=4
        )
        ax.axvline(
            u_star,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Chosen: u={u_star:.2f}",
        )

        # Add chosen return level as horizontal line
        chosen_rl = gpd_return_level(
            u_star, chosen["sigma"], chosen["xi"], chosen["lambda_per_year"], float(T)
        )
        ax.axhline(
            chosen_rl,
            color="red",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label=f"Chosen RL={chosen_rl:.2f}",
        )

        ax.set_xlabel("Threshold (u)", fontsize=10)
        ax.set_ylabel(f"{T}-year Return Level", fontsize=10)
        ax.set_title(f"Return Period: {T} years", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_dispersion_index(res: dict, figsize: tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot dispersion index to test Poisson assumption for exceedances.

    The dispersion index is the variance-to-mean ratio of inter-exceedance times.
    For a Poisson process, this should be approximately 1.

    Args:
        res: Dictionary returned by gpd_pot_ad_auto containing diagnostics.
        figsize: Figure size as (width, height) in inches.

    Returns:
        Matplotlib Figure object with dispersion index diagnostics.
    """
    diag = res["diag_df"]
    chosen = res["chosen"]
    u_star = chosen["u"]
    daily_max = res["daily_max"]

    # Calculate dispersion index for each threshold
    dispersion_indices = []
    inter_arrival_means = []
    inter_arrival_vars = []

    for _, row in diag.iterrows():
        u = row["u"]
        if row["n_exc"] >= 10:
            # Get exceedance times
            exceed_times = daily_max[daily_max > u].index
            if len(exceed_times) >= 2:
                # Calculate inter-arrival times (in days)
                inter_arrivals = (
                    np.diff(exceed_times).astype("timedelta64[D]").astype(float)
                )

                if len(inter_arrivals) > 1:
                    mean_ia = np.mean(inter_arrivals)
                    var_ia = np.var(inter_arrivals, ddof=1)
                    dispersion_idx = var_ia / mean_ia if mean_ia > 0 else np.nan

                    dispersion_indices.append(dispersion_idx)
                    inter_arrival_means.append(mean_ia)
                    inter_arrival_vars.append(var_ia)
                else:
                    dispersion_indices.append(np.nan)
                    inter_arrival_means.append(np.nan)
                    inter_arrival_vars.append(np.nan)
            else:
                dispersion_indices.append(np.nan)
                inter_arrival_means.append(np.nan)
                inter_arrival_vars.append(np.nan)
        else:
            dispersion_indices.append(np.nan)
            inter_arrival_means.append(np.nan)
            inter_arrival_vars.append(np.nan)

    diag["dispersion_index"] = dispersion_indices
    diag["inter_arrival_mean"] = inter_arrival_means
    diag["inter_arrival_var"] = inter_arrival_vars

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        "Dispersion Index: Testing Poisson Assumption", fontsize=14, fontweight="bold"
    )

    # 1. Dispersion index vs threshold
    ax = axes[0]
    valid = diag.dropna(subset=["dispersion_index"])
    ax.plot(valid["u"], valid["dispersion_index"], "b.-", linewidth=1.5, markersize=4)
    ax.axvline(
        u_star,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Chosen: u={u_star:.2f}",
    )
    ax.axhline(
        1.0,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Poisson (DI=1)",
    )

    # Add shaded region for acceptable range
    ax.axhspan(0.8, 1.2, alpha=0.2, color="green", label="Acceptable range")

    ax.set_xlabel("Threshold (u)", fontsize=10)
    ax.set_ylabel("Dispersion Index (Var/Mean)", fontsize=10)
    ax.set_title("Dispersion Index vs Threshold", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. Variance vs Mean scatter
    ax = axes[1]
    valid = diag.dropna(subset=["inter_arrival_mean", "inter_arrival_var"])
    scatter = ax.scatter(
        valid["inter_arrival_mean"],
        valid["inter_arrival_var"],
        c=valid["u"],
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add 1:1 line (Poisson expectation)
    if len(valid) > 0:
        max_val = max(
            valid["inter_arrival_mean"].max(), valid["inter_arrival_var"].max()
        )
        ax.plot(
            [0, max_val], [0, max_val], "r--", linewidth=2, label="Poisson (Var=Mean)"
        )

    # Highlight chosen threshold
    chosen_row = diag[diag["u"] == u_star].iloc[0]
    if not np.isnan(chosen_row["dispersion_index"]):
        ax.scatter(
            chosen_row["inter_arrival_mean"],
            chosen_row["inter_arrival_var"],
            s=200,
            color="red",
            marker="*",
            edgecolors="black",
            linewidth=2,
            label=f"Chosen (u={u_star:.2f})",
            zorder=5,
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Threshold (u)", fontsize=10)

    ax.set_xlabel("Mean Inter-arrival Time (days)", fontsize=10)
    ax.set_ylabel("Variance of Inter-arrival Time (days²)", fontsize=10)
    ax.set_title("Variance-Mean Relationship", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_all_diagnostics(
    res: dict,
    output_dir: str = ".",
    river_id: str | None = None,
    write_figures: bool = True,
) -> dict[str, str]:
    """Generate and save all diagnostic plots for GPD-POT analysis.

    Creates comprehensive diagnostic plots including main diagnostics,
    goodness-of-fit tests, return level analysis, L-moment ratios,
    threshold stability, and dispersion index tests.

    Args:
        res: Dictionary returned by gpd_pot_ad_auto containing diagnostics.
        output_dir: Directory to save plot files (default: current directory).
        river_id: Optional river identifier to include in plot filenames.
        write_figures: Whether to save plots to disk (default: True).

    Returns:
        Dictionary mapping plot names to file paths (empty if write_figures=False).
    """
    import os

    if not write_figures:
        print("write_figures=False, skipping diagnostic plot generation.")
        return {}

    # Create filename prefix with river_id if provided
    prefix = f"river_{river_id}_" if river_id else ""

    plot_files = {}

    # 1. Main diagnostic plots
    print("Generating main diagnostic plots...")
    fig1 = plot_gpd_diagnostics(res)
    path1 = os.path.join(output_dir, f"{prefix}gpd_pot_diagnostics.png")
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    plot_files["main_diagnostics"] = path1
    plt.close(fig1)

    # 2. QQ, PP, and Density plots
    print("Generating QQ, PP, and density plots...")
    fig2 = plot_qq_pp_density(res)
    path2 = os.path.join(output_dir, f"{prefix}gpd_pot_qq_pp_density.png")
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plot_files["qq_pp_density"] = path2
    plt.close(fig2)

    # 3. Return level plots
    print("Generating return level plots...")
    fig3 = plot_return_level(res)
    path3 = os.path.join(output_dir, f"{prefix}gpd_pot_return_levels.png")
    fig3.savefig(path3, dpi=300, bbox_inches="tight")
    plot_files["return_levels"] = path3
    plt.close(fig3)

    # 4. L-moment ratio diagram
    print("Generating L-moment ratio diagram...")
    fig4 = plot_lmoment_ratio_diagram(res)
    path4 = os.path.join(output_dir, f"{prefix}gpd_pot_lmoment_diagram.png")
    fig4.savefig(path4, dpi=300, bbox_inches="tight")
    plot_files["lmoment_diagram"] = path4
    plt.close(fig4)

    # 5. Threshold stability
    print("Generating threshold stability plots...")
    fig5 = plot_threshold_stability(res)
    path5 = os.path.join(output_dir, f"{prefix}gpd_pot_threshold_stability.png")
    fig5.savefig(path5, dpi=300, bbox_inches="tight")
    plot_files["threshold_stability"] = path5
    plt.close(fig5)

    # 6. Dispersion index
    print("Generating dispersion index plots...")
    fig6 = plot_dispersion_index(res)
    path6 = os.path.join(output_dir, f"{prefix}gpd_pot_dispersion_index.png")
    fig6.savefig(path6, dpi=300, bbox_inches="tight")
    plot_files["dispersion_index"] = path6
    plt.close(fig6)

    print(f"\nAll {len(plot_files)} diagnostic plots saved to: {output_dir}")
    return plot_files


# new distribution function for return period calculations
# based on https://doi.org/10.1002/2016WR019426
# automatic threshold selection and gpd distribution
def assign_return_periods(
    rivers: gpd.GeoDataFrame,
    discharge_dataframe: pd.DataFrame,
    return_periods: list[int | float],
    prefix: str = "Q",
    min_exceed: int = 30,
    nboot: int = 2000,
    model_root: Path | None = None,
    write_figures: bool = False,
) -> gpd.GeoDataFrame:
    """Assign return periods to rivers using GPD-POT analysis.

    Uses Generalized Pareto Distribution Peaks-Over-Threshold method with:
        - Daily maxima resampling from input time series
        - GPD-POT exceedance model fitting above threshold
        - Anderson-Darling bootstrap p-value threshold selection
        - Return level estimation for specified return periods

    Args:
        rivers: GeoDataFrame with river IDs as index that must match columns in discharge_dataframe.
        discharge_dataframe: Time series DataFrame with datetime index containing discharge data for all rivers (m³/s).
        return_periods: List of return periods in years to compute return levels for.
        prefix: Column prefix for output return level columns. Defaults to "Q".
        min_exceed: Minimum number of exceedances required for reliable GPD fit. Defaults to 30.
        nboot: Number of bootstrap samples for Anderson-Darling threshold selection. Defaults to 2000.
        model_root: Path to model root directory (for saving diagnostic plots). Optional.
        write_figures: Whether to save diagnostic plots. Defaults to False.

    Returns:
        Updated rivers GeoDataFrame with return level columns added (m³/s).

    Raises:
        TypeError: If discharge series does not have a DatetimeIndex.
        ValueError: If return periods contain non-positive values.
    """
    assert isinstance(return_periods, list)
    if not all((isinstance(T, (int, float)) and T > 0) for T in return_periods):
        raise ValueError("All return periods must be positive numbers (years > 0).")
    return_periods_arr = np.asarray(return_periods, float)

    for idx in tqdm(rivers.index, total=len(rivers), desc="GPD-POT Return Periods"):
        discharge = discharge_dataframe[idx].dropna()

        # If all values are zero, assign zeros
        if (discharge < 1e-10).all():
            print(f"Discharge all zero for river {idx}, assigning zeros.")
            for T in return_periods:
                rivers.loc[idx, f"{prefix}_{T}"] = 0.0
            continue

        if not isinstance(discharge.index, pd.DatetimeIndex):
            raise TypeError(
                f"Discharge series for river {idx} must have a DatetimeIndex."
            )

        # Try GPD-POT method
        try:
            result = gpd_pot_ad_auto(
                series=discharge,
                return_periods=return_periods_arr,
                min_exceed=min_exceed,
                nboot=nboot,
            )
            print(f"GPD-POT analysis completed for river {idx}.")
        except Exception as e:
            # If the threshold-selection routine raises, handle gracefully:
            # assign NaNs for all requested return periods and continue.
            warnings.warn(
                f"GPD-POT analysis raised an exception for river {idx}: {type(e).__name__}: {e}. "
                "Assigning NaN for return levels.",
                UserWarning,
            )
            rl_values = np.array([np.nan] * len(return_periods_arr))
        else:
            # defensive extraction of rl values
            try:
                rl_values = (
                    result.get("rl_table", pd.DataFrame())
                    .get("GPD_POT_RL", pd.Series([np.nan] * len(return_periods_arr)))
                    .values
                )
            except Exception:
                rl_values = np.array([np.nan] * len(return_periods_arr))

            # Generate diagnostic plots if requested
            if write_figures and model_root is not None:
                try:
                    plot_all_diagnostics(
                        result,
                        output_dir=str(model_root),
                        river_id=str(idx),
                        write_figures=True,
                    )
                except Exception as plot_err:
                    warnings.warn(
                        f"Failed to generate diagnostic plots for river {idx}: {plot_err}",
                        UserWarning,
                    )

        # Assign and apply warnings/guards
        MAX_Q = 400_000
        for T, rl in zip(return_periods, rl_values):
            # safe conversion to float - handle NaN / weird objects
            try:
                discharge_value = float(rl)
            except Exception:
                discharge_value = float("nan")

            # If non-finite, store NaN
            if not np.isfinite(discharge_value):
                rivers.loc[idx, f"{prefix}_{T}"] = np.nan
                continue

            # Cap values to a safe maximum rather than raising on extremely large extrapolations
            if discharge_value > MAX_Q:
                warnings.warn(
                    f"Computed return level for T={T} years for river {idx} ({discharge_value:.3g}) "
                    f"exceeds maximum cap {MAX_Q}. Capping to {MAX_Q}.",
                    UserWarning,
                )
                discharge_value = float(MAX_Q)

            # Finally assign the (sanitized) value
            rivers.loc[idx, f"{prefix}_{T}"] = discharge_value

    return rivers


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
