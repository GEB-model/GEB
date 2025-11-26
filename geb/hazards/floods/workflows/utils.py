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
from scipy.stats import genpareto, kstest
from shapely.geometry import LineString, Point
from tqdm import tqdm

from geb.types import ArrayFloat32, ArrayInt64


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
        # read subgrid elevation
        surface_elevation: xr.DataArray = xr.open_dataarray(
            model_root / "subgrid" / "dep_subgrid.tif"
        ).sel(band=1)

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
        process: subprocess.Popen[str] = subprocess.Popen(
            args=cmd,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Continuously read lines from stdout and stderr
        for line in iter(
            lambda: process.stdout.readline() or process.stderr.readline(), ""
        ):
            print(line.rstrip())
            log.write(line)
            log.flush()

        process.stdout.close()
        process.stderr.close()
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
                os.getenv("SLURM_CPUS_PER_TASK")
                or os.getenv("SLURM_CPUS_ON_NODE")
                or os.cpu_count()
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
    """
    river = rivers.loc[river_ID]
    if river["represented_in_grid"]:
        xy = _get_xy(river, up_to_downstream=True)
        if xy is not None:
            return [xy]
        else:
            print(
                f"Warning: No valid xy found for river {river_ID}. Skipping this river."
            )
            return []  # If no valid xy found, return empty list

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
                print(
                    f"Warning: No valid xy found for river {river_ID}. Skipping this river."
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
    """
    xs: list[int] = []
    ys: list[int] = []
    for points in points_per_river:
        xs.extend([p[0] for p in points])
        ys.extend([p[1] for p in points])

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
        index=river_IDs,
        columns=["river_width_alpha", "river_width_beta"],
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
def fit_gpd_mle(exceedances: np.ndarray) -> tuple[float, float]:
    """Fit GPD to positive exceedances using Maximum Likelihood Estimation.

    Fits a Generalized Pareto Distribution to exceedances (y = x - u) using
    scipy's genpareto.fit with location parameter fixed at 0.

    Args:
        exceedances: Positive exceedance values above threshold (dimensionless).

    Returns:
        Tuple of (sigma, xi) where sigma is the scale parameter and xi is
        the shape parameter of the fitted GPD.

    Raises:
        ValueError: If fewer than 6 exceedances provided for reliable fit.
    """
    y = np.asarray(exceedances, dtype=float)
    if len(y) < 6:
        raise ValueError("Too few exceedances for reliable fit")
    c_hat, loc_hat, scale_hat = genpareto.fit(y, floc=0.0)
    return float(scale_hat), float(c_hat)  # sigma, xi


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
    computes AD statistic on transformed uniforms for each simulation,
    and returns the proportion of simulated statistics >= observed statistic.

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
        ysim = genpareto.rvs(c=xi_hat, scale=sigma_hat, size=n, random_state=rng)
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
    on Anderson-Darling goodness-of-fit testing with bootstrap p-values.

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
                (u, np.nan, np.nan, n_exc, np.nan, np.nan, np.nan, np.nan)
            )
            continue

        y = exceed.values - u

        try:
            sigma, xi = fit_gpd_mle(y)
        except Exception as e:
            print(
                f"Exception in fit_gpd_mle(y) for threshold u={u}: {type(e).__name__}: {e}"
            )
            diagnostics.append(
                (u, np.nan, np.nan, n_exc, np.nan, np.nan, np.nan, np.nan)
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

        diagnostics.append((u, sigma, xi, n_exc, p_ad, ks_p, mrl_err, A_R2))

    # ---- Diagnostics to DataFrame ----
    diag_df = (
        pd.DataFrame(
            diagnostics,
            columns=["u", "sigma", "xi", "n_exc", "p_ad", "ks_p", "mrl_err", "A_R2"],
        )
        .sort_values("u")
        .reset_index(drop=True)
    )

    diag_df["xi_step"] = diag_df["xi"].diff().abs().bfill()
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
        "Automatic threshold selection for exceedances done (checked by Anderson-Darling goodness-of-fit parameter test)."
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
    }


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
