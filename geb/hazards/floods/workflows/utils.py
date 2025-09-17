"""Utility functions for flood hazard workflows."""

import os
import platform
import subprocess
from datetime import datetime
from os.path import isfile, join
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel, utils
from pyextremes import EVA
from shapely.geometry import LineString, Point
from tqdm import tqdm


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
    model_root: Path, simulation_root: Path, method: str, minimum_flood_depth: float
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

    Returns:
        The maximum flood depth downscaled to subgrid resolution.

    Raises:
        ValueError: If an unknown method is provided.
    """

    # Read SFINCS model, config and results
    model: SfincsModel = SfincsModel(
        str(simulation_root),
        mode="r",
    )
    model.read_config()
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
            water_surface_elevation: xr.DataArray = model.results["zsmax"].max(
                dim="timemax"
            )
        elif method == "final":
            # get water surface elevation at the final time step (with respect to sea level)
            water_surface_elevation: xr.DataArray = model.results["zs"].isel(timemax=-1)
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
        flood_depth_m = flood_depth_m.rio.write_crs(model.crs)

    else:
        surface_elevation = model.grid.get("dep")
        if method == "max":
            flood_depth_m = model.results["hmax"].max(dim="timemax")
        elif method == "final":
            flood_depth_m_zs = model.results["zs"].isel(time=-1) - surface_elevation
            flood_depth_m_zs = flood_depth_m_zs.compute()

            flood_depth_m = model.results["h"].isel(time=-1)
            flood_depth_m = flood_depth_m.compute()

        else:
            raise ValueError(f"Unknown method: {method}")

    print(
        f"Maximum flood depth: {float(flood_depth_m.max().values):.2f} m, "
        f"Mean flood depth: {float(flood_depth_m.mean().values):.2f} m"
    )

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
        path: The working directory for the subprocess.
        cmd: The command to run as a list of strings.
        log_file: The file to write the log output to.

    Returns:
        The return code of the subprocess.

    """
    print(f"Running SFINCS with: {cmd}")
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
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
    rising_limb_hours: float | int,
    recession_limb_hours: float | int,
) -> pd.DataFrame:
    """Create a triangular hydrograph time series.

    Args:
        peak_discharge: The peak discharge of the hydrograph.
        rising_limb_hours: The duration of the rising limb in hours.
        recession_limb_hours: The duration of the recession limb in hours.

    Returns:
        A pandas DataFrame with the time series of the hydrograph.
    """
    total_duration_hours = rising_limb_hours + recession_limb_hours
    # Create a time array with hourly intervals
    time_hours = np.arange(0, total_duration_hours + 1)
    # Create discharge array for triangular shape
    discharge = np.zeros_like(time_hours, dtype=float)
    # Rising limb: linear increase to peak
    discharge[: rising_limb_hours + 1] = np.linspace(
        0, peak_discharge, rising_limb_hours + 1
    )
    # Recession limb: linear decrease from peak
    discharge[rising_limb_hours:] = np.linspace(
        peak_discharge, 0, recession_limb_hours + 1
    )
    # Create a pandas DataFrame for the time series
    time_index = pd.date_range(
        start="2024-01-01 00:00", periods=len(time_hours), freq="h"
    )
    hydrograph_df = pd.DataFrame({"time": time_index, "discharge": discharge})
    hydrograph_df.set_index("time", inplace=True)
    return hydrograph_df


def check_docker_running() -> bool | None:
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Docker not installed properly or not running properly.")
        return False


def run_sfincs_simulation(
    model_root: Path, simulation_root: Path, gpu: bool | str = "auto"
) -> int:
    """Run SFINCS simulation using either Apptainer or Docker.

    Args:
        model_root: Path to the model root directory.
        simulation_root: Path to the simulation root directory.
            Some paths in the configuration will be made relative to this path.
            The simulation directory must be a subdirectory of the model root directory.
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
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        gpu = result.returncode == 0
        if gpu:
            print("GPU detected, running SFINCS with GPU support.")
        else:
            print("No GPU detected, running SFINCS without GPU support.")

    if gpu:
        version: str = os.getenv(
            "SFINCS_SIF_GPU", "mvanormondt/sfincs-gpu:coldeze_combo_ccall"
        )
    else:
        version: str = os.getenv(
            "SFINCS_SIF",
            "deltares/sfincs-cpu:sfincs-v2.2.0-col-dEze-Release",
        )

    if platform.system() == "Linux":
        # If not a apptainer image, add docker:// prefix
        # to the version string
        if not version.endswith(".sif"):
            version: str = "docker://" + version
        cmd: list[str] = [
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
    waterbody_ids: npt.NDArray[np.int32],
    up_to_downstream: bool = True,
) -> tuple[int, int] | None:
    """Get the first valid xy coordinate from a river's hydrography_xy list.

    Ignore coordinates that fall within a waterbody (waterbody_ids != -1).
    Starts upstream or downstream based on the up_to_downstream flag.

    Args:
        river: Series containing river information, including 'hydrography_xy'.
            This is a list of (x, y) tuples, representing the location of the river
            in the low-resolution hydrological grid.
        waterbody_ids: 2D array of waterbody IDs, where -1 indicates no waterbody.
            Also in the low-resolution hydrological grid.
        up_to_downstream: Whether to search from upstream to downstream.
            Defaults to True (starting upstream).

    Returns:
        A tuple (x, y) of the first valid coordinate, or an empty list if none found.
    """
    xys = river["hydrography_xy"]
    if not up_to_downstream:
        xys = reversed(xys)  # Reverse the order if not going downstream
    for xy in xys:
        is_waterbody = waterbody_ids[xy[1], xy[0]] != -1
        if not is_waterbody:
            return (xy[0], xy[1])
    else:
        return None  # If no valid xy found, return None


def get_representative_river_points(
    river_ID: set, rivers: pd.DataFrame, waterbody_ids: npt.NDArray[np.int32]
) -> list[tuple[float, float]]:
    river = rivers.loc[river_ID]
    if river["represented_in_grid"]:
        xy = _get_xy(river, waterbody_ids, up_to_downstream=True)
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
            xy = _get_xy(river, waterbody_ids, up_to_downstream=False)
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


def assign_return_periods(
    rivers: pd.DataFrame,
    discharge_dataframe: pd.DataFrame,
    return_periods: list[int],
    prefix: str = "Q",
) -> pd.DataFrame:
    assert isinstance(return_periods, list)
    for i, idx in tqdm(enumerate(rivers.index), total=len(rivers)):
        discharge = discharge_dataframe[idx]

        if (discharge < 1e-10).all():
            print(
                f"Discharge is all (near) zeros, skipping return period calculation, for river {idx}"
            )
            discharge_per_return_period = np.zeros_like(return_periods)
        else:
            # Fit the model and calculate return periods
            model = EVA(discharge)
            model.get_extremes(method="BM", block_size="365.2425D")
            model.fit_model()
            discharge_per_return_period = model.get_return_value(
                return_period=return_periods
            )[0]  # [1] and [2] are the uncertainty bounds

            # when only one return period is given, the result is a single value
            # instead of a list, so convert it to a list for simplicty of
            # further processing
            if len(return_periods) == 1:
                discharge_per_return_period = [discharge_per_return_period]
        for return_period, discharge_value in zip(
            return_periods, discharge_per_return_period
        ):
            rivers.loc[idx, f"{prefix}_{return_period}"] = discharge_value
            if (
                discharge_value > 400_000
            ):  # Amazon has a maximum recorded discharge of about 340,000 m3/s
                raise ValueError(
                    f"Discharge value for return period {return_period} is too high: {discharge_value} m3/s for river {idx}."
                )
    return rivers
