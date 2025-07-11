import os
import platform
import subprocess
from os.path import isfile, join
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from hydromt.log import setuplog
from hydromt_sfincs import SfincsModel
from pyextremes import EVA
from shapely.geometry import Point
from tqdm import tqdm


def make_relative_paths(config, model_root, new_root, relpath=None):
    """Return dict with paths to the new model new root."""
    if not relpath:
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


def update_forcing(
    model_root: str,
    out_root: str,
    tstart: pd.Timestamp,
    tstop: pd.Timestamp,
    precip: pd.DataFrame | None = None,
    bzs: pd.DataFrame | None = None,
    dis: pd.DataFrame | None = None,
    bnd: gpd.GeoDataFrame | None = None,
    src: gpd.GeoDataFrame | None = None,
    use_relative_path: bool = True,
    plot: bool = False,
    **config_kwargs,
) -> None:
    """Update the forcing of a sfincs model with the given forcing data.

    Parameters
    ----------
    mod : SfincsModel
        sfincs model to update
    precip, bzs, dis : pd.DataFrame, optional
        precipitation, waterlevel, discharge forcing data, by default None
    bnd, src : gpd.GeoDataFrame, optional
        waterlevel boundary and discharge source locations, by default None
    """
    # read model
    mod = SfincsModel(
        root=model_root,
        mode="r",
        write_gis=False,  # avoid writing gis files
    )
    mod.read()

    # update model root and paths in config (if use_relative_path)
    mod.set_root(out_root, mode="w+")
    if use_relative_path:
        config_kwargs.update(make_relative_paths(mod.config, model_root, out_root))

    # update model time
    mod.setup_config(
        tref=tstart.strftime("%Y%m%d %H%M%S"),
        tstart=tstart.strftime("%Y%m%d %H%M%S"),
        tstop=tstop.strftime("%Y%m%d %H%M%S"),
        **config_kwargs,
    )

    # update model forcing
    variables = []
    if precip is not None:
        mod.setup_precip_forcing(
            timeseries=precip,
        )
        mod.setup_config(**{"precipfile": "sfincs.precip"})
        variables.append("precip")
    if bzs is not None:
        mod.setup_waterlevel_forcing(timeseries=bzs, locations=bnd, merge=False)
        mod.setup_config(**{"bzsfile": "sfincs.bzs", "bndfile": "sfincs.bnd"})
        variables.append("waterlevel")
    if dis is not None:
        mod.setup_discharge_forcing(
            timeseries=dis,
            locations=src,
            merge=False,
        )
        mod.setup_config(**{"srcfile": "sfincs.src", "disfile": "sfincs.dis"})
        variables.append("discharge")

    # save model
    if use_relative_path:
        mod.write_forcing(data_vars=variables)
        mod.write_config()  # write config last
    else:
        mod.write()

    # plot
    if plot:
        mod.plot_forcing(join(mod.root, "forcing.png"))


def run_sfincs_subprocess(root: Path, cmd: list[str], log_file: Path) -> int:
    print(f"Running SFINCS with: {cmd}")
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
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


def write_zsmax_tif(
    root,
    zsmax_fn,
):
    """Write zsmax to tif."""
    mod = SfincsModel(root, mode="r")
    # get maximum waterlevel
    zsmax = mod.results["zsmax"].max("timemax")
    zsmax.fillna(-9999).raster.to_raster(
        zsmax_fn,
        driver="GTiff",
        compress="lzw",
        overwrite=True,
        nodata=-9999,
    )


def get_logger():
    return setuplog("sfincs", log_level=10)


def get_start_point(geom):
    return Point(geom.coords[0])


def get_end_point(geom):
    return Point(geom.coords[-1])


def get_discharge_for_return_periods(
    discharge, return_periods, minimum_median_discharge=100
):
    assert not np.isnan(discharge).any(), "Discharge values contain NaNs"
    assert not np.isinf(discharge).any(), "Discharge values contain infinite values"

    # If median discharge is less than threshold, return zeros
    if (
        discharge.groupby(discharge.index.year).max().median()
        < minimum_median_discharge
    ):
        return np.zeros_like(return_periods)

    # Fit the model and calculate return periods
    model = EVA(discharge)
    # Use the Block Maxima method to fit the model
    model.get_extremes(method="BM", block_size="365.2425D")
    model.fit_model()
    discharge_per_return_period = model.get_return_value(return_period=return_periods)[
        0
    ]
    # The Amazon should have a maximum recorded discharge of about 300,000 m3/s
    # so we check that the discharge values are not too high
    assert discharge_per_return_period.max() < 1e6, "Discharge values too high"
    # # Get the summary at the X-year return period
    return discharge_per_return_period


# Function to create and return a hydrograph time series
def create_hourly_hydrograph(peak_discharge, rising_limb_hours, recession_limb_hours):
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
        start="2024-01-01 00:00", periods=len(time_hours), freq="H"
    )
    hydrograph_df = pd.DataFrame({"time": time_index, "discharge": discharge})
    hydrograph_df.set_index("time", inplace=True)
    return hydrograph_df


def check_docker_running():
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


def run_sfincs_simulation(model_root, simulation_root, gpu=False) -> int:
    # Check if we are on Linux or Windows and run the appropriate script
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
        # If not a singularity image, add docker:// prefix
        # to the version string
        if not version.endswith(".sif"):
            version: str = "docker://" + version
        cmd: list[str] = [
            "singularity",
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


def get_representative_river_points(river_ID: set, rivers: pd.DataFrame):
    river = rivers.loc[river_ID]
    if river["represented_in_grid"]:
        river = rivers.loc[river_ID]
        xy = river["hydrography_xy"][0]  # get most upstream point
        return [(xy[0], xy[1])]
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
        xys = [
            (river[-1][0], river[-1][1])
            for river in representitative_rivers["hydrography_xy"]
        ]
        return xys


def get_discharge_by_river(
    river_IDs: list[int],
    points_per_river,
    discharge: xr.DataArray,
    river_width_alpha: npt.NDArray[np.float32] | None = None,
    river_width_beta: npt.NDArray[np.float32] | None = None,
) -> tuple[pd.DataFrame, xr.DataArray | None, xr.DataArray | None]:
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
    assert not np.isnan(discharge_per_point).any(), "Discharge values contain NaNs"

    if river_width_alpha is not None:
        river_width_alpha_per_point = river_width_alpha[y_points, x_points]
    else:
        river_width_alpha_per_point = None
    if river_width_beta is not None:
        river_width_beta_per_point = river_width_beta[y_points, x_points]
    else:
        river_width_beta_per_point = None

    discharge_df: pd.DataFrame = pd.DataFrame(index=discharge.time)

    i: int = 0
    for river_ID, points in zip(river_IDs, points_per_river, strict=True):
        discharge_per_river = discharge_per_point.isel(
            points=slice(i, i + len(points))
        ).sum(dim="points")
        discharge_df[river_ID] = discharge_per_river
        i += len(points)

    assert i == len(xs), "Discharge values do not match the number of points"

    return discharge_df, river_width_alpha_per_point, river_width_beta_per_point


def assign_return_periods(
    rivers: pd.DataFrame,
    discharge_dataframe: pd.DataFrame,
    return_periods: list[int],
    prefix: str = "Q",
):
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
    return rivers


def snap_to_grid(ds, reference, relative_tollerance=0.02, ydim="y", xdim="x"):
    # make sure all datasets have more or less the same coordinates
    assert np.isclose(
        ds.coords[ydim].values,
        reference[ydim].values,
        atol=abs(ds.rio.resolution()[1] * relative_tollerance),
        rtol=0,
    ).all()
    assert np.isclose(
        ds.coords[xdim].values,
        reference[xdim].values,
        atol=abs(ds.rio.resolution()[0] * relative_tollerance),
        rtol=0,
    ).all()
    return ds.assign_coords({ydim: reference[ydim], xdim: reference[xdim]})


def configure_sfincs_model(sf, model_root, simulation_root):
    """Helper function to configure SFINCS model with common settings."""
    sf.setup_config(
        alpha=0.5
    )  # alpha is the parameter for the CFL-condition reduction. Decrease for additional numerical stability, minimum value is 0.1 and maximum is 0.75 (0.5 default value)
    sf.setup_config(tspinup=86400)  # spinup time in seconds
    sf.setup_config(dtout=900)  # output time step in seconds
    # change root to the output and write
    sf.set_root(simulation_root, mode="w+")
    sf._write_gis = True
    sf.write_grid()
    sf.write_forcing()
    # update config
    sf.setup_config(**make_relative_paths(sf.config, model_root, simulation_root))
    sf.write_config()
    sf.plot_basemap(fn_out="src_points_check.png")
    sf.plot_forcing(fn_out="forcing.png")


def calculate_two_year_return(discharge_series, discharge_name):
    """Processes a discharge series to calculate the 2-year return period value."""
    # Reset index and preprocess the data
    discharge_series = discharge_series.reset_index()
    discharge_series["time"] = pd.to_datetime(discharge_series["time"])
    discharge_series = discharge_series.sort_values(by="time", ascending=True)
    discharge_series[discharge_name] = discharge_series[discharge_name].astype(float)
    discharge_series = discharge_series.dropna(subset=["discharge"])

    # Handle infinite values if any
    discharge_series = discharge_series[np.isfinite(discharge_series[discharge_name])]
    discharge_series.set_index("time", inplace=True)

    # Extract the discharge series
    discharge_series = discharge_series[discharge_name]

    # Fit the model and calculate return periods
    model = EVA(discharge_series)
    model.get_extremes(method="BM", block_size="365.2425D")
    model.fit_model()
    summary = model.get_summary(
        return_period=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000], alpha=0.95
    )

    # Get the 2-year return period value
    return summary.loc[2.0, "return value"]
