import logging
import os
import tempfile
import time
import zipfile
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rioxarray as rxr
import xarray as xr
import xclim.indices as xci
from dateutil.relativedelta import relativedelta
from isimip_client.client import ISIMIPClient
from numcodecs.zarr3 import FixedScaleOffset
from tqdm import tqdm

from geb.workflows.io import get_window

from ...workflows.io import calculate_scaling, open_zarr, to_zarr
from ..workflows.general import (
    interpolate_na_along_time_dim,
    resample_chunked,
    resample_like,
)


def reproject_and_apply_lapse_rate_temperature(
    T, elevation_forcing, elevation_target, lapse_rate=-0.0065
) -> xr.DataArray:
    assert (T.x.values == elevation_forcing.x.values).all()
    assert (T.y.values == elevation_forcing.y.values).all()
    t_at_sea_level = T - elevation_forcing * lapse_rate
    t_at_sea_level_reprojected = resample_like(
        t_at_sea_level, elevation_target, method="conservative"
    )
    T_grid = t_at_sea_level_reprojected + lapse_rate * elevation_target
    T_grid.name = T.name
    T_grid.attrs = T.attrs
    return T_grid


def get_pressure_correction_factor(DEM, g, Mo, lapse_rate):
    return (288.15 / (288.15 + lapse_rate * DEM)) ** (g * Mo / (8.3144621 * lapse_rate))


def reproject_and_apply_lapse_rate_pressure(
    pressure,
    elevation_forcing,
    elevation_target,
    g=9.80665,
    Mo=0.0289644,
    lapse_rate=-0.0065,
) -> xr.DataArray:
    """Pressure correction based on elevation lapse_rate.

    Parameters
    ----------
    dem_model : xarray.DataArray
        DataArray with high res lat/lon axis and elevation data
    g : float, default 9.80665
        gravitational constant [m s-2]

    Mo : float, default 0.0289644
        molecular weight of gas [kg / mol]
    lapse_rate : float, deafult -0.0065
        lapse rate of temperature [C m-1]

    Returns:
    -------
    press_fact : xarray.DataArray
        pressure correction factor
    """
    assert (pressure.x.values == elevation_forcing.x.values).all()
    assert (pressure.y.values == elevation_forcing.y.values).all
    pressure_at_sea_level = pressure / get_pressure_correction_factor(
        elevation_forcing, g, Mo, lapse_rate
    )  # divide by pressure factor to get pressure at sea level
    pressure_at_sea_level_reprojected = resample_like(
        pressure_at_sea_level, elevation_target, method="conservative"
    )
    pressure_grid = (
        pressure_at_sea_level_reprojected
        * get_pressure_correction_factor(elevation_target, g, Mo, lapse_rate)
    )  # multiply by pressure factor to get pressure at DEM grid, corrected for elevation
    pressure_grid.name = pressure.name
    pressure_grid.attrs = pressure.attrs

    return pressure_grid


def download_ERA5(
    folder: Path,
    variable: str,
    start_date: datetime,
    end_date: datetime,
    bounds: tuple[float, float, float, float],
    logger: logging.Logger,
) -> xr.DataArray:
    """Download ERA5 data for a specific variable and time period and save it as a zarr file.

    If the data is already downloaded, it will be opened from the local zarr file.

    Args:
        folder: Folder to store the downloaded data
        variable: Short name of the variable to download (e.g., "t2m"). Codes can be found here: https://codes.ecmwf.int/grib/param-db/
        start_date: start date of the time period to download
        end_date: end date of the time period to download
        bounds:  bounding box in the format (min_lon, min_lat, max_lon, max_lat)
        logger:   logger to use for logging

    Returns:
        Downloaded ERA5 data as an xarray DataArray.
    """
    output_fn = folder / f"{variable}.zarr"
    if output_fn.exists():
        da: xr.DataArray = open_zarr(output_fn)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        da: xr.DataArray = xr.open_dataset(
            "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",
            storage_options={"client_kwargs": {"trust_env": True}},
            chunks={},
            engine="zarr",
        )[variable].rename({"valid_time": "time", "latitude": "y", "longitude": "x"})

        da: xr.DataArray = da.drop_vars(["number", "surface", "depthBelowLandLayer"])

        buffer: float = 0.5

        # Check if region crosses the meridian (longitude=0)
        # use a slightly larger slice. The resolution is 0.1 degrees, so buffer degrees is a bit more than that (to be sure)
        if bounds[0] < 0 and bounds[2] > 0:
            # Need to handle the split across the meridian
            # Get western hemisphere part (longitude < 0)
            west_da: xr.DataArray = da.sel(
                time=slice(start_date, end_date),
                y=slice(bounds[3] + buffer, bounds[1] - buffer),
                x=slice(((bounds[0] - buffer) + 360) % 360, 360),
            )
            # Get eastern hemisphere part (longitude > 0)
            east_da: xr.DataArray = da.sel(
                time=slice(start_date, end_date),
                y=slice(bounds[3] + buffer, bounds[1] - buffer),
                x=slice(0, ((bounds[2] + buffer) + 360) % 360),
            )
            # Combine the two parts
            da: xr.DataArray = xr.concat([west_da, east_da], dim="x")
        else:
            # Regular case - doesn't cross meridian
            da: xr.DataArray = da.sel(
                time=slice(start_date, end_date),
                y=slice(bounds[3] + buffer, bounds[1] - buffer),
                x=slice(
                    ((bounds[0] - buffer) + 360) % 360,
                    ((bounds[2] + buffer) + 360) % 360,
                ),
            )

        # Reorder x to be between -180 and 180 degrees
        da: xr.DataArray = da.assign_coords(x=((da.x + 180) % 360 - 180))

        logger.info(f"Downloading ERA5 {variable} to {output_fn}")
        da.attrs["_FillValue"] = da.attrs["GRIB_missingValue"]
        da: xr.DataArray = da.raster.mask_nodata()
        da: xr.DataArray = to_zarr(
            da,
            output_fn,
            time_chunksize=get_chunk_size(da, target=1e7),
            crs=4326,
        )
    return da


def process_ERA5(
    variable: str,
    folder: Path,
    start_date: datetime,
    end_date: datetime,
    bounds: tuple[float, float, float, float],
    logger: logging.Logger,
) -> xr.DataArray:
    """Process ERA5 data for a given variable and time period.

    Downloads the data from the Climate Data Store (CDS) if not already available,
    processes it to ensure it is in the correct format, and applies de-accumulation
    for accumulated variables and interpolation of missing values.

    Args:
        variable: short name of the variable to process (e.g., "t2m"). Codes can be found here: https://codes.ecmwf.int/grib/param-db/
        folder: folder to store the downloaded data
        start_date: start date of the time period to process
        end_date: end date of the time period to process
        bounds:  bounding box in the format (min_lon, min_lat, max_lon, max_lat)
        logger:  logger to use for logging

    Raises:
        NotImplementedError: If the step type of the data is not "accum" or "instant".

    Returns:
        xr.DataArray: Processed ERA5 data as an xarray DataArray.
    """
    da: xr.DataArray = download_ERA5(
        folder, variable, start_date, end_date, bounds, logger
    )
    # assert that time is monotonically increasing with a constant step size
    assert (
        da.time.diff("time").astype(np.int64)
        == (da.time[1] - da.time[0]).astype(np.int64)
    ).all(), "time is not monotonically increasing with a constant step size"
    if da.attrs["GRIB_stepType"] == "accum":
        da: xr.DataArray = xr.where(
            da.isel(time=slice(1, None)).time.dt.hour == 1,
            da.isel(time=slice(1, None)),
            da.diff(dim="time", n=1),
        )

    elif da.attrs["GRIB_stepType"] == "instant":
        pass
    else:
        raise NotImplementedError

    da: xr.DataArray = da.rio.set_crs(4326)
    da.raster.set_crs(4326)
    da: xr.DataArray = interpolate_na_along_time_dim(da)

    return da


def plot_timeline(
    da: xr.DataArray, data: xr.DataArray, name: str, ax: plt.Axes
) -> None:
    """Plot a timeline of the data.

    Args:
        da: the original xarray DataArray containing the data to plot.
        data: the data to plot, should be a 1D xarray DataArray with a time dimension.
        name: the name of the data, used for the plot title.
        ax: the matplotlib axes to plot on.
    """
    ax.plot(data.time, data)
    ax.set_xlabel("Time")
    if "units" in da.attrs:
        ax.set_ylabel(da.attrs["units"])
    ax.set_xlim(data.time[0], data.time[-1])
    ax.set_ylim(data.min(), data.max() * 1.1)
    significant_digits: int = 6
    ax.set_title(
        f"{name} - mean: {data.mean().item():.{significant_digits}f} - min: {data.min().item():.{significant_digits}f} - max: {data.max().item():.{significant_digits}f}"
    )


def get_chunk_size(da, target: float | int = 1e8) -> int:
    """Calculate the optimal chunk size for the given xarray DataArray based on the target size.

    Args:
        da: The xarray DataArray for which to calculate the chunk size.
        target: The target size in bytes. Default is 1e8 (100 MB).

    Returns:
        The calculated chunk size in bytes.
    """
    return int(target / (da.dtype.itemsize * da.x.size * da.y.size))


class Forcing:
    def __init__(self):
        pass

    def download_isimip(
        self,
        product: str,
        variable: str,
        forcing: str | None = None,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
        simulation_round: str = "ISIMIP3a",
        climate_scenario: str = "obsclim",
        resolution: str | None = None,
        buffer: int = 0,
    ) -> xr.DataArray:
        """This method downloads ISIMIP climate data for GEB.

        It first retrieves the dataset
        metadata from the ISIMIP repository using the specified `product`, `variable`, `forcing`, and `resolution`
        parameters. It then downloads the data files that match the specified `start_date` and `end_date` parameters, and
        extracts them to the specified `download_path` directory.

        The resulting climate data is returned as an xarray dataset. The dataset is assigned the coordinate reference system
        EPSG:4326, and the spatial dimensions are set to 'lon' and 'lat'.

        Args:
            product: The name of the ISIMIP product to download.
            variable: The name of the climate variable to download.
            forcing: The name of the climate forcing to download.
            start_date: The start date of the data. Default is None.
            end_date: The end date of the data. Default is None.
            simulation_round: The ISIMIP simulation round to download data for. Default is "ISIMIP3a".
            climate_scenario: The climate scenario to download data for. Default is "obsclim".
            resolution: The resolution of the data to download. Default is None.
            buffer: The buffer size in degrees to add to the bounding box of the data to download. Default is 0.

        Returns:
            The downloaded climate data as an xarray dataset.
        """
        # if start_date is specified, end_date must be specified as well
        assert (start_date is None) == (end_date is None)

        if isinstance(start_date, datetime):
            start_date: date = start_date.date()
        if isinstance(end_date, datetime):
            end_date: date = end_date.date()

        client: ISIMIPClient = ISIMIPClient()
        download_path: Path = self.preprocessing_dir / "climate"
        if forcing is not None:
            download_path: Path = download_path / forcing
        download_path: Path = download_path / variable
        download_path.mkdir(parents=True, exist_ok=True)

        # Code to get data from disk rather than server.
        parse_files = []
        for file in os.listdir(download_path):
            if file.endswith(".nc"):
                fp = download_path / file
                parse_files.append(fp)

        # get the dataset metadata from the ISIMIP repository
        response = client.datasets(
            simulation_round=simulation_round,
            product=product,
            climate_forcing=forcing,
            climate_scenario=climate_scenario,
            climate_variable=variable,
            resolution=resolution,
        )
        assert len(response["results"]) == 1
        dataset = response["results"][0]
        files = dataset["files"]

        xmin, ymin, xmax, ymax = self.bounds
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer

        if variable == "orog":
            assert len(files) == 1
            filename = files[
                0
            ][
                "name"
            ]  # global should be included due to error in ISIMIP API .replace('_global', '')
            parse_files = [filename]
            if not (download_path / filename).exists():
                download_files = [files[0]["path"]]
            else:
                download_files = []

        else:
            assert start_date is not None and end_date is not None
            download_files = []
            parse_files = []
            for file in files:
                name: str = file["name"]
                if name.endswith(".nc"):
                    splitted_filename = name.split("_")
                    dt = splitted_filename[-1].split(".")[0]
                    if "-" in dt:
                        file_start_date, file_end_date = date.split("-")
                        file_start_date = datetime.strptime(
                            file_start_date, "%Y%m%d"
                        ).date()
                        file_end_date = datetime.strptime(
                            file_end_date, "%Y%m%d"
                        ).date()
                    elif len(dt) == 6:
                        file_start_date = datetime.strptime(dt, "%Y%m").date()
                        file_end_date = (
                            file_start_date
                            + relativedelta(months=1)
                            - relativedelta(days=1)
                        )
                    elif len(dt) == 4:  # is year
                        assert splitted_filename[-2].isdigit()
                        file_start_date = datetime.strptime(
                            splitted_filename[-2], "%Y"
                        ).date()
                        file_end_date = date(int(dt), 12, 31)
                    else:
                        raise ValueError(f"could not parse date {dt} from file {name}")

                    if not (file_end_date < start_date or file_start_date > end_date):
                        parse_files.append(file["name"].replace("_global", ""))
                        if not (
                            download_path / file["name"].replace("_global", "")
                        ).exists():
                            download_files.append(file["path"])
                elif name.endswith(".txt"):
                    parse_files.append(name)
                    if not (download_path / name).exists():
                        download_files.append(file["path"])
                else:
                    raise ValueError(f"Unknown file type {name} in ISIMIP dataset")

        if not parse_files:
            raise ValueError(
                f"No files found for variable {variable} in ISIMIP dataset {dataset['id']}"
            )

        all_nc = all(str(f).endswith(".nc") for f in parse_files)
        all_txt = all(str(f).endswith(".txt") for f in parse_files)
        assert all_nc or all_txt, "All parse_files must end with either .nc or .txt"
        if all_txt:
            assert len(parse_files) == 1, "Only one .txt file is expected"

        if download_files:
            self.logger.info(f"Requesting download of {len(download_files)} files")
            if all_nc:
                while True:
                    try:
                        response = client.cutout(
                            download_files, [ymin, ymax, xmin, xmax]
                        )
                    except requests.exceptions.HTTPError:
                        self.logger.warning(
                            "HTTPError, could not download files, retrying in 60 seconds"
                        )
                    else:
                        if response["status"] == "finished":
                            break
                        elif response["status"] == "started":
                            self.logger.info(
                                f"{response['meta']['created_files']}/{response['meta']['total_files']} files prepared on ISIMIP server for {variable}, waiting 60 seconds before retrying"
                            )
                        elif response["status"] == "queued":
                            self.logger.info(
                                f"Data preparation queued for {variable} on ISIMIP server, waiting 60 seconds before retrying"
                            )
                        elif response["status"] == "failed":
                            self.logger.info(
                                "ISIMIP internal server error, waiting 60 seconds before retrying"
                            )
                        else:
                            raise ValueError(
                                f"Could not download files: {response['status']}"
                            )
                    time.sleep(60)

                self.logger.info(f"Starting download of files for {variable}")
                # download the file when it is ready
                client.download(
                    response["file_url"],
                    path=download_path,
                    validate=False,
                    extract=False,
                )

                self.logger.info(f"Download finished for {variable}")
                # remove zip file
                zip_file = download_path / Path(
                    urlparse(response["file_url"]).path.split("/")[-1]
                )
                # make sure the file exists
                assert zip_file.exists()
                # Open the zip file
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    # Get a list of all the files in the zip file
                    file_list = [f for f in zip_ref.namelist() if f.endswith(".nc")]
                    # Extract each file one by one
                    for i, file_name in enumerate(file_list):
                        # Rename the file
                        bounds_str = ""
                        if isinstance(ymin, float):
                            bounds_str += f"_lat{ymin}"
                        else:
                            bounds_str += f"_lat{ymin:.1f}"
                        if isinstance(ymax, float):
                            bounds_str += f"to{ymax}"
                        else:
                            bounds_str += f"to{ymax:.1f}"
                        if isinstance(xmin, float):
                            bounds_str += f"lon{xmin}"
                        else:
                            bounds_str += f"lon{xmin:.1f}"
                        if isinstance(xmax, float):
                            bounds_str += f"to{xmax}"
                        else:
                            bounds_str += f"to{xmax:.1f}"
                        assert bounds_str in file_name
                        new_file_name = file_name.replace(bounds_str, "")
                        zip_ref.getinfo(file_name).filename = new_file_name
                        # Extract the file
                        if os.name == "nt":
                            max_file_path_length = 260
                        else:
                            max_file_path_length = os.pathconf("/", "PC_PATH_MAX")
                        assert (
                            len(str(download_path / new_file_name))
                            <= max_file_path_length
                        ), (
                            f"File path too long: {download_path / zip_ref.getinfo(file_name).filename}"
                        )
                        zip_ref.extract(file_name, path=download_path)
                # remove zip file
                (
                    download_path
                    / Path(urlparse(response["file_url"]).path.split("/")[-1])
                ).unlink()

            else:
                client.download(
                    "https://files.isimip.org/" + download_files[0],
                    path=download_path,
                )

        if all_nc:
            datasets: list[xr.Dataset] = [
                xr.open_dataset(download_path / file, chunks={}) for file in parse_files
            ]
            for dataset in datasets:
                assert "lat" in dataset.coords and "lon" in dataset.coords

            # make sure y is decreasing rather than increasing
            datasets: list[xr.Dataset] = [
                (
                    dataset.reindex(lat=dataset.lat[::-1])
                    if dataset.lat[0] < dataset.lat[-1]
                    else dataset
                )
                for dataset in datasets
            ]

            reference = datasets[0]
            for dataset in datasets:
                # make sure all datasets have more or less the same coordinates
                assert np.isclose(
                    dataset.coords["lat"].values,
                    reference["lat"].values,
                    atol=abs(datasets[0].rio.resolution()[1] / 50),
                    rtol=0,
                ).all()
                assert np.isclose(
                    dataset.coords["lon"].values,
                    reference["lon"].values,
                    atol=abs(datasets[0].rio.resolution()[0] / 50),
                    rtol=0,
                ).all()

            datasets: list[xr.Dataset] = [
                ds.assign_coords(
                    lon=reference["lon"].values, lat=reference["lat"].values
                )
                for ds in datasets
            ]
            if len(datasets) > 1:
                ds: xr.Dataset = xr.concat(datasets, dim="time")
            else:
                ds: xr.Dataset = datasets[0]

            if start_date is not None:
                ds = ds.sel(time=slice(start_date, end_date))
                # assert that time is monotonically increasing with a constant step size
                assert (
                    ds.time.diff("time").astype(np.int64)
                    == (ds.time[1] - ds.time[0]).astype(np.int64)
                ).all()

            ds.raster.set_spatial_dims(x_dim="lon", y_dim="lat")
            assert not ds.lat.attrs, "lat already has attributes"
            assert not ds.lon.attrs, "lon already has attributes"
            ds.lat.attrs = {
                "long_name": "latitude of grid cell center",
                "units": "degrees_north",
            }
            ds.lon.attrs = {
                "long_name": "longitude of grid cell center",
                "units": "degrees_east",
            }
            ds.raster.set_crs(4326)

            ds: xr.Dataset = ds.rename({"lon": "x", "lat": "y"})

            # check whether data is for noon or midnight. If noon, subtract 12 hours from time coordinate to align with other datasets
            if hasattr(ds, "time") and pd.to_datetime(ds.time[0].values).hour == 12:
                # subtract 12 hours from time coordinate
                self.logger.warning(
                    "Subtracting 12 hours from time coordinate to align climate datasets"
                )
                ds: xr.Dataset = ds.assign_coords(
                    time=ds.time - np.timedelta64(12, "h")
                )
            return ds[variable]

        elif all_txt:
            assert len(parse_files) == 1, "Only one .txt file is expected"
            df = pd.read_csv(
                download_path / parse_files[0], sep=r"\s+", names=["year", variable]
            ).set_index("year")
            df = df[(df.index >= start_date.year) & (df.index <= end_date.year)]
            assert len(df.columns) == 1, "Only one column expected in .txt file"
            # convert df to xarray DataArray
            da = xr.DataArray(
                df[variable].values,
                coords={
                    "time": pd.date_range(
                        start=f"{df.index[0]}-01-01",
                        end=f"{df.index[-1]}-12-31",
                        freq="YS",
                    )
                },
                dims=["time"],
            )
            return da
        else:
            raise ValueError(
                "All parse_files must end with either .nc or .txt, but got: "
                f"{parse_files}"
            )

    def plot_forcing(self, da, name):
        fig, axes = plt.subplots(4, 1, figsize=(20, 10), gridspec_kw={"hspace": 0.5})

        mask = self.grid["mask"]
        data = ((da * ~mask).sum(dim=("y", "x")) / (~mask).sum()).compute()
        assert not np.isnan(data.values).any(), "data contains NaN values"

        # plot entire timeline on the first axis
        plot_timeline(da, data, name, axes[0])

        # plot the first three years on separate axes
        for i in range(0, 3):
            year = data.time[0].dt.year + i
            year_data = data.sel(time=data.time.dt.year == year)
            if year_data.size > 0:
                plot_timeline(
                    da,
                    data.sel(time=da.time.dt.year == year),
                    f"{name} - {year.item()}",
                    axes[i + 1],
                )

        fp = self.report_dir / (name + "_timeline.png")
        fp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fp)

        plt.close(fig)

        spatial_data = da.mean(dim="time").compute()

        spatial_data.plot()

        plt.title(name)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        spatial_fp: Path = self.report_dir / (name + "_spatial.png")
        plt.savefig(spatial_fp)

        plt.close()

    def set_xy_attrs(self, da: xr.DataArray) -> None:
        """Set CF-compliant attributes for the x and y coordinates of a DataArray."""
        da.x.attrs = {"long_name": "longitude", "units": "degrees_east"}
        da.y.attrs = {"long_name": "latitude", "units": "degrees_north"}

    def set_pr_hourly(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/pr_hourly"
        da.attrs = {
            "standard_name": "precipitation_flux",
            "long_name": "Precipitation",
            "units": "kg m-2 s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0

        # maximum rainfall in one hour was 304.8 mm in 1956 in Holt, Missouri, USA
        # https://www.guinnessworldrecords.com/world-records/737965-greatest-rainfall-in-one-hour
        # we take a wide margin of 500 mm/h
        max_value = 500 / 3600  # convert to kg/m2/s
        precision = 0.01 / 3600  # 0.01 mm in kg/m2/s

        scaling_factor, out_dtype = calculate_scaling(
            0, max_value, offset=offset, precision=precision
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            time_chunksize=7 * 24,
            filters=filters,
        )
        return da

    def set_pr(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/pr"
        da.attrs = {
            "standard_name": "precipitation_flux",
            "long_name": "Precipitation",
            "units": "kg m-2 s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        # maximum rainfall in one hour was 304.8 mm in 1956 in Holt, Missouri, USA
        # https://www.guinnessworldrecords.com/world-records/737965-greatest-rainfall-in-one-hour
        # we take a wide margin of 500 mm/h
        # this function is currently daily, so the hourly value should be save
        max_value: float = 500 / 3600  # convert to kg/m2/s
        precision: float = 0.01 / 3600  # 0.01 mm in kg/m2/s

        offset: int = 0
        scaling_factor, out_dtype = calculate_scaling(
            0, max_value, offset=offset, precision=precision
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self._mask_forcing(da, value=-offset)
        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def set_rsds(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/rsds"
        da.attrs = {
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "long_name": "Surface Downwelling Shortwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, out_dtype = calculate_scaling(
            0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def set_rlds(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/rlds"
        da.attrs = {
            "standard_name": "surface_downwelling_longwave_flux_in_air",
            "long_name": "Surface Downwelling Longwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, out_dtype = calculate_scaling(
            0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def set_tas(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/tas"
        da.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C = 273.15
        offset = -15 - K_to_C  # average temperature on earth
        scaling_factor, out_dtype = calculate_scaling(
            -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )

        self.plot_forcing(da, name)
        return da

    def set_tasmax(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/tasmax"
        da.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Daily Maximum Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C = 273.15
        offset = -15 - K_to_C  # average temperature on earth
        scaling_factor, out_dtype = calculate_scaling(
            -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def set_tasmin(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/tasmin"
        da.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Daily Minimum Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C = 273.15
        offset = -15 - K_to_C  # average temperature on earth
        scaling_factor, out_dtype = calculate_scaling(
            -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def set_hurs(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/hurs"
        da.attrs = {
            "standard_name": "relative_humidity",
            "long_name": "Near-Surface Relative Humidity",
            "units": "%",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = -50
        scaling_factor, out_dtype = calculate_scaling(
            0, 100, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def _mask_forcing(self, da: xr.DataArray, value: int | float) -> xr.DataArray:
        """Mask the forcing data where the grid mask is False.

        Args:
            da: DataArray to mask.
            value: Value to use for masking where the grid mask is False.

        Returns:
            DataArray with replaced values with the same dimensions as the input DataArray.
        """
        da_: xr.DataArray = xr.where(~self.grid["mask"], da, value, keep_attrs=True)
        da_: xr.DataArray = da_.rio.write_crs(da.rio.crs)
        # restore the original dimensions order which can be changed by the mask operation
        da: xr.DataArray = da_.transpose(*da.dims)
        return da

    def set_ps(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/ps"
        da.attrs = {
            "standard_name": "surface_air_pressure",
            "long_name": "Surface Air Pressure",
            "units": "Pa",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = -100_000
        scaling_factor, out_dtype = calculate_scaling(
            30_000, 120_000, offset=offset, precision=10
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def set_sfcwind(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/sfcwind"
        da.attrs = {
            "standard_name": "wind_speed",
            "long_name": "Near-Surface Wind Speed",
            "units": "m s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, out_dtype = calculate_scaling(
            0, 120, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def set_SPEI(self, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        name: str = "climate/SPEI"
        da.attrs = {
            "units": "-",
            "long_name": "Standard Precipitation Evapotranspiration Index",
            "name": "spei",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        # this range corresponds to probabilities of lower than 0.001 and higher than 0.999
        # which should be considered non-significant
        min_SPEI = -3.09
        max_SPEI = 3.09
        da = da.clip(min=min_SPEI, max=max_SPEI)

        offset = 0
        scaling_factor, out_dtype = calculate_scaling(
            min_SPEI, max_SPEI, offset=offset, precision=0.001
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=da.dtype,
                astype=out_dtype,
            ),
        ]

        da = self._mask_forcing(da, value=-offset)
        da = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da),
        )
        self.plot_forcing(da, name)
        return da

    def setup_forcing_era5(self):
        target = self.grid["mask"]
        target.raster.set_crs(4326)

        download_args: dict[str, Any] = {
            "folder": self.preprocessing_dir / "climate" / "ERA5",
            "start_date": self.start_date
            - relativedelta(
                years=1
            ),  # include one year before the start date for SPEI calculation
            "end_date": self.end_date,
            "bounds": target.raster.bounds,
            "logger": self.logger,
        }

        pr_hourly = process_ERA5(
            "tp",  # total_precipitation
            **download_args,
        )
        elevation_forcing, elevation_target = self.get_elevation_forcing_and_grid(
            self.grid["mask"], pr_hourly, forcing_name="ERA5"
        )

        pr_hourly = pr_hourly * (1000 / 3600)  # convert from m/hr to kg/m2/s

        # ensure no negative values for precipitation, which may arise due to float precision
        pr_hourly = xr.where(pr_hourly > 0, pr_hourly, 0, keep_attrs=True)
        pr_hourly = self.set_pr_hourly(pr_hourly)  # weekly chunk size

        pr = pr_hourly.resample(time="D").mean()  # get daily mean
        pr = resample_like(pr, target, method="conservative")
        pr = self.set_pr(pr)

        hourly_tas = process_ERA5("t2m", **download_args)

        tas_avg = hourly_tas.resample(time="D").mean()
        tas_avg = reproject_and_apply_lapse_rate_temperature(
            tas_avg, elevation_forcing, elevation_target
        )
        self.set_tas(tas_avg)

        tasmax = hourly_tas.resample(time="D").max()
        tasmax = reproject_and_apply_lapse_rate_temperature(
            tasmax, elevation_forcing, elevation_target
        )
        self.set_tasmax(tasmax)

        tasmin = hourly_tas.resample(time="D").min()
        tasmin = reproject_and_apply_lapse_rate_temperature(
            tasmin, elevation_forcing, elevation_target
        )
        self.set_tasmin(tasmin)

        hourly_dew_point_tas = process_ERA5(
            "d2m",
            **download_args,
        )
        dew_point_tas = hourly_dew_point_tas.resample(time="D").mean()
        dew_point_tas = reproject_and_apply_lapse_rate_temperature(
            dew_point_tas, elevation_forcing, elevation_target
        )

        water_vapour_pressure = 0.6108 * np.exp(
            17.27 * (dew_point_tas - 273.15) / (237.3 + (dew_point_tas - 273.15))
        )  # calculate water vapour pressure (kPa)
        saturation_vapour_pressure = 0.6108 * np.exp(
            17.27 * (tas_avg - 273.15) / (237.3 + (tas_avg - 273.15))
        )

        assert water_vapour_pressure.shape == saturation_vapour_pressure.shape
        relative_humidity = (water_vapour_pressure / saturation_vapour_pressure) * 100
        self.set_hurs(relative_humidity)

        hourly_rsds = process_ERA5(
            "ssrd",  # surface_solar_radiation_downwards
            **download_args,
        )
        rsds = hourly_rsds.resample(time="D").sum() / (
            24 * 3600
        )  # get daily sum and convert from J/m2 to W/m2

        rsds = resample_like(rsds, target, method="conservative")
        self.set_rsds(rsds)

        hourly_rlds = process_ERA5(
            "strd",  # surface_thermal_radiation_downwards
            **download_args,
        )
        rlds = hourly_rlds.resample(time="D").sum() / (24 * 3600)
        rlds = resample_like(rlds, target, method="conservative")
        self.set_rlds(rlds)

        pressure = process_ERA5("sp", **download_args)
        pressure = reproject_and_apply_lapse_rate_pressure(
            pressure, elevation_forcing, elevation_target
        )
        pressure = pressure.resample(time="D").mean()
        self.set_ps(pressure)

        u_wind = process_ERA5(
            "u10",
            **download_args,
        )
        u_wind = u_wind.resample(time="D").mean()

        v_wind = process_ERA5(
            "v10",
            **download_args,
        )
        v_wind = v_wind.resample(time="D").mean()
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        wind_speed = resample_like(wind_speed, target, method="conservative")
        self.set_sfcwind(wind_speed)

    def setup_forcing_ISIMIP(self, resolution_arcsec: int, forcing: str) -> None:
        """Sets up the forcing data for GEB using ISIMIP data.

        Parameters
        ----------
        resolution_arcsec : int
            The resolution of the data in arcseconds. Supported values are 30 and 1800.
        forcing : str
            The forcing data to use. Supported values are 'chelsa-w5e5' for 30 arcsec resolution
            and ipsl-cm6a-lr, gfdl-esm4, mpi-esm1-2-hr, mri-esm2-0, and mri-esm2-0 for 1800 arcsec resolution.
        """
        if resolution_arcsec == 30:
            assert forcing == "chelsa-w5e5", (
                "Only chelsa-w5e5 is supported for 30 arcsec resolution"
            )
            # download source data from ISIMIP
            self.logger.info("setting up forcing data")
            high_res_variables: list = ["pr", "rsds", "tas", "tasmax", "tasmin"]
            self.setup_30arcsec_variables_isimip(high_res_variables)
            self.logger.info("setting up relative humidity...")
            self.setup_hurs_isimip_30arcsec()
            self.logger.info("setting up longwave radiation...")
            self.setup_longwave_isimip_30arcsec()
            self.logger.info("setting up pressure...")
            self.setup_pressure_isimip_30arcsec()
            self.logger.info("setting up wind...")
            self.setup_wind_isimip_30arcsec()
        elif resolution_arcsec == 1800:
            variables = [
                "pr",
                "rsds",
                "tas",
                "tasmax",
                "tasmin",
                "hurs",
                "rlds",
                "ps",
                "sfcwind",
            ]
            self.setup_1800arcsec_variables_isimip(forcing, variables)
        else:
            raise ValueError(
                "Only 30 arcsec and 1800 arcsec resolution is supported for ISIMIP data"
            )

    def setup_forcing(
        self,
        resolution_arcsec: int,
        forcing_name: str | None = None,
        data_source: str = "ERA5",
    ):
        """Sets up the forcing data for GEB.

        Args:
            resolution_arcsec : The resolution of the data in arcseconds. Supported values are 30 and 1800.
            data_source : The data source to use for the forcing data. Can be ERA5 or ISIMIP. Default is 'era5'.
            forcing_name : The name of the forcing data to use within the dataset. Only required for ISIMIP data.
                For ISIMIP, this can be 'chelsa-w5e5' for 30 arcsec resolution
                or 'ipsl-cm6a-lr', 'gfdl-esm4', 'mpi-esm1-2-hr', 'mri-esm2-0', or 'mri-esm2-0' for 1800 arcsec resolution.

        Notes:
            This method sets up the forcing data for GEB. It first downloads the high-resolution variables
            (precipitation, surface solar radiation, air temperature, maximum air temperature, and minimum air temperature) from
            the ISIMIP dataset for the specified time period.

            The method then sets up the relative humidity, longwave radiation, pressure, and wind data for the model. The
            relative humidity data is downloaded from the ISIMIP dataset using the `setup_hurs_isimip_30arcsec` method. The longwave radiation
            data is calculated using the air temperature and relative humidity data and the `calculate_longwave` function. The
            pressure data is downloaded from the ISIMIP dataset using the `setup_pressure_isimip_30arcsec` method. The wind data is downloaded
            from the ISIMIP dataset using the `setup_wind_isimip_30arcsec` method. All these data are first downscaled to the model grid.

            The resulting forcing data is set as forcing data in the model with names of the form 'forcing/{variable_name}'.
        """
        if data_source == "ISIMIP":
            self.setup_forcing_ISIMIP(resolution_arcsec, forcing_name)
        elif data_source == "ERA5":
            self.setup_forcing_era5()
        elif data_source == "CMIP":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(f"Unknown data source: {data_source}")

    def construct_ISIMIP_variable(
        self, variable_name: str, forcing: str | None, ssp: str
    ) -> xr.DataArray:
        self.logger.info(f"Setting up {variable_name}...")
        first_year_future_climate: int = 2015
        da_list: list = []
        if ssp == "picontrol":
            da: xr.DataArray = self.download_isimip(
                product="InputData",
                simulation_round="ISIMIP3b",
                climate_scenario=ssp,
                variable=variable_name,
                start_date=self.start_date,
                end_date=self.end_date,
                forcing=forcing,
                resolution=None,
                buffer=1,
            )
            da_list.append(da)

        if (
            (
                self.end_date.year < first_year_future_climate
                or self.start_date.year < first_year_future_climate
            )
            and ssp != "picontrol"
        ):  # isimip cutoff date between historic and future climate
            da: xr.DataArray = self.download_isimip(
                product="InputData",
                simulation_round="ISIMIP3b",
                climate_scenario="historical",
                variable=variable_name,
                start_date=self.start_date,
                end_date=self.end_date,
                forcing=forcing,
                resolution=None,
                buffer=1,
            )
            da_list.append(da)
        if (
            self.start_date.year >= first_year_future_climate
            or self.end_date.year >= first_year_future_climate
        ) and ssp != "picontrol":
            assert ssp is not None, "ssp must be specified for future climate"
            assert ssp != "historical", "historical scenarios run until 2014"
            da: xr.DataArray = self.download_isimip(
                product="InputData",
                simulation_round="ISIMIP3b",
                climate_scenario=ssp,
                variable=variable_name,
                start_date=self.start_date,
                end_date=self.end_date,
                forcing=forcing,
                resolution=None,
                buffer=1,
            )
            da_list.append(da)

        da: xr.DataArray = xr.concat(
            da_list, dim="time", combine_attrs="drop_conflicts", compat="equals"
        )  # all values and dimensions must be the same

        step_size: np.int64 = (da.time[1] - da.time[0]).astype(np.int64)

        # assert that time is monotonically increasing with a constant step size
        # check if step size is yearly
        if step_size == 365 * 24 * 3600 * 1e9 or step_size == 366 * 24 * 3600 * 1e9:
            assert (da.time.dt.year.diff("time").astype(np.int64) == 1).all(), (
                "time is not monotonically increasing with a constant step size"
            )
        else:  # if not check if step size is constant
            assert (
                da.time.diff("time").astype(np.int64)
                == (da.time[1] - da.time[0]).astype(np.int64)
            ).all(), "time is not monotonically increasing with a constant step size"

        if variable_name in ("co2",):
            pass

        elif variable_name in ("tas", "tasmin", "tasmax", "ps"):
            elevation_forcing, elevation_grid = self.get_elevation_forcing_and_grid(
                self.grid["mask"], da, forcing_name="ISIMIP"
            )

            if variable_name in ("tas", "tasmin", "tasmax"):
                da: xr.DataArray = reproject_and_apply_lapse_rate_temperature(
                    da, elevation_forcing, elevation_grid
                )
            elif variable_name == "ps":
                da: xr.DataArray = reproject_and_apply_lapse_rate_pressure(
                    da, elevation_forcing, elevation_grid
                )
            else:
                raise ValueError
        else:
            target: xr.DataArray = self.grid["mask"]
            da: xr.DataArray = resample_like(
                da, target, method="conservative"
            )  # resample to the model grid
        da.attrs["_FillValue"] = np.nan
        self.logger.info(f"Completed {variable_name}")
        return da

    def setup_1800arcsec_variables_isimip(
        self,
        forcing: str,
        variables: List[str],
    ):
        """Sets up the high-resolution climate variables for GEB.

        Args:
            forcing: The forcing data to use. Supported values are 'ipsl-cm6a-lr', 'gfdl-esm4', 'mpi-esm1-2-hr', 'mri-esm2-0', or 'mri-esm2-0'.
            variables: The list of climate variables to set up.

        Notes:
            This method sets up the high-resolution climate variables for GEB. It downloads the specified
            climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
            `download_isimip` method.

            The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
            then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

            The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """
        for variable in variables:
            da: xr.DataArray = self.construct_ISIMIP_variable(
                variable, forcing, self.ISIMIP_ssp
            )
            getattr(self, f"set_{variable}")(da)

    def setup_30arcsec_variables_isimip(self, variables: List[str]):
        """Sets up the high-resolution climate variables for GEB.

        Parameters
        ----------
        variables : list of str
            The list of climate variables to set up.
        folder: str
            The folder to save the forcing data in.

        Notes:
        -----
        This method sets up the high-resolution climate variables for GEB. It downloads the specified
        climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
        `download_isimip` method.

        The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
        then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

        The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """

        def download_variable(variable):
            self.logger.info(f"Setting up {variable}...")
            ds: xr.DataArray = self.download_isimip(
                product="InputData",
                variable=variable,
                start_date=self.start_date,
                end_date=self.end_date,
                forcing="chelsa-w5e5",
                resolution="30arcsec",
            )
            var: xr.DataArray = ds[variable].raster.clip_bbox(ds.raster.bounds)
            # TODO: Due to the offset of the MERIT grid, the snapping to the MERIT grid is not perfect
            # and thus the snapping needs to consider quite a large tollerance. We can consider interpolating
            # or this may be fixed when the we go to HydroBASINS v2
            var: xr.DataArray = self.snap_to_grid(
                var, self.grid, relative_tollerance=0.07
            )
            var.attrs["_FillValue"] = np.nan
            self.logger.info(f"Completed {variable}")
            getattr(self, f"set_{variable}")(var)

        for variable in variables:
            download_variable(variable)

    def setup_hurs_isimip_30arcsec(self):
        """Sets up the relative humidity data for GEB.

        Notes:
            This method sets up the relative humidity data for GEB. It first downloads the relative humidity
            data from the ISIMIP dataset for the specified time period using the `download_isimip` method. The data is downloaded
            at a 30 arcsec resolution.

            The method then downloads the monthly CHELSA-BIOCLIM+ relative humidity data at 30 arcsec resolution from the data
            catalog. The data is downloaded for each month in the specified time period and is clipped to the bounding box of
            the downloaded relative humidity data using the `clip_bbox` method of the `raster` object.

            The original ISIMIP data is then downscaled using the monthly CHELSA-BIOCLIM+ data. The downscaling method is adapted
            from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

            The resulting relative humidity data is set as forcing data in the model with names of the form 'climate/hurs'.
        """
        hurs_30_min: xr.DataArray = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        )["hurs"]  # some buffer to avoid edge effects / errors in ISIMIP API

        # just taking the years to simplify things
        start_year: int = self.start_date.year
        end_year: int = self.end_date.year

        chelsa_folder: Path = (
            self.preprocessing_dir / "climate" / "chelsa-bioclim+" / "hurs"
        )
        chelsa_folder.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Downloading/reading monthly CHELSA-BIOCLIM+ hurs data at 30 arcsec resolution"
        )
        hurs_ds_30sec: list[xr.DataArray] = []
        hurs_time: list[str] = []
        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                fp: Path = chelsa_folder / f"hurs_{year}_{month:02d}.zarr"
                if not fp.exists():
                    url: str = self.data_catalog.get_source(
                        f"CHELSA-BIOCLIM+_monthly_hurs_{month:02d}_{year}"
                    ).path
                    hurs: xr.DataArray = rxr.open_rasterio(
                        url,
                    ).isel(band=0)
                    hurs: xr.DataArray = hurs.isel(
                        get_window(hurs.x, hurs.y, hurs_30_min.raster.bounds, buffer=1)
                    )

                    hurs: xr.DataArray = to_zarr(hurs, fp, crs=4326, progress=False)
                else:
                    hurs: xr.DataArray = open_zarr(fp)
                hurs_ds_30sec.append(hurs)
                hurs_time.append(f"{year}-{month:02d}")

        hurs_ds_30sec: xr.DataArray = xr.concat(hurs_ds_30sec, dim="time")
        hurs_ds_30sec["time"] = pd.date_range(hurs_time[0], hurs_time[-1], freq="MS")

        hurs_output: xr.DataArray = self.full_like(
            self.other["climate/tas"], fill_value=np.nan, nodata=np.nan
        )
        hurs_ds_30sec.raster.set_crs(4326)

        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                start_month = datetime(year, month, 1)
                end_month = datetime(year, month, monthrange(year, month)[1])

                w5e5_30min_sel = hurs_30_min.sel(time=slice(start_month, end_month))
                w5e5_regridded = resample_like(
                    w5e5_30min_sel, hurs_ds_30sec, method="conservative"
                )
                w5e5_regridded = w5e5_regridded * 0.01  # convert to fraction

                w5e5_regridded_mean = w5e5_regridded.mean(
                    dim="time"
                )  # get monthly mean
                w5e5_regridded_tr = np.log(
                    w5e5_regridded / (1 - w5e5_regridded)
                )  # assume beta distribuation => logit transform
                w5e5_regridded_mean_tr = np.log(
                    w5e5_regridded_mean / (1 - w5e5_regridded_mean)
                )  # logit transform

                chelsa = (
                    hurs_ds_30sec.sel(time=start_month) * 0.0001
                )  # convert to fraction

                chelsa_tr = np.log(
                    chelsa / (1 - chelsa)
                )  # assume beta distribuation => logit transform

                difference = chelsa_tr - w5e5_regridded_mean_tr

                # apply difference to w5e5
                w5e5_regridded_tr_corr = w5e5_regridded_tr + difference
                w5e5_regridded_corr = (
                    1 / (1 + np.exp(-w5e5_regridded_tr_corr))
                ) * 100  # back transform
                w5e5_regridded_corr.raster.set_crs(4326)
                w5e5_regridded_corr_clipped = w5e5_regridded_corr.raster.clip_bbox(
                    hurs_output.raster.bounds
                )

                hurs_output.loc[dict(time=slice(start_month, end_month))] = (
                    self.snap_to_grid(
                        w5e5_regridded_corr_clipped,
                        hurs_output,
                        relative_tollerance=0.07,
                    )
                )

        self.set_hurs(hurs_output)

    def setup_longwave_isimip_30arcsec(self):
        """Sets up the longwave radiation data for GEB.

        Notes:
            This method sets up the longwave radiation data for GEB. It first downloads the relative humidity,
            air temperature, and downward longwave radiation data from the ISIMIP dataset for the specified time period using the
            `download_isimip` method. The data is downloaded at a 30 arcsec resolution.

            The method then regrids the downloaded data to the target grid using the `xe.Regridder` method. It calculates the
            saturation vapor pressure, water vapor pressure, clear-sky emissivity, all-sky emissivity, and cloud-based component
            of emissivity for the coarse and fine grids. It then downscales the longwave radiation data for the fine grid using
            the calculated all-sky emissivity and Stefan-Boltzmann constant. The downscaling method is adapted
            from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

            The resulting longwave radiation data is set as forcing data in the model with names of the form 'climate/rlds'.
        """
        x1: float = 0.43
        x2: float = 5.7
        sbc: float = 5.67e-8  # stefan boltzman constant [Js−1 m−2 K−4]

        es0: float = 6.11  # reference saturation vapour pressure  [hPa]
        T0: float = 273.15
        lv: float = 2.5e6  # latent heat of vaporization of water
        Rv: float = 461.5  # gas constant for water vapour [J K kg-1]

        hurs_coarse: xr.DataArray = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).hurs  # some buffer to avoid edge effects / errors in ISIMIP API
        tas_coarse: xr.DataArray = self.download_isimip(
            product="SecondaryInputData",
            variable="tas",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).tas  # some buffer to avoid edge effects / errors in ISIMIP API
        rlds_coarse: xr.DataArray = self.download_isimip(
            product="SecondaryInputData",
            variable="rlds",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).rlds  # some buffer to avoid edge effects / errors in ISIMIP API

        target: xr.DataArray = self.other["climate/hurs"]
        target.raster.set_crs(4326)

        hurs_coarse_regridded: xr.DataArray = resample_like(
            hurs_coarse, target, method="conservative"
        )

        tas_coarse_regridded: xr.DataArray = resample_like(
            tas_coarse, target, method="conservative"
        )
        rlds_coarse_regridded: xr.DataArray = resample_like(
            rlds_coarse, target, method="conservative"
        )

        hurs_fine: xr.DataArray = self.other["climate/hurs"]
        tas_fine: xr.DataArray = self.other["climate/tas"]

        # now ready for calculation:
        es_coarse = es0 * np.exp(
            (lv / Rv) * (1 / T0 - 1 / tas_coarse_regridded)
        )  # saturation vapor pressure
        pV_coarse = (
            hurs_coarse_regridded * es_coarse
        ) / 100  # water vapor pressure [hPa]

        es_fine = es0 * np.exp((lv / Rv) * (1 / T0 - 1 / tas_fine))
        pV_fine = (hurs_fine * es_fine) / 100  # water vapour pressure [hPa]

        e_cl_coarse = 0.23 + x1 * ((pV_coarse * 100) / tas_coarse_regridded) ** (1 / x2)
        # e_cl_coarse == clear-sky emissivity w5e5 (pV needs to be in Pa not hPa, hence *100)
        e_cl_fine = 0.23 + x1 * ((pV_fine * 100) / tas_fine) ** (1 / x2)
        # e_cl_fine == clear-sky emissivity target grid (pV needs to be in Pa not hPa, hence *100)

        e_as_coarse = rlds_coarse_regridded / (
            sbc * tas_coarse_regridded**4
        )  # all-sky emissivity w5e5
        e_as_coarse = xr.where(
            e_as_coarse < 1, e_as_coarse, 1
        )  # constrain all-sky emissivity to max 1
        delta_e = e_as_coarse - e_cl_coarse  # cloud-based component of emissivity w5e5

        e_as_fine = e_cl_fine + delta_e
        e_as_fine = xr.where(
            e_as_fine < 1, e_as_fine, 1
        )  # constrain all-sky emissivity to max 1
        lw_fine = (
            e_as_fine * sbc * tas_fine**4
        )  # downscaled lwr! assume cloud e is the same

        lw_fine: xr.DataArray = self.snap_to_grid(
            lw_fine, self.grid, relative_tollerance=0.07
        )
        self.set_rlds(lw_fine)

    def setup_pressure_isimip_30arcsec(self):
        """Sets up the surface pressure data for GEB.

        Parameters
        ----------
        folder: str
            The folder to save the forcing data in.

        Notes:
        -----
        This method sets up the surface pressure data for GEB. It then downloads
        the orography data and surface pressure data from the ISIMIP dataset for the specified time period using the
        `download_isimip` method. The data is downloaded at a 30 arcsec resolution.

        The method then regrids the orography and surface pressure data to the target grid using the `xe.Regridder` method.
        It corrects the surface pressure data for orography using the gravitational acceleration, molar mass of
        dry air, universal gas constant, and sea level standard temperature. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting surface pressure data is set as forcing data in the model with names of the form 'climate/ps'.
        """
        g = 9.80665  # gravitational acceleration [m/s2]
        M = 0.02896968  # molar mass of dry air [kg/mol]
        r0 = 8.314462618  # universal gas constant [J/(mol·K)]
        T0 = 288.16  # Sea level standard temperature  [K]

        pressure_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="psl",  # pressure at sea level
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).psl  # some buffer to avoid edge effects / errors in ISIMIP API

        target = self.other["climate/hurs"]
        target.raster.set_crs(4326)

        orography = self.download_isimip(
            product="InputData", variable="orog", forcing="chelsa-w5e5", buffer=1
        ).orog  # some buffer to avoid edge effects / errors in ISIMIP API
        # TODO: This can perhaps be a clipped version of the orography data
        orography = resample_like(orography, target, method="bilinear")

        # pressure at sea level, so we can do bilinear interpolation before
        # applying the correction for orography
        pressure_30_min_regridded = resample_like(
            pressure_30_min, target, method="conservative"
        )

        pressure_30_min_regridded_corr = pressure_30_min_regridded * np.exp(
            -(g * orography * M) / (T0 * r0)
        )

        self.set_ps(pressure_30_min_regridded_corr)

    def setup_wind_isimip_30arcsec(self):
        """This method sets up the wind data for GEB.

        It first downloads the global wind atlas data and
        regrids it to the target grid using the `xe.Regridder` method. It then downloads the 30-minute average wind data
        from the ISIMIP dataset for the specified time period and regrids it to the target grid using the `xe.Regridder`
        method.

        The method then creates a diff layer by assuming that wind follows a Weibull distribution and taking the log
        transform of the wind data. It then subtracts the log-transformed 30-minute average wind data from the
        log-transformed global wind atlas data to create the diff layer.

        The method then downloads the wind data from the ISIMIP dataset for the specified time period and regrids it to the
        target grid using the `xe.Regridder` method. It applies the diff layer to the log-transformed wind data and then
        exponentiates the result to obtain the corrected wind data. The downscaling method is adapted
        from https://github.com/johanna-malle/w5e5_downscale, which was licenced under GNU General Public License v3.0.

        The resulting wind data is set as forcing data in the model with names of the form 'climate/wind'.

        Currently the global wind atlas database is offline, so the correction is removed
        """
        global_wind_atlas = self.data_catalog.get_rasterdataset(
            "global_wind_atlas", bbox=self.grid.raster.bounds, buffer=10
        )
        target = self.grid["mask"]
        target.raster.set_crs(4326)

        global_wind_atlas_regridded = resample_like(
            global_wind_atlas, target, method="conservative"
        )

        wind_30_min_avg = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            start_date=date(2008, 1, 1),
            end_date=date(2017, 12, 31),
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind.mean(
            dim="time"
        )  # some buffer to avoid edge effects / errors in ISIMIP API

        wind_30_min_avg_regridded = resample_like(
            wind_30_min_avg, target, method="conservative"
        )

        # create diff layer:
        # assume wind follows weibull distribution => do log transform
        wind_30_min_avg_regridded_log = np.log(wind_30_min_avg_regridded)

        global_wind_atlas_regridded_log = np.log(global_wind_atlas_regridded)

        diff_layer = (
            global_wind_atlas_regridded_log - wind_30_min_avg_regridded_log
        )  # to be added to log-transformed daily

        wind_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            start_date=self.start_date,
            end_date=self.end_date,
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind  # some buffer to avoid edge effects / errors in ISIMIP API

        wind_30min_regridded = resample_like(wind_30_min, target, method="conservative")
        wind_30min_regridded_log = np.log(wind_30min_regridded)

        wind_30min_regridded_log_corr = wind_30min_regridded_log + diff_layer
        wind_30min_regridded_corr = np.exp(wind_30min_regridded_log_corr)

        wind_output_clipped = wind_30min_regridded_corr.raster.clip_bbox(
            self.grid.raster.bounds
        )

        wind_output_clipped = self.snap_to_grid(wind_output_clipped, self.grid)
        self.set_sfcwind(wind_output_clipped)

    def setup_SPEI(
        self,
        calibration_period_start: date = date(1981, 1, 1),
        calibration_period_end: date = date(2010, 1, 1),
        window_months: int = 12,
    ) -> None:
        """Sets up the Standardized Precipitation Evapotranspiration Index (SPEI).

        Note that due to the sliding window, the SPEI data will be shorter than the original data. When
        a sliding window of 12 months is used, the SPEI data will be shorter by 11 months.

        Also sets up the Generalized Extreme Value (GEV) parameters for the SPEI data, being
        the c shape (ξ), loc location (μ), and scale (σ) parameters.

        The chunks for the climate data are optimized for reading the data in xy-direction. However,
        for the SPEI calculation, the data is needs to be read in time direction. Therefore, we
        create an intermediate temporary file of the water balance wher chunks are in an intermediate
        size between the xy and time chunks.

        Args:
            calibration_period_start: The start time of the reSPEI data in ISO 8601 format (YYYY-MM-DD).
            calibration_period_end: The end time of the SPEI data in ISO 8601 format (YYYY-MM-DD). Endtime is exclusive.
            window_months: The window size in months for the SPEI calculation. Default is 12 months.
        """
        self.logger.info("setting up SPEI...")

        assert window_months <= 12, (
            "window_months must be less than or equal to 12 (otherwise we run out of climate data)"
        )
        assert window_months >= 1, (
            "window_months must be greater than or equal to 1 (otherwise we have no sliding window)"
        )

        # assert input data have the same coordinates
        assert np.array_equal(
            self.other["climate/pr"].x, self.other["climate/tasmin"].x
        )
        assert np.array_equal(
            self.other["climate/pr"].x, self.other["climate/tasmax"].x
        )
        assert np.array_equal(
            self.other["climate/pr"].y, self.other["climate/tasmin"].y
        )
        assert np.array_equal(
            self.other["climate/pr"].y, self.other["climate/tasmax"].y
        )
        if not self.other[
            "climate/pr"
        ].time.min().dt.date <= calibration_period_start and self.other[
            "climate/pr"
        ].time.max().dt.date >= calibration_period_end - timedelta(days=1):
            forcing_start_date = self.other["climate/pr"].time.min().dt.date.item()
            forcing_end_date = self.other["climate/pr"].time.max().dt.date.item()
            raise AssertionError(
                f"water data does not cover the entire calibration period, forcing data covers from {forcing_start_date} to {forcing_end_date}, "
                f"while requested calibration period is from {calibration_period_start} to {calibration_period_end}"
            )

        pet = xci.potential_evapotranspiration(
            tasmin=self.other["climate/tasmin"],
            tasmax=self.other["climate/tasmax"],
            # hurs=self.other["climate/hurs"],
            # rsds=self.other["climate/rsds"],
            # rlds=self.other["climate/rlds"],
            # rsus=self.full_like(
            #     self.other["climate/rsds"],
            #     fill_value=0,
            #     nodata=np.nan,
            #     attrs=self.other["climate/rsds"].attrs,
            # ),
            # rlus=self.full_like(
            #     self.other["climate/rsds"],
            #     fill_value=0,
            #     nodata=np.nan,
            #     attrs=self.other["climate/rsds"].attrs,
            # ),
            # sfcWind=self.other["climate/sfcwind"],
            method="BR65",
        ).astype(np.float32)

        # Compute the potential evapotranspiration
        water_budget = xci.water_budget(pr=self.other["climate/pr"], evspsblpot=pet)

        water_budget = water_budget.resample(time="MS").mean(keep_attrs=True)
        water_budget.attrs["_FillValue"] = np.nan

        temp_xy_chunk_size = 50

        with tempfile.TemporaryDirectory() as tmp_water_budget_folder:
            tmp_water_budget_file = (
                Path(tmp_water_budget_folder) / "tmp_water_budget_file.zarr"
            )
            self.logger.info("Exporting temporary water budget to zarr")
            water_budget = to_zarr(
                water_budget,
                tmp_water_budget_file,
                crs=4326,
                x_chunksize=temp_xy_chunk_size,
                y_chunksize=temp_xy_chunk_size,
                time_chunksize=50,
                time_chunks_per_shard=None,
            ).chunk({"time": -1})  # for the SPEI calculation time must not be chunked

            # We set freq to None, so that the input frequency is used (no recalculating)
            # this means that we can calculate SPEI much more efficiently, as it is not
            # rechunked in the xclim package

            # The log-logistic distribution used in SPEI has three parameters: scale, shape, and location.
            # In practice, fixing the location (floc) to 0 simplifies the fitting process and often
            # provides satisfactory results.The fitting is then effectively done with two parameters,
            # also reducing the risk of overfitting, especially with limited data.
            # When empirical data suggest that the climatic water balance values are significantly shifted,
            # a non-zero floc may better fit the distribution. However, this is not typical in routine applications.
            SPEI = xci.standardized_precipitation_evapotranspiration_index(
                wb=water_budget,
                cal_start=calibration_period_start.strftime("%Y-%m-%d"),
                cal_end=calibration_period_end.strftime("%Y-%m-%d"),
                freq=None,
                window=window_months,
                dist="fisk",  # log-logistic distribution
                method="APP",  # approximative method
                fitkwargs={
                    "floc": water_budget.min().compute().item()
                },  # location parameter, assures that the distribution is always positive
            ).astype(np.float32)

            # remove all nan values as a result of the sliding window
            SPEI: xr.DataArray = SPEI.isel(time=slice(window_months - 1, None))

            with tempfile.TemporaryDirectory() as tmp_spei_folder:
                tmp_spei_file = Path(tmp_spei_folder) / "tmp_spei_file.zarr"
                self.logger.info("Calculating SPEI and exporting to temporary file...")
                SPEI.attrs = {
                    "_FillValue": np.nan,
                }
                SPEI: xr.DataArray = to_zarr(
                    SPEI,
                    tmp_spei_file,
                    x_chunksize=temp_xy_chunk_size,
                    y_chunksize=temp_xy_chunk_size,
                    time_chunksize=10,
                    time_chunks_per_shard=None,
                    crs=4326,
                )

                self.set_SPEI(SPEI)

                self.logger.info("calculating GEV parameters...")

                # Group the data by year and find the maximum monthly sum for each year
                SPEI_yearly_min = SPEI.groupby("time.year").min(dim="time", skipna=True)
                SPEI_yearly_min = SPEI_yearly_min.dropna(dim="year")
                SPEI_yearly_min = (
                    SPEI_yearly_min.rename({"year": "time"})
                    .chunk({"time": -1})
                    .compute()
                )

                GEV = xci.stats.fit(SPEI_yearly_min, dist="genextreme").compute()

                self.set_grid(
                    GEV.sel(dparams="c").astype(np.float32), name="climate/gev_c"
                )
                self.set_grid(
                    GEV.sel(dparams="loc").astype(np.float32), name="climate/gev_loc"
                )
                self.set_grid(
                    GEV.sel(dparams="scale").astype(np.float32),
                    name="climate/gev_scale",
                )

    def get_elevation_forcing_and_grid(
        self, grid: xr.DataArray, forcing_grid: xr.DataArray, forcing_name
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Gets elevation maps for both the normal grid (target of resampling) and the forcing grid.

        Args:
            grid: the normal grid to which the forcing data is resampled
            forcing_grid: grid of the forcing data
            forcing_name: name of the forcing data, used to determine the file paths for caching

        Returns:
            elevation data for the forcing grid and the normal grid

        """
        # we also need to process the elevation data for the grid, because the
        # elevation data that is available in the model is masked to the grid
        elevation_forcing_fp: Path = (
            self.preprocessing_dir / "climate" / forcing_name / "DEM_forcing.zarr"
        )
        elevation_grid_fp: Path = self.preprocessing_dir / "climate" / "DEM.zarr"
        if elevation_forcing_fp.exists() and elevation_grid_fp.exists():
            elevation_forcing: xr.DataArray = open_zarr(elevation_forcing_fp)
            elevation_grid: xr.DataArray = open_zarr(elevation_grid_fp)
        else:
            elevation: xr.DataArray = xr.open_dataarray(
                self.data_catalog.get_source("fabdem").path
            )
            elevation: xr.DataArray = (
                elevation.isel(
                    band=0,
                    **get_window(
                        elevation.x, elevation.y, forcing_grid.rio.bounds(), buffer=500
                    ),
                )
                .raster.mask_nodata()
                .fillna(0)
                .chunk({"x": 2000, "y": 2000})
            )
            elevation_forcing: xr.DataArray = resample_chunked(
                elevation,
                forcing_grid.isel(time=0).chunk({"x": 10, "y": 10}),
                method="bilinear",
            )
            elevation_forcing: xr.DataArray = to_zarr(
                elevation_forcing,
                elevation_forcing_fp,
                crs=4326,
            )
            elevation_grid: xr.DataArray = resample_chunked(
                elevation, grid.chunk({"x": 50, "y": 50}), method="bilinear"
            )
            elevation_grid: xr.DataArray = to_zarr(
                elevation_grid, elevation_grid_fp, crs=4326
            )

        return elevation_forcing.chunk({"x": -1, "y": -1}), elevation_grid.chunk(
            {"x": -1, "y": -1}
        )

    def setup_CO2_concentration(self) -> None:
        """Aquires the CO2 concentration data for the specified SSP in ppm."""
        da: xr.DataArray = self.construct_ISIMIP_variable(
            variable_name="co2",
            forcing=None,
            ssp=self.ISIMIP_ssp,
        ).astype(np.float32)
        self.set_other(
            da,
            name="climate/CO2_ppm",
        )
