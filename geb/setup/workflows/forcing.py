import cdsapi
import calendar
import numpy as np
import xarray as xr
import concurrent.futures
import pandas as pd
from datetime import date, timedelta


def reproject_and_apply_lapse_rate_temperature(T, DEM, grid_mask, lapse_rate=-0.0065):
    DEM.raster.mask_nodata().fillna(
        0
    )  # assuming 0 for missing DEM values above the ocean
    DEM_grid = DEM.raster.reproject_like(grid_mask, method="average")
    if "dparams" in DEM_grid.coords:
        DEM_grid = DEM_grid.drop_vars(["dparams"])
    if "inplace" in DEM_grid.coords:
        DEM_grid = DEM_grid.drop_vars(["inplace"])
    DEM_forcing = DEM.raster.reproject_like(T, method="average")

    t_at_sea_level = T - DEM_forcing * lapse_rate
    t_at_sea_level_reprojected = t_at_sea_level.raster.reproject_like(
        DEM_grid, method="average"
    )
    T_grid = t_at_sea_level_reprojected + lapse_rate * DEM_grid
    T_grid.name = T.name
    T_grid.attrs = T.attrs
    return T_grid


def get_pressure_correction_factor(DEM, g, Mo, lapse_rate):
    return (288.15 / (288.15 + lapse_rate * DEM)) ** (g * Mo / (8.3144621 * lapse_rate))


def reproject_and_apply_lapse_rate_pressure(
    pressure,
    DEM,
    grid_mask,
    g=9.80665,
    Mo=0.0289644,
    lapse_rate=-0.0065,
):
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

    Returns
    -------
    press_fact : xarray.DataArray
        pressure correction factor
    """
    DEM.raster.mask_nodata().fillna(
        0
    )  # assuming 0 for missing DEM values above the ocean
    DEM_grid = DEM.raster.reproject_like(grid_mask, method="average")
    if "dparams" in DEM_grid.coords:
        DEM_grid = DEM_grid.drop_vars(["dparams"])
    if "inplace" in DEM_grid.coords:
        DEM_grid = DEM_grid.drop_vars(["inplace"])
    DEM_forcing = DEM.raster.reproject_like(pressure, method="average")

    pressure_at_sea_level = pressure / get_pressure_correction_factor(
        DEM_forcing, g, Mo, lapse_rate
    )  # divide by pressure factor to get pressure at sea level
    pressure_at_sea_level_reprojected = pressure_at_sea_level.raster.reproject_like(
        DEM_grid, method="average"
    )
    pressure_grid = (
        pressure_at_sea_level_reprojected
        * get_pressure_correction_factor(DEM_grid, g, Mo, lapse_rate)
    )  # multiply by pressure factor to get pressure at DEM grid, corrected for elevation
    pressure_grid.name = pressure.name
    pressure_grid.attrs = pressure.attrs

    return pressure_grid


def download_ERA5(
    folder,
    variables: list,
    starttime: date,
    endtime: date,
    bounds: tuple,
    logger=None,
):
    # https://cds.climate.copernicus.eu/cdsapp#!/software/app-c3s-daily-era5-statistics?tab=appcode
    # https://earthscience.stackexchange.com/questions/24156/era5-single-level-calculate-relative-humidity
    """
    Download hourly ERA5 data for a specified time frame and bounding box.

    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    """

    folder.mkdir(parents=True, exist_ok=True)

    def download(month_start):
        days_in_month = calendar.monthrange(month_start.year, month_start.month)[1]
        month_end = month_start + timedelta(days=days_in_month)
        output_fn = (
            folder
            / f"{month_start.strftime("%Y%m%d")}_{month_end.strftime("%Y%m%d")}.nc"
        )
        if output_fn.exists():
            if logger:
                logger.info(f"ERA5 data already downloaded to {output_fn}")
        else:
            (xmin, ymin, xmax, ymax) = bounds

            # add buffer to bounding box. Resolution is 0.1 degrees, so add 0.1 degrees to each side
            xmin -= 0.1
            ymin -= 0.1
            xmax += 0.1
            ymax += 0.1

            max_retries = 10
            retries = 0
            while retries < max_retries:
                try:
                    request = {
                        "product_type": "reanalysis",
                        "format": "grib",
                        "download_format": "unarchived",
                        "variable": variables,
                        "year": [f"{month_start.year}"],
                        "month": [f"{month_start.month:02d}"],
                        "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
                        "time": [
                            "00:00",
                            "01:00",
                            "02:00",
                            "03:00",
                            "04:00",
                            "05:00",
                            "06:00",
                            "07:00",
                            "08:00",
                            "09:00",
                            "10:00",
                            "11:00",
                            "12:00",
                            "13:00",
                            "14:00",
                            "15:00",
                            "16:00",
                            "17:00",
                            "18:00",
                            "19:00",
                            "20:00",
                            "21:00",
                            "22:00",
                            "23:00",
                        ],
                        "area": (
                            float(ymax),
                            float(xmin),
                            float(ymin),
                            float(xmax),
                        ),  # North, West, South, East
                    }
                    cdsapi.Client().retrieve(
                        "reanalysis-era5-land",
                        request,
                        output_fn,
                    )
                    break
                except Exception as e:
                    print(f"Download failed. Retrying... ({retries+1}/{max_retries})")
                    print(e)
                    print(request)
                    retries += 1
            if retries == max_retries:
                raise Exception("Download failed after maximum retries.")
        return output_fn

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        month_starts = pd.date_range(starttime, endtime, freq="MS").date.tolist()
        files = list(executor.map(download, month_starts))

    return files


def open_ERA5(files, variable, xy_chunksize):
    def preprocess(ds):
        # drop all variables except the variable of interest
        valid_time = (ds["time"] + ds["step"]).values.flatten()
        da = ds[variable].stack(valid_time=("time", "step"))
        da = da.assign_coords(valid_time=("valid_time", valid_time))
        da = da.rename({"valid_time": "time"})
        da = da.isel(time=slice(23, -1))
        return da

    ds = xr.open_mfdataset(
        files,
        compat="equals",  # all values and dimensions must be the same,
        combine_attrs="drop_conflicts",  # drop conflicting attributes
        engine="cfgrib",
        preprocess=preprocess,
    ).rio.set_crs(4326)

    assert "time" in ds.dims
    assert "latitude" in ds.dims
    assert "longitude" in ds.dims

    # assert that time is monotonically increasing with a constant step size
    assert (
        ds.time.diff("time").astype(np.int64)
        == (ds.time[1] - ds.time[0]).astype(np.int64)
    ).all(), "time is not monotonically increasing with a constant step size"

    # remove first time step.
    # This is an accumulation from the previous day and thus cannot be calculated
    ds = ds.isel(time=slice(1, None))
    ds = ds.chunk({"time": 24, "latitude": xy_chunksize, "longitude": xy_chunksize})
    # the ERA5 grid is sometimes not exactly regular. The offset is very minor
    # therefore we snap the grid to a regular grid, to save huge computational time
    # for a infenitesimal loss in accuracy
    ds = ds.assign_coords(
        latitude=np.linspace(
            ds["latitude"][0].item(),
            ds["latitude"][-1].item(),
            ds["latitude"].size,
            endpoint=True,
        ),
        longitude=np.linspace(
            ds["longitude"][0].item(),
            ds["longitude"][-1].item(),
            ds["longitude"].size,
            endpoint=True,
        ),
    )

    ds.raster.set_crs(4326)
    # the last few months of data may come from ERA5T (expver 5) instead of ERA5 (expver 1)
    # if so, combine that dimension
    if "expver" in ds.dims:
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # assert there is only one data variable
    assert len(ds.data_vars) == 1

    # select the variable and rename longitude and latitude variable
    ds = ds[list(ds.data_vars)[0]].rename({"longitude": "x", "latitude": "y"})

    if ds.attrs["GRIB_stepType"] == "accum":

        def xr_ERA5_accumulation_to_hourly(ds, dim):
            # Identify the axis number for the given dimension
            assert ds.time.dt.hour[0] == 1, "First time step must be at 1 UTC"
            # All chunksizes must be divisible by 24, except the last one
            assert all(chunksize % 24 == 0 for chunksize in ds.chunksizes["time"][:-1])

            def diff_with_prepend(data, dim):
                # Assert dimension is a multiple of 24
                # As the first hour is an accumulation from the first hour of the day, prepend a 0
                # to the data array before taking the diff. In this way, the output is also 24 hours
                return np.diff(data, prepend=0, axis=dim)

            # Apply the custom diff function using apply_ufunc
            return xr.apply_ufunc(
                diff_with_prepend,  # The function to apply
                ds,  # The DataArray or Dataset to which the function will be applied
                kwargs={
                    "dim": ds.get_axis_num(dim)
                },  # Additional arguments for the function
                dask="parallelized",  # Enable parallelized computation
                output_dtypes=[ds.dtype],  # Specify the output data type
            )

        # The accumulations in the short forecasts of ERA5-Land (with hourly steps from 01 to 24) are treated
        # the same as those in ERA-Interim or ERA-Interim/Land, i.e., they are accumulated from the beginning
        # of the forecast to the end of the forecast step. For example, runoff at day=D, step=12 will provide
        # runoff accumulated from day=D, time=0 to day=D, time=12. The maximum accumulation is over 24 hours,
        # i.e., from day=D, time=0 to day=D+1,time=0 (step=24).
        # forecasts are the difference between the current and previous time step
        hourly = xr_ERA5_accumulation_to_hourly(ds, "time")
    elif ds.attrs["GRIB_stepType"] == "instant":
        hourly = ds
    else:
        raise NotImplementedError

    return hourly
