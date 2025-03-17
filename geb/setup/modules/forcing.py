import numpy as np
import xarray as xr
import pandas as pd
from datetime import date, timedelta, datetime
import tempfile
import xclim.indices as xci
from calendar import monthrange
from typing import List
from tqdm import tqdm
from pathlib import Path

from ...workflows.io import to_zarr, open_zarr


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


def download_ERA5(folder, variable, starttime, endtime, bounds, logger):
    output_fn = folder / f"{variable}.zarr"
    if output_fn.exists():
        da = open_zarr(output_fn)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        da = xr.open_dataset(
            "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",
            storage_options={"client_kwargs": {"trust_env": True}},
            chunks={},
            engine="zarr",
        )[variable].rename({"valid_time": "time", "latitude": "y", "longitude": "x"})

        da = da.drop_vars(["number", "surface", "depthBelowLandLayer"])

        da = da.sel(
            time=slice(starttime, endtime),
            y=slice(bounds[3], bounds[1]),
            x=slice(bounds[0], bounds[2]),
        )
        da = da.isel(time=slice(1, None))

        logger.info(f"Downloading ERA5 {variable} to {output_fn}")
        da = to_zarr(
            da,
            output_fn,
            time_chunksize=24,
        )
    return da


def process_ERA5(variable, folder, starttime, endtime, bounds, logger):
    da = download_ERA5(folder, variable, starttime, endtime, bounds, logger)
    # assert that time is monotonically increasing with a constant step size
    assert (
        da.time.diff("time").astype(np.int64)
        == (da.time[1] - da.time[0]).astype(np.int64)
    ).all(), "time is not monotonically increasing with a constant step size"
    if da.attrs["GRIB_stepType"] == "accum":

        def xr_ERA5_accumulation_to_hourly(da, dim):
            # Identify the axis number for the given dimension
            assert da.time.dt.hour[0] == 1, "First time step must be at 1 UTC"
            # All chunksizes must be divisible by 24, except the last one
            assert all(chunksize % 24 == 0 for chunksize in da.chunksizes["time"][:-1])

            def diff_with_prepend(data, dim):
                # Assert dimension is a multiple of 24
                # As the first hour is an accumulation from the first hour of the day, prepend a 0
                # to the data array before taking the diff. In this way, the output is also 24 hours
                return np.diff(data, prepend=0, axis=dim)

            # Apply the custom diff function using apply_ufunc
            return xr.apply_ufunc(
                diff_with_prepend,  # The function to apply
                da,  # The DataArray or Dataset to which the function will be applied
                kwargs={
                    "dim": da.get_axis_num(dim)
                },  # Additional arguments for the function
                dask="parallelized",  # Enable parallelized computation
                output_dtypes=[da.dtype],  # Specify the output data type
                keep_attrs=True,  # Keep the attributes of the input DataArray or Dataset
            )

        # The accumulations in the short forecasts of ERA5-Land (with hourly steps from 01 to 24) are treated
        # the same as those in ERA-Interim or ERA-Interim/Land, i.e., they are accumulated from the beginning
        # of the forecast to the end of the forecast step. For example, runoff at day=D, step=12 will provide
        # runoff accumulated from day=D, time=0 to day=D, time=12. The maximum accumulation is over 24 hours,
        # i.e., from day=D, time=0 to day=D+1,time=0 (step=24).
        # forecasts are the difference between the current and previous time step
        da = xr_ERA5_accumulation_to_hourly(da, "time")
    elif da.attrs["GRIB_stepType"] == "instant":
        da = da
    else:
        raise NotImplementedError

    da = da.rio.set_crs(4326)
    da.raster.set_crs(4326)

    return da


class Forcing:
    def __init__(self):
        pass

    def set_pr_hourly(self, pr_hourly, *args, **kwargs):
        self.set_other(
            pr_hourly, name="climate/pr_hourly", *args, **kwargs, time_chunksize=7 * 24
        )

    def set_pr(self, pr, *args, **kwargs):
        self.set_other(pr, name="climate/pr", *args, **kwargs)

    def set_rsds(self, rsds, *args, **kwargs):
        self.set_other(rsds, name="climate/rsds", *args, **kwargs)

    def set_rlds(self, rlds, *args, **kwargs):
        self.set_other(rlds, name="climate/rlds", *args, **kwargs)

    def set_tas(self, tas, *args, **kwargs):
        self.set_other(tas, name="climate/tas", *args, **kwargs, byteshuffle=True)

    def set_tasmax(self, tasmax, *args, **kwargs):
        self.set_other(tasmax, name="climate/tasmax", *args, **kwargs, byteshuffle=True)

    def set_tasmin(self, tasmin, *args, **kwargs):
        self.set_other(tasmin, name="climate/tasmin", *args, **kwargs, byteshuffle=True)

    def set_hurs(self, hurs, *args, **kwargs):
        self.set_other(hurs, name="climate/hurs", *args, **kwargs, byteshuffle=True)

    def set_ps(self, ps, *args, **kwargs):
        self.set_other(ps, name="climate/ps", *args, **kwargs, byteshuffle=True)

    def set_sfcwind(self, sfcwind, *args, **kwargs):
        self.set_other(
            sfcwind, name="climate/sfcwind", *args, **kwargs, byteshuffle=True
        )

    def set_SPEI(self, SPEI, *args, **kwargs):
        self.set_other(SPEI, name="climate/SPEI", *args, **kwargs, byteshuffle=True)

    def setup_forcing_era5(self, starttime, endtime):
        target = self.grid["mask"]
        target.raster.set_crs(4326)

        download_args = {
            "folder": self.preprocessing_dir / "climate" / "ERA5",
            "starttime": starttime,
            "endtime": endtime,
            "bounds": target.raster.bounds,
            "logger": self.logger,
        }

        pr_hourly = process_ERA5(
            "tp",  # total_precipitation
            **download_args,
        )
        pr_hourly = pr_hourly * (1000 / 3600)  # convert from m/hr to kg/m2/s
        pr_hourly.attrs = {
            "standard_name": "precipitation_flux",
            "long_name": "Precipitation",
            "units": "kg m-2 s-1",
            "_FillValue": np.nan,
        }
        # ensure no negative values for precipitation, which may arise due to float precision
        pr_hourly = xr.where(pr_hourly > 0, pr_hourly, 0, keep_attrs=True)
        self.set_pr_hourly(pr_hourly)  # weekly chunk size

        pr = pr_hourly.resample(time="D").mean()  # get daily mean
        pr = pr.raster.reproject_like(target, method="average")
        self.set_pr(pr)

        hourly_rsds = process_ERA5(
            "ssrd",  # surface_solar_radiation_downwards
            **download_args,
        )
        rsds = hourly_rsds.resample(time="D").sum() / (
            24 * 3600
        )  # get daily sum and convert from J/m2 to W/m2
        rsds.attrs = {
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "long_name": "Surface Downwelling Shortwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }

        rsds = rsds.raster.reproject_like(target, method="average")
        self.set_rsds(rsds)

        hourly_rlds = process_ERA5(
            "strd",  # surface_thermal_radiation_downwards
            **download_args,
        )
        rlds = hourly_rlds.resample(time="D").sum() / (24 * 3600)
        rlds.attrs = {
            "standard_name": "surface_downwelling_longwave_flux_in_air",
            "long_name": "Surface Downwelling Longwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        rlds = rlds.raster.reproject_like(target, method="average")
        self.set_rlds(rlds)

        hourly_tas = process_ERA5("t2m", **download_args)

        DEM = self.data_catalog.get_rasterdataset(
            "fabdem",
            bbox=hourly_tas.raster.bounds,
            buffer=100,
            variables=["fabdem"],
        )

        hourly_tas_reprojected = reproject_and_apply_lapse_rate_temperature(
            hourly_tas, DEM, target
        )

        tas_reprojected = hourly_tas_reprojected.resample(time="D").mean()
        tas_reprojected.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_tas(tas_reprojected)

        tasmax = hourly_tas_reprojected.resample(time="D").max()
        tasmax.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Daily Maximum Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_tasmax(tasmax)

        tasmin = hourly_tas_reprojected.resample(time="D").min()
        tasmin.attrs = {
            "standard_name": "air_temperature",
            "long_name": "Daily Minimum Near-Surface Air Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_tasmin(tasmin)

        dew_point_tas = process_ERA5(
            "d2m",
            **download_args,
        )
        dew_point_tas_reprojected = reproject_and_apply_lapse_rate_temperature(
            dew_point_tas, DEM, target
        )

        water_vapour_pressure = 0.6108 * np.exp(
            17.27
            * (dew_point_tas_reprojected - 273.15)
            / (237.3 + (dew_point_tas_reprojected - 273.15))
        )  # calculate water vapour pressure (kPa)
        saturation_vapour_pressure = 0.6108 * np.exp(
            17.27
            * (hourly_tas_reprojected - 273.15)
            / (237.3 + (hourly_tas_reprojected - 273.15))
        )

        assert water_vapour_pressure.shape == saturation_vapour_pressure.shape
        relative_humidity = (water_vapour_pressure / saturation_vapour_pressure) * 100
        relative_humidity.attrs = {
            "standard_name": "relative_humidity",
            "long_name": "Near-Surface Relative Humidity",
            "units": "%",
            "_FillValue": np.nan,
        }
        relative_humidity = relative_humidity.resample(time="D").mean()
        relative_humidity = relative_humidity.raster.reproject_like(
            target, method="average"
        )
        self.set_hurs(relative_humidity)

        pressure = process_ERA5("sp", **download_args)
        pressure = reproject_and_apply_lapse_rate_pressure(pressure, DEM, target)
        pressure.attrs = {
            "standard_name": "surface_air_pressure",
            "long_name": "Surface Air Pressure",
            "units": "Pa",
            "_FillValue": np.nan,
        }
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
        wind_speed.attrs = {
            "standard_name": "wind_speed",
            "long_name": "Near-Surface Wind Speed",
            "units": "m s-1",
            "_FillValue": np.nan,
        }
        wind_speed = wind_speed.raster.reproject_like(target, method="average")
        self.set_sfcwind(wind_speed)

    def setup_forcing_ISIMIP(self, starttime, endtime, resolution_arcsec, forcing, ssp):
        if resolution_arcsec == 30:
            assert forcing == "chelsa-w5e5", (
                "Only chelsa-w5e5 is supported for 30 arcsec resolution"
            )
            # download source data from ISIMIP
            self.logger.info("setting up forcing data")
            # high_res_variables = ["pr", "rsds", "tas", "tasmax", "tasmin"]
            # self.setup_30arcsec_variables_isimip(high_res_variables, starttime, endtime)
            self.logger.info("setting up relative humidity...")
            # self.setup_hurs_isimip_30arcsec(starttime, endtime)
            self.logger.info("setting up longwave radiation...")
            # self.setup_longwave_isimip_30arcsec(starttime=starttime, endtime=endtime)
            self.logger.info("setting up pressure...")
            self.setup_pressure_isimip_30arcsec(starttime, endtime)
            self.logger.info("setting up wind...")
            self.setup_wind_isimip_30arcsec(starttime, endtime)
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
            self.setup_1800arcsec_variables_isimip(
                forcing, variables, starttime, endtime, ssp=ssp
            )
        else:
            raise ValueError(
                "Only 30 arcsec and 1800 arcsec resolution is supported for ISIMIP data"
            )

    def setup_forcing(
        self,
        starttime: date,
        endtime: date,
        data_source: str = "isimip",
        resolution_arcsec: int = 30,
        forcing: str = "chelsa-w5e5",
        ssp=None,
    ):
        """
        Sets up the forcing data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        data_source : str, optional
            The data source to use for the forcing data. Default is 'isimip'.

        Notes
        -----
        This method sets up the forcing data for GEB. It first downloads the high-resolution variables
        (precipitation, surface solar radiation, air temperature, maximum air temperature, and minimum air temperature) from
        the ISIMIP dataset for the specified time period. The data is downloaded using the `setup_30arcsec_variables_isimip`
        method.

        The method then sets up the relative humidity, longwave radiation, pressure, and wind data for the model. The
        relative humidity data is downloaded from the ISIMIP dataset using the `setup_hurs_isimip_30arcsec` method. The longwave radiation
        data is calculated using the air temperature and relative humidity data and the `calculate_longwave` function. The
        pressure data is downloaded from the ISIMIP dataset using the `setup_pressure_isimip_30arcsec` method. The wind data is downloaded
        from the ISIMIP dataset using the `setup_wind_isimip_30arcsec` method. All these data are first downscaled to the model grid.

        The resulting forcing data is set as forcing data in the model with names of the form 'forcing/{variable_name}'.
        """
        assert starttime < endtime, "Start time must be before end time"
        if data_source == "isimip":
            self.setup_forcing_ISIMIP(
                starttime, endtime, resolution_arcsec, forcing, ssp
            )
        elif data_source == "era5":
            self.setup_forcing_era5(starttime, endtime)
        elif data_source == "cmip":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(f"Unknown data source: {data_source}")

    def setup_1800arcsec_variables_isimip(
        self,
        forcing: str,
        variables: List[str],
        starttime: date,
        endtime: date,
        ssp: str,
    ):
        """
        Sets up the high-resolution climate variables for GEB.

        Parameters
        ----------
        variables : list of str
            The list of climate variables to set up.
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the high-resolution climate variables for GEB. It downloads the specified
        climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
        `download_isimip` method.

        The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
        then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

        The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """

        def download_variable(variable_name, forcing, ssp, starttime, endtime):
            self.logger.info(f"Setting up {variable_name}...")
            first_year_future_climate = 2015
            var = []
            if ssp == "picontrol":
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario=ssp,
                    variable=variable_name,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(ds[variable_name].raster.clip_bbox(ds.raster.bounds))
            if (
                (
                    endtime.year < first_year_future_climate
                    or starttime.year < first_year_future_climate
                )
                and ssp != "picontrol"
            ):  # isimip cutoff date between historic and future climate
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario="historical",
                    variable=variable_name,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(ds[variable_name].raster.clip_bbox(ds.raster.bounds))
            if (
                starttime.year >= first_year_future_climate
                or endtime.year >= first_year_future_climate
            ) and ssp != "picontrol":
                assert ssp is not None, "ssp must be specified for future climate"
                assert ssp != "historical", "historical scenarios run until 2014"
                ds = self.download_isimip(
                    product="InputData",
                    simulation_round="ISIMIP3b",
                    climate_scenario=ssp,
                    variable=variable_name,
                    starttime=starttime,
                    endtime=endtime,
                    forcing=forcing,
                    resolution=None,
                    buffer=1,
                )
                var.append(ds[variable_name].raster.clip_bbox(ds.raster.bounds))

            var = xr.concat(
                var, dim="time", combine_attrs="drop_conflicts", compat="equals"
            )  # all values and dimensions must be the same

            # assert that time is monotonically increasing with a constant step size
            assert (
                ds.time.diff("time").astype(np.int64)
                == (ds.time[1] - ds.time[0]).astype(np.int64)
            ).all(), "time is not monotonically increasing with a constant step size"

            mask = self.grid["mask"]
            mask.raster.set_crs(4326)
            var = var.rename({"lon": "x", "lat": "y"})
            if variable_name in ("tas", "tasmin", "tasmax", "ps"):
                DEM = self.data_catalog.get_rasterdataset(
                    "fabdem",
                    bbox=var.raster.bounds,
                    buffer=100,
                    variables=["fabdem"],
                )
                if variable_name in ("tas", "tasmin", "tasmax"):
                    var = reproject_and_apply_lapse_rate_temperature(var, DEM, mask)
                elif variable_name == "ps":
                    var = reproject_and_apply_lapse_rate_pressure(var, DEM, mask)
                else:
                    raise ValueError
            else:
                var = self.interpolate(var, "linear")
            var.attrs["_FillValue"] = np.nan
            self.logger.info(f"Completed {variable_name}")

            getattr(self, f"set_{variable_name}")(var)

        for variable in variables:
            download_variable(variable, forcing, ssp, starttime, endtime)

    def setup_30arcsec_variables_isimip(
        self, variables: List[str], starttime: date, endtime: date
    ):
        """
        Sets up the high-resolution climate variables for GEB.

        Parameters
        ----------
        variables : list of str
            The list of climate variables to set up.
        starttime : date
            The start time of the forcing data.
        endtime : date
            The end time of the forcing data.
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the high-resolution climate variables for GEB. It downloads the specified
        climate variables from the ISIMIP dataset for the specified time period. The data is downloaded using the
        `download_isimip` method.

        The method renames the longitude and latitude dimensions of the downloaded data to 'x' and 'y', respectively. It
        then clips the data to the bounding box of the model grid using the `clip_bbox` method of the `raster` object.

        The resulting climate variables are set as forcing data in the model with names of the form 'climate/{variable_name}'.
        """

        def download_variable(variable, starttime, endtime):
            self.logger.info(f"Setting up {variable}...")
            ds = self.download_isimip(
                product="InputData",
                variable=variable,
                starttime=starttime,
                endtime=endtime,
                forcing="chelsa-w5e5",
                resolution="30arcsec",
            )
            ds = ds.rename({"lon": "x", "lat": "y"})
            var = ds[variable].raster.clip_bbox(ds.raster.bounds)
            # TODO: Due to the offset of the MERIT grid, the snapping to the MERIT grid is not perfect
            # and thus the snapping needs to consider quite a large tollerance. We can consider interpolating
            # or this may be fixed when the we go to HydroBASINS v2
            var = self.snap_to_grid(var, self.grid, relative_tollerance=0.07)
            var.attrs["_FillValue"] = np.nan
            self.logger.info(f"Completed {variable}")
            getattr(self, f"set_{variable}")(var)

        for variable in variables:
            download_variable(variable, starttime, endtime)

    def setup_hurs_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the relative humidity data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the relative humidity data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the relative humidity data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
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
        hurs_30_min = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        )  # some buffer to avoid edge effects / errors in ISIMIP API

        # just taking the years to simplify things
        start_year = starttime.year
        end_year = endtime.year

        chelsa_folder = self.preprocessing_dir / "climate" / "chelsa-bioclim+" / "hurs"
        chelsa_folder.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Downloading/reading monthly CHELSA-BIOCLIM+ hurs data at 30 arcsec resolution"
        )
        hurs_ds_30sec, hurs_time = [], []
        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                fn = chelsa_folder / f"hurs_{year}_{month:02d}.zarr"
                if not fn.exists():
                    hurs = self.data_catalog.get_rasterdataset(
                        f"CHELSA-BIOCLIM+_monthly_hurs_{month:02d}_{year}",
                        bbox=hurs_30_min.raster.bounds,
                        buffer=1,
                    )
                    hurs = to_zarr(hurs, fn, crs=4326, progress=False)
                else:
                    hurs = open_zarr(fn)
                hurs_ds_30sec.append(hurs)
                hurs_time.append(f"{year}-{month:02d}")

        hurs_ds_30sec = xr.concat(hurs_ds_30sec, dim="time")
        hurs_ds_30sec["time"] = pd.date_range(hurs_time[0], hurs_time[-1], freq="MS")

        hurs_output = self.full_like(
            self.other["climate/tas"], fill_value=np.nan, nodata=np.nan
        )
        hurs_output.attrs = {
            "units": "%",
            "long_name": "Relative humidity",
            "_FillValue": np.nan,
        }
        hurs_ds_30sec.raster.set_crs(4326)

        for year in tqdm(range(start_year, end_year + 1)):
            for month in range(1, 13):
                start_month = datetime(year, month, 1)
                end_month = datetime(year, month, monthrange(year, month)[1])

                w5e5_30min_sel = hurs_30_min.sel(time=slice(start_month, end_month))
                w5e5_regridded = (
                    w5e5_30min_sel.raster.reproject_like(
                        hurs_ds_30sec, method="bilinear"
                    )
                    * 0.01
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
                w5e5_regridded_corr_clipped = w5e5_regridded_corr[
                    "hurs"
                ].raster.clip_bbox(hurs_output.raster.bounds)

                hurs_output.loc[dict(time=slice(start_month, end_month))] = (
                    self.snap_to_grid(
                        w5e5_regridded_corr_clipped,
                        hurs_output,
                        relative_tollerance=0.07,
                    )
                )

        self.set_hurs(hurs_output)

    def setup_longwave_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the longwave radiation data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the longwave radiation data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the longwave radiation data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
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
        x1 = 0.43
        x2 = 5.7
        sbc = 5.67e-8  # stefan boltzman constant [Js−1 m−2 K−4]

        es0 = 6.11  # reference saturation vapour pressure  [hPa]
        T0 = 273.15
        lv = 2.5e6  # latent heat of vaporization of water
        Rv = 461.5  # gas constant for water vapour [J K kg-1]

        hurs_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="hurs",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).hurs  # some buffer to avoid edge effects / errors in ISIMIP API
        tas_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="tas",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).tas  # some buffer to avoid edge effects / errors in ISIMIP API
        rlds_coarse = self.download_isimip(
            product="SecondaryInputData",
            variable="rlds",
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).rlds  # some buffer to avoid edge effects / errors in ISIMIP API

        target = self.other["climate/hurs"]
        target.raster.set_crs(4326)

        hurs_coarse_regridded = hurs_coarse.raster.reproject_like(
            target, method="bilinear"
        )

        tas_coarse_regridded = tas_coarse.raster.reproject_like(
            target, method="bilinear"
        )
        rlds_coarse_regridded = rlds_coarse.raster.reproject_like(
            target, method="bilinear"
        )

        hurs_fine = self.other["climate/hurs"]
        tas_fine = self.other["climate/tas"]

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

        lw_fine = self.snap_to_grid(lw_fine, self.grid, relative_tollerance=0.07)
        lw_fine.attrs = {
            "units": "W m-2",
            "long_name": "Surface Downwelling Longwave Radiation",
            "_FillValue": np.nan,
        }
        self.set_rlds(lw_fine)

    def setup_pressure_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the surface pressure data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the surface pressure data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the surface pressure data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
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
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).psl  # some buffer to avoid edge effects / errors in ISIMIP API

        target = self.other["climate/hurs"]
        target.raster.set_crs(4326)

        orography = self.download_isimip(
            product="InputData", variable="orog", forcing="chelsa-w5e5", buffer=1
        ).orog  # some buffer to avoid edge effects / errors in ISIMIP API
        # TODO: This can perhaps be a clipped version of the orography data
        orography = orography.raster.reproject_like(target, method="average")

        # pressure at sea level, so we can do bilinear interpolation before
        # applying the correction for orography
        pressure_30_min_regridded = pressure_30_min.raster.reproject_like(
            target, method="bilinear"
        )

        pressure_30_min_regridded_corr = pressure_30_min_regridded * np.exp(
            -(g * orography * M) / (T0 * r0)
        )
        pressure_30_min_regridded_corr.attrs = {
            "units": "Pa",
            "long_name": "surface pressure",
            "_FillValue": np.nan,
        }

        self.set_ps(pressure_30_min_regridded_corr)

    def setup_wind_isimip_30arcsec(self, starttime: date, endtime: date):
        """
        Sets up the wind data for GEB.

        Parameters
        ----------
        starttime : date
            The start time of the wind data in ISO 8601 format (YYYY-MM-DD).
        endtime : date
            The end time of the wind data in ISO 8601 format (YYYY-MM-DD).
        folder: str
            The folder to save the forcing data in.

        Notes
        -----
        This method sets up the wind data for GEB. It first downloads the global wind atlas data and
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

        global_wind_atlas_regridded = global_wind_atlas.raster.reproject_like(
            target, method="average"
        )

        wind_30_min_avg = self.download_isimip(
            product="SecondaryInputData",
            variable="sfcwind",
            starttime=date(2008, 1, 1),
            endtime=date(2017, 12, 31),
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind.mean(
            dim="time"
        )  # some buffer to avoid edge effects / errors in ISIMIP API

        wind_30_min_avg_regridded = wind_30_min_avg.raster.reproject_like(
            target, method="bilinear"
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
            starttime=starttime,
            endtime=endtime,
            forcing="w5e5v2.0",
            buffer=1,
        ).sfcWind  # some buffer to avoid edge effects / errors in ISIMIP API

        wind_30min_regridded = wind_30_min.raster.reproject_like(
            target, method="bilinear"
        )
        wind_30min_regridded_log = np.log(wind_30min_regridded)

        wind_30min_regridded_log_corr = wind_30min_regridded_log + diff_layer
        wind_30min_regridded_corr = np.exp(wind_30min_regridded_log_corr)

        wind_output_clipped = wind_30min_regridded_corr.raster.clip_bbox(
            self.grid.raster.bounds
        )

        wind_output_clipped = self.snap_to_grid(wind_output_clipped, self.grid)
        wind_output_clipped.attrs = {
            "units": "m/s",
            "long_name": "Surface Wind Speed",
            "_FillValue": np.nan,
        }
        self.set_sfcwind(wind_output_clipped)

    def setup_SPEI(
        self,
        calibration_period_start: date = date(1981, 1, 1),
        calibration_period_end: date = date(2010, 1, 1),
        window: int = 12,
    ):
        """
        Sets up the Standardized Precipitation Evapotranspiration Index (SPEI). Note that
        due to the sliding window, the SPEI data will be shorter than the original data. When
        a sliding window of 12 months is used, the SPEI data will be shorter by 11 months.

        Also sets up the Generalized Extreme Value (GEV) parameters for the SPEI data, being
        the c shape (ξ), loc location (μ), and scale (σ) parameters.

        The chunks for the climate data are optimized for reading the data in xy-direction. However,
        for the SPEI calculation, the data is needs to be read in time direction. Therefore, we
        create an intermediate temporary file of the water balance wher chunks are in an intermediate
        size between the xy and time chunks.

        Parameters
        ----------
        calibration_period_start : date
            The start time of the reSPEI data in ISO 8601 format (YYYY-MM-DD).
        calibration_period_end : date
            The end time of the SPEI data in ISO 8601 format (YYYY-MM-DD). Endtime is exclusive.
        """
        self.logger.info("setting up SPEI...")

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

        self.other["climate/tasmin"]["y"].attrs["standard_name"] = "latitude"
        self.other["climate/tasmin"]["x"].attrs["standard_name"] = "longitude"
        self.other["climate/tasmin"]["y"].attrs["units"] = "degrees_north"
        self.other["climate/tasmin"]["x"].attrs["units"] = "degrees_east"

        pet = xci.potential_evapotranspiration(
            tasmin=self.other["climate/tasmin"],
            tasmax=self.other["climate/tasmax"],
            method="BR65",
        ).astype(np.float32)

        # Compute the potential evapotranspiration
        water_budget = xci.water_budget(pr=self.other["climate/pr"], evspsblpot=pet)

        water_budget = water_budget.resample(time="MS").mean(keep_attrs=True)

        temp_xy_chunk_size = 50

        water_budget.attrs = {"units": "kg m-2 s-1", "_FillValue": np.nan}
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
            SPEI = xci.standardized_precipitation_evapotranspiration_index(
                wb=water_budget,
                cal_start=calibration_period_start,
                cal_end=calibration_period_end,
                freq=None,
                window=window,
                dist="gamma",
                method="ML",
            ).astype(np.float32)

            # remove all nan values as a result of the sliding window
            SPEI.attrs = {
                "units": "-",
                "long_name": "Standard Precipitation Evapotranspiration Index",
                "name": "spei",
                "_FillValue": np.nan,
            }

            with tempfile.TemporaryDirectory() as tmp_spei_folder:
                tmp_spei_file = Path(tmp_spei_folder) / "tmp_spei_file.zarr"
                self.logger.info("Calculating SPEI and to temporary file...")
                SPEI = to_zarr(
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
