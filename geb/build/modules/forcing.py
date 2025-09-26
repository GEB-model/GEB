import tempfile
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xclim.indices as xci
from dateutil.relativedelta import relativedelta
from zarr.codecs.numcodecs import FixedScaleOffset

from geb.build.data_catalog.base import Adapter
from geb.build.methods import build_method
from geb.workflows.io import get_window

from ...workflows.io import calculate_scaling, to_zarr
from ..workflows.general import (
    resample_like,
)


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
    """Contains methods to download and process climate forcing data for GEB."""

    def __init__(self) -> None:
        pass

    def plot_forcing(self, da, name) -> None:
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

    def set_pr(self, da: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
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
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, max_value, offset=offset, precision=precision
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        self.plot_forcing(da, name)
        return da

    def set_rsds(self, da: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
        name: str = "climate/rsds"
        da.attrs = {
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "long_name": "Surface Downwelling Shortwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        self.plot_forcing(da, name)
        return da

    def set_rlds(self, da: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
        name: str = "climate/rlds"
        da.attrs = {
            "standard_name": "surface_downwelling_longwave_flux_in_air",
            "long_name": "Surface Downwelling Longwave Radiation",
            "units": "W m-2",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, 1361, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        self.plot_forcing(da, name)
        return da

    def set_tas(self, da: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
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
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )

        self.plot_forcing(da, name)
        return da

    def set_dewpoint_tas(
        self, da: xr.DataArray, *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        name: str = "climate/dewpoint_tas"
        da.attrs = {
            "standard_name": "air_temperature_dow_point",
            "long_name": "Hourly Near-Surface Dewpoint Temperature",
            "units": "K",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        K_to_C: float = 273.15
        offset: float = -15 - K_to_C  # average temperature on earth
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, -100 + K_to_C, 60 + K_to_C, offset=offset, precision=0.1
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        self.plot_forcing(da, name)
        return da

    def set_ps(self, da: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
        name: str = "climate/ps"
        da.attrs = {
            "standard_name": "surface_air_pressure",
            "long_name": "Surface Air Pressure",
            "units": "Pa",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset: int = -100_000
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 30_000, 120_000, offset=offset, precision=10
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        self.plot_forcing(da, name)
        return da

    def set_wind(
        self, da: xr.DataArray, direction: str, *args: Any, **kwargs: Any
    ) -> xr.DataArray:
        name: str = f"climate/wind_{direction}10m"
        da.attrs = {
            "standard_name": "wind_speed",
            "long_name": "Near-Surface Wind Speed",
            "units": "m s-1",
            "_FillValue": np.nan,
        }
        self.set_xy_attrs(da)

        offset = 0
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, 0, 120, offset=offset, precision=0.1
        )
        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
            da,
            name=name,
            *args,
            **kwargs,
            byteshuffle=True,
            filters=filters,
            time_chunks_per_shard=get_chunk_size(da) // 24,
            time_chunksize=24,
        )
        self.plot_forcing(da, name)
        return da

    def set_SPEI(self, da: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
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
        scaling_factor, in_dtype, out_dtype = calculate_scaling(
            da, min_SPEI, max_SPEI, offset=offset, precision=0.001
        )

        filters: list = [
            FixedScaleOffset(
                offset=offset,
                scale=scaling_factor,
                dtype=in_dtype,
                astype=out_dtype,
            ),
        ]

        da: xr.DataArray = self.set_other(
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

    def setup_forcing_ERA5(self) -> None:
        era5_store: Adapter = self.new_data_catalog.fetch("era5")
        era5_loader: partial = partial(
            era5_store.read,
            start_date=self.start_date - relativedelta(years=1),
            end_date=self.end_date,
            bounds=self.grid["mask"].raster.bounds,
        )

        pr_hourly: xr.DataArray = era5_loader(variable="tp")
        pr_hourly: xr.DataArray = pr_hourly * (
            1000 / 3600
        )  # convert from m/hr to kg/m2/s

        # ensure no negative values for precipitation, which may arise due to float precision
        pr_hourly: xr.DataArray = xr.where(pr_hourly > 0, pr_hourly, 0, keep_attrs=True)
        pr_hourly: xr.DataArray = self.set_pr(pr_hourly)

        hourly_tas: xr.DataArray = era5_loader("t2m")
        self.set_tas(hourly_tas)

        hourly_dew_point_tas: xr.DataArray = era5_loader("d2m")
        self.set_dewpoint_tas(hourly_dew_point_tas)

        hourly_rsds: xr.DataArray = era5_loader("ssrd") / (
            24 * 3600  # convert from J/m2 to W/m2
        )  # surface_solar_radiation_downwards
        self.set_rsds(hourly_rsds)

        hourly_rlds: xr.DataArray = era5_loader(
            "strd"
        ) / (  # surface_thermal_radiation_downwards
            24 * 3600
        )  # convert from J/m2 to W/m2
        self.set_rlds(hourly_rlds)

        pressure: xr.DataArray = era5_loader("sp")
        self.set_ps(pressure)

        u_wind: xr.DataArray = era5_loader("u10")
        self.set_wind(u_wind, direction="u")

        v_wind: xr.DataArray = era5_loader("v10")
        self.set_wind(v_wind, direction="v")

        elevation_grid = self.get_elevation_forcing_and_grid(pr_hourly)

    @build_method(depends_on=["set_ssp", "set_time_range"])
    def setup_forcing(
        self,
        forcing: str = "ERA5",
        resolution_arcsec: int | None = None,
        model: str | None = None,
    ) -> None:
        """Sets up the forcing data for GEB.

        Args:
            forcing: The data source to use for the forcing data. Can be ERA5 or ISIMIP. Default is 'era5'.
            resolution_arcsec: The resolution of the data in arcseconds. Only used for ISIMIP. Supported values are 30 and 1800.
            model: The name of the forcing data to use within the dataset. Only required for ISIMIP data.
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

        Raises:
            ValueError: If an unknown data source is specified.
        """
        if forcing == "ISIMIP":
            raise NotImplementedError(
                "ISIMIP forcing is not supported anymore. We switched fully to hourly forcing data."
            )
        elif forcing == "ERA5":
            assert resolution_arcsec is None, (
                "resolution_arcsec must be None for ERA5 forcing data"
            )
            assert model is None, "model must be None for ERA5 forcing data"
            self.setup_forcing_ERA5()
        elif forcing == "CMIP":
            raise NotImplementedError("CMIP forcing data is not yet supported")
        else:
            raise ValueError(
                f"Unknown data source: {forcing}, supported are 'ISIMIP' and 'ERA5'"
            )

    @build_method(depends_on=["setup_forcing"])
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

        Raises:
            ValueError: If the input data do not have the same coordinates.
        """
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
            raise ValueError(
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
        self, forcing_grid: xr.DataArray
    ) -> xr.DataArray:
        """Gets elevation maps for both the normal grid (target of resampling) and the forcing grid.

        Args:
            grid: the normal grid to which the forcing data is resampled
            forcing_grid: grid of the forcing data
            forcing_name: name of the forcing data, used to determine the file paths for caching

        Returns:
            elevation data for the forcing grid and the normal grid

        """
        elevation = xr.open_dataarray(self.data_catalog.get_source("fabdem").path)
        elevation = elevation.isel(
            band=0,
            **get_window(
                elevation.x, elevation.y, forcing_grid.rio.bounds(), buffer=500
            ),
        )
        elevation = elevation.drop_vars("band")
        elevation = xr.where(elevation == -9999, 0, elevation)
        elevation.attrs["_FillValue"] = np.nan
        target = forcing_grid.isel(time=0).drop_vars("time")

        elevation_forcing = resample_like(elevation, target, method="bilinear")
        elevation_forcing = elevation_forcing.chunk({"x": -1, "y": -1})

        return elevation_forcing

    @build_method(depends_on=["set_ssp", "set_time_range"])
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
