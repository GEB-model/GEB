"""Data adapter for obtaining CMIP6 climate projections."""

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cdsapi
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from .base import Adapter

mapping_variables_to_cdf = {
    "near_surface_air_temperature": "tas",
    "precipitation": "pr",
}

available_scenarios = ["ssp1_1_9", "ssp1_2_6", "ssp2_4_5", "ssp3_7_0", "ssp5_8_5"]

from ...workflows.io import (
    read_zarr,
    write_zarr,
)


def to_minus180_180(da: xr.DataArray, lon_name: str = "x") -> xr.DataArray:
    """Wrap longitudes to the [-180, 180) range and sort by longitude.

    Args:
        da: Input DataArray containing a longitude coordinate.
        lon_name: Name of the longitude coordinate in ``da``. Defaults to "x".

    Returns:
        A copy of ``da`` with wrapped longitudes and ascending longitude order.
    """
    lon = da[lon_name]
    lon_wrapped = ((lon + 180) % 360) - 180
    da = da.assign_coords({lon_name: lon_wrapped})
    da = da.sortby(lon_name)
    return da


class CMIP6(Adapter):
    """Data adapter for obtaining CMIP6 climate projections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the CMIP6 data adapter."""
        super().__init__(*args, **kwargs)

    def construct_request(
        self,
        years: list[str],
        bounds: tuple[float, float, float, float],
        variable: str,
        experiment: str = "historical",
        model: str = "gfdl_esm4",
    ) -> dict[str, Any]:
        """Construct the API call dictionary for GTSM data retrieval.

        Args:
             years: A list of years to include in the request.
             bounds: A tuple containing the bounding coordinates (min_lon, min_lat, max_lon, max_lat).
             variable: The variable to retrieve (e.g., "near_surface_air_temperature" or "precipitation").
             experiment: The CMIP6 experiment to retrieve data for (e.g., "historical", "ssp5_8_5").
             model: The CMIP6 model to retrieve data from (e.g., "gfdl_esm4").
        Returns:
             A dictionary containing the parameters for the CMIP6/CDS API call.
        """
        # Add a buffer to the bounds to ensure we match the extent of the ERA5 data
        buffer: float = 3.0
        buffered_bounds: tuple[float, float, float, float] = (
            bounds[0] - buffer,
            bounds[1] - buffer,
            bounds[2] + buffer,
            bounds[3] + buffer,
        )

        area = [
            buffered_bounds[3],
            buffered_bounds[0],
            buffered_bounds[1],
            buffered_bounds[2],
        ]  # [max_lat, min_lon, min_lat, max_lon]
        request = {
            "temporal_resolution": "monthly",
            "experiment": experiment,
            "variable": variable,
            "model": model,
            "year": years,
            "area": area,
        }
        return request

    def _retrieve(self, request: dict[str, Any], output_path: str) -> None:
        """Download CDS data to a temporary file and atomically move it into place.

        Args:
            request: Parameters for the CDS API request.
            output_path: Final destination for the downloaded file.
        """
        output_file: Path = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Keep the temporary file in the target directory so the final replace stays atomic.
        with tempfile.NamedTemporaryFile() as temporary_file:
            temporary_path = Path(temporary_file.name)

            client = cdsapi.Client()
            client.retrieve(
                "projections-cmip6",
                request,
                str(temporary_path),
            )
            shutil.move(temporary_file.name, output_path)

    def calculate_deltas(
        self,
        variable: str,
        historical_data_path: Path,
        future_data_path: Path,
        start_year: int,
        end_year: int,
        representative_forcing_year: int = 2050,
    ) -> xr.Dataset:
        """Calculate the {variable} deltas from the historical and future CMIP6 data.

        for each month, the 30-year mean of a future period centred on the representative year is compared with the 30-year mean of a
        reference period (the last 30 years up to end_year). Temperature deltas are absolute
        differences, precipitation deltas are ratios. The resulting 12 monthly deltas are repeated
        for every year of the forcing period (start_year to end_year + 1).

        Args:
            variable: The variable to compute deltas for (e.g., "tas" for near-surface air temperature, "pr" for precipitation).
            historical_data_path: The path to the ZIP file containing the historical CMIP6 data.
            future_data_path: The path to the ZIP file containing the future CMIP6 data.
            start_year: The start year for calculating deltas.
            end_year: The end year for calculating deltas.
            representative_forcing_year: The year to use as a representative future projection of forcing data (e.g., 2050).
        Returns:
            An xarray Dataset containing the calculated deltas for the specified variable.
        Raises:
            ValueError: If the merged historical and future data is not a continuous monthly series, or if the selected periods are invalid.
        """
        variable_in_netcdf = mapping_variables_to_cdf.get(variable, variable)

        historical = self.unzip_and_load(historical_data_path)
        future = self.unzip_and_load(future_data_path)

        # Normalize time & drop bounds
        def normalize_time(ds: xr.Dataset) -> xr.Dataset:
            if "time_bounds" in ds:
                ds = ds.drop_vars("time_bounds")

            ds["time"] = xr.date_range(
                start=str(ds.time.dt.strftime("%Y-%m-01").values[0]),
                periods=ds.sizes["time"],
                freq="MS",
                calendar=ds.time.dt.calendar,
                use_cftime=True,
            )  # round to nearest day to avoid issues with time bounds

            return ds

        historical = normalize_time(historical)
        future = normalize_time(future)

        # Ensure grids match
        xr.testing.assert_allclose(historical.lat, future.lat)
        xr.testing.assert_allclose(historical.lon, future.lon)

        # Merge along time and sort by time to ensure proper alignment
        merged = xr.concat([historical, future], dim="time").sortby("time")

        month_index: np.ndarray = merged.indexes["time"].year * 12 + (
            merged.indexes["time"].month - 1
        )
        if not (np.diff(month_index) == 1).all():
            raise ValueError(
                "Merged historical and future CMIP6 data is not a continuous monthly series "
                f"(historical: {historical.time.values[0]} to {historical.time.values[-1]}, "
                f"future: {future.time.values[0]} to {future.time.values[-1]}). "
                "Delete the cached CMIP6 ZIP files and download them again."
            )

        # define periods and compute the representative year shift
        window_years = 30
        reference_start, reference_end = end_year - window_years + 1, end_year
        future_start = representative_forcing_year - window_years // 2 + 1
        future_end = future_start + window_years - 1

        available_start = merged.indexes["time"][0].year
        available_end = merged.indexes["time"][-1].year
        if not (
            available_start <= reference_start <= reference_end <= available_end
            and available_start <= future_start <= future_end <= available_end
        ):
            raise ValueError(
                f"Reference period ({reference_start}-{reference_end}) and future period "
                f"({future_start}-{future_end}) must lie within the available CMIP6 data "
                f"({available_start}-{available_end})."
            )

        reference = merged[variable_in_netcdf].sel(
            time=slice(f"{reference_start}-01-01", f"{reference_end}-12-31")
        )
        future = merged[variable_in_netcdf].sel(
            time=slice(f"{future_start}-01-01", f"{future_end}-12-31")
        )
        reference_climatology = reference.groupby("time.month").mean("time")
        future_climatology = future.groupby("time.month").mean("time")

        # Compute delta as either absolute difference (for temperature) or relative change (for precipitation)
        if variable_in_netcdf == "tas":
            monthly_delta = future_climatology - reference_climatology
        else:
            monthly_delta = future_climatology / reference_climatology

        # expand the 12 monthly deltas to every month of the forcing period, so the
        # output keeps the monthly time axis that the forcing adjustment expects
        forcing_time = merged.sel(
            time=slice(f"{start_year}-01-01", f"{end_year + 1}-12-31")
        ).time
        delta = monthly_delta.sel(month=forcing_time.dt.month).drop_vars("month")

        return delta.rio.write_crs("EPSG:4326")  # ensure the deltas have a CRS

    def combine_deltas_and_write_to_file(self, deltas: dict[str, xr.Dataset]) -> None:
        """Combine the individual variable deltas into a single xarray Dataset and write it to a NetCDF file.

        Args:
            deltas: A dictionary mapping variable names to their corresponding delta xarray Datasets.
        Raises:
            ValueError: If there are NaN values in the combined deltas, indicating a problem with the delta calculation.
        """
        combined = xr.Dataset()
        for variable, delta in deltas.items():
            combined[variable + "_delta"] = delta
        combined["time"] = combined.indexes["time"].to_datetimeindex(time_unit="ns")  # ty:ignore[unresolved-attribute]
        combined = combined.rename({"lon": "x", "lat": "y"}).to_array()
        # assert that there are no NaN values in the combined deltas, which would indicate a problem with the delta calculation
        if combined.isnull().any().compute().item():
            raise ValueError(
                "NaN values found in combined deltas, indicating a problem with the delta calculation."
            )
        combined.attrs["_FillValue"] = (
            np.nan
        )  # Set fill value attribute for missing data
        combined = combined.rio.write_crs(
            "EPSG:4326"
        )  # ensure the combined dataset has a CRS
        combined = to_minus180_180(
            da=combined,
            lon_name="x",
        )  # ensure longitudes are in the -180 to 180 range
        write_zarr(da=combined, path=self.path, crs=combined.rio.crs)

    def unzip_and_load(self, zip_path: Path) -> xr.Dataset:
        """Unzip the downloaded CMIP6 data and load it into an xarray Dataset.

        Args:
            zip_path: The path to the ZIP file containing the CMIP6 data.
        Returns:
            An xarray Dataset containing the CMIP6 data.
        Raises:
            ValueError: If the ZIP file does not contain exactly one .nc file.
        """
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Process historical data
            # Find the .nc file inside the zip
            nc_files = [name for name in zip_ref.namelist() if name.endswith(".nc")]

            if len(nc_files) != 1:
                raise ValueError(
                    f"Expected 1 .nc file, found {len(nc_files)}: {nc_files}"
                )

            nc_name = nc_files[0]

            with zip_ref.open(nc_name) as f:
                da = xr.open_dataset(f).load()
        return da

    def fetch(
        self,
        bounds: tuple[float, float, float, float],
        start_year: int,
        end_year: int,
        ssp_rcp: int,
        representative_forcing_year: int,
        url: str | None = None,
    ) -> CMIP6:
        """Fetch CMIP6 data and save it to the cache directory.

        Args:
            url: Not used for CMIP6, included for compatibility with base class.
            bounds: A tuple containing the bounding coordinates (min_lon, min_lat, max_lon, max_lat).
            start_year: The start year for calculating deltas.
            end_year: The end year for the data range.
            representative_forcing_year: The year to use as a representative future projection of forcing data (e.g., 2050).
            ssp_rcp: The SSP scenario to use (e.g., 245 for SSP2-4.5).
            url: Not used for CMIP6, included for compatibility with base class.
        Raises:
            ValueError: If the SSP-RCP is invalid or if the historical and future periods are not of equal length.
        Returns:
            The current instance of the CMIP6 adapter.
        """
        if self.path.exists():
            return self

        # translate the scenario string to the full SSP name
        ssp_scenario = f"ssp{str(ssp_rcp).replace('', '_')[1:6]}"
        if ssp_scenario not in available_scenarios:
            raise ValueError(
                f"Invalid SSP-RCP '{ssp_rcp}'. Available scenarios are: {available_scenarios}"
            )

        # CMIP6 "historical" experiments end in 2014 and the SSP scenarios start in 2015.
        # Requesting other years silently returns no data, which would leave a gap.
        years_historical = [str(year) for year in range(1950, 2015)]
        years_future = [str(year) for year in range(2015, 2101)]
        deltas = {}

        # fetch both precipitation and near-surface air temperature for the historical period, since we'll need both to compute deltas
        for variable in ["precipitation", "near_surface_air_temperature"]:
            historical_data_path = self.root / f"cmip6_historical_{variable}.zip"
            if not historical_data_path.exists():
                request = self.construct_request(
                    years=years_historical,
                    variable=variable,
                    bounds=bounds,
                    experiment="historical",
                    model="gfdl_esm4",
                )
                self._retrieve(request, str(historical_data_path))

            # future data
            future_data_path = (
                self.root
                / f"cmip6_{ssp_scenario}_{variable}_{representative_forcing_year}.zip"
            )
            if not future_data_path.exists():
                request = self.construct_request(
                    years=years_future,
                    variable=variable,
                    bounds=bounds,
                    experiment=ssp_scenario,
                    model="gfdl_esm4",
                )

                self._retrieve(request, str(future_data_path))

            deltas[variable] = self.calculate_deltas(
                variable=variable,
                start_year=start_year,
                end_year=end_year,
                representative_forcing_year=representative_forcing_year,
                historical_data_path=historical_data_path,
                future_data_path=future_data_path,
            )
        self.combine_deltas_and_write_to_file(deltas)
        return self

    def read(self) -> xr.DataArray:
        """Read the processed CMIP6 deltas from the local cache.

        Returns:
            An xarray DataArray containing the CMIP6 deltas.
        """
        return read_zarr(self.path)
