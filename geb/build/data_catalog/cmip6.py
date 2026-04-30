"""Data adapter for obtaining CMIP6 climate projections."""

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cdsapi
import xarray as xr

from .base import Adapter

mapping_variables_to_cdf = {
    "near_surface_air_temperature": "tas",
    "precipitation": "pr",
}


class CMIP6(Adapter):
    """Data adapter for obtaining CMIP6 climate projections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the CMIP6 data adapter."""
        super().__init__(*args, **kwargs)

    def construct_request(
        self,
        years: list[int],
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
             A dictionary containing the parameters for the GTSM API call.
        """
        area = [
            bounds[3],
            bounds[0],
            bounds[1],
            bounds[2],
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

    def download_data(self, request: dict[str, Any], output_path: str) -> None:
        """Download GTSM data using the CDS API.

        Args:
            request: A dictionary containing the parameters for the GTSM API call.
            output_path: The file path where the downloaded data will be saved.
        """
        self._retrieve(request, output_path)

    def calculate_deltas(
        self,
        variable: str,
        historical_data_path: Path,
        future_data_path: Path,
        start_year: int,
        end_year: int,
        representative_year: int = 2050,
    ) -> xr.Dataset:
        """Calculate the {variable} deltas from the historical and future CMIP6 data.

        Args:
            variable: The variable to compute deltas for (e.g., "tas" for near-surface air temperature, "pr" for precipitation).
            historical_data_path: The path to the ZIP file containing the historical CMIP6 data.
            future_data_path: The path to the ZIP file containing the future CMIP6 data.
            start_year: The start year for calculating deltas.
            end_year: The end year for calculating deltas.
            representative_year: The year to use as a representative future projection of forcing data (e.g., 2050).
        Returns:
            An xarray Dataset containing the calculated deltas for the specified variable.
        """
        variable_in_netcdf = mapping_variables_to_cdf.get(variable, variable)

        historical = self.unzip_and_load(historical_data_path)
        future = self.unzip_and_load(future_data_path)

        # Normalize time & drop bounds
        def normalize_time(ds: xr.Dataset) -> xr.Dataset:
            if "time_bounds" in ds:
                ds = ds.drop_vars("time_bounds")

            ds["time"] = xr.cftime_range(
                start=str(ds.time.dt.strftime("%Y-%m-01").values[0]),
                periods=ds.sizes["time"],
                freq="MS",
                calendar=ds.time.dt.calendar,
            )
            return ds

        historical = normalize_time(historical)
        future = normalize_time(future)

        # Ensure grids match
        xr.testing.assert_allclose(historical.lat, future.lat)
        xr.testing.assert_allclose(historical.lon, future.lon)

        # Merge along time and sort by time to ensure proper alignment
        merged = xr.concat([historical, future], dim="time").sortby("time")

        # Define periods and compute representative year shift
        shift = representative_year - end_year  # years
        initial_start, initial_end = start_year, end_year
        new_start, new_end = initial_start + shift, initial_end + shift

        # Select subsets for historical and future periods
        historical_subset = merged.sel(
            time=slice(f"{initial_start}-01-01", f"{initial_end}-12-31")
        )

        future_subset = merged.sel(time=slice(f"{new_start}-01-01", f"{new_end}-12-31"))

        # Align time axes by shifting future back
        future_subset = future_subset.assign_coords(time=historical_subset.time)
        # Force exact alignment to ensure no numpy datetime64 issues in delta calculation
        historical_subset, future_subset = xr.align(
            historical_subset, future_subset, join="exact"
        )

        # Compute delta as either absolute difference (for temperature) or relative change (for precipitation)
        if variable_in_netcdf == "tas":
            delta = (
                future_subset[variable_in_netcdf]
                - historical_subset[variable_in_netcdf]
            )
        else:
            delta = (
                future_subset[variable_in_netcdf]
                / historical_subset[variable_in_netcdf]
            )

        return delta

    def combine_deltas_and_write_to_file(self, deltas: dict[str, xr.Dataset]) -> None:
        """Combine the individual variable deltas into a single xarray Dataset and write it to a NetCDF file.

        Args:
            deltas: A dictionary mapping variable names to their corresponding delta xarray Datasets.
        """
        combined = xr.Dataset()
        for variable, delta in deltas.items():
            combined[variable + "_delta"] = delta
        combined.to_netcdf(self.path)

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
        representative_year: int = 2050,
        url: str | None = None,
    ) -> CMIP6:
        """Fetch CMIP6 data and save it to the cache directory.

        Args:
            url: Not used for CMIP6, included for compatibility with base class.
            bounds: A tuple containing the bounding coordinates (min_lon, min_lat, max_lon, max_lat).
            start_year: The start year for calculating deltas.
            end_year: The end year for the data range.
            representative_year: The year to use as a representative future projection of forcing data (e.g., 2050).
            url: Not used for CMIP6, included for compatibility with base class.
        Returns:
            The current instance of the CMIP6 adapter.
        """
        # first fetch historical data (up to 2016), then fetch future projections (2017-2100)
        years_historical = [str(year) for year in range(1950, 2017)]
        years_future = [str(year) for year in range(2017, 2101)]
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

                self.download_data(request, str(historical_data_path))

            # future data

            future_data_path = self.root / f"cmip6_future_{variable}.zip"
            if not future_data_path.exists():
                request = self.construct_request(
                    years=years_future,
                    variable=variable,
                    bounds=bounds,
                    experiment="ssp5_8_5",
                    model="gfdl_esm4",
                )

                self.download_data(request, str(future_data_path))

            deltas[variable] = self.calculate_deltas(
                variable=variable,
                start_year=start_year,
                end_year=end_year,
                representative_year=representative_year,
                historical_data_path=historical_data_path,
                future_data_path=future_data_path,
            )
        self.combine_deltas_and_write_to_file(deltas)
        return self

    def read(self) -> xr.Dataset:
        """Read the processed CMIP6 deltas from the local cache.

        Returns:
            An xarray Dataset containing the CMIP6 deltas.
        """
        return xr.open_dataset(self.path)
