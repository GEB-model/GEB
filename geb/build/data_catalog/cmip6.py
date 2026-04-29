"""Data adapter for obtaining CMIP6 climate projections."""

import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cdsapi
import xarray as xr

from .base import Adapter


class CMIP6(Adapter):
    """Data adapter for obtaining CMIP6 climate projections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the CMIP6 data adapter."""
        super().__init__(*args, **kwargs)

    def construct_request(
        self,
        years: list[int],
        bounds: tuple[float, float, float, float],
        experiment: str = "historical",
        model: str = "gfdl_esm4",
    ) -> dict[str, Any]:
        """Construct the API call dictionary for GTSM data retrieval.

        Args:
             years: A list of years to include in the request.
             bounds: A tuple containing the bounding coordinates (min_lon, min_lat, max_lon, max_lat).
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
            "variable": ["precipitation"],
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

    def calculate_deltas(self) -> None:
        """Calculate the precipitation deltas from the historical and future CMIP6 data."""
        historical_data_path = Path("cmip6_historical.zip")
        future_data_path = Path("cmip6_future.zip")

        historical = self.unzip_and_load(historical_data_path)
        future = self.unzip_and_load(future_data_path)

        # --- Normalize time & drop bounds ---
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

        # --- Ensure grids match ---
        xr.testing.assert_allclose(historical.lat, future.lat)
        xr.testing.assert_allclose(historical.lon, future.lon)

        # --- Merge along time ---
        merged = xr.concat([historical, future], dim="time").sortby("time")

        # --- Sanity check on coverage ---
        years = set(merged["time"].dt.year.values)
        assert set(range(1950, 2100)).issubset(years)

        # --- Define periods ---
        shift = 50  # years
        initial_start, initial_end = 1980, 2020
        new_start, new_end = initial_start + shift, initial_end + shift

        # --- Select subsets ---
        historical_subset = merged.sel(
            time=slice(f"{initial_start}-01-01", f"{initial_end}-12-31")
        )

        future_subset = merged.sel(time=slice(f"{new_start}-01-01", f"{new_end}-12-31"))

        # --- Align time axes by shifting future back ---
        future_subset = future_subset.assign_coords(time=historical_subset.time)
        # --- Force exact alignment ---
        historical_subset, future_subset = xr.align(
            historical_subset, future_subset, join="exact"
        )

        # --- Compute delta (fully aligned, no numpy hack) ---
        delta = future_subset["pr"] / historical_subset["pr"]

        # Avoid divide-by-zero issues
        delta = delta.where(historical_subset["pr"] != 0)

        # --- Save ---
        delta = delta.rename("precipitation_delta")
        delta.to_netcdf("cmip6_precipitation_delta.nc")

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
        start_date: datetime,
        end_date: datetime,
        bounds: tuple[float, float, float, float],
        url: str | None = None,
    ) -> CMIP6:
        """Fetch CMIP6 data and save it to the cache directory.

        Args:
            url: Not used for CMIP6, included for compatibility with base class.
            start_date: The start date for the data retrieval.
            end_date: The end date for the data retrieval.
            bounds: A tuple containing the bounding coordinates (min_lon, min_lat, max_lon,
        Returns:
            The current instance of the CMIP6 adapter.
        """
        # first fetch historical data (up to 2016), then fetch future projections (2017-2100)
        years = [str(year) for year in range(1950, 2017)]
        request = self.construct_request(
            years=years, bounds=bounds, experiment="historical", model="gfdl_esm4"
        )
        output_path = Path("cmip6_historical.zip")
        if not output_path.exists():
            self.download_data(request, str(output_path))

        # future data
        years = [str(year) for year in range(2015, 2101)]
        request = self.construct_request(
            years=years, bounds=bounds, experiment="ssp5_8_5", model="gfdl_esm4"
        )
        output_path = Path("cmip6_future.zip")
        if not output_path.exists():
            self.download_data(request, str(output_path))
        self.calculate_deltas()
        return self

    def read(self) -> None:
        """Read the downloaded CMIP6 data from the cache directory."""
        pass
