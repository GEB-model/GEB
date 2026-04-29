"""Data adapter for obtaining CMIP6 climate projections."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cdsapi

from .base import Adapter


class CMIP6(Adapter):
    """Data adapter for obtaining CMIP6 climate projections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the CMIP6 data adapter."""
        super().__init__(*args, **kwargs)

    def _variable_dir(self, variable: str) -> Path:
        """Return the directory containing cached files for a variable.

        Args:
            variable: GTSM variable name.

        Returns:
            Directory used to store the requested variable.
        """
        return self.root / variable

    def construct_request(
        self, years: list[int], bounds: tuple[float, float, float, float]
    ) -> dict[str, Any]:
        """Construct the API call dictionary for GTSM data retrieval.

        Args:
             years: A list of years to include in the request.
             bounds: A tuple containing the bounding coordinates (min_lon, min_lat, max_lon, max_lat).
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
            "experiment": "historical",
            "variable": ["near_surface_air_temperature", "precipitation"],
            "model": "gfdl_esm4",
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
        years = [str(year) for year in range(start_date.year, end_date.year + 1)]
        variable_dir: Path = self._variable_dir("temperature")
        request = self.construct_request(years=years, bounds=bounds)
        output_path = Path("cmip6.zip")
        self.download_data(request, str(output_path))

        return self

    def read(self) -> None:
        """Read the downloaded CMIP6 data from the cache directory."""
        pass
