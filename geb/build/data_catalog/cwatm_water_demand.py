"""Data adapter for CWATM water demand data."""

from typing import Any

import rioxarray  # noqa: F401
import xarray as xr

from .base import Adapter
from .workflows.google_drive import download_from_google_drive

FILES: dict[str, dict[str, str]] = {
    "industry": {
        "historical": "1QrDQ59qb4pa6PxnvePhCoKnlvvG2DhZO",
        "ssp1": "16t14PSPWJ7L1lJNh7NBePYmgR-RmZLX3",
        "ssp2": "1goboGVpWZh7u2qv9M8jsSnDHQ8wkuZPN",
        "ssp3": "1p-a__Wjp7sa5lIOCIcY0VZMU1BL4HWK2",
        "ssp5": "13c_Tw7wA7LPzcMXi6mvs6w-vW8W7MH2j",
    },
    "livestock": {
        "historical": "1z0fQgdF7BUGbmm07-mUAxgIt8OrRN9gC",
        "ssp2": "1L6DZ_GVZnJCmBqnRaMkfeZikD-PKo5RS",
    },
}


class CWATMWaterDemand(Adapter):
    """The CWATM Water Demand adapter for downloading water demand data.

    Args:
        Adapter: The base Adapter class.
    """

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the CWATMWaterDemand adapter.

        Args:
            variable: The variable to download, either 'industry' or 'livestock'.
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        self.variable = variable
        super().__init__(*args, **kwargs)

    def fetch(self, url: str | None = None) -> Adapter:
        """Download the CWATM water demand file from Google Drive.

        Args:
            url: The URL to download data from (unused, IDs are looked up in FILES).

        Returns:
            The adapter instance.

        Raises:
            ValueError: If the scenario cannot be determined or the ID is missing.
        """
        if not self.is_ready:
            # Determine scenario from filename
            # Filenames are expected to contain the scenario (e.g. historical_liv_...)
            filename = self.path.name
            scenario = next(
                (s for s in FILES[self.variable] if f"{s}_" in filename), None
            )

            if not scenario:
                raise ValueError(
                    f"Could not determine scenario from filename: {filename}. Expected one of {list(FILES[self.variable].keys())}"
                )

            file_id = FILES[self.variable][scenario]
            download_from_google_drive(file_id=file_id, file_path=self.path)

        return self

    def read(self, **kwargs: Any) -> xr.Dataset:
        """Read the CWATM water demand data.

        Args:
            **kwargs: Additional keyword arguments forwarded to xarray.

        Returns:
            The dataset.
        """
        ds = xr.open_dataset(self.path, decode_times=False, **kwargs).rename(
            {"lat": "y", "lon": "x"}
        )
        ds = ds.rio.write_crs("EPSG:4326")
        return ds


class CWATMLivestockWaterDemand(CWATMWaterDemand):
    """The CWATM Livestock Water Demand adapter.

    Args:
        CWATMWaterDemand: The base CWATMWaterDemand class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the CWATMLivestockWaterDemand adapter."""
        super().__init__("livestock", *args, **kwargs)


class CWATMIndustryWaterDemand(CWATMWaterDemand):
    """The CWATM Industry Water Demand adapter.

    Args:
        CWATMWaterDemand: The base CWATMWaterDemand class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the CWATMIndustryWaterDemand adapter."""
        super().__init__("industry", *args, **kwargs)
