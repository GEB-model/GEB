"""Data adapter for CWATM water demand data."""

import re
from pathlib import Path
from typing import Any

import requests
import xarray as xr

from geb.workflows.io import fetch_and_save

from .base import Adapter

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

    def check_quota(self, text: str, file_path: Path | None = None) -> None:
        """Check if the Google Drive quota has been exceeded.

        Args:
            text: The HTML text to check.
            file_path: The path to the file. If the quota is exceeded, the file will be deleted.
                If None, no file will be deleted.

        Raises:
            ValueError: If the quota has been exceeded.
        """
        if "Google Drive - Quota exceeded" in text:
            if file_path and file_path.exists():
                file_path.unlink()  # remove the incomplete file
            raise ValueError(
                "Too many users have viewed or downloaded this file recently. Please try accessing the file again later. If the file you are trying to access is particularly large or is shared with many people, it may take up to 24 hours to be able to view or download the file."
            )

    def fetch(self, url: str) -> Adapter:
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
            if file_id == "PLACEHOLDER":
                raise ValueError(
                    f"File ID for {self.variable} {scenario} is missing (PLACEHOLDER)."
                )

            download_path: Path = self.path.parent
            download_path.mkdir(parents=True, exist_ok=True)
            file_path = self.path

            session: requests.Session = requests.Session()

            if not file_path.exists():
                response = session.get(
                    f"https://drive.google.com/uc?export=download&id={file_id}",
                    stream=True,
                )

                # Check if we got a direct file download or an HTML page
                content_type = response.headers.get("content-type", "").lower()
                is_html = "text/html" in content_type

                if is_html:
                    # Likely a confirmation page or error
                    # We can read the text because it's small
                    html = response.text
                    self.check_quota(html)

                    # Regex-based parse of hidden inputs for large file confirmation
                    inputs = dict(re.findall(r'name="([^"]+)" value="([^"]+)"', html))

                    if "id" in inputs and "confirm" in inputs:
                        action_url_match = re.search(r'form[^>]+action="([^"]+)"', html)
                        assert action_url_match, (
                            "Could not find form action URL, perhaps Google changed their HTML?"
                        )
                        action_url = action_url_match.group(1)
                        fetch_and_save(
                            url=action_url,
                            file_path=file_path,
                            params=inputs,
                            session=session,
                        )

                        # if file is less than 100KB, it is probably an error page
                        if file_path.stat().st_size < 100_000:
                            with open(file_path, "r") as f:
                                text = f.read()
                            if "Google Drive - Quota exceeded" in text:
                                self.check_quota(text, file_path)
                    else:
                        # Unknown HTML response
                        pass
                else:
                    # Direct download
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)

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
