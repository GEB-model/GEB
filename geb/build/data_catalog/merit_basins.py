"""Data adapter for HydroLAKES data."""

import re
import tempfile
from pathlib import Path
from time import sleep
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from pyarrow.parquet import SortingColumn

from geb.workflows.io import fetch_and_save

from .base import Adapter

FILES: dict[str, dict[str, dict[str, str]]] = {
    "cat": {
        "africa": {
            "shp": "1Vv7aFAlfhhx2OO3wgvA22Ztt-uE50LD2",
            "shx": "1PCIZMH-3xPE3kEU5rCx9tcLiSHRlnz3k",
            "dbf": "1syr1BUcB1CWzr-fFVpzUSPiw0aXmTbOA",
            "cpg": "1mG9qS1HOr8N6EoJfVx8nZFpEyz3evB-f",
        },
        "europe": {
            "shp": "1kOyT6ZNq4Ga2xDgK0LX3LwzbhclFaHFs",
            "shx": "1tQd4Ylqs6R1quCxAHoP8KQCMQrbV_W4z",
            "dbf": "16cr3LwqiqXXBwWum7BLLbzitjzLxZ6tT",
            "cpg": "1VwbXAK_ga_JCJ8YFM-Pt6UxM4Er133sZ",
        },
        "north_asia": {
            "shp": "11fkpeUD-f4UYFlXFHU0rGIdOx3eLhtiU",
            "shx": "1zUxRABGWZSLClBFOWZAMId7D4CpbY8ul",
            "dbf": "104KVCQXa0ygTty8HR5onJP8GUhj7BP3j",
            "cpg": "1lXyYJ4KTlRP-Dhi1Rzw3VZol-l5wBALW",
        },
        "south_asia": {
            "shp": "1cwyCe72QHSnrOKPuqJWTjSJgoha4RR1_",
            "shx": "1F4bboo1zdGQ0VFCbgwwct8VZsJpxt0YB",
            "dbf": "1dEEfpjPtnHjc1UYCPNiPNt_FXRFMHQej",
            "cpg": "1tT6aPRNND5DpryA2lVrlMibKzWx3ZQo4",
        },
        "oceania": {
            "shp": "1KSdJd08il4pHuaOp_ghAXxuzLYJTlE01",
            "shx": "176IHweMUL6i4wUfuytQba009Q0W3E5r2",
            "dbf": "1LIJtJxsl84NwDTRVSsB7OMFXPv1jLPZS",
            "cpg": "14EhtQZ34fmfPMkHae40zCQvekNraI_Tb",
        },
        "south_america": {
            "shp": "1osIIkZ0TjpZadLLMvk6JPGuDjN1ebL8c",
            "shx": "1zA45MbShNDWowCwZUEznfuiBVGaSVi3O",
            "dbf": "1pwLrMtwoP9GM6yRLYzccAYlB1ROIMZ4X",
            "cpg": "126v6mdoT-EfkpRVxtm1LFmMyxYeV3_kk",
        },
        "north_america": {
            "shp": "1AOwUIzB17t8jaohOnlpSXCB-a0KszIH4",
            "shx": "1U6n6eYM1k7Vgwi8KNN80xRc2t0jtFbuG",
            "dbf": "1Dx5lEvwsEW5qzX6vP68eMjgj21oxL1Iy",
            "cpg": "1yP7HOQzlslJGPDlPZYp7bn7Jz4Qi0yvh",
        },
        "arctic": {
            "shp": "1bWBlkjfshw5L_dHjMpZn6J5dB1_RVNuf",
            "shx": "1bX2REVuH3NHeJtTK_OqNcfm4Nvhqt8K9",
            "dbf": "1rJ9WlK1GXh_kjTcuy0h_-G4W_sl5V9Jz",
            "cpg": "1VotyKZn1MnDTDLuF0_2RvoYs66xc7K7O",
        },
        "greenland": {
            "shp": "1blM-7L1syXkpkliubbS5CMGhchgbGSp6",
            "shx": "1wYtb5VuD1LdyH1fc583le3vZEbjc2HgE",
            "dbf": "13cbq3iYpb-zI3lx4F4Y99-8eau4eD_XG",
            "cpg": "1a-ghdPJ7dQ74pjSuFuX2iOGRuefWzr8F",
        },
    },
    "riv": {
        "africa": {
            "shp": "1xadWTqKLQqyrj9wvr_GKOJPoyQk7tFa_",
            "shx": "12hhyQJZnp_D3MYEjpr0cZPpXwOH0SBnL",
            "dbf": "1DkJyg6GHBjU7TFfG9_Qa75fVNCNvxk4G",
            "cpg": "131Z7ehWjjuf6XXN4OFN0_5Br1wTIiPdk",
            "prj": "1pu5904DYAJWbqmEP4QvpwdQzTgNHihNp",
        },
        "europe": {
            "shp": "1UKIHCe-G8cfKIxy43DpvN8W3xQMPEpW5",
            "shx": "1zv8llloH9N8DtwM_WtCBriAA3RnjVc7k",
            "dbf": "1Yp0MfoMgbIRRKWYBN3akjPmwZEIFLEDg",
            "cpg": "1Wkb9FMPoGKX_C9VF2TXQuMMfgzEdrAhd",
            "prj": "1GAZ4FE7RKmgNEr8abvDAMbZ9j9oItLaa",
        },
        "north_asia": {
            "shp": "1vxlh4Ttr7mkyuGqtVLFZ4Mr8jZ7K0uNe",
            "shx": "1dNWdRDxxnCGUadGxc7NCkQZKiExPxhjg",
            "dbf": "1ytb_PzWs_TLGkuAI_wtXZRcHhRagFTXd",
            "cpg": "1ByMnkM5GC8qLaKmyPISbyNDj-9ueE9AH",
            "prj": "1mG_g6wmmbxceLMQJqRC0eDn4xzOMCsYg",
        },
        "south_asia": {
            "shp": "18MQJ5tiSKQWKIvyqAJL-8FukYv_5rToQ",
            "shx": "176tpQOboS0iS0lwOV9x70CZMUq617zpN",
            "dbf": "1m5j_5-9WmuMHDGppinwvANVvJNfwjXYB",
            "cpg": "1M6eACzWYwdSW-95abYzUotJmETLW-RqY",
            "prj": "1CY6INtO06B3Y3LDYtTNUR3kIxG3_jnIR",
        },
        "oceania": {
            "shp": "1fokOX1Bi7vQAG_yle7gg1CYNfzdxHXkE",
            "shx": "1CBMVFoIfb5vPnfzK5cKso6MZYFU3YK0-",
            "dbf": "1M9S4Ia1ITZA3u9_ZNHEgySLmcXp4HFec",
            "cpg": "1U3CMH3y-jwlLeq1ci3yL-MKsYV03LhiI",
            "prj": "1DK3V9jkNRrfPO8-rKC0eZLFHq4lYJZ1Q",
        },
        "south_america": {
            "shp": "1wFsrj5InLik9s4nyT9IPN0IBS93T_6vM",
            "shx": "1CPqjtwKwJUtqK2xc_XWRKVx8cZ-VG5uS",
            "dbf": "1_sijWSSxl2R6Ypn8VNGykMBrCubzuXeA",
            "cpg": "1ZbcnvKaUUYVD51-DCdzhR8kDamrYCFAB",
            "prj": "1LkMUufSF5ShJM6IxQCFYXgRZxlw8x0JK",
        },
        "north_america": {
            "shp": "1zhWemhBuXnEwJJSswT4xswOGtXXUZ8q7",
            "shx": "1dwpYmiPu3gVlLD97ML3o70CPLSJyR80C",
            "dbf": "1rprGuKV66JfxPWu0AKLb16fcJk4dsrUP",
            "cpg": "1FGF3U0P9rTUZQtzuhlkGMW418hhS-4j6",
            "prj": "1JVwM74SESWaUZSme5nKC_izfrIeDGVxg",
        },
        "arctic": {
            "shp": "1Ud_AxA-988_67pCEGTtWD39_Bmt0LOog",
            "shx": "1TGAfXS1faHPgIstSRa5pJMKdtVbpSt7P",
            "dbf": "1LJJIOZgj_hKZZx0NDDjabeNdD8ZnFEO2",
            "cpg": "1qeqLKi0u6huPTDPK7D2U822XzEHUtA0u",
            "prj": "1IFan_gKFFfQrzxSZaDtarAKAuWhldqMn",
        },
        "greenland": {
            "shp": "11I02Wqe_05ZVOogUVtIvANKQ3wPhhurg",
            "shx": "1LZs6zrTS9wxKouLfcAroBKBkiKKOwoVI",
            "dbf": "1p39wCYc2QGdDuZGAgFQoAenVAYUrwpAX",
            "cpg": "1fSS7bCg29_ksgQbtq2n4QrqIj--zaJ30",
            "prj": "1QFGaYwLVRclNvzRsBDEFhWzB1amAK3Gm",
        },
    },
}


class MeritBasins(Adapter):
    """The MeritBasins adapter for downloading and processing HydroLAKES data.

    Args:
        Adapter: The base Adapter class.
    """

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the MeritBasins adapter.

        Args:
            variable: The variable to download, either 'cat' for catchments or 'riv' for river network.
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

    def fetch(self, url: str) -> Path:
        """Process HydroLAKES zip file to extract and convert to parquet.

        Args:
            url: The URL to download the HydroLAKES zip file from.

        Returns:
            Path to the processed parquet file.
        """
        if not self.is_ready:
            download_path: Path = self.root

            session: requests.Session = requests.Session()

            files: dict[str, dict[str, str]] = FILES[self.variable]
            downloaded_shp_files: list[Path] = []
            all_downloaded_files: list[Path] = []
            for continent, file_dict in files.items():
                for ext, file_id in file_dict.items():
                    file_path = (
                        download_path
                        / f"merit_basins_{self.variable}_{continent}.{ext}"
                    )

                    if not file_path.exists():
                        response = session.get(
                            f"https://drive.google.com/uc?export=download&id={file_id}",
                        )

                        self.check_quota(response.text)

                        # Case 1: small file → direct content
                        if (
                            "content-disposition"
                            in response.headers.get("content-type", "").lower()
                            or "Content-Disposition" in response.headers
                        ):
                            # Just write the file
                            with open(file_path, "wb") as f:
                                f.write(response.content)

                        else:
                            # Case 2: large file → parse HTML form
                            html = response.text

                            # Regex-based parse of hidden inputs
                            inputs = dict(
                                re.findall(r'name="([^"]+)" value="([^"]+)"', html)
                            )

                            if "id" in inputs and "confirm" in inputs:
                                action_url_match = re.search(
                                    r'form[^>]+action="([^"]+)"', html
                                )
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

                    if ext == "shp":
                        downloaded_shp_files.append(file_path)
                    all_downloaded_files.append(file_path)

            gdfs: list[gpd.GeoDataFrame] = []
            for shp_path in downloaded_shp_files:
                gdf: gpd.GeoDataFrame = gpd.read_file(shp_path)
                gdfs.append(gdf)

            merged: gpd.GeoDataFrame = pd.concat(gdfs, ignore_index=True)

            # clean memory
            for gdf in gdfs:
                del gdf
            del gdfs

            merged: gpd.GeoDataFrame = merged.set_crs("EPSG:4326")

            ascending: bool = True
            merged: gpd.GeoDataFrame = merged.sort_values(
                by="COMID", ascending=ascending
            )  # sort by COMID

            # Use a temporary file to avoid partial writes in case of errors
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent, suffix=".parquet", delete=False
            ) as tmp_file:
                tmp_path: Path = Path(tmp_file.name)
                print(
                    "Saving downloaded and processed data to temporary parquet file..."
                )
                merged.to_parquet(
                    path=tmp_path,
                    compression="gzip",
                    write_covering_bbox=True,
                    index=False,
                    sorting_columns=[SortingColumn(0, descending=not ascending)],
                    row_group_size=10_000,
                    schema_version="1.1.0",
                )
                # Atomically move the temporary file to the final path
                tmp_path.rename(self.path)

            sleep(5)  # wait a bit to ensure all file handles are closed

            for file_path in all_downloaded_files:
                file_path.unlink()  # remove downloaded files

        return self


class MeritBasinsCatchments(MeritBasins):
    """The MeritBasinsCat adapter for downloading and processing MERIT Basins catchment data.

    Args:
        MeritBasins: The base MeritBasins class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MeritBasinsCat adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__("cat", *args, **kwargs)


class MeritBasinsRivers(MeritBasins):
    """The MeritBasinsRiv adapter for downloading and processing MERIT Basins river network data.

    Args:
        MeritBasins: The base MeritBasins class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MeritBasinsRiv adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__("riv", *args, **kwargs)
