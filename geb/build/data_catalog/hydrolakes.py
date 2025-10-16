"""Data adapter for HydroLAKES data."""

import shutil
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd

from geb.workflows.io import fetch_and_save

from .base import Adapter


class HydroLakes(Adapter):
    """The HydroLakes adapter for downloading and processing HydroLAKES data.

    Args:
        Adapter: The base Adapter class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the HydroLakes adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, url: str) -> Path:
        """Process HydroLAKES zip file to extract and convert to parquet.

        Args:
            url: The URL to download the HydroLAKES zip file from.

        Returns:
            Path to the processed parquet file.
        """
        if not self.is_ready:
            download_path: Path = self.root / url.split(sep="/")[-1]
            fetch_and_save(url=url, file_path=download_path)

            uncompressed_file: Path = download_path.with_suffix(suffix="")
            with zipfile.ZipFile(file=download_path, mode="r") as zip_ref:
                zip_ref.extractall(uncompressed_file)
            download_path.unlink()  # remove zip file
            gdf: gpd.GeoDataFrame = gpd.read_file(
                filename=uncompressed_file / "HydroLAKES_polys_v10.gdb"
            )
            gdf: gpd.GeoDataFrame = gdf.rename(
                columns={
                    "Vol_total": "volume_total",
                    "Dis_avg": "average_discharge",
                    "Hylak_id": "waterbody_id",
                    "Lake_area": "average_area",
                    "Lake_type": "waterbody_type",
                }
            )
            gdf["average_area"] *= 1e6  # convert from km^2 to m^2
            gdf["volume_total"] *= 1e6
            gdf.to_parquet(
                self.path,
                engine="pyarrow",
                compression="gzip",
                compression_level=9,
                write_covering_bbox=True,
            )
            shutil.rmtree(path=uncompressed_file)  # remove uncompressed folder

        return self
