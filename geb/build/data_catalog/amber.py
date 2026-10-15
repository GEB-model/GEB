"""Read the AMBER atlas of river barriers in Europe."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

from geb.workflows.io import fetch_and_save, write_geom

from .base import Adapter


def prepare_amber(data: pd.DataFrame) -> gpd.GeoDataFrame:
    """Read barrier locations, types and heights from the atlas.

    Args:
        data: AMBER V1 records with WGS84 coordinates in degrees.

    Returns:
        Barrier points in EPSG:4326, with heights in meters.

    Raises:
        ValueError: If required fields or IDs are missing.
    """
    required_fields: set[str] = {
        "GUID",
        "Longitude_WGS84",
        "Latitude_WGS84",
        "LabelAtlas",
        "Height",
    }
    if not required_fields.issubset(data.columns):
        raise ValueError("AMBER is missing required V1 fields.")
    if data["GUID"].isna().any():
        raise ValueError("AMBER IDs must be present.")
    # The atlas repeats two barriers under different basin names.
    data = data.drop_duplicates("GUID")
    longitude: pd.Series = pd.to_numeric(data["Longitude_WGS84"], errors="coerce")
    latitude: pd.Series = pd.to_numeric(data["Latitude_WGS84"], errors="coerce")
    valid: pd.Series = longitude.between(-180, 180) & latitude.between(-90, 90)
    data = data.loc[valid].copy()
    height_m: pd.Series = pd.to_numeric(data["Height"], errors="coerce")
    barriers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "amber_id": data["GUID"],
            "dam_type": data["LabelAtlas"].str.strip().str.title(),
            "dam_hgt_m": height_m.where(np.isfinite(height_m) & (height_m > 0)),
        },
        geometry=gpd.points_from_xy(longitude[valid], latitude[valid]),
        crs=4326,
    )
    return barriers


class AMBER(Adapter):
    """Download and cache the AMBER V1 barrier points."""

    def fetch(self, url: str) -> AMBER:
        """Save the atlas as GeoParquet.

        Args:
            url: URL of the AMBER V1 ZIP archive.

        Returns:
            This adapter, ready to read.

        Raises:
            ValueError: If required atlas fields are missing.
        """  # noqa: DOC502
        if self.is_ready:
            return self
        with TemporaryDirectory(dir=self.root) as folder:
            archive: Path = Path(folder) / "amber.zip"
            fetch_and_save(url=url, file_path=archive, logger=self.logger)
            with ZipFile(archive) as zipped:
                with zipped.open("AMBER_BARRIER_ATLAS_V1.csv") as records:
                    data: pd.DataFrame = pd.read_csv(records, low_memory=False)
            write_geom(prepare_amber(data), self.path, write_covering_bbox=True)
        return self
