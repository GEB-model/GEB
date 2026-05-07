"""Data adapter for GROW global groundwater time series dataset."""

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from geb.workflows.io import RemoteFile, read_geom, write_geom

from .base import Adapter


class GROW(Adapter):
    """Adapter for the GROW global groundwater time series dataset.

    GROW contains > 200 000 groundwater depth / level time series from 55
    countries together with 36 static and dynamic Earth-system attributes.
    The dataset is published as two parquet files inside a single zip archive
    on Zenodo.

    Reference:
        Bäthge et al. (2026). GROW: A Global-Scale Time Series Dataset for
        Groundwater Studies within the Earth System.
        https://doi.org/10.5281/zenodo.15149480
    """

    @property
    def _timeseries_path(self) -> Path:
        """Path to the cached timeseries parquet file.

        Returns:
            Absolute path to grow_timeseries.parquet in the cache root.
        """
        return self.root / "grow_timeseries.parquet"

    @property
    def is_ready(self) -> bool:
        """Check whether both parquet files have already been cached.

        Returns:
            True only when both grow_attributes.parquet and
            grow_timeseries.parquet are present on disk.
        """
        return self.path.exists() and self._timeseries_path.exists()

    def _download_from_zip(self, zip_member: str, url: str) -> io.BytesIO:
        """Download a single member from the remote ZIP into memory.

        Uses HTTP Range requests so that only the requested member is
        transferred rather than the full archive.

        Args:
            zip_member: Path of the member inside the ZIP (e.g.
                ``'data/grow_attributes.parquet'``).
            url: URL of the remote ZIP archive.

        Returns:
            Raw bytes of the requested file.

        Raises:
            FileNotFoundError: If ``zip_member`` is not found in the archive.
        """
        buffer = io.BytesIO()
        with zipfile.ZipFile(RemoteFile(url), "r") as zf:
            try:
                file_info = zf.getinfo(zip_member)
            except KeyError:
                raise FileNotFoundError(
                    f"'{zip_member}' not found in remote zip at {url}"
                )

            with (
                zf.open(zip_member) as source,
                tqdm(
                    total=file_info.file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {zip_member.split('/')[-1]}",
                ) as pbar,
            ):
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    pbar.update(len(chunk))

        return buffer

    def fetch(
        self,
        url: str,
    ) -> GROW:
        """Download and cache the GROW parquet files from Zenodo.

        Both ``grow_attributes.parquet`` and ``grow_timeseries.parquet`` are
        extracted directly from the remote ZIP without downloading the full
        archive (HTTP Range requests are used).  Subsequent calls are no-ops
        once both files are present in the cache.

        Args:
            url: URL of the ``data.zip`` archive on Zenodo.

        Returns:
            GROW: The adapter instance with the cached data.
        """
        if not self.is_ready:
            if not self.path.exists():
                self.logger.info(
                    "Downloading GROW attributes parquet from Zenodo (this may take a few minutes)..."
                )
                attributes = pd.read_parquet(
                    self._download_from_zip("data/grow_attributes.parquet", url)
                )
                attributes = gpd.GeoDataFrame(
                    attributes,
                    geometry=gpd.points_from_xy(
                        attributes["longitude"], attributes["latitude"]
                    ),
                    crs="EPSG:4326",
                )
                write_geom(attributes, self.path, write_covering_bbox=True)
            if not self._timeseries_path.exists():
                self.logger.info(
                    "Downloading GROW timeseries parquet from Zenodo (this may take a few minutes)..."
                )
                self._timeseries_path.write_bytes(
                    self._download_from_zip(
                        "data/grow_timeseries.parquet", url
                    ).getvalue()
                )

        return self

    def read(self, geom: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """Read GROW attributes or timeseries data.

        Args:
            geom: A GeoDataFrame representing the geometry of interest.

        Returns:
            A tuple of:
            - attributes: A GeoDataFrame of GROW stations whose point geometry
                falls within the union of the input geometries, with all
                static and dynamic attributes included.
            - timeseries: A DataFrame of groundwater depth / level time series
                for the returned stations, with outlier change points annotated.
        """
        region_wells: gpd.GeoDataFrame = read_geom(self.path, bbox=geom.total_bounds)

        region_wells: gpd.GeoDataFrame = region_wells[
            region_wells.geometry.within(geom.union_all())
        ].set_index("GROW_ID")

        timeseries: pd.DataFrame = pd.read_parquet(
            self._timeseries_path,
            filters=[
                ("GROW_ID", "in", region_wells.index.tolist()),
            ],
            columns=[
                "GROW_ID",
                "date",
                "groundwater_depth_from_ground_elevation_m",
                "groundwater_water_level_elevation_m_asl",
                "outliers_change_points",
            ],
        )
        # drop time series with no groundwater depth measurements at all (i.e. only nulls)
        timeseries: pd.DataFrame = timeseries.dropna(
            subset=[
                "groundwater_depth_from_ground_elevation_m",
                "groundwater_water_level_elevation_m_asl",
            ],
            how="all",
        )

        for grow_id, well_timeseries in tqdm(
            timeseries.groupby("GROW_ID"),
            desc="Processing GROW timeseries",
            leave=False,
        ):
            # if no values are null, we can skip this well
            if not pd.isnull(
                well_timeseries["groundwater_depth_from_ground_elevation_m"]
            ).any():
                continue

            ground_elevation = region_wells.at[grow_id, "ground_elevation_merit_m_asl"]
            assert not np.isnan(ground_elevation)
            for idx, row in well_timeseries.iterrows():
                if not pd.isnull(row["groundwater_depth_from_ground_elevation_m"]):
                    continue
                else:
                    groundwater_depth_from_ground_elevation_m: np.float64 = (
                        ground_elevation
                        - row["groundwater_water_level_elevation_m_asl"]
                    )
                    assert not np.isnan(groundwater_depth_from_ground_elevation_m)
                    timeseries.at[idx, "groundwater_depth_from_ground_elevation_m"] = (
                        groundwater_depth_from_ground_elevation_m
                    )

        timeseries: pd.DataFrame = timeseries.drop(
            columns=[
                "groundwater_water_level_elevation_m_asl",
            ]
        )

        return region_wells, timeseries
