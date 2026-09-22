"""Adapter for WorldFloodsv2 dataset from Hugging Face."""

import tempfile
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import xarray as xr
from huggingface_hub import snapshot_download
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape
from tqdm import tqdm

from geb.geb_types import TwoDArrayInt8
from geb.workflows.io import (
    read_geom,
    read_zarr,
    write_geom,
    write_zarr,
)

from .base import Adapter

INVALID: Literal[-2] = -2
CLOUD: Literal[-1] = -1
LAND: Literal[0] = 0
FLOOD: Literal[2] = 2
PERMANENT_WATER: Literal[1] = 1


class WorldFloodsV2(Adapter):
    """Adapter for WorldFloodsv2 dataset from Hugging Face."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the WorldFloodsV2 adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(
        self,
        url: str,
    ) -> WorldFloodsV2:
        """Fetch WorldFloodsv2 data from Hugging Face.

        Args:
            url: The Hugging Face dataset ID (e.g., 'isp-uv-es/WorldFloodsv2').

        Returns:
            WorldFloodsV2: The adapter instance.
        """
        if not self.is_ready:
            self.logger.info(f"Fetching WorldFloodsv2 dataset from {url}...")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                snapshot_download(
                    repo_id=url,
                    repo_type="dataset",
                    local_dir=temp_dir,
                    allow_patterns=[
                        "*/gt/*.tif",
                        "*/PERMANENTWATERJRC/*.tif",
                        "dataset_metadata.csv",
                    ],
                )

                metadata: pd.DataFrame = pd.read_csv(temp_dir / "dataset_metadata.csv")

                all_flood_maps_vectors: list[gpd.GeoDataFrame] = []
                for folder in ("train", "test", "val"):
                    files: list[Path] = list((temp_dir / folder / "gt").glob("*.tif"))
                    for file in tqdm(
                        files, desc=f"Processing WorldFloodsV2 {folder} maps"
                    ):
                        with rasterio.open(
                            temp_dir / folder / "gt" / f"{file.stem}.tif"
                        ) as src:
                            flood_map_raster_cloud = src.read(1)
                            flood_map_raster_water = src.read(2)

                            flood_map_raster: TwoDArrayInt8 = np.full_like(
                                flood_map_raster_cloud,
                                dtype=np.int8,
                                fill_value=INVALID,
                            )  # invalid by default

                            with rasterio.open(
                                temp_dir
                                / folder
                                / "PERMANENTWATERJRC"
                                / f"{file.stem}.tif"
                            ) as src_water:
                                # Initialize the destination array
                                permanent_water = np.full_like(
                                    flood_map_raster, dtype=np.int8, fill_value=-2
                                )

                                # Reproject directly from the source dataset into the destination array
                                rasterio.warp.reproject(
                                    source=rasterio.band(src_water, 1),
                                    destination=permanent_water,
                                    src_transform=src_water.transform,
                                    src_crs=src_water.crs,
                                    dst_transform=src.transform,
                                    dst_crs=src.crs,
                                    resampling=Resampling.nearest,
                                )

                                assert not (permanent_water == -2).any(), (
                                    "Reprojection failed, some pixels remain with the fill value."
                                )

                            flood_map_raster[flood_map_raster_cloud == 2] = CLOUD
                            flood_map_raster[
                                (flood_map_raster_water == 1)
                                & (flood_map_raster_cloud == 1)
                            ] = LAND
                            flood_map_raster[
                                (flood_map_raster_water == 2)
                                & (flood_map_raster_cloud == 1)
                            ] = FLOOD
                            flood_map_raster[
                                (permanent_water == 3) & (flood_map_raster == FLOOD)
                            ] = PERMANENT_WATER

                            flood_map_raster: xr.DataArray = xr.DataArray(
                                flood_map_raster,
                                dims=["y", "x"],
                                coords={
                                    "y": np.arange(src.height) * src.transform[4]
                                    + src.transform[5]
                                    + src.transform[4] / 2,
                                    "x": np.arange(src.width) * src.transform[0]
                                    + src.transform[2]
                                    + src.transform[0] / 2,
                                },
                            ).rio.write_crs(src.crs)

                            flood_map_raster.attrs["_FillValue"] = (
                                INVALID  # Set the fill value for invalid pixels
                            )

                            flood_map_raster.attrs["encoding"] = {
                                "invalid": INVALID,
                                "cloud": CLOUD,
                                "land": LAND,
                                "flood": FLOOD,
                                "permanent_water": PERMANENT_WATER,
                            }

                            # create a vector representation of the flood map for this event
                            mask = flood_map_raster.data != INVALID
                            flood_map_vector = list(
                                shapes(
                                    mask.astype(np.int8),
                                    mask=mask,
                                    transform=src.transform,
                                    connectivity=8,
                                )
                            )

                            flood_map_vector: gpd.GeoDataFrame = (
                                gpd.GeoDataFrame.from_records(
                                    [
                                        {
                                            "geometry": shape(geom),
                                            "name": file.stem,
                                        }
                                        for geom, _ in flood_map_vector
                                    ],
                                )
                            )  # ty:ignore[invalid-assignment]

                            assert len(flood_map_vector) > 0, (
                                f"No valid geometries found for {file.stem}."
                            )

                            flood_map_vector: gpd.GeoDataFrame = (
                                flood_map_vector.set_crs(src.crs)
                            )  # ty:ignore[invalid-assignment]
                            flood_map_vector: gpd.GeoDataFrame = (
                                flood_map_vector.to_crs("EPSG:4326")
                            )

                            # merge all geometries with the same value into a single geometry
                            flood_map_vector = flood_map_vector.union_all()
                            flood_map_vector = gpd.GeoDataFrame(
                                {"geometry": [flood_map_vector], "name": [file.stem]},
                                crs="EPSG:4326",
                            )
                            all_flood_maps_vectors.append(flood_map_vector)

                            write_zarr(
                                flood_map_raster,
                                self.root / "rasters" / f"{file.stem}.zarr",
                                crs=flood_map_raster.rio.crs,
                            )

                all_flood_maps_vectors: gpd.GeoDataFrame = pd.concat(
                    all_flood_maps_vectors, ignore_index=True
                )  # ty:ignore[invalid-assignment]

                metadata: pd.DataFrame = metadata.rename(
                    columns={"event id": "name", "satellite date": "satellite_date"}
                )

                all_flood_maps_vectors: gpd.GeoDataFrame = all_flood_maps_vectors.merge(
                    metadata,
                    left_on="name",
                    right_on="name",
                    how="inner",
                )  # ty:ignore[invalid-assignment]
                write_geom(
                    all_flood_maps_vectors,
                    self.root / "flood_maps.geoparquet",
                    write_covering_bbox=True,
                )
        return self

    def read(
        self, region: gpd.GeoDataFrame
    ) -> tuple[gpd.GeoDataFrame, dict[str, xr.DataArray]]:
        """Read the WorldFloodsv2 metadata.

        Args:
            region: A GeoDataFrame representing the region of interest.

        Returns:
            The flood maps and metadata as GeoDataFrames.
        """
        floods: gpd.GeoDataFrame = read_geom(
            self.root / "flood_maps.geoparquet", bbox=region.total_bounds
        )
        # filter actual overlapping
        floods = floods[floods.intersects(region.union_all())]

        # parse observation_date to datetime and convert to UTC timezone
        floods["observation_date"] = pd.to_datetime(
            floods["satellite_date"], format="mixed", utc=True
        ).dt.tz_localize(None)

        flood_maps: dict[str, xr.DataArray] = {}
        for _, row in floods.iterrows():
            flood_name = row["name"]
            raster_path = self.root / "rasters" / f"{flood_name}.zarr"
            flood_raster = read_zarr(raster_path)
            flood_maps[flood_name] = flood_raster

        return floods, flood_maps
