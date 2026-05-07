"""Adapter for FLUXNET datasets."""

import zipfile
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from geb.workflows.io import read_geom, write_geom

from .base import Adapter


class Fluxnet(Adapter):
    """Adapter for FLUXNET datasets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Fluxnet adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def fetch(
        self,
        url: str,
    ) -> Fluxnet:
        """Fetch and process FLUXNET data.

        Because registration is required to download the data, the user must manually
        download the zip files and place them in the fluxnet folder.

        Station locations are extracted from the FLUXNET BIF (Basic Information File)
        which stores site attributes as key-value rows under the VARIABLE and DATAVALUE
        columns (LOCATION_LAT and LOCATION_LONG).

        Args:
            url: The URL where the data can be found.

        Returns:
            Fluxnet: The adapter instance with the processed data.
        """
        if not self.is_ready:
            # The root directory for this adapter
            if not self.root.exists():
                self.root.mkdir(parents=True, exist_ok=True)

            # Check if any zip files exist in the root
            zip_files = list(self.root.glob("*.zip"))

            while not zip_files:
                print(
                    "\033[91mFLUXNET data requires manual download. Please download the data from: "
                    + url
                    + f" and place the zip files at: {self.root}\033[0m"
                )
                input(
                    "\033[91mPress Enter after placing the files to continue...\033[0m"
                )
                zip_files = list(self.root.glob("*.zip"))

            station_data: list[dict] = []

            for zip_path in tqdm(
                zip_files, desc="Processing FLUXNET zip files", leave=False
            ):
                with zipfile.ZipFile(zip_path, "r") as z:
                    # The BIF (Basic Information File) holds per-station metadata as
                    # VARIABLE/DATAVALUE rows — LOCATION_LAT and LOCATION_LONG are
                    # stored as row values, not column headers.
                    bif_files = [
                        fi.filename
                        for fi in z.infolist()
                        if "_BIF_" in fi.filename
                        and fi.filename.endswith(".csv")
                        and "BIFVARINFO" not in fi.filename
                    ]

                    with z.open(bif_files[0]) as f:
                        df_bif = pd.read_csv(f)

                    station_id: str = df_bif["SITE_ID"].iloc[0]
                    lat_values = df_bif.loc[
                        df_bif["VARIABLE"] == "LOCATION_LAT", "DATAVALUE"
                    ].values
                    lon_values = df_bif.loc[
                        df_bif["VARIABLE"] == "LOCATION_LONG", "DATAVALUE"
                    ].values

                    if lat_values.size > 0 and lon_values.size > 0:
                        station_data.append(
                            {
                                "station_id": station_id,
                                "lat": float(lat_values[0]),
                                "lon": float(lon_values[0]),
                                "zip_path": str(zip_path.relative_to(self.root)),
                            }
                        )

            gdf = gpd.GeoDataFrame(
                station_data,
                geometry=gpd.points_from_xy(
                    [d["lon"] for d in station_data] if station_data else [],
                    [d["lat"] for d in station_data] if station_data else [],
                ),
                crs="EPSG:4326",
            )
            gdf: gpd.GeoDataFrame = gdf.drop(columns=["lat", "lon"], axis=1)  # ty:ignore[invalid-assignment]

            write_geom(gdf, self.path, write_covering_bbox=True)

        return self

    def read(self, geom: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """Read the processed station metadata and the latent heat time series.

        Args:
            geom: A GeoDataFrame representing the geometry of interest.

        Returns:
            A tuple of:
            - A GeoDataFrame of station metadata for stations within the input geometry.
            - A DataFrame of latent heat flux time series for those stations.
        """
        stations: gpd.GeoDataFrame = read_geom(
            self.path, bbox=geom.total_bounds
        ).set_index("station_id")  # ty:ignore[invalid-assignment]
        stations: gpd.GeoDataFrame = stations[
            stations.geometry.within(geom.union_all())
        ]

        stations_timeseries: list[pd.DataFrame] = []

        for station_id, tower in tqdm(
            stations.iterrows(),
            desc="Reading FLUXNET time series",
            leave=False,
            total=len(stations),
        ):
            zip_path: str = tower["zip_path"]

            with zipfile.ZipFile(self.root / zip_path, "r") as z:
                for file_info in z.infolist():
                    name = file_info.filename
                    if name.endswith(".csv") and "FLUXNET_FLUXMET_HH" in name:
                        with z.open(name) as f:
                            df: pd.DataFrame = pd.read_csv(f, engine="pyarrow")[
                                [
                                    "TIMESTAMP_START",
                                    "TIMESTAMP_END",
                                    "LE_F_MDS",
                                    "LE_F_MDS_QC",
                                ]
                            ]
                            df["station_id"] = station_id
                            start: pd.Series = pd.to_datetime(
                                df["TIMESTAMP_START"], format="%Y%m%d%H%M"
                            )
                            end: pd.Series = pd.to_datetime(
                                df["TIMESTAMP_END"], format="%Y%m%d%H%M"
                            )
                            df["date"] = start + (end - start) / 2
                            df["latent_heat_flux_W_per_m2"] = df["LE_F_MDS"]

                            # set any non measurement points to NA
                            df.loc[
                                df["LE_F_MDS_QC"] != 0, "latent_heat_flux_W_per_m2"
                            ] = pd.NA

                            # drop original columns and set date as index
                            df: pd.DataFrame = df.drop(
                                columns=[
                                    "TIMESTAMP_START",
                                    "TIMESTAMP_END",
                                    "LE_F_MDS",
                                    "LE_F_MDS_QC",
                                ]
                            )

                            stations_timeseries.append(df)

                        break

                else:
                    self.logger.warning(
                        f"No FLUXMET CSV found in zip for station {station_id}"
                    )

        if stations_timeseries:
            stations_timeseries: pd.DataFrame = pd.concat(
                stations_timeseries, axis=1, ignore_index=True
            )
        else:
            stations_timeseries: pd.DataFrame = pd.DataFrame(
                [],
                columns=np.array(["date", "station_id", "latent_heat_flux_W_per_m2"]),
            )

        stations: gpd.GeoDataFrame = stations.drop("zip_path", axis=1)  # ty:ignore[invalid-assignment]

        return stations, stations_timeseries
