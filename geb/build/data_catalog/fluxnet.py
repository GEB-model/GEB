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
        """Read the station metadata and daily latent heat flux of the towers in a region.

        Uses the daily FLUXMET files. LE_F_MDS is the gap-filled latent heat flux,
        LE_CORR the same flux corrected for the energy balance closure gap. Their
        quality flag (LE_F_MDS_QC) is the fraction of measured and good-quality
        gap-filled half hours in the day.

        Args:
            geom: A GeoDataFrame representing the geometry of interest.

        Returns:
            A tuple of:
            - A GeoDataFrame of station metadata (site name, IGBP land cover class)
              indexed by station_id, for stations within the input geometry.
            - A long DataFrame with columns station_id, date,
              latent_heat_flux_W_per_m2, latent_heat_flux_corrected_W_per_m2 and
              latent_heat_flux_quality_fraction.
        """
        stations: gpd.GeoDataFrame = read_geom(
            self.path, bbox=geom.total_bounds
        ).set_index("station_id")  # ty:ignore[invalid-assignment]
        stations = stations[stations.geometry.within(geom.union_all())]

        stations_timeseries: list[pd.DataFrame] = []
        site_names: dict[str, str] = {}
        igbp_classes: dict[str, str] = {}
        columns: dict[str, str] = {
            "LE_F_MDS": "latent_heat_flux_W_per_m2",
            "LE_CORR": "latent_heat_flux_corrected_W_per_m2",
            "LE_F_MDS_QC": "latent_heat_flux_quality_fraction",
        }

        for station_id, tower in tqdm(
            stations.iterrows(),
            desc="Reading FLUXNET time series",
            leave=False,
            total=len(stations),
        ):
            with zipfile.ZipFile(self.root / tower["zip_path"], "r") as z:
                names: list[str] = [info.filename for info in z.infolist()]
                bif: list[str] = [
                    n for n in names if "_BIF_" in n and n.endswith(".csv")
                ]
                if bif:
                    with z.open(bif[0]) as f:
                        df_bif: pd.DataFrame = pd.read_csv(f)
                    for variable, target in (
                        ("SITE_NAME", site_names),
                        ("IGBP", igbp_classes),
                    ):
                        values = df_bif.loc[
                            df_bif["VARIABLE"] == variable, "DATAVALUE"
                        ].values
                        if values.size > 0:
                            target[str(station_id)] = str(values[0])

                daily: list[str] = [
                    n for n in names if "FLUXNET_FLUXMET_DD" in n and n.endswith(".csv")
                ]
                if not daily:
                    self.logger.warning(
                        f"No daily FLUXMET CSV found in zip for station {station_id}"
                    )
                    continue
                with z.open(daily[0]) as f:
                    header: list[str] = pd.read_csv(f, nrows=0).columns.tolist()
                with z.open(daily[0]) as f:
                    df: pd.DataFrame = pd.read_csv(
                        f,
                        usecols=["TIMESTAMP"] + [c for c in columns if c in header],
                        na_values=[-9999],
                    )
            df = df.rename(columns=columns)
            for column in columns.values():
                if column not in df:
                    df[column] = np.nan
            df["date"] = pd.to_datetime(df["TIMESTAMP"].astype(str), format="%Y%m%d")
            df["station_id"] = str(station_id)
            stations_timeseries.append(
                df[["station_id", "date", *columns.values()]].dropna(
                    subset=["latent_heat_flux_W_per_m2"]
                )
            )

        if stations_timeseries:
            timeseries: pd.DataFrame = pd.concat(stations_timeseries, ignore_index=True)
        else:
            timeseries = pd.DataFrame(
                columns=np.array(["station_id", "date", *columns.values()])
            )

        stations = stations.drop("zip_path", axis=1)
        stations["site_name"] = [site_names[str(s)] for s in stations.index]
        stations["igbp"] = [igbp_classes[str(s)] for s in stations.index]
        return stations, timeseries
