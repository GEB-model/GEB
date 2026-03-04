"""This module contains the classes and functions processing observational data during model building."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

from geb.build.methods import build_method
from geb.build.workflows.river_snapping import (
    plot_snapping,
    snap_point_to_river_network,
)
from geb.workflows.timeseries import regularize_discharge_timeseries

from .base import BuildModelBase


def process_station_data(
    station: str, Q_station_path: Path, dt_format: str
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Parse and preprocess a station CSV read into a DataFrame.

    Args:
        station: Filename or identifier of the station (used in error messages).
        Q_station_path: Raw station DataFrame where the first row contains coordinates and data starts at row index `startrow`.
        dt_format: Datetime format string for parsing the date column.

    Returns:
        A tuple with the cleaned station DataFrame indexed by time and a tuple with station (lon, lat) as floats.

    Raises:
        ValueError: If the processed station DataFrame does not contain exactly one data column (expected 'Q'),
                    or if the first row does not contain exactly two coordinates (longitude and latitude) that can be parsed as floats.
    """
    Q_station = Q_station_path.read_text().splitlines()
    lon_str, lat_str = Q_station[0].split(",")
    station_coords = (float(lon_str), float(lat_str))

    Q_station = pd.read_csv(
        StringIO("\n".join(Q_station[2:])), delimiter=",", index_col=0, parse_dates=True
    )

    Q_station["Q"] = Q_station["Q"].astype(np.float32)  # convert to float

    Q_station = regularize_discharge_timeseries(
        Q_station
    )  # regularize the time series to ensure consistent time steps

    # Resample to hourly if frequency is higher than hourly (e.g., 15 min -> 1 h).
    # If frequency is already hourly or lower (e.g., daily), keep as is.
    assert Q_station.index.freq is not None  # ty:ignore[possibly-missing-attribute]
    if Q_station.index.freq < pd.Timedelta(hours=1):  # ty:ignore[possibly-missing-attribute]
        Q_station = Q_station.resample("h", label="left").mean()
    elif Q_station.index.freq > pd.Timedelta(  # ty:ignore[possibly-missing-attribute]
        hours=1
    ) and Q_station.index.freq < pd.Timedelta(days=1):  # ty:ignore[possibly-missing-attribute]
        Q_station = Q_station.resample("D", label="left").mean()
    elif Q_station.index.freq > pd.Timedelta(days=1):  # ty:ignore[possibly-missing-attribute]
        raise ValueError(
            f"Time step of station {station} is larger than 1 day. Please ensure the time step is hourly or daily."
        )
    else:
        pass  # keep original frequency if it's already hourly or daily

    Q_station.index.name = "time"  # rename index to time

    # delete missing values in the dataframe
    Q_station.dropna(inplace=True)  # drop missing time steps

    # checks
    if Q_station.shape[1] != 1:
        raise ValueError(f"File {station} does not have 1 column")

    if len(station_coords) != 2:
        raise ValueError(
            f"File {station} does not have 2 coordinates. .csv files of discharge stations should have the coordinates in the first row of the file"
        )
    return Q_station, (station_coords[0], station_coords[1])


class Observations(BuildModelBase):
    """Collects, parses and processes observational data for model evaluation."""

    def __init__(self) -> None:
        """Initialize the Observations class."""
        pass

    @build_method(depends_on=["setup_hydrography"], required=False)
    def setup_discharge_observations(
        self,
        max_uparea_difference_ratio: float = 0.3,
        max_spatial_difference_degrees: float = 0.1,
        custom_river_stations: str | None = None,
    ) -> None:
        """setup_discharge_observations is responsible for setting up discharge observations from the discharge observations dataset.

        It clips discharge observations to the basin area, and snaps the discharge observations locations to the locations of the GEB discharge simulations, using upstream area estimates recorded in the discharge observations.
        It also saves necessary input data for the model in the input folder, and some additional information in the output folder (e.g snapping plots).
        Additional stations can be added as csv files in the custom_stations folder in the GEB data catalog.

        Args:
            max_uparea_difference_ratio: The maximum allowed difference in upstream area between the discharge observations station and the GEB river segment, as a ratio of the discharge observations upstream area. Default is 0.3 (30%).
            max_spatial_difference_degrees: The maximum allowed spatial difference in degrees between the discharge observations station and the GEB river segment. Default is 0.1 degrees.
            custom_river_stations: Path to a folder containing custom river stations as csv files. Each csv file should have the first row containing the coordinates (longitude, latitude) and the data starting from the fourth row. Default is None, which means no custom stations are used.
        """
        # load data
        upstream_area_grid = self.grid[
            "routing/upstream_area_m2"
        ].compute()  # we need to use this one many times, so we compute it once
        upstream_area_subgrid = self.other[
            "drainage/original_d8_upstream_area_m2"
        ].compute()
        rivers = self.geom["routing/rivers"]
        region_mask = self.geom["mask"]

        # Load discharge observations dataset
        discharge_observations = self.data_catalog.fetch("GRDC").read()

        # create folders
        snapping_discharge_folder = Path(self.report_dir) / "snapping_discharge"
        snapping_discharge_folder.mkdir(parents=True, exist_ok=True)

        # Initialize discharge observation DataFrames
        obs_hourly = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
        obs_daily = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))

        # Initialize metadata GeoDataFrame from GRDC
        obs_metadata = gpd.GeoDataFrame(
            {
                "discharge_observations_station_ID": discharge_observations.id.values,
                "discharge_observations_station_name": discharge_observations.station_name.values,
                "x": discharge_observations.x.values,
                "y": discharge_observations.y.values,
                "discharge_observations_upstream_area_m2": discharge_observations.area.values
                * 1e6,  # convert km2 to m2
                "discharge_observations_river_name": discharge_observations.river_name.values,
            },
            geometry=gpd.points_from_xy(
                discharge_observations.x.values, discharge_observations.y.values
            ),
            crs="EPSG:4326",
        )

        # Assume GRDC is daily and extract it
        if "runoff_mean" in discharge_observations.data_vars:
            obs_daily = (
                discharge_observations.runoff_mean.astype(np.float32)
                .to_dataframe()
                .reset_index()
                .pivot(index="time", columns="id", values="runoff_mean")
            )
            obs_daily.index.name = "time"
            # Replace -999 with NaN in GRDC data
            obs_daily = obs_daily.replace(-999, np.nan)

        # Track which IDs belong to which frequency
        hourly_ids = set()
        daily_ids = set(obs_daily.columns.tolist())

        if custom_river_stations is not None:
            for station in os.listdir(Path(self.root).parent / custom_river_stations):
                if not station.endswith(".csv"):
                    continue

                station_name = station[:-4]
                Q_station_path = (
                    Path(self.root).parent / Path(custom_river_stations) / station
                )

                Q_station, station_coords = process_station_data(
                    station,
                    Q_station_path,
                    dt_format="%Y-%m-%d %H:%M:%S",
                )

                # Assign a unique ID for custom stations
                station_id = int(
                    max(obs_metadata["discharge_observations_station_ID"].max(), 0) + 1
                )

                # Add metadata
                new_meta = pd.DataFrame(
                    [
                        {
                            "discharge_observations_station_ID": station_id,
                            "discharge_observations_station_name": station_name,
                            "x": station_coords[0],
                            "y": station_coords[1],
                            "discharge_observations_upstream_area_m2": np.nan,  # Not provided in basic CSV
                            "discharge_observations_river_name": "Unknown",
                        }
                    ]
                )
                new_meta_gdf = gpd.GeoDataFrame(
                    new_meta,
                    geometry=gpd.points_from_xy(
                        [station_coords[0]], [station_coords[1]]
                    ),
                    crs="EPSG:4326",
                )
                obs_metadata = pd.concat(
                    [obs_metadata, new_meta_gdf], ignore_index=True
                )

                # Add data to the correct DataFrame
                if Q_station.index.to_series().diff().median() <= pd.Timedelta(hours=1):
                    obs_hourly[station_id] = Q_station["Q"]
                    hourly_ids.add(station_id)
                else:
                    obs_daily[station_id] = Q_station["Q"]
                    daily_ids.add(station_id)

        # Filter metadata by region
        obs_metadata = obs_metadata[
            obs_metadata.geometry.within(region_mask.geometry.union_all())
        ]

        if obs_metadata.empty:
            # No stations found - create empty files
            self.logger.warning(
                "No discharge stations found in the region. Creating empty files"
            )
            # Create empty snapping results Excel file with proper columns
            empty_cols = [
                "discharge_observations_station_name",
                "discharge_observations_station_ID",
                "discharge_observations_river_name",
                "discharge_observations_upstream_area_m2",
                "discharge_observations_station_coords",
                "closest_point_coords",
                "subgrid_pixel_coords",
                "snapped_grid_pixel_lonlat",
                "snapped_grid_pixel_xy",
                "GEB_upstream_area_from_subgrid",
                "GEB_upstream_area_from_grid",
                "discharge_observations_to_GEB_upstream_area_ratio",
                "snapping_distance_degrees",
            ]
            discharge_snapping_df = pd.DataFrame(columns=np.array(empty_cols))
            discharge_snapping_df.to_excel(
                self.report_dir / "snapping_discharge" / "discharge_snapping.xlsx",
                index=False,
            )

            # Create empty discharge table
            empty_discharge_df = pd.DataFrame()
            self.set_table(
                empty_discharge_df, name="discharge/discharge_observations_hourly"
            )
            self.set_table(
                empty_discharge_df, name="discharge/discharge_observations_daily"
            )

            # Create empty snapped locations geometry
            empty_geom: gpd.GeoDataFrame = gpd.GeoDataFrame(
                discharge_snapping_df,
                geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                crs="EPSG:4326",
            ).set_index(pd.Index([], name="discharge_observations_station_ID"))  # ty:ignore[invalid-assignment]
            self.set_geom(empty_geom, name="discharge/discharge_snapped_locations")

            self.logger.info("Empty discharge datasets created")

            return

        # Snapping to river
        discharge_snapping_results = []

        for _, station_row in tqdm(obs_metadata.iterrows(), total=len(obs_metadata)):
            station_id = station_row["discharge_observations_station_ID"]
            station_name = station_row["discharge_observations_station_name"]
            station_coords = (station_row["x"], station_row["y"])

            discharge_observations_uparea_m2 = station_row[
                "discharge_observations_upstream_area_m2"
            ]
            discharge_observations_rivername = station_row[
                "discharge_observations_river_name"
            ]

            # Snap station to river network
            snap_results = snap_point_to_river_network(
                point=shapely.geometry.Point(station_coords),
                rivers=rivers,
                upstream_area_grid=upstream_area_grid,
                upstream_area_subgrid=upstream_area_subgrid,
                upstream_area_m2=discharge_observations_uparea_m2,
                max_uparea_difference_ratio=max_uparea_difference_ratio,
                max_spatial_difference_degrees=max_spatial_difference_degrees,
            )

            if snap_results is None:
                self.logger.warning(
                    f"No river segment found within criteria for station {station_name} with upstream area {discharge_observations_uparea_m2} m2. Skipping this station."
                )
                continue

            # Extract results
            closest_point_coords = snap_results["closest_point_coords"]
            grid_pixel_coords = snap_results["snapped_grid_pixel_lonlat"]
            closest_river_segment = snap_results["closest_river_segment"]

            discharge_snapping_results.append(
                {
                    "discharge_observations_station_name": station_name,
                    "discharge_observations_station_ID": station_id,
                    "discharge_observations_river_name": discharge_observations_rivername,
                    "discharge_observations_upstream_area_m2": discharge_observations_uparea_m2,
                    "discharge_observations_station_coords": station_coords,
                    "closest_point_coords": closest_point_coords,
                    "subgrid_pixel_coords": snap_results["subgrid_pixel_coords"],
                    "snapped_grid_pixel_lonlat": grid_pixel_coords,
                    "snapped_grid_pixel_xy": snap_results["snapped_grid_pixel_xy"],
                    "GEB_upstream_area_from_subgrid": snap_results[
                        "geb_uparea_subgrid"
                    ],
                    "GEB_upstream_area_from_grid": snap_results["geb_uparea_grid"],
                    "discharge_observations_to_GEB_upstream_area_ratio": (
                        snap_results["geb_uparea_subgrid"]
                        / discharge_observations_uparea_m2
                        if snap_results["geb_uparea_subgrid"] is not None
                        and not np.isnan(discharge_observations_uparea_m2)
                        else np.nan
                    ),
                    "snapping_distance_degrees": snap_results["distance_degrees"],
                }
            )

            plot_snapping(
                point_id=station_id,
                output_folder=self.report_dir / "snapping_discharge",
                rivers=rivers,
                upstream_area=upstream_area_grid,
                original_coords=station_coords,
                closest_point_coords=closest_point_coords,
                closest_river_segment=closest_river_segment,
                grid_pixel_xy=snap_results["snapped_grid_pixel_xy"],
                filename_prefix="snapping_discharge",
                point_label="Original gauge",
                title=f"Upstream area grid and gauge snapping for {station_id}",
            )

        self.logger.info("Discharge snapping done for all stations")

        discharge_snapping_df = pd.DataFrame(discharge_snapping_results)

        # save to excel and parquet files
        discharge_snapping_df.to_excel(
            self.report_dir / "snapping_discharge" / "discharge_snapping.xlsx",
            index=False,
        )  # save the dataframe to an excel file

        discharge_snapping_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
            discharge_snapping_df,
            geometry=gpd.points_from_xy(
                discharge_snapping_df["snapped_grid_pixel_lonlat"].apply(
                    lambda coord: coord[0]
                ),
                discharge_snapping_df["snapped_grid_pixel_lonlat"].apply(
                    lambda coord: coord[1]
                ),
            ),
            crs="EPSG:4326",  # Set the coordinate reference system
        ).set_index("discharge_observations_station_ID")  # ty:ignore[invalid-assignment]

        # Filter the tables based on snapped stations and ensure columns exist even if empty
        snapped_ids = set(discharge_snapping_df["discharge_observations_station_ID"])

        # Prepare final hourly table
        final_hourly_cols = sorted([id for id in hourly_ids if id in snapped_ids])
        obs_hourly_final = obs_hourly.reindex(columns=final_hourly_cols).dropna(
            how="all"
        )
        if obs_hourly_final.empty:
            obs_hourly_final = pd.DataFrame(columns=np.array(final_hourly_cols))
            obs_hourly_final.index.name = "time"
        self.set_table(obs_hourly_final, name="discharge/discharge_observations_hourly")

        # Prepare final daily table
        final_daily_cols = sorted([id for id in daily_ids if id in snapped_ids])
        # Resample daily stations to a daily index to remove hourly timestamps if any
        obs_daily_final = obs_daily.reindex(columns=final_daily_cols)
        if not obs_daily_final.empty:
            # Ensure frequency is strictly daily
            obs_daily_final = (
                obs_daily_final.resample("D", label="left").mean().dropna(how="all")
            )

        if obs_daily_final.empty:
            obs_daily_final = pd.DataFrame(columns=np.array(final_daily_cols))
            obs_daily_final.index.name = "time"

        self.set_table(obs_daily_final, name="discharge/discharge_observations_daily")

        self.set_geom(
            discharge_snapping_gdf, name="discharge/discharge_snapped_locations"
        )
