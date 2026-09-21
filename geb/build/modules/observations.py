"""This module contains the classes and functions processing observational data during model building."""

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from tqdm import tqdm

from geb.build.methods import build_method
from geb.build.workflows.discharge_observations import (
    find_duplicate_discharge_stations,
)
from geb.build.workflows.discharge_snapping import (
    DischargeSnappingResults,
    group_routing_pixels,
    plot_discharge_snapping,
    snap_discharge_station,
)
from geb.workflows.timeseries import regularize_discharge_timeseries

from .base import BuildModelBase

DISCHARGE_SNAPPING_COLUMNS: list[str] = [
    "discharge_observations_station_name",
    "discharge_observations_source",
    "discharge_observations_station_ID",
    "discharge_observations_river_name",
    "discharge_observations_country_code",
    "discharge_observations_upstream_area_m2",
    "discharge_observations_station_coords",
    "original_subgrid_pixel_lonlat",
    "snapped_grid_pixel_lonlat",
    "snapped_grid_pixel_xy",
    "GEB_upstream_area_from_original_subgrid",
    "GEB_upstream_area_from_grid",
    "discharge_observations_to_GEB_upstream_area_ratio",
    "station_to_original_subgrid_distance_m",
    "snapped_river_id",
    "snapping_method",
    "timezone_utc_offset",
]


def parse_grdc_utc_offset_hours(raw_offset: float) -> float:
    """Convert a GRDC clock-style UTC offset to decimal hours.

    GRDC stores offsets using an HH.MM representation, as indicated by the
    NetCDF variable unit 00:00. For example, 6.3 represents 06:30 and
    must become 6.5 decimal hours before it is passed to pandas.

    Args:
        raw_offset: GRDC UTC offset in clock-style hours and minutes (HH.MM).

    Returns:
        UTC offset in decimal hours.

    Raises:
        ValueError: If the offset is non-finite, contains invalid minutes, or
            lies outside the UTC-12 to UTC+14 range.
    """
    if not np.isfinite(raw_offset):
        raise ValueError("GRDC UTC offset must be finite.")

    whole_hours: int = int(raw_offset)
    clock_minutes: int = round((raw_offset - whole_hours) * 100.0)
    if abs(clock_minutes) >= 60:
        raise ValueError(
            f"GRDC UTC offset {raw_offset} contains invalid clock minutes."
        )

    decimal_offset_hours: float = whole_hours + clock_minutes / 60.0
    if not -12.0 <= decimal_offset_hours <= 14.0:
        raise ValueError(f"GRDC UTC offset {raw_offset} is outside UTC-12 to UTC+14.")
    return decimal_offset_hours


def parse_custom_station_filename(
    station_path: Path,
) -> tuple[float, float, float, str]:
    """Parse coordinates, optional upstream area, and station name from a custom station file path.

    The filename stem must follow one of these two conventions:
    - lon_lat+station_name
    - lon_lat_upstream_area+station_name

    Args:
        station_path: Path to the station file.

    Returns:
        A tuple of (longitude, latitude, upstream_area_m2, station_name), where
        longitude and latitude are in degrees, upstream_area_m2 is in m2 (np.nan if not provided),
        and station_name is a string.

    Raises:
        ValueError: If the filename does not contain '+' separator, contains an invalid number of
            underscore-separated metadata parts, or coordinates/upstream area cannot be converted to floats.
    """
    if "+" not in station_path.stem:
        raise ValueError(
            f"Filename '{station_path.name}' does not contain '+' separator. "
            "Expected format: 'lon_lat+station_name.ext' or 'lon_lat_upstream_area+station_name.ext'."
        )

    metadata_str: str
    station_name: str
    metadata_str, station_name = station_path.stem.split("+", 1)

    parts: list[str] = metadata_str.split("_")
    if len(parts) == 2:
        try:
            lon: float = float(parts[0])
            lat: float = float(parts[1])
        except ValueError as err:
            raise ValueError(
                f"Filename '{station_path.name}' does not contain valid numeric coordinates. "
                "Expected format: 'lon_lat+station_name.ext' or 'lon_lat_upstream_area+station_name.ext'."
            ) from err
        upstream_area_m2: float = np.nan
    elif len(parts) == 3:
        try:
            lon: float = float(parts[0])
            lat: float = float(parts[1])
            upstream_area_m2: float = float(parts[2])
        except ValueError as err:
            raise ValueError(
                f"Filename '{station_path.name}' does not contain valid numeric coordinates or upstream area. "
                "Expected format: 'lon_lat+station_name.ext' or 'lon_lat_upstream_area+station_name.ext'."
            ) from err
    else:
        raise ValueError(
            f"Filename '{station_path.name}' contains {len(parts)} metadata parts before '+'. "
            "Expected format: 'lon_lat+station_name.ext' (2 parts) or 'lon_lat_upstream_area+station_name.ext' (3 parts)."
        )

    return lon, lat, upstream_area_m2, station_name


def _load_stations_from_zip(
    zip_path: Path,
    logger: logging.Logger | None = None,
) -> list[tuple[Path, pd.DataFrame]]:
    """Extract and read station files from a zip archive.

    Args:
        zip_path: Path to the zip file.
        logger: Optional logger instance for recording info messages for skipped files.

    Returns:
        A list of tuples of (station_file_path, station_dataframe).

    Raises:
        ValueError: If an unsupported file format is encountered in the zip archive.
    """
    stations: list[tuple[Path, pd.DataFrame]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member_info in zf.infolist():
            if member_info.is_dir():
                continue
            member_path: Path = Path(member_info.filename)
            # Skip hidden files, system files, and macOS metadata
            if member_path.name.startswith(".") or (
                len(member_path.parts) > 0 and member_path.parts[0] == "__MACOSX"
            ):
                continue
            if member_path.suffix in (".txt", ".md") or member_path.name in (
                ".DS_Store",
                "Thumbs.db",
            ):
                if logger is not None:
                    logger.info(
                        f"Ignoring file {member_info.filename} in zip archive {zip_path.name}, as it is not a .csv or .parquet file."
                    )
                continue
            if member_path.suffix == ".csv":
                with zf.open(member_info) as f:
                    q_df: pd.DataFrame = pd.read_csv(
                        f,
                        delimiter=",",
                        index_col=0,
                        parse_dates=True,
                    )
                stations.append((member_path, q_df))
            elif member_path.suffix == ".parquet":
                with zf.open(member_info) as f:
                    # BytesIO is required because pd.read_parquet needs a seekable buffer
                    q_df = pd.read_parquet(io.BytesIO(f.read())).set_index("datetime")
                stations.append((member_path, q_df))
            else:
                raise ValueError(
                    f"Unsupported file format for station {member_info.filename} in {zip_path.name}. Only .csv and .parquet are supported."
                )
    return stations


def load_custom_river_stations(
    path: Path,
    logger: logging.Logger | None = None,
) -> list[tuple[Path, pd.DataFrame]]:
    """Recursively load custom river station data from a directory, file, or zip archive.

    Scans the given path for CSV and Parquet files as well as ZIP archives containing them.
    Subdirectories are traversed. Non-data files like text and markdown files or hidden files
    are skipped.

    Args:
        path: Path to a file, directory, or zip archive containing custom station data.
        logger: Optional logger instance for recording info messages for skipped non-data files.

    Returns:
        A list of tuples containing the station file Path (used for metadata parsing and error reporting)
        and the loaded raw pd.DataFrame.

    Raises:
        ValueError: If an unsupported file format is encountered (other than supported extensions and ignored files).
    """
    stations: list[tuple[Path, pd.DataFrame]] = []
    if path.is_file():
        if path.suffix == ".zip":
            return _load_stations_from_zip(path, logger=logger)
        elif path.suffix == ".csv":
            q_df: pd.DataFrame = pd.read_csv(
                path,
                delimiter=",",
                index_col=0,
                parse_dates=True,
            )
            return [(path, q_df)]
        elif path.suffix == ".parquet":
            q_df = pd.read_parquet(path).set_index("datetime")
            return [(path, q_df)]
        elif path.suffix in (".txt", ".md") or path.name.startswith("."):
            if logger is not None:
                logger.info(
                    f"Ignoring file {path} in custom river stations, as it is not a .csv or .parquet file."
                )
            return []
        else:
            raise ValueError(
                f"Unsupported file format for station {path}. Only .csv, .parquet, and .zip are supported."
            )

    for item_path in sorted(path.rglob("*")):
        if item_path.is_dir():
            continue
        if item_path.name.startswith(".") or "__MACOSX" in item_path.parts:
            continue
        if item_path.suffix in (".txt", ".md") or item_path.name in (
            ".DS_Store",
            "Thumbs.db",
        ):
            if logger is not None:
                logger.info(
                    f"Ignoring file {item_path} in custom river stations folder, as it is not a .csv or .parquet file."
                )
            continue
        if item_path.suffix == ".zip":
            stations.extend(_load_stations_from_zip(item_path, logger=logger))
        elif item_path.suffix == ".csv":
            q_df = pd.read_csv(
                item_path,
                delimiter=",",
                index_col=0,
                parse_dates=True,
            )
            stations.append((item_path, q_df))
        elif item_path.suffix == ".parquet":
            q_df = pd.read_parquet(item_path).set_index("datetime")
            stations.append((item_path, q_df))
        else:
            raise ValueError(
                f"Unsupported file format for station {item_path}. Only .csv, .parquet, and .zip are supported."
            )

    return stations


def process_station_data(Q_station: pd.DataFrame, station_path: Path) -> pd.DataFrame:
    """Parse and preprocess a station CSV read into a DataFrame.

    Args:
        Q_station: A DataFrame read with station discharge data.
        station_path: The path to the station file.

    Returns:
        The cleaned station DataFrame indexed by time.

    Raises:
        TypeError: If the station index is not a DatetimeIndex.
        ValueError: If the processed station DataFrame does not contain exactly one data column (expected 'Q'),
            or if the time step is greater than 1 day.
    """
    if not isinstance(Q_station.index, pd.DatetimeIndex):
        raise TypeError("Station index must be a DatetimeIndex")

    # Convert any timezone-aware datetime index to UTC and make it timezone-naive
    if Q_station.index.tz is not None:
        Q_station.index = Q_station.index.tz_convert("UTC").tz_localize(None)

    Q_station["Q"] = Q_station["Q"].astype(np.float32)  # convert to float

    Q_station = regularize_discharge_timeseries(
        Q_station
    )  # regularize the time series to ensure consistent time steps

    # Resample to hourly if frequency is higher than hourly (e.g., 15 min -> 1 h).
    # If frequency is already hourly or lower (e.g., daily), keep as is.
    assert Q_station.index.freq is not None  # ty:ignore[unresolved-attribute]
    if Q_station.index.freq < pd.Timedelta(hours=1):  # ty:ignore[unresolved-attribute]
        Q_station = Q_station.resample("h", label="left").mean()
        # Represent the hour by the middle of the hour (30 minutes offset)
        Q_station.index = Q_station.index + pd.Timedelta(minutes=30)
    elif Q_station.index.freq > pd.Timedelta(  # ty:ignore[unresolved-attribute]
        hours=1
    ) and Q_station.index.freq < pd.Timedelta(days=1):  # ty:ignore[unresolved-attribute]
        Q_station = Q_station.resample("D", label="left").mean()
        # Offset index by 12 hours so that daily observations are in the middle of the day (12:00:00).
        Q_station.index = Q_station.index + pd.Timedelta(hours=12)
    elif Q_station.index.freq > pd.Timedelta(days=1):  # ty:ignore[unresolved-attribute]
        raise ValueError(
            f"Time step of station {station_path} is larger than 1 day. Please ensure the time step is hourly or daily."
        )
    else:
        # If already daily with timestamps at 00:00:00, shift to the middle of the day (12:00:00)
        if (
            Q_station.index.freq == pd.Timedelta(days=1)  # ty:ignore[unresolved-attribute]
            and len(Q_station.index) > 0
            and Q_station.index[0].hour == 0
        ):
            Q_station.index = Q_station.index + pd.Timedelta(hours=12)

    Q_station.index.name = "time"  # rename index to time

    # delete missing values in the dataframe
    Q_station.dropna(inplace=True)  # drop missing time steps

    # checks
    if Q_station.shape[1] != 1:
        raise ValueError(f"File {station_path} does not have 1 column")
    return Q_station


def _validate_discharge_observation_timestamps(
    df: pd.DataFrame, frequency: str
) -> None:
    """Validate that discharge observation timestamps are centered on the interval midpoint.

    Args:
        df: DataFrame indexed by DatetimeIndex to validate.
        frequency: Expected frequency ('hourly' or 'daily').

    Raises:
        ValueError: If frequency is 'hourly' and timestamps are not on the half hour (HH:30:00),
            or if frequency is 'daily' and timestamps are not at the middle of the day (12:00:00).
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return

    time_series: pd.Series = df.index.to_series()
    if frequency == "hourly":
        invalid_mask = (
            (time_series.dt.minute != 30)
            | (time_series.dt.second != 0)
            | (time_series.dt.microsecond != 0)
        )
        if invalid_mask.any():
            first_invalid = time_series[invalid_mask].iloc[0]
            raise ValueError(
                f"Hourly discharge observations must be timestamped on the half hour (HH:30:00). "
                f"Found invalid timestamp: {first_invalid}."
            )
    elif frequency == "daily":
        invalid_mask = (
            (time_series.dt.hour != 12)
            | (time_series.dt.minute != 0)
            | (time_series.dt.second != 0)
            | (time_series.dt.microsecond != 0)
        )
        if invalid_mask.any():
            first_invalid = time_series[invalid_mask].iloc[0]
            raise ValueError(
                f"Daily discharge observations must be timestamped at the middle of the day (12:00:00). "
                f"Found invalid timestamp: {first_invalid}."
            )
    else:
        raise ValueError(
            f"Unsupported frequency '{frequency}' for timestamp validation. Expected 'hourly' or 'daily'."
        )


class Observations(BuildModelBase):
    """Collects, parses and processes observational data for model evaluation."""

    def __init__(self) -> None:
        """Initialize the Observations class."""

    def _create_empty_discharge_datasets(self, discharge_snapping_folder: Path) -> None:
        """Create empty discharge observation tables and snapping report.

        Args:
            discharge_snapping_folder: Folder where the empty snapping report Excel file is written.
        """
        discharge_snapping_folder.mkdir(parents=True, exist_ok=True)
        discharge_snapping_df: pd.DataFrame = pd.DataFrame(
            columns=np.array(DISCHARGE_SNAPPING_COLUMNS)
        )
        discharge_snapping_df.to_excel(
            discharge_snapping_folder / "discharge_snapping.xlsx",
            index=False,
        )

        # Create empty discharge table
        empty_discharge_df: pd.DataFrame = pd.DataFrame()
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

        # Create empty station locations geometry
        empty_station_locations: gpd.GeoDataFrame = gpd.GeoDataFrame(
            columns=[
                "discharge_observations_station_name",
                "discharge_observations_source",
                "x",
                "y",
                "discharge_observations_upstream_area_m2",
                "discharge_observations_river_name",
                "discharge_observations_country_code",
                "timezone_utc_offset",
                "evaluation_exclusion_reason",
            ],
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        ).set_index(pd.Index([], name="discharge_observations_station_ID"))  # ty:ignore[invalid-assignment]
        self.set_geom(empty_station_locations, name="discharge/station_locations")

        self.logger.info("Empty discharge datasets created")

    @build_method(depends_on=["setup_hydrography"], required=False)
    def setup_discharge_observations(
        self,
        include_GRDC: bool = True,
        custom_river_stations: str | None = None,
        create_plots: bool = False,
    ) -> None:
        """Prepare and snap discharge observations.

        Stations are matched to an original subgrid river pixel using distance, upstream
        area, and river ID. The matching routing pixel has the same river ID.
        Additional stations can be added from a custom folder (or zip file) containing station files
        in either CSV or Parquet format, or zip files containing them.
        Custom station filenames must follow either the lon_lat+station_name.ext or
        lon_lat_upstream_area+station_name.ext format, where lon and lat are the station
        coordinates in degrees, upstream_area is the upstream area in m2, and ext is either
        .csv or .parquet.
        CSV files must contain a datetime index column and a Q discharge column. Parquet files
        must contain a datetime column and a Q discharge column.

        Args:
            include_GRDC: Whether to include discharge observation stations from the GRDC dataset. Default is True.
            custom_river_stations: Path to a folder, file, or zip archive containing custom river station
                files named lon_lat+station_name.ext or lon_lat_upstream_area+station_name.ext
                (coordinates in degrees, optional upstream area in m2), containing a datetime index
                and a Q discharge column (m³/s). Default is None.
            create_plots: Whether to plot each station match.

        """
        # load data
        routing_upstream_area: xr.DataArray = self.grid[
            "routing/upstream_area_m2"
        ].compute()  # we need to use this one many times, so we compute it once
        # Load once to avoid decompressing the same raster chunks for every station.
        original_subgrid_upstream_area: xr.DataArray = self.other[
            "drainage/original_d8_upstream_area_m2"
        ].compute()
        original_subgrid_river_ids: xr.DataArray = self.other[
            "drainage/original_river_ids"
        ].compute()
        routing_river_ids: xr.DataArray = self.grid["routing/river_ids"].compute()
        routing_pixels_by_river_id: dict[int, tuple[np.ndarray, ...]] = (
            group_routing_pixels(routing_river_ids)
        )
        region_mask = self.geom["mask"]
        region_geometry: shapely.Geometry = region_mask.geometry.union_all()

        # create folders
        discharge_snapping_folder: Path = Path(self.report_dir) / "discharge_snapping"
        discharge_snapping_folder.mkdir(parents=True, exist_ok=True)

        # Initialize discharge observation DataFrames
        obs_hourly = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
        hourly_ids: set[int] = set()
        daily_ids: set[int] = set()

        if include_GRDC:
            # Load discharge observations dataset
            discharge_observations = self.data_catalog.fetch("GRDC").read()

            # Initialize metadata GeoDataFrame from GRDC
            obs_metadata = gpd.GeoDataFrame(
                {
                    "discharge_observations_station_ID": discharge_observations.id.values,
                    "discharge_observations_station_name": discharge_observations.station_name.values,
                    "discharge_observations_source": "GRDC",
                    "x": discharge_observations.x.values,
                    "y": discharge_observations.y.values,
                    "discharge_observations_upstream_area_m2": discharge_observations.area.values
                    * 1e6,  # convert km2 to m2
                    "discharge_observations_river_name": discharge_observations.river_name.values,
                    "discharge_observations_country_code": discharge_observations.country.values,
                },
                geometry=gpd.points_from_xy(
                    discharge_observations.x.values, discharge_observations.y.values
                ),
                crs="EPSG:4326",
            )

            # Filter metadata by region first
            region_obs_metadata = obs_metadata[
                obs_metadata.geometry.within(region_geometry)
            ]

            needed_ids = region_obs_metadata[
                "discharge_observations_station_ID"
            ].tolist()

            # Select only filtered IDs from the xarray dataset before converting to dataframe
            obs_daily = (
                discharge_observations.runoff_mean.sel(id=needed_ids)
                .astype(np.float32)
                .transpose("time", "id")
                .to_pandas()
            )
            obs_daily.index.name = "time"
            # Replace -999 with NaN in GRDC data
            obs_daily = obs_daily.replace(-999, np.nan)
            daily_ids = set(obs_daily.columns.tolist())
        else:
            obs_metadata = gpd.GeoDataFrame(
                columns=[
                    "discharge_observations_station_ID",
                    "discharge_observations_station_name",
                    "discharge_observations_source",
                    "x",
                    "y",
                    "discharge_observations_upstream_area_m2",
                    "discharge_observations_river_name",
                    "discharge_observations_country_code",
                    "geometry",
                ],
                crs="EPSG:4326",
            )
            obs_daily = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
            needed_ids = []

        if custom_river_stations is not None:
            custom_river_stations_path: Path = Path(custom_river_stations)
            if not custom_river_stations_path.exists():
                self.logger.warning(
                    f"Custom river stations path {custom_river_stations_path} does not exist. Skipping custom stations."
                )
            else:
                loaded_stations: list[tuple[Path, pd.DataFrame]] = (
                    load_custom_river_stations(
                        custom_river_stations_path, logger=self.logger
                    )
                )

                if obs_metadata.empty:
                    next_station_id: int = 1
                else:
                    next_station_id = (
                        int(obs_metadata["discharge_observations_station_ID"].max()) + 1
                    )

                custom_metadata_records: list[dict[str, Any]] = []
                custom_hourly_series: dict[int, pd.Series] = {}
                custom_daily_series: dict[int, pd.Series] = {}

                min_x: float
                min_y: float
                max_x: float
                max_y: float
                min_x, min_y, max_x, max_y = self.bounds

                for station_path, raw_station_data in tqdm(
                    loaded_stations, desc="Loading and checking custom river stations"
                ):
                    station_name: str
                    lon: float
                    lat: float
                    upstream_area_m2: float
                    lon, lat, upstream_area_m2, station_name = (
                        parse_custom_station_filename(station_path)
                    )

                    # Only process station data if coordinates are within the model domain bounds
                    if not (min_x <= lon <= max_x and min_y <= lat <= max_y):
                        continue

                    # As a second test, check if the station point is within the actual region geometry
                    station_point: shapely.geometry.Point = shapely.geometry.Point(
                        lon, lat
                    )
                    if not station_point.within(region_geometry):
                        continue

                    Q_station: pd.DataFrame = process_station_data(
                        raw_station_data, station_path
                    )

                    station_id: int = next_station_id
                    next_station_id += 1

                    # Collect metadata record
                    custom_metadata_records.append(
                        {
                            "discharge_observations_station_ID": station_id,
                            "discharge_observations_station_name": station_name,
                            "discharge_observations_source": f"custom:{station_path.name}",
                            "x": lon,
                            "y": lat,
                            "discharge_observations_upstream_area_m2": upstream_area_m2,
                            "discharge_observations_river_name": "Unknown",
                            "discharge_observations_country_code": "",
                        }
                    )

                    # Collect series in dictionary to avoid dataframe column insertion fragmentation
                    q_series: pd.Series = Q_station["Q"].rename(station_id)
                    if Q_station.index.to_series().diff().median() <= pd.Timedelta(
                        hours=1
                    ):
                        custom_hourly_series[station_id] = q_series
                        hourly_ids.add(station_id)
                    else:
                        custom_daily_series[station_id] = q_series
                        daily_ids.add(station_id)

                if custom_metadata_records:
                    custom_meta_df: pd.DataFrame = pd.DataFrame(custom_metadata_records)
                    custom_meta_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
                        custom_meta_df,
                        geometry=gpd.points_from_xy(
                            custom_meta_df["x"], custom_meta_df["y"]
                        ),
                        crs="EPSG:4326",
                    )
                    obs_metadata = pd.concat(
                        [obs_metadata, custom_meta_gdf], ignore_index=True
                    )

                if custom_hourly_series:
                    custom_hourly_df: pd.DataFrame = pd.DataFrame(custom_hourly_series)
                    obs_hourly = (
                        pd.concat([obs_hourly, custom_hourly_df], axis=1)
                        if not obs_hourly.empty
                        else custom_hourly_df
                    )

                if custom_daily_series:
                    custom_daily_df: pd.DataFrame = pd.DataFrame(custom_daily_series)
                    obs_daily = (
                        pd.concat([obs_daily, custom_daily_df], axis=1)
                        if not obs_daily.empty
                        else custom_daily_df
                    )

        # Filter metadata by region
        obs_metadata = obs_metadata[obs_metadata.geometry.within(region_geometry)]

        # GRDC provides a fixed UTC offset relative to the national capital.
        # Custom stations are absent from this metadata and therefore default to UTC.
        obs_metadata = obs_metadata.copy()
        if include_GRDC and needed_ids:
            raw_timezone_utc_offsets: pd.Series = discharge_observations.timezone.sel(
                id=needed_ids
            ).to_pandas()
            timezone_utc_offsets: pd.Series = (
                raw_timezone_utc_offsets.fillna(0.0)
                .astype(float)
                .map(parse_grdc_utc_offset_hours)
            )
            obs_metadata["timezone_utc_offset"] = (
                obs_metadata["discharge_observations_station_ID"]
                .map(timezone_utc_offsets)
                .fillna(0.0)
                .astype(float)
            )
        else:
            obs_metadata["timezone_utc_offset"] = 0.0

        if obs_metadata.empty:
            # No stations found - create empty files
            self.logger.warning(
                "No discharge stations found in the region. Creating empty files"
            )
            self._create_empty_discharge_datasets(discharge_snapping_folder)
            return

        regional_station_ids: set[Any] = set(
            obs_metadata["discharge_observations_station_ID"]
        )
        duplicate_stations: dict[Any, list[Any]] = find_duplicate_discharge_stations(
            obs_daily.reindex(
                columns=[
                    station_id
                    for station_id in obs_daily.columns
                    if station_id in regional_station_ids
                ]
            )
        )
        daily_ids.difference_update(duplicate_stations)
        exclusion_reasons: dict[Any, str] = {}
        for station_id, matching_ids in duplicate_stations.items():
            matching_station_ids: str = ", ".join(str(value) for value in matching_ids)
            exclusion_reasons[station_id] = (
                f"Duplicate observations shared with station(s) {matching_station_ids} "
                "for at least 5 years; the correct location is unknown."
            )
        obs_metadata["evaluation_exclusion_reason"] = (
            obs_metadata["discharge_observations_station_ID"]
            .map(exclusion_reasons)
            .fillna("")
        )
        if duplicate_stations:
            self.logger.info(
                "Excluded %d stations with at least 5 years of duplicate observations.",
                len(duplicate_stations),
            )
        # Keep the station inventory independent of snapping and record-length filters.
        self.set_geom(
            gpd.GeoDataFrame(
                obs_metadata.set_index("discharge_observations_station_ID"),
                geometry="geometry",
                crs=obs_metadata.crs,
            ),
            name="discharge/station_locations",
        )

        # Snap stations directly to original subgrid river pixels.
        discharge_snapping_results: list[dict[str, Any]] = []

        for _, station_row in tqdm(obs_metadata.iterrows(), total=len(obs_metadata)):
            station_id = station_row["discharge_observations_station_ID"]
            station_name = station_row["discharge_observations_station_name"]
            station_source = station_row["discharge_observations_source"]
            station_lonlat: tuple[float, float] = (
                station_row["x"],
                station_row["y"],
            )

            station_upstream_area_m2: float = station_row[
                "discharge_observations_upstream_area_m2"
            ]
            station_river_name: str = station_row["discharge_observations_river_name"]

            snap_results: DischargeSnappingResults | None = snap_discharge_station(
                station_location=shapely.geometry.Point(station_lonlat),
                station_upstream_area_m2=(
                    float(station_upstream_area_m2)
                    if np.isfinite(station_upstream_area_m2)
                    else None
                ),
                original_subgrid_upstream_area=original_subgrid_upstream_area,
                original_subgrid_river_ids=original_subgrid_river_ids,
                routing_upstream_area=routing_upstream_area,
                routing_pixels_by_river_id=routing_pixels_by_river_id,
            )

            if snap_results is None:
                self.logger.warning(
                    "Station %s (%s) skipped: no valid snapping match found.",
                    station_id,
                    station_name,
                )
                continue

            discharge_snapping_results.append(
                {
                    "discharge_observations_station_name": station_name,
                    "discharge_observations_source": station_source,
                    "discharge_observations_station_ID": station_id,
                    "discharge_observations_river_name": station_river_name,
                    "discharge_observations_country_code": station_row.get(
                        "discharge_observations_country_code", ""
                    ),
                    "discharge_observations_upstream_area_m2": station_upstream_area_m2,
                    "discharge_observations_station_coords": station_lonlat,
                    "original_subgrid_pixel_lonlat": snap_results.original_subgrid_pixel_lonlat,
                    "snapped_grid_pixel_lonlat": snap_results.routing_pixel_lonlat,
                    "snapped_grid_pixel_xy": snap_results.routing_pixel_xy,
                    "GEB_upstream_area_from_original_subgrid": snap_results.original_subgrid_upstream_area_m2,
                    "GEB_upstream_area_from_grid": snap_results.routing_upstream_area_m2,
                    "discharge_observations_to_GEB_upstream_area_ratio": (
                        station_upstream_area_m2 / snap_results.routing_upstream_area_m2
                        if np.isfinite(station_upstream_area_m2)
                        else np.nan
                    ),
                    "station_to_original_subgrid_distance_m": (
                        snap_results.station_to_original_subgrid_distance_m
                    ),
                    "snapped_river_id": snap_results.river_id,
                    "snapping_method": "original_subgrid_pixel_v1",
                    "timezone_utc_offset": float(station_row["timezone_utc_offset"]),
                }
            )

            if create_plots:
                plot_discharge_snapping(
                    station_id=station_id,
                    output_folder=discharge_snapping_folder,
                    station_lonlat=station_lonlat,
                    snapping_result=snap_results,
                    original_subgrid_upstream_area=original_subgrid_upstream_area,
                )

        self.logger.info("Discharge snapping done for all stations")

        if not discharge_snapping_results:
            self.logger.warning(
                "No discharge stations could be snapped to the river network. Creating empty files"
            )
            self._create_empty_discharge_datasets(discharge_snapping_folder)
            return

        discharge_snapping_df: pd.DataFrame = pd.DataFrame(
            discharge_snapping_results, columns=DISCHARGE_SNAPPING_COLUMNS
        )

        # save to excel and parquet files
        discharge_snapping_df.to_excel(
            discharge_snapping_folder / "discharge_snapping.xlsx",
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
        else:
            _validate_discharge_observation_timestamps(obs_hourly_final, "hourly")
        self.set_table(obs_hourly_final, name="discharge/discharge_observations_hourly")

        # Prepare final daily table
        final_daily_cols = sorted([id for id in daily_ids if id in snapped_ids])
        # Resample daily stations to a daily index to remove hourly timestamps if any
        obs_daily_final = obs_daily.reindex(columns=final_daily_cols)
        if not obs_daily_final.empty:
            # Ensure frequency is strictly daily and timestamped in the middle of the day (12:00:00)
            obs_daily_final = (
                obs_daily_final.resample("D", label="left").mean().dropna(how="all")
            )
            obs_daily_final.index = obs_daily_final.index + pd.Timedelta(hours=12)

        if obs_daily_final.empty:
            obs_daily_final = pd.DataFrame(columns=np.array(final_daily_cols))
            obs_daily_final.index.name = "time"
        else:
            _validate_discharge_observation_timestamps(obs_daily_final, "daily")

        self.set_table(obs_daily_final, name="discharge/discharge_observations_daily")

        self.set_geom(
            discharge_snapping_gdf, name="discharge/discharge_snapped_locations"
        )

    @build_method(depends_on=["setup_hydrography"], required=True)
    def setup_meteorological_stations_observations(self) -> None:
        """Set up meteorological tower observations. Currently only latent heat."""
        # Fetch metadata to find towers in region
        stations, timeseries = self.data_catalog.fetch("fluxnet").read(geom=self.region)

        if stations.empty:
            self.logger.info("No FLUXNET towers found in the region.")

        self.set_table(
            timeseries, name="observations/meteorological_stations_timeseries"
        )
        self.set_geom(stations, name="observations/meteorological_station_locations")

    @build_method(required=True)
    def setup_groundwater_well_observations(self) -> None:
        """Set up groundwater level observations from the GROW dataset.

        Downloads (if not already cached) and reads the GROW global groundwater
        time series dataset, clips well locations to the basin area, and saves
        the time series and well locations.

        Notes:
            Data are downloaded automatically from Zenodo on first use. The
            timeseries file is ~1.7 GB; subsequent runs reuse the local cache.
        """
        wells, timeseries = self.data_catalog.fetch("grow").read(geom=self.region)

        if wells.empty:
            self.logger.info(
                "No GROW groundwater observation wells found in the region."
            )

        self.set_table(timeseries, name="observations/groundwater_well_timeseries")
        self.set_geom(wells, name="observations/groundwater_well_locations")

    @build_method(required=True)
    def setup_flood_observations(self) -> None:
        """Set up flood observations."""
        floods, flood_maps = self.data_catalog.fetch("worldfloodsv2").read(
            region=self.region
        )

        self.set_geom(floods, name="observations/floods")

        for name, flood_map in flood_maps.items():
            self.set_other(flood_map, name=f"observations/floods/{name}")
