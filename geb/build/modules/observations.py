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
from tqdm import tqdm

from geb.build.methods import build_method
from geb.build.workflows.river_snapping import (
    SnappingResults,
    plot_snapping,
    snap_point_to_river_network,
)
from geb.workflows.timeseries import regularize_discharge_timeseries

from .base import BuildModelBase


def parse_custom_station_filename(
    station_path: Path,
) -> tuple[float, float, float, str]:
    """Parse coordinates, optional upstream area, and station name from a custom station file path.

    The filename stem must follow one of these two conventions:
    - ``lon_lat+station_name``
    - ``lon_lat_upstream_area+station_name``

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
    zip_source: Path | io.BytesIO,
    zip_name: str,
    logger: logging.Logger | None = None,
) -> list[tuple[Path, pd.DataFrame]]:
    """Extract and read station files from a zip archive.

    Args:
        zip_source: Path to the zip file or BytesIO buffer containing zip archive data.
        zip_name: Display name of the zip file for logging and error reporting.
        logger: Optional logger instance for recording info messages for skipped files.

    Returns:
        A list of tuples of (station_file_path, station_dataframe).

    Raises:
        ValueError: If an unsupported file format is encountered in the zip archive.
    """
    stations: list[tuple[Path, pd.DataFrame]] = []
    with zipfile.ZipFile(zip_source, "r") as zf:
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
                        f"Ignoring file {member_info.filename} in zip archive {zip_name}, as it is not a .csv or .parquet file."
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
            elif member_path.suffix == ".zip":
                with zf.open(member_info) as f:
                    nested_stations: list[tuple[Path, pd.DataFrame]] = (
                        _load_stations_from_zip(
                            io.BytesIO(f.read()),
                            zip_name=f"{zip_name}/{member_info.filename}",
                            logger=logger,
                        )
                    )
                stations.extend(nested_stations)
            else:
                raise ValueError(
                    f"Unsupported file format for station {member_info.filename} in {zip_name}. Only .csv, .parquet, and .zip are supported."
                )
    return stations


def load_custom_river_stations(
    path: Path,
    logger: logging.Logger | None = None,
) -> list[tuple[Path, pd.DataFrame]]:
    """Recursively load custom river station data from a directory, file, or zip archive.

    Scans the given path for CSV and Parquet files as well as ZIP archives containing them.
    Subdirectories and nested archives are traversed. Non-data files like text and markdown
    files or hidden files are skipped.

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
            return _load_stations_from_zip(path, zip_name=path.name, logger=logger)
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
            stations.extend(
                _load_stations_from_zip(
                    item_path, zip_name=item_path.name, logger=logger
                )
            )
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
        Q_station: A DataFrame read with
        station_path: The path to the station file.

    Returns:
        The cleaned station DataFrame indexed by time.

    Raises:
        ValueError: If the processed station DataFrame does not contain exactly one data column (expected 'Q'),
                    or if the first row does not contain exactly two coordinates (longitude and latitude) that can be parsed as floats.
    """
    Q_station["Q"] = Q_station["Q"].astype(np.float32)  # convert to float

    Q_station = regularize_discharge_timeseries(
        Q_station
    )  # regularize the time series to ensure consistent time steps

    # Resample to hourly if frequency is higher than hourly (e.g., 15 min -> 1 h).
    # If frequency is already hourly or lower (e.g., daily), keep as is.
    assert Q_station.index.freq is not None  # ty:ignore[unresolved-attribute]
    if Q_station.index.freq < pd.Timedelta(hours=1):  # ty:ignore[unresolved-attribute]
        Q_station = Q_station.resample("h", label="left").mean()
    elif Q_station.index.freq > pd.Timedelta(  # ty:ignore[unresolved-attribute]
        hours=1
    ) and Q_station.index.freq < pd.Timedelta(days=1):  # ty:ignore[unresolved-attribute]
        Q_station = Q_station.resample("D", label="left").mean()
    elif Q_station.index.freq > pd.Timedelta(days=1):  # ty:ignore[unresolved-attribute]
        raise ValueError(
            f"Time step of station {station_path} is larger than 1 day. Please ensure the time step is hourly or daily."
        )
    else:
        pass  # keep original frequency if it's already hourly or daily

    Q_station.index.name = "time"  # rename index to time

    # delete missing values in the dataframe
    Q_station.dropna(inplace=True)  # drop missing time steps

    # checks
    if Q_station.shape[1] != 1:
        raise ValueError(f"File {station_path} does not have 1 column")
    return Q_station


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
        include_GRDC: bool = True,
        custom_river_stations: str | None = None,
        create_plots: bool = False,
    ) -> None:
        """setup_discharge_observations is responsible for setting up discharge observations from the discharge observations dataset.

        It clips discharge observations to the basin area, and snaps the discharge observations locations to the locations of the GEB discharge simulations, using upstream area estimates recorded in the discharge observations.
        It also saves necessary input data for the model in the input folder, and some additional information in the output folder (e.g snapping plots).
        Additional stations can be added from a custom folder (or zip file) containing station files in either CSV or Parquet format, or zip files containing them.
        Custom station filenames must follow either the lon_lat+station_name.ext or lon_lat_upstream_area+station_name.ext format, where lon and lat are the station coordinates in degrees, upstream_area is the upstream area in m2, and ext is either .csv or .parquet.
        CSV files must contain a datetime index column and a Q discharge column. Parquet files must contain a datetime column and a Q discharge column.

        Args:
            max_uparea_difference_ratio: The maximum allowed difference in upstream area between the discharge observations station and the GEB river segment, as a ratio of the discharge observations upstream area. Default is 0.3 (30%).
            max_spatial_difference_degrees: The maximum allowed spatial difference in degrees between the discharge observations station and the GEB river segment. Default is 0.1 degrees.
            include_GRDC: Whether to include discharge observation stations from the GRDC dataset. Default is True.
            custom_river_stations: Path to a folder or file containing custom river station files in .csv or .parquet format, or .zip archives containing them. Coordinates, optional upstream area in m2, and station name are read from the filename using the lon_lat+station_name.ext or lon_lat_upstream_area+station_name.ext convention. Default is None, which means no custom stations are used.
            create_plots: Whether to create plots of the snapping results for each station. Default is False.
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

            # Filter metadata by region first
            region_obs_metadata = obs_metadata[
                obs_metadata.geometry.within(region_mask.geometry.union_all())
            ]

            needed_ids = region_obs_metadata[
                "discharge_observations_station_ID"
            ].tolist()

            # Select only filtered IDs from the xarray dataset before converting to dataframe
            obs_daily = (
                discharge_observations.runoff_mean.sel(id=needed_ids)
                .astype(np.float32)
                .to_dataframe()
                .reset_index()
                .pivot(index="time", columns="id", values="runoff_mean")
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
                    "x",
                    "y",
                    "discharge_observations_upstream_area_m2",
                    "discharge_observations_river_name",
                    "geometry",
                ],
                crs="EPSG:4326",
            )
            obs_daily = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))

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
                max_existing_id: int = (
                    int(obs_metadata["discharge_observations_station_ID"].max())
                    if not obs_metadata.empty
                    and pd.notna(
                        obs_metadata["discharge_observations_station_ID"].max()
                    )
                    else 0
                )
                next_station_id: int = max(max_existing_id, 0) + 1

                custom_metadata_records: list[dict[str, Any]] = []
                custom_hourly_series: dict[int, pd.Series] = {}
                custom_daily_series: dict[int, pd.Series] = {}

                for station_path, raw_station_data in loaded_stations:
                    station_name: str
                    lon: float
                    lat: float
                    upstream_area_m2: float
                    lon, lat, upstream_area_m2, station_name = (
                        parse_custom_station_filename(station_path)
                    )

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
                            "x": lon,
                            "y": lat,
                            "discharge_observations_upstream_area_m2": upstream_area_m2,
                            "discharge_observations_river_name": "Unknown",
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
        obs_metadata = obs_metadata[
            obs_metadata.geometry.within(region_mask.geometry.union_all())
        ]

        # GRDC provides a fixed UTC offset relative to the national capital.
        # Custom stations are absent from this metadata and therefore default to UTC.
        obs_metadata = obs_metadata.copy()
        if include_GRDC:
            timezone_utc_offsets: pd.Series = (
                discharge_observations.timezone.to_pandas()
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
                "timezone_utc_offset",
            ]
            discharge_snapping_df = pd.DataFrame(columns=np.array(empty_cols))
            discharge_snapping_df.to_excel(
                discharge_snapping_folder / "discharge_snapping.xlsx",
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
            station_coords: tuple[float, float] = (station_row["x"], station_row["y"])

            discharge_observations_uparea_m2 = station_row[
                "discharge_observations_upstream_area_m2"
            ]
            discharge_observations_rivername = station_row[
                "discharge_observations_river_name"
            ]

            # Snap station to river network
            snap_results: SnappingResults | None = snap_point_to_river_network(
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
            closest_point_coords = snap_results.closest_point_coords
            grid_pixel_coords = snap_results.snapped_grid_pixel_lonlat
            closest_river_segment = snap_results.closest_river_segment

            discharge_snapping_results.append(
                {
                    "discharge_observations_station_name": station_name,
                    "discharge_observations_station_ID": station_id,
                    "discharge_observations_river_name": discharge_observations_rivername,
                    "discharge_observations_upstream_area_m2": discharge_observations_uparea_m2,
                    "discharge_observations_station_coords": station_coords,
                    "closest_point_coords": closest_point_coords,
                    "subgrid_pixel_coords": snap_results.subgrid_pixel_coords,
                    "snapped_grid_pixel_lonlat": grid_pixel_coords,
                    "snapped_grid_pixel_xy": snap_results.snapped_grid_pixel_xy,
                    "GEB_upstream_area_from_subgrid": snap_results.geb_uparea_subgrid,
                    "GEB_upstream_area_from_grid": snap_results.geb_uparea_grid,
                    "discharge_observations_to_GEB_upstream_area_ratio": (
                        snap_results.geb_uparea_subgrid
                        / discharge_observations_uparea_m2
                        if snap_results.geb_uparea_subgrid is not None
                        and not np.isnan(discharge_observations_uparea_m2)
                        else np.nan
                    ),
                    "snapping_distance_degrees": snap_results.distance_degrees,
                    "timezone_utc_offset": float(station_row["timezone_utc_offset"]),
                }
            )

            if create_plots:
                plot_snapping(
                    point_id=station_id,
                    output_folder=discharge_snapping_folder,
                    rivers=rivers,
                    upstream_area=upstream_area_grid,
                    original_coords=station_coords,
                    closest_point_coords=closest_point_coords,
                    closest_river_segment=closest_river_segment,
                    grid_pixel_xy=snap_results.snapped_grid_pixel_xy,
                    filename_prefix="discharge_snapping",
                    point_label="Original gauge",
                    title=f"Upstream area grid and gauge snapping for {station_id}",
                )

        self.logger.info("Discharge snapping done for all stations")

        discharge_snapping_df = pd.DataFrame(discharge_snapping_results)

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
