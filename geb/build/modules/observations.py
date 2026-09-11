"""This module contains the classes and functions processing observational data during model building."""

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


def parse_grdc_utc_offset_hours(raw_offset: float) -> float:
    """Convert a GRDC clock-style UTC offset to decimal hours.

    GRDC stores offsets using an ``HH.MM`` representation, as indicated by the
    NetCDF variable unit ``00:00``. For example, ``6.3`` represents 06:30 and
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
        custom_river_stations: str | None = None,
        create_plots: bool = False,
    ) -> None:
        """Prepare and snap discharge observations.

        Stations are matched to an original-resolution river pixel using distance, upstream
        area, and river ID. The matching routing pixel has the same river ID.

        Args:
            custom_river_stations: Folder with CSV or Parquet files named
                ``lon_lat+station_name.ext`` (coordinates in degrees), containing
                a datetime index and a ``Q`` discharge column (m³/s).
            create_plots: Whether to plot each station match.

        Raises:
            ValueError: If custom station data are invalid.
        """
        # load data
        routing_upstream_area: xr.DataArray = self.grid[
            "routing/upstream_area_m2"
        ].compute()  # we need to use this one many times, so we compute it once
        original_upstream_area: xr.DataArray = self.other[
            "drainage/original_d8_upstream_area_m2"
        ]
        original_river_ids: xr.DataArray = self.other["drainage/original_river_ids"]
        routing_river_ids: xr.DataArray = self.grid["routing/river_ids"].compute()
        routing_pixels_by_river_id: dict[int, tuple[np.ndarray, ...]] = (
            group_routing_pixels(routing_river_ids)
        )
        region_mask = self.geom["mask"]

        # Load discharge observations dataset
        discharge_observations = self.data_catalog.fetch("GRDC").read()

        # create folders
        discharge_snapping_folder: Path = Path(self.report_dir) / "discharge_snapping"
        discharge_snapping_folder.mkdir(parents=True, exist_ok=True)

        # Initialize discharge observation DataFrames
        obs_hourly = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))

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

        # Track which IDs belong to which frequency
        hourly_ids = set()

        # Filter metadata by region first
        region_obs_metadata = obs_metadata[
            obs_metadata.geometry.within(region_mask.geometry.union_all())
        ]

        needed_ids = region_obs_metadata["discharge_observations_station_ID"].tolist()

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

        if custom_river_stations is not None:
            custom_river_stations: Path = Path(custom_river_stations)
            if not custom_river_stations.exists():
                self.logger.warning(
                    f"Custom river stations folder {custom_river_stations} does not exist. Skipping custom stations."
                )
            else:
                for station_path in custom_river_stations.iterdir():
                    if station_path.suffix == ".csv":
                        Q_station: pd.DataFrame = pd.read_csv(
                            station_path,
                            delimiter=",",
                            index_col=0,
                            parse_dates=True,
                        )
                    elif station_path.suffix == ".parquet":
                        Q_station: pd.DataFrame = pd.read_parquet(
                            station_path
                        ).set_index("datetime")
                    elif station_path.suffix in (".txt", ".md"):
                        self.logger.info(
                            f"Ignoring file {station_path} in custom river stations folder, as it is not a .csv or .parquet file."
                        )
                        continue  # ignore txt files (e.g., README)
                    else:
                        raise ValueError(
                            f"Unsupported file format for station {station_path}. Only .csv and .parquet are supported."
                        )

                    station_name: str
                    lonlat_str: str
                    lonlat_str, station_name = station_path.stem.split("+", 1)
                    lon_lat: list[str] = lonlat_str.split("_", maxsplit=1)
                    assert len(lon_lat) == 2, (
                        f"Filename {station_path} does not contain valid coordinates. Expected format: 'lon_lat+stationname.csv'"
                    )
                    lon_lat: tuple[float, float] = (
                        float(lon_lat[0]),
                        float(lon_lat[1]),
                    )

                    Q_station = process_station_data(Q_station, station_path)

                    # Assign a unique ID for custom stations
                    station_id = int(
                        max(obs_metadata["discharge_observations_station_ID"].max(), 0)
                        + 1
                    )

                    # Add metadata
                    new_meta = pd.DataFrame(
                        [
                            {
                                "discharge_observations_station_ID": station_id,
                                "discharge_observations_station_name": station_name,
                                "discharge_observations_source": f"custom:{station_path.name}",
                                "x": lon_lat[0],
                                "y": lon_lat[1],
                                "discharge_observations_upstream_area_m2": np.nan,  # Not provided in basic CSV
                                "discharge_observations_river_name": "Unknown",
                            }
                        ]
                    )
                    new_meta_gdf = gpd.GeoDataFrame(
                        new_meta,
                        geometry=gpd.points_from_xy([lon_lat[0]], [lon_lat[1]]),
                        crs="EPSG:4326",
                    )
                    obs_metadata = pd.concat(
                        [obs_metadata, new_meta_gdf], ignore_index=True
                    )

                    # Add data to the correct DataFrame
                    if Q_station.index.to_series().diff().median() <= pd.Timedelta(
                        hours=1
                    ):
                        obs_hourly[station_id] = Q_station["Q"]
                        hourly_ids.add(station_id)
                    else:
                        obs_daily[station_id] = Q_station["Q"]
                        daily_ids.add(station_id)

        # Filter metadata by region
        obs_metadata = obs_metadata[
            obs_metadata.geometry.within(region_mask.geometry.union_all())
        ]

        # GRDC provides a fixed UTC offset relative to the national capital.
        # Custom stations are absent from this metadata and therefore default to UTC.
        raw_timezone_utc_offsets: pd.Series = (
            discharge_observations.timezone.to_pandas()
        )
        timezone_utc_offsets: pd.Series = (
            raw_timezone_utc_offsets.fillna(0.0)
            .astype(float)
            .map(parse_grdc_utc_offset_hours)
        )
        obs_metadata = obs_metadata.copy()
        obs_metadata["timezone_utc_offset"] = (
            obs_metadata["discharge_observations_station_ID"]
            .map(timezone_utc_offsets)
            .fillna(0.0)
            .astype(float)
        )
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
        obs_metadata["evaluation_exclusion_reason"] = (
            obs_metadata["discharge_observations_station_ID"]
            .map(
                {
                    station_id: "Duplicate observations shared with station(s) "
                    + ", ".join(str(value) for value in matching_ids)
                    + " for at least 5 years; the correct location is unknown."
                    for station_id, matching_ids in duplicate_stations.items()
                }
            )
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

        empty_cols: list[str] = [
            "discharge_observations_station_name",
            "discharge_observations_source",
            "discharge_observations_station_ID",
            "discharge_observations_river_name",
            "discharge_observations_country_code",
            "discharge_observations_upstream_area_m2",
            "discharge_observations_station_coords",
            "original_pixel_lonlat",
            "snapped_grid_pixel_lonlat",
            "snapped_grid_pixel_xy",
            "GEB_upstream_area_from_original",
            "GEB_upstream_area_from_grid",
            "discharge_observations_to_GEB_upstream_area_ratio",
            "station_to_original_distance_m",
            "snapped_river_id",
            "snapping_method",
            "timezone_utc_offset",
        ]

        if obs_metadata.empty:
            # No stations found - create empty files
            self.logger.warning(
                "No discharge stations found in the region. Creating empty files"
            )
            # Create empty snapping results Excel file with proper columns
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

        # Snap stations directly to original-resolution river pixels.
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
                station_upstream_area_m2=station_upstream_area_m2,
                original_upstream_area=original_upstream_area,
                original_river_ids=original_river_ids,
                routing_upstream_area=routing_upstream_area,
                routing_pixels_by_river_id=routing_pixels_by_river_id,
            )

            if snap_results is None:
                self.logger.warning(
                    "No valid original river pixel found for station %s. Skipping station.",
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
                    "original_pixel_lonlat": snap_results.original_pixel_lonlat,
                    "snapped_grid_pixel_lonlat": snap_results.routing_pixel_lonlat,
                    "snapped_grid_pixel_xy": snap_results.routing_pixel_xy,
                    "GEB_upstream_area_from_original": snap_results.original_upstream_area_m2,
                    "GEB_upstream_area_from_grid": snap_results.routing_upstream_area_m2,
                    "discharge_observations_to_GEB_upstream_area_ratio": (
                        station_upstream_area_m2 / snap_results.routing_upstream_area_m2
                    ),
                    "station_to_original_distance_m": (
                        snap_results.station_to_original_distance_m
                    ),
                    "snapped_river_id": snap_results.river_id,
                    "snapping_method": "original_pixel_v1",
                    "timezone_utc_offset": float(station_row["timezone_utc_offset"]),
                }
            )

            if create_plots:
                plot_discharge_snapping(
                    station_id=station_id,
                    output_folder=discharge_snapping_folder,
                    station_lonlat=station_lonlat,
                    snapping_result=snap_results,
                    original_upstream_area=original_upstream_area,
                )

        self.logger.info("Discharge snapping done for all stations")

        discharge_snapping_df = pd.DataFrame(
            discharge_snapping_results, columns=empty_cols
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
