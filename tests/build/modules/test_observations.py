"""Tests for the observations build module and helper functions."""

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geb.build.modules.observations import (
    _load_stations_from_zip,
    _validate_discharge_observation_timestamps,
    parse_custom_station_filename,
    process_station_data,
)


def test_parse_custom_station_filename_with_upstream_area() -> None:
    """Test parsing filename containing coordinates, upstream area in m2, and station name."""
    station_path: Path = Path("-0.02754_51.839051_136790000+Rib_at_Wadesmill.parquet")
    lon: float
    lat: float
    upstream_area_m2: float
    station_name: str
    lon, lat, upstream_area_m2, station_name = parse_custom_station_filename(
        station_path
    )

    assert math.isclose(lon, -0.02754)
    assert math.isclose(lat, 51.839051)
    assert math.isclose(upstream_area_m2, 136790000.0)
    assert station_name == "Rib_at_Wadesmill"


def test_parse_custom_station_filename_without_upstream_area() -> None:
    """Test parsing secondary filename format containing only coordinates and station name."""
    station_path: Path = Path("0.12345_52.6789+Sample_Station.csv")
    lon: float
    lat: float
    upstream_area_m2: float
    station_name: str
    lon, lat, upstream_area_m2, station_name = parse_custom_station_filename(
        station_path
    )

    assert math.isclose(lon, 0.12345)
    assert math.isclose(lat, 52.6789)
    assert np.isnan(upstream_area_m2)
    assert station_name == "Sample_Station"


def test_parse_custom_station_filename_missing_plus_separator() -> None:
    """Test that missing '+' separator raises ValueError."""
    station_path: Path = Path("-0.02754_51.839051_136790000_Rib_at_Wadesmill.parquet")
    with pytest.raises(ValueError, match=r"does not contain '\+' separator"):
        parse_custom_station_filename(station_path)


def test_parse_custom_station_filename_invalid_part_count() -> None:
    """Test that invalid number of metadata parts before '+' raises ValueError."""
    station_path_single: Path = Path("51.839051+Rib_at_Wadesmill.parquet")
    with pytest.raises(ValueError, match="metadata parts"):
        parse_custom_station_filename(station_path_single)

    station_path_four: Path = Path("1.0_2.0_3.0_4.0+Rib_at_Wadesmill.parquet")
    with pytest.raises(ValueError, match="metadata parts"):
        parse_custom_station_filename(station_path_four)


def test_parse_custom_station_filename_invalid_numbers() -> None:
    """Test that non-numeric coordinates or upstream area raise ValueError."""
    station_path_coords: Path = Path("invalid_51.839051+Rib_at_Wadesmill.parquet")
    with pytest.raises(ValueError, match="valid numeric coordinates"):
        parse_custom_station_filename(station_path_coords)

    station_path_area: Path = Path(
        "-0.02754_51.839051_invalid+Rib_at_Wadesmill.parquet"
    )
    with pytest.raises(ValueError, match="valid numeric coordinates or upstream area"):
        parse_custom_station_filename(station_path_area)


def test_process_station_data_timezone_aware() -> None:
    """Test that timezone-aware indices are converted to UTC and made timezone-naive."""
    # Create timezone-aware hourly dataframe in US/Eastern (UTC-5)
    dates: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", periods=5, freq="h", tz="US/Eastern"
    )
    raw_df: pd.DataFrame = pd.DataFrame(
        {"Q": [10.0, 12.0, 14.0, 16.0, 18.0]}, index=dates
    )

    processed_df: pd.DataFrame = process_station_data(
        raw_df, Path("0.0_0.0+test_station.parquet")
    )

    assert isinstance(processed_df.index, pd.DatetimeIndex)
    assert processed_df.index.tz is None
    # 12:00:00 US/Eastern is 17:00:00 UTC
    assert processed_df.index[0] == pd.Timestamp("2020-01-01 17:00:00")
    assert processed_df.index.name == "time"
    assert len(processed_df) == 5


def test_process_station_data_resample_sub_hourly() -> None:
    """Test that sub-hourly discharge data (e.g., 15 min) is resampled to hourly with a 30-minute offset."""
    dates: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:00:00", periods=8, freq="15min"
    )
    # Hour 1: [00:00, 00:15, 00:30, 00:45] -> mean is (10 + 12 + 14 + 16) / 4 = 13.0
    # Hour 2: [01:00, 01:15, 01:30, 01:45] -> mean is (18 + 20 + 22 + 24) / 4 = 21.0
    raw_df: pd.DataFrame = pd.DataFrame(
        {"Q": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]}, index=dates
    )

    processed_df: pd.DataFrame = process_station_data(
        raw_df, Path("0.0_0.0+test_station_15min.parquet")
    )

    assert isinstance(processed_df.index, pd.DatetimeIndex)
    assert processed_df.index.name == "time"
    assert len(processed_df) == 2
    # Check that timestamps have the half-hour offset to align with reporter.py substeps
    assert processed_df.index[0] == pd.Timestamp("2020-01-01 00:30:00")
    assert processed_df.index[1] == pd.Timestamp("2020-01-01 01:30:00")
    assert np.isclose(processed_df["Q"].iloc[0], 13.0)
    assert np.isclose(processed_df["Q"].iloc[1], 21.0)


def test_process_station_data_resample_sub_daily() -> None:
    """Test that sub-daily discharge data (e.g., 6h) is resampled to daily with a 12-hour offset."""
    dates: pd.DatetimeIndex = pd.date_range("2020-01-01 00:00:00", periods=8, freq="6h")
    # Day 1: [00:00, 06:00, 12:00, 18:00] -> mean is (10 + 20 + 30 + 40) / 4 = 25.0
    # Day 2: [00:00, 06:00, 12:00, 18:00] -> mean is (50 + 60 + 70 + 80) / 4 = 65.0
    raw_df: pd.DataFrame = pd.DataFrame(
        {"Q": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]}, index=dates
    )

    processed_df: pd.DataFrame = process_station_data(
        raw_df, Path("0.0_0.0+test_station_6h.parquet")
    )

    assert isinstance(processed_df.index, pd.DatetimeIndex)
    assert processed_df.index.name == "time"
    assert len(processed_df) == 2
    # Check that timestamps are at 12:00:00 (middle of the day)
    assert processed_df.index[0] == pd.Timestamp("2020-01-01 12:00:00")
    assert processed_df.index[1] == pd.Timestamp("2020-01-02 12:00:00")
    assert np.isclose(processed_df["Q"].iloc[0], 25.0)
    assert np.isclose(processed_df["Q"].iloc[1], 65.0)


def test_process_station_data_daily_offset() -> None:
    """Test that daily discharge data starting at 00:00:00 is shifted to 12:00:00."""
    dates: pd.DatetimeIndex = pd.date_range("2020-01-01 00:00:00", periods=3, freq="D")
    raw_df: pd.DataFrame = pd.DataFrame({"Q": [5.0, 10.0, 15.0]}, index=dates)

    processed_df: pd.DataFrame = process_station_data(
        raw_df, Path("0.0_0.0+test_station_daily.parquet")
    )

    assert isinstance(processed_df.index, pd.DatetimeIndex)
    assert processed_df.index.name == "time"
    assert len(processed_df) == 3
    assert processed_df.index[0] == pd.Timestamp("2020-01-01 12:00:00")
    assert processed_df.index[1] == pd.Timestamp("2020-01-02 12:00:00")
    assert processed_df.index[2] == pd.Timestamp("2020-01-03 12:00:00")
    assert np.isclose(processed_df["Q"].iloc[0], 5.0)


def test_load_stations_from_zip(tmp_path: Path) -> None:
    """Test loading station data from a zip archive containing CSV and Parquet files."""
    zip_path: Path = tmp_path / "stations.zip"
    df_csv: pd.DataFrame = pd.DataFrame(
        {"Q": [1.0, 2.0]},
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )
    df_parquet: pd.DataFrame = pd.DataFrame(
        {"datetime": pd.date_range("2020-01-01", periods=2, freq="D"), "Q": [3.0, 4.0]}
    )

    parquet_buffer: io.BytesIO = io.BytesIO()
    df_parquet.to_parquet(parquet_buffer, index=False)
    parquet_bytes: bytes = parquet_buffer.getvalue()

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("station1.csv", df_csv.to_csv())
        zf.writestr("station2.parquet", parquet_bytes)
        zf.writestr("readme.txt", "some info")

    stations: list[tuple[Path, pd.DataFrame]] = _load_stations_from_zip(zip_path)
    assert len(stations) == 2
    filenames: list[str] = [station[0].name for station in stations]
    assert "station1.csv" in filenames
    assert "station2.parquet" in filenames


def test_validate_discharge_observation_timestamps_hourly() -> None:
    """Test validation of hourly discharge observation timestamps."""
    # Valid hourly timestamps (on the half hour)
    valid_dates: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", periods=4, freq="h"
    )
    valid_df: pd.DataFrame = pd.DataFrame(
        {"101": [1.0, 2.0, 3.0, 4.0]}, index=valid_dates
    )
    _validate_discharge_observation_timestamps(valid_df, "hourly")

    # Invalid hourly timestamps (on the hour)
    invalid_dates: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:00:00", periods=4, freq="h"
    )
    invalid_df: pd.DataFrame = pd.DataFrame(
        {"101": [1.0, 2.0, 3.0, 4.0]}, index=invalid_dates
    )
    with pytest.raises(
        ValueError,
        match=r"Hourly discharge observations must be timestamped on the half hour",
    ):
        _validate_discharge_observation_timestamps(invalid_df, "hourly")


def test_validate_discharge_observation_timestamps_daily() -> None:
    """Test validation of daily discharge observation timestamps."""
    # Valid daily timestamps (at 12:00:00)
    valid_dates: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", periods=3, freq="D"
    )
    valid_df: pd.DataFrame = pd.DataFrame({"202": [5.0, 6.0, 7.0]}, index=valid_dates)
    _validate_discharge_observation_timestamps(valid_df, "daily")

    # Invalid daily timestamps (at 00:00:00)
    invalid_dates: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:00:00", periods=3, freq="D"
    )
    invalid_df: pd.DataFrame = pd.DataFrame(
        {"202": [5.0, 6.0, 7.0]}, index=invalid_dates
    )
    with pytest.raises(
        ValueError,
        match=r"Daily discharge observations must be timestamped at the middle of the day",
    ):
        _validate_discharge_observation_timestamps(invalid_df, "daily")
