"""Tests for create_validation_df in evaluate/hydrology.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geb.evaluate.hydrology import create_validation_df


def test_create_validation_df_hourly(tmp_path: Path) -> None:
    """Test creating validation dataframe with hourly observed and simulated discharge."""
    routing_dir: Path = tmp_path / "report" / "hydrology.routing"
    routing_dir.mkdir(parents=True)

    # 48 hourly steps (2 days) with 30-minute midpoint offset
    sim_times: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", periods=48, freq="h"
    )
    sim_df: pd.DataFrame = pd.DataFrame(
        {"discharge_hourly_m3_per_s_101": np.full(48, 5.0, dtype=np.float32)},
        index=sim_times,
    )
    sim_df.to_parquet(routing_dir / "discharge_hourly_m3_per_s_101.parquet")

    # Observed discharge also hourly with 30-minute offset
    obs_times: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", periods=48, freq="h"
    )
    obs_series: pd.Series = pd.Series(
        np.full(48, 4.0, dtype=np.float32), index=obs_times, name="Q"
    )

    validation_df: pd.DataFrame = create_validation_df(
        output_folder=tmp_path,
        run_name="default",
        station_id=101,
        observed_discharge=obs_series,
        correct_discharge_observations=False,
        discharge_observations_to_GEB_upstream_area_ratio=1.0,
    )

    assert validation_df.dropna().shape[0] == 48
    assert validation_df.index[0] == pd.Timestamp("2020-01-01 00:30:00")
    assert np.isclose(validation_df["discharge_observations"].iloc[0], 4.0)
    assert np.isclose(validation_df["discharge_simulations"].iloc[0], 5.0)


def test_create_validation_df_daily(tmp_path: Path) -> None:
    """Test creating validation dataframe when simulated is hourly and observed is daily (12:00:00)."""
    routing_dir: Path = tmp_path / "report" / "hydrology.routing"
    routing_dir.mkdir(parents=True)

    # 48 hourly steps (2 days) with 30-minute midpoint offset
    # Day 1 hourly values all 10.0 -> daily average = 10.0
    # Day 2 hourly values all 20.0 -> daily average = 20.0
    sim_times: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", periods=48, freq="h"
    )
    sim_values: np.ndarray = np.concatenate(
        [np.full(24, 10.0, dtype=np.float32), np.full(24, 20.0, dtype=np.float32)]
    )
    sim_df: pd.DataFrame = pd.DataFrame(
        {"discharge_hourly_m3_per_s_202": sim_values},
        index=sim_times,
    )
    sim_df.to_parquet(routing_dir / "discharge_hourly_m3_per_s_202.parquet")

    # Observed discharge is daily, timestamped at 12:00:00 (middle of the day)
    obs_times: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", periods=2, freq="D"
    )
    obs_series: pd.Series = pd.Series(
        [10.5, 19.5], index=obs_times, name="Q", dtype=np.float32
    )

    validation_df: pd.DataFrame = create_validation_df(
        output_folder=tmp_path,
        run_name="default",
        station_id=202,
        observed_discharge=obs_series,
        correct_discharge_observations=False,
        discharge_observations_to_GEB_upstream_area_ratio=1.0,
    )

    assert validation_df.dropna().shape[0] == 2
    assert validation_df.index[0] == pd.Timestamp("2020-01-01 12:00:00")
    assert validation_df.index[1] == pd.Timestamp("2020-01-02 12:00:00")
    assert np.isclose(validation_df["discharge_observations"].iloc[0], 10.5)
    assert np.isclose(validation_df["discharge_simulations"].iloc[0], 10.0)
    assert np.isclose(validation_df["discharge_observations"].iloc[1], 19.5)
    assert np.isclose(validation_df["discharge_simulations"].iloc[1], 20.0)


def test_create_validation_df_daily_invalid_timestamp_raises(tmp_path: Path) -> None:
    """Test that daily observations not timestamped at 12:00:00 raise ValueError."""
    routing_dir: Path = tmp_path / "report" / "hydrology.routing"
    routing_dir.mkdir(parents=True)

    sim_times: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", periods=48, freq="h"
    )
    sim_df: pd.DataFrame = pd.DataFrame(
        {"discharge_hourly_m3_per_s_303": np.full(48, 15.0, dtype=np.float32)},
        index=sim_times,
    )
    sim_df.to_parquet(routing_dir / "discharge_hourly_m3_per_s_303.parquet")

    # Invalid observed discharge with 00:00:00 timestamp instead of 12:00:00
    obs_times: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:00:00", periods=2, freq="D"
    )
    obs_series: pd.Series = pd.Series(
        [14.0, 16.0], index=obs_times, name="Q", dtype=np.float32
    )

    with pytest.raises(
        ValueError,
        match=r"Daily observed discharge for station 303 must be timestamped in the middle of the day",
    ):
        create_validation_df(
            output_folder=tmp_path,
            run_name="default",
            station_id=303,
            observed_discharge=obs_series,
            correct_discharge_observations=False,
            discharge_observations_to_GEB_upstream_area_ratio=1.0,
        )


def test_create_validation_df_hourly_invalid_timestamp_raises(tmp_path: Path) -> None:
    """Test that hourly observations not timestamped at HH:30:00 raise ValueError."""
    routing_dir: Path = tmp_path / "report" / "hydrology.routing"
    routing_dir.mkdir(parents=True)

    sim_times: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", periods=48, freq="h"
    )
    sim_df: pd.DataFrame = pd.DataFrame(
        {"discharge_hourly_m3_per_s_404": np.full(48, 15.0, dtype=np.float32)},
        index=sim_times,
    )
    sim_df.to_parquet(routing_dir / "discharge_hourly_m3_per_s_404.parquet")

    # Invalid observed discharge with 00:00:00 timestamp instead of 00:30:00
    obs_times: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:00:00", periods=48, freq="h"
    )
    obs_series: pd.Series = pd.Series(
        np.full(48, 14.0, dtype=np.float32), index=obs_times, name="Q"
    )

    with pytest.raises(
        ValueError,
        match=r"Hourly observed discharge for station 404 must be timestamped on the half hour",
    ):
        create_validation_df(
            output_folder=tmp_path,
            run_name="default",
            station_id=404,
            observed_discharge=obs_series,
            correct_discharge_observations=False,
            discharge_observations_to_GEB_upstream_area_ratio=1.0,
        )
