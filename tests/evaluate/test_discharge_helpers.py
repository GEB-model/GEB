"""Unit tests for discharge comparison and alignment helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geb.evaluate.workflows.discharge_helpers import load_station_discharge_comparison


def _create_mock_routing_output(
    tmp_path: Path,
    station_id: int | str,
    timestamps: pd.DatetimeIndex,
    values: list[float] | np.ndarray,
) -> Path:
    """Helper to write mock simulated discharge parquet file.

    Args:
        tmp_path: Temporary output folder.
        station_id: Station identifier.
        timestamps: Simulated discharge DatetimeIndex.
        values: Simulated discharge values (m3/s).

    Returns:
        Path to the root output folder.
    """
    routing_dir: Path = tmp_path / "report" / "hydrology.routing"
    routing_dir.mkdir(parents=True, exist_ok=True)
    simulated_series: pd.Series = pd.Series(
        values,
        index=timestamps,
        name=f"discharge_hourly_m3_per_s_{station_id}",
    )
    simulated_df: pd.DataFrame = simulated_series.to_frame()
    simulated_df.to_parquet(
        routing_dir / f"discharge_hourly_m3_per_s_{station_id}.parquet"
    )
    return tmp_path


def test_load_station_discharge_comparison_hourly(tmp_path: Path) -> None:
    """Test hourly comparison retains half-hour timestamps without resampling."""
    station_id: int = 1001
    timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", "2020-01-02 23:30:00", freq="h"
    )
    values: np.ndarray = np.ones(len(timestamps), dtype=float) * 5.0
    _create_mock_routing_output(tmp_path, station_id, timestamps, values)

    observed_series: pd.Series = pd.Series(
        values * 1.2,
        index=timestamps,
    ).asfreq("h")

    comparison: pd.DataFrame = load_station_discharge_comparison(
        output_folder=tmp_path,
        station_id=station_id,
        observed_discharge=observed_series,
        apply_upstream_area_correction=False,
        upstream_area_ratio=1.0,
        timezone_utc_offset=0.0,
    )

    assert comparison.shape[0] == len(timestamps)
    assert comparison.dropna().shape[0] == len(timestamps)
    assert comparison.index[0] == pd.Timestamp("2020-01-01 00:30:00")
    assert comparison["discharge_simulations"].iloc[0] == pytest.approx(5.0)
    assert comparison["discharge_observations"].iloc[0] == pytest.approx(6.0)


def test_load_station_discharge_comparison_daily_resampling(tmp_path: Path) -> None:
    """Test hourly simulation is resampled and shifted to midday to align with daily observation."""
    station_id: int = 1002
    hourly_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", "2020-01-03 23:30:00", freq="h"
    )
    hourly_values: np.ndarray = np.full(len(hourly_timestamps), 10.0, dtype=float)
    _create_mock_routing_output(tmp_path, station_id, hourly_timestamps, hourly_values)

    daily_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", "2020-01-03 12:00:00", freq="D"
    )
    observed_series: pd.Series = pd.Series(
        [9.0, 11.0, 10.5],
        index=daily_timestamps,
    ).asfreq("D")

    comparison: pd.DataFrame = load_station_discharge_comparison(
        output_folder=tmp_path,
        station_id=station_id,
        observed_discharge=observed_series,
        apply_upstream_area_correction=False,
        upstream_area_ratio=1.0,
        timezone_utc_offset=0.0,
    )

    assert comparison.shape[0] == 3
    assert comparison.dropna().shape[0] == 3
    assert comparison.index[0] == pd.Timestamp("2020-01-01 12:00:00")
    assert comparison["discharge_simulations"].iloc[0] == pytest.approx(10.0)


def test_load_station_discharge_comparison_invalid_hourly_obs_timestamp(
    tmp_path: Path,
) -> None:
    """Test ValueError is raised when hourly observation is not timestamped at HH:30:00."""
    station_id: int = 1003
    sim_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", "2020-01-01 10:30:00", freq="h"
    )
    _create_mock_routing_output(
        tmp_path, station_id, sim_timestamps, np.ones(len(sim_timestamps))
    )

    # Invalid hourly timestamps on the hour
    invalid_obs_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:00:00", "2020-01-01 10:00:00", freq="h"
    )
    observed_series: pd.Series = pd.Series(
        np.ones(len(invalid_obs_timestamps)), index=invalid_obs_timestamps
    ).asfreq("h")

    with pytest.raises(
        ValueError,
        match=r"Hourly observed discharge for station .* must be timestamped on the half hour",
    ):
        load_station_discharge_comparison(
            output_folder=tmp_path,
            station_id=station_id,
            observed_discharge=observed_series,
            apply_upstream_area_correction=False,
            upstream_area_ratio=1.0,
            timezone_utc_offset=0.0,
        )


def test_load_station_discharge_comparison_invalid_daily_obs_timestamp(
    tmp_path: Path,
) -> None:
    """Test ValueError is raised when daily observation is not timestamped at 12:00:00."""
    station_id: int = 1004
    sim_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", "2020-01-03 23:30:00", freq="h"
    )
    _create_mock_routing_output(
        tmp_path, station_id, sim_timestamps, np.ones(len(sim_timestamps))
    )

    # Invalid daily timestamps at midnight
    invalid_daily_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:00:00", "2020-01-03 00:00:00", freq="D"
    )
    observed_series: pd.Series = pd.Series(
        [1.0, 2.0, 3.0], index=invalid_daily_timestamps
    ).asfreq("D")

    with pytest.raises(
        ValueError,
        match=r"Daily observed discharge for station .* must be timestamped in the middle of the day",
    ):
        load_station_discharge_comparison(
            output_folder=tmp_path,
            station_id=station_id,
            observed_discharge=observed_series,
            apply_upstream_area_correction=False,
            upstream_area_ratio=1.0,
            timezone_utc_offset=0.0,
        )


def test_load_station_discharge_comparison_invalid_sim_timestamp(
    tmp_path: Path,
) -> None:
    """Test ValueError is raised when hourly simulation is not timestamped at HH:30:00."""
    station_id: int = 1005
    # Invalid hourly timestamps on the hour for simulation
    invalid_sim_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:00:00", "2020-01-01 10:00:00", freq="h"
    )
    _create_mock_routing_output(
        tmp_path,
        station_id,
        invalid_sim_timestamps,
        np.ones(len(invalid_sim_timestamps)),
    )

    valid_obs_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", "2020-01-01 10:30:00", freq="h"
    )
    observed_series: pd.Series = pd.Series(
        np.ones(len(valid_obs_timestamps)), index=valid_obs_timestamps
    ).asfreq("h")

    with pytest.raises(
        ValueError,
        match=r"Hourly simulated discharge for station .* must be timestamped on the half hour",
    ):
        load_station_discharge_comparison(
            output_folder=tmp_path,
            station_id=station_id,
            observed_discharge=observed_series,
            apply_upstream_area_correction=False,
            upstream_area_ratio=1.0,
            timezone_utc_offset=0.0,
        )
