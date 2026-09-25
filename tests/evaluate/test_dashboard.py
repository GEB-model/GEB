"""Unit tests for discharge dashboard timeseries generation and master index alignment."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geb.evaluate.workflows.dashboard import (
    _as_finite_float,
    _build_timeseries_data,
    _to_int_deltas,
    build_station_chart_data,
    determine_master_time_index,
)


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


def test_to_int_deltas() -> None:
    """Test delta encoding of integer series with resets after missing values."""
    values = [1000, 1050, 1040, None, None, 2500, 2510]
    deltas = _to_int_deltas(values)
    assert deltas == [1000, 50, -10, None, None, 2500, 10]


def test_build_timeseries_data_with_master_index() -> None:
    """Test timeseries window slicing to master index with delta integer cents scaling."""
    master_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", "2020-01-05 12:00:00", freq="D"
    )
    # Station data only covers Jan 2 to Jan 4, with a missing observation on Jan 3.
    station_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-02 12:00:00", "2020-01-04 12:00:00", freq="D"
    )
    comparison_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [12.3456, np.nan, 34.5678],
            "discharge_simulations": [11.1111, 22.2222, 33.3333],
        },
        index=station_index,
    )

    data = _build_timeseries_data(comparison_df, master_time_index=master_index)

    # Master index mode should omit redundant "time" array and window slice to active period.
    assert "time" not in data
    assert data["start"] == 1  # Jan 2 is index 1 of master_index
    assert data["scale"] == 100
    assert data["deltas"] is True
    assert len(data["observed"]) == 3
    assert len(data["simulated"]) == 3

    # Delta integer cents: Jan 2 is anchor (1235), Jan 3 is None, Jan 4 is anchor reset (3457)
    assert data["observed"] == [1235, None, 3457]
    # Simulated has no missing values: anchor 1111, then deltas 1111, 1111
    assert data["simulated"] == [1111, 1111, 1111]


def test_build_timeseries_data_without_master_index() -> None:
    """Test fallback when no master index is supplied provides integer epoch timestamps."""
    station_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", "2020-01-02 12:00:00", freq="D"
    )
    comparison_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [10.555, 20.666],
            "discharge_simulations": [10.111, 20.222],
        },
        index=station_index,
    )

    data = _build_timeseries_data(comparison_df, master_time_index=None)

    assert "time" in data
    assert data["time"] == [1577880000000, 1577966400000]
    assert data["scale"] == 100
    assert data["deltas"] is True
    assert data["observed"] == [1056, 1011]
    assert data["simulated"] == [1011, 1011]


def test_build_timeseries_data_invalid_index() -> None:
    """Test ValueError raised when dataframe does not use DatetimeIndex."""
    df_no_dt = pd.DataFrame(
        {"discharge_observations": [1.0], "discharge_simulations": [1.0]},
        index=[1],
    )
    with pytest.raises(ValueError, match="DateTimeIndex"):
        _build_timeseries_data(df_no_dt)


def test_determine_master_time_index_daily(tmp_path: Path) -> None:
    """Test master index generation for daily observations."""
    sim_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", "2020-01-10 23:30:00", freq="h"
    )
    _create_mock_routing_output(
        tmp_path,
        station_id=123,
        timestamps=sim_timestamps,
        values=np.ones(len(sim_timestamps)),
    )

    obs_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-03 12:00:00", "2020-01-08 12:00:00", freq="D"
    )

    master_idx = determine_master_time_index(
        observations_index=obs_index,
        simulation_output_folder=tmp_path,
        frequency="daily",
    )

    assert master_idx is not None
    assert len(master_idx) == 6
    assert master_idx[0] == pd.Timestamp("2020-01-03 12:00:00")
    assert master_idx[-1] == pd.Timestamp("2020-01-08 12:00:00")


def test_determine_master_time_index_hourly(tmp_path: Path) -> None:
    """Test master index generation for hourly observations."""
    sim_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", "2020-01-03 23:30:00", freq="h"
    )
    _create_mock_routing_output(
        tmp_path,
        station_id=123,
        timestamps=sim_timestamps,
        values=np.ones(len(sim_timestamps)),
    )

    obs_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:30:00", "2020-01-02 12:30:00", freq="h"
    )

    master_idx = determine_master_time_index(
        observations_index=obs_index,
        simulation_output_folder=tmp_path,
        frequency="hourly",
    )

    assert master_idx is not None
    assert len(master_idx) == 25
    assert master_idx[0] == pd.Timestamp("2020-01-01 12:30:00")
    assert master_idx[-1] == pd.Timestamp("2020-01-02 12:30:00")


def test_determine_master_time_index_edge_cases(tmp_path: Path) -> None:
    """Test edge cases such as empty observations or invalid frequency."""
    empty_index = pd.DatetimeIndex([])
    assert (
        determine_master_time_index(
            observations_index=empty_index,
            simulation_output_folder=tmp_path,
            frequency="daily",
        )
        is None
    )

    obs_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", "2020-01-05 12:00:00", freq="D"
    )
    with pytest.raises(ValueError, match="Unsupported frequency"):
        determine_master_time_index(
            observations_index=obs_index,
            simulation_output_folder=tmp_path,
            frequency="monthly",
        )

    # Missing simulation routing directory returns None
    non_existent = tmp_path / "does_not_exist"
    assert (
        determine_master_time_index(
            observations_index=obs_index,
            simulation_output_folder=non_existent,
            frequency="daily",
        )
        is None
    )


def test_build_station_chart_data_includes_timeseries() -> None:
    """Test that build_station_chart_data returns chart payload with timeseries data."""
    station_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", "2020-01-03 12:00:00", freq="D"
    )
    comparison_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [10.0, 20.0, 30.0],
            "discharge_simulations": [11.0, 21.0, 31.0],
        },
        index=station_index,
    )
    logger: logging.Logger = logging.getLogger("test")

    chart_data = build_station_chart_data(
        discharge_comparison=comparison_df,
        station_name="Test Station",
        upstream_area_ratio=1.05432,
        timezone_utc_offset=1.0,
        metrics={"KGE": 0.854321, "NSE": 0.826789},
        frequency="daily",
        logger=logger,
        include_return_period_plots=False,
        master_time_index=station_index,
    )

    assert chart_data["stationName"] == "Test Station"
    assert chart_data["metrics"]["KGE"] == 0.85
    assert chart_data["metrics"]["NSE"] == 0.83
    assert chart_data["metrics"]["upstreamAreaRatio"] == 1.05
    assert "timeseries" in chart_data
    assert chart_data["timeseries"]["scale"] == 100
    assert chart_data["timeseries"]["deltas"] is True
    assert chart_data["timeseries"]["observed"] == [1000, 1000, 1000]
    assert chart_data["timeseries"]["simulated"] == [1100, 1000, 1000]


def test_as_finite_float() -> None:
    """Test _as_finite_float rounds to 2 decimals and handles non-finite values."""
    assert _as_finite_float(0.854321) == 0.85
    assert _as_finite_float(0.856789) == 0.86
    assert _as_finite_float(10.0) == 10.0
    assert _as_finite_float(None) is None
    assert _as_finite_float(float("nan")) is None
    assert _as_finite_float(float("inf")) is None
    assert _as_finite_float(float("-inf")) is None
    # Custom decimal precision
    assert _as_finite_float(0.12345, decimals=3) == 0.123
