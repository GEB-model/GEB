"""Unit tests for discharge dashboard timeseries generation and main index alignment."""

import base64
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg

from geb.evaluate.workflows.dashboard import (
    StationChartBundleWriter,
    _as_finite_float,
    _build_station_marker_payload,
    _build_timeseries_data,
    _to_int_deltas,
    build_station_chart_data,
    determine_main_time_index,
    serialize_main_timeline,
    write_discharge_dashboard,
    write_station_chart_data,
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


def test_build_timeseries_data_with_main_index() -> None:
    """Test timeseries window slicing to main index with delta integer cents scaling."""
    main_index: pd.DatetimeIndex = pd.date_range(
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

    data = _build_timeseries_data(comparison_df, main_time_index=main_index)

    # main index mode should omit redundant "time" array and window slice to active period.
    assert "time" not in data
    assert data["start"] == 1  # Jan 2 is index 1 of main_index
    assert data["scale"] == 100
    assert data["deltas"] is True
    assert len(data["observed"]) == 3
    assert len(data["simulated"]) == 3

    # Delta integer cents: Jan 2 is anchor (1235), Jan 3 is None, Jan 4 is anchor reset (3457)
    assert data["observed"] == [1235, None, 3457]
    # Simulated has no missing values: anchor 1111, then deltas 1111, 1111
    assert data["simulated"] == [1111, 1111, 1111]


def test_build_timeseries_data_without_main_index() -> None:
    """Test fallback when no main index is supplied provides integer epoch timestamps."""
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

    data = _build_timeseries_data(comparison_df, main_time_index=None)

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


def test_determine_main_time_index_daily(tmp_path: Path) -> None:
    """Test main index generation for daily observations."""
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

    main_idx = determine_main_time_index(
        observations_index=obs_index,
        simulation_output_folder=tmp_path,
        frequency="daily",
    )

    assert main_idx is not None
    assert len(main_idx) == 6
    assert main_idx[0] == pd.Timestamp("2020-01-03 12:00:00")
    assert main_idx[-1] == pd.Timestamp("2020-01-08 12:00:00")


def test_determine_main_time_index_hourly(tmp_path: Path) -> None:
    """Test main index generation for hourly observations."""
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

    main_idx = determine_main_time_index(
        observations_index=obs_index,
        simulation_output_folder=tmp_path,
        frequency="hourly",
    )

    assert main_idx is not None
    assert len(main_idx) == 25
    assert main_idx[0] == pd.Timestamp("2020-01-01 12:30:00")
    assert main_idx[-1] == pd.Timestamp("2020-01-02 12:30:00")


def test_determine_main_time_index_edge_cases(tmp_path: Path) -> None:
    """Test edge cases such as empty observations or invalid frequency."""
    empty_index = pd.DatetimeIndex([])
    assert (
        determine_main_time_index(
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
        determine_main_time_index(
            observations_index=obs_index,
            simulation_output_folder=tmp_path,
            frequency="monthly",
        )

    # Missing simulation routing directory returns None
    non_existent = tmp_path / "does_not_exist"
    assert (
        determine_main_time_index(
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
        main_time_index=station_index,
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


def test_write_station_chart_data_gzip_base64(tmp_path: Path) -> None:
    """Test that write_station_chart_data writes compressed Base64 Gzip JS payload."""
    dashboard_file: Path = tmp_path / "eval_map.html"
    station_payload: dict[str, Any] = {
        "stationName": "Test Gzip Station",
        "frequency": "daily",
        "metrics": {"KGE": 0.91, "NSE": 0.88},
        "timeseries": {
            "start": 0,
            "scale": 100,
            "deltas": True,
            "observed": [500, 10, -5],
            "simulated": [510, 8, -3],
        },
    }

    rel_path: str = write_station_chart_data(
        dashboard_path=dashboard_file,
        station_id="station_abc_123",
        chart_data=station_payload,
    )

    chart_file: Path = tmp_path / rel_path
    assert chart_file.exists()
    assert rel_path.startswith("eval_map_charts/")
    assert rel_path.endswith(".js")

    content: str = chart_file.read_text(encoding="utf-8")
    prefix: str = 'window._gebStationChartPayload="'
    suffix: str = '";'
    assert content.startswith(prefix)
    assert content.endswith(suffix)

    b64_payload: str = content[len(prefix) : -len(suffix)]
    compressed_bytes: bytes = base64.b64decode(b64_payload)
    decompressed_json: bytes = gzip.decompress(compressed_bytes)
    recovered_data: dict[str, Any] = json.loads(decompressed_json.decode("utf-8"))

    assert recovered_data == station_payload


def test_station_chart_bundle_writer(tmp_path: Path) -> None:
    """Test that StationChartBundleWriter chunks stations into bundle files."""
    dashboard_file: Path = tmp_path / "eval_map.html"
    writer: StationChartBundleWriter = StationChartBundleWriter(
        dashboard_path=dashboard_file,
        max_stations_per_bundle=2,
    )

    data1: dict[str, Any] = {"stationName": "St1", "val": 10}
    data2: dict[str, Any] = {"stationName": "St2", "val": 20}
    data3: dict[str, Any] = {"stationName": "St3", "val": 30}

    writer.add_station("st1", data1)
    writer.add_station("st2", data2)
    # 2 stations should trigger bundle_000.js flush
    assert len(writer.station_chart_files) == 2
    assert writer.station_chart_files["st1"] == "eval_map_charts/bundle_000.js"
    assert writer.station_chart_files["st2"] == "eval_map_charts/bundle_000.js"

    writer.add_station("st3", data3)
    files: dict[str, str] = writer.finish()

    assert len(files) == 3
    assert files["st3"] == "eval_map_charts/bundle_001.js"

    # Verify bundle_000.js content
    bundle0_path: Path = tmp_path / "eval_map_charts" / "bundle_000.js"
    assert bundle0_path.exists()
    content0: str = bundle0_path.read_text(encoding="utf-8")
    prefix: str = 'window._gebStationChartBundle="'
    suffix: str = '";'
    assert content0.startswith(prefix) and content0.endswith(suffix)
    b0_data: dict[str, Any] = json.loads(
        gzip.decompress(base64.b64decode(content0[len(prefix) : -len(suffix)])).decode(
            "utf-8"
        )
    )
    assert b0_data == {"st1": data1, "st2": data2}

    # Verify bundle_001.js content
    bundle1_path: Path = tmp_path / "eval_map_charts" / "bundle_001.js"
    assert bundle1_path.exists()
    content1: str = bundle1_path.read_text(encoding="utf-8")
    b1_data: dict[str, Any] = json.loads(
        gzip.decompress(base64.b64decode(content1[len(prefix) : -len(suffix)])).decode(
            "utf-8"
        )
    )
    assert b1_data == {"st3": data3}

    # Test that reinitializing cleans up stale bundle files
    writer_reinit: StationChartBundleWriter = StationChartBundleWriter(
        dashboard_path=dashboard_file,
        max_stations_per_bundle=2,
    )
    assert not bundle0_path.exists()
    assert not bundle1_path.exists()


def test_serialize_main_timeline() -> None:
    """Test serializing regular DatetimeIndex to compact start/step/count dictionary."""
    daily_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", "2020-01-05 12:00:00", freq="D"
    )
    spec: dict[str, int] = serialize_main_timeline(daily_index)
    assert spec["start"] == int(
        daily_index[0].to_datetime64().astype("datetime64[ms]").astype("int64")
    )
    assert spec["step"] == 86400000  # 1 day in ms
    assert spec["count"] == 5

    # Hourly index
    hourly_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 00:30:00", "2020-01-01 05:30:00", freq="h"
    )
    h_spec: dict[str, int] = serialize_main_timeline(hourly_index)
    assert h_spec["step"] == 3600000  # 1 hour in ms
    assert h_spec["count"] == 6

    # Empty index raises ValueError
    with pytest.raises(ValueError, match="cannot be empty"):
        serialize_main_timeline(pd.DatetimeIndex([]))


def test_build_timeseries_data_observation_window_pruning() -> None:
    """Test that leading and trailing missing observation periods are pruned."""
    main_index: pd.DatetimeIndex = pd.date_range(
        "2020-01-01 12:00:00", "2020-01-10 12:00:00", freq="D"
    )
    # Station comparison spans Jan 1 to Jan 10, but observations only exist Jan 3 to Jan 5
    station_index: pd.DatetimeIndex = main_index.copy()
    obs_vals = [
        np.nan,
        np.nan,
        15.0,
        20.0,
        25.0,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    sim_vals = [10.0, 11.0, 14.0, 19.0, 24.0, 28.0, 30.0, 32.0, 35.0, 38.0]
    comparison_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": obs_vals,
            "discharge_simulations": sim_vals,
        },
        index=station_index,
    )

    data = _build_timeseries_data(comparison_df, main_time_index=main_index)

    # Pruned window should start at Jan 3 (index 2 of main_index) and end at Jan 5 (length 3)
    assert data["start"] == 2
    assert len(data["observed"]) == 3
    assert len(data["simulated"]) == 3
    assert data["observed"] == [1500, 500, 500]
    assert data["simulated"] == [1400, 500, 500]


def test_build_station_marker_payload() -> None:
    """Test compressing station marker payload and recovering station attributes."""
    gdf = gpd.GeoDataFrame(
        {
            "station_name": ["Station A", "Station B"],
            "upstream_area_GEB": [100.0, 400.0],
            "discharge_observations_to_GEB_upstream_area_ratio": [1.0, np.nan],
            "KGE": [0.85, -0.2],
            "timezone_utc_offset": [1.0, 2.0],
            "exclusion_reason": [None, "Too far"],
        },
        geometry=[sg.Point(8.0, 50.0), sg.Point(8.5, 50.5)],
        index=pd.Index(["st_1", "st_2"]),
    )
    metric_layers = [
        (
            folium.FeatureGroup(name="KGE", show=True),
            cm.LinearColormap(colors=["red", "green"], vmin=0.0, vmax=1.0),
            "KGE",
        )
    ]
    layer_upstream = folium.FeatureGroup(name="Upstream", show=False)
    colormap_upstream = cm.LinearColormap(colors=["blue", "yellow"], vmin=0.5, vmax=1.5)

    payload = _build_station_marker_payload(
        mapped_station_scores=gdf,
        metric_layers=metric_layers,
        layer_upstream=layer_upstream,
        colormap_upstream=colormap_upstream,
        characteristic_layers=[],
        availability_layer=None,
        characteristic_records={},
        largest_upstream_area_sqrt=20.0,
    )
    assert isinstance(payload, str)
    raw_json = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
    stations = json.loads(raw_json)
    assert len(stations) == 2
    assert stations[0]["id"] == "st_1"
    assert stations[0]["name"] == "Station A"
    assert stations[0]["coords"] == [50.0, 8.0]
    assert stations[0]["r"] == 7.5  # 5 + sqrt(100)/20 * 5 = 7.5
    assert stations[0]["tz"] == 1.0
    assert stations[0]["ex"] is None
    assert metric_layers[0][0].get_name() in stations[0]["m"]
    assert layer_upstream.get_name() in stations[0]["m"]

    assert stations[1]["id"] == "st_2"
    assert stations[1]["ex"] == "Too far"
    assert layer_upstream.get_name() not in stations[1]["m"]  # ratio was NaN


def test_write_discharge_dashboard(tmp_path: Path) -> None:
    """Test generating interactive dashboard HTML with compressed layers."""
    dashboard_file: Path = tmp_path / "discharge_evaluation_map.html"
    region = gpd.GeoDataFrame(
        geometry=[sg.box(7.0, 49.0, 9.0, 51.0)],
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame(
        {
            "uparea_m2": [5000000.0],
        },
        geometry=[sg.LineString([(7.5, 49.5), (8.5, 50.5)])],
        index=pd.Index(["riv_1"]),
        crs="EPSG:4326",
    )
    scores = gpd.GeoDataFrame(
        {
            "station_name": ["Station Test"],
            "upstream_area_GEB": [1000000.0],
            "discharge_observations_to_GEB_upstream_area_ratio": [1.02],
            "KGE": [0.88],
            "timezone_utc_offset": [1.0],
        },
        geometry=[sg.Point(8.0, 50.0)],
        index=pd.Index(["test_st"]),
        crs="EPSG:4326",
    )

    write_discharge_dashboard(
        mapped_station_scores=scores,
        output_path=dashboard_file,
        region_geom=region,
        rivers=rivers,
        station_chart_files={"test_st": "charts/test.js"},
    )
    assert dashboard_file.exists()
    html_content = dashboard_file.read_text(encoding="utf-8")
    assert "DecompressionStream" in html_content
    assert "_gebStations" in html_content
    assert "Station search" in html_content
