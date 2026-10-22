"""Unit tests for discharge dashboard timeseries generation and main index alignment."""

import base64
import datetime
import gzip
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

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
    _calculate_distribution_summary,
    _decode_varint_deltas,
    _encode_varint_deltas,
    _to_int_deltas,
    build_station_chart_data,
    create_discharge_dashboard,
    determine_main_time_index,
    read_run_completion_time,
    read_run_completion_timestamp_for_filename,
    serialize_main_timeline,
    write_discharge_dashboard,
)
from geb.evaluate.workflows.discharge_metrics import use_native_discharge_scores


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


def test_varint_deltas_roundtrip_and_sentinel() -> None:
    """Test Base64 Varint encoding and decoding with sentinel 0 for missing data."""
    # Test cases include None, zero, small positive/negative deltas, large deltas
    test_values: list[int | None] = [
        None,
        None,
        12500,
        0,
        5,
        -3,
        0,
        -100,
        None,
        200,
        -200,
        32767,
        -32768,
        500000,
        -500000,
    ]
    encoded: str = _encode_varint_deltas(test_values)
    assert isinstance(encoded, str)
    decoded: list[int | None] = _decode_varint_deltas(encoded)
    assert decoded == test_values

    # Verify sentinel 0 is used for missing data
    assert _decode_varint_deltas(_encode_varint_deltas([None])) == [None]
    assert _decode_varint_deltas(_encode_varint_deltas([0])) == [0]
    assert _decode_varint_deltas(_encode_varint_deltas([-1, 1])) == [-1, 1]


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
    assert data["deltas"] == "varint"

    # Delta integer cents: Jan 2 is anchor (1235), Jan 3 is None, Jan 4 is anchor reset (3457)
    assert _decode_varint_deltas(data["observed"]) == [1235, None, 3457]
    # Simulated has no missing values: anchor 1111, then deltas 1111, 1111
    assert _decode_varint_deltas(data["simulated"]) == [1111, 1111, 1111]


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
    assert data["deltas"] == "varint"
    assert _decode_varint_deltas(data["observed"]) == [1056, 1011]
    assert _decode_varint_deltas(data["simulated"]) == [1011, 1011]


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
    assert chart_data["timeseries"]["deltas"] == "varint"
    assert _decode_varint_deltas(chart_data["timeseries"]["observed"]) == [
        1000,
        1000,
        1000,
    ]
    assert _decode_varint_deltas(chart_data["timeseries"]["simulated"]) == [
        1100,
        1000,
        1000,
    ]


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


def test_station_chart_bundle_writer() -> None:
    """Test that StationChartBundleWriter chunks stations into compressed in-memory bundles."""
    writer: StationChartBundleWriter = StationChartBundleWriter(
        max_stations_per_bundle=2,
    )

    data1: dict[str, Any] = {"stationName": "St1", "val": 10}
    data2: dict[str, Any] = {"stationName": "St2", "val": 20}
    data3: dict[str, Any] = {"stationName": "St3", "val": 30}

    writer.add_station("st1", data1)
    writer.add_station("st2", data2)
    # 2 stations should trigger bundle flush
    assert len(writer.station_bundle_indices) == 2
    assert writer.station_bundle_indices["st1"] == 0
    assert writer.station_bundle_indices["st2"] == 0
    assert len(writer.bundles) == 1

    writer.add_station("st3", data3)
    indices: dict[str, int] = writer.finish()

    assert len(indices) == 3
    assert indices["st3"] == 1
    assert len(writer.bundles) == 2

    # Verify bundle 0 content
    b0_data: dict[str, Any] = json.loads(
        gzip.decompress(base64.b64decode(writer.bundles[0])).decode("utf-8")
    )
    assert b0_data == {"st1": data1, "st2": data2}

    # Verify bundle 1 content
    b1_data: dict[str, Any] = json.loads(
        gzip.decompress(base64.b64decode(writer.bundles[1])).decode("utf-8")
    )
    assert b1_data == {"st3": data3}


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
    assert _decode_varint_deltas(data["observed"]) == [1500, 500, 500]
    assert _decode_varint_deltas(data["simulated"]) == [1400, 500, 500]


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

    assert stations[0]["fq"] == "daily"
    assert stations[0]["v"][metric_layers[0][0].get_name()] == 0.85
    assert stations[0]["v"][layer_upstream.get_name()] == 1.0

    assert stations[1]["id"] == "st_2"
    assert stations[1]["ex"] == "Too far"
    assert layer_upstream.get_name() not in stations[1]["m"]  # ratio was NaN


def test_build_station_marker_payload_hourly() -> None:
    """Test compressing station marker payload with hourly metrics."""
    gdf = gpd.GeoDataFrame(
        {
            "station_name": ["Station Hourly"],
            "upstream_area_GEB": [100.0],
            "observation_frequency": ["hourly"],
            "KGE": [0.80],
            "KGE_daily": [0.80],
            "KGE_hourly": [0.72],
        },
        geometry=[sg.Point(8.0, 50.0)],
        index=pd.Index(["st_hourly"]),
    )
    metric_layers = [
        (
            folium.FeatureGroup(name="KGE", show=True),
            cm.LinearColormap(colors=["red", "green"], vmin=0.0, vmax=1.0),
            "KGE",
        )
    ]
    payload = _build_station_marker_payload(
        mapped_station_scores=gdf,
        metric_layers=metric_layers,
        layer_upstream=None,
        colormap_upstream=None,
        characteristic_layers=[],
        availability_layer=None,
        characteristic_records={},
        largest_upstream_area_sqrt=20.0,
    )
    stations = json.loads(gzip.decompress(base64.b64decode(payload)).decode("utf-8"))
    assert len(stations) == 1
    assert stations[0]["fq"] == "hourly"
    assert stations[0]["v"][metric_layers[0][0].get_name()] == 0.72
    assert stations[0]["vh"][metric_layers[0][0].get_name()] == 0.72
    assert stations[0]["vd"][metric_layers[0][0].get_name()] == 0.80
    assert metric_layers[0][0].get_name() in stations[0]["mh"]
    assert metric_layers[0][0].get_name() in stations[0]["md"]


def test_build_station_marker_payload_uses_high_res_snapped_location() -> None:
    """Test that station markers use high-res original_subgrid coordinates when available."""
    gdf = gpd.GeoDataFrame(
        {
            "station_name": ["Station A"],
            "upstream_area_GEB": [100.0],
            "original_subgrid_latitude": [50.12345],
            "original_subgrid_longitude": [8.12345],
        },
        geometry=[sg.Point(8.0, 50.0)],
        index=pd.Index(["st_1"]),
    )
    metric_layers = [
        (
            folium.FeatureGroup(name="KGE", show=True),
            cm.LinearColormap(colors=["red", "green"], vmin=0.0, vmax=1.0),
            "KGE",
        )
    ]
    payload = _build_station_marker_payload(
        mapped_station_scores=gdf,
        metric_layers=metric_layers,
        layer_upstream=None,
        colormap_upstream=None,
        characteristic_layers=[],
        availability_layer=None,
        characteristic_records={},
        largest_upstream_area_sqrt=20.0,
    )
    raw_json = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
    stations = json.loads(raw_json)
    assert stations[0]["coords"] == [50.12345, 8.12345]


def test_write_discharge_dashboard(tmp_path: Path) -> None:
    """Test generating a single self-contained dashboard HTML file with inlined chart bundles."""
    dashboard_file: Path = tmp_path / "discharge_evaluation_map.html"
    region: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[sg.box(7.0, 49.0, 9.0, 51.0)],
        crs="EPSG:4326",
    )
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "uparea_m2": [5000000.0],
        },
        geometry=[sg.LineString([(7.5, 49.5), (8.5, 50.5)])],
        index=pd.Index(["riv_1"]),
        crs="EPSG:4326",
    )
    scores: gpd.GeoDataFrame = gpd.GeoDataFrame(
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

    writer: StationChartBundleWriter = StationChartBundleWriter()
    writer.add_station(
        "test_st", {"stationName": "Station Test", "metrics": {"KGE": 0.88}}
    )
    station_indices: dict[str, int] = writer.finish()

    write_discharge_dashboard(
        mapped_station_scores=scores,
        output_path=dashboard_file,
        region_geom=region,
        rivers=rivers,
        chart_bundles=writer.bundles,
        station_bundle_indices=station_indices,
    )
    assert dashboard_file.exists()
    assert not (tmp_path / "discharge_evaluation_map_charts").exists()
    html_content: str = dashboard_file.read_text(encoding="utf-8")
    assert "DecompressionStream" in html_content
    assert "_gebStations" in html_content
    assert "Station search" in html_content
    assert writer.bundles[0] in html_content
    assert '<script id="geb-chart-data" type="application/json">' in html_content
    assert "stationsPane" in html_content
    assert "geb-metric-dropdown" in html_content
    assert "geb-violin-svg" in html_content
    assert "data-geb-theme" in html_content
    assert "geb-theme-btn" in html_content
    assert "geb-export-violin" in html_content
    assert "geb-export-map-btn" in html_content
    assert "Download plot as SVG" in html_content
    assert "Download table (CSV)" in html_content
    assert "Download GeoJSON" in html_content
    assert "exportStationGeoJson" in html_content
    assert "Rivers" in html_content


def test_build_station_chart_data_with_bankfull_discharge() -> None:
    """Test that build_station_chart_data includes bankfullDischarge in timeseries."""
    import logging

    from geb.evaluate.workflows.dashboard import build_station_chart_data

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "discharge_observations": [10.0, 12.0, 14.0, 11.0, 9.0],
            "discharge_simulations": [11.0, 13.0, 13.5, 10.5, 9.5],
        },
        index=idx,
    )
    logger = logging.getLogger("test")
    chart_data = build_station_chart_data(
        discharge_comparison=df,
        station_name="Test Station",
        upstream_area_ratio=1.0,
        timezone_utc_offset=0.0,
        metrics={"KGE": 0.95},
        frequency="daily",
        logger=logger,
        include_return_period_plots=False,
        bankfull_discharge=25.5,
    )
    assert "timeseries" in chart_data
    assert chart_data["timeseries"]["bankfullDischarge"] == 25.5


def test_write_dashboard_charts_from_saved_scores_with_bankfull(
    tmp_path: Path,
) -> None:
    """Test that _write_dashboard_charts_from_saved_scores picks up bankfull discharge."""
    import logging

    from geb.evaluate.workflows.dashboard import (
        _write_dashboard_charts_from_saved_scores,
    )

    run_output = tmp_path / "run"
    routing_dir = run_output / "report" / "hydrology.routing"
    routing_dir.mkdir(parents=True)

    # Write simulated hourly discharge for station
    hourly_idx = pd.date_range("2020-01-01 00:30:00", periods=48, freq="h")
    sim_df = pd.DataFrame(
        {"discharge_hourly_m3_per_s_101": [10.0] * 48},
        index=hourly_idx,
    )
    sim_df.to_parquet(routing_dir / "discharge_hourly_m3_per_s_101.parquet")

    # Write bankfull discharge table
    yearly_idx = pd.date_range("2020-01-01", periods=2, freq="YS")
    bf_df = pd.DataFrame(
        {"101": [30.0, 35.0]},
        index=yearly_idx,
    )
    bf_df.to_parquet(routing_dir / "bankfull_discharge_yearly_m3_per_s.parquet")

    # Observation table
    obs_file = tmp_path / "obs_daily.parquet"
    daily_idx = pd.date_range("2020-01-01 12:00:00", periods=2, freq="D")
    obs_df = pd.DataFrame(
        {"101": [9.5, 10.5]},
        index=daily_idx,
    )
    obs_df.to_parquet(obs_file)

    scores = gpd.GeoDataFrame(
        {
            "station_name": ["Station 101"],
            "discharge_observations_to_GEB_upstream_area_ratio": [1.1],
            "timezone_utc_offset": [0.0],
            "KGE_daily": [0.9],
        },
        index=pd.Index(["101"]),
        geometry=[sg.Point(8.0, 50.0)],
        crs="EPSG:4326",
    )

    obs_hourly_file = tmp_path / "obs_hourly.parquet"
    pd.DataFrame(index=pd.DatetimeIndex([], name="time")).to_parquet(obs_hourly_file)

    table_files = {
        "discharge/discharge_observations_daily": obs_file,
        "discharge/discharge_observations_hourly": obs_hourly_file,
    }
    logger = logging.getLogger("test")

    chart_files, chart_bundles, _ = _write_dashboard_charts_from_saved_scores(
        table_files=table_files,
        logger=logger,
        mapped_station_scores=scores,
        run_output_folder=run_output,
        correct_discharge_observations=True,
        include_return_period_plots=False,
    )
    assert "101" in chart_files
    bundle_idx: int = chart_files["101"]
    encoded: str = chart_bundles[bundle_idx]
    decoded = json.loads(gzip.decompress(base64.b64decode(encoded)).decode("utf-8"))
    assert decoded["101"]["timeseries"]["bankfullDischarge"] == 38.5
    assert "bankfull" in decoded["101"]["timeseries"]
    assert len(decoded["101"]["timeseries"]["bankfull"]) > 0


def test_build_river_chart_data() -> None:
    """Test that build_river_chart_data correctly structures the river payload."""
    from geb.evaluate.workflows.dashboard import build_river_chart_data

    profile_widths = [20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0]
    river_data: dict[str, Any] = build_river_chart_data(
        river_id="10",
        upstream_area_km2=125.5,
        time=["2020-01-01", "2021-01-01"],
        bankfull_discharge=[30.0, 35.0],
        width=[15.0, 15.0],
        depth=[2.0, 2.2],
        floodplain_side_slope_m_per_m=2.0,
        floodplain_profile_widths=profile_widths,
    )
    assert river_data["type"] == "river"
    assert river_data["riverId"] == "10"
    assert river_data["upstreamAreaKm2"] == 125.5
    assert river_data["timeseries"]["time"] == ["2020-01-01", "2021-01-01"]
    assert river_data["timeseries"]["bankfullDischarge"] == [30.0, 35.0]
    assert river_data["timeseries"]["width"] == [15.0, 15.0]
    assert river_data["timeseries"]["depth"] == [2.0, 2.2]
    assert "crossSection" in river_data
    assert river_data["crossSection"]["width"] == 15.0
    assert river_data["crossSection"]["depth"] == 2.2
    assert river_data["crossSection"]["profileWidths"] == profile_widths
    assert river_data["crossSection"]["floodplainSideSlope"] == 2.0


def test_add_river_charts_to_bundle_writer(tmp_path: Path) -> None:
    """Test that add_river_charts_to_bundle_writer adds river data into bundles."""
    from geb.evaluate.workflows.dashboard import (
        StationChartBundleWriter,
        add_river_charts_to_bundle_writer,
    )

    run_output = tmp_path / "run"
    routing_dir = run_output / "report" / "hydrology.routing"
    routing_dir.mkdir(parents=True)

    yearly_idx = pd.date_range("2020-01-01", periods=2, freq="YS")
    df_q = pd.DataFrame({"20": [40.0, 45.0]}, index=yearly_idx)
    df_d = pd.DataFrame({"20": [2.1, 2.3]}, index=yearly_idx)
    df_w = pd.DataFrame({"20": [25.0, 25.0]}, index=yearly_idx)
    df_fp_s = pd.DataFrame({"20": [3.0, 3.0]}, index=yearly_idx)
    df_q.to_parquet(routing_dir / "bankfull_discharge_rivers_yearly_m3_per_s.parquet")
    df_d.to_parquet(routing_dir / "bankfull_depth_rivers_yearly_m.parquet")
    df_w.to_parquet(routing_dir / "bankfull_width_rivers_yearly_m.parquet")
    df_fp_s.to_parquet(
        routing_dir / "floodplain_side_slope_rivers_yearly_m_per_m.parquet"
    )
    for k in range(10):
        df_k = pd.DataFrame(
            {"20": [50.0 + k * 10.0, 50.0 + k * 10.0]}, index=yearly_idx
        )
        df_k.to_parquet(
            routing_dir / f"floodplain_profile_width_{k}_rivers_yearly_m.parquet"
        )

    rivers_gdf = gpd.GeoDataFrame(
        {"uparea_m2": [50_000_000.0], "geometry": [sg.LineString([(0, 0), (1, 1)])]},
        index=pd.Index([20]),
        crs="EPSG:4326",
    )

    writer = StationChartBundleWriter(max_stations_per_bundle=50)
    add_river_charts_to_bundle_writer(
        chart_writer=writer,
        run_output_folder=run_output,
        rivers=rivers_gdf,
    )
    files = writer.finish()
    assert "river_20" in files
    bundle_idx: int = files["river_20"]
    encoded: str = writer.bundles[bundle_idx]
    decoded = json.loads(gzip.decompress(base64.b64decode(encoded)).decode("utf-8"))
    assert decoded["river_20"]["type"] == "river"
    assert decoded["river_20"]["upstreamAreaKm2"] == 50.0
    assert decoded["river_20"]["timeseries"]["bankfullDischarge"] == [40.0, 45.0]
    assert decoded["river_20"]["timeseries"]["depth"] == [2.1, 2.3]
    assert decoded["river_20"]["timeseries"]["width"] == [25.0, 25.0]
    assert decoded["river_20"]["crossSection"]["width"] == 25.0
    assert decoded["river_20"]["crossSection"]["depth"] == 2.3
    assert len(decoded["river_20"]["crossSection"]["profileWidths"]) == 10
    assert decoded["river_20"]["crossSection"]["profileWidths"][0] == 50.0
    assert decoded["river_20"]["crossSection"]["profileWidths"][9] == 140.0
    assert decoded["river_20"]["crossSection"]["floodplainSideSlope"] == 3.0


def test_metric_layer_colormap_ranges() -> None:
    """Test that KGE, mKGE, NSE, and ratio metric colormaps are properly configured.

    Ensures that efficiency metrics use 7 colors spanning from -0.5 to 1.0 with
    yellow at 0.25 and purple at 1.0, while ratio metrics apply the mirrored
    color scale centered at 1.0.
    """
    from geb.evaluate.workflows.dashboard import (
        _EFFICIENCY_METRIC_COLORS,
        _METRIC_LAYER_CONFIGS,
        _RATIO_METRIC_COLORS,
    )
    from geb.evaluate.workflows.discharge_metrics import (
        SKILL_SCORE_PLOT_CONFIG_BY_COLUMN,
    )

    dashboard_configs_by_col: dict[str, dict] = {
        cfg["col"]: cfg for cfg in _METRIC_LAYER_CONFIGS
    }

    # Verify dashboard configurations for efficiency metrics
    for metric_col in ("KGE", "KGE_modified", "NSE"):
        cfg: dict = dashboard_configs_by_col[metric_col]
        vmin: float = cfg["vmin"]
        vmax: float = cfg["vmax"]
        assert vmin == -0.5
        assert vmax == 1.0
        assert cfg["colors"] == _EFFICIENCY_METRIC_COLORS
        assert len(cfg["colors"]) == 7
        # Index 3 is yellow (#fee08b) at the exact 0.25 midpoint
        midpoint: float = (vmin + vmax) / 2.0
        assert midpoint == 0.25
        assert cfg["colors"][3] == "#fee08b"
        assert cfg["colors"][4] == "#1a9850"  # green
        assert cfg["colors"][5] == "#762a83"  # purple
        assert cfg["colors"][6] == "#ffffff"  # white at maximum (1.0)

    # Verify dashboard configurations for ratio metrics (bias and variability)
    for ratio_col in ("KGE_bias_ratio", "KGE_variability_ratio"):
        ratio_cfg: dict = dashboard_configs_by_col[ratio_col]
        assert ratio_cfg["vmin"] == 0.0
        assert ratio_cfg["vmax"] == 2.0
        assert ratio_cfg["colors"] == _RATIO_METRIC_COLORS
        assert len(ratio_cfg["colors"]) == 5
        # Red (< 1.0 underestimation)
        assert ratio_cfg["colors"][0] == "#d7191c"
        assert ratio_cfg["colors"][1] == "#fdae61"
        # Center index 2 is green (#1a9850) at optimum 1.0
        assert ratio_cfg["colors"][2] == "#1a9850"
        # Blue (> 1.0 overestimation)
        assert ratio_cfg["colors"][3] == "#92c5de"
        assert ratio_cfg["colors"][4] == "#0571b0"

    # Verify static plot configurations for KGE and NSE
    for metric_col in ("KGE", "NSE"):
        plot_cfg: dict[str, object] = SKILL_SCORE_PLOT_CONFIG_BY_COLUMN[metric_col]
        plot_vmin: float = float(cast(float, plot_cfg["vmin"]))
        plot_vmax: float = float(cast(float, plot_cfg["vmax"]))
        assert plot_vmin == -0.5
        assert plot_vmax == 1.0
        plot_midpoint: float = (plot_vmin + plot_vmax) / 2.0
        assert plot_midpoint == 0.25


def test_calculate_distribution_summary() -> None:
    """Test calculating distribution statistics and histogram bins."""
    data: pd.Series = pd.Series([0.2, 0.4, 0.6, 0.8, 1.0, np.nan, np.inf])
    dist: dict[str, Any] = _calculate_distribution_summary(
        data, vmin=0.0, vmax=1.0, num_bins=5
    )
    assert dist["count"] == 5
    assert dist["median"] == 0.6
    assert dist["mean"] == 0.6
    assert dist["min"] == 0.2
    assert dist["max"] == 1.0
    assert len(dist["bins"]) == 6
    assert len(dist["counts"]) == 5
    assert sum(dist["counts"]) == 5
    assert len(dist["density_points"]) == 80

    empty_dist: dict[str, Any] = _calculate_distribution_summary(
        pd.Series([], dtype=float), vmin=0.0, vmax=1.0
    )
    assert empty_dist["count"] == 0
    assert empty_dist["median"] is None
    assert len(empty_dist["density_points"]) == 80


def test_write_discharge_dashboard_with_lakes_and_reservoirs(tmp_path: Path) -> None:
    """Test dashboard rendering with lakes, reservoirs, and excluded station subpanels."""
    dashboard_file: Path = tmp_path / "discharge_evaluation_map_lakes.html"
    region: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[sg.box(7.0, 49.0, 9.0, 51.0)],
        crs="EPSG:4326",
    )
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"uparea_m2": [5000000.0]},
        geometry=[sg.LineString([(7.5, 49.5), (8.5, 50.5)])],
        index=pd.Index(["riv_1"]),
        crs="EPSG:4326",
    )
    scores: gpd.GeoDataFrame = gpd.GeoDataFrame(
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
    waterbodies: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "waterbody_type": [1, 2],
            "waterbody_id": [101, 102],
            "average_area": [1500000.0, 3200000.0],
            "volume_total": [5000000.0, 12000000.0],
        },
        geometry=[
            sg.Polygon([(7.2, 49.2), (7.3, 49.2), (7.3, 49.3), (7.2, 49.3)]),
            sg.Polygon([(8.2, 50.2), (8.3, 50.2), (8.3, 50.3), (8.2, 50.3)]),
        ],
        crs="EPSG:4326",
    )
    excluded_stations: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "station_name": ["Excluded Station 1"],
            "exclusion_reason": ["Missing station observations"],
            "station_longitude": [7.8],
            "station_latitude": [49.8],
            "discharge_observations_to_GEB_upstream_area_ratio": [np.nan],
            "upstream_area_GEB": [np.nan],
        },
        geometry=[sg.Point(7.8, 49.8)],
        index=pd.Index(["ex_1"]),
        crs="EPSG:4326",
    )
    writer: StationChartBundleWriter = StationChartBundleWriter()
    writer.add_station(
        "test_st", {"stationName": "Station Test", "metrics": {"KGE": 0.88}}
    )
    station_indices: dict[str, int] = writer.finish()

    write_discharge_dashboard(
        mapped_station_scores=scores,
        output_path=dashboard_file,
        region_geom=region,
        rivers=rivers,
        waterbodies=waterbodies,
        excluded_stations=excluded_stations,
        chart_bundles=writer.bundles,
        station_bundle_indices=station_indices,
    )
    assert dashboard_file.exists()
    html_content: str = dashboard_file.read_text(encoding="utf-8")
    assert "Lakes" in html_content
    assert "Reservoirs" in html_content
    assert "Rivers" in html_content
    assert "geb-exclusion-subpanel" in html_content
    assert '"name": "Lakes", "layer":' in html_content
    assert '"name": "Reservoirs", "layer":' in html_content
    assert '"active": false' in html_content
    assert '"crossOrigin": true' in html_content


def test_plot_skill_score_maps(tmp_path: Path) -> None:
    """Test fast generation of skill score maps with basemap caching."""
    from geb.evaluate.workflows.discharge_plots import plot_skill_score_maps

    region: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[sg.box(5.0, 50.0, 7.0, 52.0)],
        crs="EPSG:4326",
    )
    scores: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "station_name": ["St_1", "St_2"],
            "KGE": [0.8, 0.7],
            "KGE_correlation": [0.85, 0.75],
            "KGE_bias_ratio": [1.05, 0.95],
            "KGE_variability_ratio": [0.98, 1.02],
            "NSE": [0.75, 0.65],
            "R2": [0.8, 0.7],
            "RRMSE": [0.2, 0.3],
        },
        geometry=[sg.Point(5.5, 50.5), sg.Point(6.5, 51.5)],
        crs="EPSG:4326",
    )
    logger: logging.Logger = logging.getLogger("test_maps")

    plot_skill_score_maps(
        mapped_station_scores=scores,
        region_geom=region,
        output_folder=tmp_path,
        logger=logger,
        dpi=100,
    )

    maps_dir: Path = tmp_path / "skill_score_maps"
    assert maps_dir.exists()
    assert (maps_dir / "skill_score_map_kge_components.svg").exists()
    assert (maps_dir / "skill_score_map_kge.svg").exists()
    assert (maps_dir / "skill_score_map_nse.svg").exists()


def test_use_native_discharge_scores() -> None:
    """Test setting unsuffixed score columns using native observation frequency."""
    df: pd.DataFrame = pd.DataFrame(
        {
            "observation_frequency": ["hourly", "daily", "daily"],
            "KGE_hourly": [0.72, np.nan, 0.50],
            "KGE_daily": [0.85, 0.65, 0.60],
        },
        index=pd.Index(["st_1", "st_2", "st_3"]),
    )
    use_native_discharge_scores(df)
    assert df.loc["st_1", "KGE"] == 0.72
    assert df.loc["st_2", "KGE"] == 0.65
    assert df.loc["st_3", "KGE"] == 0.60


def test_read_run_completion_time(tmp_path: Path) -> None:
    """Test reading run completion timestamp from done.txt modification time."""
    assert read_run_completion_time(None) is None
    assert read_run_completion_timestamp_for_filename(None) is None

    run_dir: Path = tmp_path / "my_run"
    assert read_run_completion_time(run_dir) is None
    assert read_run_completion_timestamp_for_filename(run_dir) is None

    run_dir.mkdir(parents=True, exist_ok=True)
    done_file: Path = run_dir / "done.txt"
    done_file.touch()

    completion_time: str | None = read_run_completion_time(run_dir)
    assert completion_time is not None
    parsed_dt: datetime.datetime = datetime.datetime.strptime(
        completion_time, "%Y-%m-%d %H:%M:%S"
    )
    assert parsed_dt is not None

    completion_ts: str | None = read_run_completion_timestamp_for_filename(run_dir)
    assert completion_ts is not None
    assert re.match(r"^\d{8}_\d{6}$", completion_ts)

    report_dir: Path = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    assert read_run_completion_time(report_dir) is None
    assert read_run_completion_timestamp_for_filename(report_dir) is None


def test_write_discharge_dashboard_with_run_and_completion_date(tmp_path: Path) -> None:
    """Test dashboard metadata inclusion (run name and completion date in legend, themed, with no duplicate title card)."""
    dashboard_file: Path = tmp_path / "discharge_evaluation_map.html"
    run_dir: Path = tmp_path / "scenario_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "done.txt").touch()

    region: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[sg.box(7.0, 49.0, 9.0, 51.0)],
        crs="EPSG:4326",
    )
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "uparea_m2": [5000000.0],
        },
        geometry=[sg.LineString([(7.5, 49.5), (8.5, 50.5)])],
        index=pd.Index(["riv_1"]),
        crs="EPSG:4326",
    )
    scores: gpd.GeoDataFrame = gpd.GeoDataFrame(
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
        chart_bundles=[],
        station_bundle_indices={},
        run_output_folder=run_dir,
        run_name="scenario_a",
    )

    assert dashboard_file.exists()
    html_content: str = dashboard_file.read_text(encoding="utf-8")
    assert (
        "<title>Discharge Evaluation Dashboard · Run: scenario_a · Completed: "
        in html_content
    )
    assert '"runName": "scenario_a"' in html_content
    assert '"completionDate":' in html_content
    # Metadata is shown in only one place on the page (in the unified legend header)
    assert "geb-run-meta" in html_content
    assert "geb-dashboard-title-card" not in html_content
    # Strictly adheres to both dark and light themes
    assert "geb-run-meta" in html_content
    assert ".geb-run-meta{border-top:1px solid #1e293b}" in html_content
    assert ".geb-run-meta{border-top:1px solid #e2e8f0}" in html_content
    assert "data-geb-theme" in html_content
    assert "data-set-theme" in html_content


def test_create_discharge_dashboard_filename_with_date(tmp_path: Path) -> None:
    """Test that create_discharge_dashboard appends the completion timestamp (date, hour, minute, second) to the html filename."""
    eval_folder: Path = tmp_path / "evaluate"
    eval_folder.mkdir(parents=True, exist_ok=True)
    run_dir: Path = tmp_path / "scenario_b"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "done.txt").touch()

    region_box: sg.Polygon = sg.box(7.0, 49.0, 9.0, 51.0)
    scores = gpd.GeoDataFrame(
        columns=[
            "station_name",
            "upstream_area_GEB",
            "discharge_observations_to_GEB_upstream_area_ratio",
            "KGE",
            "geometry",
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    scores.to_parquet(eval_folder / "evaluation_metrics.geoparquet")

    snapped = gpd.GeoDataFrame(
        {"GEB_upstream_area_from_grid": pd.Series(dtype=float)},
        geometry=gpd.GeoSeries(crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    snapped_file: Path = tmp_path / "snapped.geoparquet"
    snapped.to_parquet(snapped_file)

    inventory = gpd.GeoDataFrame(
        {
            "station_name": pd.Series(dtype=str),
            "evaluation_exclusion_reason": pd.Series(dtype=str),
            "discharge_observations_upstream_area_m2": pd.Series(dtype=float),
        },
        geometry=gpd.GeoSeries(crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    inventory.to_parquet(tmp_path / "station_locations.geoparquet")

    mask_file: Path = tmp_path / "mask.geoparquet"
    gpd.GeoDataFrame(geometry=[region_box], crs="EPSG:4326").to_parquet(mask_file)

    rivers_file: Path = tmp_path / "rivers.geoparquet"
    gpd.GeoDataFrame(
        {
            "uparea_m2": [5000000.0],
            "is_downstream_outflow": [False],
            "is_further_downstream_outflow": [False],
        },
        geometry=[sg.LineString([(7.5, 49.5), (8.5, 50.5)])],
        index=pd.Index(["riv_1"]),
        crs="EPSG:4326",
    ).to_parquet(rivers_file)

    wb_file: Path = tmp_path / "waterbodies.geoparquet"
    gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_parquet(wb_file)

    geom_files: dict[str, Path] = {
        "mask": mask_file,
        "routing/rivers": rivers_file,
        "waterbodies/waterbody_data": wb_file,
        "discharge/discharge_snapped_locations": snapped_file,
    }

    res: dict[str, str] = create_discharge_dashboard(
        evaluation_folder=eval_folder,
        run_output_folder=run_dir,
        geometry_files=geom_files,
        table_files={},
        minimum_upstream_area_km2=0.0,
        logger=logging.getLogger("test_dashboard"),
        include_return_period_plots=False,
        run_name="scenario_b",
    )
    dashboard_created_path: Path = Path(res["dashboard"])
    assert dashboard_created_path.exists()
    assert re.search(
        r"discharge_evaluation_map_\d{8}_\d{6}\.html$", dashboard_created_path.name
    )
