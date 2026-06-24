"""Tests for the interactive discharge evaluation dashboard."""

from pathlib import Path
from types import SimpleNamespace

import folium
import pandas as pd
import pytest

from geb.cli import get_available_evaluation_methods
from geb.evaluate.hydrology import Hydrology
from geb.evaluate.workflows.dashboard import (
    _build_timeseries_payload,
    _inject_popup_chart_script,
    write_discharge_dashboard_chart_data,
)


def test_build_timeseries_payload_uses_iso_datetime_strings() -> None:
    """Test that dashboard time-series payloads keep browser-parseable dates."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [1.0, 2.0, None],
            "discharge_simulations": [1.1, 1.9, 2.7],
        },
        index=pd.date_range("2001-01-01", periods=3, freq="D"),
    )

    payload: dict[str, list[str] | list[float | None]] = _build_timeseries_payload(
        validation_df
    )

    assert payload["time"] == [
        "2001-01-01T00:00:00",
        "2001-01-02T00:00:00",
        "2001-01-03T00:00:00",
    ]
    assert payload["observed"] == [1.0, 2.0, None]


def test_build_timeseries_payload_requires_datetime_index() -> None:
    """Test that invalid chart timestamps fail before Plotly receives them."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": [1.0, 2.0],
            "discharge_simulations": [1.1, 1.9],
        },
        index=pd.Index([1400, 2600]),
    )

    with pytest.raises(ValueError, match="DateTimeIndex"):
        _build_timeseries_payload(validation_df)


def test_timeseries_payload_keeps_every_row() -> None:
    """Test that dashboard payloads retain the complete scientific time series."""
    validation_df: pd.DataFrame = pd.DataFrame(
        {
            "discharge_observations": list(range(100)),
            "discharge_simulations": list(range(100, 200)),
        },
        index=pd.date_range("2001-01-01", periods=100, freq="h"),
    )

    payload = _build_timeseries_payload(validation_df)

    assert len(payload["time"]) == len(validation_df)
    assert payload["observed"] == list(range(100))
    assert payload["simulated"] == list(range(100, 200))


def test_popup_chart_script_uses_lazy_files_and_webgl() -> None:
    """Test that exact station series are loaded lazily and rendered with WebGL."""
    folium_map: folium.Map = folium.Map(location=[0.0, 0.0])

    _inject_popup_chart_script(
        folium_map, station_chart_files={"station-1": "charts/station-1.js"}
    )
    html: str = folium_map.get_root().render()

    assert "charts/station-1.js" in html
    assert "loadStationData" in html
    assert "data.frequency === 'hourly' ? 'scattergl' : 'scatter'" in html
    assert "type: 'date'" in html
    assert "range: timeRange" in html
    assert "range: returnPeriodRange" in html
    assert "tickmode: 'array'" in html
    assert "tickvals: returnPeriodTicks" in html
    assert "%{y:,.0f} m3/s" in html
    assert "connectgaps: false" in html
    assert ".geb-popup__title{color:#0f172a" in html
    assert ".geb-popup__metrics span{background:#111827" in html
    assert "timeseriesTraceType" in html


def test_popup_chart_script_reports_interactive_loading_errors() -> None:
    """Test that missing chart data and Plotly failures are explicit."""
    folium_map: folium.Map = folium.Map(location=[0.0, 0.0])

    _inject_popup_chart_script(folium_map, station_chart_files={})
    html: str = folium_map.get_root().render()

    assert "window._stationChartFiles={}" in html
    assert "_charts.js" not in html
    assert "No interactive chart data is available." in html
    assert "Interactive charts require access to cdn.plot.ly." in html
    assert "fallbackHtml" not in html


def test_write_dashboard_chart_data_creates_station_asset(tmp_path: Path) -> None:
    """Test that full station payloads are written beside the dashboard."""
    dashboard_path: Path = tmp_path / "dashboard.html"
    chart_data = {"timeseries": {"time": ["2001-01-01T00:00:00"]}}

    relative_path: str = write_discharge_dashboard_chart_data(
        dashboard_path=dashboard_path,
        station_id="station/1",
        chart_data=chart_data,
    )

    chart_path: Path = tmp_path / relative_path
    assert chart_path.exists()
    assert '"2001-01-01T00:00:00"' in chart_path.read_text()


def test_create_discharge_dashboard_is_available_through_evaluate_cli() -> None:
    """Test that dashboard-only creation is exposed as an evaluation method."""
    assert "hydrology.create_discharge_dashboard" in get_available_evaluation_methods()


def test_create_discharge_dashboard_requires_existing_metrics(tmp_path: Path) -> None:
    """Test that dashboard-only creation requires a previous discharge evaluation."""
    evaluator: SimpleNamespace = SimpleNamespace(output_folder_evaluate=tmp_path)
    hydrology = Hydrology(model=SimpleNamespace(), evaluator=evaluator)

    with pytest.raises(FileNotFoundError, match="evaluate_discharge"):
        hydrology.create_discharge_dashboard()
