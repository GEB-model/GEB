"""Tests for the interactive discharge evaluation dashboard."""

import inspect
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


def test_popup_chart_script_pins_axes_and_uses_svg_scatter() -> None:
    """Test that interactive dashboard charts render with explicit axis ranges."""
    folium_map: folium.Map = folium.Map(location=[0.0, 0.0])

    _inject_popup_chart_script(folium_map, station_images={}, station_chart_data={})
    html: str = folium_map.get_root().render()

    assert "type: 'date'" in html
    assert "range: timeRange" in html
    assert "range: returnPeriodRange" in html
    assert "tickmode: 'array'" in html
    assert "tickvals: returnPeriodTicks" in html
    assert "%{y:,.0f} m3/s" in html
    assert "connectgaps: true" in html
    assert ".geb-popup__title{color:#0f172a" in html
    assert ".geb-popup__metrics span{background:#111827" in html
    assert "'scattergl'" not in html


def test_create_discharge_dashboard_is_available_through_evaluate_cli() -> None:
    """Test that dashboard-only creation is exposed as an evaluation method."""
    assert "hydrology.create_discharge_dashboard" in get_available_evaluation_methods()


def test_discharge_dashboards_default_to_static_png_popups() -> None:
    """Test that interactive dashboard charts are opt-in."""
    evaluate_signature = inspect.signature(Hydrology.evaluate_discharge)
    dashboard_signature = inspect.signature(Hydrology.create_discharge_dashboard)

    assert (
        evaluate_signature.parameters["interactive_dashboard_charts"].default is False
    )
    assert (
        dashboard_signature.parameters["interactive_dashboard_charts"].default is False
    )


def test_create_discharge_dashboard_requires_existing_metrics(tmp_path: Path) -> None:
    """Test that dashboard-only creation requires a previous discharge evaluation."""
    evaluator: SimpleNamespace = SimpleNamespace(output_folder_evaluate=tmp_path)
    hydrology = Hydrology(model=SimpleNamespace(), evaluator=evaluator)  # ty:ignore[invalid-argument-type]

    with pytest.raises(FileNotFoundError, match="evaluate_discharge"):
        hydrology.create_discharge_dashboard()
