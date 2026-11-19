"""Tests for the evapotranspiration evaluation against eddy covariance towers."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg

from geb.evaluate.workflows.dashboard import (
    StationChartBundleWriter,
    write_discharge_dashboard,
)
from geb.evaluate.workflows.evapotranspiration_evaluation import (
    SIMULATED_COMPONENTS,
    build_tower_dashboard_data,
    compare_towers,
    kling_gupta_efficiency,
    latent_heat_flux_to_mm_per_day,
)


def _tower_inputs(
    n_days: int = 120,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Two towers; the model transpires 80% of the observed flux at tower A.

    Args:
        n_days: Number of days.

    Returns:
        Tower locations, observation table and simulated components (mm/day).
    """
    dates: pd.DatetimeIndex = pd.date_range("2012-04-01", periods=n_days, freq="D")
    observed_mm: np.ndarray = 2.0 + np.sin(np.arange(n_days) / 10.0)
    flux_W_per_m2: np.ndarray = observed_mm * 2.45e6 / 86400.0
    quality: np.ndarray = np.ones(n_days)
    quality[:10] = 0.5  # poor days are not scored
    observations: pd.DataFrame = pd.DataFrame(
        {
            "station_id": "A",
            "date": dates,
            "latent_heat_flux_W_per_m2": flux_W_per_m2,
            "latent_heat_flux_corrected_W_per_m2": flux_W_per_m2 * 1.25,
            "latent_heat_flux_quality_fraction": quality,
        }
    )
    stations: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"site_name": ["Tower A", "Tower B"], "igbp": ["GRA", "MF"]},
        geometry=[sg.Point(8.0, 50.0), sg.Point(8.5, 50.5)],
        index=pd.Index(["A", "B"], name="station_id"),
        crs="EPSG:4326",
    )
    simulated: dict[str, pd.DataFrame] = {
        name: pd.DataFrame({"A": np.zeros(n_days), "B": np.zeros(n_days)}, index=dates)
        for name in SIMULATED_COMPONENTS
    }
    simulated["transpiration_m"]["A"] = 0.6 * observed_mm
    simulated["interception_evaporation_m"]["A"] = 0.2 * observed_mm
    # Sublimation is negative and adds to evaporation.
    simulated["sublimation_or_deposition_m"]["A"] = -0.1
    simulated["potential_evapotranspiration_m"]["A"] = 3.0
    return stations, observations, simulated


def test_latent_heat_flux_to_mm_per_day() -> None:
    """28.36 W/m2 of latent heat evaporates 1 mm per day."""
    result = latent_heat_flux_to_mm_per_day(pd.Series([2.45e6 / 86400.0]))
    np.testing.assert_allclose(result, [1.0])


def test_kling_gupta_efficiency() -> None:
    """A perfect simulation has KGE 1; too few values give NaN."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert kling_gupta_efficiency(values, values) == pytest.approx(1.0)
    assert np.isnan(kling_gupta_efficiency(values[:2], values[:2]))


def test_compare_towers_scores_good_days_only() -> None:
    """Bias uses only good-quality days and includes sublimation as evaporation."""
    stations, observations, simulated = _tower_inputs()
    comparison = compare_towers(stations, observations, simulated)
    tower = comparison.stations.loc["A"]
    assert tower["n_days"] == 110
    observed_mean: float = float((2.0 + np.sin(np.arange(120) / 10.0))[10:].mean())
    expected_bias: float = (0.8 * observed_mean + 0.1) / observed_mean
    assert tower["bias_ratio"] == pytest.approx(expected_bias, rel=1e-5)
    assert tower["bias_ratio_corrected"] == pytest.approx(
        expected_bias / 1.25, rel=1e-5
    )
    assert tower["correlation_daily"] == pytest.approx(1.0, abs=1e-6)
    assert tower["potential_mm_per_year"] == pytest.approx(3.0 * 365.25)
    # Tower B has no observations and is kept without scores.
    assert comparison.stations.loc["B", "n_days"] == 0
    assert not comparison.daily["A"]["scored"].iloc[:10].any()


def test_build_tower_dashboard_data_and_dashboard(tmp_path: Path) -> None:
    """Towers get monthly series and are written as a layer on the dashboard."""
    stations, observations, simulated = _tower_inputs()
    towers = build_tower_dashboard_data(
        compare_towers(stations, observations, simulated)
    )
    assert [t["id"] for t in towers] == ["A", "B"]
    tower_a = towers[0]
    assert tower_a["igbp_name"] == "grassland"
    assert len(tower_a["monthly"]["time"]) == len(tower_a["monthly"]["observed"]) == 4
    assert "monthly" not in towers[1]

    region = gpd.GeoDataFrame(geometry=[sg.box(7.0, 49.0, 9.0, 51.0)], crs="EPSG:4326")
    rivers = gpd.GeoDataFrame(
        {"uparea_m2": [5000000.0]},
        geometry=[sg.LineString([(7.5, 49.5), (8.5, 50.5)])],
        index=pd.Index(["riv_1"]),
        crs="EPSG:4326",
    )
    scores = gpd.GeoDataFrame(
        {
            "station_name": ["Station Test"],
            "upstream_area_GEB": [1000000.0],
            "discharge_observations_to_GEB_upstream_area_ratio": [1.0],
            "KGE": [0.8],
            "timezone_utc_offset": [1.0],
        },
        geometry=[sg.Point(8.0, 50.0)],
        index=pd.Index(["test_st"]),
        crs="EPSG:4326",
    )
    writer = StationChartBundleWriter()
    writer.add_station(
        "test_st", {"stationName": "Station Test", "metrics": {"KGE": 0.8}}
    )
    station_indices = writer.finish()
    dashboard_file: Path = tmp_path / "dashboard.html"
    write_discharge_dashboard(
        mapped_station_scores=scores,
        output_path=dashboard_file,
        region_geom=region,
        rivers=rivers,
        chart_bundles=writer.bundles,
        station_bundle_indices=station_indices,
        flux_towers=towers,
    )
    html: str = dashboard_file.read_text(encoding="utf-8")
    assert '<script id="geb-tower-data" type="application/json">' in html
    assert "Flux towers, evapotranspiration (2)" in html
    assert "Tower A" in html
