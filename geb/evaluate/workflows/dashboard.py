"""Functions for creating interactive Folium discharge evaluation maps."""

import hashlib
import html
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium import MacroElement, TileLayer
from jinja2 import Template

from geb.evaluate.workflows.discharge_characteristics import (
    DASHBOARD_CHARACTERISTICS,
    Characteristic,
)
from geb.workflows.extreme_value_analysis import ReturnPeriodModel
from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.model import GEBModel

_ESRI_TOPO_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_TOPO_ATTR = (
    "Sources: Esri, HERE, Garmin, Intermap, INCREMENT P, GEBCO, USGS, FAO, "
    "NPS, NRCan, GeoBase, IGN, Kadaster NL, Ordnance Survey, Esri Japan, "
    "METI, Mapwithyou, NOSTRA, © OpenStreetMap contributors, and the GIS "
    "user community"
)


StationMarkerIndex = dict[str, str | list[str]]
RESERVOIR_WATERBODY_TYPE: int = 2
_WATERBODY_STYLE: dict[int, dict[str, str]] = {
    RESERVOIR_WATERBODY_TYPE: {"color": "#FF8A65", "label": "Reservoir"},
}
_CHARACTERISTIC_COLORS: list[str] = [
    "#440154",
    "#414487",
    "#2A788E",
    "#22A884",
    "#7AD151",
    "#FDE725",
]
_CHARACTERISTIC_MISSING_COLOR: str = "#B8BEC7"
_CHARACTERISTIC_ZERO_COLOR: str = "#EEE8DD"
_CARAVAN_AVAILABLE_COLOR: str = "#1B9E77"
_CARAVAN_UNAVAILABLE_COLOR: str = "#9CA3AF"
_CHARACTERISTIC_COLORMAP: cm.LinearColormap = cm.LinearColormap(
    colors=_CHARACTERISTIC_COLORS,
    vmin=0.0,
    vmax=100.0,
)


class DischargeDashboardGeometries(NamedTuple):
    """Geometries required to build a discharge evaluation dashboard."""

    region: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    waterbodies: gpd.GeoDataFrame


def load_discharge_dashboard_geometries(
    model: GEBModel,
) -> DischargeDashboardGeometries:
    """Load and filter geometries used by the discharge dashboard.

    Args:
        model: GEB model containing the geometry file registry.

    Returns:
        Region boundary, dashboard river network, and waterbody geometries.
    """
    region_geom: gpd.GeoDataFrame = read_geom(model.files["geom"]["mask"])
    all_rivers: gpd.GeoDataFrame = read_geom(model.files["geom"]["routing/rivers"])
    excluded_rivers: pd.Series = (
        all_rivers["is_downstream_outflow"]
        | all_rivers["is_upstream_of_downstream_basin"]
        | all_rivers["is_further_downstream_outflow"]
    )
    waterbodies: gpd.GeoDataFrame = read_geom(
        model.files["geom"]["waterbodies/waterbody_data"]
    )
    return DischargeDashboardGeometries(
        region=region_geom,
        rivers=all_rivers.loc[~excluded_rivers].copy(),
        waterbodies=waterbodies,
    )


_METRIC_LAYER_CONFIGS: list[dict] = [
    {
        "col": "KGE",
        "name": "KGE",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": -1.0,
        "vmax": 1.0,
        "show": True,
    },
    {
        "col": "KGE_modified",
        "name": "mKGE",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": -1.0,
        "vmax": 1.0,
        "show": False,
    },
    {
        "col": "KGE_correlation",
        "name": "KGE correlation",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": 0.0,
        "vmax": 1.0,
        "show": False,
    },
    {
        "col": "KGE_bias_ratio",
        "name": "KGE bias (β)",
        "colors": ["red", "orange", "green", "orange", "red"],
        "vmin": 0.0,
        "vmax": 2.0,
        "show": False,
    },
    {
        "col": "KGE_variability_ratio",
        "name": "KGE variability (α)",
        "colors": ["red", "orange", "green", "orange", "red"],
        "vmin": 0.0,
        "vmax": 2.0,
        "show": False,
    },
    {
        "col": "NSE",
        "name": "NSE",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": -1.0,
        "vmax": 1.0,
        "show": False,
    },
    {
        "col": "R2",
        "name": "Pearson r²",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": 0.0,
        "vmax": 1.0,
        "show": False,
    },
    {
        "col": "RRMSE",
        "name": "RRMSE",
        "colors": ["green", "blue", "yellow", "orange", "red"],
        "vmin": 0.0,
        "vmax": 1.0,
        "show": False,
    },
]


class _JavascriptMacro(MacroElement):
    """Small Folium macro wrapper for dashboard JavaScript.

    Args:
        script: JavaScript inserted in Folium's script block.
    """

    def __init__(self, script: str) -> None:
        """Create a Folium macro from a script string.

        Args:
            script: JavaScript inserted in Folium's script block.
        """
        super().__init__()
        self._template = Template(
            "{%- macro script(this, kwargs) -%}\n" + script + "\n{%- endmacro -%}"
        )


def _as_finite_float(value: float | int | np.floating | None) -> float | None:
    """Convert a numeric value to a finite JSON-friendly float.

    Args:
        value: Value to convert (dimensionless unless documented by the caller).

    Returns:
        Finite float value, or None for missing, NaN, or infinite values.
    """
    if value is None:
        return None
    float_value: float = float(value)
    return float_value if np.isfinite(float_value) else None


def _characteristic_color_statistics(
    values: pd.Series,
    characteristic: Characteristic,
) -> dict[str, float | int | bool | list[float]]:
    """Calculate distribution summaries for a characteristic map legend.

    Map colours use empirical percentile ranks rather than raw-value intervals,
    so skewed variables retain spatial contrast across their observed range.
    For variables where zero means absence, the reference distribution contains
    positive values only and zero is reported as a separate category.

    Args:
        values: Characteristic values in display units.
        characteristic: Characteristic metadata, including zero handling.

    Returns:
        Reference quantiles and counts in display units.

    Raises:
        ValueError: If fewer than two finite, distinct ranked values are available.
    """
    numeric_values: pd.Series = pd.to_numeric(values, errors="coerce")
    valid_values: pd.Series = numeric_values.loc[np.isfinite(numeric_values)]
    ranked_values: pd.Series = (
        valid_values.loc[valid_values != 0.0]
        if characteristic.zero_is_distinct
        else valid_values
    )
    if len(ranked_values) < 2 or ranked_values.nunique() < 2:
        raise ValueError(
            f"Characteristic {characteristic.column} needs two distinct values."
        )
    reference_values: np.ndarray = np.nanpercentile(
        ranked_values.to_numpy(dtype=float), [0.0, 25.0, 50.0, 75.0, 100.0]
    )
    return {
        "reference_values": reference_values.astype(float).tolist(),
        "available_count": int(len(valid_values)),
        "missing_count": int(len(numeric_values) - len(valid_values)),
        "ranked_count": int(len(ranked_values)),
        "zero_count": int((valid_values == 0.0).sum()),
        "zero_is_distinct": characteristic.zero_is_distinct,
    }


def _characteristic_percentile_ranks(
    values: pd.Series,
    characteristic: Characteristic,
) -> pd.Series:
    """Rank finite characteristic values from zero to 100 percent.

    Ties receive their average empirical rank. When zero denotes absence, zero
    values remain unranked so the positive range uses the complete colour scale.

    Args:
        values: Characteristic values in display units.
        characteristic: Characteristic metadata, including zero handling.

    Returns:
        Percentile ranks (percent), aligned to ``values``; missing and distinct
        zero values are represented by NaN.
    """
    numeric_values: pd.Series = pd.to_numeric(values, errors="coerce")
    finite_values: pd.Series = numeric_values.where(np.isfinite(numeric_values))
    ranked_values: pd.Series = (
        finite_values.where(finite_values != 0.0)
        if characteristic.zero_is_distinct
        else finite_values
    )
    valid_values: pd.Series = ranked_values.dropna()
    percentile_ranks: pd.Series = pd.Series(np.nan, index=values.index, dtype=float)
    if valid_values.empty:
        return percentile_ranks
    if len(valid_values) == 1:
        percentile_ranks.loc[valid_values.index] = 50.0
        return percentile_ranks

    average_ranks: pd.Series = valid_values.rank(method="average")
    percentile_ranks.loc[valid_values.index] = (
        (average_ranks - 1.0) / (len(valid_values) - 1.0) * 100.0
    )
    return percentile_ranks


def _build_characteristic_layer_payload(
    evaluation_gdf: gpd.GeoDataFrame,
    characteristic_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build characteristic-layer metadata aligned to evaluated stations.

    Args:
        evaluation_gdf: Evaluated stations indexed by station identifier.
        characteristic_df: Curated GRDC-Caravan values in display units, with
            a unique ``station_ID`` column.

    Returns:
        Characteristic configuration and compact per-station values.

    Raises:
        ValueError: If station identifiers or usable characteristic data are
            unavailable.
    """
    if "station_ID" not in characteristic_df.columns:
        raise ValueError("Dashboard characteristics have no station_ID column.")
    if "grdc_caravan_matched" not in characteristic_df.columns:
        raise ValueError("Dashboard characteristics have no GRDC-Caravan match status.")
    if characteristic_df["station_ID"].duplicated().any():
        raise ValueError("Dashboard characteristics contain duplicate station IDs.")
    if evaluation_gdf.index.astype(str).duplicated().any():
        raise ValueError("Dashboard evaluation contains duplicate station IDs.")

    characteristic_index: pd.DataFrame = characteristic_df.copy()
    characteristic_index["station_ID"] = characteristic_index["station_ID"].astype(str)
    characteristic_index = characteristic_index.set_index("station_ID")
    evaluation_station_ids: pd.Index = pd.Index(
        evaluation_gdf.index.astype(str), name="station_ID"
    )
    # Restrict distributions to displayed stations so percentile colours and
    # legend counts describe exactly the points visible on the dashboard.
    characteristic_index = characteristic_index.reindex(evaluation_station_ids)
    characteristic_configs: list[dict[str, Any]] = []
    percentile_by_characteristic: dict[str, pd.Series] = {}
    usable_characteristics: list[Characteristic] = []
    for characteristic in DASHBOARD_CHARACTERISTICS:
        if characteristic.column not in characteristic_index.columns:
            continue
        try:
            statistics: dict[str, float | int | bool | list[float]] = (
                _characteristic_color_statistics(
                    characteristic_index[characteristic.column], characteristic
                )
            )
        except ValueError:
            continue
        percentile_by_characteristic[characteristic.column] = (
            _characteristic_percentile_ranks(
                characteristic_index[characteristic.column], characteristic
            )
        )
        characteristic_configs.append(
            {
                "column": characteristic.column,
                "label": characteristic.label,
                **statistics,
            }
        )
        usable_characteristics.append(characteristic)
    if not usable_characteristics:
        raise ValueError("No usable GRDC-Caravan dashboard characteristics found.")

    station_records: list[dict[str, Any]] = []
    for station_id in evaluation_gdf.index:
        station_id_string: str = str(station_id)
        characteristic_row: pd.Series | None = (
            characteristic_index.loc[station_id_string]
            if station_id_string in characteristic_index.index
            else None
        )
        values: dict[str, float | None] = {}
        percentiles: dict[str, float | None] = {}
        for characteristic in usable_characteristics:
            values[characteristic.column] = _as_finite_float(
                characteristic_row[characteristic.column]
                if characteristic_row is not None
                else None
            )
            percentile_series: pd.Series = percentile_by_characteristic[
                characteristic.column
            ]
            percentiles[characteristic.column] = _as_finite_float(
                percentile_series.get(station_id_string)
            )
        station_records.append(
            {
                "id": station_id_string,
                "caravan_available": bool(characteristic_row["grdc_caravan_matched"])
                if characteristic_row is not None
                and pd.notna(characteristic_row["grdc_caravan_matched"])
                else False,
                "values": values,
                "percentiles": percentiles,
            }
        )
    return {
        "characteristics": characteristic_configs,
        "stations": station_records,
    }


def _timestamp_to_isoformat(timestamp: Any) -> str:
    """Convert a dashboard timestamp to an ISO-formatted string.

    Args:
        timestamp: Timestamp-like value from a discharge time-series index.

    Returns:
        ISO-formatted timestamp string.

    Raises:
        ValueError: If ``timestamp`` is missing or cannot be represented as a
            timestamp.
    """
    timestamp_value: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp(timestamp))
    if pd.isna(timestamp_value):
        raise ValueError("Dashboard chart timestamps must not contain missing values.")
    return timestamp_value.isoformat()


def _build_timeseries_payload(
    validation_df: pd.DataFrame,
) -> dict[str, list[str] | list[float | None]]:
    """Build the popup payload for one discharge time-series chart.

    Args:
        validation_df: Observed/simulated discharge dataframe (m3/s).
    Returns:
        Dictionary with ISO timestamps and discharge values (m3/s).

    Raises:
        ValueError: If ``validation_df`` is not indexed by timestamps.
    """
    if not isinstance(validation_df.index, pd.DatetimeIndex):
        raise ValueError("validation_df must use a DateTimeIndex for dashboard charts.")

    return {
        "time": [
            _timestamp_to_isoformat(timestamp)
            for timestamp in pd.DatetimeIndex(validation_df.index)
        ],
        "observed": [
            _as_finite_float(value)
            for value in validation_df["discharge_observations"].to_numpy()
        ],
        "simulated": [
            _as_finite_float(value)
            for value in validation_df["discharge_simulations"].to_numpy()
        ],
    }


def _build_return_period_payload(
    series: pd.Series,
    return_periods_years: list[int | float],
) -> dict[str, list[float | None]]:
    """Build fitted return-period values for one discharge series.

    Args:
        series: Regular discharge time series (m3/s).
        return_periods_years: Return periods to estimate (years).

    Returns:
        Dictionary with return periods (years) and fitted discharge values (m3/s).
        Returns empty lists if the fit fails or the series is too short.
    """
    try:
        model = ReturnPeriodModel(
            series=series,
            return_periods=return_periods_years,
            fixed_shape=0.0,
            selection_strategy="first_significant",
        )
        return {
            "returnPeriod": [
                _as_finite_float(value)
                for value in model.rl_table["T_years"].to_numpy(dtype=float)
            ],
            "discharge": [
                _as_finite_float(value)
                for value in model.rl_table["GPD_POT_RL"].to_numpy(dtype=float)
            ],
        }
    except Exception:
        return {"returnPeriod": [], "discharge": []}


def build_discharge_dashboard_chart_data(
    validation_df: pd.DataFrame,
    station_name: str,
    upstream_area_ratio: float,
    metrics: dict[str, float],
    frequency: str,
) -> dict[str, Any]:
    """Build compact interactive chart data for one discharge dashboard popup.

    Args:
        validation_df: Observed/simulated discharge dataframe (m3/s).
        station_name: Human-readable station name.
        upstream_area_ratio: Observed-to-model upstream-area ratio (dimensionless).
        metrics: Discharge skill metrics such as ``KGE``, ``NSE``, and ``R2``
            (dimensionless).
        frequency: Data frequency label, for example ``"daily"`` or ``"hourly"``.

    Returns:
        Compact chart payload with discharge values (m3/s).
    """
    return_periods_years: list[int | float] = [2, 5, 10, 25, 50, 100]
    simulated_series: pd.Series = validation_df["discharge_simulations"].copy()
    simulated_series[validation_df["discharge_observations"].isna()] = np.nan
    return {
        "stationName": station_name,
        "frequency": frequency,
        "metrics": {
            "KGE": _as_finite_float(metrics.get("KGE")),
            "KGE_modified": _as_finite_float(metrics.get("KGE_modified")),
            "KGE_correlation": _as_finite_float(metrics.get("KGE_correlation")),
            "KGE_bias_ratio": _as_finite_float(metrics.get("KGE_bias_ratio")),
            "KGE_variability_ratio": _as_finite_float(
                metrics.get("KGE_variability_ratio")
            ),
            "NSE": _as_finite_float(metrics.get("NSE")),
            "R2": _as_finite_float(metrics.get("R2")),
            "RMSE": _as_finite_float(metrics.get("RMSE")),
            "RRMSE": _as_finite_float(metrics.get("RRMSE")),
            "upstreamAreaRatio": _as_finite_float(upstream_area_ratio),
        },
        "timeseries": _build_timeseries_payload(validation_df),
        "returnPeriods": {
            "observed": _build_return_period_payload(
                validation_df["discharge_observations"], return_periods_years
            ),
            "simulated": _build_return_period_payload(
                simulated_series, return_periods_years
            ),
        },
    }


def write_discharge_dashboard_chart_data(
    dashboard_path: Path,
    station_id: str,
    chart_data: dict[str, Any],
) -> str:
    """Write one exact station chart payload for lazy browser loading.

    Args:
        dashboard_path: Output path of the dashboard HTML file.
        station_id: Station identifier used to derive a stable asset filename.
        chart_data: Complete interactive chart payload.

    Returns:
        POSIX-style payload path relative to the dashboard HTML.
    """
    chart_folder: Path = dashboard_path.parent / f"{dashboard_path.stem}_charts"
    chart_folder.mkdir(parents=True, exist_ok=True)
    station_hash: str = hashlib.sha256(station_id.encode()).hexdigest()[:16]
    chart_path: Path = chart_folder / f"{station_hash}.js"
    chart_path.write_text(
        "window._gebStationChartPayload="
        + json.dumps(chart_data, separators=(",", ":"))
        + ";",
        encoding="utf-8",
    )
    return chart_path.relative_to(dashboard_path.parent).as_posix()


def _inject_popup_chart_script(
    m: folium.Map,
    station_chart_files: dict[str, str],
) -> None:
    """Add lazy-rendered interactive station plots to dashboard popups.

    Args:
        m: Folium map to inject the macro into.
        station_chart_files: Mapping of station IDs to exact chart payload files
            relative to the dashboard HTML.

    """
    chart_files_json: str = json.dumps(station_chart_files, separators=(",", ":"))
    _JavascriptMacro(
        "window._stationChartFiles=" + chart_files_json + ";\n" + """
(function(){
  var plotlyUrl = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
  var colors = { observed: '#facc15', simulated: '#38bdf8' };
  var stationChartCache = {};
  var layoutBase = {
    autosize: true,
    height: 260,
    margin: {l: 50, r: 18, t: 18, b: 42},
    paper_bgcolor: '#020617',
    plot_bgcolor: '#020617',
    font: {color: '#e2e8f0', size: 11},
    legend: {orientation: 'h', x: 0, y: 1.15},
    xaxis: {gridcolor: '#1f2937', zerolinecolor: '#334155'},
    yaxis: {gridcolor: '#1f2937', zerolinecolor: '#334155', rangemode: 'tozero'}
  };

  function ensurePlotly(callback) {
    if (window.Plotly) { callback(); return; }
    var script = document.createElement('script');
    script.src = plotlyUrl;
    script.onload = callback;
    script.onerror = function() { callback(false); };
    document.head.appendChild(script);
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function(character) {
      return ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[character];
    });
  }

  function formatNumber(value) {
    return Number.isFinite(value) ? value.toFixed(2) : 'n/a';
  }

  function metricHtml(label, value) {
    return '<span><b>' + label + '</b> ' + formatNumber(value) + '</span>';
  }

  function makeChartDiv(id) {
    return '<div id="' + id + '" class="geb-popup__chart"></div>';
  }

  function loadStationData(stationId, callback) {
    if (stationChartCache[stationId]) {
      callback(stationChartCache[stationId]);
      return;
    }
    var chartFile = window._stationChartFiles[stationId];
    if (!chartFile) {
      callback(null);
      return;
    }
    var script = document.createElement('script');
    script.src = chartFile;
    script.onload = function() {
      var data = window._gebStationChartPayload;
      delete window._gebStationChartPayload;
      if (data) stationChartCache[stationId] = data;
      script.remove();
      callback(data || null);
    };
    script.onerror = function() {
      script.remove();
      callback(null);
    };
    document.head.appendChild(script);
  }

  function finiteNumbers(values, minimumValue) {
    return (values || []).map(Number).filter(function(value) {
      return Number.isFinite(value) && (minimumValue === undefined || value >= minimumValue);
    });
  }

  function linearRange(values) {
    var numbers = finiteNumbers(values);
    if (!numbers.length) return undefined;
    var minimum = Math.min.apply(null, numbers);
    var maximum = Math.max.apply(null, numbers);
    if (minimum === maximum) {
      var padding = Math.max(Math.abs(minimum) * 0.05, 1);
      return [minimum - padding, maximum + padding];
    }
    return [minimum, maximum];
  }

  function logRange(values) {
    var numbers = finiteNumbers(values, Number.MIN_VALUE);
    if (!numbers.length) return undefined;
    var minimum = Math.min.apply(null, numbers);
    var maximum = Math.max.apply(null, numbers);
    if (minimum === maximum) {
      return [Math.log10(minimum) - 0.05, Math.log10(maximum) + 0.05];
    }
    return [Math.log10(minimum), Math.log10(maximum)];
  }

  function dateRange(values) {
    var times = (values || []).map(function(value) {
      return new Date(value).getTime();
    }).filter(Number.isFinite);
    if (!times.length) return undefined;
    return [new Date(Math.min.apply(null, times)), new Date(Math.max.apply(null, times))];
  }

  function sortedUniqueNumbers(values) {
    var seen = {};
    return finiteNumbers(values).filter(function(value) {
      var key = String(value);
      if (seen[key]) return false;
      seen[key] = true;
      return true;
    }).sort(function(firstValue, secondValue) {
      return firstValue - secondValue;
    });
  }

  function formatTick(value) {
    return Number.isInteger(value) ? String(value) : value.toPrecision(3);
  }

  function renderCharts(stationId, data) {
    var safeStationId = encodeURIComponent(stationId);
    var common = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d']};
    // SVG is reliable for daily series; WebGL keeps full-resolution hourly
    // series responsive without changing the underlying scientific data.
    var timeseriesTraceType = data.frequency === 'hourly' ? 'scattergl' : 'scatter';
    function trace(name, x, y, kind, mode, hoverTemplate) {
      return {
        x: x,
        y: y,
        name: name,
        type: kind,
        mode: mode,
        connectgaps: false,
        hovertemplate: hoverTemplate,
        line: {color: colors[name.toLowerCase()], width: 1.5},
        marker: {color: colors[name.toLowerCase()], size: 5}
      };
    }
    var timeRange = dateRange(data.timeseries.time);
    var observedReturnPeriodRange = linearRange(data.returnPeriods.observed.returnPeriod);
    var simulatedReturnPeriodRange = linearRange(data.returnPeriods.simulated.returnPeriod);
    var returnPeriodValues = [];
    if (observedReturnPeriodRange) returnPeriodValues = returnPeriodValues.concat(observedReturnPeriodRange);
    if (simulatedReturnPeriodRange) returnPeriodValues = returnPeriodValues.concat(simulatedReturnPeriodRange);
    var returnPeriodRange = logRange(returnPeriodValues);
    var returnPeriodTicks = sortedUniqueNumbers(
      data.returnPeriods.observed.returnPeriod.concat(data.returnPeriods.simulated.returnPeriod)
    );
    Plotly.newPlot('geb-time-' + safeStationId, [
      trace('Observed', data.timeseries.time, data.timeseries.observed, timeseriesTraceType, 'lines', '%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Observed</extra>'),
      trace('Simulated', data.timeseries.time, data.timeseries.simulated, timeseriesTraceType, 'lines', '%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Simulated</extra>')
    ], Object.assign({}, layoutBase, {hovermode: 'x unified', xaxis: Object.assign({}, layoutBase.xaxis, {type: 'date', range: timeRange}), yaxis: Object.assign({}, layoutBase.yaxis, {title: 'Discharge (m3/s)'})}), common);
    Plotly.newPlot('geb-return-' + safeStationId, [
      trace('Observed', data.returnPeriods.observed.returnPeriod, data.returnPeriods.observed.discharge, 'scatter', 'lines+markers', '%{x:g}-year<br>%{y:,.0f} m3/s<extra>Observed</extra>'),
      trace('Simulated', data.returnPeriods.simulated.returnPeriod, data.returnPeriods.simulated.discharge, 'scatter', 'lines+markers', '%{x:g}-year<br>%{y:,.0f} m3/s<extra>Simulated</extra>')
    ], Object.assign({}, layoutBase, {hovermode: 'x unified', xaxis: Object.assign({}, layoutBase.xaxis, {type: 'log', range: returnPeriodRange, tickmode: 'array', tickvals: returnPeriodTicks, ticktext: returnPeriodTicks.map(formatTick), title: 'Return period (years)'}), yaxis: Object.assign({}, layoutBase.yaxis, {title: 'Discharge (m3/s)'})}), common);
  }

  function renderStation(el, stationId) {
    if (el.dataset.rendered === 'true') return;
    el.dataset.rendered = 'true';
    loadStationData(stationId, function(data) {
      if (!data) {
        el.innerHTML = '<div class="geb-popup__error">No interactive chart data is available.</div>';
        return;
      }
    var metrics = data.metrics || {};
    var safeStationId = encodeURIComponent(stationId);
    el.innerHTML = '<div class="geb-popup__title">' + escapeHtml(data.stationName || stationId) + '</div>' +
      '<div class="geb-popup__subtitle">Station ' + escapeHtml(stationId) + ' · ' + escapeHtml(data.frequency || 'discharge') + '</div>' +
      '<div class="geb-popup__metrics">' + metricHtml('KGE', metrics.KGE) + metricHtml('mKGE', metrics.KGE_modified) +
      metricHtml('r', metrics.KGE_correlation) + metricHtml('β', metrics.KGE_bias_ratio) +
      metricHtml('α', metrics.KGE_variability_ratio) + metricHtml('NSE', metrics.NSE) +
      metricHtml('r²', metrics.R2) + metricHtml('RMSE', metrics.RMSE) +
      metricHtml('RRMSE', metrics.RRMSE) + metricHtml('Area ratio', metrics.upstreamAreaRatio) + '</div>' +
      '<div class="geb-popup__chart-title">Return periods</div>' + makeChartDiv('geb-return-' + safeStationId) +
      '<div class="geb-popup__chart-title">Discharge time series</div>' + makeChartDiv('geb-time-' + safeStationId);
    ensurePlotly(function(loaded) {
      if (loaded === false) {
        el.innerHTML = '<div class="geb-popup__error">Interactive charts require access to cdn.plot.ly.</div>';
        return;
      }
      renderCharts(stationId, data);
    });
    });
  }

  var style = document.createElement('style');
  style.textContent = '.geb-popup{width:820px;max-width:86vw;color:#0f172a;font-family:Inter,system-ui,sans-serif}.geb-popup__title{color:#0f172a;font-size:18px;font-weight:750}.geb-popup__subtitle{color:#475569;font-size:12px;margin-bottom:8px}.geb-popup__metrics{display:flex;gap:12px;flex-wrap:wrap;margin:6px 0 10px}.geb-popup__metrics span{background:#111827;border:1px solid #263244;border-radius:6px;color:#e2e8f0;padding:5px 8px}.geb-popup__chart{height:260px;background:#020617;border:1px solid #263244;border-radius:8px;margin-bottom:10px}.geb-popup__chart-title{color:#334155;font-weight:700;font-size:13px;margin:10px 0 4px}.geb-popup__error{color:#b91c1c;padding:18px}.geb-popup img{width:100%;height:auto;display:block}';
  document.head.appendChild(style);

"""
        "{{this._parent.get_name()}}.on('popupopen', function(e) {\n"
        "  var content = e.popup.getContent();\n"
        "  if (!content || !content.querySelector) return;\n"
        "  var el = content.querySelector('[data-station-id]');\n"
        "  if (!el) return;\n"
        "  var sid = el.getAttribute('data-station-id');\n"
        "  renderStation(el, sid);\n"
        "});\n"
        "})();\n"
    ).add_to(m)


def _inject_station_search_script(
    m: folium.Map,
    station_markers: list[StationMarkerIndex],
) -> None:
    """Add a station ID/name search control.

    Args:
        m: Folium map receiving the search control.
        station_markers: Station metadata and Folium marker variable names.
    """
    marker_index_js = json.dumps(station_markers, separators=(",", ":"))
    _JavascriptMacro(
        "var gebStationIndex="
        + marker_index_js
        + ";\n"
        + """
(function(){
  var map = {{this._parent.get_name()}};

  function resolveMarkers(station) {
    if (station._markers) return station._markers;
    station._markers = station.markers.map(function(name) {
      try { return window[name] || eval(name); } catch(error) { return null; }
    }).filter(Boolean);
    return station._markers;
  }

  function setStationVisible(station, visible) {
    resolveMarkers(station).forEach(function(marker) {
      marker.setStyle({
        opacity: visible ? 1 : 0,
        fillOpacity: visible ? 0.9 : 0
      });
      marker.options.interactive = visible;
      if (marker.getElement()) {
        marker.getElement().style.pointerEvents = visible ? '' : 'none';
      }
    });
  }

  function applySearch(query) {
    var normalizedQuery = query.trim().toLowerCase();
    var matches = [];
    gebStationIndex.forEach(function(station) {
      var haystack = (station.id + ' ' + station.name).toLowerCase();
      var visible = !normalizedQuery || haystack.indexOf(normalizedQuery) !== -1;
      if (visible) matches.push(station);
      setStationVisible(station, visible);
    });
    updateStatus(normalizedQuery, matches);
    return matches;
  }

  function updateStatus(query, matches) {
    var text = query ? matches.length + ' matching stations' : gebStationIndex.length + ' stations';
    if (query && matches.length) {
      text += ' · Enter opens first match';
    }
    status.textContent = text;
  }

  function openFirstMatch(matches) {
    if (!matches.length) return;
    var marker = resolveMarkers(matches[0])[0];
    if (!marker) return;
    map.setView(marker.getLatLng(), Math.max(map.getZoom(), 8));
    marker.openPopup();
  }

  var control = L.control({position: 'topright'});
  control.onAdd = function() {
    var root = L.DomUtil.create('div', 'geb-station-search');
    root.innerHTML = '<label for="geb-station-search-input">Station search</label>' +
      '<div class="geb-station-search__row"><input id="geb-station-search-input" type="search" placeholder="ID or name">' +
      '<button type="button" title="Clear station search">Clear</button></div>' +
      '<div class="geb-station-search__status"></div>';
    L.DomEvent.disableClickPropagation(root);
    L.DomEvent.disableScrollPropagation(root);
    return root;
  };
  control.addTo(map);

  var root = document.querySelector('.geb-station-search');
  var input = root.querySelector('input');
  var button = root.querySelector('button');
  var status = root.querySelector('.geb-station-search__status');

  input.addEventListener('input', function() { applySearch(input.value); });
  input.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
      event.preventDefault();
      openFirstMatch(applySearch(input.value));
    }
  });
  button.addEventListener('click', function() {
    input.value = '';
    input.focus();
    applySearch('');
  });

  var style = document.createElement('style');
  style.textContent = '.geb-station-search{background:#020617;color:#e2e8f0;border:1px solid #263244;border-radius:8px;padding:10px;width:250px;box-shadow:0 12px 30px rgba(0,0,0,.35);font-family:Inter,system-ui,sans-serif}.geb-station-search label{display:block;font-size:12px;font-weight:750;margin-bottom:6px}.geb-station-search__row{display:flex;gap:6px}.geb-station-search input{min-width:0;flex:1;background:#111827;color:#f8fafc;border:1px solid #334155;border-radius:6px;padding:6px 8px}.geb-station-search button{background:#1f2937;color:#f8fafc;border:1px solid #475569;border-radius:6px;padding:6px 8px;cursor:pointer}.geb-station-search__status{color:#94a3b8;font-size:11px;margin-top:6px}';
  document.head.appendChild(style);
  updateStatus('', gebStationIndex);
})();
"""
    ).add_to(m)


def _inject_station_layer_legend_script(
    discharge_map: folium.Map,
    metric_layers: list[tuple[folium.FeatureGroup, cm.LinearColormap, str]],
    upstream_layer: folium.FeatureGroup | None,
    characteristic_layers: list[tuple[folium.FeatureGroup, dict[str, Any]]],
    availability_layer: folium.FeatureGroup | None,
    caravan_available_count: int,
    station_count: int,
) -> None:
    """Add one dynamic legend and enforce one active station-value layer.

    Args:
        discharge_map: Performance map receiving the legend control.
        metric_layers: Metric feature groups, colormaps, and source columns.
        upstream_layer: Optional upstream-area-ratio feature group.
        characteristic_layers: GRDC-Caravan feature groups and metadata.
        availability_layer: Optional GRDC-Caravan coverage feature group.
        caravan_available_count: Stations matched to GRDC-Caravan.
        station_count: Total evaluated stations.
    """
    configs_by_column: dict[str, dict[str, Any]] = {
        str(config["col"]): config for config in _METRIC_LAYER_CONFIGS
    }
    legend_configs: list[dict[str, Any]] = []
    for layer, _, column in metric_layers:
        config: dict[str, Any] = configs_by_column[column]
        legend_configs.append(
            {
                "layer": layer.get_name(),
                "kind": "continuous",
                "name": config["name"],
                "colors": config["colors"],
                "minimum": config["vmin"],
                "maximum": config["vmax"],
                "show": config["show"],
            }
        )
    if upstream_layer is not None:
        legend_configs.append(
            {
                "layer": upstream_layer.get_name(),
                "kind": "continuous",
                "name": "Upstream Area Ratio",
                "colors": ["red", "orange", "yellow", "blue", "green"],
                "minimum": 0.5,
                "maximum": 2.0,
                "show": False,
            }
        )
    for layer, characteristic in characteristic_layers:
        legend_configs.append(
            {
                "layer": layer.get_name(),
                "kind": "characteristic",
                "name": characteristic["label"],
                "colors": _CHARACTERISTIC_COLORS,
                "minimum": 0.0,
                "maximum": 100.0,
                "reference_values": characteristic["reference_values"],
                "ranked_count": characteristic["ranked_count"],
                "zero_count": characteristic["zero_count"],
                "missing_count": characteristic["missing_count"],
                "zero_is_distinct": characteristic["zero_is_distinct"],
                "show": False,
            }
        )
    if availability_layer is not None:
        legend_configs.append(
            {
                "layer": availability_layer.get_name(),
                "kind": "availability",
                "name": "GRDC-Caravan data availability",
                "available_count": caravan_available_count,
                "unavailable_count": station_count - caravan_available_count,
                "available_color": _CARAVAN_AVAILABLE_COLOR,
                "unavailable_color": _CARAVAN_UNAVAILABLE_COLOR,
                "show": False,
            }
        )
    config_json: str = json.dumps(legend_configs, separators=(",", ":"))
    _JavascriptMacro(
        "var gebMetricLegendConfigs="
        + config_json
        + ";\n"
        + r"""
(function(){
  var map = {{this._parent.get_name()}};
  var configByLayer = {};
  var layerByName = {};
  var activeConfig = null;
  gebMetricLegendConfigs.forEach(function(config) {
    var layer = null;
    try { layer = window[config.layer] || eval(config.layer); } catch(error) { layer = null; }
    if (!layer) return;
    configByLayer[L.stamp(layer)] = config;
    layerByName[config.layer] = layer;
    if (config.show) activeConfig = config;
  });

  var legendControl = L.control({position: 'bottomleft'});
  legendControl.onAdd = function() {
    var element = L.DomUtil.create('div', 'geb-metric-legend');
    L.DomEvent.disableClickPropagation(element);
    return element;
  };
  legendControl.addTo(map);
  var legendRoot = map.getContainer().querySelector('.geb-metric-legend');

  function formatTick(value) {
    var numericValue = Number(value);
    var absoluteValue = Math.abs(numericValue);
    if (numericValue === 0) return '0';
    if (absoluteValue < 0.01) return numericValue.toExponential(1);
    var digits = absoluteValue >= 1000 ? 0 : absoluteValue >= 100 ? 1 : absoluteValue >= 10 ? 1 : absoluteValue >= 1 ? 2 : 3;
    return numericValue.toLocaleString(undefined, {maximumFractionDigits: digits});
  }

  function renderLegend(config) {
    if (!config) {
      legendRoot.style.display = 'none';
      return;
    }
    legendRoot.style.display = '';
    if (config.kind === 'availability') {
      legendRoot.innerHTML = '<b>' + config.name + '</b>' +
        '<div class="geb-categorical-key"><i style="background:' + config.available_color + '"></i>' +
        'Available (n=' + config.available_count + ')</div>' +
        '<div class="geb-categorical-key"><i style="background:' + config.unavailable_color + '"></i>' +
        'Not available (n=' + config.unavailable_count + ')</div>';
      return;
    }
    var ticks = [];
    for (var index = 0; index < 5; index += 1) {
      ticks.push(config.minimum + (config.maximum - config.minimum) * index / 4);
    }
    var detail = '';
    if (config.kind === 'characteristic') {
      ticks = config.reference_values;
      detail = '<div class="geb-metric-note">Colour shows empirical percentile rank (n=' +
        config.ranked_count + '); ticks are values at ranks 0 / 25 / 50 / 75 / 100.</div>' +
        (config.zero_is_distinct && config.zero_count > 0 ?
          '<div class="geb-categorical-key"><i style="background:#EEE8DD"></i>' +
          'Zero / none (n=' + config.zero_count + ')</div>' : '') +
        '<div class="geb-metric-note">Missing value: n=' + config.missing_count + '</div>';
    }
    legendRoot.innerHTML = '<b>' + config.name + '</b>' +
      '<div class="geb-metric-gradient" style="background:linear-gradient(90deg,' +
      config.colors.join(',') + ')"></div><div class="geb-metric-ticks">' +
      ticks.map(function(value) { return '<span>' + formatTick(value) + '</span>'; }).join('') +
      '</div>' + detail;
  }

  map.on('overlayadd', function(event) {
    var config = configByLayer[L.stamp(event.layer)];
    if (!config) return;
    gebMetricLegendConfigs.forEach(function(otherConfig) {
      if (otherConfig.layer === config.layer) return;
      var otherLayer = layerByName[otherConfig.layer];
      if (otherLayer && map.hasLayer(otherLayer)) map.removeLayer(otherLayer);
    });
    activeConfig = config;
    renderLegend(activeConfig);
  });
  map.on('overlayremove', function(event) {
    var config = configByLayer[L.stamp(event.layer)];
    if (config && activeConfig && config.layer === activeConfig.layer) {
      activeConfig = null;
      renderLegend(null);
    }
  });

  var style = document.createElement('style');
  style.textContent = '.geb-metric-legend{background:rgba(255,255,255,.96);border:1px solid #d8dee8;border-radius:8px;box-shadow:0 6px 20px rgba(15,23,42,.18);font-family:Inter,system-ui,sans-serif;margin-bottom:22px!important;padding:9px 10px;width:240px}.geb-metric-legend>b{color:#111827;display:block;font-size:11px;margin-bottom:6px}.geb-metric-gradient{border-radius:2px;height:9px}.geb-metric-ticks{color:#475569;display:flex;font-size:9px;justify-content:space-between;margin-top:3px}.geb-metric-note{color:#64748b;font-size:9px;line-height:1.3;margin-top:5px}.geb-categorical-key{align-items:center;color:#475569;display:flex;font-size:9px;gap:6px;margin-top:5px}.geb-categorical-key i{border:1px solid #fff;border-radius:50%;height:9px;width:9px}';
  document.head.appendChild(style);
  renderLegend(activeConfig);
})();
"""
    ).add_to(discharge_map)


def _add_station_marker(
    layer: folium.FeatureGroup,
    coords: list[float],
    radius: float,
    fill_color: str,
    popup_html: str,
    popup_width: int,
    tooltip: str,
) -> str:
    """Add a station marker and return its JavaScript variable name.

    Args:
        layer: Folium layer receiving the marker.
        coords: Marker coordinates as ``[latitude, longitude]`` (degrees).
        radius: Marker radius (pixels).
        fill_color: Marker fill color.
        popup_html: Popup placeholder HTML.
        popup_width: Popup width (pixels).
        tooltip: Marker tooltip text.

    Returns:
        Folium JavaScript variable name for the marker.
    """
    marker = folium.CircleMarker(
        location=coords,
        radius=radius,
        color="black",
        fill=True,
        fill_color=fill_color,
        fill_opacity=0.9,
        popup=folium.Popup(popup_html, max_width=popup_width),
        tooltip=tooltip,
        z_index=1000,
    )
    marker.add_to(layer)
    return marker.get_name()


def _add_metric_station_markers(
    row: pd.Series,
    metric_layers: list[tuple[folium.FeatureGroup, cm.LinearColormap, str]],
    coords: list[float],
    circle_radius: float,
    popup_html: str,
    popup_width: int,
    tooltip: str,
) -> list[str]:
    """Add one station marker to each metric layer.

    Args:
        row: Evaluation metrics for one station (dimensionless).
        metric_layers: Layers, colormaps, and metric names to render.
        coords: Marker coordinates as ``[latitude, longitude]`` (degrees).
        circle_radius: Marker radius (pixels).
        popup_html: Popup placeholder HTML.
        popup_width: Popup width (pixels).
        tooltip: Marker tooltip text.

    Returns:
        Folium JavaScript variable names for the created markers.
    """
    marker_names: list[str] = []
    for layer, colormap, metric_name in metric_layers:
        metric_value: float = row[metric_name]
        fill_color: str = colormap(metric_value) if pd.notna(metric_value) else "gray"
        marker_names.append(
            _add_station_marker(
                layer=layer,
                coords=coords,
                radius=circle_radius,
                fill_color=fill_color,
                popup_html=popup_html,
                popup_width=popup_width,
                tooltip=tooltip,
            )
        )
    return marker_names


def _format_characteristic_value(value: float) -> str:
    """Format a catchment-characteristic value for a map tooltip.

    Args:
        value: Characteristic value in the display unit stated by its label.

    Returns:
        Compact locale-independent value string.
    """
    absolute_value: float = abs(value)
    decimal_places: int = (
        0
        if absolute_value >= 1000.0
        else 1
        if absolute_value >= 100.0
        else 2
        if absolute_value >= 10.0
        else 3
    )
    return f"{value:,.{decimal_places}f}"


def _add_characteristic_station_markers(
    station_record: dict[str, Any],
    characteristic_layers: list[tuple[folium.FeatureGroup, dict[str, Any]]],
    availability_layer: folium.FeatureGroup,
    coords: list[float],
    circle_radius: float,
    popup_html: str,
    popup_width: int,
    station_tooltip: str,
) -> list[str]:
    """Add GRDC-Caravan availability and characteristic markers for a station.

    Characteristic layers contain only stations with a finite value. The
    availability layer contains every evaluated station, making absent
    GRDC-Caravan coverage explicit without treating it as a numeric zero.

    Args:
        station_record: JSON-safe station characteristic record.
        characteristic_layers: Feature groups and their characteristic metadata.
        availability_layer: Feature group showing GRDC-Caravan match status.
        coords: Marker coordinates as ``[latitude, longitude]`` (degrees).
        circle_radius: Marker radius (pixels).
        popup_html: Popup placeholder HTML.
        popup_width: Popup width (pixels).
        station_tooltip: Station identifier and name.

    Returns:
        Folium JavaScript variable names for the created markers.
    """
    marker_names: list[str] = []
    caravan_available: bool = bool(station_record["caravan_available"])
    availability_label: str = "available" if caravan_available else "not available"
    marker_names.append(
        _add_station_marker(
            layer=availability_layer,
            coords=coords,
            radius=circle_radius,
            fill_color=(
                _CARAVAN_AVAILABLE_COLOR
                if caravan_available
                else _CARAVAN_UNAVAILABLE_COLOR
            ),
            popup_html=popup_html,
            popup_width=popup_width,
            tooltip=f"{station_tooltip}<br>GRDC-Caravan data: {availability_label}",
        )
    )

    for layer, characteristic in characteristic_layers:
        value: float | None = station_record["values"][characteristic["column"]]
        if value is None:
            continue
        percentile: float | None = station_record["percentiles"][
            characteristic["column"]
        ]
        distinct_zero: bool = bool(characteristic["zero_is_distinct"]) and value == 0
        if distinct_zero:
            fill_color: str = _CHARACTERISTIC_ZERO_COLOR
            rank_text: str = "zero / none"
        elif percentile is not None:
            fill_color = _CHARACTERISTIC_COLORMAP(percentile)
            rank_text = f"percentile rank {percentile:.0f}"
        else:
            fill_color = _CHARACTERISTIC_MISSING_COLOR
            rank_text = "rank unavailable"
        marker_names.append(
            _add_station_marker(
                layer=layer,
                coords=coords,
                radius=circle_radius,
                fill_color=fill_color,
                popup_html=popup_html,
                popup_width=popup_width,
                tooltip=(
                    f"{station_tooltip}<br>{characteristic['label']}: "
                    f"{_format_characteristic_value(value)} ({rank_text})"
                ),
            )
        )
    return marker_names


def _add_waterbody_layers(
    discharge_map: folium.Map,
    waterbodies: gpd.GeoDataFrame,
) -> None:
    """Add reservoir point layers to the discharge map.

    Args:
        discharge_map: Folium map receiving waterbody layers.
        waterbodies: Waterbody GeoDataFrame with polygon geometries and
            ``waterbody_type`` identifiers. Only reservoirs are rendered.
    """
    waterbody_layers: dict[int, folium.FeatureGroup] = {
        waterbody_type: folium.FeatureGroup(name=style["label"] + "s", show=True)
        for waterbody_type, style in _WATERBODY_STYLE.items()
    }
    reservoir_mask: pd.Series = (
        waterbodies["waterbody_type"].astype(int) == RESERVOIR_WATERBODY_TYPE
    )
    reservoirs: gpd.GeoDataFrame = waterbodies.loc[reservoir_mask].copy()
    if reservoirs.empty:
        return

    waterbodies_wgs84: gpd.GeoDataFrame = reservoirs.to_crs(epsg=4326)
    for _, waterbody_row in waterbodies_wgs84.iterrows():
        waterbody_style: dict[str, str] = _WATERBODY_STYLE[RESERVOIR_WATERBODY_TYPE]

        centroid = waterbody_row.geometry.centroid
        area_km2: float | None = (
            float(waterbody_row["average_area"]) / 1e6
            if "average_area" in waterbody_row.index
            else None
        )
        volume_km3: float | None = (
            float(waterbody_row["volume_total"]) / 1e9
            if "volume_total" in waterbody_row.index
            else None
        )
        popup_lines: list[str] = [
            f"<b>{waterbody_style['label']}</b> "
            f"(ID {waterbody_row.get('waterbody_id', '?')})<br>"
        ]
        if area_km2 is not None:
            popup_lines.append(f"Area: {area_km2:.1f} km²<br>")
        if volume_km3 is not None:
            popup_lines.append(f"Volume: {volume_km3:.3f} km³<br>")

        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=5,
            color="black",
            weight=0.5,
            fill=True,
            fill_color=waterbody_style["color"],
            fill_opacity=0.8,
            popup=folium.Popup("".join(popup_lines), max_width=200),
            tooltip=f"{waterbody_style['label']} {waterbody_row.get('waterbody_id', '')}",
            z_index=500,
        ).add_to(waterbody_layers[RESERVOIR_WATERBODY_TYPE])

    for waterbody_layer in waterbody_layers.values():
        waterbody_layer.add_to(discharge_map)


def create_discharge_folium_map(
    evaluation_gdf: gpd.GeoDataFrame,
    output_path: Path,
    region_geom: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    station_chart_files: dict[str, str],
    waterbodies: gpd.GeoDataFrame | None = None,
    characteristic_df: pd.DataFrame | None = None,
    minimum_river_upstream_area_km2: float = 5000.0,
) -> folium.Map:
    """Create an interactive Folium discharge evaluation map.

    Stations are shown as circle markers coloured by each discharge metric
    (switchable via layer control) and sized by upstream area.  An optional
    upstream-area-ratio layer is included when all stations have the ratio
    available.  Station popup charts are lazy-rendered with Plotly when the
    popup is opened. Reservoirs are rendered as dot markers when ``waterbodies``
    is provided; lakes are skipped because they make large dashboards slow.
    GRDC-Caravan characteristics are optional station layers on the same map,
    together with a separate data-availability layer.

    Args:
        evaluation_gdf: Per-station GeoDataFrame with discharge metric columns,
            ``upstream_area_GEB``,
            ``discharge_observations_to_GEB_upstream_area_ratio``, and a
            point geometry.
        output_path: Full path (including filename) where the HTML file is
            saved.
        region_geom: Basin/region boundary GeoDataFrame used to fit the map
            extent and render the catchment outline.
        rivers: River network GeoDataFrame (geometry only; rivers are rendered
            at uniform width).
        station_chart_files: Exact interactive chart payload files keyed by
            station ID string.
        waterbodies: Optional GeoDataFrame with columns ``waterbody_type``
            (2 = reservoir) and polygon geometries. Centroids are used for dot
            placement.
        characteristic_df: Optional station table containing ``station_ID`` and
            the curated GRDC-Caravan characteristics in display units.
        minimum_river_upstream_area_km2: Minimum upstream area (km²) for
            rivers shown on the map. Larger values reduce file size.  Rivers
            with an ``uparea_m2`` column smaller than this threshold are
            dropped before rendering.  Set to ``0`` to include all rivers.

    Returns:
        The Folium map object (already saved to ``output_path``).
    """
    min_lon, min_lat, max_lon, max_lat = region_geom.total_bounds
    map_center: list[float] = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    discharge_map = folium.Map(location=map_center, tiles=None)
    TileLayer(
        tiles=_ESRI_TOPO_TILES,
        attr=_ESRI_TOPO_ATTR,
        name="Topographic Map",
    ).add_to(discharge_map)
    discharge_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(30, 30))
    folium.GeoJson(
        region_geom,
        name="Catchment",
        style_function=lambda feature: {
            "fillColor": "none",
            "color": "black",
            "weight": 2,
        },
        z_index=1,
    ).add_to(discharge_map)

    rivers_for_map: gpd.GeoDataFrame = rivers
    if minimum_river_upstream_area_km2 > 0 and "uparea_m2" in rivers_for_map.columns:
        rivers_for_map = rivers_for_map[
            rivers_for_map["uparea_m2"] >= minimum_river_upstream_area_km2 * 1e6
        ]
    folium.GeoJson(
        rivers_for_map[["geometry"]].to_json(),
        name="Rivers",
        style_function=lambda feature: {
            "color": "#4A90D9",
            "weight": 1.0,
            "opacity": 0.6,
        },
        z_index=2,
    ).add_to(discharge_map)

    metric_layers: list[tuple[folium.FeatureGroup, cm.LinearColormap, str]] = [
        (
            folium.FeatureGroup(name=cfg["name"], show=cfg["show"]),
            cm.LinearColormap(
                colors=cfg["colors"],
                vmin=cfg["vmin"],
                vmax=cfg["vmax"],
                caption=cfg["name"],
            ),
            cfg["col"],
        )
        for cfg in _METRIC_LAYER_CONFIGS
    ]

    layer_upstream: folium.FeatureGroup | None = None
    colormap_upstream: cm.LinearColormap | None = None
    if (
        not evaluation_gdf["discharge_observations_to_GEB_upstream_area_ratio"]
        .isna()
        .any()
    ):
        colormap_upstream = cm.LinearColormap(
            colors=["red", "orange", "yellow", "blue", "green"],
            vmin=0.5,
            vmax=2.0,
            caption="Upstream Area Ratio",
        )
        layer_upstream = folium.FeatureGroup(name="Upstream Area Ratio", show=False)

    characteristic_layers: list[tuple[folium.FeatureGroup, dict[str, Any]]] = []
    availability_layer: folium.FeatureGroup | None = None
    characteristic_records: dict[str, dict[str, Any]] = {}
    caravan_available_count: int = 0
    if characteristic_df is not None:
        characteristic_payload: dict[str, Any] = _build_characteristic_layer_payload(
            evaluation_gdf=evaluation_gdf,
            characteristic_df=characteristic_df,
        )
        characteristic_layers = [
            (
                folium.FeatureGroup(
                    name=f"GRDC-Caravan · {characteristic['label']}",
                    show=False,
                ),
                characteristic,
            )
            for characteristic in characteristic_payload["characteristics"]
        ]
        availability_layer = folium.FeatureGroup(
            name="GRDC-Caravan · Data availability",
            show=False,
        )
        characteristic_records = {
            str(station["id"]): station
            for station in characteristic_payload["stations"]
        }
        caravan_available_count = sum(
            bool(station["caravan_available"])
            for station in characteristic_payload["stations"]
        )

    largest_upstream_area_sqrt: float = math.sqrt(
        evaluation_gdf["upstream_area_GEB"].max()
    )

    popup_width = 800
    station_marker_index: list[StationMarkerIndex] = []

    for station_id, row in evaluation_gdf.iterrows():
        coords: list[float] = [row.geometry.y, row.geometry.x]
        station_id_str: str = str(station_id)
        station_name: str = (
            str(row["station_name"])
            if "station_name" in row.index and pd.notna(row["station_name"])
            else station_id_str
        )
        escaped_station_id: str = html.escape(station_id_str, quote=True)
        popup_html = (
            f"<div class='geb-popup' data-station-id='{escaped_station_id}' "
            f"style='width:{popup_width}px;'>Loading interactive charts...</div>"
        )
        tooltip = f"{station_id_str}: {station_name}"

        # Scale circle radius by upstream area (range 5–10 px).
        circle_radius: float = (
            5 + math.sqrt(row["upstream_area_GEB"]) / largest_upstream_area_sqrt * 5
        )
        station_marker_names: list[str] = _add_metric_station_markers(
            row=row,
            metric_layers=metric_layers,
            coords=coords,
            circle_radius=circle_radius,
            popup_html=popup_html,
            popup_width=popup_width,
            tooltip=tooltip,
        )

        if layer_upstream is not None and colormap_upstream is not None:
            color_upstream = colormap_upstream(
                float(row["discharge_observations_to_GEB_upstream_area_ratio"])
            )
            if isinstance(color_upstream, str) and color_upstream != "nan":
                station_marker_names.append(
                    _add_station_marker(
                        layer=layer_upstream,
                        coords=coords,
                        radius=10,
                        fill_color=color_upstream,
                        popup_html=popup_html,
                        popup_width=popup_width,
                        tooltip=tooltip,
                    )
                )

        if availability_layer is not None:
            station_record: dict[str, Any] = characteristic_records[station_id_str]
            station_marker_names.extend(
                _add_characteristic_station_markers(
                    station_record=station_record,
                    characteristic_layers=characteristic_layers,
                    availability_layer=availability_layer,
                    coords=coords,
                    circle_radius=circle_radius,
                    popup_html=popup_html,
                    popup_width=popup_width,
                    station_tooltip=tooltip,
                )
            )

        station_marker_index.append(
            {
                "id": station_id_str,
                "name": station_name,
                "markers": station_marker_names,
            }
        )

    for layer, _, _ in metric_layers:
        layer.add_to(discharge_map)

    if layer_upstream is not None and colormap_upstream is not None:
        layer_upstream.add_to(discharge_map)

    for characteristic_layer, _ in characteristic_layers:
        characteristic_layer.add_to(discharge_map)
    if availability_layer is not None:
        availability_layer.add_to(discharge_map)

    _inject_station_layer_legend_script(
        discharge_map=discharge_map,
        metric_layers=metric_layers,
        upstream_layer=layer_upstream,
        characteristic_layers=characteristic_layers,
        availability_layer=availability_layer,
        caravan_available_count=caravan_available_count,
        station_count=len(evaluation_gdf),
    )

    _inject_popup_chart_script(discharge_map, station_chart_files)
    _inject_station_search_script(discharge_map, station_marker_index)

    # Waterbodies: render reservoirs only; lakes make the dashboard too heavy.
    if waterbodies is not None and not waterbodies.empty:
        _add_waterbody_layers(discharge_map, waterbodies)

    folium.LayerControl(collapsed=False).add_to(discharge_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    discharge_map.save(str(output_path))
    return discharge_map
