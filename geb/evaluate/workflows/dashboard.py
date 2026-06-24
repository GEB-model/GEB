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
    minimum_river_upstream_area_km2: float = 5000.0,
) -> folium.Map:
    """Create an interactive Folium discharge evaluation map.

    Stations are shown as circle markers coloured by each discharge metric
    (switchable via layer control) and sized by upstream area.  An optional
    upstream-area-ratio layer is included when all stations have the ratio
    available.  Station popup charts are lazy-rendered with Plotly when the
    popup is opened. Reservoirs are rendered as dot markers when ``waterbodies``
    is provided; lakes are skipped because they make large dashboards slow.

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
        minimum_river_upstream_area_km2: Minimum upstream area (km²) for
            rivers shown on the map. Larger values reduce file size.  Rivers
            with an ``uparea_m2`` column smaller than this threshold are
            dropped before rendering.  Set to ``0`` to include all rivers.

    Returns:
        The Folium map object (already saved to ``output_path``).
    """
    min_lon, min_lat, max_lon, max_lat = region_geom.total_bounds
    map_center: list[float] = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    discharge_map = folium.Map(
        location=map_center,
        tiles=TileLayer(
            tiles=_ESRI_TOPO_TILES,
            attr=_ESRI_TOPO_ATTR,
            name="Topographic Map",
        ),
    )
    discharge_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(30, 30))

    folium.GeoJson(
        region_geom,
        name="Catchment",
        style_function=lambda x: {"fillColor": "none", "color": "black", "weight": 2},
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
        style_function=lambda x: {"color": "#4A90D9", "weight": 1.0, "opacity": 0.6},
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

        station_marker_index.append(
            {
                "id": station_id_str,
                "name": station_name,
                "markers": station_marker_names,
            }
        )

    for layer, colormap, _ in metric_layers:
        colormap.add_to(discharge_map)
        layer.add_to(discharge_map)

    if layer_upstream is not None and colormap_upstream is not None:
        colormap_upstream.add_to(discharge_map)
        layer_upstream.add_to(discharge_map)

    _inject_popup_chart_script(discharge_map, station_chart_files)
    _inject_station_search_script(discharge_map, station_marker_index)

    # Waterbodies: render reservoirs only; lakes make the dashboard too heavy.
    if waterbodies is not None and not waterbodies.empty:
        _add_waterbody_layers(discharge_map, waterbodies)

    folium.LayerControl(collapsed=False).add_to(discharge_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    discharge_map.save(str(output_path))
    return discharge_map
