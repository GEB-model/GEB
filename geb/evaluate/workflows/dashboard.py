"""Functions for creating interactive Folium discharge evaluation maps."""

import hashlib
import html
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.element import Figure
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
    """Load the geometries used by the discharge dashboard.

    Args:
        model: GEB model containing the geometry file registry.

    Returns:
        Region boundary, river network, and waterbodies.
    """
    region_geom: gpd.GeoDataFrame = read_geom(model.files["geom"]["mask"])
    all_rivers: gpd.GeoDataFrame = read_geom(model.files["geom"]["routing/rivers"])
    waterbodies: gpd.GeoDataFrame = read_geom(
        model.files["geom"]["waterbodies/waterbody_data"]
    )
    return DischargeDashboardGeometries(
        region=region_geom,
        rivers=all_rivers,
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


def _prepare_characteristic_values(
    values: pd.Series,
) -> tuple[dict[str, int | list[float]], pd.Series] | None:
    """Prepare legend values and percentile ranks for one characteristic.

    Map colours use empirical percentile ranks rather than raw-value intervals,
    so skewed variables retain spatial contrast. Zero is a valid observed value
    and is ranked consistently with every other finite value.

    Args:
        values: Characteristic values in display units.

    Returns:
        Legend statistics and aligned percentile ranks, or `None` when fewer
        than two distinct finite values are available.
    """
    numeric_values: pd.Series = pd.to_numeric(values, errors="coerce")
    finite_values: pd.Series = numeric_values.where(np.isfinite(numeric_values))
    valid_values: pd.Series = finite_values.dropna()
    if len(valid_values) < 2 or valid_values.nunique() < 2:
        return None

    reference_values: np.ndarray = np.nanpercentile(
        valid_values.to_numpy(dtype=float), [0.0, 25.0, 50.0, 75.0, 100.0]
    )
    percentile_ranks: pd.Series = pd.Series(np.nan, index=values.index, dtype=float)
    average_ranks: pd.Series = valid_values.rank(method="average")
    percentile_ranks.loc[valid_values.index] = (
        (average_ranks - 1.0) / (len(valid_values) - 1.0) * 100.0
    )
    statistics: dict[str, int | list[float]] = {
        "reference_values": reference_values.astype(float).tolist(),
        "missing_count": int(len(numeric_values) - len(valid_values)),
        "ranked_count": int(len(valid_values)),
    }
    return statistics, percentile_ranks


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
        prepared_values: tuple[dict[str, int | list[float]], pd.Series] | None = (
            _prepare_characteristic_values(characteristic_index[characteristic.column])
        )
        if prepared_values is None:
            continue
        statistics, percentile_ranks = prepared_values
        percentile_by_characteristic[characteristic.column] = percentile_ranks
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
    for station_id, characteristic_row in characteristic_index.iterrows():
        values: dict[str, float | None] = {}
        percentiles: dict[str, float | None] = {}
        for characteristic in usable_characteristics:
            values[characteristic.column] = _as_finite_float(
                characteristic_row[characteristic.column]
            )
            percentiles[characteristic.column] = _as_finite_float(
                percentile_by_characteristic[characteristic.column].loc[station_id]
            )
        station_records.append(
            {
                "id": str(station_id),
                "caravan_available": bool(characteristic_row["grdc_caravan_matched"])
                if pd.notna(characteristic_row["grdc_caravan_matched"])
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
        model: ReturnPeriodModel = ReturnPeriodModel(
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
    except Exception as error:
        # A failed extreme-value fit should not remove otherwise valid station
        # charts, but it must remain visible to users diagnosing the output.
        logging.getLogger(__name__).warning(
            "Could not fit dashboard return periods: %s", error
        )
        return {"returnPeriod": [], "discharge": []}


def build_discharge_dashboard_chart_data(
    validation_df: pd.DataFrame,
    station_name: str,
    upstream_area_ratio: float,
    timezone_utc_offset: float,
    metrics: dict[str, float],
    frequency: str,
    include_return_period_plots: bool = False,
) -> dict[str, Any]:
    """Build compact interactive chart data for one discharge dashboard popup.

    Args:
        validation_df: Observed/simulated discharge dataframe (m3/s).
        station_name: Human-readable station name.
        upstream_area_ratio: Observed-to-model upstream-area ratio (dimensionless).
        timezone_utc_offset: Fixed GRDC UTC offset used to construct local
            calendar days (hours). The source offset does not vary for daylight
            saving time.
        metrics: Discharge skill metrics such as ``KGE``, ``NSE``, and ``R2``
            (dimensionless).
        frequency: Data frequency label, for example ``"daily"`` or ``"hourly"``.
        include_return_period_plots: Whether to fit and include return-period
            curves. Defaults to False to avoid expensive extreme-value fits.

    Returns:
        Compact chart payload with discharge values (m3/s).

    Raises:
        ValueError: If validation_df does not use a DateTimeIndex.
    """
    if not isinstance(validation_df.index, pd.DatetimeIndex):
        raise ValueError("validation_df must use a DateTimeIndex for dashboard charts.")

    chart_data: dict[str, Any] = {
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
            "timezoneUtcOffset": _as_finite_float(timezone_utc_offset),
        },
        "timeseries": _build_timeseries_payload(validation_df),
    }
    if include_return_period_plots:
        # Fit only on request: fitting every station dominates dashboard creation.
        return_periods_years: list[int | float] = [2, 5, 10, 25, 50, 100]
        simulated_series: pd.Series = validation_df["discharge_simulations"].copy()
        simulated_series[validation_df["discharge_observations"].isna()] = np.nan
        chart_data["returnPeriods"] = {
            "observed": _build_return_period_payload(
                validation_df["discharge_observations"], return_periods_years
            ),
            "simulated": _build_return_period_payload(
                simulated_series, return_periods_years
            ),
        }
    return chart_data


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
    Plotly.newPlot('geb-time-' + safeStationId, [
      trace('Observed', data.timeseries.time, data.timeseries.observed, timeseriesTraceType, 'lines', '%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Observed</extra>'),
      trace('Simulated', data.timeseries.time, data.timeseries.simulated, timeseriesTraceType, 'lines', '%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Simulated</extra>')
    ], Object.assign({}, layoutBase, {hovermode: 'x unified', xaxis: Object.assign({}, layoutBase.xaxis, {type: 'date', range: timeRange}), yaxis: Object.assign({}, layoutBase.yaxis, {title: 'Discharge (m3/s)'})}), common);
    if (data.returnPeriods) {
      var observedReturnPeriodRange = linearRange(data.returnPeriods.observed.returnPeriod);
      var simulatedReturnPeriodRange = linearRange(data.returnPeriods.simulated.returnPeriod);
      var returnPeriodValues = [];
      if (observedReturnPeriodRange) returnPeriodValues = returnPeriodValues.concat(observedReturnPeriodRange);
      if (simulatedReturnPeriodRange) returnPeriodValues = returnPeriodValues.concat(simulatedReturnPeriodRange);
      var returnPeriodRange = logRange(returnPeriodValues);
      var returnPeriodTicks = sortedUniqueNumbers(
        data.returnPeriods.observed.returnPeriod.concat(data.returnPeriods.simulated.returnPeriod)
      );
      Plotly.newPlot('geb-return-' + safeStationId, [
        trace('Observed', data.returnPeriods.observed.returnPeriod, data.returnPeriods.observed.discharge, 'scatter', 'lines+markers', '%{x:g}-year<br>%{y:,.0f} m3/s<extra>Observed</extra>'),
        trace('Simulated', data.returnPeriods.simulated.returnPeriod, data.returnPeriods.simulated.discharge, 'scatter', 'lines+markers', '%{x:g}-year<br>%{y:,.0f} m3/s<extra>Simulated</extra>')
      ], Object.assign({}, layoutBase, {hovermode: 'x unified', xaxis: Object.assign({}, layoutBase.xaxis, {type: 'log', range: returnPeriodRange, tickmode: 'array', tickvals: returnPeriodTicks, ticktext: returnPeriodTicks.map(formatTick), title: 'Return period (years)'}), yaxis: Object.assign({}, layoutBase.yaxis, {title: 'Discharge (m3/s)'})}), common);
    }
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
      metricHtml('RRMSE', metrics.RRMSE) + metricHtml('Area ratio', metrics.upstreamAreaRatio) +
      metricHtml('Fixed UTC offset (h)', metrics.timezoneUtcOffset) + '</div>' +
      (data.returnPeriods ? '<div class="geb-popup__chart-title">Return periods</div>' + makeChartDiv('geb-return-' + safeStationId) : '') +
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
                "missing_count": characteristic["missing_count"],
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
        assert percentile is not None, (
            f"Finite characteristic {characteristic['column']} has no rank."
        )
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
        formatted_value: str = f"{value:,.{decimal_places}f}"
        fill_color: str = _CHARACTERISTIC_COLORMAP(percentile)
        rank_text: str = f"percentile rank {percentile:.0f}"
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
                    f"{formatted_value} ({rank_text})"
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


def format_fixed_utc_offset(offset_hours: float) -> str:
    """Format a fixed UTC offset for dashboard labels.

    Args:
        offset_hours: Fixed offset from UTC (hours).

    Returns:
        Label such as ``UTC+02:00`` or ``UTC-03:30``.

    Raises:
        ValueError: If the offset is non-finite or outside UTC-12 to UTC+14.
    """
    if not np.isfinite(offset_hours) or not -12.0 <= offset_hours <= 14.0:
        raise ValueError("UTC offset must be finite and between UTC-12 and UTC+14.")
    absolute_minutes: int = round(abs(offset_hours) * 60.0)
    whole_hours, minutes = divmod(absolute_minutes, 60)
    sign: str = "+" if offset_hours >= 0.0 else "-"
    return f"UTC{sign}{whole_hours:02d}:{minutes:02d}"


def _haversine_distance_km(
    first_longitude: float,
    first_latitude: float,
    second_longitude: float,
    second_latitude: float,
) -> float:
    """Calculate great-circle distance between two coordinates.

    Args:
        first_longitude: First longitude (degrees east).
        first_latitude: First latitude (degrees north).
        second_longitude: Second longitude (degrees east).
        second_latitude: Second latitude (degrees north).

    Returns:
        Great-circle distance (km).
    """
    earth_radius_km: float = 6371.0088
    first_lon_rad: float = math.radians(first_longitude)
    first_lat_rad: float = math.radians(first_latitude)
    second_lon_rad: float = math.radians(second_longitude)
    second_lat_rad: float = math.radians(second_latitude)
    longitude_difference: float = second_lon_rad - first_lon_rad
    latitude_difference: float = second_lat_rad - first_lat_rad
    haversine_value: float = (
        math.sin(latitude_difference / 2.0) ** 2
        + math.cos(first_lat_rad)
        * math.cos(second_lat_rad)
        * math.sin(longitude_difference / 2.0) ** 2
    )
    return earth_radius_km * 2.0 * math.asin(math.sqrt(min(1.0, haversine_value)))


def _build_snapping_qc_station(
    station_id: str,
    station_name: str,
    row: pd.Series,
) -> dict[str, Any]:
    """Build one compact station record for on-demand snapping visualization.

    Excluded stations show the reason alongside their snapping diagnostics.

    Args:
        station_id: GRDC station identifier.
        station_name: Human-readable station name.
        row: Evaluation row with snapping metadata.

    Returns:
        Coordinates, tooltip, status color, and popup text.

    Raises:
        ValueError: If a coordinate is invalid or a fixed UTC offset is invalid.
    """
    gauge_longitude: float = float(row["station_longitude"])
    gauge_latitude: float = float(row["station_latitude"])
    if any(
        pd.isna(row.get(column))
        for column in (
            "original_longitude",
            "original_latitude",
            "snapped_grid_longitude",
            "snapped_grid_latitude",
        )
    ):
        reason: str = html.escape(
            str(row.get("exclusion_reason", "No valid river match."))
        )
        station_label: str = f"{html.escape(station_name)} ({html.escape(station_id)})"
        area_m2: float = float(row.get("upstream_area_GRDC", np.nan))
        area_label: str = (
            f"{area_m2 / 1e6:,.1f} km²"
            if np.isfinite(area_m2) and area_m2 > 0
            else "missing"
        )
        return {
            "id": station_id,
            "locations": [[gauge_latitude, gauge_longitude]],
            "color": "#DC2626",
            "tooltip": f"{station_label}<br>EXCLUDED: {reason}",
            "popup": f"<b>{station_label}</b><br><b>EXCLUDED</b><br>{reason}<br>GRDC area: {area_label}<br>Gauge: {gauge_latitude:.5f}, {gauge_longitude:.5f}",
        }
    original_longitude: float = float(row["original_longitude"])
    original_latitude: float = float(row["original_latitude"])
    routing_longitude: float = float(row["snapped_grid_longitude"])
    routing_latitude: float = float(row["snapped_grid_latitude"])
    grdc_area_km2: float = float(row["upstream_area_GRDC"]) / 1_000_000.0
    routing_area_km2: float = float(row["upstream_area_GEB"]) / 1_000_000.0
    original_area_km2: float = float(row["upstream_area_GEB_original"]) / 1_000_000.0
    station_to_routing_distance_km: float = _haversine_distance_km(
        gauge_longitude,
        gauge_latitude,
        routing_longitude,
        routing_latitude,
    )
    station_to_original_distance_km: float = (
        float(row["station_to_original_distance_m"]) / 1000.0
    )
    original_to_routing_distance_km: float = _haversine_distance_km(
        original_longitude,
        original_latitude,
        routing_longitude,
        routing_latitude,
    )
    grdc_routing_area_ratio: float = (
        grdc_area_km2 / routing_area_km2 if routing_area_km2 > 0 else float("nan")
    )
    original_grdc_area_ratio: float = (
        original_area_km2 / grdc_area_km2 if grdc_area_km2 > 0 else float("nan")
    )
    routing_original_area_ratio: float = (
        routing_area_km2 / original_area_km2 if original_area_km2 > 0 else float("nan")
    )
    timezone_label: str = format_fixed_utc_offset(float(row["timezone_utc_offset"]))
    area_warning: bool = not 0.9 <= routing_original_area_ratio <= 1.1
    status_label: str = "ROUTING AREA WARNING" if area_warning else "PASS"
    status_color: str = "#EA580C" if area_warning else "#16A34A"
    exclusion_reason: str = (
        str(row.get("exclusion_reason", ""))
        if pd.notna(row.get("exclusion_reason", ""))
        else ""
    )
    if exclusion_reason:
        status_label = "EXCLUDED"
        status_color = "#DC2626"
    escaped_name: str = html.escape(station_name)
    escaped_id: str = html.escape(station_id)
    popup_html: str = (
        f"<b>{escaped_name}</b> ({escaped_id})<br>"
        f"<b>Snapping QC: <span style='color:{status_color}'>{status_label}</span></b><br>"
        f"GRDC gauge: {gauge_latitude:.5f}, {gauge_longitude:.5f}<br>"
        f"Selected original pixel: {original_latitude:.5f}, {original_longitude:.5f}<br>"
        f"Routing pixel: {routing_latitude:.5f}, {routing_longitude:.5f}<br>"
        f"River ID: {int(row['snapped_river_id'])}<br>"
        f"Gauge–routing distance: {station_to_routing_distance_km:.2f} km<br>"
        f"Gauge–original distance: {station_to_original_distance_km:.3f} km<br>"
        f"Original–routing distance: {original_to_routing_distance_km:.3f} km<br>"
        f"GRDC area: {grdc_area_km2:,.1f} km²<br>"
        f"Original area: {original_area_km2:,.1f} km²<br>"
        f"Routing area: {routing_area_km2:,.1f} km²<br>"
        f"Original/GRDC area ratio: {original_grdc_area_ratio:.3f}<br>"
        f"Routing/original area ratio: {routing_original_area_ratio:.3f}<br>"
        f"GRDC/routing area ratio: {grdc_routing_area_ratio:.3f}<br>"
        f"Daily aggregation offset: {timezone_label} (fixed; no DST)<br>"
        "Observation day: local midnight to midnight<br>"
        f"{html.escape(exclusion_reason)}"
    )
    tooltip: str = (
        f"{escaped_id}: {escaped_name}<br>Snapping QC: {status_label}"
        f"<br>Original/GRDC area: {original_grdc_area_ratio:.3f}; "
        f"routing/original: {routing_original_area_ratio:.3f}"
        f"<br>Gauge–routing: {station_to_routing_distance_km:.2f} km"
        f"<br>{timezone_label} fixed"
        "<br>Daily window: local midnight to midnight"
    )
    line_locations: list[list[float]] = [
        [gauge_latitude, gauge_longitude],
        [original_latitude, original_longitude],
        [routing_latitude, routing_longitude],
    ]
    if any(
        not np.isfinite(latitude)
        or not np.isfinite(longitude)
        or not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
        for latitude, longitude in line_locations
    ):
        raise ValueError("Snapping coordinates must be finite geographic coordinates.")
    return {
        "id": escaped_id,
        "locations": line_locations,
        "color": status_color,
        "tooltip": tooltip,
        "popup": popup_html,
    }


def _inject_snapping_qc_script(
    discharge_map: folium.Map,
    layer: folium.FeatureGroup,
    stations: list[dict[str, Any]],
) -> None:
    """Render snapping features lazily and only around the visible map extent.

    Args:
        discharge_map: Map receiving the snapping interaction and legend.
        layer: Initially empty, disabled overlay shown in the layer control.
        stations: Compact station records with coordinates in degrees and HTML
            diagnostics from _build_snapping_qc_station.

    Notes:
        Overview points and connectors share a canvas renderer. Numbered markers
        are limited to 150 visible stations at zoom 9 or higher, or one selected
        station in overview mode. Permanent labels are created only from zoom 11.
        The inactive overlay has no station layers or popup objects.
    """
    script: str = r"""
(function() {
  var map = GEB_SNAPPING_MAP;
  var layer = GEB_SNAPPING_LAYER;
  var stations = GEB_SNAPPING_STATIONS;
  var visibleLayers = new Map();
  var renderer = null;
  var selectedLayer = null;
  var selectedIndex = null;
  var currentMode = '';
  var updateTimer = null;
  var popup = null;
  var markerStyles = [
    ['1', 'Original gauge', '#BE123C', '50%'],
    ['2', 'Selected original pixel', '#92400E', '3px'],
    ['3', 'Snapped model cell', '#1D4ED8', '0']
  ];
  var control = L.control({position: 'bottomright'});
  control.onAdd = function() {
    var box = L.DomUtil.create('div', 'geb-snapping-legend');
    box.style.display = 'none';
    box.innerHTML = '<b>Station snapping: 1 → 2 → 3</b>' +
      '<div><i style="background:#BE123C;border-radius:50%">1</i> Original gauge (observations)</div>' +
      '<div><i style="background:#92400E;border-radius:3px">2</i> Selected original pixel</div>' +
      '<div><i style="background:#1D4ED8">3</i> Snapped model-cell centre (simulation)</div>' +
      '<small>Overview dots: green = PASS, orange = area warning, red = excluded.<br>' +
      'Click a dot for its three snapping steps, or zoom in.<br>' +
      'Dashed lines connect the three locations.<br>' +
      'Coincident symbols can overlap; each popup lists all coordinates.</small>' +
      '<small class="geb-snapping-status"></small>';
    L.DomEvent.disableClickPropagation(box);
    L.DomEvent.disableScrollPropagation(box);
    return box;
  };
  control.addTo(map);
  var style = document.createElement('style');
  style.textContent = '.geb-snapping-legend{background:white;color:#111827;padding:12px;border-radius:8px;box-shadow:0 2px 10px #0004;font:12px system-ui;max-width:310px}' +
    '.geb-snapping-legend div{margin-top:6px;display:flex;align-items:center;gap:8px}' +
    '.geb-snapping-legend i{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;color:white;font-style:normal;font-weight:bold}' +
    '.geb-snapping-legend small{display:block;margin-top:8px;line-height:1.5}';
  document.head.appendChild(style);

  function openPopup(station, heading, location) {
    if (!popup) popup = L.popup({maxWidth: 440});
    popup.setLatLng(location).setContent(heading + station.popup).openOn(map);
  }
  function detail(index, labels) {
    var station = stations[index];
    var group = L.featureGroup();
    var connector = L.polyline(station.locations, {
      renderer: renderer, color: station.color, weight: 3, opacity: 0.85, dashArray: '6 4'
    }).addTo(group);
    connector.on('click', function(event) {openPopup(station, '', event.latlng);});
    station.locations.forEach(function(coordinates, position) {
      var spec = markerStyles[position];
      var marker = L.marker(coordinates, {icon: L.divIcon({
        iconSize: [26, 26], iconAnchor: [13, 13], className: 'geb-snapping-icon',
        html: '<span style="display:flex;align-items:center;justify-content:center;width:26px;height:26px;box-sizing:border-box;background:' + spec[2] +
          ';border:2px solid white;border-radius:' + spec[3] + ';color:white;font:bold 14px system-ui;box-shadow:0 1px 5px #0008;">' + spec[0] + '</span>'
      })}).addTo(group);
      marker.bindTooltip(spec[0] + '. ' + spec[1] + ' · ' + station.id, {
        permanent: labels, direction: position === 0 ? 'left' : 'right', className: 'geb-snapping-label'
      });
      marker.on('click', function() {
        openPopup(station, '<h4>' + spec[0] + '. ' + spec[1] + '</h4>', coordinates);
      });
    });
    return group;
  }
  function overview(index) {
    var station = stations[index];
    var dot = L.circleMarker(station.locations[0], {
      renderer: renderer, radius: 5, color: station.color, fillColor: station.color,
      fillOpacity: 0.85, weight: 1
    });
    dot.bindTooltip(station.tooltip);
    dot.on('click', function() {
      if (selectedLayer) layer.removeLayer(selectedLayer);
      selectedIndex = index;
      selectedLayer = detail(index, map.getZoom() >= 11).addTo(layer);
      openPopup(station, '', station.locations[0]);
    });
    return dot;
  }
  function clear() {
    layer.clearLayers();
    visibleLayers.clear();
    selectedLayer = null;
    selectedIndex = null;
    currentMode = '';
    if (popup) map.closePopup(popup);
  }
  function update() {
    updateTimer = null;
    if (!map.hasLayer(layer)) return;
    if (!renderer) renderer = L.canvas({padding: 0.2});
    var bounds = map.getBounds().pad(0.1);
    var visible = [];
    stations.forEach(function(station, index) {
      // Include a connector crossing the viewport even if its gauge is outside.
      if (bounds.intersects(L.latLngBounds(station.locations))) visible.push(index);
    });
    var detailed = map.getZoom() >= 9 && visible.length <= 150;
    var labels = map.getZoom() >= 11;
    var mode = detailed ? (labels ? 'labelled' : 'detailed') : 'overview';
    var previousSelection = selectedIndex;
    if (mode !== currentMode) {clear(); currentMode = mode; selectedIndex = previousSelection;}
    var wanted = new Set(visible);
    visibleLayers.forEach(function(feature, index) {
      if (!wanted.has(index)) {layer.removeLayer(feature); visibleLayers.delete(index);}
    });
    visible.forEach(function(index) {
      if (!visibleLayers.has(index)) {
        visibleLayers.set(index, (detailed ? detail(index, labels) : overview(index)).addTo(layer));
      }
    });
    if (selectedLayer && !wanted.has(selectedIndex)) {
      layer.removeLayer(selectedLayer); selectedLayer = null; selectedIndex = null;
      if (popup) map.closePopup(popup);
    }
    control.getContainer().querySelector('.geb-snapping-status').textContent =
      visible.length + ' stations in view. ' + (detailed ? (labels ? 'Numbered steps and labels shown.' : 'Zoom further for labels.') :
        'Zoom in for all steps (up to 150 stations), or click a dot.');
  }
  function schedule() {
    // Coalesce zoomend/moveend without waiting for an animation frame.
    if (map.hasLayer(layer) && updateTimer === null) updateTimer = setTimeout(update, 50);
  }
  function overlayChanged(event) {
    if (event.layer !== layer) return;
    var enabled = map.hasLayer(layer);
    control.getContainer().style.display = enabled ? '' : 'none';
    if (enabled) schedule();
    else {
      if (updateTimer !== null) clearTimeout(updateTimer);
      updateTimer = null;
      clear();
      if (renderer && map.hasLayer(renderer)) map.removeLayer(renderer);
    }
  }
  map.on('overlayadd overlayremove', overlayChanged);
  map.on('moveend zoomend', schedule);
})();
"""
    script = script.replace("GEB_SNAPPING_MAP", discharge_map.get_name()).replace(
        "GEB_SNAPPING_LAYER", layer.get_name()
    )
    # Folium renders nested templates more than once. Escape template openers
    # inside station strings as well as script-closing HTML characters.
    station_json: str = (
        json.dumps(stations, separators=(",", ":"), ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace("{{", "\\u007b\\u007b")
        .replace("{%", "\\u007b%")
        .replace("{#", "\\u007b#")
    )
    script = script.replace("GEB_SNAPPING_STATIONS", station_json)
    _JavascriptMacro(script).add_to(discharge_map)


def _add_river_layers(
    discharge_map: folium.Map,
    rivers: gpd.GeoDataFrame,
    overview_minimum_area_km2: float,
    detailed_minimum_zoom: int,
) -> None:
    """Add overview and zoomed MERIT river layers with river ID tooltips.

    Args:
        discharge_map: Map receiving the river layers.
        rivers: MERIT river segments in WGS84, indexed by river ID.
        overview_minimum_area_km2: Minimum upstream area shown below the detailed
            zoom level (km²).
        detailed_minimum_zoom: First zoom level showing every river segment.

    Raises:
        ValueError: If the detailed zoom level is outside the Leaflet range.
    """
    if not 0 <= detailed_minimum_zoom <= 22:
        raise ValueError("detailed_minimum_zoom must be between 0 and 22.")
    if rivers.empty:
        return

    valid_rivers: gpd.GeoDataFrame = rivers.loc[
        rivers.geometry.notna() & ~rivers.geometry.is_empty
    ].copy()
    overview_exclusions: list[str] = [
        name
        for name in (
            "is_downstream_outflow",
            "is_upstream_of_downstream_basin",
            "is_further_downstream_outflow",
        )
        if name in valid_rivers.columns
    ]
    overview_rivers: gpd.GeoDataFrame = valid_rivers
    if overview_exclusions:
        overview_rivers = overview_rivers.loc[
            ~overview_rivers[overview_exclusions].any(axis=1)
        ]
    if overview_minimum_area_km2 > 0 and "uparea_m2" in overview_rivers.columns:
        overview_rivers = overview_rivers.loc[
            overview_rivers["uparea_m2"] >= overview_minimum_area_km2 * 1e6
        ]

    def display_data(selected: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Keep only compact river properties needed by the browser.

        Args:
            selected: River segments to convert.

        Returns:
            River IDs, upstream areas (km²), and geometries.
        """
        upstream_area_km2: pd.Series = (
            selected["uparea_m2"] / 1e6
            if "uparea_m2" in selected.columns
            else pd.Series(np.nan, index=selected.index)
        )
        return gpd.GeoDataFrame(
            {
                "river_id": selected.index.astype(str),
                "upstream_area_km2": upstream_area_km2.to_numpy(),
            },
            geometry=selected.geometry.to_numpy(),
            crs=selected.crs,
        )

    def river_tooltip() -> folium.GeoJsonTooltip:
        """Create a tooltip owned by one GeoJSON layer.

        Returns:
            Tooltip showing the MERIT river ID and upstream area (km²).
        """
        return folium.GeoJsonTooltip(
            fields=["river_id", "upstream_area_km2"],
            aliases=["River ID:", "Upstream area (km²):"],
            localize=True,
            sticky=True,
        )

    overview_layer: folium.FeatureGroup = folium.FeatureGroup(
        name="Major rivers overview", show=True
    )
    folium.GeoJson(
        display_data(overview_rivers).to_json(drop_id=True),
        style_function=lambda _feature: {
            "color": "#4A90D9",
            "weight": 1.2,
            "opacity": 0.65,
        },
        tooltip=river_tooltip(),
        highlight_function=lambda _feature: {"weight": 4, "opacity": 1},
    ).add_to(overview_layer)
    overview_layer.add_to(discharge_map)

    # A small visual simplification keeps the complete network responsive.
    detailed_rivers: gpd.GeoDataFrame = valid_rivers.copy()
    detailed_rivers.geometry = detailed_rivers.geometry.simplify(
        0.0004, preserve_topology=False
    )
    detailed_layer: folium.FeatureGroup = folium.FeatureGroup(
        name=f"MERIT river network (zoom {detailed_minimum_zoom}+)", show=False
    )
    folium.GeoJson(
        display_data(detailed_rivers).to_json(drop_id=True),
        style_function=lambda _feature: {
            "color": "#2563EB",
            "weight": 1.5,
            "opacity": 0.8,
        },
        tooltip=river_tooltip(),
        highlight_function=lambda _feature: {"weight": 5, "opacity": 1},
        smooth_factor=1.0,
    ).add_to(detailed_layer)
    detailed_layer.add_to(discharge_map)

    script: str = """
(function() {
  var map = GEB_RIVER_MAP;
  var overview = GEB_RIVER_OVERVIEW;
  var detailed = GEB_RIVER_DETAILED;
  var minimumZoom = GEB_RIVER_MINIMUM_ZOOM;
  function updateRivers() {
    if (map.getZoom() >= minimumZoom) {
      if (map.hasLayer(overview)) map.removeLayer(overview);
      if (!map.hasLayer(detailed)) map.addLayer(detailed);
    } else {
      if (map.hasLayer(detailed)) map.removeLayer(detailed);
      if (!map.hasLayer(overview)) map.addLayer(overview);
    }
  }
  map.on('zoomend', updateRivers);
  updateRivers();
})();
"""
    script = script.replace("GEB_RIVER_MAP", discharge_map.get_name())
    script = script.replace("GEB_RIVER_OVERVIEW", overview_layer.get_name())
    script = script.replace("GEB_RIVER_DETAILED", detailed_layer.get_name())
    script = script.replace("GEB_RIVER_MINIMUM_ZOOM", str(detailed_minimum_zoom))
    _JavascriptMacro(script).add_to(discharge_map)


def create_discharge_folium_map(
    evaluation_gdf: gpd.GeoDataFrame,
    output_path: Path,
    region_geom: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    station_chart_files: dict[str, str],
    waterbodies: gpd.GeoDataFrame | None = None,
    characteristic_df: pd.DataFrame | None = None,
    minimum_river_upstream_area_km2: float = 5000.0,
    detailed_river_minimum_zoom: int = 8,
    excluded_stations: gpd.GeoDataFrame | None = None,
) -> folium.Map:
    """Create an interactive Folium discharge evaluation map.

    Stations are shown as circle markers coloured by each discharge metric
    (switchable via layer control) and sized by upstream area.  An optional
    upstream-area-ratio layer is included when all stations have the ratio
    available.  Station popup charts are lazy-rendered with Plotly when the
    popup is opened. Reservoirs are rendered as dot markers when ``waterbodies``
    is provided; lakes are skipped because they make large dashboards slow.
    GRDC-Caravan characteristics are optional station layers on the same map,
    together with a separate data-availability layer. A snapping-QC layer shows
    the original gauge, selected original pixel, routing pixel, connecting
    line, upstream-area agreement, distance, and fixed UTC offset.
    Topographic and satellite backgrounds are selectable in the layer control.
    Snapping features load only when enabled: canvas overview dots at regional
    scales and numbered steps for at most 150 visible stations when zoomed in.

    Args:
        evaluation_gdf: Per-station GeoDataFrame with discharge metric columns,
            ``upstream_area_GEB``,
            ``discharge_observations_to_GEB_upstream_area_ratio``, and a
            point geometry.
        output_path: Full path (including filename) where the HTML file is
            saved.
        region_geom: Basin/region boundary GeoDataFrame used to fit the map
            extent and render the catchment outline.
        rivers: WGS84 river network shown on the map.
        station_chart_files: Exact interactive chart payload files keyed by
            station ID string.
        waterbodies: Optional GeoDataFrame with columns ``waterbody_type``
            (2 = reservoir) and polygon geometries. Centroids are used for dot
            placement.
        characteristic_df: Optional station table containing ``station_ID`` and
            the curated GRDC-Caravan characteristics in display units.
        excluded_stations: Stations omitted from summary scores, with an exclusion
            reason. Available charts remain accessible for diagnostic use.
        minimum_river_upstream_area_km2: Minimum upstream area (km²) for the
            river overview shown at regional zoom levels.
        detailed_river_minimum_zoom: Zoom level at which the complete MERIT
            river network replaces the overview.
    Returns:
        The Folium map object (already saved to ``output_path``).
    """
    min_lon, min_lat, max_lon, max_lat = region_geom.total_bounds
    if excluded_stations is not None and not excluded_stations.empty:
        evaluation_gdf = evaluation_gdf.copy()
        # Apply current exclusion reasons when displaying previously saved scores.
        excluded_ids: pd.Index = evaluation_gdf.index.intersection(
            excluded_stations.index
        )
        evaluation_gdf.loc[excluded_ids, "exclusion_reason"] = excluded_stations.loc[
            excluded_ids, "exclusion_reason"
        ]
    map_center: list[float] = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    discharge_map = folium.Map(location=map_center, tiles=None, prefer_canvas=True)
    TileLayer(
        tiles=_ESRI_TOPO_TILES,
        attr=_ESRI_TOPO_ATTR,
        name="Topographic Map",
    ).add_to(discharge_map)
    TileLayer(
        tiles="Esri.WorldImagery",
        name="Satellite imagery",
        show=False,
        max_zoom=19,
    ).add_to(discharge_map)
    discharge_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(30, 30))
    folium.GeoJson(
        region_geom,
        name="Catchment",
        style_function=lambda _feature: {
            "fillColor": "none",
            "color": "black",
            "weight": 2,
        },
        z_index=1,
    ).add_to(discharge_map)

    _add_river_layers(
        discharge_map,
        rivers,
        minimum_river_upstream_area_km2,
        detailed_river_minimum_zoom,
    )

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
            caption="GRDC / routing upstream area",
        )
        layer_upstream = folium.FeatureGroup(
            name="GRDC / routing upstream area", show=False
        )

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

    popup_width: int = 800
    station_marker_index: list[StationMarkerIndex] = []
    snapping_qc_layer: folium.FeatureGroup = folium.FeatureGroup(
        name="Station snapping QC (gauge → original pixel → routing grid)",
        show=False,
    )
    snapping_columns: set[str] = {
        "station_longitude",
        "station_latitude",
        "snapped_grid_longitude",
        "snapped_grid_latitude",
        "original_longitude",
        "original_latitude",
        "upstream_area_GRDC",
        "upstream_area_GEB",
        "upstream_area_GEB_original",
        "station_to_original_distance_m",
        "snapped_river_id",
        "snapping_method",
        "timezone_utc_offset",
    }
    snapping_qc_available: bool = snapping_columns.issubset(evaluation_gdf.columns)
    snapping_stations: list[dict[str, Any]] = []

    for station_id, row in evaluation_gdf.iterrows():
        coords: list[float] = [row.geometry.y, row.geometry.x]
        station_id_str: str = str(station_id)
        station_name: str = (
            str(row["station_name"])
            if "station_name" in row.index and pd.notna(row["station_name"])
            else station_id_str
        )
        escaped_station_id: str = html.escape(station_id_str, quote=True)
        popup_html: str = (
            f"<div class='geb-popup' data-station-id='{escaped_station_id}' "
            f"style='width:{popup_width}px;'>Loading interactive charts...</div>"
        )
        if pd.notna(row.get("exclusion_reason")):
            popup_html = (
                "<div style='color:#b91c1c;padding:8px'><b>Diagnostic only — excluded from evaluation.</b><br>"
                + html.escape(str(row["exclusion_reason"]))
                + "</div>"
                + popup_html
            )
        timezone_tooltip: str = (
            f"<br>{format_fixed_utc_offset(float(row['timezone_utc_offset']))} fixed"
            if "timezone_utc_offset" in row.index
            and pd.notna(row["timezone_utc_offset"])
            else ""
        )
        tooltip: str = f"{station_id_str}: {station_name}{timezone_tooltip}"
        if pd.notna(row.get("exclusion_reason")):
            tooltip += "<br>Diagnostic only — excluded from evaluation"

        if snapping_qc_available:
            snapping_record: dict[str, Any] = _build_snapping_qc_station(
                station_id=station_id_str,
                station_name=station_name,
                row=row,
            )
            snapping_stations.append(snapping_record)

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
            color_upstream: str | tuple[int, int, int, int] = colormap_upstream(
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
    if excluded_stations is not None and not excluded_stations.empty:
        area_distance_count: int = 0
        excluded_layer: folium.FeatureGroup = folium.FeatureGroup(
            name=f"Excluded stations ({len(excluded_stations)})", show=False
        )
        for station_id, row in excluded_stations.iterrows():
            record: dict[str, Any] = _build_snapping_qc_station(
                str(station_id), str(row["station_name"]), row
            )
            area_distance_failure: bool = str(row["exclusion_reason"]).startswith(
                (
                    "Missing GRDC",
                    "Missing upstream",
                    "Routing upstream",
                    "Routing pixel",
                    "No valid river",
                )
            )
            area_distance_count += int(area_distance_failure)
            record["color"] = "#DC2626" if area_distance_failure else "#D97706"
            if station_id not in evaluation_gdf.index and len(record["locations"]) == 3:
                snapping_stations.append(record)
            excluded_popup: str = record["popup"]
            if str(station_id) in station_chart_files:
                excluded_popup += (
                    "<hr><b>Diagnostic only — excluded from evaluation.</b>"
                    f"<div class='geb-popup' data-station-id='{html.escape(str(station_id), quote=True)}'>"
                    "Loading interactive charts...</div>"
                )
            marker: folium.CircleMarker = folium.CircleMarker(
                location=record["locations"][0],
                radius=7 if area_distance_failure else 5,
                color=record["color"],
                fill=True,
                fill_opacity=0.8,
                tooltip=record["tooltip"],
                popup=folium.Popup(excluded_popup, max_width=850),
            )
            marker.add_to(excluded_layer)
            station_marker_index.append(
                {
                    "id": str(station_id),
                    "name": str(row["station_name"]),
                    "markers": [marker.get_name()],
                }
            )
        excluded_layer.add_to(discharge_map)
        station_count: int = len(evaluation_gdf.index.union(excluded_stations.index))
        cast(Figure, discharge_map.get_root()).html.add_child(
            folium.Element(
                f"<div id='{excluded_layer.get_name()}_legend' style='display:none;position:fixed;top:12px;left:55px;z-index:1000;background:white;padding:8px;border:1px solid #ccc'>"
                f"<b>{station_count} stations shown</b><br>"
                f"<span style='color:#DC2626'>● Area / distance exclusions: {area_distance_count}</span><br>"
                f"<span style='color:#D97706'>● Other exclusions: {len(excluded_stations) - area_distance_count}</span>"
                "<br>Hover or click a station for its exclusion reason.</div>"
            )
        )
        _JavascriptMacro(f"""
(function() {{
  var map = {discharge_map.get_name()};
  var layer = {excluded_layer.get_name()};
  var legend = document.getElementById('{excluded_layer.get_name()}_legend');
  function updateExclusionLegend() {{
    legend.style.display = map.hasLayer(layer) ? '' : 'none';
  }}
  layer.on('add remove', updateExclusionLegend);
  updateExclusionLegend();
}})();
""").add_to(discharge_map)
    if snapping_stations:
        snapping_qc_layer.add_to(discharge_map)
        _inject_snapping_qc_script(discharge_map, snapping_qc_layer, snapping_stations)

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
    if (
        "exclusion_reason" in evaluation_gdf
        and evaluation_gdf["exclusion_reason"].notna().any()
    ):
        cast(Figure, discharge_map.get_root()).html.add_child(
            folium.Element(
                "<div style='position:fixed;bottom:25px;left:15px;z-index:1000;background:white;padding:10px;max-width:350px'>"
                "<b>Diagnostic scores included</b><br>Excluded stations retain KGE and time series for inspection. "
                "They do not contribute to evaluation summaries. See station warnings.</div>"
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    discharge_map.save(str(output_path))
    return discharge_map
