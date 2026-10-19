{%- macro html(this, kwargs) -%}
<script id="geb-chart-data" type="application/json">
{{ this.data | script_json }}
</script>
{%- endmacro -%}
{%- macro script(this, kwargs) -%}
(function(){
  var chartData = null;
  function getChartData() {
    if (!chartData) {
      var element = document.getElementById("geb-chart-data");
      chartData = (element && element.textContent) ? JSON.parse(element.textContent) : {};
    }
    return chartData;
  }
  var plotlyUrl = "https://cdn.plot.ly/plotly-2.35.2.min.js";
  var stationChartCache = {};
  var activePlotlyElements = {};

  var darkColors = {
    observed: "#facc15",
    simulated: "#38bdf8",
    bankfull: "#f87171",
    width: "#38bdf8",
    depth: "#4ade80"
  };
  var lightColors = {
    observed: "#d97706",
    simulated: "#0284c7",
    bankfull: "#dc2626",
    width: "#0284c7",
    depth: "#16a34a"
  };

  function getActiveTheme() {
    return window._gebCurrentTheme || document.documentElement.getAttribute("data-geb-theme") || "light";
  }

  function getThemeColors(theme) {
    return (theme === "dark") ? darkColors : lightColors;
  }

  function getPlotlyLayout(theme, customOpts) {
    var isDark = (theme === "dark");
    var base = {
      autosize: true,
      height: (customOpts && customOpts.height) || 250,
      margin: {l: 45, r: 15, t: 15, b: 38},
      paper_bgcolor: isDark ? "#020617" : "#ffffff",
      plot_bgcolor: isDark ? "#020617" : "#ffffff",
      font: {color: isDark ? "#e2e8f0" : "#0f172a", size: 11, family: "Inter, system-ui, -apple-system, sans-serif"},
      legend: {orientation: "h", x: 0, y: 1.15, font: {color: isDark ? "#e2e8f0" : "#0f172a"}},
      xaxis: {
        gridcolor: isDark ? "#1e293b" : "#f1f5f9",
        zerolinecolor: isDark ? "#334155" : "#cbd5e1",
        tickfont: {color: isDark ? "#94a3b8" : "#475569"}
      },
      yaxis: {
        gridcolor: isDark ? "#1e293b" : "#f1f5f9",
        zerolinecolor: isDark ? "#334155" : "#cbd5e1",
        rangemode: "tozero",
        tickfont: {color: isDark ? "#94a3b8" : "#475569"}
      }
    };
    return Object.assign({}, base, customOpts || {});
  }

  function ensurePlotly(callback) {
    if (window.Plotly) { callback(); return; }
    var script = document.createElement("script");
    script.src = plotlyUrl;
    script.onload = callback;
    script.onerror = function() { callback(false); };
    document.head.appendChild(script);
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function(character) {
      return ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[character];
    });
  }

  function formatNumber(value) {
    return Number.isFinite(value) ? value.toFixed(2) : "n/a";
  }

  function metricHtml(label, value) {
    return "<span><b>" + label + "</b> " + formatNumber(value) + "</span>";
  }

  function makeChartDiv(id, heightPx) {
    var h = heightPx || 250;
    return '<div id="' + id + '" class="geb-popup__chart" style="height:' + h + 'px"></div>';
  }

  function downloadBlob(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function() { URL.revokeObjectURL(url); }, 500);
  }

  function downloadDataUrl(dataUrl, filename) {
    var a = document.createElement("a");
    a.href = dataUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function unpackStationData(rawPayload, callback) {
    if (!rawPayload) {
      callback(null);
      return;
    }
    try {
      var binaryStr = atob(rawPayload);
      var bytes = new Uint8Array(binaryStr.length);
      for (var i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      var stream = new Response(bytes).body.pipeThrough(new DecompressionStream("gzip"));
      new Response(stream).text().then(function(decompressedText) {
        try {
          callback(JSON.parse(decompressedText));
        } catch (parseErr) {
          callback(null);
        }
      }).catch(function() {
        callback(null);
      });
    } catch (err) {
      callback(null);
    }
  }

  var pendingBundleCallbacks = {};

  function loadStationData(stationId, callback) {
    if (stationChartCache[stationId]) {
      callback(stationChartCache[stationId]);
      return;
    }
    var data = getChartData();
    var stationBundleIndices = data.stations || {};
    var bundles = data.bundles || [];

    var bundleIdx = stationBundleIndices[stationId];
    if (bundleIdx === undefined || bundleIdx === null) {
      callback(null);
      return;
    }

    var rawPayload = bundles[bundleIdx];
    if (!rawPayload) {
      callback(null);
      return;
    }

    if (pendingBundleCallbacks[bundleIdx]) {
      pendingBundleCallbacks[bundleIdx].push(function() {
        callback(stationChartCache[stationId] || null);
      });
      return;
    }
    pendingBundleCallbacks[bundleIdx] = [function() {
      callback(stationChartCache[stationId] || null);
    }];

    unpackStationData(rawPayload, function(bundleData) {
      if (bundleData && typeof bundleData === "object") {
        Object.assign(stationChartCache, bundleData);
        bundles[bundleIdx] = null;
      }
      var cbs = pendingBundleCallbacks[bundleIdx] || [];
      delete pendingBundleCallbacks[bundleIdx];
      cbs.forEach(function(cb) { cb(); });
    });
  }

  function resolveTimeline(spec, startIndex, length) {
    if (!spec) return [];
    if (Array.isArray(spec)) {
      return (startIndex > 0 || length < spec.length)
        ? spec.slice(startIndex, startIndex + length)
        : spec;
    }
    if (typeof spec === "object" && typeof spec.start === "number" && typeof spec.step === "number") {
      var count = length || spec.count || 0;
      var start = spec.start + (startIndex || 0) * spec.step;
      var arr = new Array(count);
      for (var i = 0; i < count; i++) {
        arr[i] = start + i * spec.step;
      }
      return arr;
    }
    return [];
  }

  function decodeVarintDeltas(base64Str, scale) {
    if (!base64Str) return [];
    var s = scale || 100;
    var binaryStr = atob(base64Str);
    var len = binaryStr.length;
    var bytes = new Uint8Array(len);
    for (var k = 0; k < len; k++) {
      bytes[k] = binaryStr.charCodeAt(k);
    }

    var out = [];
    var i = 0;
    var prev = null;

    while (i < len) {
      var code = 0;
      var shift = 0;
      while (true) {
        var b = bytes[i++];
        code |= (b & 0x7F) << shift;
        if ((b & 0x80) === 0) break;
        shift += 7;
      }

      if (code === 0) {
        out.push(null);
        prev = null;
      } else {
        var zz = code - 1;
        var d = (zz >>> 1) ^ -(zz & 1);
        if (prev === null) {
          out.push(d / s);
          prev = d;
        } else {
          prev += d;
          out.push(prev / s);
        }
      }
    }
    return out;
  }

  function finiteNumbers(values, minimumValue) {
    return (values || []).filter(function(value) { return value !== null; }).map(Number).filter(function(value) {
      return Number.isFinite(value) && (minimumValue === undefined || value >= minimumValue);
    });
  }

  function linearRange(values) {
    var numbers = finiteNumbers(values);
    if (!numbers.length) return undefined;
    var minimum = numbers.reduce(function(a, b) { return Math.min(a, b); }, Infinity);
    var maximum = numbers.reduce(function(a, b) { return Math.max(a, b); }, -Infinity);
    if (minimum === maximum) {
      var padding = Math.max(Math.abs(minimum) * 0.05, 1);
      return [minimum - padding, maximum + padding];
    }
    return [minimum, maximum];
  }

  function logRange(values) {
    var numbers = finiteNumbers(values, Number.MIN_VALUE);
    if (!numbers.length) return undefined;
    var minimum = numbers.reduce(function(a, b) { return Math.min(a, b); }, Infinity);
    var maximum = numbers.reduce(function(a, b) { return Math.max(a, b); }, -Infinity);
    if (minimum === maximum) {
      return [Math.log10(minimum) - 0.05, Math.log10(maximum) + 0.05];
    }
    return [Math.log10(minimum), Math.log10(maximum)];
  }

  function dateRange(values) {
    if (!values || !values.length) return undefined;
    var first = new Date(values[0]);
    var last = new Date(values[values.length - 1]);
    if (Number.isFinite(first.getTime()) && Number.isFinite(last.getTime())) {
      return [first, last];
    }
    var times = (values || []).map(function(value) {
      return new Date(value).getTime();
    }).filter(Number.isFinite);
    if (!times.length) return undefined;
    return [new Date(times.reduce(function(a, b) { return Math.min(a, b); }, Infinity)), new Date(times.reduce(function(a, b) { return Math.max(a, b); }, -Infinity))];
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

  function getStationCoordinates(stationId) {
    var all = (window._gebStations || []).concat(window._gebExcludedStations || []);
    for (var i = 0; i < all.length; i++) {
      if (all[i].id === stationId && all[i].coords) {
        return [all[i].coords[1], all[i].coords[0]]; // [lon, lat] for GeoJSON
      }
    }
    return null;
  }

  function updatePlotTheme(gd, theme) {
    if (!gd || !gd.data || !window.Plotly) return;
    var isDark = (theme === "dark");
    var colors = getThemeColors(theme);
    var layoutUpdate = {
      paper_bgcolor: isDark ? "#020617" : "#ffffff",
      plot_bgcolor: isDark ? "#020617" : "#ffffff",
      "font.color": isDark ? "#e2e8f0" : "#0f172a",
      "legend.font.color": isDark ? "#e2e8f0" : "#0f172a",
      "xaxis.gridcolor": isDark ? "#1e293b" : "#f1f5f9",
      "xaxis.zerolinecolor": isDark ? "#334155" : "#cbd5e1",
      "xaxis.tickfont.color": isDark ? "#94a3b8" : "#475569",
      "yaxis.gridcolor": isDark ? "#1e293b" : "#f1f5f9",
      "yaxis.zerolinecolor": isDark ? "#334155" : "#cbd5e1",
      "yaxis.tickfont.color": isDark ? "#94a3b8" : "#475569"
    };
    Plotly.relayout(gd, layoutUpdate);

    var lineColors = [];
    var markerColors = [];
    for (var i = 0; i < gd.data.length; i++) {
      var t = gd.data[i];
      var key = (t.name || "").toLowerCase().split(" ")[0];
      var col = colors[key] || (t.line ? t.line.color : "#f87171");
      lineColors.push(col);
      markerColors.push(col);
    }
    Plotly.restyle(gd, {
      "line.color": lineColors,
      "marker.color": markerColors
    });
  }

  window.addEventListener("gebThemeChange", function(e) {
    var theme = e.detail.theme;
    Object.keys(activePlotlyElements).forEach(function(id) {
      var gd = document.getElementById(id);
      if (gd) {
        updatePlotTheme(gd, theme);
        try { Plotly.Plots.resize(gd); } catch(err) {}
      }
    });
  });

  function exportPlotlySvg(graphDivId, targetTheme, filename) {
    var gd = document.getElementById(graphDivId);
    if (!gd || !gd.data || !window.Plotly) return;

    var isTargetDark = (targetTheme === "dark");
    var targetColors = getThemeColors(targetTheme);
    var currentTheme = getActiveTheme();

    var origTraces = [];
    for (var i = 0; i < gd.data.length; i++) {
      var tr = gd.data[i];
      origTraces.push({
        visible: tr.visible,
        showlegend: (tr.showlegend !== undefined ? tr.showlegend : true),
        lineColor: (tr.line ? tr.line.color : null),
        markerColor: (tr.marker ? tr.marker.color : null)
      });
    }
    var origLayout = {
      paper_bgcolor: gd.layout.paper_bgcolor,
      plot_bgcolor: gd.layout.plot_bgcolor,
      fontColor: gd.layout.font ? gd.layout.font.color : null,
      legendFontColor: (gd.layout.legend && gd.layout.legend.font) ? gd.layout.legend.font.color : null,
      xGrid: gd.layout.xaxis ? gd.layout.xaxis.gridcolor : null,
      xZero: gd.layout.xaxis ? gd.layout.xaxis.zerolinecolor : null,
      xTick: gd.layout.xaxis && gd.layout.xaxis.tickfont ? gd.layout.xaxis.tickfont.color : null,
      yGrid: gd.layout.yaxis ? gd.layout.yaxis.gridcolor : null,
      yZero: gd.layout.yaxis ? gd.layout.yaxis.zerolinecolor : null,
      yTick: gd.layout.yaxis && gd.layout.yaxis.tickfont ? gd.layout.yaxis.tickfont.color : null
    };

    var exportShowLegends = [];
    var exportLineColors = [];
    var exportMarkerColors = [];

    for (var j = 0; j < gd.data.length; j++) {
      var t = gd.data[j];
      var isHidden = (t.visible === "legendonly" || t.visible === false);
      exportShowLegends.push(isHidden ? false : true);

      var key = (t.name || "").toLowerCase().split(" ")[0];
      var color = targetColors[key] || (t.line ? t.line.color : "#f87171");
      exportLineColors.push(color);
      exportMarkerColors.push(color);
    }

    Plotly.restyle(gd, {
      "showlegend": exportShowLegends,
      "line.color": exportLineColors,
      "marker.color": exportMarkerColors
    });
    Plotly.relayout(gd, {
      paper_bgcolor: isTargetDark ? "#020617" : "#ffffff",
      plot_bgcolor: isTargetDark ? "#020617" : "#ffffff",
      "font.color": isTargetDark ? "#e2e8f0" : "#0f172a",
      "legend.font.color": isTargetDark ? "#e2e8f0" : "#0f172a",
      "xaxis.gridcolor": isTargetDark ? "#1e293b" : "#f1f5f9",
      "xaxis.zerolinecolor": isTargetDark ? "#334155" : "#cbd5e1",
      "xaxis.tickfont.color": isTargetDark ? "#94a3b8" : "#475569",
      "yaxis.gridcolor": isTargetDark ? "#1e293b" : "#f1f5f9",
      "yaxis.zerolinecolor": isTargetDark ? "#334155" : "#cbd5e1",
      "yaxis.tickfont.color": isTargetDark ? "#94a3b8" : "#475569"
    });

    Plotly.toImage(gd, {
      format: "svg",
      width: 900,
      height: gd.offsetHeight || 250
    }).then(function(svgUrl) {
      downloadDataUrl(svgUrl, filename + "_" + targetTheme + ".svg");
    }).finally(function() {
      var restShowLegends = [];
      var restLineColors = [];
      var restMarkerColors = [];
      var activeColors = getThemeColors(currentTheme);

      for (var k = 0; k < origTraces.length; k++) {
        restShowLegends.push(origTraces[k].showlegend);
        var kName = (gd.data[k].name || "").toLowerCase().split(" ")[0];
        var kCol = activeColors[kName] || origTraces[k].lineColor;
        restLineColors.push(kCol);
        restMarkerColors.push(kCol);
      }
      Plotly.restyle(gd, {
        "showlegend": restShowLegends,
        "line.color": restLineColors,
        "marker.color": restMarkerColors
      });
      Plotly.relayout(gd, {
        paper_bgcolor: origLayout.paper_bgcolor,
        plot_bgcolor: origLayout.plot_bgcolor,
        "font.color": origLayout.fontColor,
        "legend.font.color": origLayout.legendFontColor,
        "xaxis.gridcolor": origLayout.xGrid,
        "xaxis.zerolinecolor": origLayout.xZero,
        "xaxis.tickfont.color": origLayout.xTick,
        "yaxis.gridcolor": origLayout.yGrid,
        "yaxis.zerolinecolor": origLayout.yZero,
        "yaxis.tickfont.color": origLayout.yTick
      });
    });
  }

  function exportStationCsv(stationId, stationName, timeline, observed, simulated, bankfull) {
    var header = ["date", "observed_m3_s", "simulated_m3_s"];
    if (bankfull) header.push("bankfull_m3_s");
    var lines = [header.join(",")];
    for (var i = 0; i < timeline.length; i++) {
      var d = new Date(timeline[i]).toISOString().slice(0, 10);
      var obs = (observed[i] !== null && observed[i] !== undefined && !isNaN(observed[i])) ? observed[i] : "";
      var sim = (simulated[i] !== null && simulated[i] !== undefined && !isNaN(simulated[i])) ? simulated[i] : "";
      var row = [d, obs, sim];
      if (bankfull) {
        var bk = (bankfull[i] !== null && bankfull[i] !== undefined && !isNaN(bankfull[i])) ? bankfull[i] : "";
        row.push(bk);
      }
      lines.push(row.join(","));
    }
    var blob = new Blob([lines.join("\n")], {type: "text/csv;charset=utf-8;"});
    downloadBlob(blob, stationId + "_timeseries.csv");
  }

  function exportStationGeoJson(stationId, data, timeline, observed, simulated, bankfull) {
    var coords = getStationCoordinates(stationId);
    var geometry = coords ? {
      type: "Point",
      coordinates: coords
    } : null;

    var tsRecords = [];
    if (timeline && timeline.length) {
      for (var i = 0; i < timeline.length; i++) {
        var d = new Date(timeline[i]).toISOString().slice(0, 10);
        var record = {
          date: d,
          observed_m3_s: (observed && observed[i] !== null && observed[i] !== undefined && !isNaN(observed[i])) ? observed[i] : null,
          simulated_m3_s: (simulated && simulated[i] !== null && simulated[i] !== undefined && !isNaN(simulated[i])) ? simulated[i] : null
        };
        if (bankfull) {
          record.bankfull_m3_s = (bankfull[i] !== null && bankfull[i] !== undefined && !isNaN(bankfull[i])) ? bankfull[i] : null;
        }
        tsRecords.push(record);
      }
    }

    var feature = {
      type: "Feature",
      id: stationId,
      geometry: geometry,
      properties: {
        station_id: stationId,
        station_name: data.stationName || stationId,
        frequency: data.frequency || "discharge",
        metrics: data.metrics || {},
        timeseries: tsRecords
      }
    };

    if (data.returnPeriods) {
      var obsRp = [];
      var simRp = [];
      if (data.returnPeriods.observed && data.returnPeriods.observed.returnPeriod) {
        for (var j = 0; j < data.returnPeriods.observed.returnPeriod.length; j++) {
          obsRp.push({
            return_period_years: data.returnPeriods.observed.returnPeriod[j],
            discharge_m3_s: data.returnPeriods.observed.discharge[j]
          });
        }
      }
      if (data.returnPeriods.simulated && data.returnPeriods.simulated.returnPeriod) {
        for (var k = 0; k < data.returnPeriods.simulated.returnPeriod.length; k++) {
          simRp.push({
            return_period_years: data.returnPeriods.simulated.returnPeriod[k],
            discharge_m3_s: data.returnPeriods.simulated.discharge[k]
          });
        }
      }
      feature.properties.return_periods = {
        observed: obsRp,
        simulated: simRp
      };
    }

    var blob = new Blob([JSON.stringify(feature, null, 2)], {type: "application/geo+json;charset=utf-8;"});
    downloadBlob(blob, stationId + ".geojson");
  }

  // Icons for Plotly modebar buttons (512x512)
  var svgCameraIcon = {
    width: 512,
    height: 512,
    path: "M512 144v288c0 26.5-21.5 48-48 48H48c-26.5 0-48-21.5-48-48V144c0-26.5 21.5-48 48-48h88l18.3-36.6C163.5 44.8 178.6 32 195.2 32h121.6c16.6 0 31.7 12.8 38.9 27.4L376 96h88c26.5 0 48 21.5 48 48zM256 400c70.7 0 128-57.3 128-128s-57.3-128-128-128-128 57.3-128 128 57.3 128 128 128zm0-40c-48.5 0-88-39.5-88-88s39.5-88 88-88 88 39.5 88 88-39.5 88-88 88z"
  };

  var tableCsvIcon = {
    width: 512,
    height: 512,
    path: "M64 32C28.7 32 0 60.7 0 96v320c0 35.3 28.7 64 64 64h384c35.3 0 64-28.7 64-64V96c0-35.3-28.7-64-64-64H64zm32 64h128v80H96V96zm160 0h160v80H256V96zm-160 112h128v80H96V208zm160 0h160v80H256V208zm-160 112h128v80H96V320zm160 0h160v80H256V320z"
  };

  var geojsonIcon = {
    width: 512,
    height: 512,
    path: "M256 32C150 32 64 118 64 224c0 135 174 249 181 254 6.7 4.5 15.3 4.5 22 0 7-5 181-119 181-254 0-106-86-192-192-192zm0 256a64 64 0 1 1 0-128 64 64 0 0 1 0 128z"
  };

  function renderCharts(stationId, data) {
    var safeStationId = encodeURIComponent(stationId);
    var currentTheme = getActiveTheme();
    var currentColors = getThemeColors(currentTheme);

    function trace(name, x, y, kind, mode, hoverTemplate, lineOpts) {
      var key = name.toLowerCase().split(" ")[0];
      var col = currentColors[key] || "#f87171";
      var line = {color: col, width: 1.5};
      if (lineOpts) { Object.assign(line, lineOpts); }
      return {
        x: x,
        y: y,
        name: name,
        type: kind,
        mode: mode,
        connectgaps: false,
        hovertemplate: hoverTemplate,
        line: line,
        marker: {color: col, size: 5}
      };
    }

    if (data.timeseries) {
      var dataStore = getChartData();
      var globalTimeline = dataStore.timeline || null;
      var rawTimeline = (data.timeseries.time)
        ? data.timeseries.time
        : (globalTimeline && globalTimeline[data.frequency] ? globalTimeline[data.frequency] : globalTimeline);
      var startIndex = data.timeseries.start || 0;

      var scale = data.timeseries.scale || 100;
      var observed = decodeVarintDeltas(data.timeseries.observed, scale);
      var simulated = decodeVarintDeltas(data.timeseries.simulated, scale);
      var bankfull = data.timeseries.bankfull
        ? decodeVarintDeltas(data.timeseries.bankfull, scale)
        : null;

      var seriesLength = (observed && observed.length)
        || (simulated && simulated.length)
        || 0;
      var timeline = resolveTimeline(rawTimeline, startIndex, seriesLength);
      if (timeline && timeline.length) {
        var timeRange = dateRange(timeline);
        var traces = [
          trace("Observed", timeline, observed, "scatter", "lines", "%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Observed</extra>"),
          trace("Simulated", timeline, simulated, "scatter", "lines", "%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Simulated</extra>")
        ];
        if (bankfull && timeline && timeline.length) {
          traces.push(
            trace("Bankfull", timeline, bankfull, "scatter", "lines", "%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Bankfull</extra>", {dash: "dash"})
          );
        }
        var timeLayout = getPlotlyLayout(currentTheme, {
          height: 250,
          hovermode: "x unified",
          xaxis: {type: "date", range: timeRange},
          yaxis: {title: "Discharge (m3/s)"}
        });
        var timeChartId = "geb-time-" + safeStationId;
        var timeConfig = {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["toImage", "select2d", "lasso2d"],
          modeBarButtonsToAdd: [
            {
              name: "Download plot as SVG",
              icon: svgCameraIcon,
              click: function(gd) {
                exportPlotlySvg(gd.id, getActiveTheme(), safeStationId + "_discharge_timeseries");
              }
            },
            {
              name: "Download table (CSV)",
              icon: tableCsvIcon,
              click: function() {
                exportStationCsv(stationId, data.stationName, timeline, observed, simulated, bankfull);
              }
            },
            {
              name: "Download GeoJSON",
              icon: geojsonIcon,
              click: function() {
                exportStationGeoJson(stationId, data, timeline, observed, simulated, bankfull);
              }
            }
          ]
        };
        Plotly.newPlot(timeChartId, traces, timeLayout, timeConfig);
        activePlotlyElements[timeChartId] = true;
      }
    }

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
      var returnLayout = getPlotlyLayout(currentTheme, {
        height: 250,
        hovermode: "x unified",
        xaxis: {
          type: "log",
          range: returnPeriodRange,
          tickmode: "array",
          tickvals: returnPeriodTicks,
          ticktext: returnPeriodTicks.map(formatTick),
          title: "Return period (years)"
        },
        yaxis: {title: "Discharge (m3/s)"}
      });
      var returnChartId = "geb-return-" + safeStationId;
      var returnConfig = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["toImage", "select2d", "lasso2d"],
        modeBarButtonsToAdd: [
          {
            name: "Download plot as SVG",
            icon: svgCameraIcon,
            click: function(gd) {
              exportPlotlySvg(gd.id, getActiveTheme(), safeStationId + "_return_periods");
            }
          }
        ]
      };
      Plotly.newPlot(returnChartId, [
        trace("Observed", data.returnPeriods.observed.returnPeriod, data.returnPeriods.observed.discharge, "scatter", "lines+markers", "%{x:g}-year<br>%{y:,.0f} m3/s<extra>Observed</extra>"),
        trace("Simulated", data.returnPeriods.simulated.returnPeriod, data.returnPeriods.simulated.discharge, "scatter", "lines+markers", "%{x:g}-year<br>%{y:,.0f} m3/s<extra>Simulated</extra>")
      ], returnLayout, returnConfig);
      activePlotlyElements[returnChartId] = true;
    }
  }

  function renderRiver(el, riverKey, data) {
    var safeRiverId = encodeURIComponent(riverKey);
    var riverId = data.riverId;
    var ts = data.timeseries;
    var times = ts.time;
    var qValues = ts.bankfullDischarge;
    var wValues = ts.width;
    var dValues = ts.depth;
    var cs = data.crossSection || {};

    var lastW = cs.width != null ? cs.width : (wValues && wValues.length > 0 ? wValues[wValues.length - 1] : 10.0);
    var lastD = cs.depth != null ? cs.depth : (dValues && dValues.length > 0 ? dValues[dValues.length - 1] : 2.0);
    var fpWidth = cs.floodplainWidthM != null ? cs.floodplainWidthM : (cs.floodplainWidth != null ? cs.floodplainWidth : 0.0);
    var fpSlope = cs.floodplainSideSlopeMPerM != null ? cs.floodplainSideSlopeMPerM : (cs.floodplainSideSlope != null ? cs.floodplainSideSlope : 2.0);

    var geomSummary = "Channel: " + lastW.toFixed(1) + " m × " + lastD.toFixed(2) + " m" + (fpWidth > 0 ? " · Floodplain bench: " + fpWidth.toFixed(1) + " m" : "") + " · Valley side slope: " + fpSlope.toFixed(1) + " m/m (H:V)";

    el.innerHTML = '<div class="geb-popup__title">River segment ' + escapeHtml(riverId) + '</div>' +
      '<div class="geb-popup__subtitle">Upstream area: ' + formatNumber(data.upstreamAreaKm2) + ' km² · Bankfull hydraulic geometry over time</div>' +
      '<div class="geb-popup__chart-title">Bankfull discharge</div>' +
      makeChartDiv('geb-river-q-' + safeRiverId, 175) +
      '<div class="geb-popup__chart-title">Channel width and depth over time</div>' +
      makeChartDiv('geb-river-geom-' + safeRiverId, 175) +
      '<div class="geb-popup__chart-title">Channel and floodplain cross section (end of run)</div>' +
      '<div style="font-size:11px;margin-bottom:4px;color:#64748b">' + escapeHtml(geomSummary) + '</div>' +
      makeChartDiv('geb-river-cs-' + safeRiverId, 210);

    ensurePlotly(function(loaded) {
      if (loaded === false) {
        el.innerHTML = '<div class="geb-popup__error">Interactive charts require access to cdn.plot.ly.</div>';
        return;
      }
      var currentTheme = getActiveTheme();
      var currentColors = getThemeColors(currentTheme);

      var qConfig = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["toImage", "select2d", "lasso2d"],
        modeBarButtonsToAdd: [
          {
            name: "Download plot as SVG",
            icon: svgCameraIcon,
            click: function(gd) {
              exportPlotlySvg(gd.id, getActiveTheme(), safeRiverId + "_bankfull_discharge");
            }
          }
        ]
      };

      var qTrace = {
        x: times,
        y: qValues,
        name: "Bankfull discharge",
        type: "scatter",
        mode: "lines+markers",
        line: {color: currentColors.bankfull || "#f87171", width: 2},
        marker: {color: currentColors.bankfull || "#f87171", size: 6},
        hovertemplate: "%{x|%Y}<br>Discharge: %{y:,.2f} m³/s<extra>Bankfull discharge</extra>"
      };
      var qLayout = getPlotlyLayout(currentTheme, {
        height: 175,
        hovermode: "x unified",
        xaxis: {type: "date"},
        yaxis: {title: "Discharge (m³/s)"}
      });
      var qId = "geb-river-q-" + safeRiverId;
      Plotly.newPlot(qId, [qTrace], qLayout, qConfig);
      activePlotlyElements[qId] = true;

      var geomConfig = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["toImage", "select2d", "lasso2d"],
        modeBarButtonsToAdd: [
          {
            name: "Download plot as SVG",
            icon: svgCameraIcon,
            click: function(gd) {
              exportPlotlySvg(gd.id, getActiveTheme(), safeRiverId + "_channel_geometry");
            }
          }
        ]
      };

      var wTrace = {
        x: times,
        y: wValues,
        name: "Width (m)",
        type: "scatter",
        mode: "lines+markers",
        line: {color: currentColors.width || "#38bdf8", width: 2},
        marker: {color: currentColors.width || "#38bdf8", size: 6},
        hovertemplate: "%{x|%Y}<br>Width: %{y:,.2f} m<extra>Width</extra>"
      };
      var dTrace = {
        x: times,
        y: dValues,
        name: "Depth (m)",
        type: "scatter",
        mode: "lines+markers",
        line: {color: currentColors.depth || "#4ade80", width: 2},
        marker: {color: currentColors.depth || "#4ade80", size: 6},
        hovertemplate: "%{x|%Y}<br>Depth: %{y:,.2f} m<extra>Depth</extra>"
      };
      var geomLayout = getPlotlyLayout(currentTheme, {
        height: 175,
        hovermode: "x unified",
        xaxis: {type: "date"},
        yaxis: {title: "Meters (m)"}
      });
      var geomId = "geb-river-geom-" + safeRiverId;
      Plotly.newPlot(geomId, [wTrace, dTrace], geomLayout, geomConfig);
      activePlotlyElements[geomId] = true;

      // Cross section coordinates: channel bed -> flat floodplain benches -> valley side slopes
      var halfW = lastW * 0.5;
      var halfBench = (lastW + fpWidth) * 0.5;
      var wallHeight = Math.max(lastD * 0.5, 1.0);
      var wallRun = fpSlope * wallHeight;

      var xCoords = [];
      var yCoords = [];

      // Left valley wall
      xCoords.push(-(halfBench + wallRun));
      yCoords.push(lastD + wallHeight);
      xCoords.push(-halfBench);
      yCoords.push(lastD);

      // Left flat floodplain bench
      if (fpWidth > 0.0) {
        xCoords.push(-halfW);
        yCoords.push(lastD);
      }

      // Parabolic channel bed (21 points)
      var nPts = 20;
      for (var p = 0; p <= nPts; p++) {
        var u = -1.0 + (2.0 * p / nPts);
        xCoords.push(u * halfW);
        yCoords.push(lastD * u * u);
      }

      // Right flat floodplain bench
      if (fpWidth > 0.0) {
        xCoords.push(halfW);
        yCoords.push(lastD);
      }

      // Right valley wall
      xCoords.push(halfBench);
      yCoords.push(lastD);
      xCoords.push(halfBench + wallRun);
      yCoords.push(lastD + wallHeight);

      var csConfig = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["toImage", "select2d", "lasso2d"],
        modeBarButtonsToAdd: [
          {
            name: "Download plot as SVG",
            icon: svgCameraIcon,
            click: function(gd) {
              exportPlotlySvg(gd.id, getActiveTheme(), safeRiverId + "_cross_section");
            }
          }
        ]
      };

      var groundTrace = {
        x: xCoords,
        y: yCoords,
        name: "Ground profile",
        type: "scatter",
        mode: "lines",
        line: {color: currentTheme === "dark" ? "#f59e0b" : "#b45309", width: 2.5},
        hovertemplate: "Distance: %{x:,.2f} m<br>Elevation: %{y:,.2f} m<extra>Ground profile</extra>"
      };

      var bankfullLine = {
        x: [-halfW, halfW],
        y: [lastD, lastD],
        name: "Bankfull stage (" + lastD.toFixed(2) + " m)",
        type: "scatter",
        mode: "lines",
        line: {color: "#94a3b8", width: 1.5, dash: "dot"},
        hovertemplate: "Bankfull stage: %{y:,.2f} m<extra></extra>"
      };

      var csLayout = getPlotlyLayout(currentTheme, {
        height: 210,
        hovermode: "closest",
        xaxis: {
          title: "Distance across river (m)",
          zeroline: true,
          zerolinecolor: currentTheme === "dark" ? "#334155" : "#cbd5e1"
        },
        yaxis: {
          title: "Height above river bed (m)",
          rangemode: "tozero"
        }
      });

      var csId = "geb-river-cs-" + safeRiverId;
      Plotly.newPlot(csId, [groundTrace, bankfullLine], csLayout, csConfig);
      activePlotlyElements[csId] = true;

      setTimeout(function() {
        Plotly.Plots.resize(qId);
        Plotly.Plots.resize(geomId);
        Plotly.Plots.resize(csId);
      }, 60);
    });
  }

  function renderStation(el, stationId) {
    if (el.dataset.rendered === "true") return;
    el.dataset.rendered = "true";
    loadStationData(stationId, function(data) {
      if (!data) {
        el.innerHTML = '<div class="geb-popup__error">No interactive chart data is available.</div>';
        return;
      }
      if (data.type === "river") {
        renderRiver(el, stationId, data);
        return;
      }
      var metrics = data.metrics || {};
      var safeStationId = encodeURIComponent(stationId);

      var returnPeriodsHtml = data.returnPeriods
        ? '<div class="geb-popup__chart-title">Return periods</div>' + makeChartDiv("geb-return-" + safeStationId, 250)
        : "";

      var timeseriesHtml = data.timeseries
        ? '<div class="geb-popup__chart-title">Discharge time series</div>' + makeChartDiv("geb-time-" + safeStationId, 250)
        : "";

      el.innerHTML = '<div class="geb-popup__title">' + escapeHtml(data.stationName || stationId) + '</div>' +
        '<div class="geb-popup__subtitle">Station ' + escapeHtml(stationId) + ' · ' + escapeHtml(data.frequency || "discharge") + '</div>' +
        '<div class="geb-popup__metrics">' + metricHtml("KGE", metrics.KGE) + metricHtml("mKGE", metrics.KGE_modified) +
        metricHtml("r", metrics.KGE_correlation) + metricHtml("β", metrics.KGE_bias_ratio) +
        metricHtml("α", metrics.KGE_variability_ratio) + metricHtml("NSE", metrics.NSE) +
        metricHtml("r²", metrics.R2) + metricHtml("RMSE", metrics.RMSE) +
        metricHtml("RRMSE", metrics.RRMSE) + metricHtml("Area ratio", metrics.upstreamAreaRatio) +
        metricHtml("Fixed UTC offset (h)", metrics.timezoneUtcOffset) +
        ((data.timeseries && data.timeseries.bankfullDischarge != null) ? metricHtml("Bankfull (m³/s)", data.timeseries.bankfullDischarge) : "") + '</div>' +
        returnPeriodsHtml +
        timeseriesHtml;

      ensurePlotly(function(loaded) {
        if (loaded === false) {
          el.innerHTML = '<div class="geb-popup__error">Interactive charts require access to cdn.plot.ly.</div>';
          return;
        }
        renderCharts(stationId, data);
        setTimeout(function() {
          if (data.timeseries) Plotly.Plots.resize("geb-time-" + safeStationId);
          if (data.returnPeriods) Plotly.Plots.resize("geb-return-" + safeStationId);
        }, 60);
      });
    });
  }

  var style = document.createElement("style");
  style.textContent =
    ".geb-popup{width:750px;max-width:82vw;box-sizing:border-box;overflow-x:hidden;font-family:Inter,system-ui,-apple-system,sans-serif}" +
    ".geb-popup:empty:before{content:\"Loading interactive charts...\";display:block;padding:12px;color:#64748b;font-style:italic}" +
    ".geb-popup__title{font-size:17px;font-weight:750;letter-spacing:-0.2px}" +
    ".geb-popup__subtitle{font-size:11px;margin-bottom:8px}" +
    ".geb-popup__metrics{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px}" +
    ".geb-popup__metrics span{border-radius:5px;padding:3px 7px;font-size:11px}" +
    ".geb-popup__chart{width:100%;border-radius:8px;margin-bottom:10px;box-sizing:border-box;overflow:hidden}" +
    ".geb-popup__chart-title{font-weight:700;font-size:12px;margin:8px 0 3px}" +
    ".geb-popup__error{color:#dc2626;padding:18px;font-weight:500}" +
    "[data-geb-theme=\"dark\"] .leaflet-popup-content-wrapper,[data-geb-theme=\"dark\"] .leaflet-popup-tip{background:#0b1120!important;color:#f8fafc!important;box-shadow:0 10px 35px rgba(0,0,0,0.6)!important;border:1px solid #334155!important}" +
    "[data-geb-theme=\"dark\"] .leaflet-popup-close-button{color:#94a3b8!important}" +
    "[data-geb-theme=\"dark\"] .geb-popup__title{color:#f8fafc}" +
    "[data-geb-theme=\"dark\"] .geb-popup__subtitle{color:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-popup__metrics span{background:#1e293b;border:1px solid #334155;color:#e2e8f0}" +
    "[data-geb-theme=\"dark\"] .geb-popup__chart{background:#020617;border:1px solid #1e293b}" +
    "[data-geb-theme=\"dark\"] .geb-popup__chart-title{color:#cbd5e1}" +
    "[data-geb-theme=\"dark\"] .js-plotly-plot .plotly .modebar-btn path{fill:#ffffff!important}" +
    "[data-geb-theme=\"dark\"] .js-plotly-plot .plotly .modebar-btn:hover path{fill:#38bdf8!important}" +
    "[data-geb-theme=\"light\"] .leaflet-popup-content-wrapper,[data-geb-theme=\"light\"] .leaflet-popup-tip{background:#ffffff!important;color:#0f172a!important;box-shadow:0 10px 30px rgba(15,23,42,0.18)!important;border:1px solid #cbd5e1!important}" +
    "[data-geb-theme=\"light\"] .leaflet-popup-close-button{color:#475569!important}" +
    "[data-geb-theme=\"light\"] .geb-popup__title{color:#0f172a}" +
    "[data-geb-theme=\"light\"] .geb-popup__subtitle{color:#475569}" +
    "[data-geb-theme=\"light\"] .geb-popup__metrics span{background:#f1f5f9;border:1px solid #cbd5e1;color:#0f172a}" +
    "[data-geb-theme=\"light\"] .geb-popup__chart{background:#ffffff;border:1px solid #e2e8f0}" +
    "[data-geb-theme=\"light\"] .geb-popup__chart-title{color:#334155}";
  document.head.appendChild(style);

  function handlePopupOpen(e) {
    var container = (e.popup && e.popup.getElement ? e.popup.getElement() : null) ||
                    (e.popup && e.popup._container) ||
                    (e.popup && e.popup.getContent && e.popup.getContent().nodeType ? e.popup.getContent() : null) ||
                    document.querySelector(".leaflet-popup-pane");
    if (!container) return;
    var el = container.querySelector("[data-station-id]");
    if (!el) return;
    var sid = el.getAttribute("data-station-id");
    renderStation(el, sid);
  }

  {{this._parent.get_name()}}.on("popupopen", function(e) {
    handlePopupOpen(e);
    if (e.popup && e.popup.on) {
      e.popup.off("contentupdate", handlePopupOpen).on("contentupdate", handlePopupOpen);
    }
  });
})();
{%- endmacro -%}
