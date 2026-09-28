(function(){
  var macroData = {{ this.data | script_json }};
  var stationChartFiles = (macroData && macroData.stations) ? macroData.stations : macroData;
  var globalTimeline = (macroData && macroData.timeline) ? macroData.timeline : null;
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

  function unpackStationData(rawPayload, callback) {
    if (!rawPayload) {
      callback(null);
      return;
    }
    if (typeof rawPayload === 'object') {
      callback(rawPayload);
      return;
    }
    try {
      var binaryStr = atob(rawPayload);
      var bytes = new Uint8Array(binaryStr.length);
      for (var i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      var stream = new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'));
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
    var chartFile = stationChartFiles[stationId];
    if (!chartFile) {
      callback(null);
      return;
    }
    if (pendingBundleCallbacks[chartFile]) {
      pendingBundleCallbacks[chartFile].push(function() {
        callback(stationChartCache[stationId] || null);
      });
      return;
    }
    pendingBundleCallbacks[chartFile] = [function() {
      callback(stationChartCache[stationId] || null);
    }];

    var script = document.createElement('script');
    script.src = chartFile;
    script.onload = function() {
      var rawBundle = window._gebStationChartBundle;
      delete window._gebStationChartBundle;
      var rawSingle = window._gebStationChartPayload;
      delete window._gebStationChartPayload;
      if (script.remove) script.remove();

      function notifyPending() {
        var cbs = pendingBundleCallbacks[chartFile] || [];
        delete pendingBundleCallbacks[chartFile];
        cbs.forEach(function(cb) { cb(); });
      }

      if (rawBundle !== undefined) {
        unpackStationData(rawBundle, function(bundleData) {
          if (bundleData && typeof bundleData === 'object') {
            Object.assign(stationChartCache, bundleData);
          }
          notifyPending();
        });
      } else if (rawSingle !== undefined) {
        unpackStationData(rawSingle, function(singleData) {
          if (singleData && typeof singleData === 'object') {
            stationChartCache[stationId] = singleData;
          }
          notifyPending();
        });
      } else {
        notifyPending();
      }
    };
    script.onerror = function() {
      if (script.remove) script.remove();
      var cbs = pendingBundleCallbacks[chartFile] || [];
      delete pendingBundleCallbacks[chartFile];
      cbs.forEach(function(cb) { cb(); });
    };
    document.head.appendChild(script);
  }

  function resolveTimeline(spec, startIndex, length) {
    if (!spec) return [];
    if (Array.isArray(spec)) {
      return (startIndex > 0 || length < spec.length)
        ? spec.slice(startIndex, startIndex + length)
        : spec;
    }
    if (typeof spec === 'object' && typeof spec.start === 'number' && typeof spec.step === 'number') {
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

  function unscale(values, scale) {
    if (!values) return [];
    if (!scale || scale === 1) return values;
    return values.map(function(v) { return v !== null ? v / scale : null; });
  }

  function decodeDeltas(deltas, scale) {
    if (!deltas) return [];
    var s = scale || 100;
    var out = [];
    var prev = null;
    for (var i = 0; i < deltas.length; i++) {
      var d = deltas[i];
      if (d === null) {
        out.push(null);
        prev = null;
      } else if (prev === null) {
        out.push(d / s);
        prev = d;
      } else {
        prev += d;
        out.push(prev / s);
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

  function renderCharts(stationId, data) {
    var safeStationId = encodeURIComponent(stationId);
    var common = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d']};
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
    if (data.timeseries) {
      var rawTimeline = (data.timeseries.time)
        ? data.timeseries.time
        : (globalTimeline && globalTimeline[data.frequency] ? globalTimeline[data.frequency] : globalTimeline);
      var startIndex = data.timeseries.start || 0;
      var seriesLength = (data.timeseries.observed && data.timeseries.observed.length)
        || (data.timeseries.simulated && data.timeseries.simulated.length)
        || 0;
      var timeline = resolveTimeline(rawTimeline, startIndex, seriesLength);
      if (timeline && timeline.length) {

        var scale = data.timeseries.scale || 100;
        var isDelta = Boolean(data.timeseries.deltas);
        var observed = isDelta
          ? decodeDeltas(data.timeseries.observed, scale)
          : unscale(data.timeseries.observed, scale);
        var simulated = isDelta
          ? decodeDeltas(data.timeseries.simulated, scale)
          : unscale(data.timeseries.simulated, scale);

        var timeRange = dateRange(timeline);
        Plotly.newPlot('geb-time-' + safeStationId, [
          trace('Observed', timeline, observed, 'scatter', 'lines', '%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Observed</extra>'),
          trace('Simulated', timeline, simulated, 'scatter', 'lines', '%{x|%b %Y}<br>%{y:,.0f} m3/s<extra>Simulated</extra>')
        ], Object.assign({}, layoutBase, {hovermode: 'x unified', xaxis: Object.assign({}, layoutBase.xaxis, {type: 'date', range: timeRange}), yaxis: Object.assign({}, layoutBase.yaxis, {title: 'Discharge (m3/s)'})}), common);
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
      (data.timeseries ? '<div class="geb-popup__chart-title">Discharge time series</div>' + makeChartDiv('geb-time-' + safeStationId) : '');
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
  style.textContent = '.geb-popup{width:820px;max-width:86vw;color:#0f172a;font-family:Inter,system-ui,sans-serif}.geb-popup:empty:before{content:"Loading interactive charts...";display:block;padding:12px;color:#64748b;font-style:italic}.geb-popup__title{color:#0f172a;font-size:18px;font-weight:750}.geb-popup__subtitle{color:#475569;font-size:12px;margin-bottom:8px}.geb-popup__metrics{display:flex;gap:12px;flex-wrap:wrap;margin:6px 0 10px}.geb-popup__metrics span{background:#111827;border:1px solid #263244;border-radius:6px;color:#e2e8f0;padding:5px 8px}.geb-popup__chart{height:260px;background:#020617;border:1px solid #263244;border-radius:8px;margin-bottom:10px}.geb-popup__chart-title{color:#334155;font-weight:700;font-size:13px;margin:10px 0 4px}.geb-popup__error{color:#b91c1c;padding:18px}.geb-popup img{width:100%;height:auto;display:block}';
  document.head.appendChild(style);

  function handlePopupOpen(e) {
    var container = (e.popup && e.popup.getElement ? e.popup.getElement() : null) ||
                    (e.popup && e.popup._container) ||
                    (e.popup && e.popup.getContent && e.popup.getContent().nodeType ? e.popup.getContent() : null) ||
                    document.querySelector('.leaflet-popup-pane');
    if (!container) return;
    var el = container.querySelector('[data-station-id]');
    if (!el) return;
    var sid = el.getAttribute('data-station-id');
    renderStation(el, sid);
  }

  {{this._parent.get_name()}}.on('popupopen', function(e) {
    handlePopupOpen(e);
    if (e.popup && e.popup.on) {
      e.popup.off('contentupdate', handlePopupOpen).on('contentupdate', handlePopupOpen);
    }
  });
})();
