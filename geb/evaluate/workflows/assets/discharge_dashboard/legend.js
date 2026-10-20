(function(){
  var gebMetricLegendConfigs = {{ this.data | script_json }};
  var map = {{this._parent.get_name()}};
  var configByLayer = {};
  var layerByName = {};
  var activeConfig = null;
  gebMetricLegendConfigs.forEach(function(config) {
    var layer = window[config.layer];
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
