(function() {
  var layerGroup = {{ this.data.layer }};
  var payload = {{ this.data.payload | script_json }};

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, function(c) {
      return ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[c];
    });
  }

  async function decompress(base64Str) {
    var binary = atob(base64Str);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    var stream = new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'));
    return await new Response(stream).json();
  }

  decompress(payload).then(function(geoJsonData) {
    var features = geoJsonData.features || [];

    // Compute min and max width across all river segments
    var minWidth = Infinity;
    var maxWidth = -Infinity;
    for (var i = 0; i < features.length; i++) {
      var w = features[i].properties && features[i].properties.width_m;
      if (w != null && !isNaN(w) && w > 0) {
        if (w < minWidth) minWidth = w;
        if (w > maxWidth) maxWidth = w;
      }
    }
    var hasWidth = isFinite(minWidth) && isFinite(maxWidth);
    var widthDiff = (hasWidth && maxWidth > minWidth) ? (maxWidth - minWidth) : 1;

    // Compute min and max depth across all river segments
    var minDepth = Infinity;
    var maxDepth = -Infinity;
    for (var j = 0; j < features.length; j++) {
      var d = features[j].properties && features[j].properties.depth_m;
      if (d != null && !isNaN(d) && d > 0) {
        if (d < minDepth) minDepth = d;
        if (d > maxDepth) maxDepth = d;
      }
    }
    var hasDepth = isFinite(minDepth) && isFinite(maxDepth);
    var depthDiff = (hasDepth && maxDepth > minDepth) ? (maxDepth - minDepth) : 1;

    var minWeight = 2;
    var maxWeight = 6;

    // Linear scaling between min and max river width
    function getWeight(feature) {
      if (!hasWidth) return minWeight;
      var w = feature && feature.properties && feature.properties.width_m;
      if (w == null || isNaN(w) || w <= 0) return minWeight;
      var norm = (w - minWidth) / widthDiff;
      norm = Math.max(0, Math.min(1, norm));
      return minWeight + norm * (maxWeight - minWeight);
    }

    // Color gradient from light blue (shallow) to dark blue (deep)
    // Light sky blue (#7dd3fc): [125, 211, 252]
    // Dark navy blue (#1e3a8a): [30, 58, 138]
    var shallowR = 125, shallowG = 211, shallowB = 252;
    var deepR = 30, deepG = 58, deepB = 138;

    var noDepthColor = "#4b5563"; // Dark grey for rivers without depth data

    function getColor(feature) {
      if (!hasDepth) return noDepthColor;
      var d = feature && feature.properties && feature.properties.depth_m;
      if (d == null || isNaN(d) || d <= 0) return noDepthColor;
      var norm = (d - minDepth) / depthDiff;
      norm = Math.max(0, Math.min(1, norm));
      var r = Math.round(shallowR + norm * (deepR - shallowR));
      var g = Math.round(shallowG + norm * (deepG - shallowG));
      var b = Math.round(shallowB + norm * (deepB - shallowB));
      return "rgb(" + r + "," + g + "," + b + ")";
    }

    var riverLayer = L.geoJson(geoJsonData, {
      style: function(feature) {
        return {
          color: getColor(feature),
          weight: getWeight(feature),
          opacity: 0.85
        };
      },
      onEachFeature: function(feature, layer) {
        var properties = feature.properties || {};
        var riverId = properties.river_id || '';
        var tooltip = document.createElement('div');
        var uparea = properties.uparea_m2 ? (properties.uparea_m2 / 1e6).toLocaleString() : 'N/A';
        var widthText = properties.width_m != null ? ' · Width: ' + properties.width_m + ' m' : '';
        var depthText = properties.depth_m != null ? ' · Depth: ' + properties.depth_m + ' m' : '';
        tooltip.textContent = 'River ID: ' + riverId + ' · Upstream area: ' + uparea + ' km²' + widthText + depthText + ' (click for charts)';
        layer.bindTooltip(tooltip, {sticky: true});

        var popupHtml = '<div class="geb-popup" data-station-id="river_' + escapeHtml(riverId) + '" data-river-id="' + escapeHtml(riverId) + '"></div>';
        layer.bindPopup(popupHtml, {maxWidth: 820});

        var baseWeight = getWeight(feature);
        layer.on({
          mouseover: function(e) {
            e.target.setStyle({weight: Math.max(baseWeight + 3, 5), opacity: 1, color: '#f59e0b'});
          },
          mouseout: function(e) {
            riverLayer.resetStyle(e.target);
          }
        });
      }
    });
    riverLayer.addTo(layerGroup);
  });
})();
