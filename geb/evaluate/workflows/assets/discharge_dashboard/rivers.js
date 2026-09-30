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
    var minUparea = Infinity;
    var maxUparea = -Infinity;
    for (var i = 0; i < features.length; i++) {
      var up = (features[i].properties && features[i].properties.uparea_m2) || 0;
      if (up > 0) {
        if (up < minUparea) minUparea = up;
        if (up > maxUparea) maxUparea = up;
      }
    }
    if (!isFinite(minUparea) || minUparea <= 0) minUparea = 1;
    if (!isFinite(maxUparea) || maxUparea <= minUparea) maxUparea = minUparea + 1;

    var logMin = Math.log(minUparea);
    var logMax = Math.log(maxUparea);
    var logDiff = (logMax > logMin) ? (logMax - logMin) : 1;

    var minWeight = 2;
    var maxWeight = 6;

    function getWeight(feature) {
      var up = (feature && feature.properties && feature.properties.uparea_m2) || 0;
      if (up <= 0) return minWeight;
      var logVal = Math.log(Math.max(up, minUparea));
      var norm = (logVal - logMin) / logDiff;
      norm = Math.max(0, Math.min(1, norm));
      return minWeight + norm * (maxWeight - minWeight);
    }

    var riverLayer = L.geoJson(geoJsonData, {
      style: function(feature) {
        return {
          color: "#4A90D9",
          weight: getWeight(feature),
          opacity: 0.75
        };
      },
      onEachFeature: function(feature, layer) {
        var properties = feature.properties || {};
        var riverId = properties.river_id || '';
        var tooltip = document.createElement('div');
        var uparea = properties.uparea_m2 ? (properties.uparea_m2 / 1e6).toLocaleString() : 'N/A';
        tooltip.textContent = 'River ID: ' + riverId + ' · Upstream area: ' + uparea + ' km² (click for charts)';
        layer.bindTooltip(tooltip, {sticky: true});

        var popupHtml = '<div class="geb-popup" data-station-id="river_' + escapeHtml(riverId) + '" data-river-id="' + escapeHtml(riverId) + '"></div>';
        layer.bindPopup(popupHtml, {maxWidth: 820});

        var baseWeight = getWeight(feature);
        layer.on({
          mouseover: function(e) {
            e.target.setStyle({weight: Math.max(baseWeight + 3, 5), opacity: 1, color: '#38bdf8'});
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
