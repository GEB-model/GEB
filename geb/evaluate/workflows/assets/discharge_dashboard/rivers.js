(function() {
  var layerGroup = {{ this.data.layer }};
  var payload = {{ this.data.payload | script_json }};

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
    var riverLayer = L.geoJson(geoJsonData, {
      style: function(_feature) {
        return {
          color: "#4A90D9",
          weight: 1.2,
          opacity: 0.65
        };
      },
      onEachFeature: function(feature, layer) {
        var properties = feature.properties || {};
        var tooltip = document.createElement('div');
        var uparea = properties.uparea_m2 ? (properties.uparea_m2 / 1e6).toLocaleString() : 'N/A';
        tooltip.textContent = 'River ID: ' + (properties.river_id || '') + ' · Upstream area: ' + uparea + ' km²';
        layer.bindTooltip(tooltip, {sticky: true});
        layer.on({
          mouseover: function(e) {
            e.target.setStyle({weight: 4, opacity: 1});
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
