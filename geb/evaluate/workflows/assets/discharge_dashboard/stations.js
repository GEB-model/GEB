(function() {
  var macroData = {{ this.data | script_json }};
  var payload = macroData.payload;
  var upstreamLayer = macroData.upstreamLayer || '';
  var availabilityLayer = macroData.availabilityLayer || '';
  var caravanAvailableColor = macroData.caravanAvailableColor || '#22c55e';
  var caravanUnavailableColor = macroData.caravanUnavailableColor || '#94a3b8';

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

  decompress(payload).then(function(stations) {
    var avLayer = availabilityLayer ? window[availabilityLayer] : null;

    stations.forEach(function(station) {
      station._markers = [];
      var escapedId = escapeHtml(station.id);
      var escapedName = escapeHtml(station.name);
      var baseTooltip = escapedId + ': ' + escapedName;
      if (station.tz !== null && station.tz !== undefined) {
        baseTooltip += '<br><span data-geb-utc-offset-hours="' + station.tz + '"></span> fixed';
      }
      if (station.ex) {
        baseTooltip += '<br>Diagnostic only — excluded from evaluation';
      }

      var popupHtml = '';
      if (station.ex) {
        popupHtml += '<div style="color:#b91c1c;padding:8px"><b>Diagnostic only — excluded from evaluation.</b><br>' +
          escapeHtml(station.ex) + '</div>';
      }
      popupHtml += '<div class="geb-popup" data-station-id="' + escapedId + '"></div>';

      // Metric & upstream layers
      if (station.m) {
        for (var layerName in station.m) {
          var layer = window[layerName];
          if (!layer) continue;
          var radius = (layerName === upstreamLayer) ? 10 : station.r;
          var marker = L.circleMarker(station.coords, {
            radius: radius,
            color: 'black',
            fill: true,
            fillColor: station.m[layerName],
            fillOpacity: 0.9,
            zIndexOffset: 1000
          });
          marker.bindPopup(popupHtml, {maxWidth: 800});
          marker.bindTooltip(baseTooltip);
          marker.addTo(layer);
          station._markers.push(marker);
        }
      }

      // Availability layer
      if (avLayer && station.av !== null && station.av !== undefined) {
        var isAv = !!station.av;
        var avColor = isAv ? caravanAvailableColor : caravanUnavailableColor;
        var avLabel = isAv ? 'available' : 'not available';
        var avMarker = L.circleMarker(station.coords, {
          radius: station.r,
          color: 'black',
          fill: true,
          fillColor: avColor,
          fillOpacity: 0.9,
          zIndexOffset: 1000
        });
        avMarker.bindPopup(popupHtml, {maxWidth: 800});
        avMarker.bindTooltip(baseTooltip + '<br>GRDC-Caravan data: ' + avLabel);
        avMarker.addTo(avLayer);
        station._markers.push(avMarker);
      }

      // Characteristic layers
      if (station.ch) {
        for (var chLayerName in station.ch) {
          var chLayer = window[chLayerName];
          if (!chLayer) continue;
          var chData = station.ch[chLayerName];
          var chMarker = L.circleMarker(station.coords, {
            radius: station.r,
            color: 'black',
            fill: true,
            fillColor: chData[0],
            fillOpacity: 0.9,
            zIndexOffset: 1000
          });
          chMarker.bindPopup(popupHtml, {maxWidth: 800});
          chMarker.bindTooltip(baseTooltip + '<br>' + chData[1]);
          chMarker.addTo(chLayer);
          station._markers.push(chMarker);
        }
      }
    });

    window._gebStations = stations;
    window.dispatchEvent(new CustomEvent('gebStationsReady', {detail: stations}));
  });
})();
