(function() {
  var macroData = {{ this.data | script_json }};
  var payload = macroData.payload;
  var upstreamLayer = macroData.upstreamLayer || '';
  var availabilityLayer = macroData.availabilityLayer || '';
  var caravanAvailableColor = macroData.caravanAvailableColor || '#22c55e';
  var caravanUnavailableColor = macroData.caravanUnavailableColor || '#94a3b8';
  var map = {{ this._parent.get_name() }};

  if (!map.getPane("stationsPane")) {
    var sp = map.createPane("stationsPane");
    sp.style.zIndex = "620";
    sp.style.pointerEvents = "auto";
  }
  if (!map._paneRenderers) {
    map._paneRenderers = {};
  }
  if (!map._paneRenderers["stationsPane"]) {
    map._paneRenderers["stationsPane"] = L.svg({pane: "stationsPane"}).addTo(map);
  }

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

  var isDecompressed = false;
  var pendingLayers = [];
  var populatedLayers = {};
  var stampToLayerName = {};
  var stationsData = [];

  function populateLayer(layerName) {
    if (!isDecompressed) {
      if (pendingLayers.indexOf(layerName) === -1) pendingLayers.push(layerName);
      return;
    }
    if (populatedLayers[layerName]) return;
    var layer = window[layerName];
    if (!layer) return;
    populatedLayers[layerName] = true;

    var isCharLayer = false;
    for (var k = 0; k < stationsData.length; k++) {
      if (stationsData[k].ch && stationsData[k].ch[layerName]) {
        isCharLayer = true;
        break;
      }
    }

    stationsData.forEach(function(station) {
      var marker = null;
      if (station.m && station.m[layerName] !== undefined) {
        var radius = (layerName === upstreamLayer) ? 10 : station.r;
        var fillColor = (window._gebCurrentFrequency === 'hourly' && station.mh && station.mh[layerName])
          ? station.mh[layerName]
          : (window._gebCurrentFrequency === 'daily' && station.md && station.md[layerName])
            ? station.md[layerName]
            : station.m[layerName];
        marker = L.circleMarker(station.coords, {
          pane: 'stationsPane',
          radius: radius,
          color: 'black',
          weight: 1.2,
          fill: true,
          fillColor: fillColor,
          fillOpacity: 0.9,
          zIndexOffset: 1000
        });
        marker.bindPopup(station._popupHtml, {maxWidth: 800});
        marker.bindTooltip(station._baseTooltip);
      } else if (layerName === availabilityLayer && station.av !== null && station.av !== undefined) {
        var isAv = !!station.av;
        var avColor = isAv ? caravanAvailableColor : caravanUnavailableColor;
        var avLabel = isAv ? 'available' : 'not available';
        marker = L.circleMarker(station.coords, {
          pane: 'stationsPane',
          radius: station.r,
          color: 'black',
          weight: 1.2,
          fill: true,
          fillColor: avColor,
          fillOpacity: 0.9,
          zIndexOffset: 1000
        });
        marker.bindPopup(station._popupHtml, {maxWidth: 800});
        marker.bindTooltip(station._baseTooltip + '<br>GRDC-Caravan data: ' + avLabel);
      } else if (station.ch && station.ch[layerName]) {
        var chData = station.ch[layerName];
        marker = L.circleMarker(station.coords, {
          pane: 'stationsPane',
          radius: station.r,
          color: 'black',
          weight: 1.2,
          fill: true,
          fillColor: chData[0],
          fillOpacity: 0.9,
          zIndexOffset: 1000
        });
        marker.bindPopup(station._popupHtml, {maxWidth: 800});
        marker.bindTooltip(station._baseTooltip + '<br>' + chData[1]);
      } else if (isCharLayer) {
        marker = L.circleMarker(station.coords, {
          pane: 'stationsPane',
          radius: station.r,
          color: 'black',
          weight: 1,
          fill: true,
          fillColor: '#94a3b8',
          fillOpacity: 0.45,
          zIndexOffset: 1000
        });
        marker.bindPopup(station._popupHtml, {maxWidth: 800});
        marker.bindTooltip(station._baseTooltip + '<br>Missing characteristic data');
      }

      if (marker) {
        var isVis = (station._searchVisible !== false) && (station._isVisible !== false);
        if (!isVis) {
          marker.setStyle({opacity: 0, fillOpacity: 0});
          marker.options.interactive = false;
          if (marker.getElement()) {
            marker.getElement().style.pointerEvents = 'none';
          }
        }
        marker.addTo(layer);
        station._markers.push(marker);
      }
    });
    if (layer && layer.bringToFront) {
      try { layer.bringToFront(); } catch(e) {}
    }
  }

  window._gebPopulateStationLayer = populateLayer;

  decompress(payload).then(function(stations) {
    stationsData = stations;
    isDecompressed = true;

    stations.forEach(function(station) {
      station._markers = [];
      var escapedId = escapeHtml(station.id);
      var escapedName = escapeHtml(station.name);
      var baseTooltip = escapedId + ': ' + escapedName;
      if (station.fq) {
        baseTooltip += ' · ' + escapeHtml(station.fq);
      }
      if (station.tz !== null && station.tz !== undefined) {
        baseTooltip += '<br><span data-geb-utc-offset-hours="' + station.tz + '"></span> fixed';
      }
      if (station.ex) {
        baseTooltip += '<br>Diagnostic only — excluded from evaluation';
      }
      station._baseTooltip = baseTooltip;

      var popupHtml = '';
      if (station.ex) {
        popupHtml += '<div style="color:#b91c1c;padding:8px"><b>Diagnostic only — excluded from evaluation.</b><br>' +
          escapeHtml(station.ex) + '</div>';
      }
      popupHtml += '<div class="geb-popup" data-station-id="' + escapedId + '"></div>';
      station._popupHtml = popupHtml;

      if (station.m) {
        Object.keys(station.m).forEach(function(name) {
          var layerObj = window[name];
          if (layerObj) stampToLayerName[L.stamp(layerObj)] = name;
        });
      }
      if (station.ch) {
        Object.keys(station.ch).forEach(function(name) {
          var layerObj = window[name];
          if (layerObj) stampToLayerName[L.stamp(layerObj)] = name;
        });
      }
    });

    if (availabilityLayer) {
      var avLayerObj = window[availabilityLayer];
      if (avLayerObj) stampToLayerName[L.stamp(avLayerObj)] = availabilityLayer;
    }

    pendingLayers.forEach(function(name) {
      populateLayer(name);
    });
    pendingLayers = [];

    Object.keys(stampToLayerName).forEach(function(stamp) {
      var layerName = stampToLayerName[stamp];
      var layer = window[layerName];
      if (layer && map.hasLayer(layer)) {
        populateLayer(layerName);
      }
    });

    map.on('layeradd', function(event) {
      var layerName = stampToLayerName[L.stamp(event.layer)];
      if (layerName) {
        populateLayer(layerName);
      }
    });

    window._gebStations = stations;
    window.dispatchEvent(new CustomEvent('gebStationsReady', {detail: stations}));
  });
})();
