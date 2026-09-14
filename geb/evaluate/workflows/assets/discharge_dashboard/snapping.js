(function() {
  var map = {{ this._parent.get_name() }};
  var layer = {{ this.data.layer }};
  var stations = {{ this.data.stations | script_json }};
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
