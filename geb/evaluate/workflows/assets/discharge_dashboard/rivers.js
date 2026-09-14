(function() {
  var map = {{ this._parent.get_name() }};
  var overview = {{ this.data.overview }};
  var detailed = {{ this.data.detailed }};
  var minimumZoom = {{ this.data.minimum_zoom }};
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
