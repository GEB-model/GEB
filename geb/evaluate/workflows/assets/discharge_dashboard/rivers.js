(function() {
  var rivers = {{ this.data.layer }};
  rivers.eachLayer(function(layer) {
    var properties = layer.feature.properties;
    var tooltip = document.createElement('div');
    tooltip.textContent = 'River ID: ' + properties.river_id +
      ' · Upstream area: ' + (properties.uparea_m2 / 1e6).toLocaleString() + ' km²';
    layer.bindTooltip(tooltip, {sticky: true});
  });
})();
