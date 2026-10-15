(function() {
  var map = {{ this._parent.get_name() }};
  var layer = {{ this.data }};
  var legend = document.getElementById({{ (this.data + '_legend') | script_json }});
  function updateExclusionLegend() {
    legend.style.display = map.hasLayer(layer) ? '' : 'none';
  }
  layer.on('add remove', updateExclusionLegend);
  updateExclusionLegend();
})();
