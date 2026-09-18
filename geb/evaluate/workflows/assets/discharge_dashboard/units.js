(function() {
  var map = {{ this._parent.get_name() }};
  function formatUtcOffset(offsetHours) {
    if (!Number.isFinite(offsetHours)) return 'missing';
    var absoluteMinutes = Math.round(Math.abs(offsetHours) * 60);
    var wholeHours = Math.floor(absoluteMinutes / 60);
    var minutes = absoluteMinutes % 60;
    var sign = offsetHours >= 0 ? '+' : '-';
    return 'UTC' + sign + String(wholeHours).padStart(2, '0') + ':' + String(minutes).padStart(2, '0');
  }
  function formatUnits(container) {
    if (!container) return;
    [['area-m2', 1e6, 1], ['volume-m3', 1e9, 3], ['distance-m', 1e3, 3]].forEach(function(unit) {
      var attribute = 'data-geb-' + unit[0];
      container.querySelectorAll('[' + attribute + ']').forEach(function(element) {
        var value = Number(element.getAttribute(attribute));
        element.textContent = Number.isFinite(value) ? (value / unit[1]).toLocaleString(undefined, {
          minimumFractionDigits: unit[2], maximumFractionDigits: unit[2]
        }) : 'missing';
      });
    });
    container.querySelectorAll('[data-geb-utc-offset-hours]').forEach(function(element) {
      var value = Number(element.getAttribute('data-geb-utc-offset-hours'));
      element.textContent = formatUtcOffset(value);
    });
  }
  function formatPopup(event) {formatUnits(event.target.getElement());}
  map.on('popupopen', function(event) {
    formatUnits(event.popup.getElement());
    // Snapping reuses a popup when another station is selected.
    event.popup.off('contentupdate', formatPopup).on('contentupdate', formatPopup);
  });
  map.on('tooltipopen', function(event) {formatUnits(event.tooltip.getElement());});
})();
