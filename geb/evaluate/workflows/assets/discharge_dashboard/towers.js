{%- macro html(this, kwargs) -%}
<script id="geb-tower-data" type="application/json">
{{ this.data.towers | script_json }}
</script>
{%- endmacro -%}
{%- macro script(this, kwargs) -%}
(function() {
  // Eddy covariance (FLUXNET) towers: observed vs simulated evapotranspiration.
  var layerGroup = {{ this.data.layer }};
  var element = document.getElementById("geb-tower-data");
  var towers = (element && element.textContent) ? JSON.parse(element.textContent) : [];
  var plotlyUrl = "https://cdn.plot.ly/plotly-2.35.2.min.js";

  function loadPlotly(callback) {
    if (window.Plotly) { callback(); return; }
    var existing = document.querySelector('script[src="' + plotlyUrl + '"]');
    if (existing) { existing.addEventListener("load", callback); return; }
    var script = document.createElement("script");
    script.src = plotlyUrl;
    script.onload = callback;
    document.head.appendChild(script);
  }

  function isDark() {
    var theme = window._gebCurrentTheme || document.documentElement.getAttribute("data-geb-theme") || "light";
    return theme === "dark";
  }

  // Simulated / observed: blue = too little evaporation, red = too much.
  function biasColor(ratio) {
    if (ratio === null || ratio === undefined) { return "#94a3b8"; }
    var stops = [[0.6, "#1d4ed8"], [0.8, "#60a5fa"], [0.95, "#e5e7eb"], [1.05, "#e5e7eb"], [1.2, "#f87171"], [1.4, "#b91c1c"]];
    if (ratio <= stops[0][0]) { return stops[0][1]; }
    for (var i = 1; i < stops.length; i++) {
      if (ratio <= stops[i][0]) { return stops[i][1]; }
    }
    return stops[stops.length - 1][1];
  }

  function fmt(value, digits, unit) {
    if (value === null || value === undefined) { return "–"; }
    return Number(value).toFixed(digits) + (unit || "");
  }

  function popupHtml(tower, chartId) {
    var s = tower.scores;
    var title = tower.id + (tower.name ? " · " + tower.name : "");
    var cover = tower.igbp ? tower.igbp + (tower.igbp_name ? " (" + tower.igbp_name + ")" : "") : "";
    var rows = [
      ["Observed ET", fmt(s.observed_mm_per_year, 0, " mm/yr")],
      ["Simulated ET", fmt(s.simulated_mm_per_year, 0, " mm/yr")],
      ["Simulated potential ET", fmt(s.potential_mm_per_year, 0, " mm/yr")],
      ["Bias (sim/obs)", fmt(s.bias_ratio, 2) + " · energy-balance corrected " + fmt(s.bias_ratio_corrected, 2)],
      ["Daily correlation", fmt(s.correlation_daily, 2)],
      ["Daily KGE", fmt(s.KGE_daily, 2)],
      ["Days scored", fmt(s.n_days, 0)]
    ];
    var table = rows.map(function(r) {
      return "<tr><td style='padding-right:8px;opacity:.75'>" + r[0] + "</td><td><b>" + r[1] + "</b></td></tr>";
    }).join("");
    var note = tower.monthly ? "" : "<div style='margin-top:6px;opacity:.75'>No overlapping tower and model days.</div>";
    return "<div style='min-width:320px'><b>" + title + "</b><br><span style='opacity:.75'>Eddy covariance tower" +
      (cover ? " · " + cover : "") + "</span><table style='margin-top:6px;font-size:12px'>" + table + "</table>" + note +
      (tower.monthly ? "<div id='" + chartId + "' style='width:520px;height:260px;margin-top:6px'></div>" +
        "<div style='font-size:11px;opacity:.7'>Monthly means of days with ≥80% good tower data (mm/day). Model values at the land unit under the tower.</div>" : "") +
      "</div>";
  }

  function drawChart(tower, chartId) {
    var m = tower.monthly;
    if (!m) { return; }
    var dark = isDark();
    var lineObs = dark ? "#facc15" : "#d97706";
    var lineSim = dark ? "#38bdf8" : "#0284c7";
    var traces = [
      {x: m.time, y: m.transpiration, name: "Transpiration", type: "scatter", mode: "lines", stackgroup: "sim",
       line: {width: 0}, fillcolor: dark ? "rgba(74,222,128,0.35)" : "rgba(22,163,74,0.25)"},
      {x: m.time, y: m.interception, name: "Interception", type: "scatter", mode: "lines", stackgroup: "sim",
       line: {width: 0}, fillcolor: dark ? "rgba(56,189,248,0.30)" : "rgba(2,132,199,0.22)"},
      {x: m.time, y: m.soil, name: "Soil evaporation", type: "scatter", mode: "lines", stackgroup: "sim",
       line: {width: 0}, fillcolor: dark ? "rgba(251,146,60,0.35)" : "rgba(234,88,12,0.22)"},
      {x: m.time, y: m.simulated, name: "Simulated ET", type: "scatter", mode: "lines", line: {color: lineSim, width: 2}},
      {x: m.time, y: m.potential, name: "Potential ET", type: "scatter", mode: "lines", line: {color: lineSim, width: 1, dash: "dot"}},
      {x: m.time, y: m.observed, name: "Observed ET", type: "scatter", mode: "lines", line: {color: lineObs, width: 2}},
      {x: m.time, y: m.observed_corrected, name: "Observed (energy-balance corrected)", type: "scatter", mode: "lines",
       line: {color: lineObs, width: 1, dash: "dash"}}
    ];
    var layout = {
      autosize: true, height: 260, margin: {l: 40, r: 10, t: 10, b: 30},
      paper_bgcolor: dark ? "#020617" : "#ffffff", plot_bgcolor: dark ? "#020617" : "#ffffff",
      font: {color: dark ? "#e2e8f0" : "#0f172a", size: 10},
      legend: {orientation: "h", y: -0.18, font: {size: 9}},
      yaxis: {title: {text: "mm/day"}, rangemode: "tozero", gridcolor: dark ? "#1e293b" : "#f1f5f9"},
      xaxis: {gridcolor: dark ? "#1e293b" : "#f1f5f9"}
    };
    Plotly.newPlot(chartId, traces, layout, {displaylogo: false, responsive: true});
  }

  towers.forEach(function(tower, index) {
    var chartId = "geb-tower-chart-" + index;
    var color = biasColor(tower.scores.bias_ratio);
    var icon = L.divIcon({
      className: "geb-tower-icon",
      html: "<svg width='18' height='18' viewBox='0 0 18 18'><polygon points='9,1 17,16 1,16' fill='" + color +
        "' stroke='#0f172a' stroke-width='1.2'/></svg>",
      iconSize: [18, 18],
      iconAnchor: [9, 12]
    });
    var marker = L.marker([tower.lat, tower.lon], {icon: icon, zIndexOffset: 1000});
    marker.bindTooltip("Flux tower " + tower.id + (tower.igbp ? " (" + tower.igbp + ")" : "") +
      " · bias " + fmt(tower.scores.bias_ratio, 2));
    marker.bindPopup(popupHtml(tower, chartId), {maxWidth: 560});
    marker.on("popupopen", function() {
      loadPlotly(function() { drawChart(tower, chartId); });
    });
    marker.addTo(layerGroup);
  });
})();
{%- endmacro -%}
