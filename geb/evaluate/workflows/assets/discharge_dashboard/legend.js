(function(){
  var macroData = {{ this.data | script_json }};
  var metricConfigs = macroData.metrics || [];
  var basemapConfigs = macroData.basemaps || [];
  var overlayConfigs = macroData.overlays || [];
  var excludedStationIndex = macroData.excludedStations || [];
  window._gebExcludedStations = excludedStationIndex;
  var exclusionSummary = macroData.exclusionSummary || null;
  var snappingLayerName = macroData.snappingLayer || null;
  var map = {{this._parent.get_name()}};
  var currentFrequency = "all";
  window._gebCurrentFrequency = currentFrequency;

  // Theme setup
  var initialTheme = "light";
  try {
    var stored = localStorage.getItem("geb-theme");
    if (stored === "dark" || stored === "light") {
      initialTheme = stored;
    } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      initialTheme = "dark";
    }
  } catch(e) {}
  window._gebCurrentTheme = initialTheme;
  document.documentElement.setAttribute("data-geb-theme", initialTheme);

  function setDashboardTheme(newTheme) {
    if (window._gebCurrentTheme === newTheme) return;

    var currentPopup = map._popup;
    var wasOpen = Boolean(currentPopup && map.hasLayer(currentPopup));
    var popupSource = currentPopup ? currentPopup._source : null;
    var popupLatLng = currentPopup ? currentPopup.getLatLng() : null;

    window._gebCurrentTheme = newTheme;
    document.documentElement.setAttribute("data-geb-theme", newTheme);
    try { localStorage.setItem("geb-theme", newTheme); } catch(e) {}

    var themeButtons = menuRoot.querySelectorAll("[data-set-theme]");
    themeButtons.forEach(function(btn) {
      if (btn.getAttribute("data-set-theme") === newTheme) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    window.dispatchEvent(new CustomEvent("gebThemeChange", {detail: {theme: newTheme}}));

    if (wasOpen && !map.hasLayer(currentPopup)) {
      if (popupSource && popupSource.openPopup) {
        popupSource.openPopup();
      } else if (popupLatLng) {
        currentPopup.setLatLng(popupLatLng);
        currentPopup.addTo(map);
      }
    }
  }

  // Ensure stationsPane exists so circles are always stacked above rivers/topography
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

  var configByLayer = {};
  var layerByName = {};
  var activeConfig = null;
  metricConfigs.forEach(function(config) {
    var layer = window[config.layer];
    if (!layer) return;
    configByLayer[L.stamp(layer)] = config;
    layerByName[config.layer] = layer;
    if (config.show && !activeConfig) activeConfig = config;
  });
  if (!activeConfig && metricConfigs.length > 0) {
    activeConfig = metricConfigs[0];
  }

  var unifiedControl = L.control({position: "topright"});
  unifiedControl.onAdd = function() {
    var element = L.DomUtil.create("div", "geb-unified-menu");
    L.DomEvent.disableClickPropagation(element);
    L.DomEvent.disableScrollPropagation(element);
    return element;
  };
  unifiedControl.addTo(map);
  var menuRoot = map.getContainer().querySelector(".geb-unified-menu");

  function downloadBlob(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function() { URL.revokeObjectURL(url); }, 500);
  }

  function formatTick(value) {
    if (value === null || value === undefined || isNaN(value)) return "n/a";
    var numericValue = Number(value);
    var absoluteValue = Math.abs(numericValue);
    if (numericValue === 0) return "0";
    if (absoluteValue < 0.01) return numericValue.toExponential(1);
    var digits = absoluteValue >= 1000 ? 0 : absoluteValue >= 100 ? 1 : absoluteValue >= 10 ? 1 : absoluteValue >= 1 ? 2 : 3;
    return numericValue.toLocaleString(undefined, {maximumFractionDigits: digits});
  }

  function exportViolinSvg(config, targetTheme) {
    var dist = config.distribution || {};
    var density = dist.density_points || [];
    if (!density || density.length < 2) return;

    var isDark = (targetTheme === 'dark');
    var bgColor = isDark ? '#0b1120' : '#ffffff';
    var textColor = isDark ? '#f8fafc' : '#0f172a';
    var subTextColor = isDark ? '#94a3b8' : '#475569';
    var axisColor = isDark ? '#475569' : '#cbd5e1';
    var violinFill = isDark ? '#38bdf8' : '#0284c7';
    var violinStroke = isDark ? '#38bdf8' : '#0284c7';
    var medianStroke = isDark ? '#f8fafc' : '#0f172a';

    var totalW = 340;
    var totalH = 380;
    var axisX = 55;
    var plotTopY = 65;
    var plotBottomY = 330;
    var plotH = plotBottomY - plotTopY;
    var centerX = 180;
    var halfW = 85;
    var n = density.length;

    // 2-sided symmetrical violin polygon
    var pathD = 'M ' + centerX + ' ' + plotBottomY;
    for (var i = 0; i < n; i++) {
      var ry = plotBottomY - (i / (n - 1)) * plotH;
      var rx = centerX + (density[i] * halfW);
      pathD += ' L ' + rx.toFixed(1) + ' ' + ry.toFixed(1);
    }
    pathD += ' L ' + centerX + ' ' + plotTopY;
    for (var j = n - 1; j >= 0; j--) {
      var ly = plotBottomY - (j / (n - 1)) * plotH;
      var lx = centerX - (density[j] * halfW);
      pathD += ' L ' + lx.toFixed(1) + ' ' + ly.toFixed(1);
    }
    pathD += ' Z';

    var strokeD = 'M ' + centerX + ' ' + plotBottomY;
    for (var k = 0; k < n; k++) {
      var sy = plotBottomY - (k / (n - 1)) * plotH;
      var sx = centerX + (density[k] * halfW);
      strokeD += ' L ' + sx.toFixed(1) + ' ' + sy.toFixed(1);
    }
    strokeD += ' L ' + centerX + ' ' + plotTopY;
    for (var m = n - 1; m >= 0; m--) {
      var my = plotBottomY - (m / (n - 1)) * plotH;
      var mx = centerX - (density[m] * halfW);
      strokeD += ' L ' + mx.toFixed(1) + ' ' + my.toFixed(1);
    }
    strokeD += ' Z';

    var medianSvg = '';
    if (dist.median !== null && dist.median !== undefined && config.maximum > config.minimum) {
      var medFactor = (dist.median - config.minimum) / (config.maximum - config.minimum);
      if (medFactor >= 0 && medFactor <= 1) {
        var medY = (plotBottomY - medFactor * plotH).toFixed(1);
        var idxFloat = medFactor * (n - 1);
        var i0 = Math.floor(idxFloat);
        var i1 = Math.min(n - 1, i0 + 1);
        var frac = idxFloat - i0;
        var medDensity = density[i0] * (1 - frac) + density[i1] * frac;
        var medLeftX = (centerX - medDensity * halfW).toFixed(1);
        var medRightX = (centerX + medDensity * halfW).toFixed(1);
        medianSvg = '<line x1="' + medLeftX + '" y1="' + medY + '" x2="' + medRightX + '" y2="' + medY +
          '" stroke="' + medianStroke + '" stroke-width="2" stroke-linecap="round" />' +
          '<text x="' + (Number(medRightX) + 7).toFixed(1) + '" y="' + (Number(medY) + 3.5) + '" fill="' + textColor +
          '" font-size="10" font-family="Inter,sans-serif" font-weight="bold">Median: ' + formatTick(dist.median) + '</text>';
      }
    }

    var ticks = (config.ticks && config.ticks.length) ? config.ticks : [
      config.minimum,
      config.minimum + (config.maximum - config.minimum) * 0.25,
      config.minimum + (config.maximum - config.minimum) * 0.5,
      config.minimum + (config.maximum - config.minimum) * 0.75,
      config.maximum
    ];
    var ticksSvg = '';
    ticks.forEach(function(tv) {
      if (config.maximum > config.minimum) {
        var tf = (tv - config.minimum) / (config.maximum - config.minimum);
        if (tf >= -0.01 && tf <= 1.01) {
          var ty = (plotBottomY - tf * plotH).toFixed(1);
          ticksSvg += '<line x1="' + (axisX - 4) + '" y1="' + ty + '" x2="' + axisX + '" y2="' + ty + '" stroke="' + axisColor + '" stroke-width="1" />' +
            '<text x="' + (axisX - 7) + '" y="' + (Number(ty) + 3.5) + '" fill="' + subTextColor +
            '" font-size="10" font-family="Inter,sans-serif" text-anchor="end">' + formatTick(tv) + '</text>';
        }
      }
    });

    var statsText = 'n=' + (dist.count || 0);
    if (dist.median !== null) statsText += ' · Median=' + formatTick(dist.median);
    if (dist.mean !== null) statsText += ' · Mean=' + formatTick(dist.mean);

    var svgContent = '<svg xmlns="http://www.w3.org/2000/svg" width="' + totalW + '" height="' + totalH + '" viewBox="0 0 ' + totalW + ' ' + totalH + '">' +
      '<rect width="100%" height="100%" fill="' + bgColor + '" />' +
      '<text x="' + axisX + '" y="26" fill="' + textColor + '" font-size="13" font-family="Inter,sans-serif" font-weight="bold">' + (config.name || '') + '</text>' +
      '<text x="' + axisX + '" y="44" fill="' + subTextColor + '" font-size="10.5" font-family="Inter,sans-serif">' + statsText + '</text>' +
      '<line x1="' + axisX + '" y1="' + plotTopY + '" x2="' + axisX + '" y2="' + plotBottomY + '" stroke="' + axisColor + '" stroke-width="1" />' +
      '<line x1="' + centerX + '" y1="' + plotTopY + '" x2="' + centerX + '" y2="' + plotBottomY + '" stroke="' + axisColor + '" stroke-width="1" stroke-dasharray="2,2" />' +
      '<path d="' + pathD + '" fill="' + violinFill + '" fill-opacity="0.35" />' +
      '<path d="' + strokeD + '" fill="none" stroke="' + violinStroke + '" stroke-width="1.5" stroke-linejoin="round" />' +
      ticksSvg +
      medianSvg +
      '</svg>';

    var blob = new Blob([svgContent], {type: 'image/svg+xml;charset=utf-8'});
    var cleanName = (config.name || 'metric').replace(/[^a-z0-9]/gi, '_').toLowerCase();
    downloadBlob(blob, cleanName + '_distribution_' + targetTheme + '.svg');
  }

  function drawRoundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.lineTo(x + w - r, y);
    c.quadraticCurveTo(x + w, y, x + w, y + r);
    c.lineTo(x + w, y + h - r);
    c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    c.lineTo(x + r, y + h);
    c.quadraticCurveTo(x, y + h, x, y + h - r);
    c.lineTo(x, y + r);
    c.quadraticCurveTo(x, y, x + r, y);
    c.closePath();
  }

  function drawSimplifiedMapLegend(ctx, w, h, isDark, config) {
    var activeStationLayer = config ? (layerByName[config.layer] || window[config.layer]) : null;
    var hasStations = activeStationLayer ? map.hasLayer(activeStationLayer) : false;

    var visibleItems = [];

    if (hasStations) {
      visibleItems.push({
        label: 'Stations',
        draw: function(c, x, y) {
          c.beginPath();
          c.arc(x + 5, y - 3.5, 4, 0, 2 * Math.PI);
          c.fillStyle = isDark ? '#38bdf8' : '#0284c7';
          c.fill();
          c.strokeStyle = isDark ? '#ffffff' : '#0f172a';
          c.lineWidth = 1;
          c.stroke();
        }
      });
    }

    if (overlayConfigs && overlayConfigs.length > 0) {
      overlayConfigs.forEach(function(ov) {
        var layer = window[ov.layer];
        if (!layer || !map.hasLayer(layer)) return;

        var id = (ov.id || '').toLowerCase();
        var name = (ov.name || '').toLowerCase();

        if (id === 'rivers') {
          visibleItems.push({
            label: 'Rivers',
            draw: function(c, x, y) {
              c.beginPath();
              c.moveTo(x, y - 3.5);
              c.lineTo(x + 14, y - 3.5);
              c.strokeStyle = '#38bdf8';
              c.lineWidth = 2;
              c.stroke();
            }
          });
        } else if (id === 'catchment') {
          visibleItems.push({
            label: 'Catchment',
            draw: function(c, x, y) {
              c.strokeStyle = isDark ? '#94a3b8' : '#475569';
              c.lineWidth = 1.2;
              c.strokeRect(x, y - 7.5, 12, 8);
            }
          });
        } else if (name.indexOf('lake') !== -1) {
          visibleItems.push({
            label: 'Lakes',
            draw: function(c, x, y) {
              c.beginPath();
              c.arc(x + 5, y - 3.5, 4.5, 0, 2 * Math.PI);
              c.fillStyle = '#38bdf8';
              c.fill();
              c.strokeStyle = isDark ? '#ffffff' : '#0284c7';
              c.lineWidth = 1;
              c.stroke();
            }
          });
        } else if (name.indexOf('reservoir') !== -1) {
          visibleItems.push({
            label: ov.name,
            draw: function(c, x, y) {
              c.beginPath();
              c.arc(x + 5, y - 3.5, 4.5, 0, 2 * Math.PI);
              c.fillStyle = name.indexOf('during spinup') !== -1 ? '#8B5CF6'
                : name.indexOf('during simulation') !== -1 ? '#E11D48'
                : name.indexOf('after simulation') !== -1 ? '#94A3B8'
                : '#FF8A65';
              c.fill();
              c.strokeStyle = isDark ? '#ffffff' : '#c2410c';
              c.lineWidth = 1;
              c.stroke();
            }
          });
        } else if (id === 'exclusions' || name.indexOf('excluded') !== -1) {
          visibleItems.push({
            label: 'Excluded stations',
            draw: function(c, x, y) {
              c.beginPath();
              c.arc(x + 5, y - 3.5, 4, 0, 2 * Math.PI);
              c.fillStyle = '#dc2626';
              c.fill();
              c.strokeStyle = isDark ? '#ffffff' : '#7f1d1d';
              c.lineWidth = 1;
              c.stroke();
            }
          });
        } else if (id === 'snapping' || name.indexOf('snapping') !== -1) {
          visibleItems.push({
            label: 'Snapping details',
            draw: function(c, x, y) {
              c.beginPath();
              c.arc(x + 5, y - 3.5, 4, 0, 2 * Math.PI);
              c.fillStyle = '#1d4ed8';
              c.fill();
              c.strokeStyle = isDark ? '#ffffff' : '#1e3a8a';
              c.lineWidth = 1;
              c.stroke();
            }
          });
        } else {
          visibleItems.push({
            label: ov.name,
            draw: function(c, x, y) {
              c.beginPath();
              c.arc(x + 5, y - 3.5, 3.5, 0, 2 * Math.PI);
              c.fillStyle = isDark ? '#38bdf8' : '#0284c7';
              c.fill();
            }
          });
        }
      });
    }

    var showMetric = hasStations && config;
    if (!showMetric && visibleItems.length === 0) return;

    var pad = 24;
    var legendW = 280;

    ctx.font = '9.5px Inter, system-ui, sans-serif';
    var maxContentW = legendW - 24;
    var rows = [[]];
    var curRowW = 0;

    visibleItems.forEach(function(item) {
      var itemW = 16 + ctx.measureText(item.label).width + 12;
      if (curRowW > 0 && curRowW + itemW > maxContentW) {
        rows.push([item]);
        curRowW = itemW;
      } else {
        rows[rows.length - 1].push(item);
        curRowW += itemW;
      }
    });

    var legendH = 14;
    if (showMetric) {
      if (config.kind === 'continuous' || config.kind === 'characteristic') {
        legendH += 44;
      } else if (config.kind === 'availability') {
        legendH += 34;
      } else {
        legendH += 22;
      }
    } else if (visibleItems.length > 0) {
      legendH += 20;
    }

    if (visibleItems.length > 0) {
      legendH += (rows.length * 18);
    }
    legendH += 6;

    var lx = pad;
    var ly = h - legendH - pad;

    var bg = isDark ? 'rgba(15, 23, 42, 0.92)' : 'rgba(255, 255, 255, 0.94)';
    var border = isDark ? '#334155' : '#cbd5e1';
    var textColor = isDark ? '#f8fafc' : '#0f172a';
    var subText = isDark ? '#94a3b8' : '#475569';

    ctx.save();
    ctx.shadowColor = 'rgba(0, 0, 0, 0.25)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 2;
    drawRoundRect(ctx, lx, ly, legendW, legendH, 8);
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    ctx.stroke();

    var currentY = ly + 16;

    if (showMetric) {
      ctx.font = 'bold 11px Inter, system-ui, -apple-system, sans-serif';
      ctx.fillStyle = textColor;
      ctx.textAlign = 'left';
      ctx.fillText(config.name || 'Station Scores', lx + 12, currentY);

      if (config.kind === 'continuous' || config.kind === 'characteristic') {
        var barX = lx + 12;
        var barY = currentY + 7;
        var barW = legendW - 24;
        var barH = 7;

        var grad = ctx.createLinearGradient(barX, 0, barX + barW, 0);
        var colors = config.colors || ['#38bdf8'];
        for (var k = 0; k < colors.length; k++) {
          grad.addColorStop(k / (colors.length - 1), colors[k]);
        }
        ctx.fillStyle = grad;
        ctx.fillRect(barX, barY, barW, barH);
        ctx.strokeStyle = border;
        ctx.lineWidth = 1;
        ctx.strokeRect(barX, barY, barW, barH);

        ctx.font = '9px Inter, system-ui, sans-serif';
        ctx.fillStyle = subText;
        ctx.textAlign = 'left';
        ctx.fillText(formatTick(config.minimum), barX, barY + barH + 11);
        ctx.textAlign = 'center';
        var midVal = (config.minimum + config.maximum) / 2;
        ctx.fillText(formatTick(midVal), barX + barW / 2, barY + barH + 11);
        ctx.textAlign = 'right';
        ctx.fillText(formatTick(config.maximum), barX + barW, barY + barH + 11);
        currentY = barY + barH + 20;
      } else if (config.kind === 'availability') {
        var dotY = currentY + 12;
        ctx.font = '9.5px Inter, system-ui, sans-serif';
        ctx.fillStyle = config.available_color || '#22c55e';
        ctx.beginPath(); ctx.arc(lx + 16, dotY - 3.5, 4, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = textColor; ctx.textAlign = 'left';
        ctx.fillText('Available', lx + 24, dotY);
        ctx.fillStyle = config.unavailable_color || '#ef4444';
        ctx.beginPath(); ctx.arc(lx + 105, dotY - 3.5, 4, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = textColor;
        ctx.fillText('Not available', lx + 113, dotY);
        currentY = dotY + 14;
      } else {
        currentY += 16;
      }
    } else if (visibleItems.length > 0) {
      ctx.font = 'bold 11px Inter, system-ui, -apple-system, sans-serif';
      ctx.fillStyle = textColor;
      ctx.textAlign = 'left';
      ctx.fillText('Map Features', lx + 12, currentY);
      currentY += 16;
    }

    ctx.font = '9.5px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    rows.forEach(function(row) {
      var itemX = lx + 12;
      row.forEach(function(item) {
        item.draw(ctx, itemX, currentY);
        ctx.fillStyle = textColor;
        ctx.fillText(item.label, itemX + 15, currentY);
        itemX += 16 + ctx.measureText(item.label).width + 12;
      });
      currentY += 18;
    });

    ctx.restore();
  }

  function exportMapImage() {
    var mapContainer = map.getContainer();
    var w = mapContainer.clientWidth;
    var h = mapContainer.clientHeight;
    var isDark = (window._gebCurrentTheme === 'dark');
    var mapRect = mapContainer.getBoundingClientRect();
    var scale = 2;

    var exportCanvas = document.createElement('canvas');
    exportCanvas.width = Math.round(w * scale);
    exportCanvas.height = Math.round(h * scale);
    var ctx = exportCanvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.scale(scale, scale);
    ctx.fillStyle = isDark ? '#020617' : '#f8fafc';
    ctx.fillRect(0, 0, w, h);

    // 1. Draw tile images onto exportCanvas
    var tileImgs = mapContainer.querySelectorAll('.leaflet-tile-pane img');
    tileImgs.forEach(function(img) {
      if (img.complete && img.naturalWidth > 0 && img.style.display !== 'none' && img.style.visibility !== 'hidden') {
        var r = img.getBoundingClientRect();
        try {
          ctx.drawImage(img, r.left - mapRect.left, r.top - mapRect.top, r.width, r.height);
        } catch(e) {}
      }
    });

    // 2. Draw canvas vector layers (rivers, catchment, stations, waterbodies)
    var canvases = mapContainer.querySelectorAll('.leaflet-pane canvas');
    canvases.forEach(function(c) {
      if (c !== exportCanvas && c.width > 0 && c.height > 0 && c.style.display !== 'none' && c.style.visibility !== 'hidden') {
        var r = c.getBoundingClientRect();
        try {
          ctx.drawImage(c, r.left - mapRect.left, r.top - mapRect.top, r.width, r.height);
        } catch(e) {}
      }
    });

    // 3. Draw marker images if present
    var markerImgs = mapContainer.querySelectorAll('.leaflet-marker-pane img, .leaflet-marker-icon');
    markerImgs.forEach(function(mImg) {
      if (mImg.tagName === 'IMG' && mImg.complete && mImg.naturalWidth > 0 && mImg.style.display !== 'none' && mImg.style.visibility !== 'hidden') {
        var mr = mImg.getBoundingClientRect();
        try {
          ctx.drawImage(mImg, mr.left - mapRect.left, mr.top - mapRect.top, mr.width, mr.height);
        } catch(me) {}
      }
    });

    // 4. Draw SVG elements if any
    var svgs = mapContainer.querySelectorAll('.leaflet-pane svg');
    var svgDrawPromises = [];
    svgs.forEach(function(svg) {
      if (svg.style.display !== 'none' && svg.style.visibility !== 'hidden') {
        var r = svg.getBoundingClientRect();
        var dx = r.left - mapRect.left;
        var dy = r.top - mapRect.top;
        var xml = new XMLSerializer().serializeToString(svg);
        var blob = new Blob([xml], {type: 'image/svg+xml;charset=utf-8'});
        var url = URL.createObjectURL(blob);
        var p = new Promise(function(resolve) {
          var img = new Image();
          img.onload = function() {
            try {
              ctx.drawImage(img, dx, dy, r.width, r.height);
            } catch(err) {}
            URL.revokeObjectURL(url);
            resolve();
          };
          img.onerror = function() {
            URL.revokeObjectURL(url);
            resolve();
          };
          img.src = url;
        });
        svgDrawPromises.push(p);
      }
    });

    Promise.all(svgDrawPromises).then(function() {
      drawSimplifiedMapLegend(ctx, w, h, isDark, activeConfig);
      exportCanvas.toBlob(function(blob) {
        if (blob) {
          downloadBlob(blob, 'geb_discharge_map.png');
        }
      }, 'image/png');
    });
  }

  function renderViolinSvg(config, dist) {
    var density = (dist && dist.density_points) ? dist.density_points : [];
    if (!density || density.length < 2) return '';
    var totalW = 252;
    var totalH = 46;
    var baselineY = 44;
    var peakHeight = 38;
    var n = density.length;

    var pathD = 'M 0 ' + baselineY;
    for (var i = 0; i < n; i++) {
      var x = (i / (n - 1)) * totalW;
      var y = baselineY - (density[i] * peakHeight);
      pathD += ' L ' + x.toFixed(1) + ' ' + y.toFixed(1);
    }
    pathD += ' L ' + totalW + ' ' + baselineY + ' Z';

    var strokeD = '';
    for (var j = 0; j < n; j++) {
      var sx = (j / (n - 1)) * totalW;
      var sy = baselineY - (density[j] * peakHeight);
      strokeD += (j === 0 ? 'M ' : ' L ') + sx.toFixed(1) + ' ' + sy.toFixed(1);
    }

    var gradId = 'geb-vgrad-' + config.layer;
    var colors = config.colors || ['#38bdf8'];
    var gradStops = '';
    for (var k = 0; k < colors.length; k++) {
      var offset = (colors.length > 1) ? Math.round((k / (colors.length - 1)) * 100) : 0;
      gradStops += '<stop offset="' + offset + '%" stop-color="' + colors[k] + '" stop-opacity="0.78" />';
    }

    var medianLine = '';
    if (dist.median !== null && dist.median !== undefined && config.maximum > config.minimum) {
      var medFactor = (dist.median - config.minimum) / (config.maximum - config.minimum);
      if (medFactor >= 0 && medFactor <= 1) {
        var medX = (medFactor * totalW).toFixed(1);
        medianLine = '<line class="geb-violin-median" x1="' + medX + '" y1="3" x2="' + medX + '" y2="' + baselineY +
          '" stroke-width="2" stroke-dasharray="3,2"><title>Median: ' +
          formatTick(dist.median) + '</title></line>';
      }
    }

    return '<svg class="geb-violin-svg" viewBox="0 0 ' + totalW + ' ' + totalH + '" preserveAspectRatio="none">' +
      '<defs><linearGradient id="' + gradId + '" x1="0%" y1="0%" x2="100%" y2="0%">' +
      gradStops + '</linearGradient></defs>' +
      '<path d="' + pathD + '" fill="url(#' + gradId + ')" />' +
      '<path class="geb-violin-outline" d="' + strokeD + '" fill="none" stroke-width="1.4" stroke-linejoin="round" />' +
      medianLine + '</svg>';
  }

  function recalculateStatistics(config) {
    if (!config) return;
    if (config.kind === "availability") {
      var avCount = 0;
      var unavCount = 0;
      (window._gebStations || []).forEach(function(st) {
        if (!st.ex && st._isVisible !== false) {
          if (st.av === 1) avCount++;
          else if (st.av === 0) unavCount++;
        }
      });
      config.available_count = avCount;
      config.unavailable_count = unavCount;
      return;
    }

    var values = [];
    (window._gebStations || []).forEach(function(st) {
      if (st.ex || st._isVisible === false) return;
      var val = undefined;
      if (currentFrequency === "hourly" && st.vh && st.vh[config.layer] !== undefined) {
        val = st.vh[config.layer];
      } else if (st.v && st.v[config.layer] !== undefined) {
        val = st.v[config.layer];
      }
      if (val !== undefined && val !== null && !isNaN(val) && isFinite(val)) {
        values.push(Number(val));
      }
    });

    var count = values.length;
    if (count === 0) {
      config.distribution = {
        count: 0,
        median: null,
        mean: null,
        min: null,
        max: null,
        p25: null,
        p75: null,
        density_points: []
      };
      return;
    }

    values.sort(function(a, b) { return a - b; });
    var min = values[0];
    var max = values[count - 1];
    var sum = 0;
    for (var i = 0; i < count; i++) {
      sum += values[i];
    }
    var mean = sum / count;
    var median = (count % 2 === 1)
      ? values[Math.floor(count / 2)]
      : (values[count / 2 - 1] + values[count / 2]) / 2;

    var p25 = values[Math.floor(count * 0.25)];
    var p75 = values[Math.floor(count * 0.75)];

    var densityPoints = [];
    var numPoints = 80;
    var vmin = config.minimum;
    var vmax = config.maximum;
    var step = (vmax - vmin) / (numPoints - 1);

    if (count >= 2) {
      var sumSqDiff = 0;
      for (var i = 0; i < count; i++) {
        sumSqDiff += (values[i] - mean) * (values[i] - mean);
      }
      var variance = sumSqDiff / (count - 1);
      var std = Math.sqrt(variance);
      var iqr = p75 - p25;
      var hFactor = (iqr > 0 && iqr < std * 1.34) ? (iqr / 1.34) : std;
      if (hFactor <= 0) hFactor = std || 1.0;
      var bandwidth = 0.55 * Math.pow(count, -0.2) * hFactor;
      if (bandwidth <= 1e-4) bandwidth = (vmax - vmin) / 40;

      var maxD = 0;
      var sqrt2Pi = Math.sqrt(2 * Math.PI);
      for (var j = 0; j < numPoints; j++) {
        var t = vmin + j * step;
        var sumK = 0;
        for (var i = 0; i < count; i++) {
          var u = (t - values[i]) / bandwidth;
          sumK += Math.exp(-0.5 * u * u);
        }
        var d = sumK / (count * bandwidth * sqrt2Pi);
        densityPoints.push(d);
        if (d > maxD) maxD = d;
      }
      var normPoints = [];
      for (var j = 0; j < numPoints; j++) {
        normPoints.push(maxD > 0 ? Math.round((densityPoints[j] / maxD) * 10000) / 10000 : 0);
      }
      densityPoints = normPoints;
    } else {
      densityPoints = new Array(numPoints).fill(0);
    }

    config.distribution = {
      count: count,
      median: median,
      mean: mean,
      min: min,
      max: max,
      p25: p25,
      p75: p75,
      density_points: densityPoints
    };
  }

  function renderDistributionSection(config) {
    if (window._gebStations && window._gebStations.length > 0) {
      recalculateStatistics(config);
    }
    if (config.kind === 'availability') {
      var totalAv = (config.available_count || 0) + (config.unavailable_count || 0);
      var avPct = totalAv > 0 ? Math.round((config.available_count / totalAv) * 100) : 0;
      var unPct = 100 - avPct;
      return '<div class="geb-categorical-key"><i style="background:' + config.available_color + '"></i>' +
        'Available: <b>' + config.available_count + '</b> (' + avPct + '%)</div>' +
        '<div class="geb-categorical-key"><i style="background:' + config.unavailable_color + '"></i>' +
        'Not available: <b>' + config.unavailable_count + '</b> (' + unPct + '%)</div>' +
        '<div class="geb-availability-bar"><span style="width:' + avPct + '%;background:' + config.available_color + '"></span>' +
        '<span style="width:' + unPct + '%;background:' + config.unavailable_color + '"></span></div>';
    }

    var dist = config.distribution || {};
    var statsText = '';
    if (dist.count !== null && dist.count !== undefined) {
      var statParts = [];
      if (dist.median !== null && dist.median !== undefined) {
        statParts.push('<span>Median: <b>' + formatTick(dist.median) + '</b></span>');
      }
      if (dist.mean !== null && dist.mean !== undefined) {
        statParts.push('<span>Mean: <b>' + formatTick(dist.mean) + '</b></span>');
      }
      statParts.push('<span>n = <b>' + dist.count + '</b></span>');
      statsText = statParts.join('<span class="geb-stat-bullet">·</span>');
    }

    var exportBtnHtml = '';
    if (dist && dist.density_points && dist.density_points.length > 1) {
      exportBtnHtml = '<button type="button" class="geb-modebar-btn" id="geb-export-violin" title="Download plot as SVG" aria-label="Download plot as SVG">' +
        '<svg width="15" height="15" viewBox="0 0 512 512"><path d="M512 144v288c0 26.5-21.5 48-48 48H48c-26.5 0-48-21.5-48-48V144c0-26.5 21.5-48 48-48h88l18.3-36.6C163.5 44.8 178.6 32 195.2 32h121.6c16.6 0 31.7 12.8 38.9 27.4L376 96h88c26.5 0 48 21.5 48 48zM256 400c70.7 0 128-57.3 128-128s-57.3-128-128-128-128 57.3-128 128 57.3 128 128 128zm0-40c-48.5 0-88-39.5-88-88s39.5-88 88-88 88 39.5 88 88-39.5 88-88 88z"/></svg>' +
        '</button>';
    }
    var headerHtml = '<div class="geb-violin-header"><div class="geb-violin-stats">' + statsText + '</div>' + exportBtnHtml + '</div>';
    var violinSvg = renderViolinSvg(config, dist);

    var ticks = [];
    if (config.ticks && Array.isArray(config.ticks) && config.ticks.length > 0) {
      ticks = config.ticks;
    } else {
      for (var index = 0; index < 5; index += 1) {
        ticks.push(config.minimum + (config.maximum - config.minimum) * index / 4);
      }
    }

    var detail = '';
    if (config.kind === 'characteristic') {
      ticks = config.reference_values || ticks;
      detail = '<div class="geb-metric-note">Colour shows percentile rank (0 to 100). Ticks: ranks 0 / 25 / 50 / 75 / 100.</div>' +
        (config.missing_count ? '<div class="geb-metric-note">Missing value: n=' + config.missing_count + '</div>' : '');
    }

    var gradientHtml = '<div class="geb-metric-gradient" style="background:linear-gradient(90deg,' +
      config.colors.join(',') + ')"></div>' +
      '<div class="geb-metric-ticks">' +
      ticks.map(function(value) { return '<span>' + formatTick(value) + '</span>'; }).join('') +
      '</div>' + detail;

    return '<div class="geb-violin-container">' + headerHtml + violinSvg + gradientHtml + '</div>';
  }

  function renderMenu() {
    var categories = {};
    metricConfigs.forEach(function(c) {
      var cat = c.category || "Evaluation Metrics";
      if (!categories[cat]) categories[cat] = [];
      categories[cat].push(c);
    });

    var selectOptions = "";
    Object.keys(categories).forEach(function(cat) {
      selectOptions += "<optgroup label=\"" + cat + "\">";
      categories[cat].forEach(function(c) {
        var selected = (activeConfig && c.layer === activeConfig.layer) ? " selected" : "";
        selectOptions += "<option value=\"" + c.layer + "\"" + selected + ">" + c.name + "</option>";
      });
      selectOptions += "</optgroup>";
    });

    var basemapsHtml = "";
    if (basemapConfigs.length > 0) {
      basemapsHtml = "<div class=\"geb-basemap-row\">" +
        basemapConfigs.map(function(bm) {
          var layer = window[bm.layer];
          var isChecked = layer ? map.hasLayer(layer) : bm.active;
          return "<label class=\"geb-basemap-label\"><input type=\"radio\" name=\"geb-basemap-choice\" value=\"" +
            bm.layer + "\"" + (isChecked ? " checked" : "") + "> " + bm.name + "</label>";
        }).join("") + "</div>";
    }

    var overlaysHtml = "";
    if (overlayConfigs.length > 0) {
      overlaysHtml = "<div class=\"geb-overlays-list\">" +
        overlayConfigs.map(function(ov) {
          var layer = window[ov.layer];
          var isChecked = layer ? map.hasLayer(layer) : ov.active;
          var subpanelHtml = "";

          if (ov.id === "exclusions" && exclusionSummary) {
            subpanelHtml = "<div class=\"geb-subpanel\" id=\"geb-exclusion-subpanel\" style=\"" + (isChecked ? "" : "display:none;") + "\">" +
              "<div class=\"geb-subpanel-row\"><span style=\"color:#dc2626\">●</span> Area / distance: <b>" + exclusionSummary.areaDistanceCount + "</b></div>" +
              "<div class=\"geb-subpanel-row\"><span style=\"color:#d97706\">●</span> Other reasons: <b>" + exclusionSummary.otherCount + "</b></div>" +
              "<div class=\"geb-subpanel-note\">Hover or click a red station marker for reason.</div>" +
              "</div>";
          } else if (ov.id === "snapping") {
            subpanelHtml = "<div class=\"geb-subpanel\" id=\"geb-snapping-subpanel\" style=\"" + (isChecked ? "" : "display:none;") + "\">" +
              "<div class=\"geb-subpanel-title\">Snapping: 1 → 2 → 3</div>" +
              "<div class=\"geb-snap-step\"><i style=\"background:#BE123C;border-radius:50%\">1</i> Original gauge (obs)</div>" +
              "<div class=\"geb-snap-step\"><i style=\"background:#92400E;border-radius:3px\">2</i> Subgrid pixel</div>" +
              "<div class=\"geb-snap-step\"><i style=\"background:#1D4ED8;border-radius:0\">3</i> Model cell centre (sim)</div>" +
              "<div class=\"geb-subpanel-note\">Overview dots: green = PASS, orange = warning, red = excluded. Click a dot or zoom in for numbered steps.</div>" +
              "<div class=\"geb-snapping-status\"></div>" +
              "</div>";
          }

          return "<div class=\"geb-overlay-entry\">" +
            "<label class=\"geb-overlay-item\"><input type=\"checkbox\" data-overlay-layer=\"" +
            ov.layer + "\"" + (isChecked ? " checked" : "") + "> " + ov.name + "</label>" +
            subpanelHtml +
            "</div>";
        }).join("") + "</div>";
    }

    var isDark = (window._gebCurrentTheme === "dark");
    var themePill = "<div class=\"geb-theme-pill\">" +
      "<button type=\"button\" class=\"geb-theme-btn" + (!isDark ? " active" : "") + "\" data-set-theme=\"light\" title=\"Light mode\">☀️ Light</button>" +
      "<button type=\"button\" class=\"geb-theme-btn" + (isDark ? " active" : "") + "\" data-set-theme=\"dark\" title=\"Dark mode\">🌙 Dark</button>" +
      "</div>";

    menuRoot.innerHTML = "<div class=\"geb-unified-header\">" +
      "<div class=\"geb-header-title-box\">" +
      "<b>Dashboard</b>" +
      themePill +
      "</div>" +
      "<button type=\"button\" class=\"geb-unified-toggle\" title=\"Collapse / expand menu\">−</button>" +
      "</div>" +
      "<div class=\"geb-unified-body\">" +
      "<div class=\"geb-menu-section\">" +
      "<label for=\"geb-station-search-input\" class=\"geb-menu-label\">Station search</label>" +
      "<div class=\"geb-station-search__row\">" +
      "<input id=\"geb-station-search-input\" type=\"search\" placeholder=\"ID or name\">" +
      "<button type=\"button\" id=\"geb-station-search-clear\" title=\"Clear search\">Clear</button>" +
      "</div>" +
      "<div class=\"geb-station-search__status\"></div>" +
      "</div>" +
      "<div class=\"geb-menu-section\">" +
      "<label class=\"geb-menu-label\">Observation Frequency</label>" +
      "<div class=\"geb-freq-pill\">" +
      "<button type=\"button\" class=\"geb-freq-btn" + (currentFrequency === "all" ? " active" : "") + "\" data-freq=\"all\" title=\"Show all stations\">All</button>" +
      "<button type=\"button\" class=\"geb-freq-btn" + (currentFrequency === "daily" ? " active" : "") + "\" data-freq=\"daily\" title=\"Show only daily stations\">Daily</button>" +
      "<button type=\"button\" class=\"geb-freq-btn" + (currentFrequency === "hourly" ? " active" : "") + "\" data-freq=\"hourly\" title=\"Show only hourly stations\">Hourly</button>" +
      "</div>" +
      "</div>" +
      "<div class=\"geb-menu-section\">" +
      "<label for=\"geb-metric-dropdown\" class=\"geb-menu-label\">Station Variable</label>" +
      "<select id=\"geb-metric-dropdown\" class=\"geb-metric-dropdown\">" + selectOptions + "</select>" +
      "<div id=\"geb-distribution-container\">" +
      (activeConfig ? renderDistributionSection(activeConfig) : "") +
      "</div>" +
      "</div>" +
      "<div class=\"geb-menu-section geb-layers-section\">" +
      "<div class=\"geb-layers-header\">" +
      "<span class=\"geb-menu-label\">Map Layers</span>" +
      "<button type=\"button\" class=\"geb-export-btn geb-map-export-btn\" id=\"geb-export-map-btn\" title=\"Export current map view as image\">📷 Export map</button>" +
      "</div>" +
      basemapsHtml +
      overlaysHtml +
      "</div>" +
      "</div>";

    attachMenuHandlers();
  }

  function switchStationLayer(layerName) {
    var targetConfig = null;
    metricConfigs.forEach(function(c) {
      if (c.layer === layerName) targetConfig = c;
    });
    if (!targetConfig) return;

    if (activeConfig && activeConfig.layer !== targetConfig.layer) {
      var oldLayer = layerByName[activeConfig.layer];
      if (oldLayer && map.hasLayer(oldLayer)) {
        map.removeLayer(oldLayer);
      }
    }

    var newLayer = layerByName[targetConfig.layer];
    if (newLayer) {
      if (window._gebPopulateStationLayer) {
        window._gebPopulateStationLayer(targetConfig.layer);
      }
      if (!map.hasLayer(newLayer)) {
        newLayer.addTo(map);
      }
      if (newLayer.bringToFront) {
        try { newLayer.bringToFront(); } catch(e) {}
      }
    }
    activeConfig = targetConfig;
    renderMenu();
  }

  function switchBasemap(layerName) {
    basemapConfigs.forEach(function(bm) {
      var layer = window[bm.layer];
      if (!layer) return;
      if (bm.layer === layerName) {
        if (!map.hasLayer(layer)) layer.addTo(map);
      } else {
        if (map.hasLayer(layer)) map.removeLayer(layer);
      }
    });
  }

  function toggleOverlay(layerName, isChecked) {
    var layer = window[layerName];
    if (layer) {
      if (isChecked) {
        if (!map.hasLayer(layer)) layer.addTo(map);
      } else {
        if (map.hasLayer(layer)) map.removeLayer(layer);
      }
    }
    if (exclusionSummary && layerName === exclusionSummary.layer) {
      var exPanel = menuRoot.querySelector("#geb-exclusion-subpanel");
      if (exPanel) exPanel.style.display = isChecked ? "" : "none";
    }
    if (snappingLayerName && layerName === snappingLayerName) {
      var snapPanel = menuRoot.querySelector("#geb-snapping-subpanel");
      if (snapPanel) snapPanel.style.display = isChecked ? "" : "none";
    }
  }

  function resolveMarkers(station) {
    if (station._markers && station._markers.length > 0) return station._markers;
    if (station.markers) {
      station._markers = station.markers.map(function(name) {
        return window[name] || null;
      }).filter(Boolean);
      return station._markers;
    }
    return [];
  }

  var lastQuery = "";

  function updateFilterAndDistribution() {
    window._gebCurrentFrequency = currentFrequency;
    var normalizedQuery = (lastQuery || "").trim().toLowerCase();
    var allStations = (window._gebStations || []).concat(excludedStationIndex || []);
    var matches = [];

    allStations.forEach(function(station) {
      var haystack = ((station.id || "") + " " + (station.name || "")).toLowerCase();
      var searchMatch = !normalizedQuery || haystack.indexOf(normalizedQuery) !== -1;
      station._searchMatch = searchMatch;

      var freqMatch = true;
      if (currentFrequency === "daily") {
        freqMatch = (station.fq === "daily");
      } else if (currentFrequency === "hourly") {
        freqMatch = (station.fq === "hourly");
      }
      station._freqMatch = freqMatch;

      var visible = searchMatch && freqMatch;
      station._isVisible = visible;
      station._searchVisible = visible;

      resolveMarkers(station).forEach(function(marker) {
        var fillColor = (currentFrequency === "hourly" && station.mh && activeConfig && station.mh[activeConfig.layer])
          ? station.mh[activeConfig.layer]
          : (station.m && activeConfig && station.m[activeConfig.layer]);

        var styleOpts = {
          opacity: visible ? 1 : 0,
          fillOpacity: visible ? (marker.options.fillOpacity || 0.9) : 0
        };
        if (fillColor) {
          styleOpts.fillColor = fillColor;
        }
        marker.setStyle(styleOpts);
        marker.options.interactive = visible;
        if (marker.getElement()) {
          marker.getElement().style.pointerEvents = visible ? "" : "none";
        }
      });

      if (visible) {
        matches.push(station);
      }
    });

    var statusEl = menuRoot.querySelector(".geb-station-search__status");
    if (statusEl) {
      var freqLabel = currentFrequency === "all" ? "" : (" " + currentFrequency);
      var text = normalizedQuery
        ? matches.length + " matching" + freqLabel + " stations"
        : matches.length + freqLabel + " stations";
      if (normalizedQuery && matches.length) text += " · Enter opens match";
      statusEl.textContent = text;
    }

    if (activeConfig) {
      recalculateStatistics(activeConfig);
      var distContainer = menuRoot.querySelector("#geb-distribution-container");
      if (distContainer) {
        distContainer.innerHTML = renderDistributionSection(activeConfig);
        var violinBtn = distContainer.querySelector("#geb-export-violin");
        if (violinBtn) {
          violinBtn.addEventListener("click", function() {
            exportViolinSvg(activeConfig, window._gebCurrentTheme || "light");
          });
        }
      }
    }

    return matches;
  }

  function applySearch(query) {
    lastQuery = query;
    return updateFilterAndDistribution();
  }

  function openFirstMatch(matches) {
    if (!matches || !matches.length) return;
    var markers = resolveMarkers(matches[0]);
    var marker = markers[0];
    if (!marker) return;
    map.setView(marker.getLatLng(), Math.max(map.getZoom(), 8));
    marker.openPopup();
  }

  function attachMenuHandlers() {
    var toggleBtn = menuRoot.querySelector(".geb-unified-toggle");
    var body = menuRoot.querySelector(".geb-unified-body");
    if (toggleBtn && body) {
      toggleBtn.addEventListener("click", function() {
        var isHidden = body.style.display === "none";
        body.style.display = isHidden ? "" : "none";
        toggleBtn.textContent = isHidden ? "−" : "+";
      });
    }

    var themeButtons = menuRoot.querySelectorAll("[data-set-theme]");
    themeButtons.forEach(function(btn) {
      L.DomEvent.disableClickPropagation(btn);
      btn.addEventListener("click", function(e) {
        L.DomEvent.stop(e);
        setDashboardTheme(btn.getAttribute("data-set-theme"));
      });
    });

    var freqButtons = menuRoot.querySelectorAll(".geb-freq-btn");
    freqButtons.forEach(function(btn) {
      L.DomEvent.disableClickPropagation(btn);
      btn.addEventListener("click", function(e) {
        L.DomEvent.stop(e);
        var targetFreq = btn.getAttribute("data-freq");
        if (targetFreq === currentFrequency) return;
        currentFrequency = targetFreq;
        freqButtons.forEach(function(b) {
          if (b.getAttribute("data-freq") === currentFrequency) {
            b.classList.add("active");
          } else {
            b.classList.remove("active");
          }
        });
        updateFilterAndDistribution();
      });
    });

    var dropdown = menuRoot.querySelector("#geb-metric-dropdown");
    if (dropdown) {
      dropdown.addEventListener("change", function(e) {
        switchStationLayer(e.target.value);
      });
    }

    var violinBtn = menuRoot.querySelector("#geb-export-violin");
    if (violinBtn && activeConfig) {
      violinBtn.addEventListener("click", function() {
        exportViolinSvg(activeConfig, window._gebCurrentTheme || "light");
      });
    }

    var mapExportBtn = menuRoot.querySelector("#geb-export-map-btn");
    if (mapExportBtn) {
      mapExportBtn.addEventListener("click", function() {
        exportMapImage();
      });
    }

    var basemapRadios = menuRoot.querySelectorAll("input[name=\"geb-basemap-choice\"]");
    basemapRadios.forEach(function(radio) {
      radio.addEventListener("change", function(e) {
        switchBasemap(e.target.value);
      });
    });

    var overlayChecks = menuRoot.querySelectorAll("input[data-overlay-layer]");
    overlayChecks.forEach(function(check) {
      check.addEventListener("change", function(e) {
        toggleOverlay(e.target.getAttribute("data-overlay-layer"), e.target.checked);
      });
    });

    var searchInput = menuRoot.querySelector("#geb-station-search-input");
    var searchClear = menuRoot.querySelector("#geb-station-search-clear");
    if (searchInput) {
      searchInput.value = lastQuery;
      searchInput.addEventListener("input", function() {
        applySearch(searchInput.value);
      });
      searchInput.addEventListener("keydown", function(event) {
        if (event.key === "Enter") {
          event.preventDefault();
          openFirstMatch(applySearch(searchInput.value));
        }
      });
    }
    if (searchClear && searchInput) {
      searchClear.addEventListener("click", function() {
        searchInput.value = "";
        searchInput.focus();
        applySearch("");
      });
    }

    applySearch(lastQuery);
  }

  window.addEventListener("gebStationsReady", function() {
    updateFilterAndDistribution();
  });

  if (window._gebStations && window._gebStations.length > 0) {
    updateFilterAndDistribution();
  }

  if (activeConfig) {
    var initLayer = layerByName[activeConfig.layer];
    if (initLayer && !map.hasLayer(initLayer)) {
      initLayer.addTo(map);
    }
    if (window._gebPopulateStationLayer) {
      window._gebPopulateStationLayer(activeConfig.layer);
    }
  }

  var style = document.createElement("style");
  style.textContent =
    ".geb-unified-menu{border-radius:9px;box-shadow:0 8px 30px rgba(15,23,42,.2);font-family:Inter,system-ui,-apple-system,sans-serif;padding:0;width:282px;box-sizing:border-box;overflow:hidden}" +
    ".geb-unified-header{padding:8px 12px;display:flex;align-items:center;justify-content:space-between;font-size:12px;font-weight:700}" +
    ".geb-header-title-box{display:flex;align-items:center;gap:8px}" +
    ".geb-theme-pill{display:inline-flex;background:rgba(255,255,255,0.12);padding:2px;border-radius:12px;gap:2px}" +
    ".geb-theme-btn{background:transparent;border:none;color:#cbd5e1;font-size:10px;font-weight:600;padding:2px 6px;border-radius:10px;cursor:pointer;line-height:1.2;transition:background 0.15s,color 0.15s}" +
    ".geb-theme-btn.active{background:#ffffff;color:#0f172a}" +
    ".geb-unified-toggle{border-radius:4px;width:22px;height:22px;line-height:1;font-size:14px;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;border:none}" +
    ".geb-unified-body{padding:10px 12px;max-height:82vh;overflow-y:auto}" +
    ".geb-menu-section{margin-bottom:10px;padding-bottom:9px}" +
    ".geb-menu-section:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none}" +
    ".geb-menu-label{display:block;font-size:11px;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.4px}" +
    ".geb-layers-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}" +
    ".geb-layers-header .geb-menu-label{margin-bottom:0}" +
    ".geb-freq-pill{display:flex;width:100%;border-radius:6px;padding:2px;gap:2px;box-sizing:border-box}" +
    ".geb-freq-btn{flex:1;background:transparent;border:none;font-size:11px;font-weight:600;padding:4px 6px;border-radius:4px;cursor:pointer;line-height:1.2;text-align:center;transition:background 0.15s,color 0.15s}" +
    ".geb-station-search__row{display:flex;gap:5px}" +
    ".geb-station-search__row input{flex:1;min-width:0;border-radius:5px;padding:5px 7px;font-size:12px}" +
    ".geb-station-search__row button{border:none;border-radius:5px;padding:5px 9px;font-size:11px;font-weight:600;cursor:pointer}" +
    ".geb-station-search__status{font-size:10px;margin-top:4px}" +
    ".geb-metric-dropdown{width:100%;padding:5px 7px;font-size:12px;font-weight:600;border-radius:6px;outline:none;cursor:pointer}" +
    ".geb-violin-container{margin-top:6px}" +
    ".geb-violin-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;gap:8px}" +
    ".geb-violin-stats{display:flex;align-items:center;flex-wrap:wrap;gap:6px;font-size:10px;line-height:1.2}" +
    ".geb-stat-bullet{margin:0 2px;opacity:0.4;font-weight:700;user-select:none}" +
    ".geb-modebar-btn{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;padding:0;border:none;background:transparent;cursor:pointer;border-radius:4px;transition:all .15s ease}" +
    ".geb-modebar-btn svg{fill:#475569;transition:fill .15s ease}" +
    ".geb-modebar-btn:hover{background:rgba(0,0,0,.06)}" +
    ".geb-modebar-btn:hover svg{fill:#0284c7}" +
    ".geb-violin-svg{display:block;width:100%;height:44px;border-radius:4px 4px 0 0;overflow:visible}" +
    ".geb-metric-gradient{border-radius:0 0 3px 3px;height:8px;margin-top:1px}" +
    ".geb-metric-ticks{display:flex;font-size:9px;justify-content:space-between;margin-top:3px;font-variant-numeric:tabular-nums}" +
    ".geb-metric-note{font-size:9px;line-height:1.3;margin-top:4px}" +
    ".geb-categorical-key{align-items:center;display:flex;font-size:11px;gap:7px;margin-top:5px}" +
    ".geb-categorical-key i{border-radius:50%;height:10px;width:10px}" +
    ".geb-availability-bar{display:flex;height:7px;border-radius:3px;overflow:hidden;margin-top:5px}" +
    ".geb-violin-export-row{display:flex;align-items:center;justify-content:space-between;margin-top:6px;padding-top:4px;border-top:1px dashed #e2e8f0}" +
    ".geb-export-label{font-size:10px;font-weight:600}" +
    ".geb-violin-actions{display:inline-flex;gap:4px}" +
    ".geb-export-btn{border-radius:4px;padding:2px 6px;font-size:10px;font-weight:600;cursor:pointer;line-height:1.2;transition:background 0.15s, border-color 0.15s}" +
    ".geb-basemap-row{display:flex;gap:12px;margin-bottom:6px;font-size:11px}" +
    ".geb-basemap-label{display:inline-flex;align-items:center;gap:4px;cursor:pointer;font-weight:500}" +
    ".geb-overlays-list{display:flex;flex-direction:column;gap:5px;font-size:11px}" +
    ".geb-overlay-item{display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:500}" +".geb-overlay-entry{display:flex;flex-direction:column}" +".geb-subpanel{margin:3px 0 5px 18px;padding:6px 8px;border-radius:6px;font-size:10px;line-height:1.4}" +".geb-subpanel-title{font-weight:700;margin-bottom:3px}" +".geb-subpanel-row{display:flex;align-items:center;gap:5px;margin:1px 0}" +".geb-subpanel-note{display:block;margin-top:4px;font-size:9px;line-height:1.3}" +".geb-snap-step{display:flex;align-items:center;gap:6px;margin:2px 0}" +".geb-snap-step i{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;color:white;font-style:normal;font-weight:bold;font-size:9px}" +".geb-snapping-status{margin-top:4px;font-size:9px;font-weight:600}" +
    // Dark Theme Styles
    "[data-geb-theme=\"dark\"] .geb-unified-menu{background:rgba(15,23,42,.98);border:1px solid #334155;color:#f8fafc}" +
    "[data-geb-theme=\"dark\"] .geb-unified-header{background:#020617;color:#f8fafc;border-bottom:1px solid #1e293b}" +
    "[data-geb-theme=\"dark\"] .geb-unified-toggle{background:#1e293b;color:#f8fafc}" +
    "[data-geb-theme=\"dark\"] .geb-menu-section{border-bottom:1px solid #1e293b}" +
    "[data-geb-theme=\"dark\"] .geb-menu-label{color:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-freq-pill{background:#1e293b;border:1px solid #334155}" +
    "[data-geb-theme=\"dark\"] .geb-freq-btn{color:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-freq-btn.active{background:#38bdf8;color:#0f172a}" +
    "[data-geb-theme=\"dark\"] .geb-freq-btn:hover:not(.active){color:#f8fafc}" +
    "[data-geb-theme=\"dark\"] .geb-station-search__row input{background:#1e293b;color:#f8fafc;border:1px solid #475569}" +
    "[data-geb-theme=\"dark\"] .geb-station-search__row button{background:#334155;color:#f8fafc}" +
    "[data-geb-theme=\"dark\"] .geb-station-search__status{color:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-metric-dropdown{background:#1e293b;color:#f8fafc;border:1px solid #475569}" +
    "[data-geb-theme=\"dark\"] .geb-violin-stats{color:#cbd5e1}" +
    "[data-geb-theme=\"dark\"] .geb-violin-stats b{color:#ffffff}" +
    "[data-geb-theme=\"dark\"] .geb-stat-bullet{color:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-modebar-btn svg{fill:#ffffff}" +
    "[data-geb-theme=\"dark\"] .geb-modebar-btn:hover{background:rgba(255,255,255,.12)}" +
    "[data-geb-theme=\"dark\"] .geb-modebar-btn:hover svg{fill:#38bdf8}" +
    "[data-geb-theme=\"dark\"] .geb-violin-svg{background:#1e293b;border:1px solid #334155}" +
    "[data-geb-theme=\"dark\"] .geb-violin-outline{stroke:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-violin-median{stroke:#ffffff}" +
    "[data-geb-theme=\"dark\"] .geb-metric-ticks{color:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-metric-note{color:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-categorical-key{color:#cbd5e1}" +
    "[data-geb-theme=\"dark\"] .geb-categorical-key i{border:1px solid #cbd5e1}" +
    "[data-geb-theme=\"dark\"] .geb-availability-bar{background:#1e293b}" +
    "[data-geb-theme=\"dark\"] .geb-violin-export-row{border-top-color:#334155}" +
    "[data-geb-theme=\"dark\"] .geb-export-label{color:#94a3b8}" +
    "[data-geb-theme=\"dark\"] .geb-export-btn{background:#1e293b;color:#f8fafc;border:1px solid #334155}" +
    "[data-geb-theme=\"dark\"] .geb-export-btn:hover{background:#334155;border-color:#475569}" +
    "[data-geb-theme=\"dark\"] .geb-basemap-row,[data-geb-theme=\"dark\"] .geb-overlays-list{color:#cbd5e1}" +"[data-geb-theme=\"dark\"] .geb-subpanel{background:#1e293b;border:1px solid #334155;color:#e2e8f0}" +"[data-geb-theme=\"dark\"] .geb-subpanel-note{color:#94a3b8}" +"[data-geb-theme=\"dark\"] .geb-snapping-status{color:#38bdf8}" +
    // Light Theme Styles
    "[data-geb-theme=\"light\"] .geb-unified-menu{background:rgba(255,255,255,.98);border:1px solid #cbd5e1;color:#0f172a}" +
    "[data-geb-theme=\"light\"] .geb-unified-header{background:#0f172a;color:#f8fafc;border-bottom:1px solid #334155}" +
    "[data-geb-theme=\"light\"] .geb-unified-toggle{background:#1e293b;color:#f8fafc}" +
    "[data-geb-theme=\"light\"] .geb-menu-section{border-bottom:1px solid #e2e8f0}" +
    "[data-geb-theme=\"light\"] .geb-menu-label{color:#334155}" +
    "[data-geb-theme=\"light\"] .geb-freq-pill{background:#f1f5f9;border:1px solid #cbd5e1}" +
    "[data-geb-theme=\"light\"] .geb-freq-btn{color:#64748b}" +
    "[data-geb-theme=\"light\"] .geb-freq-btn.active{background:#0284c7;color:#ffffff}" +
    "[data-geb-theme=\"light\"] .geb-freq-btn:hover:not(.active){color:#0f172a}" +
    "[data-geb-theme=\"light\"] .geb-station-search__row input{background:#f8fafc;color:#0f172a;border:1px solid #cbd5e1}" +
    "[data-geb-theme=\"light\"] .geb-station-search__row button{background:#1e293b;color:#f8fafc}" +
    "[data-geb-theme=\"light\"] .geb-station-search__status{color:#64748b}" +
    "[data-geb-theme=\"light\"] .geb-metric-dropdown{background:#f8fafc;color:#0f172a;border:1px solid #94a3b8}" +
    "[data-geb-theme=\"light\"] .geb-violin-stats{color:#475569}" +
    "[data-geb-theme=\"light\"] .geb-violin-stats b{color:#0f172a}" +
    "[data-geb-theme=\"light\"] .geb-stat-bullet{color:#94a3b8}" +
    "[data-geb-theme=\"light\"] .geb-modebar-btn svg{fill:#475569}" +
    "[data-geb-theme=\"light\"] .geb-modebar-btn:hover{background:rgba(15,23,42,.08)}" +
    "[data-geb-theme=\"light\"] .geb-modebar-btn:hover svg{fill:#0f172a}" +
    "[data-geb-theme=\"light\"] .geb-violin-svg{background:#f8fafc;border:1px solid #e2e8f0}" +
    "[data-geb-theme=\"light\"] .geb-violin-outline{stroke:#1e293b}" +
    "[data-geb-theme=\"light\"] .geb-violin-median{stroke:#0f172a}" +
    "[data-geb-theme=\"light\"] .geb-metric-ticks{color:#475569}" +
    "[data-geb-theme=\"light\"] .geb-metric-note{color:#64748b}" +
    "[data-geb-theme=\"light\"] .geb-categorical-key{color:#334155}" +
    "[data-geb-theme=\"light\"] .geb-categorical-key i{border:1px solid #94a3b8}" +
    "[data-geb-theme=\"light\"] .geb-availability-bar{background:#e2e8f0}" +
    "[data-geb-theme=\"light\"] .geb-violin-export-row{border-top-color:#e2e8f0}" +
    "[data-geb-theme=\"light\"] .geb-export-label{color:#475569}" +
    "[data-geb-theme=\"light\"] .geb-export-btn{background:#f1f5f9;color:#0f172a;border:1px solid #cbd5e1}" +
    "[data-geb-theme=\"light\"] .geb-export-btn:hover{background:#e2e8f0;border-color:#94a3b8}" +
    "[data-geb-theme=\"light\"] .geb-basemap-row,[data-geb-theme=\"light\"] .geb-overlays-list{color:#1e293b}" +"[data-geb-theme=\"light\"] .geb-subpanel{background:#f1f5f9;border:1px solid #e2e8f0;color:#1e293b}" +"[data-geb-theme=\"light\"] .geb-subpanel-note{color:#64748b}" +"[data-geb-theme=\"light\"] .geb-snapping-status{color:#0284c7}";
  document.head.appendChild(style);

  renderMenu();
})();
