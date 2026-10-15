(async function() {
    const map = {{ this._parent.get_name() }};
    const config = {{ this.data | script_json }};
    const style = document.createElement('style');
    style.textContent = `
      :root{--ww-ink:#142637;--ww-muted:#64748b;--ww-line:#dbe4ec;--ww-teal:#087f8c}
      .ww{font:14px/1.5 system-ui,-apple-system,sans-serif;color:var(--ww-ink);box-sizing:border-box}
      .ww.ww *{box-sizing:border-box}.ww button,.ww input,.ww select{font:inherit}
      .ww button,.ww select{cursor:pointer}.ww button:focus-visible,.ww input:focus-visible,.ww a:focus-visible{outline:3px solid #38bdf8;outline-offset:2px}
      .ww-shell{position:fixed;inset:0 auto 0 0;width:370px;background:#fff;z-index:1000;display:flex;flex-direction:column;box-shadow:2px 0 15px #14263718}
      .ww-head{padding:22px 24px 16px;border-bottom:1px solid var(--ww-line)}
      .ww-nav{display:flex;gap:16px;font-size:12px;margin-bottom:18px}.ww-nav a{color:var(--ww-muted);text-decoration:none}.ww-nav strong{color:var(--ww-teal)}
      .ww h1{font-size:25px;letter-spacing:-.8px;line-height:1.2;margin:0 0 8px}.ww h2{font-size:18px;margin:0 0 8px}.ww p{margin:8px 0}.ww-small{font-size:12px;color:var(--ww-muted)}
      .ww-scroll{overflow:auto;flex:1;padding:18px 24px 120px}.ww-search{width:100%;padding:11px 12px;border:1px solid var(--ww-line);border-radius:8px;background:#f8fafc}
      .ww-filters{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}.ww-chip{border:1px solid var(--ww-line);background:#fff;border-radius:20px;padding:5px 10px;font-size:12px!important}.ww-chip[aria-pressed=true]{background:#142637;color:white;border-color:#142637}
      .ww-count{font-size:12px;color:var(--ww-muted);margin-bottom:12px}.ww-list{display:flex;flex-direction:column;gap:6px;max-height:245px;overflow:auto}
      .ww-item{width:100%;display:flex;align-items:center;gap:10px;text-align:left;background:#fff;border:1px solid var(--ww-line);border-radius:8px;padding:9px 11px}.ww-item:hover{background:#f1f7f9}.ww-item[aria-current=true]{border-color:var(--ww-teal);background:#eef9fa}.ww-dot{flex:none;width:11px;height:11px;border-radius:50%;background:var(--dot)}.ww-item small{display:block;color:var(--ww-muted)}
      .ww-details{margin-top:20px;border-top:1px solid var(--ww-line);padding-top:18px}.ww-badge{display:inline-block;font-size:12px;padding:4px 9px;border-radius:5px;background:#eef2f6;margin-bottom:12px}
      .ww-stats{display:grid;grid-template-columns:1fr 1fr;gap:10px}.ww-stat{background:#f4f8fa;border-radius:8px;padding:11px}.ww-stat b{display:block;font-size:21px;letter-spacing:-.5px}.ww-stat span{font-size:11px;color:var(--ww-muted)}
      .ww-opening{margin:12px 0}.ww-track{height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden}.ww-track div{height:100%;background:var(--ww-teal);transition:width .15s}
      .ww-chart{width:100%;height:135px;display:block;cursor:crosshair}.ww-chart-label{font-size:12px;font-weight:600;margin-top:14px}.ww-empty{padding:16px;background:#f4f8fa;border-radius:8px;color:var(--ww-muted)}
      .ww-time{position:fixed;bottom:22px;left:394px;right:24px;z-index:1000;background:#fff;border:1px solid var(--ww-line);border-radius:12px;padding:14px 20px;box-shadow:0 4px 24px #14263722}
      .ww-time-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.ww-time button{border:1px solid var(--ww-line);background:#f4f8fa;padding:7px 12px;border-radius:7px}.ww-time .ww-play{background:var(--ww-teal);color:white;border-color:var(--ww-teal);min-width:75px}.ww-time input[type=range]{width:100%;accent-color:var(--ww-teal);margin:12px 0 0}.ww-time input[type=datetime-local]{border:1px solid var(--ww-line);border-radius:5px;padding:5px;max-width:220px}.ww-time select{border:1px solid var(--ww-line);border-radius:5px;padding:5px;background:white}.ww-date{font-size:16px;font-weight:650;flex:1}
      .ww-legend{position:fixed;right:24px;top:22px;z-index:999;padding:12px 16px;background:#fffffff5;border-radius:9px;box-shadow:0 3px 18px #14263718;font-size:12px}.ww-legend summary{cursor:pointer;font-weight:650}.ww-legend div{display:flex;align-items:center;gap:8px;margin:4px 0}
      .leaflet-bottom.leaflet-right{bottom:155px}.ww-marker{background:none;border:0}.ww-marker-inner{min-width:42px;min-height:30px;display:flex;align-items:center;justify-content:center;padding:3px 6px;border-radius:7px;color:white;background:var(--dot);border:2px solid white;box-shadow:0 2px 7px #0004;font:700 11px system-ui}.ww-marker-selected .ww-marker-inner{outline:3px solid #142637}.ww-marker-opening .ww-marker-inner{box-shadow:0 0 0 5px #22c55e55,0 2px 7px #0004}
      @media(min-width:761px){ #${map.getContainer().id}{left:370px!important;width:calc(100% - 370px)!important}}
      @media(max-width:760px){.ww-shell{width:100%;height:44%;inset:0 0 auto;box-shadow:0 2px 15px #14263718}.ww-head{padding:12px 16px}.ww-nav{margin-bottom:6px}.ww h1{font-size:20px}.ww-head p{display:none}.ww-scroll{padding:12px 16px}.ww-list{max-height:100px}.ww-time{left:12px;right:12px;bottom:20px;padding:10px 12px}.ww-legend{top:calc(44% + 12px);right:12px;padding:8px;font-size:10px} #${map.getContainer().id}{top:44%!important;height:56%!important}.leaflet-bottom.leaflet-right{bottom:210px}.ww-date{font-size:13px}.ww-time input[type=datetime-local]{max-width:185px}}
    `;
    document.head.appendChild(style);
    const shell = document.createElement('aside');
    shell.className = 'ww ww-shell';
    shell.innerHTML =
        `<header class="ww-head"><nav class="ww-nav"><a id="ww-back">← Discharge dashboard</a><strong>AMBER waterworks</strong></nav><h1>Waterworks in motion</h1><p class="ww-small">Explore modeled barriers, their openings and the water flowing through them.</p></header><div class="ww-scroll"><input class="ww-search" type="search" aria-label="Search AMBER ID or structure type" placeholder="Search ID or structure type…"><div class="ww-filters" role="group" aria-label="Filter waterworks"></div><div class="ww-count" aria-live="polite">Loading waterworks…</div><div class="ww-list"></div><section class="ww-details"><div class="ww-empty">Select a structure on the map or in the list to inspect its operation and flow.</div></section><p class="ww-small">GEB simulation · Gate opening is a modeled fraction of the dam width, not an observed gate position. Fixed means no gate operation in this model.</p></div>`;
    document.body.appendChild(shell);
    shell.querySelector('#ww-back').href = config.discharge_url;
    const controls = document.createElement('section');
    controls.className = 'ww ww-time';
    controls.setAttribute('aria-label', 'Simulation timeline');
    controls.innerHTML =
        `<div class="ww-time-head"><button class="ww-play" disabled aria-label="Play timeline">▶ Play</button><button class="ww-prev" disabled aria-label="Previous timestamp">‹</button><button class="ww-next" disabled aria-label="Next timestamp">›</button><span class="ww-date">Loading simulation…</span><input type="datetime-local" aria-label="Jump to simulation date (UTC)" disabled><select aria-label="Playback step"><option value="1">1 hour / step</option><option value="24">1 day / step</option><option value="168">1 week / step</option></select></div><input type="range" min="0" max="0" value="0" aria-label="Simulation time" disabled><div class="ww-small ww-time-note">Opening: end-of-hour snapshot · Flow: hourly river discharge · Dates: UTC</div>`;
    document.body.appendChild(controls);
    const legend = document.createElement('div');
    legend.className = 'ww ww-legend';
    legend.innerHTML =
        `<details open><summary>Modeled operation</summary><div><i class="ww-dot" style="--dot:#15803d"></i>100% open</div><div><i class="ww-dot" style="--dot:#087f8c"></i>Partly open (10–100%)</div><div><i class="ww-dot" style="--dot:#b45309"></i>Minimum opening (10%)</div><div><i class="ww-dot" style="--dot:#475569"></i>Fixed barrier ◆</div><div><i class="ww-dot" style="--dot:#94a3b8"></i>Unavailable / excluded</div><div>Green halo: opening increased</div></details>`;
    document.body.appendChild(legend);
    legend.querySelector("details").open = window.innerWidth > 760;
    L.control.zoom({
        position: 'bottomright'
    }).addTo(map);
    map.invalidateSize();
    let payload;
    try {
        const bytes = Uint8Array.from(atob(config.payload), c => c.charCodeAt(0));
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(
            'gzip'));
        payload = JSON.parse(await new Response(stream).text());
    } catch (error) {
        shell.querySelector('.ww-count').textContent =
            'Waterworks data could not be loaded.';
        controls.querySelector('.ww-date').textContent = 'Timeline unavailable';
        shell.querySelector('.ww-details').textContent =
            'Use a current browser with gzip decompression support.';
        console.error(error);
        return;
    }
    const structures = payload.structures,
        times = payload.timeline;
    const markers = [],
        layer = L.layerGroup().addTo(map);
    let index = 0,
        selected = -1,
        filter = 'modeled',
        query = '',
        timer = null;
    const colors = {
        open: '#15803d',
        partial: '#087f8c',
        minimum: '#b45309',
        fixed: '#475569',
        unknown: '#94a3b8',
        excluded: '#94a3b8'
    };
    const escape = value => String(value).replace(/[&<>"']/g, c => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    } [c]));
    const value = (s, key, at = index) => s.series[key]?.[at] ?? null;
    const opening = s => {
        const v = value(s, 'open_fraction');
        return v !== null && v >= 0 && v <= 1 ? v : null;
    };
    const status = s => s.operation === 'gate' ? (opening(s) === null ? 'unknown' : opening(
            s) >= .9999 ? 'open' : opening(s) <= .1001 ? 'minimum' : 'partial') : s
        .operation;
    const label = s => ({
        open: 'Fully open',
        partial: 'Partly open',
        minimum: 'Minimum opening',
        fixed: 'Fixed barrier',
        unknown: 'Runtime data unavailable',
        excluded: 'Excluded from model'
    } [status(s)]);
    const format = v => v === null ? 'Unavailable' : new Intl.NumberFormat('en', {
        maximumFractionDigits: 2
    }).format(v);
    const stamp = t => new Date(t).toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
    const visible = s => {
        const match = (s.id + ' ' + s.type).toLowerCase().includes(query);
        return match && (filter === 'all' || filter === 'modeled' && s.included ||
            filter === 'gate' && s.operation === 'gate' || filter === 'fixed' && s
            .operation === 'fixed' || filter === 'excluded' && !s.included);
    };
    for (const [key, text] of [
            ['modeled', 'Modeled'],
            ['gate', 'Gated'],
            ['fixed', 'Fixed'],
            ['excluded', 'Excluded'],
            ['all', 'All']
        ]) {
        const button = document.createElement('button');
        button.className = 'ww-chip';
        button.textContent = text;
        button.dataset.filter = key;
        button.setAttribute('aria-pressed', String(key === filter));
        button.onclick = () => {
            filter = key;
            shell.querySelectorAll('.ww-chip').forEach(b => b.setAttribute(
                'aria-pressed', String(b.dataset.filter === key)));
            render();
        };
        shell.querySelector('.ww-filters').appendChild(button);
    }
    structures.forEach((s, i) => {
        const marker = L.marker([s.lat, s.lon], {
            keyboard: true
        });
        marker.bindTooltip('', {
            direction: 'top',
            offset: [0, -12]
        });
        marker.on('click', () => select(i));
        markers.push(marker);
    });
    /** Select a catalog object and expose its modeled diagnostics. @param {number} i Structure index. @returns {void} */
    function select(i) {
        selected = i;
        map.panTo([structures[i].lat, structures[i].lon]);
        render();
        shell.querySelector('.ww-details').scrollIntoView({
            block: 'nearest',
            behavior: 'smooth'
        });
    }
    /** Draw gap-aware histories with a time cursor, preserving peaks during reduction.
     * @param {HTMLCanvasElement} canvas Chart surface.
     * @param {Object} s Structure and its reported series.
     * @param {string[]} keys Report quantities to plot.
     * @param {string[]} palette Line colors.
     * @returns {void}
     */
    function plot(canvas, s, keys, palette) {
        const width = canvas.clientWidth || 300,
            height = 135,
            ratio = window.devicePixelRatio || 1;
        canvas.width = width * ratio;
        canvas.height = height * ratio;
        const ctx = canvas.getContext('2d');
        ctx.scale(ratio, ratio);
        const left = 42,
            right = width - 8,
            top = 12,
            bottom = 106;
        let maximum = 0,
            minimum = 0;
        for (const key of keys)
            for (const v of s.series[key] || [])
                if (v !== null) {
                    maximum = Math.max(maximum, v);
                    minimum = Math.min(minimum, v);
                }
        if (keys[0] === 'open_fraction') {
            maximum = 1;
            minimum = 0;
        }
        if (maximum === minimum) maximum = minimum + 1;
        const x = i => left + (right - left) * (times.length > 1 ? (times[i] - times[0]) / (
            times.at(-1) - times[0]) : 0);
        const y = v => bottom - (bottom - top) * (v - minimum) / (maximum - minimum);
        ctx.font = '10px system-ui';
        ctx.fillStyle = '#64748b';
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        for (let j = 0; j < 3; j++) {
            const v = minimum + (maximum - minimum) * j / 2;
            ctx.beginPath();
            ctx.moveTo(left, y(v));
            ctx.lineTo(right, y(v));
            ctx.stroke();
            ctx.fillText(keys[0] === 'open_fraction' ? Math.round(v * 100) + '%' : format(
                v), 0, y(v) + 3);
        }
        keys.forEach((key, k) => {
            const values = s.series[key] || [];
            ctx.strokeStyle = palette[k];
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            let connected = false;
            // Keep each pixel's extrema so short discharge peaks remain visible.
            let bucket = -1,
                entries = [];
            const flush = () => {
                if (!entries.length) return;
                const chosen = [entries[0], entries.reduce((a, b) => a[1] < b[
                    1] ? a : b), entries.reduce((a, b) => a[1] > b[1] ?
                    a : b), entries.at(-1)].sort((a, b) => a[0] - b[0]);
                for (const [i, v] of chosen) {
                    if (connected) ctx.lineTo(x(i), y(v));
                    else ctx.moveTo(x(i), y(v));
                    connected = true;
                }
                entries = [];
            };
            values.forEach((v, i) => {
                if (v === null || v < 0 && key === 'open_fraction') {
                    flush();
                    connected = false;
                    return;
                }
                const pixel = Math.floor(x(i));
                if (pixel !== bucket) {
                    flush();
                    bucket = pixel;
                }
                entries.push([i, v]);
            });
            flush();
            ctx.stroke();
        });
        ctx.strokeStyle = '#142637';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x(index), top);
        ctx.lineTo(x(index), bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#64748b';
        if (times.length) {
            ctx.fillText(new Date(times[0]).toISOString().slice(0, 10), left, 126);
            ctx.textAlign = 'right';
            ctx.fillText(new Date(times.at(-1)).toISOString().slice(0, 10), right, 126);
        }
        canvas.onclick = event => {
            const fraction = Math.max(0, Math.min(1, (event.offsetX - left) / (right -
                left)));
            const timestamp = times[0] + fraction * (times.at(-1) - times[0]);
            setIndex(nearest(timestamp));
        };
    }
    /** Update the selected structure, including units and missing-data explanations. @returns {void} */
    function details() {
        const target = shell.querySelector('.ww-details');
        if (selected < 0) return;
        const s = structures[selected],
            v = opening(s);
        const description = s.operation === 'gate' ?
            'The modeled opening grows from 10% to 100% as upstream depth rises from 65% to 90% of bankfull depth.' :
            s.operation === 'fixed' ?
            'This structure has a fixed crest in GEB. Water can overflow it; its opening does not change.' :
            !s.included ?
            'This catalog structure is excluded from river-barrier routing. It may be represented as a lake or reservoir.' :
            'Rebuild setup_weirs and rerun the simulation with report._waterworks enabled to inspect operation.';
        target.innerHTML =
            `<button class="ww-chip ww-back-list">↑ Back to structures</button><div class="ww-small" style="margin-top:12px">AMBER · ${escape(s.type)}</div><h2>Structure ${escape(s.id)}</h2><span class="ww-badge">${escape(label(s))}</span><div class="ww-stats"><div class="ww-stat"><b>${format(value(s,'outflow_m3_s'))}</b><span>Barrier outflow · m³/s</span></div><div class="ww-stat"><b>${format(value(s,'inflow_m3_s'))}</b><span>Upstream river inflow · m³/s</span></div></div>${s.operation==='gate'?`<div class="ww-opening"><strong>${v===null?'Opening unavailable':Math.round(v*100)+'% open'}</strong><div class="ww-track"><div style="width:${v===null?0:v*100}%"></div></div></div>`:''}<p class="ww-small">${escape(description)}</p>${s.height!==null?`<p class="ww-small">Effective crest height: ${format(s.height)} m above the river bed.</p>`:''}${!s.included?`<p class="ww-small">Exclusion: ${escape(s.reason||'No reason recorded')}</p>`:''}<div class="ww-chart-label">Flow history · m³/s</div><div class="ww-small"><span style="color:#087f8c">━ Outflow</span> &nbsp; <span style="color:#b45309">━ Upstream inflow</span></div><canvas class="ww-chart ww-flow" role="img" aria-label="Flow history; click to select a time"></canvas>${s.operation==='gate'?'<div class="ww-chart-label">Opening history</div><canvas class="ww-chart ww-gates" role="img" aria-label="Opening history; click to select a time"></canvas>':''}<p class="ww-small">Outflow combines passage through the opening and over the crest. Inflow sums immediate upstream river links and excludes local runoff, abstractions and return flows. Click a history to move the time slider.</p>`;
        target.querySelector(".ww-back-list").onclick = () => shell.querySelector(
            ".ww-scroll").scrollTo({
            top: 0,
            behavior: "smooth"
        });
        if (times.length) {
            plot(target.querySelector('.ww-flow'), s, ['outflow_m3_s', 'inflow_m3_s'], [
                '#087f8c', '#b45309'
            ]);
            if (s.operation === 'gate') plot(target.querySelector('.ww-gates'), s, [
                'open_fraction'
            ], ['#087f8c']);
        }
    }
    /** Synchronize map states, the filtered list and the selected time. @returns {void} */
    function render() {
        let visibleCount = 0,
            openingCount = 0;
        const list = shell.querySelector('.ww-list');
        list.replaceChildren();
        structures.forEach((s, i) => {
            const shown = visible(s),
                v = opening(s),
                previous = value(s, 'open_fraction', index - 1);
            const increased = s.operation === 'gate' && v !== null && index > 0 &&
                previous !== null && previous >= 0 && v > previous + .001;
            if (increased && shown) openingCount++;
            if (shown) {
                visibleCount++;
                if (!layer.hasLayer(markers[i])) layer.addLayer(markers[i]);
            } else layer.removeLayer(markers[i]);
            if (!shown) return;
            const color = colors[status(s)] || colors.unknown;
            const markerText = s.operation === 'fixed' ? '◆' : v !== null ? Math
                .round(v * 100) + '%' : s.included ? '?' : '×';
            const iconKey = markerText + ':' + color + ':' + (i === selected) +
                ':' + increased;
            // Fixed barriers rarely change; avoid replacing their DOM during playback.
            if (markers[i]._waterworksIconKey !== iconKey) {
                markers[i].setIcon(L.divIcon({
                    className: 'ww-marker' + (i === selected ?
                        ' ww-marker-selected' : '') + (increased ?
                        ' ww-marker-opening' : ''),
                    html: `<div class="ww-marker-inner" style="--dot:${color}">${markerText}</div>`,
                    iconSize: [46, 32],
                    iconAnchor: [23, 16]
                }));
                markers[i]._waterworksIconKey = iconKey;
            }
            markers[i].setTooltipContent(
                `<strong>AMBER ${escape(s.id)}</strong><br>${escape(s.type)} · ${label(s)}<br>${v!==null?Math.round(v*100)+'% open · ':''}${format(value(s,'outflow_m3_s'))} m³/s outflow`
            );
            const element = markers[i].getElement();
            if (element) element.setAttribute('aria-label',
                `AMBER ${s.id}, ${s.type}, ${label(s)}`);
            // Bound the sidebar while all matching structures remain on the map.
            if (list.children.length < 100) {
                const button = document.createElement('button');
                button.className = 'ww-item';
                button.setAttribute('aria-current', String(i === selected));
                button.innerHTML =
                    `<i class="ww-dot" style="--dot:${color}"></i><span><strong>${escape(s.id)}</strong> · ${escape(s.type)}<small>${label(s)}${v!==null?' · '+Math.round(v*100)+'%':''}</small></span>`;
                button.onclick = () => select(i);
                list.appendChild(button);
            }
        });
        shell.querySelector('.ww-count').textContent =
            `${visibleCount} structures shown · ${openingCount} opening since previous sample${visibleCount>100?' · first 100 listed':''}`;
        if (!visibleCount) list.innerHTML =
            '<div class="ww-empty">No matching structures. Try another filter or search.</div>';
        details();
        if (times.length) {
            controls.querySelector('.ww-date').textContent = stamp(times[index]);
            controls.querySelector('[type=range]').value = index;
            controls.querySelector('[type=datetime-local]').value = new Date(times[index])
                .toISOString().slice(0, 16);
        }
    }
    /** Find the closest report timestamp by binary search. @param {number} timestamp Epoch milliseconds. @returns {number} Timeline index. */
    function nearest(timestamp) {
        let low = 0,
            high = times.length - 1;
        while (low < high) {
            const middle = (low + high) >> 1;
            if (times[middle] < timestamp) low = middle + 1;
            else high = middle;
        }
        return low > 0 && timestamp - times[low - 1] < times[low] - timestamp ? low - 1 :
            low;
    }
    /** Stop playback and restore the play button. @returns {void} */
    function stop() {
        if (timer) clearInterval(timer);
        timer = null;
        controls.querySelector('.ww-play').textContent = '▶ Play';
    }
    /** Select a bounded timeline sample and stop at the final sample. @param {number} next Requested index. @returns {void} */
    function setIndex(next) {
        index = Math.max(0, Math.min(times.length - 1, next));
        if (index === times.length - 1) stop();
        render();
    }
    shell.querySelector('.ww-search').oninput = event => {
        query = event.target.value.toLowerCase().trim();
        render();
    };
    controls.querySelector('[type=range]').oninput = event => {
        stop();
        setIndex(Number(event.target.value));
    };
    controls.querySelector('[type=datetime-local]').onchange = event => {
        const date = Date.parse(event.target.value + 'Z');
        if (Number.isFinite(date)) {
            stop();
            setIndex(nearest(date));
        }
    };
    controls.querySelector('.ww-prev').onclick = () => {
        stop();
        setIndex(index - 1);
    };
    controls.querySelector('.ww-next').onclick = () => {
        stop();
        setIndex(index + 1);
    };
    controls.querySelector('.ww-play').onclick = () => {
        if (timer) {
            stop();
            return;
        }
        if (index === times.length - 1) index = 0;
        controls.querySelector('.ww-play').textContent = 'Ⅱ Pause';
        timer = setInterval(() => {
            const hours = Number(controls.querySelector('select').value);
            const next = Math.max(index + 1, nearest(times[index] + hours *
                3600000));
            setIndex(next);
        }, 450);
    };
    if (times.length) {
        controls.querySelectorAll('button,input').forEach(element => element.disabled =
            false);
        controls.querySelector('[type=range]').max = times.length - 1;
        controls.querySelector('[type=datetime-local]').min = new Date(times[0])
            .toISOString().slice(0, 16);
        controls.querySelector('[type=datetime-local]').max = new Date(times.at(-1))
            .toISOString().slice(0, 16);
    } else {
        controls.querySelector('.ww-date').textContent = 'No hourly reports available';
        controls.querySelector('.ww-time-note').textContent =
            'Rebuild setup_weirs and rerun with report._waterworks: true. Catalog inspection is available.';
    }
    if (!structures.length) shell.querySelector('.ww-details').innerHTML =
        '<div class="ww-empty">No AMBER structures are available for this model region.</div>';
    render();
    window.addEventListener('resize', () => {
        map.invalidateSize();
        details();
    });
    window.addEventListener('pagehide', stop);
})();
