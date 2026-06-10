"""Functions for creating interactive Folium discharge evaluation maps."""

import base64
import json
import math
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium import MacroElement, TileLayer
from jinja2 import Template

_ESRI_TOPO_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_TOPO_ATTR = (
    "Sources: Esri, HERE, Garmin, Intermap, INCREMENT P, GEBCO, USGS, FAO, "
    "NPS, NRCan, GeoBase, IGN, Kadaster NL, Ordnance Survey, Esri Japan, "
    "METI, Mapwithyou, NOSTRA, © OpenStreetMap contributors, and the GIS "
    "user community"
)


def _inject_station_images_macro(
    m: folium.Map,
    station_images: dict[str, dict[str, str]],
) -> None:
    """Inject a JS global containing station image data and a lazy-load handler.

    Images are stored once in ``window._stationImages`` and populated into
    popup ``<img>`` placeholders only when the popup is opened via a Leaflet
    ``popupopen`` event.

    Args:
        m: Folium map to inject the macro into.
        station_images: Mapping of station ID string to a dict with keys
            ``"returnPeriod"`` and ``"timeSeries"`` holding
            ``data:image/png;base64,…`` URIs.
    """
    images_js = json.dumps(station_images)

    class _StationImagesMacro(MacroElement):
        def __init__(self, images: str) -> None:
            super().__init__()
            self._template = Template(
                "{%- macro script(this, kwargs) -%}\n"
                "window._stationImages=" + images + ";\n"
                "{{this._parent.get_name()}}.on('popupopen', function(e) {\n"
                "  var content = e.popup.getContent();\n"
                "  if (!content || !content.querySelector) return;\n"
                "  var el = content.querySelector('[data-station-id]');\n"
                "  if (!el) return;\n"
                "  var sid = el.getAttribute('data-station-id');\n"
                "  var data = window._stationImages[sid];\n"
                "  if (!data) return;\n"
                "  var rp = el.querySelector('.rp-img');\n"
                "  var ts = el.querySelector('.ts-img');\n"
                "  if (rp) rp.src = data.returnPeriod;\n"
                "  if (ts) ts.src = data.timeSeries;\n"
                "});\n"
                "{%- endmacro -%}"
            )

    _StationImagesMacro(images_js).add_to(m)


def _build_metric_colormaps() -> tuple[
    cm.LinearColormap, cm.LinearColormap, cm.LinearColormap
]:
    """Create the standard KGE, NSE, and KGE-correlation colormaps (0 red → 1 green).

    Returns:
        Tuple of (colormap_correlation, colormap_kge, colormap_nse).
    """
    colors = ["red", "orange", "yellow", "blue", "green"]
    colormap_correlation = cm.LinearColormap(
        colors=colors, vmin=0, vmax=1, caption="KGE correlation"
    )
    colormap_kge = cm.LinearColormap(colors=colors, vmin=0, vmax=1, caption="KGE")
    colormap_nse = cm.LinearColormap(colors=colors, vmin=0, vmax=1, caption="NSE")
    return colormap_correlation, colormap_kge, colormap_nse


def create_discharge_folium_map(
    evaluation_gdf: gpd.GeoDataFrame,
    output_path: Path,
    timeseries_plot_folder: Path,
    return_period_plot_folder: Path,
    region_geom: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    waterbodies: gpd.GeoDataFrame | None = None,
) -> folium.Map:
    """Create an interactive Folium discharge evaluation map.

    Stations are shown as circle markers coloured by KGE, NSE, or KGE correlation
    (switchable via layer control) and sized by upstream area.  River widths
    are scaled by mean discharge.  An optional upstream-area-ratio layer is
    included when all stations have the ratio available.  Station PNG plots
    are lazy-loaded via a JS global to avoid duplicating large base64 strings
    across metric layers.  Lakes and reservoirs are rendered as dot markers
    when ``waterbodies`` is provided.

    Args:
        evaluation_gdf: Per-station GeoDataFrame with columns ``KGE``,
            ``NSE``, ``KGE_correlation``, ``upstream_area_GEB``,
            ``discharge_observations_to_GEB_upstream_area_ratio``, and a
            point geometry.
        output_path: Full path (including filename) where the HTML file is
            saved.
        timeseries_plot_folder: Directory containing
            ``timeseries_plot_<id>.png`` for each station.
        return_period_plot_folder: Directory containing
            ``return_period_fit_<id>.png`` for each station.
        region_geom: Basin/region boundary GeoDataFrame used to fit the map
            extent and render the catchment outline.
        rivers: River network GeoDataFrame with a ``discharge_m3_per_s``
            column used to scale river line widths.
        waterbodies: Optional GeoDataFrame with columns ``waterbody_type``
            (1 = lake, 2 = reservoir, 3 = lake control) and polygon
            geometries. Centroids are used for dot placement.

    Returns:
        The Folium map object (already saved to ``output_path``).
    """
    min_lon, min_lat, max_lon, max_lat = region_geom.total_bounds
    map_center: list[float] = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    m = folium.Map(
        location=map_center,
        tiles=TileLayer(
            tiles=_ESRI_TOPO_TILES,
            attr=_ESRI_TOPO_ATTR,
            name="Topographic Map",
        ),
    )
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(30, 30))

    folium.GeoJson(
        region_geom,
        name="Catchment",
        style_function=lambda x: {"fillColor": "none", "color": "black", "weight": 2},
        z_index=1,
    ).add_to(m)

    max_discharge = np.nanmax(rivers["discharge_m3_per_s"])
    max_discharge_sqrt: float | None = (
        None if np.isnan(max_discharge) else math.sqrt(max_discharge.item())
    )
    min_line_weight, max_line_weight = 0.5, 5.0

    def _river_style(feature: dict) -> dict:
        discharge = feature["properties"]["discharge_m3_per_s"]
        if discharge is None or max_discharge_sqrt is None:
            return {"color": "gray", "weight": min_line_weight}
        return {
            "color": "blue",
            "weight": (
                math.sqrt(discharge)
                / max_discharge_sqrt
                * (max_line_weight - min_line_weight)
                + min_line_weight
            ),
        }

    folium.GeoJson(
        rivers[["geometry", "discharge_m3_per_s"]].to_json(),
        name="Rivers",
        style_function=_river_style,
        z_index=2,
    ).add_to(m)

    colormap_correlation, colormap_kge, colormap_nse = _build_metric_colormaps()

    layer_upstream: folium.FeatureGroup | None = None
    colormap_upstream: cm.LinearColormap | None = None
    if (
        not evaluation_gdf["discharge_observations_to_GEB_upstream_area_ratio"]
        .isna()
        .any()
    ):
        colormap_upstream = cm.LinearColormap(
            colors=["red", "orange", "yellow", "blue", "green"],
            vmin=0.5,
            vmax=2.0,
            caption="Upstream Area Ratio",
        )
        layer_upstream = folium.FeatureGroup(name="Upstream Area Ratio", show=False)

    largest_upstream_area_sqrt: float = math.sqrt(
        evaluation_gdf["upstream_area_GEB"].max()
    )

    layer_kge = folium.FeatureGroup(name="KGE", show=True)
    layer_nse = folium.FeatureGroup(name="NSE", show=False)
    layer_correlation = folium.FeatureGroup(name="KGE correlation", show=False)

    popup_width = 800

    # Encode each station's PNG images as base64 and store them in a dict that is
    # injected as a single JS global.
    station_images: dict[str, dict[str, str]] = {}

    for station_id, row in evaluation_gdf.iterrows():
        coords: list[float] = [row.geometry.y, row.geometry.x]

        rp_path = return_period_plot_folder / f"return_period_fit_{station_id}.png"
        ts_path = timeseries_plot_folder / f"timeseries_plot_{station_id}.png"
        with open(rp_path, "rb") as img_file:
            encoded_rp = base64.b64encode(img_file.read()).decode("utf-8")
        with open(ts_path, "rb") as img_file:
            encoded_ts = base64.b64encode(img_file.read()).decode("utf-8")

        station_images[str(station_id)] = {
            "returnPeriod": f"data:image/png;base64,{encoded_rp}",
            "timeSeries": f"data:image/png;base64,{encoded_ts}",
        }

        # Popup HTML contains only empty placeholders. The popupopen handler
        # (injected by _inject_station_images_macro) fills src on demand.
        popup_html = (
            f"<div data-station-id='{station_id}' style='width:{popup_width}px;'>"
            f"<img class='rp-img' style='width:100%;height:auto;display:block;'>"
            f"<img class='ts-img' style='width:100%;height:auto;display:block;'>"
            f"</div>"
        )

        # Scale circle radius by upstream area (range 5–10 px).
        circle_radius: float = (
            5 + math.sqrt(row["upstream_area_GEB"]) / largest_upstream_area_sqrt * 5
        )

        for layer, colormap, metric in [
            (layer_correlation, colormap_correlation, "KGE_correlation"),
            (layer_kge, colormap_kge, "KGE"),
            (layer_nse, colormap_nse, "NSE"),
        ]:
            fill_color = colormap(row[metric]) if pd.notna(row[metric]) else "gray"
            folium.CircleMarker(
                location=coords,
                radius=circle_radius,
                color="black",
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=popup_width),
                z_index=1000,
            ).add_to(layer)

        if layer_upstream is not None and colormap_upstream is not None:
            color_upstream = colormap_upstream(
                float(row["discharge_observations_to_GEB_upstream_area_ratio"])
            )
            if not isinstance(color_upstream, str) or color_upstream == "nan":
                continue
            folium.CircleMarker(
                location=coords,
                radius=10,
                color="black",
                fill=True,
                fill_color=color_upstream,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=popup_width),
                z_index=1000,
            ).add_to(layer_upstream)

    for colormap, layer in [
        (colormap_correlation, layer_correlation),
        (colormap_kge, layer_kge),
        (colormap_nse, layer_nse),
    ]:
        colormap.add_to(m)
        layer.add_to(m)

    if layer_upstream is not None and colormap_upstream is not None:
        colormap_upstream.add_to(m)
        layer_upstream.add_to(m)

    # Inject all station image data as a single JS global.
    _inject_station_images_macro(m, station_images)

    # Waterbodies: render lakes and reservoirs as dot markers at polygon centroids.
    if waterbodies is not None and not waterbodies.empty:
        # Type constants match geb/hydrology/waterbodies.py
        _WATERBODY_STYLE: dict[int, dict[str, str]] = {
            1: {"color": "#4FC3F7", "label": "Lake"},
            2: {"color": "#FF8A65", "label": "Reservoir"},
            3: {"color": "#81C784", "label": "Lake (controlled)"},
        }
        wb_layers: dict[int, folium.FeatureGroup] = {
            wtype: folium.FeatureGroup(name=style["label"] + "s", show=True)
            for wtype, style in _WATERBODY_STYLE.items()
        }
        waterbodies_wgs84 = waterbodies.to_crs(epsg=4326)
        for _, wb_row in waterbodies_wgs84.iterrows():
            wtype = int(wb_row["waterbody_type"])
            style = _WATERBODY_STYLE.get(wtype)
            if style is None:
                continue
            centroid = wb_row.geometry.centroid
            area_km2: float | None = (
                float(wb_row["average_area"]) / 1e6
                if "average_area" in wb_row.index
                else None
            )
            volume_km3: float | None = (
                float(wb_row["volume_total"]) / 1e9
                if "volume_total" in wb_row.index
                else None
            )
            popup_lines = [
                f"<b>{style['label']}</b> (ID {wb_row.get('waterbody_id', '?')})<br>"
            ]
            if area_km2 is not None:
                popup_lines.append(f"Area: {area_km2:.1f} km²<br>")
            if volume_km3 is not None:
                popup_lines.append(f"Volume: {volume_km3:.3f} km³<br>")
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=5,
                color="black",
                weight=0.5,
                fill=True,
                fill_color=style["color"],
                fill_opacity=0.8,
                popup=folium.Popup("".join(popup_lines), max_width=200),
                tooltip=f"{style['label']} {wb_row.get('waterbody_id', '')}",
                z_index=500,
            ).add_to(wb_layers[wtype])
        for wb_layer in wb_layers.values():
            wb_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return m
