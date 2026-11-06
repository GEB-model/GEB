"""Stream European bedrock age and lithology from the official BGR map service."""

from typing import Any

import folium
from folium.raster_layers import WmsTileLayer

GEOLOGY_WMS_URL: str = "https://services.bgr.de/wms/inspire_ge/igme5000eu/"
GEOLOGY_SOURCE_URL: str = (
    "https://services.bgr.de/arcgis/rest/services/inspire_ge/igme5000eu/MapServer"
)


def add_geology_layers(
    discharge_map: folium.Map,
) -> list[tuple[WmsTileLayer, dict[str, Any]]]:
    """Add initially hidden European bedrock age and rock-type overlays.

    Notes:
        IGME5000 depicts pre-Quaternary bedrock at 1:5 million, including where
        younger sediments cover it. It does not describe soil or permeability.
        Tiles load only when enabled; dashboard creation makes no web requests.
        WMS 1.1.1 avoids latitude/longitude axis ambiguity in EPSG:4326.

    Args:
        discharge_map: Dashboard receiving the overlays beneath station markers.

    Returns:
        WMS layers and menu metadata, including selected official legend colours.

    Raises:
        None.
    """
    definitions: list[dict[str, Any]] = [
        {
            "id": "geology_age",
            "name": "Bedrock age (Europe, BGR)",
            "wms_layer": "1",
            "rest_layer": 3,
            "note": "D = Devonian age, not a rock type or karst class. Colours differ from the Wikimedia map.",
            "classes": [
                {"label": "Lower Devonian", "color": "#e6b359"},
                {"label": "Middle Devonian", "color": "#f2cc73"},
                {"label": "Upper Devonian", "color": "#f2e6a6"},
                {"label": "Frasnian (Upper Devonian)", "color": "#f2edad"},
            ],
        },
        {
            "id": "geology_lithology",
            "name": "Bedrock rock type (Europe, BGR)",
            "wms_layer": "2",
            "rest_layer": 2,
            "note": "Rock type helps distinguish sandstone, carbonate and evaporite units of the same age. Only the first listed rock type determines the map colour.",
            "classes": [
                {"label": "Sandstone", "color": "#f2d973"},
                {"label": "Clay", "color": "#faf2bf"},
                {"label": "Limestone", "color": "#66b3f2"},
                {"label": "Dolomite", "color": "#73a6f2"},
                {"label": "Gypsum or anhydrite", "color": "#b3e6f2"},
                {"label": "Compound material", "color": "#ffebd4"},
            ],
        },
    ]
    layers: list[tuple[WmsTileLayer, dict[str, Any]]] = []
    definition: dict[str, Any]
    for index, definition in enumerate(definitions):
        pane_name: str = f"{definition['id']}Pane"
        folium.map.CustomPane(
            pane_name, z_index=270 + index, pointer_events=False
        ).add_to(discharge_map)
        layer: WmsTileLayer = WmsTileLayer(
            url=GEOLOGY_WMS_URL,
            layers=definition["wms_layer"],
            styles="default",
            fmt="image/png",
            transparent=True,
            version="1.1.1",
            name=definition["name"],
            attr=f'<a href="{GEOLOGY_SOURCE_URL}">IGME5000 © BGR Hannover</a>',
            overlay=True,
            show=False,
            opacity=0.65,
            pane=pane_name,
            min_zoom=3,
            # The source is a continental overview; avoid implying local detail.
            max_native_zoom=9,
            max_zoom=19,
            bounds=[[24, -35], [83, 70]],
            cross_origin=True,
        )
        layer.add_to(discharge_map)
        layers.append(
            (
                layer,
                {
                    **definition,
                    "layer": layer.get_name(),
                    "active": False,
                    "opacity": 0.65,
                    "legend_url": f"{GEOLOGY_SOURCE_URL}/legend",
                    "source_url": GEOLOGY_SOURCE_URL,
                },
            )
        )
    return layers
