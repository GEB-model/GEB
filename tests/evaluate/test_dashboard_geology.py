"""Check optional streamed bedrock layers without depending on a live service."""

from typing import Any

import folium
import pytest
from folium.raster_layers import WmsTileLayer

from geb.evaluate.workflows.dashboard_geology import (
    GEOLOGY_WMS_URL,
    add_geology_layers,
)


@pytest.mark.parametrize("position", [0, 1])
def test_geology_layer_options(position: int) -> None:
    """Keep coarse, attributed WMS layers hidden until explicitly enabled.

    Args:
        position: Age or lithology layer index.

    Returns:
        None.

    Raises:
        AssertionError: If layer identity, extent or visibility is incorrect.
    """  # noqa: DOC202, DOC502 - Document assertion failures and the None return.
    discharge_map: folium.Map = folium.Map()
    layers: list[tuple[WmsTileLayer, dict[str, Any]]] = add_geology_layers(
        discharge_map
    )
    layer, metadata = layers[position]
    assert layer.url == GEOLOGY_WMS_URL
    # WMS and REST identifiers differ; using REST IDs here can display faults.
    assert layer.options["layers"] == str(position + 1)
    assert layer.options["version"] == "1.1.1"
    assert layer.options["transparent"] is True
    assert layer.options["crossOrigin"] is True
    assert layer.options["maxNativeZoom"] == 9
    assert layer.options["bounds"] == [[24, -35], [83, 70]]
    assert layer.show is False
    assert metadata["active"] is False
    assert metadata["layer"] == layer.get_name()
    assert metadata["classes"]
    assert "IGME5000" in str(layer.options["attribution"])
    assert f"{layer.get_name()}.addTo(" not in discharge_map.get_root().render()


def test_geology_layers_are_distinct() -> None:
    """Keep age and rock-type layers independently adjustable and labelled.

    Returns:
        None.

    Raises:
        AssertionError: If layers share identifiers or confuse age with rock type.
    """  # noqa: DOC202, DOC502 - Document assertion failures and the None return.
    discharge_map: folium.Map = folium.Map()
    layers: list[tuple[WmsTileLayer, dict[str, Any]]] = add_geology_layers(
        discharge_map
    )
    assert {metadata["id"] for _, metadata in layers} == {
        "geology_age",
        "geology_lithology",
    }
    assert layers[0][0].options["pane"] != layers[1][0].options["pane"]
    assert "not a rock type" in layers[0][1]["note"]
    assert "first listed rock type" in layers[1][1]["note"]
