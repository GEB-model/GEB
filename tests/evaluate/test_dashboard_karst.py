"""Tests for embedded model karst coverage in discharge dashboards."""

# Project docstrings include None returns and assertion failures in tests.
# ruff: noqa: DOC202, DOC502

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import shapely.geometry as sg
import xarray as xr
from PIL import Image

from geb.evaluate.workflows.dashboard import write_discharge_dashboard
from geb.evaluate.workflows.dashboard_elevation import add_karst_layer
from geb.workflows.io import write_zarr


@pytest.fixture
def karst_path(tmp_path: Path) -> Path:
    """Save a small model karst raster with zero, partial and full coverage.

    Args:
        tmp_path: Temporary data directory.

    Returns:
        Model-style karst Zarr path; source values are fractions (0–1).

    Raises:
        OSError: If test raster data cannot be saved.
    """
    raster: xr.DataArray = xr.DataArray(
        np.array(
            [[np.nan, 0, 0.25], [0, 0.825, 1], [0.25, 0.825, 1]], dtype=np.float32
        ),
        dims=("y", "x"),
        coords={"y": [51, 50, 49], "x": [7, 8, 9]},
        name="karst_fraction",
        attrs={"_FillValue": np.nan},
    )
    path: Path = tmp_path / "karst.zarr"
    write_zarr(raster.chunk({"x": 3, "y": 3}), path, crs=4326, progress=False)
    return path


def test_karst_layer_preserves_coverage(karst_path: Path) -> None:
    """Preserve coverage colours and keep zero and nodata pixels transparent.

    Args:
        karst_path: Small model karst raster.

    Returns:
        None.

    Raises:
        AssertionError: If image values, legend units or default visibility change.
    """
    dashboard: folium.Map = folium.Map(tiles=None)
    result: Any = add_karst_layer(dashboard, karst_path)
    assert result is not None
    layer, metadata = result
    assert layer.show is False
    assert layer.options["pane"] == "karstPane"
    assert layer.options["opacity"] == 0.7
    assert metadata["minimum_percent"] == 0
    assert metadata["maximum_percent"] == 100
    assert (metadata["width"], metadata["height"]) == (3, 3)
    image: Image.Image = Image.open(
        BytesIO(base64.b64decode(layer.url.split(",", 1)[1]))
    )
    pixels: npt.NDArray[np.uint8] = np.asarray(image)
    assert np.count_nonzero(pixels[:, :, 3] == 0) == 3
    palette: matplotlib.colors.Colormap = matplotlib.colormaps["YlOrBr"]
    expected: set[tuple[int, ...]] = {
        tuple(palette(value, bytes=True)) for value in (0.25, 0.825, 1.0)
    }
    assert {tuple(pixel) for pixel in pixels.reshape(-1, 4) if pixel[3] > 0} == expected
    assert f"{layer.get_name()}.addTo(" not in dashboard.get_root().render()


@pytest.mark.parametrize("value", [0.0, 0.825, 1.0, np.nan, -0.1, 1.1])
def test_karst_coverage_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    """Handle uniform zero/full coverage, missing coverage and invalid fractions.

    Args:
        tmp_path: Temporary data directory.
        monkeypatch: Fixture replacing the model raster reader.
        value: Coverage fraction; NaN denotes missing data.

    Returns:
        None.

    Raises:
        AssertionError: If boundaries or invalid fractions are handled incorrectly.
    """
    path: Path = tmp_path / "karst.zarr"
    path.touch()
    raster: xr.DataArray = xr.DataArray(
        np.full((3, 3), value, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [51, 50, 49], "x": [7, 8, 9]},
    ).rio.write_crs(4326)
    monkeypatch.setattr(
        "geb.evaluate.workflows.dashboard_elevation.read_zarr", lambda _: raster
    )
    dashboard: folium.Map = folium.Map(tiles=None)
    if value < 0 or value > 1:
        with pytest.raises(ValueError, match="within"):
            add_karst_layer(dashboard, path)
        return
    result: Any = add_karst_layer(dashboard, path, max_size=2)
    if np.isnan(value):
        assert result is None
    else:
        assert result is not None
        assert result[1]["minimum_percent"] == 0
        assert result[1]["maximum_percent"] == 100
        assert result[1]["width"] == result[1]["height"] == 2
        if value == 0:
            image: Image.Image = Image.open(
                BytesIO(base64.b64decode(result[0].url.split(",", 1)[1]))
            )
            assert not np.asarray(image)[:, :, 3].any()


def test_dashboard_includes_karst_controls(karst_path: Path, tmp_path: Path) -> None:
    """Wire coverage metadata, opacity controls and PNG export into the dashboard.

    Args:
        karst_path: Model karst raster fixture.
        tmp_path: Temporary dashboard output directory.

    Returns:
        None.

    Raises:
        AssertionError: If the generated dashboard omits karst controls or units.
    """
    region: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=[sg.box(7, 49, 9, 51)], crs=4326
    )
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"uparea_m2": [5e6]},
        geometry=[sg.LineString([(7.5, 49.5), (8.5, 50.5)])],
        index=pd.Index(["river"]),
        crs=4326,
    )
    scores: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "station_name": ["Test"],
            "upstream_area_GEB": [1e6],
            "discharge_observations_to_GEB_upstream_area_ratio": [1.0],
            "KGE": [0.8],
            "timezone_utc_offset": [1.0],
        },
        geometry=[sg.Point(8, 50)],
        index=pd.Index(["test"]),
        crs=4326,
    )
    output: Path = tmp_path / "karst_dashboard.html"
    write_discharge_dashboard(
        scores, output, region, rivers, [], {}, karst_path=karst_path
    )
    html: str = output.read_text()
    assert '"id": "karst"' in html
    assert '"minimum_percent": 0.0' in html
    assert '"maximum_percent": 100.0' in html
    assert "Karst coverage" in html
    assert "geb-karst-subpanel" in html
    assert "data-karst-opacity" in html
    assert "Karst coverage: 0–100%" in html
    assert "zero coverage is transparent" in html
    assert "data:image/png;base64," in html
