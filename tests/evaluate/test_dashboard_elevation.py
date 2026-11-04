"""Tests for embedded FathomDEM dashboard previews."""

# Project docstrings document None returns and assertions in tests.
# ruff: noqa: DOC202, DOC502

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import shapely.geometry as sg
import xarray as xr
from PIL import Image

from geb.evaluate.workflows.dashboard import write_discharge_dashboard
from geb.evaluate.workflows.dashboard_elevation import add_fathom_dem_layer
from geb.workflows.io import write_zarr


@pytest.fixture
def elevation_path(tmp_path: Path) -> Path:
    """Write a small georeferenced DEM with nodata and below-sea-level elevations.

    Args:
        tmp_path: Temporary data directory.

    Returns:
        Model-style Zarr DEM path; elevations are in meters.

    Raises:
        OSError: If test data cannot be written.
    """
    values: npt.NDArray[np.float32] = np.arange(64, dtype=np.float32).reshape(8, 8) - 30
    values[:, :2] = np.nan
    elevation: xr.DataArray = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.linspace(51, 49, 8), "x": np.linspace(7, 9, 8)},
        name="elevation",
        attrs={"_FillValue": np.nan},
    )
    path: Path = tmp_path / "dem.zarr"
    write_zarr(elevation.chunk({"x": 4, "y": 4}), path, crs=4326, progress=False)
    return path


def test_embedded_dem_preview(elevation_path: Path) -> None:
    """Keep the DEM hidden initially, bounded in size and transparent at nodata.

    Args:
        elevation_path: Small model DEM fixture.

    Returns:
        None.

    Raises:
        AssertionError: If georeferencing, size or nodata transparency is lost.
    """
    dashboard: folium.Map = folium.Map(tiles=None)
    result: Any = add_fathom_dem_layer(dashboard, elevation_path, max_size=4)
    assert result is not None
    layer, metadata = result
    assert layer.show is False
    assert layer.options["pane"] == "elevationPane"
    assert layer.options["opacity"] == 0.7
    assert metadata["minimum_m"] > -9999
    assert metadata["minimum_m"] < 0
    assert metadata["width"] <= 4 and metadata["height"] <= 4
    assert layer.bounds[0] == pytest.approx([48.857142857, 6.857142857])
    assert layer.bounds[1] == pytest.approx([51.142857143, 9.142857143])
    image: Image.Image = Image.open(
        BytesIO(base64.b64decode(layer.url.split(",", 1)[1]))
    )
    assert image.size == (4, 4)
    assert (np.asarray(image)[:, :, 3] == 0).any()
    assert (np.asarray(image)[:, :, 3] == 255).any()
    assert f"{layer.get_name()}.addTo(" not in dashboard.get_root().render()


@pytest.mark.parametrize("value", [np.nan, 12.5])
def test_empty_and_flat_dem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    """Skip empty rasters and safely normalize constant elevations.

    Args:
        tmp_path: Temporary directory.
        monkeypatch: Fixture replacing DEM input.
        value: Constant elevation (meters), or nodata.

    Returns:
        None.

    Raises:
        AssertionError: If an empty or flat DEM produces invalid metadata.
    """
    path: Path = tmp_path / "dem.zarr"
    path.touch()
    elevation: xr.DataArray = xr.DataArray(
        np.full((3, 3), value, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [51, 50, 49], "x": [7, 8, 9]},
    ).rio.write_crs(4326)
    monkeypatch.setattr(
        "geb.evaluate.workflows.dashboard_elevation.read_zarr", lambda _: elevation
    )
    result: Any = add_fathom_dem_layer(folium.Map(tiles=None), path)
    if np.isnan(value):
        assert result is None
    else:
        assert result[1]["minimum_m"] == pytest.approx(value)
        assert result[1]["maximum_m"] == pytest.approx(value)


def test_dem_validation(elevation_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject invalid preview sizes, missing datasets and absent georeferencing.

    Args:
        elevation_path: Existing DEM path.
        monkeypatch: Fixture replacing the DEM reader.

    Returns:
        None.

    Raises:
        AssertionError: If invalid inputs do not raise an explicit exception.
    """
    dashboard: folium.Map = folium.Map(tiles=None)
    with pytest.raises(ValueError, match="at least two"):
        add_fathom_dem_layer(dashboard, elevation_path, max_size=1)
    with pytest.raises(FileNotFoundError):
        add_fathom_dem_layer(dashboard, elevation_path / "missing")
    elevation: xr.DataArray = xr.DataArray(np.ones((2, 2)), dims=("y", "x"))
    monkeypatch.setattr(
        "geb.evaluate.workflows.dashboard_elevation.read_zarr", lambda _: elevation
    )
    with pytest.raises(ValueError, match="coordinate reference"):
        add_fathom_dem_layer(dashboard, elevation_path)


def test_dashboard_includes_dem_controls(elevation_path: Path, tmp_path: Path) -> None:
    """Render the DEM menu, units, opacity control and image export support.

    Args:
        elevation_path: Existing model DEM.
        tmp_path: Temporary dashboard output directory.

    Returns:
        None.

    Raises:
        AssertionError: If the generated dashboard omits DEM metadata or controls.
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
    output: Path = tmp_path / "dashboard.html"
    write_discharge_dashboard(
        scores,
        output,
        region,
        rivers,
        [],
        {},
        elevation_path=elevation_path,
    )
    html: str = output.read_text()
    assert "Elevation (FathomDEM)" in html
    assert '"id": "dem"' in html
    assert "data:image/png;base64," in html
    assert "data-dem-opacity" in html
    assert "geb-dem-subpanel" in html
    assert "Elevation (m)" in html
    assert ".leaflet-image-layer" in html
