"""Tests for the groundwater build module."""

import logging
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon

from geb.build.methods import build_method
from geb.build.modules.groundwater import GroundWater


def test_setup_groundwater_creates_boundary_heads() -> None:
    """Test that setup_groundwater saves boundary_heads grid with 1-cell buffer surrounding model domain."""
    build_method.logger = logging.getLogger("test")

    x: np.ndarray = np.array([4.0, 5.0, 6.0])
    y: np.ndarray = np.array([53.0, 52.0, 51.0])
    grid_da: xr.DataArray = xr.DataArray(
        np.full((3, 3), 10.0, dtype=np.float32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        attrs={"_FillValue": -9999.0},
    ).rio.write_crs("EPSG:4326")

    mock_gw: MagicMock = MagicMock()
    mock_gw.logger = logging.getLogger("test")
    mock_gw.grid = {"landsurface/elevation_m": grid_da}
    mock_gw.bounds = (3.5, 50.5, 6.5, 53.5)
    mock_gw.set_grid.side_effect = lambda da, name, **kwargs: da

    saved_other_grids: dict[str, xr.DataArray] = {}
    mock_gw.set_other.side_effect = lambda da, name, **kwargs: (
        saved_other_grids.setdefault(name, da)
    )

    def full_like(
        da: xr.DataArray,
        fill_value: float,
        nodata: float | None = None,
        **kwargs: object,
    ) -> xr.DataArray:
        res = da.copy()
        res.values[...] = fill_value
        return res

    mock_gw.full_like.side_effect = full_like

    def fetch(name: str, **kwargs: object) -> MagicMock:
        reader: MagicMock = MagicMock()
        if name == "why_map":
            gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
                {
                    "HYGEO2": [10, 20],
                    "geometry": [
                        Polygon([(3, 50), (7, 50), (7, 54), (3, 54)]),
                        Polygon([(3, 50), (7, 50), (7, 54), (3, 54)]),
                    ],
                },
                crs="EPSG:4326",
            )
            reader.read.return_value = gdf
        else:
            gx: np.ndarray = np.linspace(-10.0, 20.0, 31)
            gy: np.ndarray = np.linspace(70.0, 30.0, 41)
            da: xr.DataArray = xr.DataArray(
                np.full((41, 31), 50.0, dtype=np.float32),
                coords={"y": gy, "x": gx},
                dims=("y", "x"),
                attrs={"_FillValue": -9999.0},
            ).rio.write_crs("EPSG:4326")
            reader.read.return_value = da
        return reader

    mock_gw.data_catalog.fetch.side_effect = fetch

    GroundWater.setup_groundwater(mock_gw)

    assert "groundwater/boundary_heads" in saved_other_grids
    boundary_heads: xr.DataArray = saved_other_grids["groundwater/boundary_heads"]
    # Original grid is 3x3, extended grid is 1 cell larger on all sides -> 5x5
    assert boundary_heads.shape == (1, 5, 5)
    assert not np.isnan(boundary_heads.values).any()

    assert "groundwater/boundary_mask" in saved_other_grids
    boundary_mask: xr.DataArray = saved_other_grids["groundwater/boundary_mask"]
    assert boundary_mask.shape == (5, 5)
    assert boundary_mask.values.dtype == bool
    # 3x3 model grid inside 5x5: boundary cells are strictly outside the 3x3 domain
    assert boundary_mask.values[1:-1, 1:-1].sum() == 0
    # Outer halo cells touching the 3x3 domain orthogonally: 3 on each of the 4 edges = 12 cells
    assert boundary_mask.values.sum() == 12

    assert "groundwater/boundary_hydraulic_conductivity" in saved_other_grids
    boundary_k: xr.DataArray = saved_other_grids[
        "groundwater/boundary_hydraulic_conductivity"
    ]
    assert boundary_k.shape == (1, 5, 5)

    assert "groundwater/boundary_layer_boundary_elevation" in saved_other_grids
    boundary_layer_elev: xr.DataArray = saved_other_grids[
        "groundwater/boundary_layer_boundary_elevation"
    ]
    assert boundary_layer_elev.shape == (2, 5, 5)
    assert not np.isnan(boundary_layer_elev.values[:, boundary_mask.values]).any()
