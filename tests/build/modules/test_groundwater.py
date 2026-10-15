"""Tests for the groundwater build module."""

import logging
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon, box

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


@pytest.mark.parametrize("lazy_mask", [False, True])
@pytest.mark.parametrize("reverse_x", [False, True])
@pytest.mark.parametrize("reverse_y", [False, True])
@pytest.mark.parametrize("scenario", ["overlap", "outside", "empty", "inactive"])
def test_setup_karst_spatial_coverage(
    reverse_x: bool, reverse_y: bool, scenario: str, lazy_mask: bool
) -> None:
    """Save weighted polygon coverage in the correct active grid cells.

    Args:
        lazy_mask: Whether to use a Dask-backed mask, as in a loaded model.
        reverse_x: Whether to reverse the grid's horizontal coordinate order.
        reverse_y: Whether to reverse the grid's vertical coordinate order.
        scenario: Overlapping polygons, polygons outside the grid, an empty
            catalog, or a grid with no active cells.
    """
    build_method.logger = logging.getLogger("test")
    # Equal-area coordinates make rectangle overlap fractions exact and independent
    # of the projection implementation used by the coverage calculation.
    mask: xr.DataArray = xr.DataArray(
        [[False, False, False, True], [False, False, False, False]],
        coords={"y": [15.0, 5.0], "x": [5.0, 15.0, 25.0, 35.0]},
        dims=("y", "x"),
    ).rio.write_crs("EPSG:6933")
    # Northern cells: full continuous, full discontinuous, half mixed, inactive.
    # Southern cells: full continuous, half discontinuous, half continuous plus
    # half mixed, and no overlap. Fractions are independent of input row order.
    expected: xr.DataArray = xr.DataArray(
        [[0.825, 0.4, 0.325, np.nan], [0.825, 0.2, 0.5 * 0.825 + 0.5 * 0.65, 0.0]],
        coords=mask.coords,
        dims=mask.dims,
    )
    # Input order and labels deliberately differ from spatial order and rock codes.
    # Two overlapping class-1 polygons must be dissolved before area is counted.
    polygons: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {"karst_fraction": [0.4, 0.825, 0.825, 0.65, 0.4, 0.825, 0.825, 0.65, 0.4]},
        geometry=[
            box(10, 0, 15, 10),
            box(0, 0, 10, 10),
            box(0, 10, 7, 20),
            box(20, 10, 25, 20),
            box(10, 10, 20, 20),
            box(3, 10, 10, 20),
            box(20, 0, 25, 10),
            box(25, 0, 30, 10),
            box(30, 10, 40, 20),
        ],
        index=[90, 12, 71, 3, 50, 21, 8, 31, 19],
        crs=mask.rio.crs,
    )
    polygons.attrs["source"] = "Synthetic karst map"
    if scenario == "empty":
        polygons = polygons.iloc[:0].copy()
        expected.values[~mask.values] = 0.0
    elif scenario == "inactive":
        mask.values[:] = True
        expected.values[:] = np.nan
    elif scenario == "outside":
        polygons.geometry = polygons.translate(xoff=1000.0)
        expected.values[~mask.values] = 0.0
    if reverse_x:
        mask = mask.isel(x=slice(None, None, -1))
        expected = expected.isel(x=slice(None, None, -1))
    if reverse_y:
        mask = mask.isel(y=slice(None, None, -1))
        expected = expected.isel(y=slice(None, None, -1))
    if lazy_mask:
        mask = mask.chunk({"y": 1, "x": 2})
    model: MagicMock = MagicMock()
    model.logger = logging.getLogger("test")
    model.grid = {"mask": mask}
    model.data_catalog.fetch.return_value.read.return_value = polygons

    GroundWater.setup_karst(model)

    model.data_catalog.fetch.assert_called_once_with("wokam")
    model.set_grid.assert_called_once()
    assert model.set_grid.call_args.kwargs["name"] == "groundwater/karst_fraction"
    saved: xr.DataArray = model.set_grid.call_args.args[0]
    xr.testing.assert_allclose(saved, expected, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(np.isnan(saved.values), mask.values)
    assert saved.dtype == np.float32
    assert saved.rio.crs == mask.rio.crs
    assert saved.attrs["units"] == "1"
    assert saved.attrs["source"] == "Synthetic karst map"
