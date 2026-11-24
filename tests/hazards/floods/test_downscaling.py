"""Tests for the parallel replacement of hydromt-sfincs' flood map downscaling."""

import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine
from hydromt_sfincs.workflows import downscale_floodmap

from geb.hazards.floods.workflows.downscaling import downscale_flood_depth
from geb.workflows.io import read_zarr

LOGGER: logging.Logger = logging.getLogger(__name__)

CRS: int = 32631
N_ROWS: int = 60
N_COLS: int = 80
CELL_SIZE: float = 20.0
PIXELS_PER_CELL: int = 8


def create_inputs(tmp_path: Path) -> tuple[xr.DataArray, Path]:
    """Creates a water level grid with dry cells and an aligned subgrid elevation file.

    Args:
        tmp_path: Directory for the elevation file.

    Returns:
        The water level per cell (NaN in dry cells) and the path of the subgrid
        elevation GeoTIFF.
    """
    rng = np.random.default_rng(3)
    # south-up, as the hydromt-sfincs regular grid
    grid_transform = Affine(CELL_SIZE, 0, 500000.0, 0, CELL_SIZE, 5000000.0)
    x = grid_transform.c + (np.arange(N_COLS) + 0.5) * CELL_SIZE
    y = grid_transform.f + (np.arange(N_ROWS) + 0.5) * CELL_SIZE
    xx, yy = np.meshgrid(x - x[0], y - y[0])
    level = (10.0 - 0.002 * xx + 0.5 * np.sin(yy / 200.0)).astype(np.float32)
    # a dry band and scattered dry cells
    level[:, 50:60] = np.nan
    level[rng.random(level.shape) < 0.05] = np.nan
    water_level = xr.DataArray(
        level, coords={"y": y, "x": x}, dims=("y", "x")
    ).rio.write_crs(CRS)

    pixel = CELL_SIZE / PIXELS_PER_CELL
    height, width = N_ROWS * PIXELS_PER_CELL, N_COLS * PIXELS_PER_CELL
    px, py = np.meshgrid(np.arange(width) * pixel, np.arange(height) * pixel)
    elevation = (9.0 - 0.002 * px + 1.5 * np.sin(px / 37.0) * np.cos(py / 53.0)).astype(
        np.float32
    )
    elevation[:20, :30] = np.nan  # no elevation data
    path = tmp_path / "dep_subgrid.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=np.float32,
        crs=CRS,
        transform=grid_transform * Affine.scale(1 / PIXELS_PER_CELL),
        nodata=np.nan,
    ) as dst:
        dst.write(elevation, 1)
    return water_level, path


@pytest.mark.parametrize("block_size_pixels", [256, 512])
def test_downscale_flood_depth_own_cell_level(
    tmp_path: Path, block_size_pixels: int
) -> None:
    """The flood depth is the water level of each pixel's own cell minus the elevation.

    hydromt-sfincs' bilinear downscaling floods a subset of these pixels: it keeps
    pixels within half a cell of a dry cell dry.

    Args:
        tmp_path: Temporary directory.
        block_size_pixels: Block size, to test the results do not depend on it.
    """
    water_level, elevation_path = create_inputs(tmp_path)
    output_path = tmp_path / "flood_depth.zarr"
    downscale_flood_depth(
        water_level=water_level,
        subgrid_elevation_path=elevation_path,
        minimum_flood_depth=0.05,
        output_path=output_path,
        logger=LOGGER,
        block_size_pixels=block_size_pixels,
        n_workers=3,
    )
    with rasterio.open(elevation_path) as src:
        elevation = src.read(1)
    level_on_pixels = np.repeat(
        np.repeat(water_level.values, PIXELS_PER_CELL, axis=0), PIXELS_PER_CELL, axis=1
    )
    with np.errstate(invalid="ignore"):
        expected = level_on_pixels - elevation
        expected = np.where(expected > 0.05, expected, np.nan)
    flood_depth = read_zarr(output_path)
    assert flood_depth.rio.crs.to_epsg() == CRS
    assert flood_depth.shape == expected.shape
    assert np.isfinite(expected).sum() > 1000
    np.testing.assert_allclose(flood_depth.values, expected, atol=1e-6)

    reference_path = tmp_path / "hydromt.tif"
    with np.errstate(invalid="ignore"):
        downscale_floodmap(
            zsmax=water_level,
            dep=elevation_path,
            hmin=0.05,
            reproj_method="bilinear",
            floodmap_fn=reference_path,
            nrmax=200,
        )
    with rasterio.open(reference_path) as src:
        hydromt_wet = np.isfinite(src.read(1))
    own_cell_wet = np.isfinite(flood_depth.values)
    assert own_cell_wet.sum() > hydromt_wet.sum()
    # away from dry cells both methods flood about the same pixels
    assert (hydromt_wet & ~own_cell_wet).sum() < 0.01 * hydromt_wet.sum()


def test_downscale_flood_depth_dry_model(tmp_path: Path) -> None:
    """A model without wet cells gives an empty flood map.

    Args:
        tmp_path: Temporary directory.
    """
    water_level, elevation_path = create_inputs(tmp_path)
    water_level[:] = np.nan
    output_path = tmp_path / "flood_depth.zarr"
    downscale_flood_depth(
        water_level=water_level,
        subgrid_elevation_path=elevation_path,
        minimum_flood_depth=0.05,
        output_path=output_path,
        logger=LOGGER,
    )
    flood_depth = read_zarr(output_path)
    assert flood_depth.isnull().all()


def test_downscale_flood_depth_requires_aligned_grids(tmp_path: Path) -> None:
    """A subgrid raster that is not aligned with the grid is refused.

    Args:
        tmp_path: Temporary directory.
    """
    water_level, elevation_path = create_inputs(tmp_path)
    shifted = water_level.assign_coords(x=water_level.x + 3.0)
    with pytest.raises(ValueError, match="not aligned"):
        downscale_flood_depth(
            water_level=shifted,
            subgrid_elevation_path=elevation_path,
            minimum_flood_depth=0.05,
            output_path=tmp_path / "flood_depth.zarr",
            logger=LOGGER,
        )
