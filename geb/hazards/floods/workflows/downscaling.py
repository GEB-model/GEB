"""Fast, parallel replacement for hydromt-sfincs' file-based flood map downscaling.

hydromt-sfincs' `downscale_floodmap` (with the subgrid elevation as a file) processes
the subgrid raster block by block in one thread: it reads every block, also where
the model is dry, resamples the water level of the whole grid onto each block, and
opens, writes and closes the output file for each block. For a fine subgrid over a
large basin (e.g. 1 m pixels over the Meuse) this takes more than an hour.

`downscale_flood_depth` computes the flood depth as the water level of each pixel's
own cell minus the subgrid elevation, kept where deeper than the minimum depth. This
is the flat water level per cell that SFINCS' subgrid tables assume. (hydromt-sfincs
interpolates the water level bilinearly between cell centres, with dry cells as
no data, so that every pixel within half a cell of a dry cell stays dry: flood
edges shrink by up to half a cell and rivers one cell wide lose most of their water.
In July 2021 tests on the Meuse, the own-cell level gave a CSI 0.3-1.7 percentage
points higher.) Further:

- only blocks with wet cells are read and written; the other chunks of the output
  are not stored and read as no data;
- blocks are processed in parallel threads (rasterio, numpy and the zarr compressor
  release the GIL), each block being one chunk of the output zarr, so that the flood
  map is written once, directly in GEB's zarr format.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dask.array
import numpy as np
import rasterio
import xarray as xr
import zarr
from affine import Affine
from rasterio.windows import Window
from zarr.codecs.zstd import ZstdCodec

from geb.workflows.io import to_wkt

# blocks (and output chunks) are multiples of this size, in pixels
_BLOCK_SIZE_MULTIPLE: int = 256


def downscale_flood_depth(
    water_level: xr.DataArray,
    subgrid_elevation_path: Path,
    minimum_flood_depth: float,
    output_path: Path,
    logger: logging.Logger,
    block_size_pixels: int = 4096,
    compression_level: int = 18,
    n_workers: int | None = None,
) -> None:
    """Downscales the water level of a regular SFINCS grid to a flood depth map.

    Each pixel gets the water level of its own cell. The subgrid raster must be
    aligned with the grid (an integer number of pixels per cell).

    Args:
        water_level: Water level per grid cell (m), NaN in dry or inactive cells,
            with the grid's coordinates and CRS.
        subgrid_elevation_path: Path to the subgrid elevation GeoTIFF
            (dep_subgrid.tif).
        minimum_flood_depth: Pixels with a smaller depth (m) are not flooded (NaN).
        output_path: Path of the flood depth zarr to write (GEB's zarr format, as
            written by `geb.workflows.io.write_zarr`; the array is named after the
            file).
        logger: Logger for progress messages.
        block_size_pixels: Block size (and chunk size of the output) in pixels along
            each axis, rounded to a multiple of 256; bounds the memory per thread.
        compression_level: Zstandard compression level of the output.
        n_workers: Number of threads; default: the CPUs available to this process.

    Raises:
        ValueError: If the subgrid raster is not aligned with the grid.
    """
    if n_workers is None:
        n_workers = len(os.sched_getaffinity(0))
    start_time: float = time.perf_counter()
    block_size_pixels = max(
        _BLOCK_SIZE_MULTIPLE,
        block_size_pixels // _BLOCK_SIZE_MULTIPLE * _BLOCK_SIZE_MULTIPLE,
    )

    level: np.ndarray = np.asarray(water_level.values, dtype=np.float32)
    n_rows, n_cols = level.shape
    grid_transform: Affine = water_level.rio.transform(recalc=True)

    with rasterio.open(subgrid_elevation_path) as src:
        height, width = src.height, src.width
        subgrid_transform: Affine = src.transform
        subgrid_crs = src.crs
        elevation_nodata = src.nodata
    pixels_per_cell: int = width // n_cols
    expected_transform: Affine = grid_transform * Affine.scale(1 / pixels_per_cell)
    if (
        width != n_cols * pixels_per_cell
        or height != n_rows * pixels_per_cell
        or not subgrid_transform.almost_equals(expected_transform, precision=1e-9)
    ):
        raise ValueError(
            "The subgrid elevation raster is not aligned with the SFINCS grid"
        )

    # blocks of pixels with wet cells
    wet: np.ndarray = np.isfinite(level)
    blocks: list[tuple[int, int, int, int]] = []
    for row_start in range(0, height, block_size_pixels):
        row_end: int = min(row_start + block_size_pixels, height)
        cell_row_start: int = row_start // pixels_per_cell
        cell_row_end: int = -(-row_end // pixels_per_cell)
        for col_start in range(0, width, block_size_pixels):
            col_end: int = min(col_start + block_size_pixels, width)
            cell_col_start: int = col_start // pixels_per_cell
            cell_col_end: int = -(-col_end // pixels_per_cell)
            if wet[cell_row_start:cell_row_end, cell_col_start:cell_col_end].any():
                blocks.append((row_start, row_end, col_start, col_end))
    total_blocks: int = -(-height // block_size_pixels) * -(-width // block_size_pixels)
    logger.info(
        f"Downscaling flood depth: {len(blocks)} of {total_blocks} blocks of "
        f"{block_size_pixels} x {block_size_pixels} pixels have wet cells, "
        f"{n_workers} threads"
    )

    local = threading.local()

    def elevation_source() -> rasterio.io.DatasetReader:
        """Subgrid elevation dataset, one per thread (datasets are not thread-safe).

        Returns:
            The opened dataset.
        """
        dataset = getattr(local, "dataset", None)
        if dataset is None:
            dataset = rasterio.open(subgrid_elevation_path)
            local.dataset = dataset
        return dataset

    def process_block(block: tuple[int, int, int, int]) -> np.ndarray | None:
        """Flood depth of one block.

        Args:
            block: Pixel rows and columns (start, end, start, end) of the block.

        Returns:
            Flood depth (m, NaN where not flooded), or None if nothing is flooded.
        """
        row_start, row_end, col_start, col_end = block
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        elevation: np.ndarray = (
            elevation_source().read(1, window=window).astype(np.float32)
        )
        if elevation_nodata is not None and not np.isnan(elevation_nodata):
            elevation[elevation == elevation_nodata] = np.nan
        if np.isnan(elevation).all():
            return None

        # water level of each pixel's own cell
        level_on_pixels: np.ndarray = level[
            np.arange(row_start, row_end)[:, np.newaxis] // pixels_per_cell,
            np.arange(col_start, col_end)[np.newaxis, :] // pixels_per_cell,
        ]
        with np.errstate(invalid="ignore"):
            depth: np.ndarray = level_on_pixels - elevation
            depth = np.where(depth > minimum_flood_depth, depth, np.float32(np.nan))
        if np.isnan(depth).all():
            return None
        return depth.astype(np.float32)

    # empty zarr array (chunks are only stored when written)
    name: str = output_path.stem
    x: np.ndarray = subgrid_transform.c + (np.arange(width) + 0.5) * subgrid_transform.a
    y: np.ndarray = (
        subgrid_transform.f + (np.arange(height) + 0.5) * subgrid_transform.e
    )
    template = xr.DataArray(
        dask.array.full(
            (height, width),
            np.nan,
            dtype=np.float32,
            chunks=(block_size_pixels, block_size_pixels),
        ),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name=name,
        attrs={"_FillValue": np.nan, "_CRS": {"wkt": to_wkt(subgrid_crs)}},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = ZstdCodec(level=compression_level)
    template.to_zarr(
        store=output_path,
        mode="w",
        encoding={
            name: {
                "chunks": (block_size_pixels, block_size_pixels),
                "compressors": (compressor,),
                "fill_value": np.nan,
            },
            "x": {"compressors": (compressor,), "chunks": (width,)},
            "y": {"compressors": (compressor,), "chunks": (height,)},
        },
        zarr_format=3,
        consolidated=False,
        compute=False,
    )
    target = zarr.open_array(store=output_path, zarr_format=3, path=name, mode="r+")

    def process_and_write(block: tuple[int, int, int, int]) -> bool:
        """Computes and writes the flood depth of one block (one output chunk).

        Args:
            block: Pixel rows and columns (start, end, start, end) of the block.

        Returns:
            Whether the block has flooding.
        """
        depth: np.ndarray | None = process_block(block)
        if depth is None:
            return False
        row_start, row_end, col_start, col_end = block
        target[row_start:row_end, col_start:col_end] = depth
        return True

    n_flooded: int = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        n_done: int = 0
        next_report: int = 1
        for has_flooding in executor.map(process_and_write, blocks):
            n_flooded += int(has_flooding)
            n_done += 1
            if n_done >= next_report or n_done == len(blocks):
                logger.info(
                    f"Downscaled blocks: {n_done}/{len(blocks)} "
                    f"({time.perf_counter() - start_time:.0f} s)"
                )
                next_report = n_done + max(1, len(blocks) // 20)
    logger.info(
        f"Flood depth written: {n_flooded} blocks with flooding "
        f"({time.perf_counter() - start_time:.0f} s in total)"
    )
