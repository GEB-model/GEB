"""Set initial SFINCS water levels from the DEM elevations in burned river cells."""

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.windows
import xarray as xr
from hydromt_sfincs.workflows import merge_multi_dataarrays


def river_initial_water_level(
    root: Path,
    DEMs: list[dict[str, Any]],
    grid_mask: xr.DataArray,
) -> xr.DataArray:
    """Create an initial water-level map from the unburned DEM in river cells.

    The burned subgrid DEM is compared with the unburned subgrid DEM. River burning only
    lowers the DEM, so every lowered subgrid pixel is a river pixel. For each SFINCS
    cell containing river pixels, the minimum unburned subgrid DEM elevation at those pixels
    becomes the initial water level.

    Args:
        root: SFINCS model directory containing ``subgrid/dep_subgrid.tif``.
        DEMs: DEMs used to build the SFINCS subgrid, as configured for the hydrodynamic model.
        grid_mask: SFINCS active-cell mask, used as the output grid template.

    Returns:
        Initial water levels (m), with NaN in cells without burned river pixels.

    Raises:
        ValueError: If the subgrid raster dimensions do not match the SFINCS grid.
    """
    nmax, mmax = grid_mask.shape
    initial_level: xr.DataArray = xr.full_like(grid_mask, np.nan, dtype=np.float32)
    # Same merge arguments as SFINCSRootModel.build passes to subgrid.create.
    elevation_list: list[dict[str, Any]] = [
        {
            "da": DEM["elevation"]["elevation"],
            "reproj_method": "bilinear",
            **{
                key: DEM[key]
                for key in ("zmin", "zmax", "offset", "merge_method")
                if key in DEM
            },
        }
        for DEM in DEMs
    ]

    with rasterio.open(root / "subgrid" / "dep_subgrid.tif") as burned_dem_file:
        if burned_dem_file.transform.e <= 0:
            raise ValueError("Expected the subgrid DEM to have south-up rows.")
        if (
            burned_dem_file.width % mmax != 0
            or burned_dem_file.height % nmax != 0
            or burned_dem_file.width // mmax != burned_dem_file.height // nmax
        ):
            raise ValueError("Subgrid DEM dimensions do not match the SFINCS grid.")

        pixels_per_cell: int = burned_dem_file.width // mmax
        # Pixel size in the units of the grid CRS (degrees for geographic grids).
        pixel_width: float = burned_dem_file.transform.a
        pixel_height: float = burned_dem_file.transform.e
        for row_start in range(0, nmax, 1000):
            row_end: int = min(row_start + 1000, nmax)
            active_cells: np.ndarray = grid_mask.values[row_start:row_end] > 0
            if not active_cells.any():
                continue

            window: rasterio.windows.Window = rasterio.windows.Window.from_slices(
                rows=(pixels_per_cell * row_start, pixels_per_cell * row_end),
                cols=(0, burned_dem_file.width),
            )
            burned_dem: np.ndarray = burned_dem_file.read(1, window=window).astype(
                np.float32
            )
            ys: np.ndarray = (
                burned_dem_file.transform.f
                + (pixels_per_cell * row_start + np.arange(burned_dem.shape[0]) + 0.5)
                * pixel_height
            )
            xs: np.ndarray = (
                burned_dem_file.transform.c
                + (np.arange(burned_dem.shape[1]) + 0.5) * pixel_width
            )

            window_grid: xr.DataArray = xr.DataArray(
                np.zeros(burned_dem.shape, dtype=np.float32),
                coords={"y": ys, "x": xs},
                dims=("y", "x"),
            )
            window_grid.raster.set_crs(grid_mask.raster.crs)
            active_pixels: np.ndarray = np.repeat(
                np.repeat(active_cells, pixels_per_cell, axis=0),
                pixels_per_cell,
                axis=1,
            )
            with np.errstate(invalid="ignore"):
                merged_dem: xr.DataArray = merge_multi_dataarrays(
                    da_list=elevation_list,
                    da_like=window_grid,
                    interp_method="linear",
                )
                if np.isnan(merged_dem.values[active_pixels]).any():
                    merged_dem = merged_dem.raster.interpolate_na(
                        method="rio_idw", extrapolate=True
                    )
                original_dem: np.ndarray = merged_dem.values.astype(np.float32)
            river_pixels: np.ndarray = burned_dem < original_dem
            cell_shape: tuple[int, int, int, int] = (
                row_end - row_start,
                pixels_per_cell,
                mmax,
                pixels_per_cell,
            )
            river_pixel_count: np.ndarray = river_pixels.reshape(cell_shape).sum(
                axis=(1, 3)
            )
            cell_river_surface: np.ndarray = (
                np.where(river_pixels, original_dem, np.inf)
                .reshape(cell_shape)
                .min(axis=(1, 3))
            )
            local_rows, cols = np.where(
                (river_pixel_count > 0) & (grid_mask.values[row_start:row_end] > 0)
            )
            initial_level.values[row_start + local_rows, cols] = cell_river_surface[
                local_rows, cols
            ]

    initial_level.attrs["_FillValue"] = np.nan
    return initial_level


def set_river_initial_water_level(
    simulation: Any,
    DEMs: list[dict[str, Any]],
) -> str:
    """Set SFINCS initial water levels from the unburned DEM in river cells.

    Args:
        simulation: SFINCS simulation with its HydroMT-SFINCS model initialized.
        DEMs: DEMs used to build the SFINCS subgrid, as configured for the hydrodynamic model.

    Returns:
        A short summary for the SFINCS log.
    """
    sfincs_model = simulation.sfincs_model
    initial_level: xr.DataArray = river_initial_water_level(
        root=simulation.root_path,
        DEMs=DEMs,
        grid_mask=sfincs_model.grid.mask,
    )
    filled_cells: int = int(initial_level.notnull().sum().item())
    if filled_cells == 0:
        return "No burned river cells found; initial water levels were not changed."

    sfincs_model.initial_conditions.create(
        ini=initial_level,
        reproj_method="nearest",
    )
    # initial_conditions.create only sets the map in memory; the simulation does not
    # write the grid, so write the initial water level file here.
    sfincs_model.grid.write_map(
        map_fn=simulation.path / "sfincs.ini",
        data=sfincs_model.grid.data["ini"].values,
        mask=sfincs_model.grid.mask.values,
    )
    sfincs_model.config.write()
    return f"DEM-based initial water level set in {filled_cells} burned river cells"
