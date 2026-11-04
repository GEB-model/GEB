"""Embed bounded elevation and karst previews in standalone discharge dashboards."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import folium
import matplotlib
import numpy as np
import numpy.typing as npt
import xarray as xr
from folium.raster_layers import ImageOverlay
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

from geb.workflows.io import read_zarr


def _add_model_raster_layer(
    discharge_map: folium.Map,
    raster_path: Path,
    *,
    layer_id: str,
    layer_name: str,
    palette_name: str,
    pane_z_index: int,
    max_size: int,
    attribution: str,
    value_limits: tuple[float, float] | None = None,
    resampling: Resampling = Resampling.bilinear,
    transparent_zero: bool = False,
) -> tuple[ImageOverlay, dict[str, Any]] | None:
    """Embed a georeferenced model raster without loading its full resolution.

    Notes:
        The sampled image is projected to Web Mercator before embedding. Bounds
        and nodata remain aligned with the source; computation is sequential to
        bound native source-chunk memory on shared compute nodes.

    Args:
        discharge_map: Dashboard receiving the initially hidden image.
        raster_path: Model Zarr raster with x/y coordinates and a CRS.
        layer_id: Stable menu and pane identifier.
        layer_name: Human-readable map layer label.
        palette_name: Matplotlib colour palette name.
        pane_z_index: Pane ordering above basemaps and below vector features.
        max_size: Maximum preview width and height (pixels), at least two.
        attribution: Source credit displayed by Leaflet.
        value_limits: Optional valid range and colour scale in source units.
            Without limits, use the sampled raster's finite minimum and maximum.
        resampling: Reprojection method; nearest preserves coverage values.
        transparent_zero: Whether zero-valued pixels should be transparent.

    Returns:
        Hidden image layer and menu metadata in source units, or None when no
        finite raster values are available. Image dimensions are in pixels.

    Raises:
        ValueError: If preview size, raster georeferencing or values are invalid.
        FileNotFoundError: If the model raster dataset does not exist.
    """
    if max_size < 2:
        raise ValueError("Raster preview size must be at least two pixels.")
    if not raster_path.exists():
        raise FileNotFoundError(f"Model raster dataset does not exist: {raster_path}")
    with read_zarr(raster_path) as raster:
        if set(raster.dims) != {"y", "x"} or min(raster.shape) < 2:
            raise ValueError("Model raster must be a two-dimensional x/y raster.")
        if raster.rio.crs is None:
            raise ValueError("Model raster needs a coordinate reference system.")
        width: int = raster.sizes["x"]
        height: int = raster.sizes["y"]
        ratio: float = min(1.0, max_size / max(width, height))
        preview_width: int = max(2, round(width * ratio))
        preview_height: int = max(2, round(height * ratio))
        x_indices: npt.NDArray[np.int64] = np.minimum(
            ((np.arange(preview_width) + 0.5) * width / preview_width).astype(np.int64),
            width - 1,
        )
        y_indices: npt.NDArray[np.int64] = np.minimum(
            ((np.arange(preview_height) + 0.5) * height / preview_height).astype(
                np.int64
            ),
            height - 1,
        )
        sampled: xr.DataArray = raster.isel(x=x_indices, y=y_indices)
        # Limit source-chunk concurrency on large rasters, including shared nodes
        # where Dask's default worker count can exceed the Slurm CPU allocation.
        values: npt.NDArray[np.float32] = np.array(
            sampled.transpose("y", "x").compute(scheduler="synchronous").values,
            dtype=np.float32,
            copy=True,
        )
        nodata: float | None = raster.rio.nodata
        if nodata is not None:
            values[values == nodata] = np.nan
        values[~np.isfinite(values)] = np.nan
        if not np.isfinite(values).any():
            return None
        if value_limits is not None and (
            np.nanmin(values) < value_limits[0] or np.nanmax(values) > value_limits[1]
        ):
            raise ValueError(f"Raster values must be within {value_limits}.")
        # A north-up image must preserve orientation even for ascending y axes.
        if raster.y[0] < raster.y[-1]:
            values = values[::-1, :]
        if raster.x[0] > raster.x[-1]:
            values = values[:, ::-1]
        bounds: tuple[float, float, float, float] = raster.rio.bounds()
        projected_bounds: tuple[float, float, float, float] = transform_bounds(
            raster.rio.crs, "EPSG:3857", *bounds
        )
        if not np.isfinite(projected_bounds).all():
            raise ValueError("Model raster bounds cannot be displayed in Web Mercator.")
        projected: npt.NDArray[np.float32] = np.full(
            (preview_height, preview_width), np.nan, dtype=np.float32
        )
        reproject(
            source=values,
            destination=projected,
            src_transform=from_bounds(*bounds, preview_width, preview_height),
            src_crs=raster.rio.crs,
            src_nodata=np.nan,
            dst_transform=from_bounds(*projected_bounds, preview_width, preview_height),
            dst_crs="EPSG:3857",
            dst_nodata=np.nan,
            resampling=resampling,
        )
    valid: npt.NDArray[np.bool_] = np.isfinite(projected)
    if not valid.any():
        return None
    minimum: float = (
        value_limits[0] if value_limits is not None else float(np.nanmin(projected))
    )
    maximum: float = (
        value_limits[1] if value_limits is not None else float(np.nanmax(projected))
    )
    # Constant fields still need a nonzero normalization range.
    scale: float = maximum - minimum if maximum > minimum else 1.0
    normalized: npt.NDArray[np.float32] = np.nan_to_num(
        (projected - minimum) / scale, nan=0.0
    ).astype(np.float32, copy=False)
    palette: matplotlib.colors.Colormap = matplotlib.colormaps[palette_name]
    rgba: npt.NDArray[np.uint8] = palette(normalized, bytes=True)
    rgba[~valid | ((projected == 0) & transparent_zero), 3] = 0
    png: BytesIO = BytesIO()
    Image.fromarray(rgba).save(png, format="PNG")
    image_url: str = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode(
        "ascii"
    )
    west: float
    south: float
    east: float
    north: float
    west, south, east, north = transform_bounds(
        "EPSG:3857", "EPSG:4326", *projected_bounds
    )
    pane_name: str = "elevationPane" if layer_id == "dem" else f"{layer_id}Pane"
    folium.map.CustomPane(pane_name, z_index=pane_z_index, pointer_events=False).add_to(
        discharge_map
    )
    layer: ImageOverlay = ImageOverlay(
        image=image_url,
        bounds=[[south, west], [north, east]],
        name=layer_name,
        show=False,
        opacity=0.7,
        zindex=8,
        pixelated=resampling == Resampling.nearest,
        pane=pane_name,
        attribution=attribution,
    )
    layer.add_to(discharge_map)
    colors: list[str] = [
        matplotlib.colors.to_hex(palette(value)) for value in np.linspace(0, 1, 9)
    ]
    return layer, {
        "id": layer_id,
        "name": layer_name,
        "layer": layer.get_name(),
        "active": False,
        "opacity": 0.7,
        "minimum": minimum,
        "maximum": maximum,
        "colors": colors,
        "width": preview_width,
        "height": preview_height,
    }


def add_fathom_dem_layer(
    discharge_map: folium.Map, elevation_path: Path, max_size: int = 1024
) -> tuple[ImageOverlay, dict[str, Any]] | None:
    """Add a hidden FathomDEM overview with an elevation legend.

    Args:
        discharge_map: Dashboard receiving the overlay.
        elevation_path: Model FathomDEM Zarr raster (elevation in meters).
        max_size: Maximum preview width and height (pixels), at least two.

    Returns:
        Image layer and legend metadata in meters, or None for an empty DEM.

    Raises:
        ValueError: If the raster dimensions or georeferencing are invalid.
        FileNotFoundError: If the model DEM dataset is missing.
    """  # noqa: DOC502 - Document exceptions propagated by the raster loader.
    result: tuple[ImageOverlay, dict[str, Any]] | None = _add_model_raster_layer(
        discharge_map,
        elevation_path,
        layer_id="dem",
        layer_name="Elevation (FathomDEM)",
        palette_name="terrain",
        pane_z_index=250,
        max_size=max_size,
        attribution="FathomDEM © Hawker et al. (2025), CC BY-NC-SA 4.0",
    )
    if result is not None:
        result[1]["minimum_m"] = result[1].pop("minimum")
        result[1]["maximum_m"] = result[1].pop("maximum")
    return result


def add_karst_layer(
    discharge_map: folium.Map, karst_path: Path, max_size: int = 2048
) -> tuple[ImageOverlay, dict[str, Any]] | None:
    """Add model karst coverage with a fixed percentage legend and clear zeroes.

    Notes:
        Nearest-neighbour sampling preserves model fractions at boundaries.
        Zero coverage and nodata are transparent, leaving the basemap visible.
        Coverage describes mapped geology; runoff capture also depends on the
        model's karst settings.

    Args:
        discharge_map: Dashboard receiving the initially hidden overlay.
        karst_path: Model groundwater/karst_fraction Zarr raster (fractions 0–1).
        max_size: Maximum preview width and height (pixels), at least two.

    Returns:
        Image layer and legend metadata in percent (0–100), or None for an
        entirely missing raster. An all-zero raster remains available as a layer.

    Raises:
        ValueError: If fractions lie outside 0–1 or georeferencing is invalid.
        FileNotFoundError: If the model karst dataset is missing.
    """  # noqa: DOC502 - Document exceptions propagated by the raster loader.
    result: tuple[ImageOverlay, dict[str, Any]] | None = _add_model_raster_layer(
        discharge_map,
        karst_path,
        layer_id="karst",
        layer_name="Karst coverage",
        palette_name="YlOrBr",
        pane_z_index=350,
        max_size=max_size,
        attribution="Karst coverage: model groundwater/karst_fraction (WOKAM)",
        value_limits=(0.0, 1.0),
        resampling=Resampling.nearest,
        transparent_zero=True,
    )
    if result is not None:
        result[1]["minimum_percent"] = 100 * result[1].pop("minimum")
        result[1]["maximum_percent"] = 100 * result[1].pop("maximum")
    return result
