"""Snap discharge gauges directly to high-resolution river pixels."""

import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import QuadMesh
from pyproj import Geod
from scipy.ndimage import value_indices
from shapely.geometry import Point

MAX_SNAP_DISTANCE_M: float = 1500.0
MAX_AREA_DIFFERENCE_FRACTION: float = 0.1
GEOD: Geod = Geod(ellps="WGS84")


def group_routing_pixels(
    routing_river_ids: xr.DataArray,
) -> dict[int, tuple[np.ndarray, ...]]:
    """Group routing pixel indices by river ID.

    Args:
        routing_river_ids: Routing river IDs with dimensions y, x.

    Returns:
        Routing pixel indices keyed by river ID.

    Raises:
        ValueError: If the raster does not have dimensions y, x.
    """
    if routing_river_ids.dims != ("y", "x"):
        raise ValueError("River IDs must have dimensions (y, x).")
    river_id_values: np.ndarray = np.where(
        np.isfinite(routing_river_ids.values), routing_river_ids.values, -1
    ).astype(np.int64)
    return value_indices(river_id_values, ignore_value=-1)


class DischargeSnappingResults(NamedTuple):
    """Selected pixel centers (degrees), indices, areas (m²), and distance (m)."""

    original_pixel_lonlat: tuple[float, float]
    routing_pixel_lonlat: tuple[float, float]
    routing_pixel_xy: tuple[int, int]
    original_upstream_area_m2: float
    routing_upstream_area_m2: float
    station_to_original_distance_m: float
    river_id: int


def snap_discharge_station(
    station_location: Point,
    station_upstream_area_m2: float,
    original_upstream_area: xr.DataArray,
    original_river_ids: xr.DataArray,
    routing_upstream_area: xr.DataArray,
    routing_pixels_by_river_id: dict[int, tuple[np.ndarray, ...]],
) -> DischargeSnappingResults | None:
    """Snap a GRDC station to an original-resolution and routing pixel.

    The original pixel must be within 1,500 m, have an area within 10% of
    the GRDC area, and have a river ID. The routing pixel is the closest valid
    pixel with the same river ID. An original-to-routing area difference above 10%
    produces a warning but does not change the selected routing pixel.

    Args:
        station_location: GRDC station longitude and latitude (degrees, WGS84).
        station_upstream_area_m2: Reported GRDC upstream area (m²).
        original_upstream_area: Original-resolution upstream areas (m²), with
            WGS84 x/y axes.
        original_river_ids: Aligned original-resolution river IDs; NaN and
            negative IDs are missing.
        routing_upstream_area: Routing upstream areas (m²), with WGS84 x/y axes.
        routing_pixels_by_river_id: Routing pixel indices keyed by river ID.

    Returns:
        Selected pixels, or None if no valid match is found.

    Raises:
        ValueError: If coordinates are invalid or original rasters do not align.
    """
    if station_location.is_empty or not (
        np.isfinite(station_location.x)
        and np.isfinite(station_location.y)
        and -180 <= station_location.x <= 180
        and -90 <= station_location.y <= 90
    ):
        raise ValueError("Gauge coordinates must be finite WGS84 coordinates.")
    if not np.isfinite(station_upstream_area_m2) or station_upstream_area_m2 <= 0:
        return None
    if (
        original_upstream_area.dims != ("y", "x")
        or original_river_ids.dims != ("y", "x")
        or original_upstream_area.shape != original_river_ids.shape
    ):
        raise ValueError(
            "Original area and river ID rasters must have the same y, x shape."
        )

    # 1. Read the original-resolution window within 1.5 km of the station.
    latitude_margin: float = MAX_SNAP_DISTANCE_M / 110_000.0
    furthest_search_latitude: float = min(
        abs(station_location.y) + latitude_margin, 90.0
    )
    longitude_scale: float = max(np.cos(np.deg2rad(furthest_search_latitude)), 1e-12)
    longitude_margin: float = min(
        180.0,
        latitude_margin / longitude_scale,
    )
    original_columns: np.ndarray = np.flatnonzero(
        np.abs((original_upstream_area.x.values - station_location.x + 180) % 360 - 180)
        <= longitude_margin
    )
    original_rows: np.ndarray = np.flatnonzero(
        np.abs(original_upstream_area.y.values - station_location.y) <= latitude_margin
    )
    if not original_rows.size or not original_columns.size:
        return None

    local_original_areas: xr.DataArray = original_upstream_area.isel(
        y=original_rows, x=original_columns
    )
    local_original_river_ids: xr.DataArray = original_river_ids.isel(
        y=original_rows, x=original_columns
    )
    local_original_area_values: np.ndarray = local_original_areas.values
    local_original_river_id_values: np.ndarray = local_original_river_ids.values

    # 2. Keep original pixels with a river ID and area within 10% of GRDC.
    valid_original_pixels: np.ndarray = (
        np.isfinite(local_original_area_values)
        & (
            np.abs(local_original_area_values - station_upstream_area_m2)
            <= MAX_AREA_DIFFERENCE_FRACTION * station_upstream_area_m2
        )
        & np.isfinite(local_original_river_id_values)
        & (local_original_river_id_values >= 0)
    )
    candidate_original_rows: np.ndarray
    candidate_original_columns: np.ndarray
    candidate_original_rows, candidate_original_columns = np.nonzero(
        valid_original_pixels
    )
    if not candidate_original_rows.size:
        return None

    candidate_original_longitudes: np.ndarray = local_original_areas.x.values[
        candidate_original_columns
    ]
    candidate_original_latitudes: np.ndarray = local_original_areas.y.values[
        candidate_original_rows
    ]
    station_to_original_distances_m: np.ndarray = GEOD.inv(
        np.full(candidate_original_longitudes.shape, station_location.x),
        np.full(candidate_original_latitudes.shape, station_location.y),
        candidate_original_longitudes,
        candidate_original_latitudes,
    )[2]

    # 3. Select the closest original pixel and reject distances above 1.5 km.
    selected_original_index: int = int(station_to_original_distances_m.argmin())
    # Micrometer rounding prevents geodesic roundoff rejecting the exact boundary.
    station_to_original_distance_m: float = round(
        float(station_to_original_distances_m[selected_original_index]), 6
    )
    if station_to_original_distance_m > MAX_SNAP_DISTANCE_M:
        return None
    original_pixel_lonlat: tuple[float, float] = (
        float(candidate_original_longitudes[selected_original_index]),
        float(candidate_original_latitudes[selected_original_index]),
    )
    selected_original_row: int = int(candidate_original_rows[selected_original_index])
    selected_original_column: int = int(
        candidate_original_columns[selected_original_index]
    )
    river_id: int = int(
        local_original_river_id_values[selected_original_row, selected_original_column]
    )
    original_upstream_area_m2: float = float(
        local_original_area_values[selected_original_row, selected_original_column]
    )

    # 4. Select the closest valid routing pixel with the same MERIT river ID.
    if river_id not in routing_pixels_by_river_id:
        return None
    routing_rows: np.ndarray
    routing_columns: np.ndarray
    routing_rows, routing_columns = routing_pixels_by_river_id[river_id]
    routing_area_values: np.ndarray = routing_upstream_area.values
    valid_routing_areas: np.ndarray = np.isfinite(
        routing_area_values[routing_rows, routing_columns]
    ) & (routing_area_values[routing_rows, routing_columns] > 0)
    routing_rows = routing_rows[valid_routing_areas]
    routing_columns = routing_columns[valid_routing_areas]
    if not routing_rows.size:
        return None
    routing_longitudes: np.ndarray = routing_upstream_area.x.values[routing_columns]
    routing_latitudes: np.ndarray = routing_upstream_area.y.values[routing_rows]
    original_to_routing_distances_m: np.ndarray = GEOD.inv(
        np.full(routing_longitudes.shape, original_pixel_lonlat[0]),
        np.full(routing_latitudes.shape, original_pixel_lonlat[1]),
        routing_longitudes,
        routing_latitudes,
    )[2]
    selected_routing_index: int = int(original_to_routing_distances_m.argmin())
    routing_column: int = int(routing_columns[selected_routing_index])
    routing_row: int = int(routing_rows[selected_routing_index])
    routing_upstream_area_m2: float = float(
        routing_area_values[routing_row, routing_column]
    )

    # 5. Keep the closest routing pixel but warn when its area differs by over 10%.
    routing_area_difference: float = (
        abs(routing_upstream_area_m2 - original_upstream_area_m2)
        / original_upstream_area_m2
    )
    if routing_area_difference > MAX_AREA_DIFFERENCE_FRACTION:
        warnings.warn(
            f"GRDC station at ({station_location.x}, {station_location.y}): "
            "routing and original upstream areas differ by "
            f"{routing_area_difference:.1%}.",
            UserWarning,
            stacklevel=2,
        )
    return DischargeSnappingResults(
        original_pixel_lonlat=original_pixel_lonlat,
        routing_pixel_lonlat=(
            float(routing_longitudes[selected_routing_index]),
            float(routing_latitudes[selected_routing_index]),
        ),
        routing_pixel_xy=(routing_column, routing_row),
        original_upstream_area_m2=original_upstream_area_m2,
        routing_upstream_area_m2=routing_upstream_area_m2,
        station_to_original_distance_m=station_to_original_distance_m,
        river_id=river_id,
    )


def plot_discharge_snapping(
    station_id: int | str,
    output_folder: Path,
    station_lonlat: tuple[float, float],
    snapping_result: DischargeSnappingResults,
    original_upstream_area: xr.DataArray,
) -> None:
    """Save a gauge-to-original-to-routing plot without river centerlines.

    Args:
        station_id: Station identifier used in the filename.
        output_folder: Existing destination directory.
        station_lonlat: Gauge longitude and latitude (degrees).
        snapping_result: Selected pixels and upstream areas (m²).
        original_upstream_area: Original-resolution upstream area raster (m²).

    Raises:
        OSError: If the output cannot be written.
    """
    if not output_folder.is_dir():
        raise OSError(f"Snapping plot directory does not exist: {output_folder}")
    coordinates: np.ndarray = np.asarray(
        [
            station_lonlat,
            snapping_result.original_pixel_lonlat,
            snapping_result.routing_pixel_lonlat,
        ]
    )
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(9, 7))
    local_original_upstream_area: xr.DataArray = original_upstream_area.isel(
        x=np.flatnonzero(
            (original_upstream_area.x.values >= coordinates[:, 0].min() - 0.02)
            & (original_upstream_area.x.values <= coordinates[:, 0].max() + 0.02)
        ),
        y=np.flatnonzero(
            (original_upstream_area.y.values >= coordinates[:, 1].min() - 0.02)
            & (original_upstream_area.y.values <= coordinates[:, 1].max() + 0.02)
        ),
    )
    background: QuadMesh = ax.pcolormesh(
        local_original_upstream_area.x.values,
        local_original_upstream_area.y.values,
        local_original_upstream_area.values,
        shading="auto",
    )
    fig.colorbar(background, ax=ax, label="Original upstream area (m²)")
    ax.plot(coordinates[:, 0], coordinates[:, 1], "k--", linewidth=1)
    for index, label in enumerate(
        ("GRDC gauge", "Selected original pixel", "GEB routing pixel")
    ):
        ax.scatter(*coordinates[index], label=f"{index + 1}. {label}", zorder=3)
    ax.set_title(
        f"Station {station_id}: "
        "Original-pixel distance "
        f"{snapping_result.station_to_original_distance_m:.0f} m"
    )
    ax.legend()
    fig.savefig(
        output_folder / f"discharge_snapping_{station_id}.svg", bbox_inches="tight"
    )
    plt.close(fig)
