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

    original_subgrid_pixel_lonlat: tuple[float, float]
    routing_pixel_lonlat: tuple[float, float]
    routing_pixel_xy: tuple[int, int]
    original_subgrid_upstream_area_m2: float
    routing_upstream_area_m2: float
    station_to_original_subgrid_distance_m: float
    river_id: int


def snap_discharge_station(
    station_location: Point,
    station_upstream_area_m2: float | None,
    original_subgrid_upstream_area: xr.DataArray,
    original_subgrid_river_ids: xr.DataArray,
    routing_upstream_area: xr.DataArray,
    routing_pixels_by_river_id: dict[int, tuple[np.ndarray, ...]],
) -> DischargeSnappingResults | None:
    """Snap a gauging station to an original subgrid and routing pixel.

    The snapping is done in 4 steps:
    1. Read a bounding window around the station to limit raster access.
    2. Filter the selected subgrid to keep only pixels with a river ID and within the station-GEB upstream area tolerance (MAX_AREA_DIFFERENCE_FRACTION).
    3. Select the closest original subgrid pixel and reject distances above the snapping distance (MAX_SNAP_DISTANCE_M).
    4. Select the closest valid routing pixel with the same MERIT river ID.

    The window is rectangular, so step 3 checks the actual geodesic distance
    against MAX_SNAP_DISTANCE_M. Custom stations without an area use distance only.
    Area differences between resolutions produce a warning, not a failed snap:
    selecting a farther routing pixel would change the intended location.

    Args:
        station_location: Station longitude and latitude (degrees, WGS84).
        station_upstream_area_m2: Gauge upstream area (m²), or None for custom
            stations without area metadata.
        original_subgrid_upstream_area: Original subgrid upstream areas (m²), with
            WGS84 x/y axes.
        original_subgrid_river_ids: Aligned original subgrid river IDs; NaN and
            negative IDs are missing.
        routing_upstream_area: Routing upstream areas (m²), with WGS84 x/y axes.
        routing_pixels_by_river_id: Routing row and column indices keyed by river ID.

    Returns:
        Selected pixels, or None if no valid match is found.

    Raises:
        ValueError: If station coordinates are invalid.
    """
    if station_location.is_empty or not (
        -180 <= station_location.x <= 180 and -90 <= station_location.y <= 90
    ):
        raise ValueError("Gauge coordinates must be finite WGS84 coordinates.")
    if station_upstream_area_m2 is not None and (
        not np.isfinite(station_upstream_area_m2) or station_upstream_area_m2 <= 0
    ):
        return None  # Invalid station area; cannot snap to original subgrid

    # 1. Read a bounding window around the station to limit raster access.
    latitude_margin: float = MAX_SNAP_DISTANCE_M / 110_000.0
    furthest_search_latitude: float = min(
        abs(station_location.y) + latitude_margin, 90.0
    )
    longitude_scale: float = max(np.cos(np.deg2rad(furthest_search_latitude)), 1e-12)
    longitude_margin: float = min(
        180.0,
        latitude_margin / longitude_scale,
    )
    original_subgrid_columns: np.ndarray = np.flatnonzero(
        np.abs(
            (original_subgrid_upstream_area.x.values - station_location.x + 180) % 360
            - 180
        )
        <= longitude_margin
    )
    original_subgrid_rows: np.ndarray = np.flatnonzero(
        np.abs(original_subgrid_upstream_area.y.values - station_location.y)
        <= latitude_margin
    )

    local_original_subgrid_areas: xr.DataArray = original_subgrid_upstream_area.isel(
        y=original_subgrid_rows, x=original_subgrid_columns
    )
    local_original_subgrid_river_ids: xr.DataArray = original_subgrid_river_ids.isel(
        y=original_subgrid_rows, x=original_subgrid_columns
    )
    local_original_subgrid_area_values: np.ndarray = local_original_subgrid_areas.values
    local_original_subgrid_river_id_values: np.ndarray = (
        local_original_subgrid_river_ids.values
    )

    # 2. Filter the selected subgrid to keep only pixels with a river ID and within the station-GEB upstream area tolerance (MAX_AREA_DIFFERENCE_FRACTION).
    valid_original_subgrid_pixels: np.ndarray = (
        np.isfinite(local_original_subgrid_area_values)
        & (local_original_subgrid_area_values > 0)
        & np.isfinite(local_original_subgrid_river_id_values)
        & (local_original_subgrid_river_id_values >= 0)
    )
    if station_upstream_area_m2 is not None:
        valid_original_subgrid_pixels &= (
            np.abs(local_original_subgrid_area_values - station_upstream_area_m2)
            <= MAX_AREA_DIFFERENCE_FRACTION * station_upstream_area_m2
        )  # Filter pixels by max upstream area difference
    candidate_original_subgrid_rows: np.ndarray
    candidate_original_subgrid_columns: np.ndarray
    candidate_original_subgrid_rows, candidate_original_subgrid_columns = np.nonzero(
        valid_original_subgrid_pixels
    )
    if not candidate_original_subgrid_rows.size:
        return None  # No valid original subgrid pixels within the area tolerance

    candidate_original_subgrid_longitudes: np.ndarray = (
        local_original_subgrid_areas.x.values[candidate_original_subgrid_columns]
    )
    candidate_original_subgrid_latitudes: np.ndarray = (
        local_original_subgrid_areas.y.values[candidate_original_subgrid_rows]
    )

    # 3. Select the closest original subgrid pixel and reject distances above the snapping distance (MAX_SNAP_DISTANCE_M).
    station_to_original_subgrid_distances_m: np.ndarray = GEOD.inv(
        np.full(candidate_original_subgrid_longitudes.shape, station_location.x),
        np.full(candidate_original_subgrid_latitudes.shape, station_location.y),
        candidate_original_subgrid_longitudes,
        candidate_original_subgrid_latitudes,
    )[2]  # Compute distances from station to candidate original subgrid pixels

    selected_original_subgrid_index: int = int(
        station_to_original_subgrid_distances_m.argmin()
    )
    station_to_original_subgrid_distance_m: float = round(
        float(station_to_original_subgrid_distances_m[selected_original_subgrid_index]),
        6,
    )
    if station_to_original_subgrid_distance_m > MAX_SNAP_DISTANCE_M:
        return None
    original_subgrid_pixel_lonlat: tuple[float, float] = (
        float(candidate_original_subgrid_longitudes[selected_original_subgrid_index]),
        float(candidate_original_subgrid_latitudes[selected_original_subgrid_index]),
    )
    selected_original_subgrid_row: int = int(
        candidate_original_subgrid_rows[selected_original_subgrid_index]
    )
    selected_original_subgrid_column: int = int(
        candidate_original_subgrid_columns[selected_original_subgrid_index]
    )
    river_id: int = int(
        local_original_subgrid_river_id_values[
            selected_original_subgrid_row, selected_original_subgrid_column
        ]
    )
    original_subgrid_upstream_area_m2: float = float(
        local_original_subgrid_area_values[
            selected_original_subgrid_row, selected_original_subgrid_column
        ]
    )

    # 4. Select the closest valid routing pixel with the same MERIT river ID.
    if river_id not in routing_pixels_by_river_id:
        return None  # No routing pixels for this river ID
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
    original_subgrid_to_routing_distances_m: np.ndarray = GEOD.inv(
        np.full(routing_longitudes.shape, original_subgrid_pixel_lonlat[0]),
        np.full(routing_latitudes.shape, original_subgrid_pixel_lonlat[1]),
        routing_longitudes,
        routing_latitudes,
    )[2]
    selected_routing_index: int = int(original_subgrid_to_routing_distances_m.argmin())
    routing_column: int = int(routing_columns[selected_routing_index])
    routing_row: int = int(routing_rows[selected_routing_index])
    routing_pixel_xy: tuple[int, int] = (routing_column, routing_row)
    routing_pixel_lonlat: tuple[float, float] = (
        float(routing_longitudes[selected_routing_index]),
        float(routing_latitudes[selected_routing_index]),
    )
    routing_upstream_area_m2: float = float(
        routing_area_values[routing_row, routing_column]
    )
    # Area differences between resolutions should not change the selected pixel.
    routing_area_difference: float = (
        abs(routing_upstream_area_m2 - original_subgrid_upstream_area_m2)
        / original_subgrid_upstream_area_m2
    )
    if routing_area_difference > MAX_AREA_DIFFERENCE_FRACTION:
        warnings.warn(
            f"Gauging station at ({station_location.x}, {station_location.y}): "
            "routing and original subgrid upstream areas differ by "
            f"{routing_area_difference:.1%}.",
            UserWarning,
            stacklevel=2,
        )
    result: DischargeSnappingResults = DischargeSnappingResults(
        original_subgrid_pixel_lonlat=original_subgrid_pixel_lonlat,
        routing_pixel_lonlat=routing_pixel_lonlat,
        routing_pixel_xy=routing_pixel_xy,
        original_subgrid_upstream_area_m2=original_subgrid_upstream_area_m2,
        routing_upstream_area_m2=routing_upstream_area_m2,
        station_to_original_subgrid_distance_m=station_to_original_subgrid_distance_m,
        river_id=river_id,
    )

    return result


def plot_discharge_snapping(
    station_id: int | str,
    output_folder: Path,
    station_lonlat: tuple[float, float],
    snapping_result: DischargeSnappingResults,
    original_subgrid_upstream_area: xr.DataArray,
) -> None:
    """Save a gauge-to-original-subgrid-to-routing plot.

    Args:
        station_id: Station identifier used in the filename.
        output_folder: Existing destination directory.
        station_lonlat: Gauge longitude and latitude (degrees).
        snapping_result: Selected pixels and upstream areas (m²).
        original_subgrid_upstream_area: Original subgrid upstream area raster (m²).

    Raises:
        OSError: If the output cannot be written.
    """
    if not output_folder.is_dir():
        raise OSError(f"Snapping plot directory does not exist: {output_folder}")
    coordinates: np.ndarray = np.asarray(
        [
            station_lonlat,
            snapping_result.original_subgrid_pixel_lonlat,
            snapping_result.routing_pixel_lonlat,
        ]
    )
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(9, 7))
    local_original_subgrid_upstream_area: xr.DataArray = (
        original_subgrid_upstream_area.isel(
            x=np.flatnonzero(
                (
                    original_subgrid_upstream_area.x.values
                    >= coordinates[:, 0].min() - 0.02
                )
                & (
                    original_subgrid_upstream_area.x.values
                    <= coordinates[:, 0].max() + 0.02
                )
            ),
            y=np.flatnonzero(
                (
                    original_subgrid_upstream_area.y.values
                    >= coordinates[:, 1].min() - 0.02
                )
                & (
                    original_subgrid_upstream_area.y.values
                    <= coordinates[:, 1].max() + 0.02
                )
            ),
        )
    )
    background: QuadMesh = ax.pcolormesh(
        local_original_subgrid_upstream_area.x.values,
        local_original_subgrid_upstream_area.y.values,
        local_original_subgrid_upstream_area.values,
        shading="auto",
    )
    fig.colorbar(background, ax=ax, label="Original subgrid upstream area (m²)")
    ax.plot(coordinates[:, 0], coordinates[:, 1], "k--", linewidth=1)
    for index, label in enumerate(
        ("Gauging station", "Selected original subgrid pixel", "GEB routing pixel")
    ):
        ax.scatter(*coordinates[index], label=f"{index + 1}. {label}", zorder=3)
    ax.set_title(
        f"Station {station_id}: "
        "Original-pixel distance "
        f"{snapping_result.station_to_original_subgrid_distance_m:.0f} m"
    )
    ax.legend()
    fig.savefig(
        output_folder / f"discharge_snapping_{station_id}.svg", bbox_inches="tight"
    )
    plt.close(fig)
