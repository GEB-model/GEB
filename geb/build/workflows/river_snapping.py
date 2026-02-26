"""This module contains functions for snapping points to the river network and visualizing the results."""

from __future__ import annotations

import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from shapely.ops import nearest_points

from geb.workflows.io import get_window


def plot_snapping(
    point_id: int | str,
    output_folder: Path,
    rivers: gpd.GeoDataFrame,
    upstream_area: xr.DataArray,
    original_coords: tuple[float, float],
    closest_point_coords: tuple[float, float],
    closest_river_segment: gpd.GeoDataFrame,
    grid_pixel_xy: tuple[int, int],
    filename_prefix: str = "snapping",
    point_label: str = "Original point",
    title: str | None = None,
) -> None:
    """Create and save a map visualizing point snapping results to the river network.

    This function produces a map centered on the original point location that shows:
    - the original point location,
    - the closest point on the river centerline,
    - a square around the selected grid pixel,
    - the upstream area raster (within the map extent), and
    - the closest river segment.

    The resulting figure is saved as a SVG file inside the provided output_folder.

    Args:
        point_id: Identifier or name of the point used in the plot title and output filename.
        output_folder: Path to the directory where the SVG file will be saved.
        rivers: GeoDataFrame containing river centerlines used for plotting.
        upstream_area: xarray DataArray with upstream area values used for background plotting.
        original_coords: Tuple (lon, lat) of the original point coordinates.
        closest_point_coords: Tuple (lon, lat) of the closest point on the river centerline.
        closest_river_segment: GeoDataFrame containing the selected river segment to highlight.
        grid_pixel_xy: Tuple (x, y) indices of the snapped grid pixel.
        filename_prefix: Prefix for the output filename. Default is "snapping".
        point_label: Label for the original point in the legend. Default is "Original point".
        title: Optional custom title for the plot. If None, a default title is generated.

    """
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(15, 10)
    )
    ax.coastlines()  # ty:ignore[unresolved-attribute]
    ax.add_feature(cfeature.BORDERS)  # ty:ignore[unresolved-attribute]
    ax.add_feature(cfeature.LAND)  # ty:ignore[unresolved-attribute]
    ax.add_feature(cfeature.OCEAN)  # ty:ignore[unresolved-attribute]
    ax.add_feature(cfeature.LAKES)  # ty:ignore[unresolved-attribute]

    # Set the extent to zoom in around the point location
    buffer = 0.05  # Adjust this value to control the zoom level

    xmin = original_coords[0] - buffer
    xmax = original_coords[0] + buffer
    ymin = original_coords[1] - buffer
    ymax = original_coords[1] + buffer

    ax.set_extent(  # ty:ignore[unresolved-attribute]
        [
            xmin,
            xmax,
            ymin,
            ymax,
        ],
        crs=ccrs.PlateCarree(),
    )

    ax.scatter(
        original_coords[0],
        original_coords[1],
        color="red",
        marker="o",
        s=30,
        label=point_label,
        zorder=3,
    )
    ax.scatter(
        closest_point_coords[0],
        closest_point_coords[1],
        color="black",
        marker="o",
        s=30,
        label="Closest point to river",
        zorder=3,
    )

    # Plot the snapped grid pixel as a square
    # snapped grid pixel coordinates
    lon = upstream_area.x.isel(x=grid_pixel_xy[0]).values.item()
    lat = upstream_area.y.isel(y=grid_pixel_xy[1]).values.item()

    # Get resolution
    dx = float((upstream_area.x[1] - upstream_area.x[0]).values.item())
    dy = float((upstream_area.y[1] - upstream_area.y[0]).values.item())

    # Plot square
    rect = mpatches.Rectangle(
        (lon - dx / 2, lat - dy / 2),
        dx,
        dy,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label="Grid pixel",
        zorder=3,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(rect)

    # Select the upstream area within the extent of the plot
    upstream_area_within_extent = upstream_area.isel(
        get_window(
            upstream_area.x,
            upstream_area.y,
            bounds=(xmin, ymin, xmax, ymax),
            buffer=1,
            raise_on_buffer_out_of_bounds=False,
            raise_on_out_of_bounds=False,
        )
    )

    upstream_area_within_extent.plot(
        ax=ax,
        cmap="viridis",
        cbar_kwargs={"label": "Upstream area [m2]"},
        zorder=0,
        alpha=1,
    )  # ty:ignore[missing-argument]
    rivers.plot(ax=ax, color="blue", linewidth=1)
    closest_river_segment.plot(
        ax=ax, color="green", linewidth=3, label="Closest river segment"
    )

    plot_title = (
        title
        if title is not None
        else f"Upstream area grid and snapping for {point_id}"
    )
    ax.set_title(plot_title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.savefig(
        output_folder / f"{filename_prefix}_{point_id}.svg",
        bbox_inches="tight",
    )
    plt.close()


def snap_point_to_river_network(
    point: shapely.geometry.Point,
    rivers: gpd.GeoDataFrame,
    upstream_area_grid: xr.DataArray,
    upstream_area_subgrid: xr.DataArray,
    upstream_area_m2: float | None = None,
    max_uparea_difference_ratio: float = 0.3,
    max_spatial_difference_degrees: float = 0.1,
) -> dict | None:
    """Snap a point to the river network grid.

    This function finds the closest river segment (optionally matching upstream area),
    snaps the point to that segment, and then finds the corresponding cell in the
    low-resolution grid that is part of the river network.

    Args:
        point: The location to snap (shapely Point).
        rivers: GeoDataFrame of river segments. Must contain 'uparea_m2' and 'hydrography_xy'.
        upstream_area_grid: Low-resolution upstream area grid.
        upstream_area_subgrid: High-resolution upstream area subgrid.
        upstream_area_m2: Optional target upstream area for matching (m2).
        max_uparea_difference_ratio: Max allowed difference ratio for upstream area.
        max_spatial_difference_degrees: Max allowed spatial distance in degrees.

    Returns:
        Dictionary with snapping results or None if no segment found.
        The dictionary contains:
            - closest_point_coords: (lon, lat) on the river line.
            - subgrid_pixel_coords: (lon, lat) of the corresponding high-res pixel.
            - snapped_grid_pixel_lonlat: (lon, lat) of the snapped low-res grid cell.
            - snapped_grid_pixel_xy: (x_idx, y_idx) indices in the low-res grid.
            - geb_uparea_subgrid: Upstream area from the subgrid (m2).
            - geb_uparea_grid: Upstream area from the low-res grid (m2).
            - distance_degrees: Distance from original point to segment (degrees).
            - closest_river_segment: The selected river segment (GeoDataFrame).
    """
    # Calculate distances and sort from closest to furthest
    rivers = rivers.copy()
    rivers = rivers[
        rivers["represented_in_grid"]
    ]  # filter to river segments that are represented in the grid

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        rivers["station_distance"] = rivers.geometry.distance(point)

    rivers_sorted = rivers.sort_values(by="station_distance")

    # We first need to find the best river segment to snap to, based on distance and (optionally) upstream area.
    best_river_segment = select_river_segment(
        rivers_sorted=rivers_sorted,
        max_spatial_difference_degrees=max_spatial_difference_degrees,
        upstream_area_m2=upstream_area_m2,
        max_uparea_difference_ratio=max_uparea_difference_ratio,
    )

    if best_river_segment is None:
        return None

    # Then along the selected river segment, we find the closest point on the river line to the original point.
    # The first point returned by nearest_points is the point itself,
    # the second is the closest point on the linestring.
    _, closest_point_on_riverline = nearest_points(
        point, best_river_segment.iloc[0].geometry
    )

    # Next, we find the corresponding cell in the high-resolution subgrid that is part of the river network.
    river_cell_in_subgrid = upstream_area_subgrid.sel(
        x=closest_point_on_riverline.x,
        y=closest_point_on_riverline.y,
        method="nearest",
    ).compute()

    # then we find the river cell in the low resolution grid that is closest to the snapped
    # river point, and that is part of the same river segment.
    # hydrography_xy contains the list of (x,y) coordinates in the low-res grid that belong to the river segment.
    hydrography_xy = best_river_segment.iloc[0]["hydrography_xy"]
    river_points_and_xy = []
    for xy in hydrography_xy:
        river_points_and_xy.append(
            (
                shapely.geometry.Point(
                    upstream_area_grid.x.values[xy[0]].item(),
                    upstream_area_grid.y.values[xy[1]].item(),
                ),
                xy,
            )
        )
    closest_river_point_and_xy = min(
        river_points_and_xy,
        key=lambda x: shapely.distance(x[0], closest_point_on_riverline),
    )

    # Determine if we should include upstream area values in the result
    include_uparea = upstream_area_m2 is not None and not np.isnan(upstream_area_m2)

    return {
        "closest_point_coords": (
            float(closest_point_on_riverline.x),
            float(closest_point_on_riverline.y),
        ),
        "subgrid_pixel_coords": (
            river_cell_in_subgrid.x.item(),
            river_cell_in_subgrid.y.item(),
        ),
        "snapped_grid_pixel_lonlat": (
            closest_river_point_and_xy[0].x,
            closest_river_point_and_xy[0].y,
        ),
        "snapped_grid_pixel_xy": closest_river_point_and_xy[1],
        "geb_uparea_subgrid": (
            (river_cell_in_subgrid.x.item(), river_cell_in_subgrid.y.item())
            if include_uparea
            else None
        ),
        "geb_uparea_grid": (
            upstream_area_grid.isel(
                x=closest_river_point_and_xy[1][0], y=closest_river_point_and_xy[1][1]
            ).item()
            if include_uparea
            else None
        ),
        "distance_degrees": best_river_segment.iloc[0].station_distance,
        "closest_river_segment": best_river_segment,
    }


def select_river_segment(
    rivers_sorted: gpd.GeoDataFrame,
    max_spatial_difference_degrees: float,
    upstream_area_m2: float | None = None,
    max_uparea_difference_ratio: float = 0.3,
) -> gpd.GeoDataFrame | None:
    """Select the closest river segment that matches optional upstream area criteria.

    Args:
        rivers_sorted: GeoDataFrame of river segments sorted by spatial distance. Must contain 'station_distance' and 'uparea_m2'.
        max_spatial_difference_degrees: The maximum allowed spatial difference in degrees.
        upstream_area_m2: Optional target upstream area (m2). If provided (and not NaN), matching is performed.
        max_uparea_difference_ratio: The maximum allowed difference ratio for upstream area matching.

    Returns:
        The selected river segment or None if no segment meets the criteria.
    """
    if upstream_area_m2 is None or np.isnan(upstream_area_m2):
        closest_river_segment = rivers_sorted.head(1)
    else:
        # add upstream area criteria
        upstream_area_diff = max_uparea_difference_ratio * upstream_area_m2
        closest_river_segment = rivers_sorted[
            (rivers_sorted["uparea_m2"] > (upstream_area_m2 - upstream_area_diff))
            & (rivers_sorted["uparea_m2"] < (upstream_area_m2 + upstream_area_diff))
        ].head(1)

    if closest_river_segment.empty:
        return None

    if closest_river_segment.iloc[0].station_distance > max_spatial_difference_degrees:
        return None

    return closest_river_segment.iloc[0:1]
