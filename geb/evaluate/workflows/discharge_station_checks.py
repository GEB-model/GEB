"""Check station snapping and the location recorded with simulated discharge."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Geod

GEOD: Geod = Geod(ellps="WGS84")
DASHBOARD_COLUMNS: dict[str, str] = {
    "discharge_observations_station_name": "station_name",
    "discharge_observations_upstream_area_m2": "upstream_area_GRDC",
    "GEB_upstream_area_from_original": "upstream_area_GEB_original",
    "GEB_upstream_area_from_grid": "upstream_area_GEB",
}


def station_exclusion_reason(
    station: pd.Series,
    report_path: Path,
    report_station: pd.Series | None = None,
) -> str:
    """Return why a station cannot be scored, or an empty string.

    Args:
        station: Built station metadata with areas in m² and coordinates in degrees.
        report_path: Simulated station discharge parquet file.
        report_station: Optional location saved for a legacy report.

    Returns:
        Exclusion reason. Empty means the distance and report checks pass.

    """
    original_area: float = float(station["GEB_upstream_area_from_original"])
    routing_area: float = float(station["GEB_upstream_area_from_grid"])
    if (
        not np.isfinite(original_area)
        or original_area <= 0
        or not np.isfinite(routing_area)
    ):
        return "Missing upstream area."
    original_pixel: np.ndarray = np.asarray(station["original_pixel_lonlat"])
    routing_pixel: np.ndarray = np.asarray(station["snapped_grid_pixel_lonlat"])
    original_to_routing_distance_m: float = GEOD.inv(*original_pixel, *routing_pixel)[2]
    if (
        not np.isfinite(original_to_routing_distance_m)
        or round(original_to_routing_distance_m, 6) > 1500
    ):
        return "Routing pixel is more than 1.5 km from the original pixel."
    if not report_path.exists():
        return "No simulated discharge report."
    metadata: dict[bytes, bytes] = pq.read_schema(report_path).metadata or {}
    location: dict[str, Any] = (
        json.loads(metadata.get(b"pandas", b"{}"))
        .get("attributes", {})
        .get("station_location", {})
    )
    if not location and report_station is not None:
        location = {
            "pixel_xy": list(report_station["snapped_grid_pixel_xy"]),
            "longitude_latitude": list(report_station["snapped_grid_pixel_lonlat"]),
            "upstream_area_m2": float(report_station["GEB_upstream_area_from_grid"]),
        }
    if not location:
        return "Report location is unknown. Rerun the simulation."
    if (
        location.get("pixel_xy") != list(station["snapped_grid_pixel_xy"])
        or not np.allclose(
            location.get("longitude_latitude", [np.nan, np.nan]),
            routing_pixel,
            rtol=0,
            atol=1e-8,
        )
        or not np.isclose(
            location.get("upstream_area_m2", np.nan), routing_area, rtol=1e-6
        )
    ):
        return (
            "Report uses an older routing cell or upstream area. Rerun the simulation."
        )
    return ""


def excluded_station_locations(
    stations: gpd.GeoDataFrame, report_folder: Path
) -> gpd.GeoDataFrame:
    """Build dashboard records for stations that fail routing or report checks.

    Args:
        stations: Built station locations and upstream areas (m²).
        report_folder: Run report directory containing hydrology.routing.

    Returns:
        Excluded stations with dashboard coordinates (degrees) and reasons.

    """
    legacy_locations_path: Path = (
        report_folder / "hydrology.routing" / "discharge_station_locations.geoparquet"
    )
    legacy_locations: gpd.GeoDataFrame = (
        gpd.read_parquet(legacy_locations_path)
        if legacy_locations_path.exists()
        else gpd.GeoDataFrame()
    )
    reasons: pd.Series = pd.Series("", index=stations.index)
    for station_id, station in stations.iterrows():
        reasons.loc[station_id] = station_exclusion_reason(
            station,
            report_folder
            / "hydrology.routing"
            / f"discharge_hourly_m3_per_s_{station_id}.parquet",
            legacy_locations.loc[station_id]
            if station_id in legacy_locations.index
            else None,
        )
    rejected: gpd.GeoDataFrame = stations.loc[reasons.ne("")]
    if rejected.empty:
        return gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs=stations.crs
        )
    excluded: gpd.GeoDataFrame = rejected.rename(columns=DASHBOARD_COLUMNS)[
        [*DASHBOARD_COLUMNS.values(), "geometry"]
    ]
    excluded.index.name = "station_ID"
    excluded["exclusion_reason"] = reasons
    for target, source in (
        ("station", "discharge_observations_station_coords"),
        ("original", "original_pixel_lonlat"),
        ("snapped_grid", "snapped_grid_pixel_lonlat"),
    ):
        excluded[[f"{target}_longitude", f"{target}_latitude"]] = np.asarray(
            rejected[source].tolist()
        )
    for column in (
        "station_to_original_distance_m",
        "snapped_river_id",
        "snapping_method",
        "timezone_utc_offset",
    ):
        excluded[column] = rejected.get(column, 0.0)
    return excluded


def complete_dashboard_stations(
    evaluated: gpd.GeoDataFrame,
    excluded: gpd.GeoDataFrame,
    snapped: gpd.GeoDataFrame,
    snapped_path: Path,
    minimum_upstream_area_km2: float = 0.0,
) -> gpd.GeoDataFrame:
    """Keep every regional station visible regardless of evaluation filters.

    Args:
        evaluated: Stations with validated or diagnostic scores.
        excluded: Stations already excluded by snapping or report checks.
        snapped: Already loaded snapped stations, with upstream areas in m².
        snapped_path: Built snapped locations file, next to the station inventory.
        minimum_upstream_area_km2: Minimum routing area allowed for evaluation (km²).

    Returns:
        Excluded stations including gauges without a snap or usable time series.
    """
    inventory_path: Path = snapped_path.with_name("station_locations.geoparquet")
    inventory: gpd.GeoDataFrame = (
        gpd.read_parquet(inventory_path) if inventory_path.exists() else snapped
    )
    saved_reasons: pd.Series = inventory.get(
        "evaluation_exclusion_reason", pd.Series("", index=inventory.index)
    ).fillna("")
    saved_reasons = saved_reasons.loc[saved_reasons.ne("")]
    missing_ids: pd.Index = (
        inventory.index.difference(evaluated.index)
        .union(saved_reasons.index)
        .difference(excluded.index)
    )
    missing: gpd.GeoDataFrame = inventory.loc[missing_ids].rename(
        columns=DASHBOARD_COLUMNS
    )[["station_name", "upstream_area_GRDC", "geometry"]]
    reasons: pd.Series = pd.Series(
        "No valid river match within 1.5 km and 10% upstream area.",
        index=inventory.index,
    )
    area: pd.Series = inventory["discharge_observations_upstream_area_m2"]
    reasons.loc[~np.isfinite(area) | (area <= 0)] = "Missing GRDC upstream area."
    reasons.loc[reasons.index.intersection(snapped.index)] = (
        "No paired discharge record meets the evaluation period or record-length requirements."
    )
    small: pd.Index = snapped.index[
        snapped["GEB_upstream_area_from_grid"] < minimum_upstream_area_km2 * 1e6
    ]
    reasons.loc[reasons.index.intersection(small)] = (
        f"Routing upstream area is below the {minimum_upstream_area_km2:g} km² evaluation threshold."
    )
    missing["exclusion_reason"] = reasons.reindex(missing.index)
    missing["station_longitude"] = missing.geometry.x
    missing["station_latitude"] = missing.geometry.y
    missing.index.name = "station_ID"
    result: gpd.GeoDataFrame = gpd.GeoDataFrame(
        pd.concat([excluded, missing]),
        geometry="geometry",
        crs=inventory.crs,
    )
    # Duplicate reasons override older saved scores and other exclusion reasons.
    saved_ids: pd.Index = result.index.intersection(saved_reasons.index)
    result.loc[saved_ids, "exclusion_reason"] = saved_reasons.loc[saved_ids]
    return result
