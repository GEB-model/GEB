"""Check station snapping and the location recorded with simulated discharge."""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Geod

GEOD: Geod = Geod(ellps="WGS84")
DASHBOARD_COLUMNS: dict[str, str] = {
    "discharge_observations_station_name": "station_name",
    "discharge_observations_upstream_area_m2": "upstream_area_GRDC",
    "GEB_upstream_area_from_original_subgrid": "upstream_area_GEB_original_subgrid",
    "GEB_upstream_area_from_grid": "upstream_area_GEB",
}


def get_station_exclusion_reason(
    station: pd.Series,
    report_path: Path,
) -> str:
    """Return why a station cannot be scored, or an empty string.

    Args:
        station: Built station metadata with areas in m² and coordinates in degrees.
        report_path: Simulated station discharge parquet file.

    Returns:
        Exclusion reason. Empty means the distance and report checks pass.

    """
    original_subgrid_area: float = float(
        station["GEB_upstream_area_from_original_subgrid"]
    )
    routing_area: float = float(station["GEB_upstream_area_from_grid"])
    if (
        not np.isfinite(original_subgrid_area)
        or original_subgrid_area <= 0
        or not np.isfinite(routing_area)
        or routing_area <= 0
    ):
        return "Missing upstream area."
    original_subgrid_pixel: np.ndarray = np.asarray(
        station["original_subgrid_pixel_lonlat"]
    )
    routing_pixel: np.ndarray = np.asarray(station["snapped_grid_pixel_lonlat"])
    original_subgrid_to_routing_distance_m: float = GEOD.inv(
        *original_subgrid_pixel, *routing_pixel
    )[2]
    if (
        not np.isfinite(original_subgrid_to_routing_distance_m)
        or round(original_subgrid_to_routing_distance_m, 6) > 1500
    ):
        return "Routing pixel is more than 1.5 km from the original subgrid pixel."
    if not report_path.exists():
        return "No simulated discharge report."
    metadata: dict[bytes, bytes] = pq.read_schema(report_path).metadata or {}
    location: dict[str, Any] = (
        json.loads(metadata.get(b"pandas", b"{}"))
        .get("attributes", {})
        .get("station_location", {})
    )
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


def find_excluded_stations(
    stations: gpd.GeoDataFrame, report_folder: Path
) -> gpd.GeoDataFrame:
    """Identify stations excluded from evaluation and prepare their map metadata.

    Args:
        stations: Built station locations and upstream areas (m²).
        report_folder: Run report directory containing hydrology.routing.

    Returns:
        Excluded stations with dashboard coordinates (degrees) and reasons.

    """
    reasons: pd.Series = pd.Series("", index=stations.index)
    for station_id, station in stations.iterrows():
        reasons.loc[station_id] = get_station_exclusion_reason(
            station,
            report_folder
            / "hydrology.routing"
            / f"discharge_hourly_m3_per_s_{station_id}.parquet",
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
        ("original_subgrid", "original_subgrid_pixel_lonlat"),
        ("routing_grid", "snapped_grid_pixel_lonlat"),
    ):
        excluded[[f"{target}_longitude", f"{target}_latitude"]] = np.asarray(
            rejected[source].tolist()
        )
    for column in (
        "station_to_original_subgrid_distance_m",
        "snapped_river_id",
        "snapping_method",
        "timezone_utc_offset",
    ):
        excluded[column] = rejected[column]
    return excluded


def find_dashboard_excluded_stations(
    evaluated: gpd.GeoDataFrame,
    excluded: gpd.GeoDataFrame,
    snapped: gpd.GeoDataFrame,
    snapped_path: Path,
    minimum_upstream_area_km2: float,
) -> gpd.GeoDataFrame:
    """Identify stations omitted by snapping, reporting, or evaluation filters.

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
    inventory: gpd.GeoDataFrame = gpd.read_parquet(inventory_path)
    saved_reasons: pd.Series = inventory["evaluation_exclusion_reason"].fillna("")
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
    reasons.loc[~np.isfinite(area) | (area <= 0)] = "Missing station upstream area."
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
