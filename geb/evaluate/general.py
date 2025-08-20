#!/usr/bin/env python
"""Generates a dashboard.

Creates an interactive Plotly dashboard with a Scattermapbox layer of sample
locations for several environmental variables over an animated background
implemented by swapping pre-generated image frames (data URIs) in a Mapbox
image layer. All data and animation frames are preloaded at import so the
Dash callback does no heavy work or external requests (aside from client
Mapbox tile fetching).

The animated image layer is now georeferenced via ANIM_EXTENT (Zuid-Limburg)
while the map displays the full Netherlands extent via MAP_EXTENT.
Both are configurable module-level dictionaries.
"""

import base64
import os
import random
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Sequence

import geopandas as gpd
import hydromt_sfincs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rioxarray  # noqa: F401
import xarray as xr
from dash import Dash, Input, Output, State, dcc, html
from dotenv import load_dotenv
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon, box

load_dotenv()

random.seed(42)
sites: list[dict[str, str | float]] = [
    {"site_id": "A01", "name": "Alpha", "lat": 52.37, "lon": 4.90},
    {"site_id": "B02", "name": "Bravo", "lat": 51.92, "lon": 4.48},
    {"site_id": "C03", "name": "Charlie", "lat": 50.85, "lon": 4.35},
    {"site_id": "D04", "name": "Delta", "lat": 53.22, "lon": 6.56},
]
variables: list[str] = ["houses"]
n_days: int = 30
base_date = datetime.utcnow().date()
dates: list[datetime.date] = [base_date - timedelta(days=i) for i in range(n_days)][
    ::-1
]

records: list[dict[str, object]] = []
for variable_name in variables:
    for site in sites:
        level: float = random.uniform(10, 100)
        for current_date in dates:
            level += random.uniform(-5, 5)
            if variable_name == "nitrate":
                value = max(0.0, level / 10 + random.uniform(-0.5, 0.5))
            elif variable_name == "temperature":
                idx: int = dates.index(current_date)
                value = (
                    10 + 10 * np.sin((idx / n_days) * 2 * np.pi) + random.uniform(-1, 1)
                )
            else:
                value = max(0.0, level + random.uniform(-10, 10))
            records.append(
                {
                    "site_id": site["site_id"],
                    "site_name": site["name"],
                    "lat": site["lat"],
                    "lon": site["lon"],
                    "variable": variable_name,
                    "date": current_date,
                    "value": float(value),
                }
            )
df: pd.DataFrame = pd.DataFrame(records)
latest_date = df["date"].max()
latest: pd.DataFrame = df[df["date"] == latest_date]
default_var: str = "houses"


MODEL_FOLDER = Path("tests/tmp/model")
houses = gpd.read_parquet(
    MODEL_FOLDER / "input" / "geom" / "assets" / "buildings.geoparquet"
)
region = gpd.read_parquet(
    MODEL_FOLDER / "input" / "geom" / "mask.geoparquet"
)  # (NEW) region geometry for viewport filtering


# (NEW) Normalize region CRS to EPSG:4326 and derive outline + initial viewport
def _build_region_outline(gdf: gpd.GeoDataFrame) -> tuple[list[float], list[float]]:
    """Extract region exterior boundaries as a single polyline with None separators.

    Notes:
        - All geometries are reprojected to EPSG:4326 (degrees) if needed.
        - Only exterior rings are included (holes skipped) for a clean boundary.
        - Multiple polygons / multipolygons are concatenated with None sentinels
          so Plotly renders separated line segments.

    Args:
        gdf (gpd.GeoDataFrame): Region geometry GeoDataFrame (any polygonal CRS).

    Returns:
        tuple:
            list[float]: Longitudes (degrees) with None separators.
            list[float]: Latitudes (degrees) with None separators.

    Raises:
        ValueError: If GeoDataFrame empty or contains no polygonal geometries.
    """
    if gdf.empty:
        raise ValueError("Region GeoDataFrame is empty; cannot derive outline.")
    if gdf.crs is None:
        raise ValueError("Region GeoDataFrame has no CRS; cannot reproject.")
    if gdf.crs.to_string().lower() not in ("epsg:4326", "crs:84"):
        gdf = gdf.to_crs(epsg=4326)
    lons: list[float] = []
    lats: list[float] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polys: list[Polygon] = []
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys.extend(list(geom.geoms))
        for poly in polys:
            xs, ys = poly.exterior.xy
            # Append ring
            for x, y in zip(xs, ys):
                lons.append(float(x))
                lats.append(float(y))
            # Separator for next ring
            lons.append(None)  # type: ignore
            lats.append(None)  # type: ignore
    if not lons:
        raise ValueError("No polygonal exterior coordinates found for region outline.")
    # Drop trailing None to avoid an empty segment
    if lons[-1] is None:
        lons.pop()
        lats.pop()
    return lons, lats


# (NEW) Compute region-based initial viewport
_region_4326 = (
    region.to_crs(epsg=4326)
    if region.crs and region.crs.to_string().lower() not in ("epsg:4326", "crs:84")
    else region
)
minx, miny, maxx, maxy = _region_4326.total_bounds
REGION_EXTENT: dict[str, float] = {
    "lon_min": float(minx),
    "lon_max": float(maxx),
    "lat_min": float(miny),
    "lat_max": float(maxy),
}
REGION_OUTLINE_LON, REGION_OUTLINE_LAT = _build_region_outline(_region_4326)


def _compute_initial_zoom(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    pad_factor: float = 1.05,
) -> float:
    """Approximate a Mapbox zoom level to fit provided bounds.

    Args:
        lon_min (float): Western longitude (degrees).
        lon_max (float): Eastern longitude (degrees).
        lat_min (float): Southern latitude (degrees).
        lat_max (float): Northern latitude (degrees).
        pad_factor (float): Multiplicative padding (>1 expands view) (dimensionless).

    Returns:
        float: Approximate zoom (dimensionless).

    Raises:
        ValueError: If bounds invalid.
    """
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError("Invalid bounds for zoom computation.")
    lon_span: float = (lon_max - lon_min) * pad_factor
    if lon_span <= 0:
        return 6.0
    # degrees per 512 tile at zoom z = 360 / 2^z => z = log2(360 / span)
    zoom_est: float = np.log2(360.0 / lon_span)
    return float(max(2.0, min(12.0, zoom_est)))


_INITIAL_CENTER_LON: float = (REGION_EXTENT["lon_min"] + REGION_EXTENT["lon_max"]) / 2
_INITIAL_CENTER_LAT: float = (REGION_EXTENT["lat_min"] + REGION_EXTENT["lat_max"]) / 2
_INITIAL_ZOOM: float = _compute_initial_zoom(**REGION_EXTENT)
# ------------------------------------------------------------------
# Map / animation configuration (all preloaded)  (REPLACED BBOX_* constants)
# ------------------------------------------------------------------
# Overall map extent (approx Netherlands)
MAP_EXTENT: dict[str, float] = {
    "lon_min": 3.0,
    "lon_max": 7.5,
    "lat_min": 50.5,
    "lat_max": 53.7,
}
# Animation (image layer) georeferenced extent (Zuid-Limburg focus)
ANIM_EXTENT: dict[str, float] = {
    "lon_min": 5.55,
    "lon_max": 6.10,
    "lat_min": 50.70,
    "lat_max": 51.05,
}
ANIM_N_FRAMES: int = 20
ANIM_INTERVAL_MS: int = 400
ANIM_FRAME_WIDTH_PX: int = 600
ANIM_FRAME_HEIGHT_PX: int = 400
ANIM_PARTICLES: int = 28
ANIM_SEED: int = 123
# NEW: Graph interaction configuration enabling scroll zoom + persistent modebar
GRAPH_CONFIG: dict[str, object] = {
    "scrollZoom": True,
    "doubleClick": "reset",
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["zoomInMapbox", "zoomOutMapbox", "resetViewMapbox"],
}

# Early fetch & validate Mapbox token (fail fast).
MAPBOX_TOKEN: str | None = os.getenv("MAPBOX_TOKEN")
if not MAPBOX_TOKEN:
    raise ValueError(
        "MAPBOX_TOKEN environment variable not set; required for Mapbox rendering."
    )


def _validate_extent(ext: dict[str, float], name: str) -> None:
    """Validate an extent dictionary holding lon/lat bounds (degrees)."""
    required = {"lon_min", "lon_max", "lat_min", "lat_max"}
    if set(ext.keys()) != required:
        raise ValueError(f"{name} must have keys {required}, got {ext.keys()}.")
    if not (ext["lon_min"] < ext["lon_max"] and ext["lat_min"] < ext["lat_max"]):
        raise ValueError(f"{name} bounds invalid (min must be < max): {ext}.")


_validate_extent(MAP_EXTENT, "MAP_EXTENT")
_validate_extent(ANIM_EXTENT, "ANIM_EXTENT")

# Assets directory used for Dash static files. Use working-dir assets so Dash will serve files at /assets.
ASSETS_DIR: Path = Path.cwd() / "assets"


# ------------------------------------------------------------------
# Flood raster loading (replaces synthetic animation frame generation)
# ------------------------------------------------------------------
# (Wrap existing loader usage in conditional; keep function for potential later use.)
def _load_flood_raster_frames(
    simulation_root: Path,
    target_crs: str = "EPSG:3857",
    cmap: str | None = None,
    assets_root: Path | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Load flood raster NetCDF and save PNG/WebP frames to Dash assets, returning URLs.

    Notes:
        - Frames are written to <assets_root>/frames/frame_XXXX.(png|webp).
        - Returning URLs reduces JSON payload size; the browser can cache/stream images.
        - If assets_root is None it defaults to the module-level ASSETS_DIR so Dash (which is
          configured to serve ASSETS_DIR) can find the files.

    Args:
        simulation_root (Path): Path to SFINCS simulation run directory.
        target_crs (str): Desired CRS for Mapbox (default EPSG:3857).
        cmap (str | None): Matplotlib-style colormap name (unused currently).
        assets_root (Path | None): Root assets folder where 'frames/' will be created.
                                   If None, defaults to ASSETS_DIR.

    Returns:
        tuple:
            list[str]: Relative URLs to saved image files (one per time slice).
            dict[str,float]: Geographic extent (lon_min, lon_max, lat_min, lat_max) in degrees.

    Raises:
        FileNotFoundError: If simulation files missing.
        ValueError: If no suitable 2D variable is found or data is empty/invalid.
    """
    # Use the shared ASSETS_DIR so Dash serves the written frames at /assets/...
    if assets_root is None:
        assets_root = ASSETS_DIR
    frames_dir = assets_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    mod: hydromt_sfincs.SfincsModel = hydromt_sfincs.SfincsModel(
        simulation_root,
        mode="r",
    )
    flood_depth_per_timestep = mod.results["zsmax"]
    elevation = mod.grid["dep"]

    def _simple_rgba(norm_val: float) -> tuple[int, int, int]:
        v = max(0.0, min(1.0, norm_val))
        if v < 0.33:
            t = v / 0.33
            return (0, int(255 * t), 255)
        if v < 0.66:
            t = (v - 0.33) / 0.33
            return (int(255 * t), 255, int(255 * (1 - t)))
        t = (v - 0.66) / 0.34
        return (255, int(255 * (1 - t)), 0)

    frames_urls: list[str] = []
    time_slices = flood_depth_per_timestep.timemax.values

    vmin = 0
    vmax = flood_depth_per_timestep.max().compute().item()

    # Convert each slice to RGBA and save to disk. Limit to first 10 for safety (as before).
    saved_extent = None
    for idx, time in enumerate(time_slices):
        flood_depth = flood_depth_per_timestep.sel(timemax=time)

        downscaled_flood_map = hydromt_sfincs.utils.downscale_floodmap(
            zsmax=flood_depth,
            dep=elevation,
            hmin=0.05,
            reproj_method="bilinear",
        ).rio.reproject(target_crs)

        data = downscaled_flood_map.values.astype("float32")
        norm = (data - vmin) / (vmax - vmin)
        h, w = norm.shape[-2], norm.shape[-1]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask = np.isfinite(norm)
        valid_vals = norm[mask]

        if valid_vals.size:
            flat_norm = valid_vals
            r_arr = np.empty_like(flat_norm)
            g_arr = np.empty_like(flat_norm)
            b_arr = np.empty_like(flat_norm)
            # Map colors for valid pixels
            for i, nv in enumerate(flat_norm):
                r, g, b = _simple_rgba(float(nv))
                r_arr[i] = r
                g_arr[i] = g
                b_arr[i] = b
            rgba[..., 0][mask] = r_arr
            rgba[..., 1][mask] = g_arr
            rgba[..., 2][mask] = b_arr
            rgba[..., 3][mask] = 200  # semi-opaque

        img = Image.fromarray(rgba, mode="RGBA")

        # Try WebP for smaller files if supported, fallback to PNG.
        use_webp = True
        filename = f"frame_{idx:04d}.webp"
        file_path = frames_dir / filename
        try:
            if use_webp:
                img.save(file_path, format="WEBP", quality=80)
            else:
                raise OSError("force PNG path")
        except Exception:
            filename = f"frame_{idx:04d}.png"
            file_path = frames_dir / filename
            img.save(file_path, format="PNG")

        # Dash serves assets at /assets/..., so construct that relative URL.
        frames_urls.append(f"/assets/frames/{filename}")

        # Record extent using the last processed downscaled map (reproject to EPSG:4326).
        saved_extent = downscaled_flood_map.rio.reproject("EPSG:4326").rio.bounds()

    if saved_extent is None:
        raise ValueError("No frames were produced for flood animation.")

    minx, miny, maxx, maxy = saved_extent
    extent = {
        "lon_min": float(minx),
        "lon_max": float(maxx),
        "lat_min": float(miny),
        "lat_max": float(maxy),
    }

    return frames_urls, extent


# Use the same ASSETS_DIR when writing frames so Dash can serve them from the configured assets folder.
_ANIM_FRAMES, _FLOOD_MAP_EXTENT = _load_flood_raster_frames(
    MODEL_FOLDER
    / "simulation_root"
    / "default"
    / "SFINCS"
    / "run"
    / "simulations"
    / "20210712T090000 - 20210720T090000",
    assets_root=ASSETS_DIR,
)
_ANIM_FRAME_COUNT: int = len(_ANIM_FRAMES)
if _ANIM_FRAME_COUNT == 0:
    raise ValueError(
        "No animation frames available (raster disabled & placeholder failed)."
    )

HOUSE_RANDOM_SEED: int = 999  # retained (unused when no sampling)
HOUSE_VIEW_MARGIN_FACTOR: float = 0.12
HOUSE_COLOR_SAT: float = 0.65
HOUSE_COLOR_VAL: float = 0.85
HOUSE_COLOR_TIME_FACTOR: float = 0.55
HOUSE_VIEW_FILTER: bool = True  # (NEW) enable viewport-based filtering
HOUSE_MAX_VIEW: int | None = None  # (NEW) optional cap per frame (None = no cap)
HOUSE_TILE_SIZE_DEG: float = (
    0.01  # (NEW) tile size in degrees for viewport pre-filter (lon/lat)
)
HOUSE_COORD_DECIMALS: int = 5  # (NEW) lon/lat decimal places retained (degrees)


def _extract_polygons_from_gdf(
    gdf: gpd.GeoDataFrame,
    extent: dict[str, float],
    max_features: int | None = None,
) -> list[dict[str, object]]:
    """Extract exterior polygon rings inside an extent with bbox + hue seed.

    Notes:
        - Holes ignored (fills remain solid for speed).
        - MultiPolygons exploded to individual polygons.
        - Adds precomputed bounding box (lon/lat degrees) & a stable hue_seed.

    Args:
        gdf (gpd.GeoDataFrame): Source GeoDataFrame (any CRS).
        extent (dict[str, float]): Geographic bounds (degrees).
        max_features (int | None): Optional cap; random deterministic sample if exceeded.

    Returns:
        list[dict[str, object]]: Each dict has keys: lon, lat, bbox, hue_seed.

    Raises:
        ValueError: On missing CRS, geometry column, or no polygons in extent.
    """
    if gdf.empty:
        raise ValueError("House GeoDataFrame is empty.")
    if "geometry" not in gdf.columns:
        raise ValueError("GeoDataFrame missing 'geometry' column.")
    if gdf.geometry.is_empty.all():
        raise ValueError("All geometries in houses GeoDataFrame are empty.")
    if gdf.crs is None:
        raise ValueError("House GeoDataFrame has no CRS; cannot reproject.")

    if gdf.crs.to_string().lower() not in ("epsg:4326", "crs:84"):
        gdf = gdf.to_crs(epsg=4326)

    extent_geom = box(
        extent["lon_min"], extent["lat_min"], extent["lon_max"], extent["lat_max"]
    )
    gdf_clip = gdf[gdf.intersects(extent_geom)].copy()
    if gdf_clip.empty:
        raise ValueError("No house geometries intersect animation extent.")

    polygons: list[Polygon] = []
    for geom in gdf_clip.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(list(geom.geoms))
    if not polygons:
        raise ValueError("No Polygon / MultiPolygon geometries found after filtering.")

    # Optional downsampling if too many polygons (keeps determinism).
    if max_features is not None and len(polygons) > max_features:
        random.seed(HOUSE_RANDOM_SEED)
        polygons = random.sample(polygons, max_features)

    extracted: list[dict[str, object]] = []
    for idx, poly in enumerate(polygons):
        xs, ys = poly.exterior.xy
        lon_seq: list[float] = list(xs)
        lat_seq: list[float] = list(ys)
        # Ensure closure (defensive)
        if lon_seq[0] != lon_seq[-1] or lat_seq[0] != lat_seq[-1]:
            lon_seq.append(lon_seq[0])
            lat_seq.append(lat_seq[0])
        min_lon = min(lon_seq)
        max_lon = max(lon_seq)
        min_lat = min(lat_seq)
        max_lat = max(lat_seq)
        # Simple stable hue seed (0..1) based on index; could also hash attributes if needed.
        hue_seed: float = (idx * 0.61803398875) % 1.0
        extracted.append(
            {
                "lon": lon_seq,
                "lat": lat_seq,
                "bbox": (min_lon, max_lon, min_lat, max_lat),
                "hue_seed": hue_seed,
            }
        )
    return extracted


def _quantize_polygons(
    polygons: list[dict[str, object]], decimals: int
) -> list[dict[str, object]]:
    """Quantize (round) polygon exterior coordinates to reduce GeoJSON size.

    Rounds longitude / latitude (degrees) to a fixed number of decimal places to
    shrink the serialized GeoJSON payload with negligible visual impact.

    Notes:
        - Only exterior rings are present (holes were ignored upstream).
        - Input list is not mutated; a new list of dicts is returned.
        - Precision of 5 decimals ~ 1.1 m at equator; adjust via HOUSE_COORD_DECIMALS.

    Args:
        polygons (list[dict[str, object]]): Polygon dicts each containing:
            'lon' (list[float]): Longitudes (degrees).
            'lat' (list[float]): Latitudes (degrees).
            'bbox' (tuple[float,float,float,float]): (min_lon,max_lon,min_lat,max_lat) (degrees).
            'hue_seed' (float): Stable color seed in [0,1).
        decimals (int): Number of decimal places to retain (>=0) (dimensionless).

    Returns:
        list[dict, object]: New polygon dicts with rounded lon/lat (degrees).

    Raises:
        ValueError: If decimals < 0.
    """
    if decimals < 0:
        raise ValueError("decimals must be >= 0 for coordinate quantization.")
    quantized: list[dict[str, object]] = []
    for poly in polygons:
        lon_list: list[float] = poly["lon"]  # type: ignore
        lat_list: list[float] = poly["lat"]  # type: ignore
        q_lon: list[float] = [round(v, decimals) for v in lon_list]
        q_lat: list[float] = [round(v, decimals) for v in lat_list]
        quantized.append(
            {
                "lon": q_lon,
                "lat": q_lat,
                "bbox": poly["bbox"],
                "hue_seed": poly["hue_seed"],
            }
        )
    return quantized


# Build full polygon list once (no downsampling; max_features=None)
_HOUSE_POLYGONS: list[dict[str, object]] = _extract_polygons_from_gdf(
    houses,
    ANIM_EXTENT,
)
_HOUSE_POLYGONS = _quantize_polygons(_HOUSE_POLYGONS, HOUSE_COORD_DECIMALS)
N_HOUSES: int = len(_HOUSE_POLYGONS)


def _build_houses_geojson_with_ids(
    polygons: list[dict, object],
) -> tuple[dict, list[int], list[float], list[list[float]]]:
    """Assemble GeoJSON + parallel arrays (ids, hue seeds, bboxes).

    Adds bbox to each feature's properties to allow clientside viewport filtering.

    Args:
        polygons (list[dict[str, object]]): Polygon definitions with lon/lat & hue_seed.

    Returns:
        tuple:
            dict: FeatureCollection GeoJSON.
            list[int]: Feature ids (sequential).
            list[float]: Hue seeds (0-1).
            list[list[float]]: BBoxes [minLon, maxLon, minLat, maxLat] per feature.

    Raises:
        ValueError: If polygons empty or malformed.
    """
    if not polygons:
        raise ValueError("No polygons available to build houses GeoJSON.")
    features: list[dict] = []
    ids: list[int] = []
    hue_seeds: list[float] = []
    bboxes: list[list[float]] = []
    for idx, poly in enumerate(polygons):
        lon: list[float] = poly["lon"]  # type: ignore
        lat: list[float] = poly["lat"]  # type: ignore
        if not lon or not lat or len(lon) != len(lat):
            raise ValueError("Polygon coordinate arrays are empty or mismatched.")
        ring = list(zip(lon, lat))
        min_lon = min(lon)
        max_lon = max(lon)
        min_lat = min(lat)
        max_lat = max(lat)
        feature = {
            "type": "Feature",
            "id": idx,
            "properties": {"_hid": idx, "bbox": [min_lon, max_lon, min_lat, max_lat]},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }
        features.append(feature)
        ids.append(idx)
        hue_seeds.append(float(poly["hue_seed"]))  # type: ignore
        bboxes.append([min_lon, max_lon, min_lat, max_lat])
    return {"type": "FeatureCollection", "features": features}, ids, hue_seeds, bboxes


# Build single-trace resources (used in dcc.Store)
houses_geojson, house_ids, house_hue_seeds, house_bboxes = (
    _build_houses_geojson_with_ids(_HOUSE_POLYGONS)
)


# (NEW) Tile index construction ------------------------------------------------
def _build_house_tiles(
    bboxes: list[list[float]],
    extent: dict[str, float],
    tile_size_deg: float,
) -> tuple[dict, dict[str, list[int]]]:
    """Build a simple regular degree-based tile index for house bboxes.

    Tiles cover the animation extent with fixed degree size to reduce per-frame
    filtering cost: only tiles intersecting the viewport are inspected.

    Args:
        bboxes (list[list[float]]): Per-house bbox [minLon,maxLon,minLat,maxLat].
        extent (dict[str,float]): Animation extent with lon_min,lon_max,lat_min,lat_max.
        tile_size_deg (float): Tile size (degrees) for both lon and lat (square tiles).

    Returns:
        tuple:
            dict: Tile metadata {lonMin,latMin,tileSizeLon,tileSizeLat,nLon,nLat}.
            dict[str,list[int]]: Mapping tileKey "i_j" to list of house indices.

    Raises:
        ValueError: If tile_size_deg invalid or inputs empty.
    """
    if not bboxes:
        raise ValueError("No bboxes provided for tile index.")
    if tile_size_deg <= 0:
        raise ValueError("tile_size_deg must be > 0.")
    lon_min = extent["lon_min"]
    lon_max = extent["lon_max"]
    lat_min = extent["lat_min"]
    lat_max = extent["lat_max"]
    n_lon = max(1, int((lon_max - lon_min) / tile_size_deg) + 1)
    n_lat = max(1, int((lat_max - lat_min) / tile_size_deg) + 1)
    tile_buckets: dict[str, list[int]] = {}
    for idx, (b_min_lon, b_max_lon, b_min_lat, b_max_lat) in enumerate(bboxes):
        # Determine tile index range covering this bbox
        i0 = max(0, int((b_min_lon - lon_min) / tile_size_deg))
        i1 = min(n_lon - 1, int((b_max_lon - lon_min) / tile_size_deg))
        j0 = max(0, int((b_min_lat - lat_min) / tile_size_deg))
        j1 = min(n_lat - 1, int((b_max_lat - lat_min) / tile_size_deg))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                key = f"{i}_{j}"
                bucket = tile_buckets.get(key)
                if bucket is None:
                    tile_buckets[key] = [idx]
                else:
                    bucket.append(idx)
    meta = {
        "lonMin": lon_min,
        "latMin": lat_min,
        "tileSizeLon": tile_size_deg,
        "tileSizeLat": tile_size_deg,
        "nLon": n_lon,
        "nLat": n_lat,
    }
    return meta, tile_buckets


house_tile_meta, house_tile_buckets = _build_house_tiles(
    house_bboxes, ANIM_EXTENT, HOUSE_TILE_SIZE_DEG
)


# ------------------------------------------------------------------
# Build a cache of complete figures (variable x frame)
# ------------------------------------------------------------------
# (REPLACED) Use region-derived center/zoom
# _CENTER_LON / _CENTER_LAT no longer used below.
_FIGURE_CACHE: Dict[str, List[dict]] = {v: [] for v in variables}
for v in variables:
    for f_idx in range(_ANIM_FRAME_COUNT):
        fig = go.Figure()
        # No site/environmental traces added; houses are handled client-side as choroplethmapbox
        # (NEW) region boundary trace
        fig.add_trace(
            go.Scattermapbox(
                lon=REGION_OUTLINE_LON,
                lat=REGION_OUTLINE_LAT,
                mode="lines",
                line=dict(color="black", width=2),
                name="Region",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Make figure responsive so it fills the Graph container (which will be fullscreen).
        # Use zero margins to eliminate whitespace around the map.
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_white",
            dragmode="zoom",
            uirevision="map",
            mapbox=dict(
                accesstoken=MAPBOX_TOKEN,
                style="light",
                center=dict(lat=_INITIAL_CENTER_LAT, lon=_INITIAL_CENTER_LON),
                zoom=_INITIAL_ZOOM,
                layers=(
                    [
                        dict(
                            below="",
                            source=_ANIM_FRAMES[f_idx],
                            sourcetype="image",
                            coordinates=[
                                [ANIM_EXTENT["lon_min"], ANIM_EXTENT["lat_max"]],
                                [ANIM_EXTENT["lon_max"], ANIM_EXTENT["lat_max"]],
                                [ANIM_EXTENT["lon_max"], ANIM_EXTENT["lat_min"]],
                                [ANIM_EXTENT["lon_min"], ANIM_EXTENT["lat_min"]],
                            ],
                        )
                    ]
                ),
            ),
            # Remove footer annotation so it doesn't consume layout space.
            annotations=[],
        )
        _FIGURE_CACHE[v].append(fig.to_dict())


# Build lightweight base figures (first frame only) for clientside animation.
_BASE_FIGS: Dict[str, dict] = {v: _FIGURE_CACHE[v][0] for v in variables}


# ------------------------------------------------------------------
# Public assembly function (returns cloned cached figure)
# ------------------------------------------------------------------
def assemble_figure(variable: str, frame_index: int | None = None) -> go.Figure:
    """Return a cached figure for the given variable/frame without recomputation."""
    if variable not in _FIGURE_CACHE:
        raise ValueError(f"Variable '{variable}' not in cache.")
    if frame_index is None:
        frame_index = 0
    if frame_index < 0 or frame_index >= _ANIM_FRAME_COUNT:
        raise ValueError(
            f"frame_index {frame_index} out of range 0..{_ANIM_FRAME_COUNT - 1}"
        )
    return go.Figure(_FIGURE_CACHE[variable][frame_index])


def build_dashboard_figure(
    selected_var: str,
    allowed_vars: Sequence[str] | None = None,
    n_time_steps: int = 0,
) -> go.Figure:
    """Backward-compatible builder (ignores n_time_steps, uses frame 0)."""
    vars_list: list[str] = (
        list(allowed_vars) if allowed_vars is not None else list(variables)
    )
    if not vars_list:
        raise ValueError("allowed_vars is empty; at least one variable is required.")
    if selected_var not in vars_list:
        raise ValueError(f"Selected variable '{selected_var}' not in {vars_list}")
    return assemble_figure(selected_var, frame_index=0)


# ------------------------------------------------------------------
# Dash app (callback only swaps cached figure; no processing)
# ------------------------------------------------------------------
def create_dash_app() -> Dash:
    """Create Dash app with clientside color animation for a single houses trace.

    Notes:
        - Houses stored once as a Choroplethmapbox trace (GeoJSON).
        - Background animation swaps Mapbox image layer source.
        - Per-frame house colors computed clientside from hue seeds (no large caches).
    """
    # Ensure Dash is configured to serve the same assets directory where frames were written.
    app = Dash(__name__, assets_folder=str(ASSETS_DIR))
    # Fullscreen root container; remove maxWidth wrapper so map can use entire viewport.
    app.layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "height": "100vh",
            "width": "100vw",
            "margin": "0",
            "padding": "0",
            "overflow": "hidden",  # avoid scrollbars from children
        },
        children=[
            # Header removed so the map occupies the entire screen.
            dcc.Graph(
                id="dashboard-graph",
                figure=assemble_figure(default_var, frame_index=0),
                style={
                    "height": "100vh",
                    "width": "100vw",
                    "margin": "0",
                    "padding": "0",
                },
                config=GRAPH_CONFIG,
            ),
            dcc.Interval(id="anim-interval", interval=ANIM_INTERVAL_MS, n_intervals=0),
            dcc.Store(
                id="dash-anim-data",
                data={
                    "baseFigures": _BASE_FIGS,
                    "frames": _ANIM_FRAMES,
                    "houseSeeds": house_hue_seeds,
                    "houseIds": house_ids,
                    "housesGeoJSON": houses_geojson,
                    "housesBBoxes": house_bboxes,
                    "colorTimeFactor": HOUSE_COLOR_TIME_FACTOR,
                    "viewFilter": HOUSE_VIEW_FILTER,
                    "maxView": HOUSE_MAX_VIEW,
                    "viewMarginFactor": HOUSE_VIEW_MARGIN_FACTOR,
                    "tileMeta": house_tile_meta,
                    "tileBuckets": house_tile_buckets,
                    "floodExtent": _FLOOD_MAP_EXTENT,
                    "rasterEnabled": True,
                },
            ),
        ],
    )

    # Clientside callback: swaps background frame + updates houses z colors.
    client_js = f"""
    function(n_intervals, animData, currentFig){{
        // Single-mode app: variable fixed to default_var
        const variable = "{default_var}";
         const tStart = performance.now();
         let tPrev = tStart;
         function logStep(label){{
             const now = performance.now();
             console.log('[anim]', label, (now - tPrev).toFixed(1) + 'ms', 'total', (now - tStart).toFixed(1) + 'ms');
             tPrev = now;
         }}

        // Ensure a one–time Mapbox watcher that records the true map bounds after any move/zoom.
        function ensureBoundsWatcher(){{
            const gd = document.getElementById('dashboard-graph');
            if(!gd || gd._boundsWatcherAdded) return;
            const pollLimit = 40;
            let attempts = 0;
            function tryAttach(){{
                attempts++;
                // Plotly stores the Mapbox map instance internally.
                const mb = gd._fullLayout && gd._fullLayout.mapbox && gd._fullLayout.mapbox._subplot && gd._fullLayout.mapbox._subplot.map;
                if(mb){{
                    const update = () => {{
                        try {{
                            const b = mb.getBounds();
                            const c = mb.getCenter();
                            window._mapboxViewport = {{
                                lonMin: b.getWest(),
                                lonMax: b.getEast(),
                                latMin: b.getSouth(),
                                latMax: b.getNorth(),
                                zoom: mb.getZoom(),
                                centerLon: c.lng,
                                centerLat: c.lat
                            }};
                        }} catch(e) {{
                            // swallow; will retry on next event
                        }}
                    }};
                    // Update aggressively on move for responsiveness; moveend/zoomend finalize.
                    mb.on('move', update);
                    mb.on('moveend', update);
                    mb.on('zoomend', update);
                    update();
                    gd._boundsWatcherAdded = true;
                    console.log('[anim] bounds watcher attached');
                }} else if(attempts < pollLimit) {{
                    setTimeout(tryAttach, 100);
                }}
            }}
            tryAttach();
        }}
        ensureBoundsWatcher();

        if(!animData) return window.dash_clientside.no_update;
        const rasterEnabled = !!animData.rasterEnabled;

        // frames still needed for phase timing; ensure non-empty
        const frames = animData.frames;
        if(!frames || !frames.length) return window.dash_clientside.no_update;
        const base = animData.baseFigures?.[variable];
        if(!base) return window.dash_clientside.no_update;

        if(!window._houseDashCache) window._houseDashCache = {{
            lastVar: null,
            viewportKey: null,
            indices: null,
            geojson: null,
            unfiltered: false
        }};
        const cache = window._houseDashCache;

        const idx = n_intervals % frames.length;
        const phase = idx / frames.length;
        const seeds = animData.houseSeeds || [];
        const bboxes = animData.housesBBoxes || [];
        const factor = animData.colorTimeFactor || 0.55;
        const doFilter = !!animData.viewFilter;
        const maxView = animData.maxView == null ? null : animData.maxView;
        const marginFactor = animData.viewMarginFactor == null ? 0.12 : animData.viewMarginFactor;
        const tileMeta = animData.tileMeta || null;
        const tileBuckets = animData.tileBuckets || null;

        let currentVar = null;
        if(currentFig?.layout?.title?.text){{
            const m = currentFig.layout.title.text.match(/Environmental Dashboard \\(([^)]+)\\)/);
            if(m) currentVar = m[1];
        }}

        const reuse = (currentVar === variable) && !!currentFig;
        let fig = reuse ? currentFig : JSON.parse(JSON.stringify(base));
        logStep('figure prepared (reuse=' + reuse + ')');

        // Background frame (only when raster enabled)
        if(!fig.layout) fig.layout = {{}};
        if(!fig.layout.mapbox) fig.layout.mapbox = {{}};
        if(!fig.layout.mapbox.layers) fig.layout.mapbox.layers = [];

        if(rasterEnabled){{
            const floodExt = animData.floodExtent;
            if(!floodExt) return window.dash_clientside.no_update;
            const coordinates = [
                [floodExt.lon_min, floodExt.lat_max],
                [floodExt.lon_max, floodExt.lat_max],
                [floodExt.lon_max, floodExt.lat_min],
                [floodExt.lon_min, floodExt.lat_min]
            ];
            layerConfig = {{
                sourcetype: "image",
                type: "raster",
                source: frames[idx],
                coordinates: coordinates,
                paint: {{
                    "raster-resampling": "nearest"
                }},
            }}
            
            if(!fig.layout.mapbox.layers.length){{
                fig.layout.mapbox.layers.push(layerConfig);
            }} else {{
                fig.layout.mapbox.layers[0] = layerConfig;
            }}
        }} else {{
            // Remove any existing raster layer to avoid stale imagery
            if(fig.layout.mapbox.layers.length) {{
                fig.layout.mapbox.layers = [];
            }}
        }}
        logStep('background frame updated');

        const zFull = seeds.map(s => (s + phase * factor) % 1.0);
        logStep('full color array computed');

        let centerLon, centerLat, zoom;
        let viewLonMin, viewLonMax, viewLatMin, viewLatMax;
        const stored = window._mapboxViewport;
        if(stored && Number.isFinite(stored.lonMin) && Number.isFinite(stored.lonMax)){{
            centerLon = stored.centerLon;
            centerLat = stored.centerLat;
            zoom = stored.zoom;
            viewLonMin = stored.lonMin;
            viewLonMax = stored.lonMax;
            viewLatMin = stored.latMin;
            viewLatMax = stored.latMax;

            // Expand by marginFactor proportionally to current span.
            const lonSpanBase = viewLonMax - viewLonMin;
            const latSpanBase = viewLatMax - viewLatMin;
            const expandLon = lonSpanBase * marginFactor;
            const expandLat = latSpanBase * marginFactor;
            viewLonMin -= expandLon;
            viewLonMax += expandLon;
            viewLatMin -= expandLat;
            viewLatMax += expandLat;
        }} else {{
            // Fallback heuristic (center + zoom -> approximate square)
            centerLon = fig.layout.mapbox.center?.lon;
            centerLat = fig.layout.mapbox.center?.lat;
            zoom = fig.layout.mapbox.zoom;
            if(centerLon == null || centerLat == null || zoom == null){{
                centerLon = {(MAP_EXTENT["lon_min"] + MAP_EXTENT["lon_max"]) / 2};
                centerLat = {(MAP_EXTENT["lat_min"] + MAP_EXTENT["lat_max"]) / 2};
                zoom = 6;
            }}
            const lonSpanRaw = 360 / Math.pow(2, zoom);
            let lonSpan = lonSpanRaw * (1 + marginFactor) * 1.05;
            let latSpan = lonSpan;
            viewLonMin = centerLon - lonSpan/2;
            viewLonMax = centerLon + lonSpan/2;
            viewLatMin = centerLat - latSpan/2;
            viewLatMax = centerLat + latSpan/2;
        }}

        // Tile-size padding so edge-aligned features are not missed
        if(animData.tileMeta){{
            const padDeg = Math.max(animData.tileMeta.tileSizeLon, animData.tileMeta.tileSizeLat) * 0.75;
            viewLonMin -= padDeg;
            viewLonMax += padDeg;
            viewLatMin -= padDeg;
            viewLatMax += padDeg;
        }} else {{
            const padDeg = 0.01;
            viewLonMin -= padDeg;
            viewLonMax += padDeg;
            viewLatMin -= padDeg;
            viewLatMax += padDeg;
        }}
        logStep('viewport resolved (precise=' + (!!stored) + ')');

        if(cache.lastVar !== variable){{
            cache.viewportKey = null;
            cache.indices = null;
            cache.geojson = null;
            cache.unfiltered = false;
            cache.lastVar = variable;
        }}

        let indices;
        let rebuilt = false;
        if(doFilter && tileMeta && tileBuckets){{
            // Use actual bounds in key (quantize to reduce churn)
            const q = (v)=> v.toFixed(4);
            const viewportKey = [q(viewLonMin), q(viewLonMax), q(viewLatMin), q(viewLatMax)].join('|');
            if(cache.viewportKey !== viewportKey || !cache.indices){{
                const tLonSize = tileMeta.tileSizeLon;
                const tLatSize = tileMeta.tileSizeLat;
                const baseLon = tileMeta.lonMin;
                const baseLat = tileMeta.latMin;
                const nLon = tileMeta.nLon;
                const nLat = tileMeta.nLat;
                const eps = 1e-9;
                const iMin = Math.max(0, Math.floor((viewLonMin - baseLon - eps)/tLonSize));
                const iMax = Math.min(nLon-1, Math.floor((viewLonMax - baseLon + eps)/tLonSize));
                const jMin = Math.max(0, Math.floor((viewLatMin - baseLat - eps)/tLatSize));
                const jMax = Math.min(nLat-1, Math.floor((viewLatMax - baseLat + eps)/tLatSize));
                const seen = new Set();
                for(let i=iMin;i<=iMax;i++) for(let j=jMin;j<=jMax;j++) {{
                    const bucket = tileBuckets[i + "_" + j];
                    if(!bucket) continue;
                    for(const h of bucket) seen.add(h);
                }}
                const next = [];
                for(const h of seen){{
                    const bb = bboxes[h];
                    if(!bb) continue;
                    if(!(bb[1] < viewLonMin || bb[0] > viewLonMax || bb[3] < viewLatMin || bb[2] > viewLatMax)){{
                        next.push(h);
                    }}
                }}
                if(maxView !== null && next.length > maxView) next.length = maxView;
                indices = next.sort((a,b)=>a-b);
                const fullFeatures = animData.housesGeoJSON?.features || [];
                const features = indices.map(i => fullFeatures[i]).filter(Boolean);
                cache.geojson = {{type: 'FeatureCollection', features}};
                cache.indices = indices;
                cache.viewportKey = viewportKey;
                cache.unfiltered = false;
                rebuilt = true;
                logStep('filtering + geojson rebuild');
            }} else {{
                indices = cache.indices;
                logStep('filtering skipped (viewport key match)');
            }}
        }} else {{
            if(!cache.unfiltered || !cache.geojson){{
                const fullFeatures = animData.housesGeoJSON?.features || [];
                cache.geojson = {{type: 'FeatureCollection', features: fullFeatures}};
                cache.indices = seeds.map((_,i)=>i);
                cache.unfiltered = true;
                rebuilt = true;
                logStep('unfiltered geojson build');
            }} else {{
                logStep('unfiltered geojson reuse');
            }}
            indices = cache.indices;
        }}

        const zSubset = indices.map(i => zFull[i]);  // (same logic both paths)
        if(rebuilt) logStep('z subset (after rebuild)'); else logStep('z subset (reuse path)');

        let houseTrace = null;
        for(const t of (fig.data || [])){{
            if(t.type === 'choroplethmapbox' && t.name === 'Houses'){{ houseTrace = t; break; }}
        }}
        if(!houseTrace){{
            if(!fig.data) fig.data = [];
            fig.data.push({{
                type: 'choroplethmapbox',
                name: 'Houses',
                geojson: cache.geojson,
                locations: indices,
                featureidkey: 'properties._hid',
                z: zSubset,
                zmin: 0.0,
                zmax: 1.0,
                colorscale: 'Turbo',
                showscale: false,
                marker: {{line: {{width: 0.2, color: 'black'}}}},
                hoverinfo: 'skip',
                showlegend: false
            }});
            logStep('created house trace');
        }} else {{
            if(rebuilt){{
                houseTrace.geojson = cache.geojson;
                houseTrace.locations = indices;
            }}
            houseTrace.z = zSubset;
            logStep(rebuilt ? 'updated house trace (geojson+z)' : 'updated house trace (z only)');
        }}

        logStep('callback complete');
        return reuse ? {{...fig}} : fig;
    }}
    """
    app.clientside_callback(
        client_js,
        Output("dashboard-graph", "figure"),
        Input("anim-interval", "n_intervals"),
        State("dash-anim-data", "data"),
        State("dashboard-graph", "figure"),
    )
    return app


# Clean single entry point (removed obsolete legacy block that followed previously).
if __name__ == "__main__":
    dash_app = create_dash_app()
    dash_app.run(debug=True, host="0.0.0.0", port=8050)
