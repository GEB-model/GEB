#!/usr/bin/env python
"""generate_dashboard.py

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
from typing import Dict, List, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon, box  # (NEW) geometry helpers

load_dotenv()

# ------------------------------------------------------------------
# Data generation (example only) – precomputed once
# ------------------------------------------------------------------
random.seed(42)
sites: list[dict[str, str | float]] = [
    {"site_id": "A01", "name": "Alpha", "lat": 52.37, "lon": 4.90},
    {"site_id": "B02", "name": "Bravo", "lat": 51.92, "lon": 4.48},
    {"site_id": "C03", "name": "Charlie", "lat": 50.85, "lon": 4.35},
    {"site_id": "D04", "name": "Delta", "lat": 53.22, "lon": 6.56},
]
variables: list[str] = ["flow", "nitrate", "temperature"]
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
default_var: str = variables[0]

from pathlib import Path

MODEL_FOLDER = Path("tests/tmp/model")
houses = gpd.read_parquet(
    MODEL_FOLDER / "input" / "geom" / "assets" / "buildings.geoparquet"
)

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


# ------------------------------------------------------------------
# Frame generation helpers (unchanged logic, executed once)
# ------------------------------------------------------------------
def _generate_animation_frames(
    n_frames: int,
    width_px: int,
    height_px: int,
    n_particles: int,
    seed: int | None = None,
) -> list[Image.Image]:
    """Generate individual RGBA frames (not a single GIF) for map animation."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1.")
    if width_px <= 0 or height_px <= 0:
        raise ValueError("width_px and height_px must be > 0.")
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1.")
    if seed is not None:
        random.seed(seed)
    particles: list[dict[str, float]] = [
        {
            "x": random.uniform(0, width_px),
            "y": random.uniform(0, height_px),
            "vx": random.uniform(-2.2, 2.2),
            "vy": random.uniform(-2.2, 2.2),
            "r": random.uniform(7, 16),
        }
        for _ in range(n_particles)
    ]
    frames: list[Image.Image] = []
    for f_idx in range(n_frames):
        phase: float = f_idx / max(1, n_frames - 1)
        img = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for p in particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            if p["x"] < 0 or p["x"] > width_px:
                p["vx"] *= -1
                p["x"] = max(0, min(width_px, p["x"]))
            if p["y"] < 0 or p["y"] > height_px:
                p["vy"] *= -1
                p["y"] = max(0, min(height_px, p["y"]))
            hue: float = (phase + (p["x"] / width_px) * 0.3) % 1.0
            r_col: int = int(120 + 110 * hue)
            g_col: int = int(60 + 150 * (1 - hue))
            b_col: int = int(200 * (0.4 + 0.6 * (1 - abs(0.5 - hue))))
            alpha: int = 135
            x0: float = p["x"] - p["r"]
            y0: float = p["y"] - p["r"]
            x1: float = p["x"] + p["r"]
            y1: float = p["y"] + p["r"]
            draw.ellipse((x0, y0, x1, y1), fill=(r_col, g_col, b_col, alpha))
        frames.append(img)
    return frames


def _encode_frames_to_data_uri(frames: list[Image.Image]) -> list[str]:
    """Convert frames to PNG data URIs for embedding as Mapbox image sources."""
    data_uris: list[str] = []
    for frame in frames:
        buffer = BytesIO()
        frame.save(buffer, format="PNG")
        encoded: str = base64.b64encode(buffer.getvalue()).decode("ascii")
        data_uris.append(f"data:image/png;base64,{encoded}")
    return data_uris


_ANIM_FRAMES: list[str] = _encode_frames_to_data_uri(
    _generate_animation_frames(
        n_frames=ANIM_N_FRAMES,
        width_px=ANIM_FRAME_WIDTH_PX,
        height_px=ANIM_FRAME_HEIGHT_PX,
        n_particles=ANIM_PARTICLES,
        seed=ANIM_SEED,
    )
)
_ANIM_FRAME_COUNT: int = len(_ANIM_FRAMES)


# ------------------------------------------------------------------
# (REVISED) Animated house polygon configuration (now NO limiting)
# ------------------------------------------------------------------
HOUSE_MAX_FEATURES: int | None = (
    None  # (CHANGED) None => do not downsample; include all houses
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


# Build full polygon list once (no downsampling; max_features=None)
_HOUSE_POLYGONS: list[dict[str, object]] = _extract_polygons_from_gdf(
    houses,
    ANIM_EXTENT,
    max_features=None,  # (CHANGED) include all available houses
)
N_HOUSES: int = len(_HOUSE_POLYGONS)


def _build_houses_geojson_with_ids(
    polygons: list[dict[str, object]],
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
# Precompute per-variable site traces (Scattermapbox) once
# ------------------------------------------------------------------
def _build_site_trace(variable: str) -> go.Scattermapbox:
    """Create a single Scattermapbox trace for the latest snapshot of a variable."""
    if variable not in variables:
        raise ValueError(f"Variable '{variable}' not in {variables}")
    subset: pd.DataFrame = latest[latest["variable"] == variable]
    if subset.empty:
        raise ValueError(f"No latest data for variable '{variable}'.")
    return go.Scattermapbox(
        lat=subset["lat"],
        lon=subset["lon"],
        text=[
            f"{row.site_name}<br>{variable}: {row.value:,.2f}"
            for row in subset.itertuples()
        ],
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=14,
            color=subset["value"],
            colorscale="Viridis",
            cmin=float(subset["value"].min()),
            cmax=float(subset["value"].max()),
            showscale=True,
        ),
        name="Sites",
        customdata=subset["site_id"],
    )


_SITE_TRACE_JSON: Dict[str, dict] = {
    v: _build_site_trace(v).to_plotly_json() for v in variables
}

# ------------------------------------------------------------------
# Build a cache of complete figures (variable x frame) (UPDATED to use MAP_EXTENT & ANIM_EXTENT)
# ------------------------------------------------------------------
_CENTER_LON: float = (MAP_EXTENT["lon_min"] + MAP_EXTENT["lon_max"]) / 2
_CENTER_LAT: float = (MAP_EXTENT["lat_min"] + MAP_EXTENT["lat_max"]) / 2

_FIGURE_CACHE: Dict[str, List[dict]] = {v: [] for v in variables}
for v in variables:
    for f_idx in range(_ANIM_FRAME_COUNT):
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(**_SITE_TRACE_JSON[v]))
        # (REMOVED) house traces here for performance; added per viewport in callback
        fig.update_layout(
            title=f"Environmental Dashboard ({v})",
            height=600,
            margin=dict(l=40, r=40, t=80, b=40),
            template="plotly_white",
            dragmode="zoom",
            uirevision="map",
            mapbox=dict(
                accesstoken=MAPBOX_TOKEN,
                style="light",
                center=dict(lat=_CENTER_LAT, lon=_CENTER_LON),
                zoom=6.0,
                layers=[
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
                ],
            ),
            annotations=[
                dict(
                    text="Background overlay + viewport-filtered animated houses.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.1,
                    font=dict(size=12, color="gray"),
                )
            ],
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
    app = Dash(__name__)
    app.layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "maxWidth": "1100px",
            "margin": "0 auto",
        },
        children=[
            html.H2("Environmental Dashboard (Animated Houses via JS)"),
            html.Div(
                style={"display": "flex", "gap": "1rem", "alignItems": "center"},
                children=[
                    html.Label("Variable:", htmlFor="variable-dropdown"),
                    dcc.Dropdown(
                        id="variable-dropdown",
                        options=[
                            {"label": v.capitalize(), "value": v} for v in variables
                        ],
                        value=default_var,
                        clearable=False,
                        style={"width": "250px"},
                    ),
                ],
            ),
            dcc.Graph(
                id="dashboard-graph",
                figure=assemble_figure(default_var, frame_index=0),
                style={"height": "650px"},
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
                    "tileMeta": house_tile_meta,  # (NEW) tile metadata
                    "tileBuckets": house_tile_buckets,  # (NEW) tile -> indices
                },
            ),
        ],
    )

    # Clientside callback: swaps background frame + updates houses z colors.
    client_js = f"""
    function(variable, n_intervals, animData, currentFig){{
        if(!animData || !variable) return window.dash_clientside.no_update;
        const frames = animData.frames;
        if(!frames || !frames.length) return window.dash_clientside.no_update;
        const base = animData.baseFigures?.[variable];
        if(!base) return window.dash_clientside.no_update;

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
        let fig = (currentVar === variable && currentFig)
            ? JSON.parse(JSON.stringify(currentFig))
            : JSON.parse(JSON.stringify(base));

        if(!fig.layout) fig.layout = {{}};
        if(!fig.layout.mapbox) fig.layout.mapbox = {{}};
        if(!fig.layout.mapbox.layers) fig.layout.mapbox.layers = [];
        if(!fig.layout.mapbox.layers.length){{
            fig.layout.mapbox.layers.push({{
                below: "",
                sourcetype: "image",
                source: frames[idx],
                coordinates: [
                    [{ANIM_EXTENT["lon_min"]}, {ANIM_EXTENT["lat_max"]}],
                    [{ANIM_EXTENT["lon_max"]}, {ANIM_EXTENT["lat_max"]}],
                    [{ANIM_EXTENT["lon_max"]}, {ANIM_EXTENT["lat_min"]}],
                    [{ANIM_EXTENT["lon_min"]}, {ANIM_EXTENT["lat_min"]}]
                ]
            }});
        }} else {{
            fig.layout.mapbox.layers[0].source = frames[idx];
        }}

        const zFull = seeds.map(s => (s + phase * factor) % 1.0);

        // Viewport approximation
        let centerLon = fig.layout.mapbox.center?.lon;
        let centerLat = fig.layout.mapbox.center?.lat;
        let zoom = fig.layout.mapbox.zoom;
        if(centerLon == null || centerLat == null || zoom == null){{
            centerLon = {(MAP_EXTENT["lon_min"] + MAP_EXTENT["lon_max"]) / 2};
            centerLat = {(MAP_EXTENT["lat_min"] + MAP_EXTENT["lat_max"]) / 2};
            zoom = 6;
        }}
        const lonSpanRaw = 360 / Math.pow(2, zoom);
        let lonSpan = lonSpanRaw * (1 + marginFactor);
        let latSpan = lonSpan;
        const viewLonMin = centerLon - lonSpan/2;
        const viewLonMax = centerLon + lonSpan/2;
        const viewLatMin = centerLat - latSpan/2;
        const viewLatMax = centerLat + latSpan/2;

        let indices = [];
        if(doFilter && tileMeta && tileBuckets){{
            // Tile-based coarse filter
            const tLonSize = tileMeta.tileSizeLon;
            const tLatSize = tileMeta.tileSizeLat;
            const baseLon = tileMeta.lonMin;
            const baseLat = tileMeta.latMin;
            const nLon = tileMeta.nLon;
            const nLat = tileMeta.nLat;

            const iMin = Math.max(0, Math.floor((viewLonMin - baseLon)/tLonSize));
            const iMax = Math.min(nLon-1, Math.floor((viewLonMax - baseLon)/tLonSize));
            const jMin = Math.max(0, Math.floor((viewLatMin - baseLat)/tLatSize));
            const jMax = Math.min(nLat-1, Math.floor((viewLatMax - baseLat)/tLatSize));

            const seen = new Set();
            for(let i=iMin; i<=iMax; i++) {{
                for(let j=jMin; j<=jMax; j++) {{
                    const key = i + "_" + j;
                    const bucket = tileBuckets[key];
                    if(!bucket) continue;
                    for(const h of bucket) seen.add(h);
                }}
            }}
            // Optional fine bbox refinement to drop houses partially outside
            for(const h of seen){{
                const bb = bboxes[h];
                if(!bb) continue;
                if(!(bb[1] < viewLonMin || bb[0] > viewLonMax || bb[3] < viewLatMin || bb[2] > viewLatMax)){{
                    indices.push(h);
                }}
            }}
        }} else {{
            indices = seeds.map((_,i)=>i);
        }}

        if(maxView !== null && indices.length > maxView){{
            indices = indices.slice(0, maxView);
        }}

        const fullFeatures = animData.housesGeoJSON?.features || [];
        const features = indices.map(i => fullFeatures[i]).filter(Boolean);
        const filteredGeoJSON = {{type: "FeatureCollection", features}};
        const locs = indices;
        const z = indices.map(i => zFull[i]);

        let houseTrace = null;
        for(const t of (fig.data || [])){{
            if(t.type === "choroplethmapbox" && t.name === "Houses"){{
                houseTrace = t; break;
            }}
        }}
        if(!houseTrace){{
            if(!fig.data) fig.data = [];
            fig.data.push({{
                type: "choroplethmapbox",
                name: "Houses",
                geojson: filteredGeoJSON,
                locations: locs,
                featureidkey: "properties._hid",
                z: z,
                zmin: 0.0,
                zmax: 1.0,
                colorscale: "Turbo",
                showscale: false,
                marker: {{line: {{width: 0.2, color: "black"}}}},
                hoverinfo: "skip",
                showlegend: false
            }});
        }} else {{
            houseTrace.geojson = filteredGeoJSON;
            houseTrace.locations = locs;
            houseTrace.z = z;
        }}

        return fig;
    }}
    """
    app.clientside_callback(
        client_js,
        Output("dashboard-graph", "figure"),
        Input("variable-dropdown", "value"),
        Input("anim-interval", "n_intervals"),
        State("dash-anim-data", "data"),
        State("dashboard-graph", "figure"),
    )
    return app


# Clean single entry point (removed obsolete legacy block that followed previously).
if __name__ == "__main__":
    dash_app = create_dash_app()
    dash_app.run(debug=False, host="0.0.0.0", port=8050)
