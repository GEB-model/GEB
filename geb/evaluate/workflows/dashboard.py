"""Build discharge dashboard maps and station chart files.

This module contains functions for the map, station charts, and map layers.
JavaScript files in assets/discharge_dashboard control the interactive features.
_JavascriptMacro adds these scripts and their data to the HTML file.
"""

import hashlib
import html
import json
import logging
import math
from pathlib import Path
from typing import Any, NamedTuple, cast

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.element import Figure
from folium import MacroElement, TileLayer
from jinja2 import Environment
from jinja2.utils import htmlsafe_json_dumps
from pandas.api.typing import NaTType

from geb.evaluate.workflows import discharge_characteristics, discharge_helpers
from geb.evaluate.workflows.discharge_characteristics import (
    DASHBOARD_CATCHMENT_CHARACTERISTICS,
    CatchmentCharacteristic,
)
from geb.evaluate.workflows.discharge_helpers import load_station_discharge_comparison
from geb.evaluate.workflows.discharge_metrics import (
    DischargeMetrics,
    use_daily_discharge_scores,
)
from geb.evaluate.workflows.discharge_station_checks import (
    GEOD,
    find_dashboard_excluded_stations,
    find_excluded_stations,
)
from geb.hydrology.routing import select_active_rivers
from geb.workflows.extreme_value_analysis import ReturnPeriodModel
from geb.workflows.io import read_geom

_ESRI_TOPO_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
)

_ESRI_TOPO_ATTR = (
    "Sources: Esri, HERE, Garmin, Intermap, INCREMENT P, GEBCO, USGS, FAO, "
    "NPS, NRCan, GeoBase, IGN, Kadaster NL, Ordnance Survey, Esri Japan, "
    "METI, Mapwithyou, NOSTRA, © OpenStreetMap contributors, and the GIS "
    "user community"
)

StationMarkerIndex = dict[str, str | list[str]]

RESERVOIR_WATERBODY_TYPE: int = 2

_WATERBODY_STYLE: dict[int, dict[str, str]] = {
    RESERVOIR_WATERBODY_TYPE: {"color": "#FF8A65", "label": "Reservoir"},
}

_CHARACTERISTIC_COLORS: list[str] = [
    "#440154",
    "#414487",
    "#2A788E",
    "#22A884",
    "#7AD151",
    "#FDE725",
]

_CARAVAN_AVAILABLE_COLOR: str = "#1B9E77"

_CARAVAN_UNAVAILABLE_COLOR: str = "#9CA3AF"

_CHARACTERISTIC_COLORMAP: cm.LinearColormap = cm.LinearColormap(
    colors=_CHARACTERISTIC_COLORS,
    vmin=0.0,
    vmax=100.0,
)


class DischargeDashboardGeometries(NamedTuple):
    """Geometries required to build a discharge evaluation dashboard."""

    region: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    waterbodies: gpd.GeoDataFrame


_METRIC_LAYER_CONFIGS: list[dict] = [
    {
        "col": "KGE",
        "name": "KGE",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": -1.0,
        "vmax": 1.0,
        "show": True,
    },
    {
        "col": "KGE_modified",
        "name": "mKGE",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": -1.0,
        "vmax": 1.0,
        "show": False,
    },
    {
        "col": "KGE_correlation",
        "name": "KGE correlation",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": 0.0,
        "vmax": 1.0,
        "show": False,
    },
    {
        "col": "KGE_bias_ratio",
        "name": "KGE bias (β)",
        "colors": ["red", "orange", "green", "orange", "red"],
        "vmin": 0.0,
        "vmax": 2.0,
        "show": False,
    },
    {
        "col": "KGE_variability_ratio",
        "name": "KGE variability (α)",
        "colors": ["red", "orange", "green", "orange", "red"],
        "vmin": 0.0,
        "vmax": 2.0,
        "show": False,
    },
    {
        "col": "NSE",
        "name": "NSE",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": -1.0,
        "vmax": 1.0,
        "show": False,
    },
    {
        "col": "R2",
        "name": "Pearson r²",
        "colors": ["red", "orange", "yellow", "blue", "green"],
        "vmin": 0.0,
        "vmax": 1.0,
        "show": False,
    },
    {
        "col": "RRMSE",
        "name": "RRMSE",
        "colors": ["green", "blue", "yellow", "orange", "red"],
        "vmin": 0.0,
        "vmax": 1.0,
        "show": False,
    },
]


class _JavascriptMacro(MacroElement):
    """Embed a packaged dashboard script and its data in the exported HTML."""

    def __init__(self, filename: str, data: Any) -> None:
        """Load a dashboard script with data available to its Jinja template.

        Args:
            filename: JavaScript filename in assets/discharge_dashboard.
            data: Script data serialized for safe embedding in Folium templates.

        """
        super().__init__()
        self.data: Any = data
        script_path: Path = (
            Path(__file__).parent / "assets" / "discharge_dashboard" / filename
        )
        # Folium renders scripts twice; escape data on the first pass so the
        # second pass cannot interpret station text as a template.
        environment: Environment = Environment()
        environment.filters["script_json"] = _script_json
        self._template = environment.from_string(
            "{%- macro script(this, kwargs) -%}\n"
            + script_path.read_text(encoding="utf-8")
            + "\n{%- endmacro -%}"
        )


def create_discharge_dashboard(
    evaluation_folder: Path,
    run_output_folder: Path,
    geometry_files: dict[str, Path],
    table_files: dict[str, Path],
    minimum_upstream_area_km2: float,
    logger: logging.Logger,
    correct_discharge_observations: bool = False,
    output_filename: str = "discharge_evaluation_map.html",
    include_return_period_plots: bool = True,
) -> dict[str, str]:
    """Create only the discharge evaluation dashboard.

    This reuses ``evaluation_metrics.geoparquet`` from a previous
    ``evaluate_discharge`` run. Data for the interactive Plotly charts are
    rebuilt from the reported discharge time series. Static station plots
    and skill-score plots are not regenerated.
    Excluded stations retain diagnostic KGE and time series with a warning.

    Args:
        evaluation_folder: Folder containing saved discharge evaluation metrics.
        run_output_folder: Output folder of the selected simulation run.
        geometry_files: Model geometry paths keyed by dataset name.
        table_files: Model observation table paths keyed by dataset name.
        minimum_upstream_area_km2: Minimum routing area for evaluation (km²).
        logger: Model logger for progress and diagnostics.
        correct_discharge_observations: Whether to correct simulated discharge
            by the observed-to-GEB upstream-area ratio (dimensionless), matching
            the option in ``evaluate_discharge``.
        output_filename: Dashboard HTML filename written inside the discharge
            evaluation output folder, or an absolute output path.
        include_return_period_plots: Whether to calculate and plot return-period
            curves in station popups. Defaults to True.

    Returns:
        Dictionary with the created dashboard path.

    Raises:
        FileNotFoundError: If saved discharge evaluation metrics do not exist.
        ValueError: If ``output_filename`` is empty or
            saved metrics use obsolete snapping.
    """
    if not output_filename:
        raise ValueError("output_filename must not be empty.")
    output_path: Path = Path(output_filename)

    metrics_path: Path = evaluation_folder / "evaluation_metrics.geoparquet"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "No discharge evaluation metrics found. Run "
            "`geb evaluate hydrology.evaluate_discharge` before creating "
            "only the dashboard."
        )

    mapped_station_scores: gpd.GeoDataFrame = gpd.read_parquet(metrics_path)
    snapped_locations: gpd.GeoDataFrame = read_geom(
        geometry_files["discharge/discharge_snapped_locations"]
    )
    # Keep stations that failed snapping or visualization on the dashboard
    # while excluding them from evaluation summaries.
    excluded_stations: gpd.GeoDataFrame = find_excluded_stations(
        snapped_locations,
        run_output_folder / "report",
    )
    if not excluded_stations.empty:
        mapped_station_scores["exclusion_reason"] = excluded_stations[
            "exclusion_reason"
        ].reindex(mapped_station_scores.index)
    diagnostic_path: Path = metrics_path.with_name("diagnostic_metrics.geoparquet")
    if diagnostic_path.exists():
        mapped_excluded_scores: gpd.GeoDataFrame = gpd.read_parquet(diagnostic_path)
        mapped_station_scores = gpd.GeoDataFrame(
            pd.concat(
                [
                    mapped_station_scores,
                    mapped_excluded_scores.loc[
                        ~mapped_excluded_scores.index.isin(mapped_station_scores.index)
                    ],
                ]
            ),
            geometry="geometry",
            crs=mapped_station_scores.crs,
        )
    # Remove this extra guard in January 2027, after old builds are updated.
    if not mapped_station_scores.empty and (
        "snapping_method" not in mapped_station_scores.columns
        or not (
            mapped_station_scores["snapping_method"] == "original_subgrid_pixel_v1"
        ).all()
    ):
        raise ValueError(
            "Saved discharge metrics predate original-subgrid-pixel snapping. Rebuild hydrography "
            "and discharge observations, rerun station discharge reporting and "
            "hydrology.evaluate_discharge before creating the dashboard."
        )
    excluded_stations = find_dashboard_excluded_stations(
        mapped_station_scores,
        excluded_stations,
        snapped_locations,
        Path(geometry_files["discharge/discharge_snapped_locations"]),
        minimum_upstream_area_km2,
    )
    n_stations: int = len(mapped_station_scores)

    logger.info("Creating discharge dashboard for %d stations.", n_stations)

    dashboard_station_scores: gpd.GeoDataFrame = mapped_station_scores.copy()
    use_daily_discharge_scores(dashboard_station_scores)
    dashboard_characteristics: pd.DataFrame | None = None
    if not dashboard_station_scores.empty:
        dashboard_characteristics = (
            discharge_characteristics.load_dashboard_catchment_characteristics(
                mapped_station_scores=dashboard_station_scores,
                logger=logger,
            )
        )

    logger.info("Loading dashboard geometries...")
    dashboard_geometries: DischargeDashboardGeometries = (
        load_discharge_dashboard_geometries(geometry_files)
    )

    dashboard_path: Path = evaluation_folder / output_path
    logger.info("Preparing interactive chart data...")
    station_dashboard_chart_files, chart_timelines = (
        _write_dashboard_charts_from_saved_scores(
            table_files=table_files,
            logger=logger,
            mapped_station_scores=mapped_station_scores,
            run_output_folder=run_output_folder,
            correct_discharge_observations=correct_discharge_observations,
            dashboard_path=dashboard_path,
            include_return_period_plots=include_return_period_plots,
        )
    )

    logger.info("Creating the dashboard HTML file...")
    write_discharge_dashboard(
        mapped_station_scores=dashboard_station_scores,
        output_path=dashboard_path,
        region_geom=dashboard_geometries.region,
        rivers=dashboard_geometries.rivers,
        station_chart_files=station_dashboard_chart_files,
        waterbodies=dashboard_geometries.waterbodies,
        station_characteristics=dashboard_characteristics,
        excluded_stations=excluded_stations,
        chart_timeline=chart_timelines,
    )
    logger.info("Discharge evaluation dashboard created: %s", dashboard_path)
    logger.info(
        "Tip: If station charts do not appear, download the dashboard HTML "
        "and its charts folder to the same local directory."
    )
    return {"dashboard": str(dashboard_path)}


def write_discharge_dashboard(
    mapped_station_scores: gpd.GeoDataFrame,
    output_path: Path,
    region_geom: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    station_chart_files: dict[str, str],
    waterbodies: gpd.GeoDataFrame | None = None,
    station_characteristics: pd.DataFrame | None = None,
    excluded_stations: gpd.GeoDataFrame | None = None,
    chart_timeline: list[int] | dict[str, list[int]] | None = None,
) -> folium.Map:
    """Save the discharge map with station charts, score layers, and snapping characteristics (e.g., station IDs, upstream areas).

    Charts and snapping features load on demand. Station size reflects upstream
    area; score and attribute layers share a dynamic legend.

    Args:
        mapped_station_scores: Per-station GeoDataFrame with discharge metric columns,
            ``upstream_area_GEB``,
            ``discharge_observations_to_GEB_upstream_area_ratio``, and a
            point geometry.
        output_path: Full path (including filename) where the HTML file is
            saved.
        region_geom: Basin/region boundary GeoDataFrame used to fit the map
            extent and draw the catchment outline.
        rivers: WGS84 river network shown on the map.
        station_chart_files: Interactive chart data files keyed by
            station ID string.
        waterbodies: Optional GeoDataFrame with columns ``waterbody_type``
            (2 = reservoir) and polygon geometries. Centroids are used for dot
            placement.
        station_characteristics: Optional station table containing ``station_ID`` and
            the selected GRDC-Caravan attributes in display units.
        excluded_stations: Stations omitted from summary scores, with an exclusion
            reason. Available charts remain accessible for diagnostic use.
        chart_timeline: Optional shared integer timestamp array (epoch ms) or
            dictionary of timelines by frequency for popup time-series charts.
            Defaults to None.

    Returns:
        The Folium map object (already saved to ``output_path``).
    """
    min_lon, min_lat, max_lon, max_lat = region_geom.total_bounds
    if excluded_stations is not None and not excluded_stations.empty:
        mapped_station_scores = mapped_station_scores.copy()
        # Apply current exclusion reasons when displaying previously saved scores.
        excluded_ids: pd.Index = mapped_station_scores.index.intersection(
            excluded_stations.index
        )
        mapped_station_scores.loc[excluded_ids, "exclusion_reason"] = (
            excluded_stations.loc[excluded_ids, "exclusion_reason"]
        )
    map_center: list[float] = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    discharge_map = folium.Map(location=map_center, tiles=None, prefer_canvas=True)
    TileLayer(
        tiles=_ESRI_TOPO_TILES,
        attr=_ESRI_TOPO_ATTR,
        name="Topographic Map",
    ).add_to(discharge_map)
    TileLayer(
        tiles="Esri.WorldImagery",
        name="Satellite imagery",
        show=False,
        max_zoom=19,
    ).add_to(discharge_map)
    discharge_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(30, 30))
    folium.GeoJson(
        region_geom,
        name="Catchment",
        style_function=lambda _feature: {
            "fillColor": "none",
            "color": "black",
            "weight": 2,
        },
        z_index=1,
    ).add_to(discharge_map)

    _add_river_layers(
        discharge_map,
        rivers,
    )

    metric_layers: list[tuple[folium.FeatureGroup, cm.LinearColormap, str]] = [
        (
            folium.FeatureGroup(name=cfg["name"], show=cfg["show"]),
            cm.LinearColormap(
                colors=cfg["colors"],
                vmin=cfg["vmin"],
                vmax=cfg["vmax"],
                caption=cfg["name"],
            ),
            cfg["col"],
        )
        for cfg in _METRIC_LAYER_CONFIGS
    ]

    layer_upstream: folium.FeatureGroup | None = None
    colormap_upstream: cm.LinearColormap | None = None
    if (
        not mapped_station_scores["discharge_observations_to_GEB_upstream_area_ratio"]
        .isna()
        .all()
    ):
        colormap_upstream = cm.LinearColormap(
            colors=["red", "orange", "yellow", "blue", "green"],
            vmin=0.5,
            vmax=2.0,
            caption="Station / routing upstream area",
        )
        layer_upstream = folium.FeatureGroup(
            name="Station / routing upstream area", show=False
        )

    characteristic_layers: list[tuple[folium.FeatureGroup, dict[str, Any]]] = []
    availability_layer: folium.FeatureGroup | None = None
    characteristic_records: dict[str, dict[str, Any]] = {}
    caravan_available_count: int = 0
    if station_characteristics is not None:
        characteristic_data: dict[str, Any] = _build_characteristic_layer_data(
            mapped_station_scores=mapped_station_scores,
            station_characteristics=station_characteristics,
        )
        characteristic_layers = [
            (
                folium.FeatureGroup(
                    name=f"GRDC-Caravan · {characteristic['label']}",
                    show=False,
                ),
                characteristic,
            )
            for characteristic in characteristic_data["characteristics"]
        ]
        availability_layer = folium.FeatureGroup(
            name="GRDC-Caravan · Data availability",
            show=False,
        )
        characteristic_records = {
            str(station["id"]): station for station in characteristic_data["stations"]
        }
        caravan_available_count = sum(
            bool(station["caravan_available"])
            for station in characteristic_data["stations"]
        )

    largest_upstream_area_sqrt: float = math.sqrt(
        mapped_station_scores["upstream_area_GEB"].max()
    )

    popup_width: int = 800
    station_marker_index: list[StationMarkerIndex] = []
    snapping_layer: folium.FeatureGroup = folium.FeatureGroup(
        name="Station snapping details (gauge → original subgrid pixel → routing grid)",
        show=False,
    )
    snapping_columns: set[str] = {
        "station_longitude",
        "station_latitude",
        "routing_grid_longitude",
        "routing_grid_latitude",
        "original_subgrid_longitude",
        "original_subgrid_latitude",
        "upstream_area_GRDC",
        "upstream_area_GEB",
        "upstream_area_GEB_original_subgrid",
        "station_to_original_subgrid_distance_m",
        "snapped_river_id",
        "snapping_method",
        "timezone_utc_offset",
    }
    snapping_available: bool = snapping_columns.issubset(mapped_station_scores.columns)
    snapping_stations: list[dict[str, Any]] = []

    for station_id, row in mapped_station_scores.iterrows():
        coords: list[float] = [row.geometry.y, row.geometry.x]
        station_id_str: str = str(station_id)
        station_name: str = str(row["station_name"])
        escaped_station_id: str = html.escape(station_id_str, quote=True)
        popup_html: str = (
            f"<div class='geb-popup' data-station-id='{escaped_station_id}' "
            f"style='width:{popup_width}px;'>Loading interactive charts...</div>"
        )
        if pd.notna(row.get("exclusion_reason")):
            popup_html = (
                "<div style='color:#b91c1c;padding:8px'><b>Diagnostic only — excluded from evaluation.</b><br>"
                + html.escape(str(row["exclusion_reason"]))
                + "</div>"
                + popup_html
            )
        timezone_tooltip: str = (
            "<br><span data-geb-utc-offset-hours="
            f"'{float(row['timezone_utc_offset'])}'></span> fixed"
        )
        tooltip: str = (
            f"{escaped_station_id}: {html.escape(station_name, quote=True)}"
            f"{timezone_tooltip}"
        )
        if pd.notna(row.get("exclusion_reason")):
            tooltip += "<br>Diagnostic only — excluded from evaluation"

        if snapping_available:
            snapping_record: dict[str, Any] = _build_snapping_station(
                station_id=station_id_str,
                station_name=station_name,
                row=row,
            )
            snapping_stations.append(snapping_record)

        # Scale circle radius by upstream area (range 5–10 px).
        circle_radius: float = (
            5 + math.sqrt(row["upstream_area_GEB"]) / largest_upstream_area_sqrt * 5
        )
        station_marker_names: list[str] = _add_metric_station_markers(
            row=row,
            metric_layers=metric_layers,
            coords=coords,
            circle_radius=circle_radius,
            popup_html=popup_html,
            popup_width=popup_width,
            tooltip=tooltip,
        )

        if (
            layer_upstream is not None
            and colormap_upstream is not None
            and np.isfinite(row["discharge_observations_to_GEB_upstream_area_ratio"])
        ):
            color_upstream: str | tuple[int, int, int, int] = colormap_upstream(
                float(row["discharge_observations_to_GEB_upstream_area_ratio"])
            )
            if isinstance(color_upstream, str) and color_upstream != "nan":
                upstream_marker = folium.CircleMarker(
                    location=coords,
                    radius=10,
                    color="black",
                    fill=True,
                    fill_color=color_upstream,
                    fill_opacity=0.9,
                    popup=folium.Popup(popup_html, max_width=popup_width),
                    tooltip=tooltip,
                    z_index=1000,
                )
                upstream_marker.add_to(layer_upstream)
                station_marker_names.append(upstream_marker.get_name())

        if availability_layer is not None:
            station_record: dict[str, Any] = characteristic_records[station_id_str]
            station_marker_names.extend(
                _add_characteristic_station_markers(
                    station_record=station_record,
                    characteristic_layers=characteristic_layers,
                    availability_layer=availability_layer,
                    coords=coords,
                    circle_radius=circle_radius,
                    popup_html=popup_html,
                    popup_width=popup_width,
                    station_tooltip=tooltip,
                )
            )

        station_marker_index.append(
            {
                "id": station_id_str,
                "name": station_name,
                "markers": station_marker_names,
            }
        )

    for layer, _, _ in metric_layers:
        layer.add_to(discharge_map)

    if layer_upstream is not None and colormap_upstream is not None:
        layer_upstream.add_to(discharge_map)

    for characteristic_layer, _ in characteristic_layers:
        characteristic_layer.add_to(discharge_map)
    if availability_layer is not None:
        availability_layer.add_to(discharge_map)
    if excluded_stations is not None and not excluded_stations.empty:
        area_distance_count: int = 0
        excluded_layer: folium.FeatureGroup = folium.FeatureGroup(
            name=f"Excluded stations ({len(excluded_stations)})", show=False
        )
        for station_id, row in excluded_stations.iterrows():
            record: dict[str, Any] = _build_snapping_station(
                str(station_id), str(row["station_name"]), row
            )
            area_distance_failure: bool = str(row["exclusion_reason"]).startswith(
                (
                    "Missing station",
                    "Missing upstream",
                    "Routing upstream",
                    "Routing pixel",
                    "No valid river",
                )
            )
            area_distance_count += int(area_distance_failure)
            record["color"] = "#DC2626" if area_distance_failure else "#D97706"
            if (
                station_id not in mapped_station_scores.index
                and len(record["locations"]) == 3
            ):
                snapping_stations.append(record)
            excluded_popup: str = record["popup"]
            if str(station_id) in station_chart_files:
                excluded_popup += (
                    "<hr><b>Diagnostic only — excluded from evaluation.</b>"
                    f"<div class='geb-popup' data-station-id='{html.escape(str(station_id), quote=True)}'>"
                    "Loading interactive charts...</div>"
                )
            marker: folium.CircleMarker = folium.CircleMarker(
                location=record["locations"][0],
                radius=7 if area_distance_failure else 5,
                color=record["color"],
                fill=True,
                fill_opacity=0.8,
                tooltip=record["tooltip"],
                popup=folium.Popup(excluded_popup, max_width=850),
            )
            marker.add_to(excluded_layer)
            station_marker_index.append(
                {
                    "id": str(station_id),
                    "name": str(row["station_name"]),
                    "markers": [marker.get_name()],
                }
            )
        excluded_layer.add_to(discharge_map)
        station_count: int = len(
            mapped_station_scores.index.union(excluded_stations.index)
        )
        cast(Figure, discharge_map.get_root()).html.add_child(
            folium.Element(
                f"<div id='{excluded_layer.get_name()}_legend' style='display:none;position:fixed;top:12px;left:55px;z-index:1000;background:white;padding:8px;border:1px solid #ccc'>"
                f"<b>{station_count} stations shown</b><br>"
                f"<span style='color:#DC2626'>● Area / distance exclusions: {area_distance_count}</span><br>"
                f"<span style='color:#D97706'>● Other exclusions: {len(excluded_stations) - area_distance_count}</span>"
                "<br>Hover or click a station for its exclusion reason.</div>"
            )
        )
        _JavascriptMacro("exclusions.js", excluded_layer.get_name()).add_to(
            discharge_map
        )
    if snapping_stations:
        snapping_layer.add_to(discharge_map)
        # The inactive overlay stays empty; JavaScript creates visible features lazily.
        _JavascriptMacro(
            "snapping.js",
            {
                "layer": snapping_layer.get_name(),
                "stations": snapping_stations,
            },
        ).add_to(discharge_map)

    _inject_station_layer_legend_script(
        discharge_map=discharge_map,
        metric_layers=metric_layers,
        upstream_layer=layer_upstream,
        characteristic_layers=characteristic_layers,
        availability_layer=availability_layer,
        caravan_available_count=caravan_available_count,
        station_count=len(mapped_station_scores),
    )

    chart_macro_data: dict[str, Any] = {
        "stations": station_chart_files,
        "timeline": chart_timeline,
    }
    _JavascriptMacro("charts.js", chart_macro_data).add_to(discharge_map)
    _JavascriptMacro("search.js", station_marker_index).add_to(discharge_map)

    if waterbodies is not None and not waterbodies.empty:
        _add_reservoir_layer(discharge_map, waterbodies)

    _JavascriptMacro("units.js", {}).add_to(discharge_map)
    folium.LayerControl(collapsed=False).add_to(discharge_map)
    if (
        "exclusion_reason" in mapped_station_scores
        and mapped_station_scores["exclusion_reason"].notna().any()
    ):
        cast(Figure, discharge_map.get_root()).html.add_child(
            folium.Element(
                "<div style='position:fixed;bottom:25px;left:15px;z-index:1000;background:white;padding:10px;max-width:350px'>"
                "<b>Diagnostic scores included</b><br>Excluded stations retain KGE and time series for inspection. "
                "They do not contribute to evaluation summaries. See station warnings.</div>"
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    discharge_map.save(str(output_path))
    return discharge_map


def load_discharge_dashboard_geometries(
    geometry_files: dict[str, Path],
) -> DischargeDashboardGeometries:
    """Load the geometries used by the discharge dashboard.

    Args:
        geometry_files: Model geometry paths keyed by dataset name.

    Returns:
        Region boundary, river network, and waterbodies.
    """
    region_geom: gpd.GeoDataFrame = read_geom(geometry_files["mask"])
    all_rivers: gpd.GeoDataFrame = read_geom(geometry_files["routing/rivers"])
    waterbodies: gpd.GeoDataFrame = read_geom(
        geometry_files["waterbodies/waterbody_data"]
    )
    return DischargeDashboardGeometries(
        region=region_geom,
        rivers=select_active_rivers(all_rivers),
        waterbodies=waterbodies,
    )


def determine_master_time_index(
    observations_index: pd.DatetimeIndex,
    simulation_output_folder: Path,
    frequency: str,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex | None:
    """Determine a continuous master time index spanning the evaluation period.

    Notes:
        Ensures all station charts share an identical, aligned time grid for a
        given frequency (e.g. daily midpoint at 12:00:00, or hourly at half-hour).

    Args:
        observations_index: DatetimeIndex of the station observations table.
        simulation_output_folder: Path to model run output folder containing
            the hydrology routing report files.
        frequency: Observation frequency label ("daily" or "hourly").
        period_start: Optional evaluation period start timestamp.
        period_end: Optional evaluation period end timestamp.

    Returns:
        Continuous master DatetimeIndex, or None if simulation files or
        observations are unavailable.

    Raises:
        ValueError: If frequency is not "daily" or "hourly".
    """
    if frequency not in ("daily", "hourly"):
        raise ValueError(
            f"Unsupported frequency '{frequency}'. Expected 'daily' or 'hourly'."
        )

    if observations_index.empty:
        return None

    routing_dir: Path = simulation_output_folder / "report" / "hydrology.routing"
    if not routing_dir.exists():
        return None

    first_parquet_file: Path | None = next(
        routing_dir.glob("discharge_hourly_m3_per_s_*.parquet"), None
    )
    if first_parquet_file is None:
        return None

    # Read the simulation index from the first available station routing file.
    simulated_index: pd.Index = pd.read_parquet(first_parquet_file).index
    if not isinstance(simulated_index, pd.DatetimeIndex) or simulated_index.empty:
        return None

    if frequency == "daily":
        # Match GEB daily aggregation: noon (12:00:00) midpoint timestamps.
        sim_start: pd.Timestamp = simulated_index.min().floor("D") + pd.Timedelta(
            hours=12
        )
        sim_end: pd.Timestamp = simulated_index.max().floor("D") + pd.Timedelta(
            hours=12
        )
        freq_step: str = "D"
    else:
        # Match GEB hourly timestamp convention (half-hour HH:30:00).
        sim_start = simulated_index.min()
        sim_end = simulated_index.max()
        freq_step = "h"

    start_timestamp: pd.Timestamp = max(observations_index.min(), sim_start)
    end_timestamp: pd.Timestamp = min(observations_index.max(), sim_end)

    if period_start is not None:
        aligned_period_start: pd.Timestamp = (
            cast(
                pd.Timestamp,
                period_start.floor("D") + pd.Timedelta(hours=12),
            )
            if frequency == "daily"
            else period_start
        )
        start_timestamp = max(start_timestamp, aligned_period_start)

    if period_end is not None:
        aligned_period_end: pd.Timestamp = (
            cast(
                pd.Timestamp,
                period_end.floor("D") + pd.Timedelta(hours=12),
            )
            if frequency == "daily"
            else period_end
        )
        end_timestamp = min(end_timestamp, aligned_period_end)

    if start_timestamp > end_timestamp:
        return None

    return pd.date_range(start=start_timestamp, end=end_timestamp, freq=freq_step)


def _write_dashboard_charts_from_saved_scores(
    table_files: dict[str, Path],
    logger: logging.Logger,
    mapped_station_scores: gpd.GeoDataFrame,
    run_output_folder: Path,
    correct_discharge_observations: bool,
    dashboard_path: Path,
    include_return_period_plots: bool = True,
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """Save interactive chart data for stations with saved evaluation scores.

    Args:
        table_files: Model observation table paths keyed by dataset name.
        logger: Model logger for progress and diagnostics.
        mapped_station_scores: Saved per-station discharge evaluation metrics.
        run_output_folder: Model output folder for the selected run.
        correct_discharge_observations: Whether to correct simulated discharge
            by the observed-to-GEB upstream-area ratio (dimensionless).
        dashboard_path: Output path of the dashboard HTML file.
        include_return_period_plots: Whether to fit and include return-period
            curves. Defaults to True.

    Returns:
        Tuple of (mapping from station ID to chart data file, mapping from frequency to timeline).

    Raises:
        ValueError: If saved metrics are missing required station columns.
    """
    if mapped_station_scores.empty:
        return {}, {}
    required_columns: set[str] = {
        "station_name",
        "discharge_observations_to_GEB_upstream_area_ratio",
    }
    missing_columns: set[str] = required_columns.difference(
        mapped_station_scores.columns
    )
    if missing_columns:
        raise ValueError(
            "Saved discharge evaluation metrics are missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    observations_by_frequency: dict[str, pd.DataFrame] = (
        discharge_helpers.load_discharge_observations(table_files)
    )
    saved_scores_by_station_id: dict[str, pd.Series] = {
        str(station_id): station_row
        for station_id, station_row in mapped_station_scores.iterrows()
    }

    # Count total work up front for progress reporting.
    total_work: int = sum(
        sum(
            str(station_id) in saved_scores_by_station_id
            for station_id in observations.columns
        )
        for observations in observations_by_frequency.values()
        if not observations.empty
    )
    logger.info(
        "Processing %d station-frequency combinations...",
        total_work,
    )

    station_dashboard_chart_files: dict[str, str] = {}
    chart_timelines: dict[str, list[int]] = {}
    processed: int = 0
    for (
        frequency_label,
        observations_by_station,
    ) in observations_by_frequency.items():
        if observations_by_station.empty:
            continue
        master_time_index: pd.DatetimeIndex | None = determine_master_time_index(
            observations_index=cast(pd.DatetimeIndex, observations_by_station.index),
            simulation_output_folder=run_output_folder,
            frequency=frequency_label,
        )
        if master_time_index is not None:
            chart_timelines[frequency_label] = (
                master_time_index.astype("datetime64[ms]").astype("int64")
            ).tolist()
        for station_id in observations_by_station.columns:
            station_id_text: str = str(station_id)
            if station_id_text not in saved_scores_by_station_id:
                continue
            station_row: pd.Series = saved_scores_by_station_id[station_id_text]
            upstream_area_ratio: float = float(
                station_row["discharge_observations_to_GEB_upstream_area_ratio"]
            )
            observed_discharge_series: pd.Series = observations_by_station[station_id]

            timezone_utc_offset: float = float(station_row["timezone_utc_offset"])
            discharge_comparison: pd.DataFrame = load_station_discharge_comparison(
                output_folder=run_output_folder,
                station_id=station_id,
                observed_discharge=observed_discharge_series,
                apply_upstream_area_correction=correct_discharge_observations,
                upstream_area_ratio=upstream_area_ratio,
                timezone_utc_offset=timezone_utc_offset,
            )
            metrics: dict[str, float] = {
                metric_name: float(station_row[f"{metric_name}_{frequency_label}"])
                for metric_name in DischargeMetrics._fields
                if f"{metric_name}_{frequency_label}" in station_row.index
            }
            station_dashboard_chart_files[station_id_text] = write_station_chart_data(
                dashboard_path=dashboard_path,
                station_id=station_id_text,
                chart_data=build_station_chart_data(
                    discharge_comparison=discharge_comparison,
                    station_name=str(station_row["station_name"]),
                    upstream_area_ratio=upstream_area_ratio,
                    timezone_utc_offset=timezone_utc_offset,
                    metrics=metrics,
                    frequency=frequency_label,
                    logger=logger,
                    include_return_period_plots=include_return_period_plots,
                    master_time_index=master_time_index,
                ),
            )

            processed += 1
            if processed % 100 == 0:
                logger.info(
                    "  %d / %d processed...",
                    processed,
                    total_work,
                )

    logger.info(
        "Chart data built for %d stations.",
        len(station_dashboard_chart_files),
    )
    return station_dashboard_chart_files, chart_timelines


def write_station_chart_data(
    dashboard_path: Path,
    station_id: str,
    chart_data: dict[str, Any],
) -> str:
    """Save station chart data for the browser to load when the popup opens.

    Args:
        dashboard_path: Output path of the dashboard HTML file.
        station_id: Station identifier used to derive a stable asset filename.
        chart_data: Data for the interactive station charts.

    Returns:
        Path to the chart data file relative to the dashboard HTML, using forward slashes.
    """
    chart_folder: Path = dashboard_path.parent / f"{dashboard_path.stem}_charts"
    chart_folder.mkdir(parents=True, exist_ok=True)
    station_hash: str = hashlib.sha256(station_id.encode()).hexdigest()[:16]
    chart_path: Path = chart_folder / f"{station_hash}.js"
    chart_path.write_text(
        "window._gebStationChartPayload="
        + json.dumps(chart_data, separators=(",", ":"))
        + ";",
        encoding="utf-8",
    )
    return chart_path.relative_to(dashboard_path.parent).as_posix()


def build_station_chart_data(
    discharge_comparison: pd.DataFrame,
    station_name: str,
    upstream_area_ratio: float,
    timezone_utc_offset: float,
    metrics: dict[str, float],
    frequency: str,
    logger: logging.Logger,
    include_return_period_plots: bool = True,
    master_time_index: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    """Prepare chart data for one station popup in the discharge dashboard.

    Args:
        discharge_comparison: Observed/simulated discharge dataframe (m3/s).
        station_name: Human-readable station name.
        upstream_area_ratio: Observed-to-model upstream-area ratio (dimensionless).
        timezone_utc_offset: Fixed station UTC offset used to construct local
            calendar days (hours). The source offset does not vary for daylight
            saving time.
        metrics: Discharge skill metrics such as ``KGE``, ``NSE``, and ``R2``
            (dimensionless).
        frequency: Data frequency label, for example ``"daily"`` or ``"hourly"``.
        logger: Model logger for return-period fit diagnostics.
        include_return_period_plots: Whether to fit and include return-period
            curves. Defaults to True.
        master_time_index: Optional master DatetimeIndex shared across all stations
            to align time series onto a single dashboard timeline.

    Returns:
        Chart data with discharge values (m3/s).

    Raises:
        ValueError: If discharge_comparison does not use a DateTimeIndex.
    """
    if not isinstance(discharge_comparison.index, pd.DatetimeIndex):
        raise ValueError(
            "discharge_comparison must use a DateTimeIndex for dashboard charts."
        )

    chart_metrics: dict[str, float | None] = {
        "KGE": _as_finite_float(metrics.get("KGE")),
        "KGE_modified": _as_finite_float(metrics.get("KGE_modified")),
        "KGE_correlation": _as_finite_float(metrics.get("KGE_correlation")),
        "KGE_bias_ratio": _as_finite_float(metrics.get("KGE_bias_ratio")),
        "KGE_variability_ratio": _as_finite_float(metrics.get("KGE_variability_ratio")),
        "NSE": _as_finite_float(metrics.get("NSE")),
        "R2": _as_finite_float(metrics.get("R2")),
        "RMSE": _as_finite_float(metrics.get("RMSE")),
        "RRMSE": _as_finite_float(metrics.get("RRMSE")),
        "upstreamAreaRatio": _as_finite_float(upstream_area_ratio),
        "timezoneUtcOffset": _as_finite_float(timezone_utc_offset),
    }
    for key, value in metrics.items():
        if key not in chart_metrics:
            chart_metrics[key] = _as_finite_float(value)

    chart_data: dict[str, Any] = {
        "stationName": station_name,
        "frequency": frequency,
        "metrics": chart_metrics,
        "timeseries": _build_timeseries_data(
            discharge_comparison=discharge_comparison,
            master_time_index=master_time_index,
        ),
    }
    if include_return_period_plots:
        # Fit only on request: fitting every station dominates dashboard creation.
        return_periods_years: list[int | float] = [2, 5, 10, 25, 50, 100]
        simulated_series: pd.Series = discharge_comparison[
            "discharge_simulations"
        ].copy()
        simulated_series[discharge_comparison["discharge_observations"].isna()] = np.nan
        chart_data["returnPeriods"] = {
            "observed": _build_return_period_data(
                discharge_comparison["discharge_observations"],
                return_periods_years,
                logger,
            ),
            "simulated": _build_return_period_data(
                simulated_series, return_periods_years, logger
            ),
        }
    return chart_data


def _scale_series_to_int_cents(series: pd.Series) -> list[int | None]:
    """Scale discharge values to integer cents (100 * m3/s) with None for NaNs.

    Args:
        series: Discharge series (m3/s).

    Returns:
        List of scaled integer values, or None for non-finite values.
    """
    vals: np.ndarray = series.to_numpy(dtype=float)
    valid_mask: np.ndarray = np.isfinite(vals)
    res: np.ndarray = np.full(len(vals), None, dtype=object)
    res[valid_mask] = np.round(vals[valid_mask] * 100.0).astype(int)
    return res.tolist()


def _to_int_deltas(values: list[int | None]) -> list[int | None]:
    """Encode integer values as deltas relative to the preceding value.

    Notes:
        When a value follows a missing value (None) or appears at the start of
        the series, the absolute value is stored as an anchor. Subsequent finite
        values store the relative delta (current - previous).

    Args:
        values: Sequence of integer values (cents, 100 * m3/s) or None for missing data.

    Returns:
        Delta-encoded sequence with initial absolute values and relative differences.
    """
    deltas: list[int | None] = []
    previous_value: int | None = None
    for value in values:
        if value is None:
            deltas.append(None)
            previous_value = None
        elif previous_value is None:
            deltas.append(value)
            previous_value = value
        else:
            deltas.append(value - previous_value)
            previous_value = value
    return deltas


def _build_timeseries_data(
    discharge_comparison: pd.DataFrame,
    master_time_index: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    """Prepare data for one discharge time-series chart in a popup.

    Notes:
        If ``master_time_index`` is provided, the dataframe is windowed to the
        overlapping active period and aligned with the shared dashboard timeline
        using a start index offset, eliminating leading and trailing missing values.
        Missing observation values within the window are represented as ``None``
        (serialized to JSON ``null``). Discharge values are scaled by 100, stored
        as integers, and delta-encoded to minimize JSON payload size.

    Args:
        discharge_comparison: Observed/simulated discharge dataframe (m3/s).
        master_time_index: Optional master DatetimeIndex shared across all stations.
            When provided, redundant leading/trailing nulls and the "time" array
            are omitted, storing only the start index offset.

    Returns:
        Dictionary with delta-encoded integer observed/simulated values, start index,
        and scale factor.

    Raises:
        ValueError: If ``discharge_comparison`` is not indexed by timestamps.
    """
    if not isinstance(discharge_comparison.index, pd.DatetimeIndex):
        raise ValueError(
            "discharge_comparison must use a DateTimeIndex for dashboard charts."
        )

    if master_time_index is not None and not discharge_comparison.empty:
        common_start: pd.Timestamp = max(
            discharge_comparison.index[0], master_time_index[0]
        )
        common_end: pd.Timestamp = min(
            discharge_comparison.index[-1], master_time_index[-1]
        )
        if common_start <= common_end:
            start_index: int = int(cast(int, master_time_index.get_loc(common_start)))
            end_index: int = int(cast(int, master_time_index.get_loc(common_end))) + 1
            window_slice: pd.DatetimeIndex = master_time_index[start_index:end_index]
            comparison: pd.DataFrame = discharge_comparison.reindex(window_slice)
        else:
            comparison = discharge_comparison.iloc[0:0]
            start_index = 0

        return {
            "start": start_index,
            "scale": 100,
            "deltas": True,
            "observed": _to_int_deltas(
                _scale_series_to_int_cents(comparison["discharge_observations"])
            ),
            "simulated": _to_int_deltas(
                _scale_series_to_int_cents(comparison["discharge_simulations"])
            ),
        }

    comparison = discharge_comparison
    return {
        "time": comparison.index.astype("datetime64[ms]").astype("int64").tolist(),
        "scale": 100,
        "deltas": True,
        "observed": _to_int_deltas(
            _scale_series_to_int_cents(comparison["discharge_observations"])
        ),
        "simulated": _to_int_deltas(
            _scale_series_to_int_cents(comparison["discharge_simulations"])
        ),
    }


def _build_return_period_data(
    series: pd.Series,
    return_periods_years: list[int | float],
    logger: logging.Logger,
) -> dict[str, list[float | None]]:
    """Build fitted return-period values for one discharge series.

    Args:
        series: Regular discharge time series (m3/s).
        return_periods_years: Return periods to estimate (years).
        logger: Model logger for fit diagnostics.

    Returns:
        Dictionary with return periods (years) and fitted discharge values (m3/s).
        Returns empty lists if the fit fails or the series is too short.
    """
    try:
        model: ReturnPeriodModel = ReturnPeriodModel(
            series=series,
            return_periods=return_periods_years,
            fixed_shape=0.0,
            selection_strategy="first_significant",
        )
        return {
            "returnPeriod": [
                _as_finite_float(value)
                for value in model.rl_table["T_years"].to_numpy(dtype=float)
            ],
            "discharge": [
                _as_finite_float(value)
                for value in model.rl_table["GPD_POT_RL"].to_numpy(dtype=float)
            ],
        }
    except (ValueError, RuntimeError, FloatingPointError) as error:
        # A failed extreme-value fit should not remove otherwise valid station
        # charts, but it must remain visible to users diagnosing the output.
        logger.warning("Could not fit dashboard return periods: %s", error)
        return {"returnPeriod": [], "discharge": []}


def _timestamp_to_isoformat(timestamp: Any) -> str:
    """Convert a dashboard timestamp to an ISO-formatted string.

    Args:
        timestamp: Timestamp-like value from a discharge time-series index.

    Returns:
        ISO-formatted timestamp string.

    Raises:
        ValueError: If ``timestamp`` is missing or cannot be represented as a
            timestamp.
    """
    timestamp_value: pd.Timestamp | NaTType = pd.Timestamp(timestamp)
    if not isinstance(timestamp_value, pd.Timestamp):
        raise ValueError("Dashboard chart timestamps must not contain missing values.")
    return timestamp_value.isoformat()


def _build_characteristic_layer_data(
    mapped_station_scores: gpd.GeoDataFrame,
    station_characteristics: pd.DataFrame,
) -> dict[str, Any]:
    """Build characteristic-layer metadata aligned to evaluated stations.

    Args:
        mapped_station_scores: Evaluated stations indexed by station identifier.
        station_characteristics: Curated GRDC-Caravan values in display units, with
            a unique ``station_ID`` column.

    Returns:
        Catchment attribute settings and values for each station.

    Raises:
        ValueError: If station identifiers or usable characteristic data are
            unavailable.
    """
    if "station_ID" not in station_characteristics.columns:
        raise ValueError("Dashboard characteristics have no station_ID column.")
    if "grdc_caravan_matched" not in station_characteristics.columns:
        raise ValueError("Dashboard characteristics have no GRDC-Caravan match status.")
    if station_characteristics["station_ID"].duplicated().any():
        raise ValueError("Dashboard characteristics contain duplicate station IDs.")
    if mapped_station_scores.index.astype(str).duplicated().any():
        raise ValueError("Dashboard evaluation contains duplicate station IDs.")

    characteristics_by_station: pd.DataFrame = station_characteristics.copy()
    characteristics_by_station["station_ID"] = characteristics_by_station[
        "station_ID"
    ].astype(str)
    characteristics_by_station = characteristics_by_station.set_index("station_ID")
    evaluation_station_ids: pd.Index = pd.Index(
        mapped_station_scores.index.astype(str), name="station_ID"
    )
    # Restrict distributions to displayed stations so percentile colours and
    # legend counts describe exactly the points visible on the dashboard.
    characteristics_by_station = characteristics_by_station.reindex(
        evaluation_station_ids
    )
    characteristic_configs: list[dict[str, Any]] = []
    percentile_by_characteristic: dict[str, pd.Series] = {}
    usable_characteristics: list[CatchmentCharacteristic] = []
    for characteristic in DASHBOARD_CATCHMENT_CHARACTERISTICS:
        if characteristic.column not in characteristics_by_station.columns:
            continue
        prepared_values: tuple[dict[str, int | list[float]], pd.Series] | None = (
            _prepare_characteristic_values(
                characteristics_by_station[characteristic.column]
            )
        )
        if prepared_values is None:
            continue
        statistics, percentile_ranks = prepared_values
        percentile_by_characteristic[characteristic.column] = percentile_ranks
        characteristic_configs.append(
            {
                "column": characteristic.column,
                "label": characteristic.label,
                **statistics,
            }
        )
        usable_characteristics.append(characteristic)
    station_records: list[dict[str, Any]] = []
    for station_id, characteristic_row in characteristics_by_station.iterrows():
        values: dict[str, float | None] = {}
        percentiles: dict[str, float | None] = {}
        for characteristic in usable_characteristics:
            values[characteristic.column] = _as_finite_float(
                characteristic_row[characteristic.column]
            )
            percentiles[characteristic.column] = _as_finite_float(
                percentile_by_characteristic[characteristic.column].loc[station_id]
            )
        station_records.append(
            {
                "id": str(station_id),
                "caravan_available": bool(characteristic_row["grdc_caravan_matched"])
                if pd.notna(characteristic_row["grdc_caravan_matched"])
                else False,
                "values": values,
                "percentiles": percentiles,
            }
        )
    return {
        "characteristics": characteristic_configs,
        "stations": station_records,
    }


def _prepare_characteristic_values(
    values: pd.Series,
) -> tuple[dict[str, int | list[float]], pd.Series] | None:
    """Prepare legend values and percentile ranks for one characteristic.

    Map colours use empirical percentile ranks rather than raw-value intervals,
    so skewed variables retain spatial contrast. Zero is a valid observed value
    and is ranked consistently with every other finite value.

    Args:
        values: Catchment characteristic values in display units.

    Returns:
        Legend statistics and aligned percentile ranks, or `None` when fewer
        than two distinct finite values are available.
    """
    numeric_values: pd.Series = pd.to_numeric(values, errors="coerce")
    finite_values: pd.Series = numeric_values.where(np.isfinite(numeric_values))
    valid_values: pd.Series = finite_values.dropna()
    if len(valid_values) < 2 or valid_values.nunique() < 2:
        return None

    reference_values: np.ndarray = np.nanpercentile(
        valid_values.to_numpy(dtype=float), [0.0, 25.0, 50.0, 75.0, 100.0]
    )
    percentile_ranks: pd.Series = pd.Series(np.nan, index=values.index, dtype=float)
    average_ranks: pd.Series = valid_values.rank(method="average")
    percentile_ranks.loc[valid_values.index] = (
        (average_ranks - 1.0) / (len(valid_values) - 1.0) * 100.0
    )
    statistics: dict[str, int | list[float]] = {
        "reference_values": [round(float(value), 2) for value in reference_values],
        "missing_count": int(len(numeric_values) - len(valid_values)),
        "ranked_count": int(len(valid_values)),
    }
    return statistics, percentile_ranks


def _add_metric_station_markers(
    row: pd.Series,
    metric_layers: list[tuple[folium.FeatureGroup, cm.LinearColormap, str]],
    coords: list[float],
    circle_radius: float,
    popup_html: str,
    popup_width: int,
    tooltip: str,
) -> list[str]:
    """Add one station marker to each metric layer.

    Args:
        row: Evaluation metrics for one station (dimensionless).
        metric_layers: Layers, colormaps, and metric names to show.
        coords: Marker coordinates as ``[latitude, longitude]`` (degrees).
        circle_radius: Marker radius (pixels).
        popup_html: Popup placeholder HTML.
        popup_width: Popup width (pixels).
        tooltip: Marker tooltip text.

    Returns:
        Folium JavaScript variable names for the created markers.
    """
    marker_names: list[str] = []
    for layer, colormap, metric_name in metric_layers:
        metric_value: float = row.get(metric_name, np.nan)
        fill_color: str = colormap(metric_value) if pd.notna(metric_value) else "gray"
        marker = folium.CircleMarker(
            location=coords,
            radius=circle_radius,
            color="black",
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=popup_width),
            tooltip=tooltip,
            z_index=1000,
        )
        marker.add_to(layer)
        marker_names.append(marker.get_name())
    return marker_names


def _add_characteristic_station_markers(
    station_record: dict[str, Any],
    characteristic_layers: list[tuple[folium.FeatureGroup, dict[str, Any]]],
    availability_layer: folium.FeatureGroup,
    coords: list[float],
    circle_radius: float,
    popup_html: str,
    popup_width: int,
    station_tooltip: str,
) -> list[str]:
    """Add GRDC-Caravan availability and characteristic markers for a station.

    Catchment characteristic layers contain only stations with a finite value. The
    availability layer contains every evaluated station, making absent
    GRDC-Caravan coverage explicit without treating it as a numeric zero.

    Args:
        station_record: JSON-safe station characteristic record.
        characteristic_layers: Feature groups and their characteristic metadata.
        availability_layer: Feature group showing GRDC-Caravan match status.
        coords: Marker coordinates as ``[latitude, longitude]`` (degrees).
        circle_radius: Marker radius (pixels).
        popup_html: Popup placeholder HTML.
        popup_width: Popup width (pixels).
        station_tooltip: Station identifier and name.

    Returns:
        Folium JavaScript variable names for the created markers.
    """
    marker_names: list[str] = []
    caravan_available: bool = bool(station_record["caravan_available"])
    availability_label: str = "available" if caravan_available else "not available"
    availability_marker = folium.CircleMarker(
        location=coords,
        radius=circle_radius,
        color="black",
        fill=True,
        fill_color=(
            _CARAVAN_AVAILABLE_COLOR
            if caravan_available
            else _CARAVAN_UNAVAILABLE_COLOR
        ),
        fill_opacity=0.9,
        popup=folium.Popup(popup_html, max_width=popup_width),
        tooltip=f"{station_tooltip}<br>GRDC-Caravan data: {availability_label}",
        z_index=1000,
    )
    availability_marker.add_to(availability_layer)
    marker_names.append(availability_marker.get_name())

    for layer, characteristic in characteristic_layers:
        value: float | None = station_record["values"][characteristic["column"]]
        if value is None:
            continue
        percentile: float | None = station_record["percentiles"][
            characteristic["column"]
        ]
        assert percentile is not None, (
            f"Finite characteristic {characteristic['column']} has no rank."
        )
        absolute_value: float = abs(value)
        decimal_places: int = (
            0
            if absolute_value >= 1000.0
            else 1
            if absolute_value >= 100.0
            else 2
            if absolute_value >= 10.0
            else 3
        )
        formatted_value: str = f"{value:,.{decimal_places}f}"
        fill_color: str = _CHARACTERISTIC_COLORMAP(percentile)
        rank_text: str = f"percentile rank {percentile:.0f}"
        characteristic_marker = folium.CircleMarker(
            location=coords,
            radius=circle_radius,
            color="black",
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=popup_width),
            tooltip=(
                f"{station_tooltip}<br>{characteristic['label']}: "
                f"{formatted_value} ({rank_text})"
            ),
            z_index=1000,
        )
        characteristic_marker.add_to(layer)
        marker_names.append(characteristic_marker.get_name())
    return marker_names


def _inject_station_layer_legend_script(
    discharge_map: folium.Map,
    metric_layers: list[tuple[folium.FeatureGroup, cm.LinearColormap, str]],
    upstream_layer: folium.FeatureGroup | None,
    characteristic_layers: list[tuple[folium.FeatureGroup, dict[str, Any]]],
    availability_layer: folium.FeatureGroup | None,
    caravan_available_count: int,
    station_count: int,
) -> None:
    """Add one dynamic legend and enforce one active station-value layer.

    Args:
        discharge_map: Performance map receiving the legend control.
        metric_layers: Metric feature groups, colormaps, and source columns.
        upstream_layer: Optional upstream-area-ratio feature group.
        characteristic_layers: GRDC-Caravan feature groups and metadata.
        availability_layer: Optional GRDC-Caravan coverage feature group.
        caravan_available_count: Stations matched to GRDC-Caravan.
        station_count: Total evaluated stations.
    """
    configs_by_column: dict[str, dict[str, Any]] = {
        str(config["col"]): config for config in _METRIC_LAYER_CONFIGS
    }
    legend_configs: list[dict[str, Any]] = []
    for layer, _, column in metric_layers:
        config: dict[str, Any] = configs_by_column[column]
        legend_configs.append(
            {
                "layer": layer.get_name(),
                "kind": "continuous",
                "name": config["name"],
                "colors": config["colors"],
                "minimum": config["vmin"],
                "maximum": config["vmax"],
                "show": config["show"],
            }
        )
    if upstream_layer is not None:
        legend_configs.append(
            {
                "layer": upstream_layer.get_name(),
                "kind": "continuous",
                "name": "Upstream Area Ratio",
                "colors": ["red", "orange", "yellow", "blue", "green"],
                "minimum": 0.5,
                "maximum": 2.0,
                "show": False,
            }
        )
    for layer, characteristic in characteristic_layers:
        legend_configs.append(
            {
                "layer": layer.get_name(),
                "kind": "characteristic",
                "name": characteristic["label"],
                "colors": _CHARACTERISTIC_COLORS,
                "minimum": 0.0,
                "maximum": 100.0,
                "reference_values": characteristic["reference_values"],
                "ranked_count": characteristic["ranked_count"],
                "missing_count": characteristic["missing_count"],
                "show": False,
            }
        )
    if availability_layer is not None:
        legend_configs.append(
            {
                "layer": availability_layer.get_name(),
                "kind": "availability",
                "name": "GRDC-Caravan data availability",
                "available_count": caravan_available_count,
                "unavailable_count": station_count - caravan_available_count,
                "available_color": _CARAVAN_AVAILABLE_COLOR,
                "unavailable_color": _CARAVAN_UNAVAILABLE_COLOR,
                "show": False,
            }
        )
    _JavascriptMacro("legend.js", legend_configs).add_to(discharge_map)


def _add_river_layers(
    discharge_map: folium.Map,
    rivers: gpd.GeoDataFrame,
) -> None:
    """Show all active rivers, with upstream areas formatted in the browser.

    Args:
        discharge_map: Map receiving the river network.
        rivers: Active WGS84 river segments indexed by river ID, with areas in m².
    """
    river_data: gpd.GeoDataFrame = rivers[["uparea_m2", "geometry"]].copy()
    river_data["river_id"] = rivers.index.astype(str)
    river_layer: folium.GeoJson = folium.GeoJson(
        river_data.to_json(drop_id=True),
        name="Active rivers",
        style_function=lambda _feature: {
            "color": "#4A90D9",
            "weight": 1.2,
            "opacity": 0.65,
        },
        highlight_function=lambda _feature: {"weight": 4, "opacity": 1},
    )
    river_layer.add_to(discharge_map)
    _JavascriptMacro("rivers.js", {"layer": river_layer.get_name()}).add_to(
        discharge_map
    )


def _add_reservoir_layer(
    discharge_map: folium.Map,
    waterbodies: gpd.GeoDataFrame,
) -> None:
    """Add reservoir point layers to the discharge map.

    Args:
        discharge_map: Folium map receiving waterbody layers.
        waterbodies: Waterbody GeoDataFrame with polygon geometries and
            ``waterbody_type`` identifiers. Only reservoirs are shown.
    """
    waterbody_layers: dict[int, folium.FeatureGroup] = {
        waterbody_type: folium.FeatureGroup(name=style["label"] + "s", show=True)
        for waterbody_type, style in _WATERBODY_STYLE.items()
    }
    reservoir_mask: pd.Series = (
        waterbodies["waterbody_type"].astype(int) == RESERVOIR_WATERBODY_TYPE
    )
    reservoirs: gpd.GeoDataFrame = waterbodies.loc[reservoir_mask].copy()
    if reservoirs.empty:
        return

    waterbodies_wgs84: gpd.GeoDataFrame = reservoirs.to_crs(epsg=4326)
    for _, waterbody_row in waterbodies_wgs84.iterrows():
        waterbody_style: dict[str, str] = _WATERBODY_STYLE[RESERVOIR_WATERBODY_TYPE]

        centroid = waterbody_row.geometry.centroid
        area_m2: float | None = (
            float(waterbody_row["average_area"])
            if "average_area" in waterbody_row.index
            else None
        )
        volume_m3: float | None = (
            float(waterbody_row["volume_total"])
            if "volume_total" in waterbody_row.index
            else None
        )
        popup_lines: list[str] = [
            f"<b>{waterbody_style['label']}</b> "
            f"(ID {waterbody_row.get('waterbody_id', '?')})<br>"
        ]
        if area_m2 is not None:
            popup_lines.append(
                f"Area: <span data-geb-area-m2='{area_m2}'></span> km²<br>"
            )
        if volume_m3 is not None:
            popup_lines.append(
                f"Volume: <span data-geb-volume-m3='{volume_m3}'></span> km³<br>"
            )

        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=5,
            color="black",
            weight=0.5,
            fill=True,
            fill_color=waterbody_style["color"],
            fill_opacity=0.8,
            popup=folium.Popup("".join(popup_lines), max_width=200),
            tooltip=f"{waterbody_style['label']} {waterbody_row.get('waterbody_id', '')}",
            z_index=500,
        ).add_to(waterbody_layers[RESERVOIR_WATERBODY_TYPE])

    for waterbody_layer in waterbody_layers.values():
        waterbody_layer.add_to(discharge_map)


def _build_snapping_station(
    station_id: str,
    station_name: str,
    row: pd.Series,
) -> dict[str, Any]:
    """Prepare station data to show snapping details when requested.

    Excluded stations show the reason alongside their snapping diagnostics.

    Args:
        station_id: Gauging station identifier.
        station_name: Human-readable station name.
        row: Evaluation row with snapping metadata.

    Returns:
        Coordinates, tooltip, status color, and popup text.
    """
    gauge_longitude: float = float(row["station_longitude"])
    gauge_latitude: float = float(row["station_latitude"])
    if any(
        pd.isna(row.get(column))
        for column in (
            "original_subgrid_longitude",
            "original_subgrid_latitude",
            "routing_grid_longitude",
            "routing_grid_latitude",
        )
    ):
        reason: str = html.escape(
            str(row.get("exclusion_reason", "No valid river match."))
        )
        station_label: str = f"{html.escape(station_name)} ({html.escape(station_id)})"
        area_m2: float = float(row.get("upstream_area_GRDC", np.nan))
        area_label: str = (
            f"<span data-geb-area-m2='{area_m2}'></span> km²"
            if np.isfinite(area_m2) and area_m2 > 0
            else "missing"
        )
        return {
            "id": station_id,
            "locations": [[gauge_latitude, gauge_longitude]],
            "color": "#DC2626",
            "tooltip": f"{station_label}<br>EXCLUDED: {reason}",
            "popup": f"<b>{station_label}</b><br><b>EXCLUDED</b><br>{reason}<br>Station area: {area_label}<br>Gauge: {gauge_latitude:.5f}, {gauge_longitude:.5f}",
        }
    original_subgrid_longitude: float = float(row["original_subgrid_longitude"])
    original_subgrid_latitude: float = float(row["original_subgrid_latitude"])
    routing_longitude: float = float(row["routing_grid_longitude"])
    routing_latitude: float = float(row["routing_grid_latitude"])
    station_area_m2: float = float(row["upstream_area_GRDC"])
    routing_area_m2: float = float(row["upstream_area_GEB"])
    original_subgrid_area_m2: float = float(row["upstream_area_GEB_original_subgrid"])
    station_to_routing_distance_m: float = GEOD.inv(
        gauge_longitude,
        gauge_latitude,
        routing_longitude,
        routing_latitude,
    )[2]
    station_to_original_subgrid_distance_m: float = float(
        row["station_to_original_subgrid_distance_m"]
    )
    original_subgrid_to_routing_distance_m: float = GEOD.inv(
        original_subgrid_longitude,
        original_subgrid_latitude,
        routing_longitude,
        routing_latitude,
    )[2]
    station_routing_area_ratio: float = (
        station_area_m2 / routing_area_m2 if routing_area_m2 > 0 else float("nan")
    )
    original_subgrid_station_area_ratio: float = (
        original_subgrid_area_m2 / station_area_m2
        if station_area_m2 > 0
        else float("nan")
    )
    routing_original_subgrid_area_ratio: float = (
        routing_area_m2 / original_subgrid_area_m2
        if original_subgrid_area_m2 > 0
        else float("nan")
    )
    timezone_offset_hours: float = float(row["timezone_utc_offset"])
    area_warning: bool = not 0.9 <= routing_original_subgrid_area_ratio <= 1.1
    status_label: str = "ROUTING AREA WARNING" if area_warning else "PASS"
    status_color: str = "#EA580C" if area_warning else "#16A34A"
    exclusion_reason: str = (
        str(row.get("exclusion_reason", ""))
        if pd.notna(row.get("exclusion_reason", ""))
        else ""
    )
    if exclusion_reason:
        status_label = "EXCLUDED"
        status_color = "#DC2626"
    escaped_name: str = html.escape(station_name)
    escaped_id: str = html.escape(station_id)
    popup_html: str = (
        f"<b>{escaped_name}</b> ({escaped_id})<br>"
        f"<b>Snapping details: <span style='color:{status_color}'>{status_label}</span></b><br>"
        f"Gauging station: {gauge_latitude:.5f}, {gauge_longitude:.5f}<br>"
        f"Selected original subgrid: {original_subgrid_latitude:.5f}, {original_subgrid_longitude:.5f}<br>"
        f"Routing pixel: {routing_latitude:.5f}, {routing_longitude:.5f}<br>"
        f"River ID: {int(row['snapped_river_id'])}<br>"
        f"Gauge–routing distance: <span data-geb-distance-m='{station_to_routing_distance_m}'></span> km<br>"
        f"Gauge–original subgrid distance: <span data-geb-distance-m='{station_to_original_subgrid_distance_m}'></span> km<br>"
        f"Original subgrid–routing distance: <span data-geb-distance-m='{original_subgrid_to_routing_distance_m}'></span> km<br>"
        f"Station area: <span data-geb-area-m2='{station_area_m2}'></span> km²<br>"
        f"Original subgrid area: <span data-geb-area-m2='{original_subgrid_area_m2}'></span> km²<br>"
        f"Routing area: <span data-geb-area-m2='{routing_area_m2}'></span> km²<br>"
        f"Original subgrid/station area ratio: {original_subgrid_station_area_ratio:.2f}<br>"
        f"Routing/original subgrid area ratio: {routing_original_subgrid_area_ratio:.2f}<br>"
        f"Station/routing area ratio: {station_routing_area_ratio:.2f}<br>"
        f"Daily aggregation offset: <span data-geb-utc-offset-hours='{timezone_offset_hours}'></span> (fixed; no DST)<br>"
        "Observation day: local midnight to midnight<br>"
        f"{html.escape(exclusion_reason)}"
    )
    tooltip: str = (
        f"{escaped_id}: {escaped_name}<br>Snapping details: {status_label}"
        f"<br>Original subgrid/Station area: {original_subgrid_station_area_ratio:.2f}; "
        f"routing/original subgrid: {routing_original_subgrid_area_ratio:.2f}"
        f"<br>Gauge–routing: <span data-geb-distance-m='{station_to_routing_distance_m}'></span> km"
        f"<br><span data-geb-utc-offset-hours='{timezone_offset_hours}'></span> fixed"
        "<br>Daily window: local midnight to midnight"
    )
    line_locations: list[list[float]] = [
        [gauge_latitude, gauge_longitude],
        [original_subgrid_latitude, original_subgrid_longitude],
        [routing_latitude, routing_longitude],
    ]
    return {
        "id": escaped_id,
        "locations": line_locations,
        "color": status_color,
        "tooltip": tooltip,
        "popup": popup_html,
    }


def _script_json(value: Any) -> str:
    """Serialize script data without HTML or Jinja template delimiters.

    Args:
        value: JSON-compatible dashboard data.

    Returns:
        JSON that remains safe when Folium processes the script templates more than once.
    """
    return (
        str(htmlsafe_json_dumps(value))
        .replace("{{", "\\u007b\\u007b")
        .replace("{%", "\\u007b%")
        .replace("{#", "\\u007b#")
    )


def _as_finite_float(
    value: float | int | np.floating | None, decimals: int = 2
) -> float | None:
    """Convert a numeric value to a rounded finite JSON-friendly float.

    Args:
        value: Value to convert (dimensionless unless documented by the caller).
        decimals: Number of decimal places to round to. Defaults to 2.

    Returns:
        Finite float value rounded to the specified decimals, or None for missing,
        NaN, or infinite values.
    """
    # JSON has no NaN or infinity literals; null also creates gaps in Plotly charts.
    if value is None:
        return None
    float_value: float = float(value)
    return round(float_value, decimals) if np.isfinite(float_value) else None
