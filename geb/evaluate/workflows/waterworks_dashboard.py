"""Build the AMBER waterworks page from catalog and runtime diagnostics."""

import base64
import gzip
import html
import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from branca.element import Figure


def build_waterworks_payload(
    barriers: gpd.GeoDataFrame | None,
    run_output_folder: Path | None,
) -> dict[str, Any]:
    """Join AMBER structures to hourly diagnostics using their exact grid pixels.

    Notes:
        Missing reports remain unavailable rather than inferring gate operation
        from catalog type. Excluded catalog objects are retained for inspection.
        Runtime locations are used for matched modeled objects.

    Args:
        barriers: Barrier catalog including source, inclusion, ID and geometry.
        run_output_folder: Simulation folder containing weir_heights.csv and report.

    Returns:
        JSON-compatible structures, epoch-millisecond timeline, and diagnostics.
        Flows are in m³/s; opening fractions are dimensionless. Timeline labels
        mark hour ends (30 minutes after the reporter's mid-hour timestamps).

    Raises:
        ValueError: If catalog geometry or report timestamps are invalid, or
            runtime pixel identifiers are duplicated.
    """
    payload: dict[str, Any] = {"structures": [], "timeline": []}
    if barriers is None or barriers.empty:
        return payload
    required: set[str] = {"source", "barrier_id", "included"}
    if not required.issubset(barriers.columns) or barriers.crs is None:
        raise ValueError(
            "Barrier catalog requires source, barrier_id, included and a CRS."
        )
    amber: gpd.GeoDataFrame = barriers.loc[barriers.source.eq("AMBER")].to_crs(4326)
    amber = amber.loc[amber.geometry.notna() & ~amber.geometry.is_empty]
    if not amber.geometry.geom_type.eq("Point").all():
        raise ValueError("AMBER catalog must contain point geometries.")
    runtime: pd.DataFrame = pd.DataFrame()
    if run_output_folder is not None:
        metadata_path: Path = run_output_folder / "weir_heights.csv"
        if metadata_path.exists():
            runtime = pd.read_csv(metadata_path)
            if {"grid_row", "grid_column", "grid_cell_index"}.issubset(runtime.columns):
                if runtime.duplicated(["grid_row", "grid_column"]).any():
                    raise ValueError("Runtime waterworks grid pixels must be unique.")
                runtime = runtime.set_index(["grid_row", "grid_column"])
            else:
                runtime = pd.DataFrame()
    if not runtime.empty:
        if {"grid_row", "grid_column"}.issubset(amber.columns):
            modeled_pixels: pd.MultiIndex = pd.MultiIndex.from_frame(
                amber.loc[amber.included.fillna(False), ["grid_row", "grid_column"]]
            )
            runtime = runtime.loc[runtime.index.isin(modeled_pixels)]
        else:
            runtime = pd.DataFrame()
    series: dict[str, pd.DataFrame] = {}
    timeline: pd.DatetimeIndex = pd.DatetimeIndex([])
    grid_keys: list[str] = []
    if not runtime.empty:
        grid_keys = runtime.grid_cell_index.astype(str).tolist()
    quantity: str
    for quantity in ("open_fraction", "outflow_m3_s", "inflow_m3_s"):
        if run_output_folder is None or not grid_keys:
            continue
        report_path: Path = (
            run_output_folder
            / "report"
            / "hydrology.routing"
            / f"waterworks_{quantity}.parquet"
        )
        if not report_path.exists():
            continue
        # Read only matched AMBER columns; large runs may also report many GDW barriers.
        available_columns: set[str] = set(pq.read_schema(report_path).names)
        selected_columns: list[str] = [
            key for key in grid_keys if key in available_columns
        ]
        report: pd.DataFrame = pd.read_parquet(report_path, columns=selected_columns)
        if not isinstance(report.index, pd.DatetimeIndex) or report.index.hasnans:
            raise ValueError("Waterworks reports require valid datetime indices.")
        if report.index.has_duplicates or not report.index.is_monotonic_increasing:
            raise ValueError("Waterworks timestamps must be unique and increasing.")
        report.columns = report.columns.astype(str)
        series[quantity] = report.loc[:, report.columns.intersection(grid_keys)]
        timeline = timeline.union(report.index)
    # Flow reports are centered on the hour; opening snapshots are sampled at its end.
    payload["timeline"] = (
        (timeline + pd.Timedelta(minutes=30)).as_unit("ms").asi8
    ).tolist()
    row: pd.Series
    for _, row in amber.iterrows():
        if not np.isfinite([row.geometry.x, row.geometry.y]).all():
            continue
        included: bool = pd.notna(row.included) and bool(row.included)
        structure: dict[str, Any] = {
            "id": str(row.barrier_id),
            "type": str(row.get("barrier_type", "Unknown")),
            "included": included,
            "reason": str(row.get("exclusion_reason", "")),
            "lat": float(row.geometry.y),
            "lon": float(row.geometry.x),
            "operation": "unknown" if included else "excluded",
            "height": None,
            "series": {},
        }
        pixel: tuple[Any, Any] = (row.get("grid_row", -1), row.get("grid_column", -1))
        if included and not runtime.empty and pixel in runtime.index:
            metadata: pd.Series = runtime.loc[pixel]
            structure.update(
                operation="gate" if bool(metadata.instream_dam) else "fixed",
                height=float(metadata.effective_height_m),
                lat=float(metadata.latitude_deg),
                lon=float(metadata.longitude_deg),
            )
            grid_key: str = str(int(metadata.grid_cell_index))
            for quantity, report in series.items():
                if grid_key in report.columns:
                    values: np.ndarray = (
                        report[grid_key].reindex(timeline).to_numpy(dtype=float)
                    )
                    structure["series"][quantity] = [
                        round(float(value), 4) if np.isfinite(value) else None
                        for value in values
                    ]
        payload["structures"].append(structure)
    return payload


def write_waterworks_dashboard(
    output_path: Path,
    discharge_path: Path,
    region: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    barriers: gpd.GeoDataFrame | None,
    run_output_folder: Path | None,
) -> None:
    """Write a linked AMBER map with playback and clickable diagnostic histories.

    Args:
        output_path: Destination HTML file, beside the discharge dashboard.
        discharge_path: Discharge dashboard file used by the navigation link.
        region: Region boundary with a CRS.
        rivers: River geometries with a CRS.
        barriers: Optional AMBER/GDW barrier catalog.
        run_output_folder: Optional simulation folder containing hourly reports.

    Returns:
        None. Writes a standalone HTML page using the dashboard's existing Leaflet.

    Raises:
        ValueError: If barrier records or hourly report timestamps are invalid.
        OSError: If the HTML file cannot be written.
    """  # noqa: DOC202, DOC502
    # Import here to reuse the dashboard's safe template embedding without a cycle.
    from .dashboard import _JavascriptMacro

    payload: dict[str, Any] = build_waterworks_payload(barriers, run_output_folder)
    bounds: np.ndarray = region.to_crs(4326).total_bounds
    waterworks_map: folium.Map = folium.Map(
        location=[
            float((bounds[1] + bounds[3]) / 2),
            float((bounds[0] + bounds[2]) / 2),
        ],
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri, HERE, Garmin, OpenStreetMap contributors",
        prefer_canvas=True,
        zoom_control=False,
    )
    waterworks_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    folium.GeoJson(
        region.to_crs(4326),
        name="Catchment",
        style_function=lambda _: {"color": "#94a3b8", "weight": 1, "fillOpacity": 0},
    ).add_to(waterworks_map)
    if not rivers.empty:
        folium.GeoJson(
            rivers.to_crs(4326)[["geometry"]],
            name="Rivers",
            style_function=lambda _: {
                "color": "#93c5da",
                "weight": 1.5,
                "opacity": 0.7,
            },
        ).add_to(waterworks_map)
    encoded: str = base64.b64encode(
        gzip.compress(
            json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        )
    ).decode("ascii")
    _JavascriptMacro(
        "waterworks.js",
        {
            "payload": encoded,
            "discharge_url": quote(discharge_path.name, safe=""),
        },
    ).add_to(waterworks_map)
    cast(Figure, waterworks_map.get_root()).html.add_child(
        folium.Element(
            "<title>AMBER waterworks · GEB</title>"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waterworks_map.save(str(output_path))


def waterworks_navigation(output_path: Path) -> str:
    """Return a safely escaped link to the companion AMBER page.

    Args:
        output_path: Discharge dashboard path.

    Returns:
        Navigation HTML linking sibling files.

    Raises:
        ValueError: If the path has no filename.
    """
    if not output_path.name:
        raise ValueError("Dashboard path requires a filename.")
    filename: str = html.escape(
        quote(output_path.stem + "_amber.html", safe=""), quote=True
    )
    return (
        '<a href="' + filename + '" style="position:fixed;top:12px;left:65px;'
        "z-index:1001;background:#0f172a;color:white;padding:12px 18px;border-radius:9px;"
        'font:600 14px system-ui;text-decoration:none;box-shadow:0 3px 12px #0003">'
        "AMBER waterworks ↗</a>"
    )
