"""Create a publication-ready folder of simulated station discharge (first draft version)."""

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.evaluate.hydrology import Hydrology


PUBLICATION_README_TEMPLATE: str = """# GEB station discharge simulations

This folder contains raw GEB simulated discharge for {station_count} gauging
stations included in the discharge evaluation for run `{run_name}`.

## Contents

- `station_catalog.csv`: station identity and source, original gauge location,
  and the exact snapped model-cell location.
- `simulations/*.parquet`: the original hourly GEB reporter file for each
  station. Files are named `discharge_hourly_m3_per_s_<station_id>.parquet` and
  contain a `discharge_hourly_m3_per_s_<station_id>` column and datetime index.
- `evaluation_metrics.xlsx`: derived station-level evaluation results.

Observed discharge is deliberately excluded. GRDC does not permit downloaded
observations to be redistributed to third parties or via the internet.
Observations are available directly from the GRDC Data Portal:
https://grdc.bafg.de/data/data_portal/.

Coordinates use WGS 84 longitude/latitude (`EPSG:4326`). Discharge is in cubic
metres per second (`m3 s-1`). Simulations are unmodified reporter values: no
observation-based upstream-area correction or temporal resampling is applied.
"""


def _station_id_text(station_id: object) -> str:
    """Return a stable text representation of a station identifier.

    Args:
        station_id: Numeric or textual gauging-station identifier.

    Returns:
        Identifier without a spreadsheet-style decimal suffix.

    Raises:
        ValueError: If the identifier is missing or empty.
    """
    if pd.isna(station_id):
        raise ValueError("Station identifier cannot be missing.")
    identifier: str = str(station_id).strip()
    if not identifier:
        raise ValueError("Station identifier cannot be empty.")
    try:
        numeric_identifier: float = float(identifier)
        if numeric_identifier.is_integer():
            return str(int(numeric_identifier))
    except ValueError:
        pass
    return identifier


def _coordinate_pair(value: object, field_name: str) -> tuple[float, float]:
    """Validate and unpack a stored coordinate pair.

    Args:
        value: Two-value array-like coordinate pair.
        field_name: Field name used in validation errors.

    Returns:
        Coordinate pair as finite floating-point values.

    Raises:
        ValueError: If the value does not contain two finite coordinates.
    """
    coordinates: np.ndarray = np.asarray(value, dtype=float).reshape(-1)
    if len(coordinates) != 2 or not np.isfinite(coordinates).all():
        raise ValueError(f"{field_name} must contain two finite coordinates.")
    return float(coordinates[0]), float(coordinates[1])


def export_discharge_publication_data(
    self: Hydrology,
    run_name: str = "default",
) -> Path:
    """Export raw station simulations and metadata for data deposition.

    The package deliberately excludes observed discharge, which may be
    subject to redistribution restrictions. Run ``evaluate_discharge``
    before calling this method.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        run_name: Name of the GEB simulation run.

    Returns:
        Path to the completed publication-data folder.
    """
    snapped_locations: gpd.GeoDataFrame = read_geom(
        self.model.files["geom"]["discharge/discharge_snapped_locations"]
    )
    publication_folder: Path = (
        self.evaluate_discharge_output_folder / "publication_data"
    )
    run_output_folder: Path = (
        Path(self.model.config["general"]["output_folder"]) / run_name
    )
    create_discharge_publication_package(
        routing_folder=run_output_folder / "report" / "hydrology.routing",
        evaluation_metrics_xlsx=(
            self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx"
        ),
        output_folder=publication_folder,
        run_name=run_name,
        snapped_locations=snapped_locations,
    )
    self.model.logger.info(
        "Created discharge publication folder at %s.", publication_folder
    )
    return publication_folder


def create_discharge_publication_package(
    routing_folder: Path,
    evaluation_metrics_xlsx: Path,
    snapped_locations: gpd.GeoDataFrame,
    output_folder: Path,
    run_name: str,
) -> Path:
    """Collect raw station simulations and identifying metadata for publication.

    Report locations and saved evaluation locations must match the current
    snapping metadata before any publication files are replaced.

    Args:
        routing_folder: Folder containing raw station discharge reports.
        evaluation_metrics_xlsx: Station evaluation spreadsheet to include.
        snapped_locations: Gauge-to-model snapping metadata.
        output_folder: Final publication folder.
        run_name: Name of the GEB simulation run.

    Returns:
        Path to the completed publication folder.

    Raises:
        FileNotFoundError: If a required metric or simulation file is missing.
        ValueError: If there are no evaluated stations, or station identifiers
            or snapping metadata are invalid, or report/evaluation locations
            do not match the current snapping metadata.
    """
    if not evaluation_metrics_xlsx.exists():
        raise FileNotFoundError(
            f"Missing evaluation metrics: {evaluation_metrics_xlsx}"
        )
    if not routing_folder.exists():
        raise FileNotFoundError(f"Missing routing report folder: {routing_folder}")

    station_scores: pd.DataFrame = pd.read_excel(evaluation_metrics_xlsx)
    if "station_ID" not in station_scores.columns:
        raise ValueError("Evaluation metrics contain no station_ID column.")
    if station_scores.empty:
        raise ValueError("No evaluated stations are available for publication.")
    station_ids: list[str] = station_scores["station_ID"].map(_station_id_text).tolist()
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("Evaluation metrics contain duplicate station IDs.")

    snapping_df: gpd.GeoDataFrame = snapped_locations.copy()
    snapping_df.index = snapping_df.index.map(_station_id_text)
    if snapping_df.index.has_duplicates:
        raise ValueError("Snapping metadata contain duplicate station IDs.")
    missing_station_ids: set[str] = set(station_ids) - set(snapping_df.index)
    if missing_station_ids:
        raise ValueError(
            "Snapping metadata are missing "
            f"{len(missing_station_ids)} evaluated stations."
        )

    selected_stations: gpd.GeoDataFrame = snapping_df.loc[station_ids]
    for station_id in station_ids:
        report_path: Path = (
            routing_folder / f"discharge_hourly_m3_per_s_{station_id}.parquet"
        )
        if not report_path.exists():
            raise FileNotFoundError(f"Missing simulated discharge: {report_path}")
        report_metadata: dict[bytes, bytes] = pq.read_schema(report_path).metadata or {}
        report_location: dict[str, Any] = (
            json.loads(report_metadata.get(b"pandas", b"{}"))
            .get("attributes", {})
            .get("station_location", {})
        )
        station: pd.Series = selected_stations.loc[station_id]
        if (
            not report_location
            or report_location.get("pixel_xy") != list(station["snapped_grid_pixel_xy"])
            or not np.allclose(
                report_location.get("longitude_latitude", [np.nan, np.nan]),
                station["snapped_grid_pixel_lonlat"],
                rtol=0,
                atol=1e-8,
            )
            or not np.isclose(
                report_location.get("upstream_area_m2", np.nan),
                station["GEB_upstream_area_from_grid"],
                rtol=1e-6,
            )
        ):
            raise ValueError(
                f"Report location for station {station_id} does not match current "
                "snapping metadata. Rerun the simulation and discharge evaluation."
            )
    location_columns: list[str] = [
        "routing_grid_longitude",
        "routing_grid_latitude",
        "upstream_area_GEB",
    ]
    if not set(location_columns).issubset(station_scores.columns):
        raise ValueError(
            "Evaluation locations are missing. Rerun discharge evaluation."
        )
    if not np.allclose(
        station_scores[location_columns[:2]].to_numpy(dtype=float),
        np.asarray(
            selected_stations["snapped_grid_pixel_lonlat"].tolist(), dtype=float
        ),
        rtol=0,
        atol=1e-8,
    ) or not np.allclose(
        station_scores["upstream_area_GEB"].to_numpy(dtype=float),
        selected_stations["GEB_upstream_area_from_grid"].to_numpy(dtype=float),
        rtol=1e-6,
    ):
        raise ValueError(
            "Evaluation locations differ from current snapping metadata. "
            "Rerun discharge evaluation."
        )

    staging_folder: Path = output_folder.with_name(f".{output_folder.name}.building")
    if staging_folder.exists():
        shutil.rmtree(staging_folder)
    simulations_folder: Path = staging_folder / "simulations"
    simulations_folder.mkdir(parents=True)

    catalog_rows: list[dict[str, object]] = []
    for station_id in station_ids:
        station_row: pd.Series = snapping_df.loc[station_id]
        station_lon, station_lat = _coordinate_pair(
            station_row["discharge_observations_station_coords"],
            "discharge_observations_station_coords",
        )
        snapped_lon, snapped_lat = _coordinate_pair(
            station_row["snapped_grid_pixel_lonlat"],
            "snapped_grid_pixel_lonlat",
        )
        simulation_filename: str = f"discharge_hourly_m3_per_s_{station_id}.parquet"
        source_path: Path = routing_folder / simulation_filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing simulated discharge: {source_path}")
        # Preserve the reporter output byte-for-byte as the primary model result.
        shutil.copy2(source_path, simulations_folder / simulation_filename)

        station_source_value: object = station_row.get(
            "discharge_observations_source", "GRDC"
        )
        station_source: str = (
            "GRDC"
            if pd.isna(station_source_value)
            else str(station_source_value).strip()
        )
        catalog_rows.append(
            {
                "station_id": station_id,
                "station_name": " ".join(
                    str(station_row["discharge_observations_station_name"]).split()
                ),
                "station_metadata_source": station_source,
                "station_longitude": station_lon,
                "station_latitude": station_lat,
                "snapped_model_longitude": snapped_lon,
                "snapped_model_latitude": snapped_lat,
            }
        )

    pd.DataFrame(catalog_rows).to_csv(
        staging_folder / "station_catalog.csv", index=False
    )
    shutil.copy2(evaluation_metrics_xlsx, staging_folder / "evaluation_metrics.xlsx")
    (staging_folder / "README.md").write_text(
        PUBLICATION_README_TEMPLATE.format(
            station_count=len(station_ids), run_name=run_name
        ),
        encoding="utf-8",
    )

    if output_folder.exists():
        shutil.rmtree(output_folder)
    staging_folder.replace(output_folder)
    return output_folder
