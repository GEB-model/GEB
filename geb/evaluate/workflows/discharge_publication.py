"""Create a publication-ready folder of simulated station discharge."""

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


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


def _write_readme(output_path: Path, station_count: int, run_name: str) -> None:
    """Document the contents and data-use constraints of the export.

    Args:
        output_path: README file to create.
        station_count: Number of exported stations.
        run_name: GEB simulation run name.
    """
    readme_text: str = f"""# GEB station discharge simulations

This folder contains raw GEB simulated discharge for {station_count} gauging
stations from run `{run_name}`.

## Contents

- `station_catalog.csv`: station identity and source, original gauge location,
  and the exact snapped model-cell location.
- `simulations/*.parquet`: the original hourly GEB reporter file for each
  station. Files are named `discharge_hourly_m3_per_s_<station_id>.parquet` and
  contain a `discharge_hourly_m3_per_s_<station_id>` column and datetime index.
- `evaluation_metrics.xlsx`: derived station-level evaluation results.

Observed discharge is deliberately excluded. GRDC does not permit downloaded
observations to be redistributed to third parties or via the internet.
Authorized observations are available directly from the GRDC Data Portal:
https://grdc.bafg.de/data/data_portal/.

Coordinates use WGS 84 longitude/latitude (`EPSG:4326`). Discharge is in cubic
metres per second (`m3 s-1`). Simulations are unmodified reporter values: no
observation-based upstream-area correction or temporal resampling is applied.
"""
    output_path.write_text(readme_text, encoding="utf-8")


def create_discharge_publication_package(
    routing_folder: Path,
    evaluation_metrics_xlsx: Path,
    snapped_locations: gpd.GeoDataFrame,
    output_folder: Path,
    run_name: str,
) -> Path:
    """Collect raw station simulations and identifying metadata for publication.

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
        ValueError: If station identifiers or snapping metadata are invalid.
    """
    if not evaluation_metrics_xlsx.exists():
        raise FileNotFoundError(
            f"Missing evaluation metrics: {evaluation_metrics_xlsx}"
        )
    if not routing_folder.exists():
        raise FileNotFoundError(f"Missing routing report folder: {routing_folder}")

    evaluation_df: pd.DataFrame = pd.read_excel(evaluation_metrics_xlsx)
    if "station_ID" not in evaluation_df.columns:
        raise ValueError("Evaluation metrics contain no station_ID column.")
    station_ids: list[str] = evaluation_df["station_ID"].map(_station_id_text).tolist()
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
    _write_readme(staging_folder / "README.md", len(station_ids), run_name)

    if output_folder.exists():
        shutil.rmtree(output_folder)
    staging_folder.replace(output_folder)
    return output_folder
