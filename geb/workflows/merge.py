"""Merge multiple GEB model cluster outputs into a single merged model directory.

The merged directory has the same layout as any individual cluster scenario
directory (``input/``, ``output/``, ``model.yml``), so it can be passed
directly to ``geb evaluate`` without any special multi-region handling.

Merged items
------------
* **output report subdirectories** – all parquet files found under
  ``output/report/<run_name>/`` are symlinked (zero copy) into the merged
  directory. This covers ``hydrology.routing`` (per-station discharge) as
  well as any other reporter subdirectory that may be present, making the
  merge step forward-compatible with future reporters.
* **input geometry** – rivers, catchment mask, and discharge-station
  geometries are concatenated across clusters (GeoParquet).
* **input tables** – discharge observation parquets are concatenated along
  the station axis (columns), preserving the DatetimeIndex.
* **model.yml** – a minimal config that inherits from the parent
  ``<models_dir>/model.yml``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# Relative paths (from cluster scenario base dir) for geometry files that
# must be concatenated across clusters for discharge evaluation.
_GEOM_FILES_TO_MERGE: list[str] = [
    "input/geom/routing/rivers.geoparquet",
    "input/geom/mask.geoparquet",
    "input/geom/discharge/discharge_snapped_locations.geoparquet",
]

# Relative paths for tabular files whose stations (columns) should be merged.
# Each file uses a DatetimeIndex for rows and station IDs for columns.
_TABLE_FILES_TO_MERGE: list[str] = [
    "input/table/discharge/discharge_observations_hourly.parquet",
    "input/table/discharge/discharge_observations_daily.parquet",
]

# Ordered pipeline phases; presence of ``<phase>.done`` in the scenario dir marks
# completion.  Used for diagnostic status logging before the merge.
PHASES: list[str] = ["build", "spinup", "run", "evaluate"]


def _get_phase_status(scenario_dir: Path) -> dict[str, bool]:
    """Return completion status for each pipeline phase of a cluster.

    Args:
        scenario_dir: Scenario subdirectory of a cluster (e.g.
            ``.../Europe_004/base``), where ``<phase>.done`` sentinels are
            written.

    Returns:
        Mapping of phase name → ``True`` if the ``<phase>.done`` sentinel
        file exists, ``False`` otherwise.
    """
    return {phase: (scenario_dir / f"{phase}.done").exists() for phase in PHASES}


def _log_cluster_status_table(models_dir: Path, cluster_prefix: str) -> None:
    """Log a formatted phase-completion table for all matching cluster dirs.

    Scans all ``<cluster_prefix>_*`` subdirectories and logs one row per
    cluster showing which pipeline phases have completed.  Useful as a
    pre-merge diagnostic so the user can see which clusters will be included.

    Args:
        models_dir: Directory containing the cluster subdirectories.
        cluster_prefix: Prefix used for cluster directory names.
    """
    rows: dict[str, dict[str, bool]] = {}
    for cluster_dir in sorted(models_dir.glob(f"{cluster_prefix}_*")):
        if not cluster_dir.is_dir():
            continue
        # Pick the first scenario subdirectory found (typically 'base').
        scenario_dir = next(
            (d for d in sorted(cluster_dir.iterdir()) if d.is_dir()), None
        )
        rows[cluster_dir.name] = (
            _get_phase_status(scenario_dir)
            if scenario_dir is not None
            else {p: False for p in PHASES}
        )

    if not rows:
        return

    col_w = 10
    header = f"{'Cluster':<20}" + "".join(f"  {p:<{col_w}}" for p in PHASES)
    logger.info(header)
    logger.info("-" * len(header))
    for cluster, statuses in rows.items():
        cells = "".join(f"  {'done' if statuses[p] else '·':<{col_w}}" for p in PHASES)
        logger.info("%s%s", f"{cluster:<20}", cells)


def _find_cluster_base_dirs(
    models_dir: Path,
    cluster_prefix: str,
    run_name: str,
) -> dict[str, Path]:
    """Return scenario base dirs for clusters that have completed the run phase.

    A cluster is considered complete when its ``run.done`` sentinel file
    exists and at least one output report subdirectory is present under
    ``output/report/<run_name>/``.  The scenario subdirectory (e.g. ``base``)
    is discovered dynamically so that it does not need to be hardcoded.

    Args:
        models_dir: Directory containing the cluster subdirectories.
        cluster_prefix: Prefix used for cluster directory names (e.g.
            ``"Europe"``).
        run_name: Name of the model run to look for inside
            ``output/report/<run_name>/``.

    Returns:
        Mapping of cluster name → scenario directory path, only for clusters
        whose ``run.done`` sentinel file and output run directory both exist.
    """
    result: dict[str, Path] = {}
    for cluster_dir in sorted(models_dir.glob(f"{cluster_prefix}_*")):
        if not cluster_dir.is_dir():
            continue
        for scenario_dir in sorted(cluster_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            run_dir = scenario_dir / "output" / "report" / run_name
            if (scenario_dir / "run.done").exists() and run_dir.is_dir():
                result[cluster_dir.name] = scenario_dir
                break  # take the first matching scenario per cluster
    return result


def _merge_geoparquets(
    cluster_bases: dict[str, Path],
    relative_path: str,
    dest_path: Path,
) -> None:
    """Concatenate a GeoParquet file from each cluster and write the result.

    Clusters that are missing the file are skipped with a warning so that a
    single missing geometry does not abort the entire merge.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        relative_path: Path to the GeoParquet file relative to the base
            directory (e.g. ``"input/geom/routing/rivers.geoparquet"``).
        dest_path: Output path for the merged GeoParquet file.

    Raises:
        RuntimeError: If no source files are found for the given path across
            all clusters.
    """
    frames: list[gpd.GeoDataFrame] = []
    for cluster_name, base_dir in cluster_bases.items():
        src = base_dir / relative_path
        if not src.exists():
            logger.warning(
                "%s: %s not found, skipping cluster.", cluster_name, relative_path
            )
            continue
        frames.append(gpd.read_parquet(src))

    if not frames:
        raise RuntimeError(
            f"No source files found for '{relative_path}' across any cluster."
        )

    merged: gpd.GeoDataFrame = gpd.GeoDataFrame(pd.concat(frames), crs=frames[0].crs)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(dest_path)


def _merge_table_parquets(
    cluster_bases: dict[str, Path],
    relative_path: str,
    dest_path: Path,
) -> None:
    """Concatenate a tabular parquet across clusters along the column (station) axis.

    Each cluster's parquet is expected to have a DatetimeIndex (rows) and one
    column per station. Concatenating along axis=1 produces a merged table
    with all stations while preserving the time index.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        relative_path: Path to the parquet file relative to the base
            directory (e.g.
            ``"input/table/discharge/discharge_observations_hourly.parquet"``).
        dest_path: Output path for the merged parquet file.

    Raises:
        RuntimeError: If no source files are found for the given path across
            all clusters.
    """
    frames: list[pd.DataFrame] = []
    for cluster_name, base_dir in cluster_bases.items():
        src = base_dir / relative_path
        if not src.exists():
            logger.warning(
                "%s: %s not found, skipping cluster.", cluster_name, relative_path
            )
            continue
        frames.append(pd.read_parquet(src))

    if not frames:
        raise RuntimeError(
            f"No source files found for '{relative_path}' across any cluster."
        )

    # Stations are columns; concat on axis=1 to get a single wide table.
    merged_df: pd.DataFrame = pd.concat(frames, axis=1)
    duplicate_cols = merged_df.columns[merged_df.columns.duplicated()].tolist()
    if duplicate_cols:
        logger.warning(
            "Duplicate station columns found in '%s' (appear in >1 cluster): %s. "
            "Keeping the first occurrence.",
            relative_path,
            duplicate_cols,
        )
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep="first")]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(dest_path)


def _symlink_output_report_files(
    cluster_bases: dict[str, Path],
    run_name: str,
    merged_base: Path,
) -> int:
    """Symlink all parquet files from every report subdirectory into the merged dir.

    All subdirectories found under ``output/report/<run_name>/`` in any
    cluster are included, so the merge is forward-compatible with reporter
    modules added in the future (e.g. ``hydrology.landsurface``,
    ``hydrology.water_demand``).  Using symlinks avoids duplicating
    potentially large time-series data.

    Station IDs are globally unique across clusters, so there are no
    filename collisions between clusters.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        run_name: Name of the model run (e.g. ``"default"``).
        merged_base: Root of the merged model directory.

    Returns:
        Total number of symlinks created.
    """
    dest_run_dir = merged_base / "output" / "report" / run_name

    n_links = 0
    for base_dir in cluster_bases.values():
        src_run_dir = base_dir / "output" / "report" / run_name
        for src_file in sorted(src_run_dir.rglob("*.parquet")):
            # Preserve the subdirectory structure relative to the run dir
            # (e.g. hydrology.routing/discharge_hourly_m3_per_s_12345.parquet).
            rel = src_file.relative_to(src_run_dir)
            dest_file = dest_run_dir / rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()
            # Use absolute path so the symlink remains valid regardless of
            # the working directory from which the merged model is accessed.
            os.symlink(src_file.resolve(), dest_file)
            n_links += 1

    return n_links


def merge_model_outputs(
    models_dir: Path,
    run_name: str = "default",
    cluster_prefix: str = "Europe",
    merged_dir_name: str = "merged",
    overwrite: bool = False,
) -> Path:
    """Merge outputs from multiple GEB model clusters into a single merged model.

    Creates ``<models_dir>/<merged_dir_name>/`` with the same directory
    layout as any individual cluster scenario directory so it can be passed
    directly to ``geb evaluate``.

    Merged contents
    ---------------
    * ``output/report/<run_name>/`` — symlinks to all parquet files from
      every cluster's report directory (all reporter subdirectories,
      including ``hydrology.routing`` and any future ones).
    * ``input/geom/routing/rivers.geoparquet`` — merged river network.
    * ``input/geom/mask.geoparquet`` — merged catchment mask.
    * ``input/geom/discharge/discharge_snapped_locations.geoparquet`` —
      merged discharge station locations.
    * ``input/table/discharge/discharge_observations_hourly.parquet`` —
      merged observation table (all stations as columns).
    * ``input/table/discharge/discharge_observations_daily.parquet`` —
      merged observation table (all stations as columns).
    * ``model.yml`` — minimal config that inherits from
      ``<models_dir>/model.yml``.

    Args:
        models_dir: Directory containing the cluster subdirectories (e.g.
            ``/path/to/models/large_scale6``).
        run_name: Name of the model run whose outputs are merged (e.g.
            ``"default"``).
        cluster_prefix: Prefix used for cluster directory names (e.g.
            ``"Europe"`` matches ``Europe_000``, ``Europe_001``, …).
        merged_dir_name: Name for the merged directory inside
            ``models_dir``. Defaults to ``"merged"``.
        overwrite: If ``True``, overwrite an existing merged directory.

    Returns:
        Path to the created merged model directory.

    Raises:
        FileNotFoundError: If ``models_dir`` does not exist.
        FileExistsError: If the merged directory already exists and
            ``overwrite`` is ``False``.
        RuntimeError: If no completed clusters are found.
    """
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    merged_base = models_dir / merged_dir_name

    if merged_base.exists() and not overwrite:
        raise FileExistsError(
            f"Merged directory already exists: {merged_base}. "
            "Pass overwrite=True (or --overwrite on the CLI) to replace it."
        )

    logger.info("Cluster phase status in %s:", models_dir)
    _log_cluster_status_table(models_dir, cluster_prefix)

    logger.info("Scanning for clusters ready to merge (run.done present) …")
    cluster_bases = _find_cluster_base_dirs(models_dir, cluster_prefix, run_name)

    if not cluster_bases:
        raise RuntimeError(
            f"No clusters matching '{cluster_prefix}_*' with 'run.done' and a "
            f"'{run_name}' output run directory were found in {models_dir}."
        )

    logger.info(
        "Found %d completed cluster(s): %s",
        len(cluster_bases),
        ", ".join(cluster_bases),
    )

    # Output: symlink all parquet files under every report subdirectory.
    logger.info("Symlinking output report parquets …")
    n_links = _symlink_output_report_files(cluster_bases, run_name, merged_base)
    logger.info("Created %d symlink(s).", n_links)

    # Input geometry: rivers, catchment mask, station locations.
    logger.info("Merging input geometry files …")
    for rel_path in _GEOM_FILES_TO_MERGE:
        dest = merged_base / rel_path
        logger.debug("  %s", rel_path)
        _merge_geoparquets(cluster_bases, rel_path, dest)

    # Input tables: discharge observations (stations are columns).
    logger.info("Merging input observation tables …")
    for rel_path in _TABLE_FILES_TO_MERGE:
        dest = merged_base / rel_path
        logger.debug("  %s", rel_path)
        _merge_table_parquets(cluster_bases, rel_path, dest)

    # Write a minimal model.yml that inherits from the parent config so that
    # all shared settings (CRS, time range, etc.) are picked up automatically.
    model_yml = merged_base / "model.yml"
    model_yml.parent.mkdir(parents=True, exist_ok=True)
    model_yml.write_text("inherits: ../model.yml\n")
    logger.info("Wrote %s", model_yml)

    logger.info("Merged model ready at: %s", merged_base)
    return merged_base
