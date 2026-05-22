"""Merge multiple GEB model cluster outputs into a single merged model directory.

The merged directory mirrors the layout of an individual cluster scenario
directory and can be passed directly to ``geb evaluate``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# Geometry files (GeoParquet) to concatenate across clusters.
_GEOM_FILES_TO_MERGE: list[str] = [
    "input/geom/routing/rivers.geoparquet",
    "input/geom/mask.geoparquet",
    "input/geom/discharge/discharge_snapped_locations.geoparquet",
]

# Tabular observation files to merge along the station (column) axis.
# Each file has a DatetimeIndex for rows and one column per station.
_TABLE_FILES_TO_MERGE: list[str] = [
    "input/table/discharge/discharge_observations_hourly.parquet",
    "input/table/discharge/discharge_observations_daily.parquet",
]

# Ordered pipeline phases; ``<phase>.done`` sentinels mark completion.
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
    """Concatenate a GeoParquet file from each cluster and write the merged result.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        relative_path: Path to the GeoParquet file relative to the base directory.
        dest_path: Output path for the merged GeoParquet file.

    Raises:
        RuntimeError: If no source files are found across any cluster.
    """
    frames: list[gpd.GeoDataFrame] = []
    for cluster_name, base_dir in cluster_bases.items():
        src = base_dir / relative_path
        if not src.exists():
            logger.warning("%s: %s not found, skipping.", cluster_name, relative_path)
            continue
        frames.append(gpd.read_parquet(src))

    if not frames:
        raise RuntimeError(
            f"No source files found for '{relative_path}' across any cluster."
        )

    with warnings.catch_warnings():
        # Pandas raises a FutureWarning when concatenating frames with mixed-dtype
        # columns; harmless here since we control the output schema.
        warnings.simplefilter("ignore", FutureWarning)
        merged = gpd.GeoDataFrame(pd.concat(frames), crs=frames[0].crs)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(dest_path)


def _merge_table_parquets(
    cluster_bases: dict[str, Path],
    relative_path: str,
    dest_path: Path,
) -> None:
    """Concatenate a tabular parquet from each cluster along the station (column) axis.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        relative_path: Path to the parquet file relative to the base directory.
        dest_path: Output path for the merged parquet file.

    Raises:
        RuntimeError: If no source files are found across any cluster.
    """
    frames: list[pd.DataFrame] = []
    for cluster_name, base_dir in cluster_bases.items():
        src = base_dir / relative_path
        if not src.exists():
            logger.warning("%s: %s not found, skipping.", cluster_name, relative_path)
            continue
        frames.append(pd.read_parquet(src))

    if not frames:
        raise RuntimeError(
            f"No source files found for '{relative_path}' across any cluster."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        merged_df = pd.concat(frames, axis=1)
    duplicate_cols = merged_df.columns[merged_df.columns.duplicated()].tolist()
    if duplicate_cols:
        logger.warning(
            "Duplicate station columns in '%s' (appear in >1 cluster): %s. Keeping first.",
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
    """Symlink all parquet files from every cluster's report subdirectory into the merged dir.

    Uses absolute symlinks so the merged directory remains valid regardless of
    working directory. Station IDs are globally unique, so there are no filename
    collisions between clusters.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        run_name: Name of the model run (e.g. ``"default"``).
        merged_base: Root of the merged model directory.

    Returns:
        Total number of symlinks created.
    """
    dest_run_dir = merged_base / "output" / "report" / run_name

    n_links = 0
    for cluster_name, base_dir in cluster_bases.items():
        src_run_dir = base_dir / "output" / "report" / run_name
        cluster_links = 0
        for src_file in sorted(src_run_dir.rglob("*.parquet")):
            rel = src_file.relative_to(src_run_dir)
            dest_file = dest_run_dir / rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()
            dest_file.symlink_to(src_file.resolve())
            cluster_links += 1
        logger.info(
            "  output/report/%s: %d files symlinked from %s",
            run_name,
            cluster_links,
            cluster_name,
        )
        n_links += cluster_links

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

    Args:
        models_dir: Directory containing the cluster subdirectories.
        run_name: Name of the model run whose outputs are merged.
        cluster_prefix: Prefix used for cluster directory names (e.g. ``"Europe"``).
        merged_dir_name: Name for the merged directory inside ``models_dir``.
        overwrite: If ``True``, overwrite an existing merged directory.

    Returns:
        Path to the created merged model base directory.

    Raises:
        FileNotFoundError: If ``models_dir`` does not exist.
        FileExistsError: If the merged directory already exists and ``overwrite`` is ``False``.
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

    logger.info("Scanning for clusters with run.done and '%s' output …", run_name)
    cluster_bases = _find_cluster_base_dirs(models_dir, cluster_prefix, run_name)

    if not cluster_bases:
        raise RuntimeError(
            f"No clusters matching '{cluster_prefix}_*' with 'run.done' and a "
            f"'{run_name}' output run directory were found in {models_dir}."
        )

    logger.info(
        "Merging %d cluster(s): %s",
        len(cluster_bases),
        ", ".join(cluster_bases),
    )

    # Output: symlink all parquet files under every report subdirectory.
    logger.info("[output] Symlinking report parquets …")
    n_links = _symlink_output_report_files(cluster_bases, run_name, merged_base)
    logger.info("[output] %d symlink(s) created.", n_links)

    # Input geometry: rivers, catchment mask, station locations.
    logger.info("[input] Merging geometry files …")
    for rel_path in _GEOM_FILES_TO_MERGE:
        dest = merged_base / rel_path
        logger.info("  %s", rel_path)
        _merge_geoparquets(cluster_bases, rel_path, dest)

    # Input tables: discharge observations (stations are columns).
    logger.info("[input] Merging observation tables …")
    for rel_path in _TABLE_FILES_TO_MERGE:
        dest = merged_base / rel_path
        logger.info("  %s", rel_path)
        _merge_table_parquets(cluster_bases, rel_path, dest)

    # input/files.yml — required by the runner to discover all registered files.
    # Copy from the first available cluster; all clusters share the same schema.
    first_cluster_name, first_base = next(iter(cluster_bases.items()))
    src_files_yml = first_base / "input" / "files.yml"
    dest_files_yml = merged_base / "input" / "files.yml"
    dest_files_yml.parent.mkdir(parents=True, exist_ok=True)
    dest_files_yml.write_bytes(src_files_yml.read_bytes())
    logger.info("[input] Copied input/files.yml from %s.", first_cluster_name)

    # input/build_complete.txt — sentinel required by GEBModel.verify_build_complete().
    build_complete = merged_base / "input" / "build_complete.txt"
    build_complete.touch()
    logger.info("[input] Created input/build_complete.txt.")

    # model.yml — minimal config that inherits shared settings from the parent.
    model_yml = merged_base / "model.yml"
    model_yml.parent.mkdir(parents=True, exist_ok=True)
    model_yml.write_text("inherits: ../model.yml\n")
    logger.info("Wrote %s", model_yml)

    logger.info("Merged model ready at: %s", merged_base)
    return merged_base
