"""Merge multiple GEB model cluster outputs into a single merged model directory."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

PHASES = ["build", "spinup", "run", "evaluate"]

# GeoParquet files with spatially unique rows per cluster → concatenate row-wise.
GEOM_FILES = [
    "input/geom/mask.geoparquet",
    "input/geom/routing/rivers.geoparquet",
    "input/geom/routing/subbasins.geoparquet",
    "input/geom/discharge/discharge_snapped_locations.geoparquet",
    "input/geom/waterbodies/waterbody_data.geoparquet",
]

# Timeseries tables where each column is a unique station ID → concatenate column-wise.
TABLE_FILES_COLUMNS = [
    "input/table/discharge/discharge_observations_hourly.parquet",
    "input/table/discharge/discharge_observations_daily.parquet",
]

# Tables where each cluster contributes different rows → concatenate row-wise.
# Only include files that are actually present in at least some clusters.
TABLE_FILES_ROWS: list[str] = []


def _log_cluster_status(models_dir: Path, cluster_prefix: str) -> None:
    """Log a phase-completion table for all matching cluster directories.

    Args:
        models_dir: Directory containing the cluster subdirectories.
        cluster_prefix: Prefix used for cluster directory names.
    """
    col_w = 10
    header = f"{'Cluster':<20}" + "".join(f"  {p:<{col_w}}" for p in PHASES)
    logger.info(header)
    logger.info("-" * len(header))
    for cluster_dir in sorted(models_dir.glob(f"{cluster_prefix}_*")):
        if not cluster_dir.is_dir():
            continue
        scenario_dir = next(
            (d for d in sorted(cluster_dir.iterdir()) if d.is_dir()), None
        )
        statuses = (
            {p: (scenario_dir / f"{p}.done").exists() for p in PHASES}
            if scenario_dir
            else {p: False for p in PHASES}
        )
        cells = "".join(f"  {'done' if statuses[p] else '·':<{col_w}}" for p in PHASES)
        logger.info("%s%s", f"{cluster_dir.name:<20}", cells)


def _find_cluster_dirs(
    models_dir: Path, cluster_prefix: str, run_name: str
) -> dict[str, Path]:
    """Find cluster scenario dirs that have completed the run phase.

    Args:
        models_dir: Directory containing the cluster subdirectories.
        cluster_prefix: Prefix used for cluster directory names (e.g. ``"Europe"``).
        run_name: Name of the model run to look for inside ``output/report/<run_name>/``.

    Returns:
        Mapping of cluster name → scenario directory path, only for completed clusters.
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
                break
    return result


def _collect_frames(cluster_bases: dict[str, Path], rel_path: str) -> list:
    """Load a file from each cluster, skipping missing ones with a warning.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        rel_path: Path to the file relative to each scenario base directory.

    Returns:
        List of loaded DataFrames or GeoDataFrames. Empty if none found.
    """
    read = gpd.read_parquet if rel_path.endswith(".geoparquet") else pd.read_parquet
    frames = []
    for cluster_name, base_dir in cluster_bases.items():
        src = base_dir / rel_path
        if not src.exists():
            logger.warning("%s: %s not found, skipping.", cluster_name, rel_path)
            continue
        frames.append(read(src))
    return frames


def _merge_and_write(
    cluster_bases: dict[str, Path], rel_path: str, dest: Path, axis: int = 0
) -> None:
    """Merge a file from all clusters and write the result to dest.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        rel_path: Path to the file relative to each scenario base directory.
        dest: Output path for the merged file.
        axis: Concat axis — 0 for row-wise, 1 for column-wise.
    """
    frames = _collect_frames(cluster_bases, rel_path)
    if not frames:
        logger.warning("  %s → not found in any cluster, skipping.", rel_path)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        merged = pd.concat(frames, axis=axis)

    if axis == 1:
        # Warn about and drop duplicate station columns (same gauge in >1 cluster).
        dupes = merged.columns[merged.columns.duplicated()].tolist()
        if dupes:
            logger.warning(
                "Duplicate columns in '%s', keeping first: %s", rel_path, dupes
            )
            merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]

    dest.parent.mkdir(parents=True, exist_ok=True)
    if rel_path.endswith(".geoparquet"):
        merged = gpd.GeoDataFrame(merged, crs=frames[0].crs)
        merged.to_parquet(dest)
        logger.info(
            "  %s → %d features from %d cluster(s)", rel_path, len(merged), len(frames)
        )
    else:
        merged.to_parquet(dest)
        logger.info(
            "  %s → %d rows, %d cols from %d cluster(s)",
            rel_path,
            len(merged),
            len(merged.columns),
            len(frames),
        )


def _symlink_output_reports(
    cluster_bases: dict[str, Path], run_name: str, merged_base: Path
) -> int:
    """Symlink all report parquet files from every cluster into the merged directory.

    Station IDs are globally unique across clusters so there are no filename collisions.

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
            dest_file = dest_run_dir / src_file.relative_to(src_run_dir)
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()
            dest_file.symlink_to(src_file.resolve())
            cluster_links += 1
        logger.info("  %s: %d report files symlinked", cluster_name, cluster_links)
        n_links += cluster_links
    return n_links


def _symlink_remaining_inputs(
    cluster_bases: dict[str, Path], merged_base: Path, already_merged: set[str]
) -> None:
    """Symlink input files not already merged from the first available cluster.

    These files (forcing grids, lookup tables, rasters, etc.) are spatially
    identical across clusters, so we only need one copy.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        merged_base: Root of the merged model directory.
        already_merged: Relative paths already written by a merge step — skip these.
    """
    first_base = next(iter(cluster_bases.values()))
    for src in sorted((first_base / "input").rglob("*")):
        if not src.is_file():
            continue
        rel = str(src.relative_to(first_base))
        if rel in already_merged:
            continue
        dest = merged_base / rel
        if dest.exists() or dest.is_symlink():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(src.resolve())


def merge_model_outputs(
    models_dir: Path,
    run_name: str = "default",
    cluster_prefix: str = "Europe",
    merged_dir_name: str = "merged",
    overwrite: bool = False,
) -> Path:
    """Merge outputs from multiple GEB model clusters into a single merged model.

    Creates ``<models_dir>/<merged_dir_name>/`` with the same layout as any
    individual cluster scenario directory so it can be passed to ``geb evaluate``.

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

    merged_base = models_dir / merged_dir_name / "base"
    if merged_base.exists() and not overwrite:
        raise FileExistsError(
            f"Merged directory already exists: {merged_base}. "
            "Pass overwrite=True (or --overwrite on the CLI) to replace it."
        )

    logger.info("Cluster phase status in %s:", models_dir)
    _log_cluster_status(models_dir, cluster_prefix)

    logger.info("Scanning for clusters with run.done and '%s' output …", run_name)
    cluster_bases = _find_cluster_dirs(models_dir, cluster_prefix, run_name)
    if not cluster_bases:
        raise RuntimeError(
            f"No clusters matching '{cluster_prefix}_*' with 'run.done' and a "
            f"'{run_name}' output directory were found in {models_dir}."
        )
    logger.info(
        "Merging %d cluster(s): %s", len(cluster_bases), ", ".join(cluster_bases)
    )

    merged_paths: set[str] = set()

    logger.info("[output] Symlinking report parquets …")
    n_links = _symlink_output_reports(cluster_bases, run_name, merged_base)
    logger.info("[output] %d symlink(s) created.", n_links)

    logger.info("[input] Merging geometry files …")
    for rel_path in GEOM_FILES:
        _merge_and_write(cluster_bases, rel_path, merged_base / rel_path, axis=0)
        merged_paths.add(rel_path)

    logger.info("[input] Merging station tables (column-wise) …")
    for rel_path in TABLE_FILES_COLUMNS:
        _merge_and_write(cluster_bases, rel_path, merged_base / rel_path, axis=1)
        merged_paths.add(rel_path)

    logger.info("[input] Merging waterbody tables (row-wise) …")
    for rel_path in TABLE_FILES_ROWS:
        _merge_and_write(cluster_bases, rel_path, merged_base / rel_path, axis=0)
        merged_paths.add(rel_path)

    logger.info("[input] Symlinking remaining input files …")
    _symlink_remaining_inputs(cluster_bases, merged_base, merged_paths)

    # Sentinel required by GEBModel.verify_build_complete().
    (merged_base / "input" / "build_complete.txt").touch()

    # Minimal config that inherits shared settings from the parent directory.
    model_yml = merged_base / "model.yml"
    model_yml.parent.mkdir(parents=True, exist_ok=True)
    model_yml.write_text("inherits: ../../model.yml\n")

    logger.info("Merged model ready at: %s", merged_base)
    return merged_base
