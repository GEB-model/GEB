"""Merge multiple GEB model cluster outputs into a single merged model directory."""

from __future__ import annotations

import logging
import shutil
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

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


def _find_cluster_dirs(
    models_dir: Path, cluster_prefix: str, run_name: str, logger: logging.Logger
) -> dict[str, Path]:
    """Return cluster name → scenario dir for every cluster that has a run output directory.

    Args:
        models_dir: Directory containing the cluster subdirectories.
        cluster_prefix: Prefix used for cluster directory names (e.g. ``"Europe"``).
        run_name: Name of the run to look for inside ``output/report/<run_name>/``.
        logger: Logger for progress messages.

    Returns:
        Mapping of cluster name → scenario directory path.
    """
    result: dict[str, Path] = {}
    for cluster_dir in sorted(models_dir.glob(f"{cluster_prefix}_*")):
        if not cluster_dir.is_dir():
            continue
        for scenario_dir in sorted(cluster_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            run_dir = scenario_dir / "output" / "report" / run_name
            if run_dir.is_dir():
                result[cluster_dir.name] = scenario_dir
                break
            else:
                logger.debug(
                    "%s: run output directory not found at %s, skipping.",
                    cluster_dir.name,
                    run_dir,
                )
    return result


def _collect_frames(
    cluster_bases: dict[str, Path], rel_path: str, logger: logging.Logger
) -> list:
    """Load ``rel_path`` from each cluster, skipping missing ones with a warning.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        rel_path: Path to the file relative to each scenario base directory.
        logger: Logger for progress messages.

    Returns:
        List of loaded DataFrames / GeoDataFrames; empty if none found.
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
    cluster_bases: dict[str, Path],
    rel_path: str,
    dest: Path,
    logger: logging.Logger,
    axis: int = 0,
) -> None:
    """Concatenate ``rel_path`` from all clusters and write to ``dest``.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        rel_path: Path to the file relative to each scenario base directory.
        dest: Output path for the merged file.
        logger: Logger for progress messages.
        axis: Concat axis — 0 row-wise, 1 column-wise.
    """
    frames = _collect_frames(cluster_bases, rel_path, logger)
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
    cluster_bases: dict[str, Path],
    run_name: str,
    merged_base: Path,
    logger: logging.Logger,
) -> int:
    """Symlink every report parquet from each cluster into the merged directory.

    Args:
        cluster_bases: Mapping of cluster name → scenario base directory.
        run_name: Name of the model run (e.g. ``"default"``).
        merged_base: Root of the merged model directory.
        logger: Logger for progress messages.

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
                continue
            dest_file.symlink_to(src_file.resolve())
            cluster_links += 1
        logger.info("  %s: %d report files symlinked", cluster_name, cluster_links)
        n_links += cluster_links
    return n_links


def _symlink_remaining_inputs(
    cluster_bases: dict[str, Path], merged_base: Path, already_merged: set[str]
) -> None:
    """Symlink input files from the first cluster that haven't been merged yet.

    Forcing grids, rasters, and lookup tables are identical across clusters,
    so one symlinked copy is sufficient.

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
    logger: logging.Logger | None = None,
) -> Path:
    """Merge GEB cluster outputs into a single model directory for evaluation.

    Creates ``<models_dir>/<merged_dir_name>/base/`` with the same layout as a
    normal scenario directory so it can be passed directly to ``geb evaluate``.

    Args:
        models_dir: Directory containing the cluster subdirectories.
        run_name: Name of the model run to merge.
        cluster_prefix: Prefix of cluster directory names (e.g. ``"Europe"``).
        merged_dir_name: Name for the merged output directory inside ``models_dir``.
        overwrite: Overwrite an existing merged directory.
        logger: Logger for progress messages. Defaults to the module logger.

    Returns:
        Path to the merged model base directory.

    Raises:
        FileNotFoundError: If ``models_dir`` does not exist.
        FileExistsError: If the merged directory already exists and ``overwrite`` is ``False``.
        RuntimeError: If no clusters with the expected run output are found.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    merged_base = models_dir / merged_dir_name / "base"
    if merged_base.exists():
        if not overwrite:
            raise FileExistsError(
                f"Merged directory already exists: {merged_base}. "
                "Pass overwrite=True (or --overwrite on the CLI) to replace it."
            )
        shutil.rmtree(merged_base)

    logger.info(
        "Scanning for clusters with '%s' output directory in %s …", run_name, models_dir
    )
    cluster_bases = _find_cluster_dirs(models_dir, cluster_prefix, run_name, logger)
    if not cluster_bases:
        raise RuntimeError(
            f"No clusters matching '{cluster_prefix}_*' with a '{run_name}' "
            f"output directory were found in {models_dir}."
        )
    logger.info(
        "Merging %d cluster(s): %s", len(cluster_bases), ", ".join(cluster_bases)
    )

    merged_paths: set[str] = set()

    logger.info("[output] Symlinking report parquets …")
    n_links = _symlink_output_reports(cluster_bases, run_name, merged_base, logger)
    logger.info("[output] %d symlink(s) created.", n_links)

    logger.info("[input] Merging geometry files …")
    for rel_path in GEOM_FILES:
        _merge_and_write(
            cluster_bases, rel_path, merged_base / rel_path, logger, axis=0
        )
        merged_paths.add(rel_path)

    logger.info("[input] Merging station tables (column-wise) …")
    for rel_path in TABLE_FILES_COLUMNS:
        _merge_and_write(
            cluster_bases, rel_path, merged_base / rel_path, logger, axis=1
        )
        merged_paths.add(rel_path)

    logger.info("[input] Symlinking remaining input files …")
    _symlink_remaining_inputs(cluster_bases, merged_base, merged_paths)

    # Sentinel required by GEBModel.verify_build_complete().
    (merged_base / "input" / "build_complete.txt").touch()

    # Inherit from the shared parent config when available (common but not guaranteed).
    model_yml = merged_base / "model.yml"
    model_yml.parent.mkdir(parents=True, exist_ok=True)
    parent_config = models_dir / "model.yml"
    if parent_config.exists():
        model_yml.write_text("inherits: ../../model.yml\n")
    else:
        model_yml.write_text("")
        logger.warning(
            "No parent model.yml found at %s; wrote an empty model.yml — "
            "provide your own configuration before running 'geb evaluate'.",
            parent_config,
        )

    logger.info("Merged model ready at: %s", merged_base)
    return merged_base
