"""Merge multiple GEB model cluster outputs into a single model directory."""

import logging
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geb.workflows.io import read_geom, read_table, write_geom, write_table

GEOPARQUET_FILES_TO_MERGE = [
    "input/geom/mask.geoparquet",
    "input/geom/routing/rivers.geoparquet",
    "input/geom/routing/subbasins.geoparquet",
    "input/geom/discharge/discharge_snapped_locations.geoparquet",
    "input/geom/waterbodies/waterbody_data.geoparquet",
]

PARQUET_FILES_TO_MERGE = [
    "input/table/discharge/discharge_observations_hourly.parquet",
    "input/table/discharge/discharge_observations_daily.parquet",
]


def merge_model_outputs(
    models_dir: Path,
    logger: logging.Logger,
    run_name: str = "default",
    cluster_prefix: str = "Europe",
    merged_dir_name: str = "merged",
    overwrite: bool = False,
) -> Path:
    """Merge GEB cluster outputs into one model directory for evaluation.

    Args:
        models_dir: Directory containing the cluster subdirectories.
        logger: Logger for progress messages.
        run_name: Name of the model run to merge.
        cluster_prefix: Prefix of cluster directory names (e.g. ``"Europe"``).
        merged_dir_name: Name for the merged output directory inside ``models_dir``.
        overwrite: Overwrite an existing merged directory.

    Returns:
        Path to the merged model directory.

    Raises:
        FileNotFoundError: If ``models_dir`` or an expected cluster input file is missing.
        FileExistsError: If the merged directory already exists and ``overwrite`` is ``False``.
        RuntimeError: If no clusters with the expected run output are found.
    """
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    merged_model_dir = models_dir / merged_dir_name / "base"
    if merged_model_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Merged directory already exists: {merged_model_dir}. "
                "Pass overwrite=True (or --overwrite on the CLI) to replace it."
            )
        shutil.rmtree(merged_model_dir)

    cluster_scenario_dirs: dict[str, Path] = {}
    for cluster_dir in sorted(models_dir.glob(f"{cluster_prefix}_*")):
        if not cluster_dir.is_dir():
            continue
        for scenario_dir in sorted(cluster_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            if (scenario_dir / "output" / "report" / run_name).is_dir():
                cluster_scenario_dirs[cluster_dir.name] = scenario_dir
                break

    if not cluster_scenario_dirs:
        raise RuntimeError(
            f"No clusters matching '{cluster_prefix}_*' with a '{run_name}' "
            f"output directory were found in {models_dir}."
        )
    logger.info("Merging clusters: %s", ", ".join(cluster_scenario_dirs))

    merged_input_paths: set[str] = set()

    # Report files stay in the original clusters; links avoid copying large outputs.
    symlink_count = 0
    for scenario_dir in cluster_scenario_dirs.values():
        cluster_report_dir = scenario_dir / "output" / "report" / run_name
        for report_file in sorted(cluster_report_dir.rglob("*.parquet")):
            merged_report_file = (
                merged_model_dir
                / "output"
                / "report"
                / run_name
                / report_file.relative_to(cluster_report_dir)
            )
            merged_report_file.parent.mkdir(parents=True, exist_ok=True)
            if not merged_report_file.exists() and not merged_report_file.is_symlink():
                merged_report_file.symlink_to(report_file.resolve())
                symlink_count += 1
    logger.info("[output] %d report symlink(s) created.", symlink_count)

    for relative_path in GEOPARQUET_FILES_TO_MERGE:
        cluster_geometries: list[gpd.GeoDataFrame] = []
        for cluster_name, scenario_dir in cluster_scenario_dirs.items():
            input_file = scenario_dir / relative_path
            if not input_file.exists():
                raise FileNotFoundError(
                    f"Expected '{relative_path}' in cluster '{cluster_name}', but it was not found at {input_file}."
                )
            cluster_geometries.append(read_geom(input_file))

        # Cluster geometries cover different areas, so they are stacked row-wise.
        merged_geometry = gpd.GeoDataFrame(
            pd.concat(cluster_geometries), crs=cluster_geometries[0].crs
        )
        output_file = merged_model_dir / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_geom(merged_geometry, output_file)
        merged_input_paths.add(relative_path)
        logger.info("[input] %s: %d features", relative_path, len(merged_geometry))

    for relative_path in PARQUET_FILES_TO_MERGE:
        cluster_tables: list[pd.DataFrame] = []
        for cluster_name, scenario_dir in cluster_scenario_dirs.items():
            input_file = scenario_dir / relative_path
            if not input_file.exists():
                raise FileNotFoundError(
                    f"Expected '{relative_path}' in cluster '{cluster_name}', but it was not found at {input_file}."
                )
            cluster_tables.append(read_table(input_file))

        # Station IDs are columns, so cluster tables are joined side by side.
        merged_table = pd.concat(cluster_tables, axis=1)
        output_file = merged_model_dir / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_table(merged_table, output_file)
        merged_input_paths.add(relative_path)
        logger.info("[input] %s: %d columns", relative_path, len(merged_table.columns))

    # Remaining inputs are shared build inputs, so one cluster can serve as reference.
    reference_scenario_dir = next(iter(cluster_scenario_dirs.values()))
    for input_file in sorted((reference_scenario_dir / "input").rglob("*")):
        relative_path = str(input_file.relative_to(reference_scenario_dir))
        merged_input_file = merged_model_dir / relative_path
        if (
            input_file.is_file()
            and relative_path not in merged_input_paths
            and not merged_input_file.exists()
            and not merged_input_file.is_symlink()
        ):
            merged_input_file.parent.mkdir(parents=True, exist_ok=True)
            merged_input_file.symlink_to(input_file.resolve())

    (merged_model_dir / "input" / "build_complete.txt").touch()

    merged_model_config = merged_model_dir / "model.yml"
    parent_model_config = models_dir / "model.yml"
    if parent_model_config.exists():
        merged_model_config.write_text("inherits: ../../model.yml\n")
    else:
        merged_model_config.write_text("")
        logger.warning("No parent model.yml found at %s.", parent_model_config)

    logger.info("Merged model ready at: %s", merged_model_dir)
    return merged_model_dir
