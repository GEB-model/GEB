"""Load external discharge scores and align them with GEB stations.

Standalone exports filter external tables to all known GEB station keys.
Scientific comparison plots use pairwise matching after applying upstream-area
thresholds, so every plotted GEB value has an aligned external value.
"""

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geb.workflows.io import read_geom

# Download manually from https://zenodo.org/records/6390219.
EXTERNAL_EVALUATION_FOLDER_NAME: str = "external_evaluation_data"
UTRECHT_EVALUATION_FILE_NAME: str = "Utrecht_1KM_daily_discharge.csv"
# Download metrics.tgz manually from https://zenodo.org/records/10397664 and
# save it under this fixed name in the external evaluation folder.
EXTERNAL_METRICS_ARCHIVE_FILE_NAME: str = "google_streamflow_metrics.tgz"
GOOGLE_MODEL_NAME: str = "Google Streamflow"
GLOFAS_MODEL_NAME: str = "GloFAS"
UTRECHT_MODEL_NAME: str = "Utrecht"
GOOGLE_METRIC_ROOT: Path = Path(
    "metrics/hydrograph_metrics/per_metric/google/2014/dual_lstm/"
    "hydrologically_separated"
)
GLOFAS_METRIC_ROOT: Path = Path(
    "metrics/hydrograph_metrics/per_metric/glofas/2014/glofas_prediction"
)
ARCHIVE_MODEL_METRIC_ROOTS: dict[str, Path] = {
    GOOGLE_MODEL_NAME: GOOGLE_METRIC_ROOT,
    GLOFAS_MODEL_NAME: GLOFAS_METRIC_ROOT,
}
ARCHIVE_METRIC_FILES: dict[str, str] = {
    "KGE": "KGE.csv",
    "NSE": "NSE.csv",
    "KGE_correlation": "Pearson-r.csv",
    "KGE_bias_ratio": "Beta-KGE.csv",
    "KGE_variability_ratio": "Alpha-NSE.csv",
}
ARCHIVE_LEAD_TIME_COLUMN: str = "0"


def _get_external_evaluation_folder(input_folder: Path) -> Path:
    """Get the external score folder for ordinary and merged model layouts.

    Args:
        input_folder: Model input folder.

    Returns:
        Folder containing optional external evaluation data.
    """
    resolved_input_folder: Path = input_folder.resolve()
    model_folder: Path = resolved_input_folder.parent
    if model_folder.name == "base":
        # Merged models use <models>/<merged name>/base/input, while ordinary
        # models keep external data directly beside their input folder.
        model_folder = model_folder.parents[1]
    return model_folder / EXTERNAL_EVALUATION_FOLDER_NAME


@dataclass(frozen=True)
class MatchedSkillScores:
    """GEB and external scores aligned to the same gauging stations.

    Args:
        geb: Filtered GEB skill-score table.
        external: External skill-score table in the same station order.
        minimum_upstream_area_km2: Effective GEB upstream-area threshold (km2).
    """

    geb: pd.DataFrame
    external: pd.DataFrame
    minimum_upstream_area_km2: float


def format_grdc_station_key(station_id: object) -> str | None:
    """Format one station ID as a GRDC-style key.

    Args:
        station_id: Raw station ID value.

    Returns:
        Uppercase key such as ``"GRDC_1234567"``, or `None` for missing values.
    """
    if pd.isna(station_id):
        return None
    station_id_text: str = str(station_id).strip().upper()
    if not station_id_text or station_id_text == "NAN":
        return None
    if station_id_text.startswith("GRDC_"):
        return station_id_text
    try:
        station_id_text = str(int(float(station_id_text)))
    except ValueError:
        pass
    return f"GRDC_{station_id_text}"


def _add_match_keys(table: pd.DataFrame) -> pd.DataFrame:
    """Add station-name and GRDC keys used for matching.

    Args:
        table: Skill-score table with optional ``station_name`` and
            ``station_ID`` columns.

    Returns:
        Copy of the table with ``station_name_key`` and ``station_id_key``.
    """
    keyed_table: pd.DataFrame = table.copy()
    if "station_name" in keyed_table.columns:
        keyed_table["station_name_key"] = (
            keyed_table["station_name"].fillna("").astype(str).str.strip().str.upper()
        )
    else:
        keyed_table["station_name_key"] = ""
    if "station_ID" in keyed_table.columns:
        keyed_table["station_id_key"] = (
            keyed_table["station_ID"].map(format_grdc_station_key).fillna("")
        )
    else:
        keyed_table["station_id_key"] = ""
    return keyed_table


def _read_model_metrics_from_archive(
    archive: tarfile.TarFile,
    archive_path: Path,
    metric_root: Path,
) -> pd.DataFrame:
    """Read one model's daily skill scores from the metrics archive.

    Args:
        archive: Open local metrics archive.
        archive_path: Path to ``google_streamflow_metrics.tgz``.
        metric_root: Model-specific directory inside the archive.

    Returns:
        Per-GRDC-station daily skill scores.

    Raises:
        ValueError: If an expected metric member or lead-time column is missing.
    """
    metric_series: dict[str, pd.Series] = {}
    for metric_name, metric_file_name in ARCHIVE_METRIC_FILES.items():
        member_name: str = str(metric_root / metric_file_name)
        try:
            metric_file = archive.extractfile(member_name)
        except KeyError as error:
            raise ValueError(
                f"{archive_path} is missing expected member {member_name}."
            ) from error
        if metric_file is None:
            raise ValueError(
                f"{archive_path} is missing expected member {member_name}."
            )

        metric_df: pd.DataFrame = pd.read_csv(metric_file, index_col=0)
        if ARCHIVE_LEAD_TIME_COLUMN not in metric_df.columns:
            raise ValueError(
                f"{member_name} is missing lead-time column "
                f"{ARCHIVE_LEAD_TIME_COLUMN!r}."
            )
        values: pd.Series = pd.to_numeric(
            metric_df[ARCHIVE_LEAD_TIME_COLUMN], errors="coerce"
        )
        values.name = metric_name
        metric_series[metric_name] = values

    model_df: pd.DataFrame = pd.concat(metric_series.values(), axis=1)
    model_df.index = model_df.index.map(str).str.strip().str.upper()
    model_df["R2"] = model_df["KGE_correlation"] ** 2
    return model_df.dropna(how="all")


def load_external_skill_scores(
    input_folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Read fixed local Utrecht, Google, and GloFAS skill-score files.

    Args:
        input_folder: Model input folder. For a merged model, the shared external
            data directory is in the top-level folder alongside the merged and
            cluster model folders.
        logger: Logger used for diagnostics.

    Returns:
        External skill-score tables keyed by model label.
    """
    external_evaluation_folder: Path = _get_external_evaluation_folder(input_folder)
    if not external_evaluation_folder.is_dir():
        logger.info(
            "No optional external evaluation folder found at %s; showing GEB only.",
            external_evaluation_folder,
        )
        return {}
    logger.info(
        "Reading external evaluation data from %s.",
        external_evaluation_folder,
    )

    external_models: dict[str, pd.DataFrame] = {}
    utrecht_path: Path = external_evaluation_folder / UTRECHT_EVALUATION_FILE_NAME
    if utrecht_path.exists():
        utrecht_df: pd.DataFrame = pd.read_csv(utrecht_path, index_col=0)
        utrecht_df.index = utrecht_df.index.map(str).str.strip().str.upper()
        external_models[UTRECHT_MODEL_NAME] = utrecht_df

    metrics_archive_path: Path = (
        external_evaluation_folder / EXTERNAL_METRICS_ARCHIVE_FILE_NAME
    )
    if metrics_archive_path.exists():
        with tarfile.open(metrics_archive_path, mode="r:gz") as archive:
            external_models.update(
                {
                    model_name: _read_model_metrics_from_archive(
                        archive=archive,
                        archive_path=metrics_archive_path,
                        metric_root=metric_root,
                    )
                    for model_name, metric_root in ARCHIVE_MODEL_METRIC_ROOTS.items()
                }
            )

    if not external_models:
        logger.info(
            "No external evaluation data found in %s; expected %s and/or %s.",
            external_evaluation_folder,
            UTRECHT_EVALUATION_FILE_NAME,
            EXTERNAL_METRICS_ARCHIVE_FILE_NAME,
        )
        return external_models

    for model_name, model_df in external_models.items():
        duplicate_count: int = int(model_df.index.duplicated(keep="first").sum())
        if duplicate_count:
            logger.info(
                "External model '%s': keeping the first row for %d duplicate "
                "station keys.",
                model_name,
                duplicate_count,
            )
            model_df = model_df[~model_df.index.duplicated(keep="first")].copy()
            external_models[model_name] = model_df
        logger.info(
            "Loaded external model '%s' metrics for %d stations.",
            model_name,
            len(model_df),
        )
    return external_models


def filter_external_skill_scores(
    external_models: dict[str, pd.DataFrame],
    station_keys: set[str],
    output_folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Match external model tables to GEB stations.

    Args:
        external_models: External model skill-score tables keyed by model label.
        station_keys: Uppercase station-name or GRDC keys from GEB.
        output_folder: Folder where matched external tables are saved.
        logger: Logger used for diagnostics.

    Returns:
        Non-empty matched external tables keyed by model label.
    """
    matched_external_models: dict[str, pd.DataFrame] = {}
    output_folder.mkdir(parents=True, exist_ok=True)
    station_keys_upper: set[str] = {station_key.upper() for station_key in station_keys}
    for model_name, all_stations_df in external_models.items():
        matched_df: pd.DataFrame = all_stations_df[
            all_stations_df.index.isin(station_keys_upper)
        ].copy()
        logger.info(
            "External model '%s': %d/%d external stations matched.",
            model_name,
            len(matched_df),
            len(all_stations_df),
        )
        matched_df.to_excel(
            output_folder / f"external_evaluation_filtered_{model_name}.xlsx"
        )
        if not matched_df.empty:
            matched_external_models[model_name] = matched_df
    return matched_external_models


def load_geb_station_keys(
    evaluation_metrics_path: Path,
    snapped_locations_path: Path,
) -> set[str]:
    """Get GEB station keys for matching external skill-score tables.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`.
        snapped_locations_path: Path to discharge snapped-locations geometry.

    Returns:
        Uppercase station-name and GRDC-style station ID keys.
    """
    if evaluation_metrics_path.exists():
        evaluation_df: pd.DataFrame = pd.read_excel(evaluation_metrics_path)
        if not evaluation_df.empty:
            keyed_evaluation_df: pd.DataFrame = _add_match_keys(evaluation_df)
            station_keys: set[str] = set(keyed_evaluation_df["station_name_key"])
            station_keys.update(keyed_evaluation_df["station_id_key"])
            station_keys.discard("")
            return station_keys

    snapped_locations = read_geom(snapped_locations_path)
    station_keys: set[str] = set(
        snapped_locations["discharge_observations_station_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    station_keys.update(
        station_key
        for station_key in snapped_locations.index.to_series().map(
            format_grdc_station_key
        )
        if station_key is not None
    )
    station_keys.discard("")
    return station_keys


def match_external_skill_scores(
    evaluation_df: pd.DataFrame,
    external_models: dict[str, pd.DataFrame],
    output_folder: Path,
    logger: logging.Logger,
    minimum_upstream_area_km2: float,
) -> dict[str, MatchedSkillScores]:
    """Prepare matched GEB-vs-external scores for each external model.

    Args:
        evaluation_df: Loaded and upstream-area-filtered GEB metrics.
        external_models: Loaded external metrics keyed by model name.
        output_folder: Folder where matched external tables are saved.
        logger: Logger used for diagnostics.
        minimum_upstream_area_km2: Minimum modeled upstream-area threshold (km2).

    Returns:
        Plot inputs keyed by external model label.
    """
    matched_scores: dict[str, MatchedSkillScores] = {}
    if evaluation_df.empty or not external_models:
        return matched_scores

    keyed_evaluation_df: pd.DataFrame = _add_match_keys(evaluation_df)
    for model_name, external_model_df in external_models.items():
        eligible_geb_df: pd.DataFrame = keyed_evaluation_df[
            keyed_evaluation_df["upstream_area_GEB"]
            >= minimum_upstream_area_km2 * 1_000_000.0
        ].copy()
        external_station_keys: set[str] = set(external_model_df.index.str.upper())
        matched_geb_df: pd.DataFrame = eligible_geb_df[
            eligible_geb_df["station_name_key"].isin(external_station_keys)
            | eligible_geb_df["station_id_key"].isin(external_station_keys)
        ].copy()
        if matched_geb_df.empty:
            continue

        matched_external_keys: pd.Series = matched_geb_df["station_name_key"].where(
            matched_geb_df["station_name_key"].isin(external_station_keys),
            matched_geb_df["station_id_key"],
        )
        matched_external_df: pd.DataFrame = external_model_df.reindex(
            matched_external_keys
        ).copy()
        matched_external_df.index = matched_geb_df.index

        if "KGE" in matched_geb_df.columns and "KGE" in matched_external_df.columns:
            matched_geb_df["KGE_difference"] = (
                pd.to_numeric(matched_geb_df["KGE"], errors="coerce").to_numpy()
                - pd.to_numeric(matched_external_df["KGE"], errors="coerce").to_numpy()
            )

        matched_external_df.to_excel(
            output_folder / f"external_evaluation_filtered_{model_name}.xlsx"
        )
        logger.info(
            "Pairwise external model '%s': %d matched stations.",
            model_name,
            len(matched_geb_df),
        )

        matched_scores[model_name] = MatchedSkillScores(
            geb=matched_geb_df.drop(
                columns=["station_name_key", "station_id_key"], errors="ignore"
            ),
            external=matched_external_df,
            minimum_upstream_area_km2=minimum_upstream_area_km2,
        )
    return matched_scores
