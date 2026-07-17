"""Prepare external discharge skill-score comparisons."""

import logging
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from geb.workflows.io import read_geom

PLOTTED_SKILL_SCORE_COLUMNS: tuple[str, ...] = (
    "KGE",
    "KGE_correlation",
    "KGE_bias_ratio",
    "KGE_variability_ratio",
    "NSE",
    "R2",
    "RRMSE",
)

# Some external products document recommended minimum basin sizes. These are
# applied to GEB rows in pairwise plots so comparisons use similar station sets.
EXTERNAL_MODEL_MINIMUM_UPSTREAM_AREA_KM2: dict[str, float] = {
    "utrecht": 400.0,
    "glofas": 500.0,
}

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
}
ARCHIVE_LEAD_TIME_COLUMN: str = "0"


@dataclass(frozen=True)
class SkillScorePlotInputs:
    """Prepared discharge skill-score tables for plotting.

    Args:
        evaluation_df: Filtered GEB skill-score table.
        external_models: External skill-score tables keyed by model label.
        minimum_upstream_area_km2: Effective GEB upstream-area threshold (km2).
    """

    evaluation_df: pd.DataFrame
    external_models: dict[str, pd.DataFrame]
    minimum_upstream_area_km2: float | None = None


def _format_grdc_station_key(station_id: object) -> str | None:
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
    try:
        station_id_text = str(int(float(station_id_text)))
    except ValueError:
        station_id_text = station_id_text.removeprefix("GRDC_")
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
        keyed_table["station_id_key"] = keyed_table["station_ID"].map(
            lambda station_id: _format_grdc_station_key(station_id) or ""
        )
    else:
        keyed_table["station_id_key"] = ""
    return keyed_table


def _station_keys_from_evaluation(evaluation_df: pd.DataFrame) -> set[str]:
    """Get station-name and GRDC keys from a GEB evaluation table.

    Args:
        evaluation_df: GEB discharge evaluation table.

    Returns:
        Uppercase station keys used for matching external tables.
    """
    keyed_evaluation_df: pd.DataFrame = _add_match_keys(evaluation_df)
    station_keys: set[str] = set(keyed_evaluation_df["station_name_key"])
    station_keys.update(keyed_evaluation_df["station_id_key"])
    station_keys.discard("")
    return station_keys


def _drop_rows_with_missing_plotted_scores(
    skill_score_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep rows that have values for every available plotted score.

    Args:
        skill_score_df: GEB or external skill-score table.

    Returns:
        Complete-score rows. If no plotted columns exist, the table is returned
        unchanged.
    """
    score_columns: list[str] = [
        column_name
        for column_name in PLOTTED_SKILL_SCORE_COLUMNS
        if column_name in skill_score_df.columns
    ]
    if not score_columns:
        return skill_score_df
    numeric_scores: pd.DataFrame = skill_score_df[score_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    return skill_score_df[numeric_scores.notna().all(axis=1)].copy()


def _external_minimum_area_km2(model_name: str) -> float | None:
    """Get the recommended minimum basin size for an external model.

    Args:
        model_name: External model label.

    Returns:
        Minimum upstream area (km2), or `None` when no recommendation is known.
    """
    model_name_lower: str = model_name.lower()
    for model_key, minimum_area_km2 in EXTERNAL_MODEL_MINIMUM_UPSTREAM_AREA_KM2.items():
        if model_key in model_name_lower:
            return minimum_area_km2
    return None


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


def _read_external_evaluation_archive(
    archive_path: Path,
) -> dict[str, pd.DataFrame]:
    """Read fixed Google Streamflow and GloFAS scores from one local archive.

    Args:
        archive_path: Path to ``google_streamflow_metrics.tgz``.

    Returns:
        Per-GRDC-station daily skill scores keyed by model name.
    """
    with tarfile.open(archive_path, mode="r:gz") as archive:
        external_models: dict[str, pd.DataFrame] = {
            model_name: _read_model_metrics_from_archive(
                archive=archive,
                archive_path=archive_path,
                metric_root=metric_root,
            )
            for model_name, metric_root in ARCHIVE_MODEL_METRIC_ROOTS.items()
        }
    return external_models


def read_external_evaluation_raw(
    model_folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Read fixed local Utrecht, Google, and GloFAS skill-score files.

    Args:
        model_folder: Merged-model input folder. The external data directory is
            located two levels above it in the main model folder.
        logger: Logger used for diagnostics.

    Returns:
        External skill-score tables keyed by model label.
    """
    folder: Path = (
        model_folder.resolve().parent.parent / EXTERNAL_EVALUATION_FOLDER_NAME
    )
    if not folder.is_dir():
        logger.info(
            "No optional external evaluation folder found at %s; showing GEB only.",
            folder,
        )
        return {}
    logger.info("Reading external evaluation data from %s.", folder.resolve())

    external_models: dict[str, pd.DataFrame] = {}
    utrecht_path: Path = folder / UTRECHT_EVALUATION_FILE_NAME
    if utrecht_path.exists():
        utrecht_df: pd.DataFrame = pd.read_csv(utrecht_path, index_col=0)
        utrecht_df.index = utrecht_df.index.map(str).str.strip().str.upper()
        external_models[UTRECHT_MODEL_NAME] = utrecht_df

    metrics_archive_path: Path = folder / EXTERNAL_METRICS_ARCHIVE_FILE_NAME
    if metrics_archive_path.exists():
        external_models.update(_read_external_evaluation_archive(metrics_archive_path))

    if not external_models:
        logger.info(
            "No external evaluation data found in %s; expected %s and/or %s.",
            folder,
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


def prepare_external_evaluation(
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
        matched_df = _drop_rows_with_missing_plotted_scores(matched_df)
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


def get_external_model_output_suffix(model_name: str) -> str:
    """Create a stable filename suffix for an external-model plot.

    Args:
        model_name: External model label.

    Returns:
        Lowercase filename suffix beginning with `_matched_`.
    """
    suffix_text: str = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"_matched_{suffix_text}"


def load_geb_skill_score_metrics(
    evaluation_metrics_path: Path,
    minimum_upstream_area_km2: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load GEB discharge skill scores.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`.
        minimum_upstream_area_km2: Minimum modeled upstream-area threshold (km2).
        logger: Logger used for diagnostics.

    Returns:
        GEB skill-score table. Daily columns are copied to the short plotting
        names such as ``KGE`` because summary plots compare daily metrics.
    """
    if not evaluation_metrics_path.exists():
        return pd.DataFrame()

    evaluation_df: pd.DataFrame = pd.read_excel(evaluation_metrics_path)
    if evaluation_df.empty:
        return evaluation_df

    for metric_column in PLOTTED_SKILL_SCORE_COLUMNS:
        daily_metric_column: str = f"{metric_column}_daily"
        if daily_metric_column in evaluation_df.columns:
            evaluation_df[metric_column] = evaluation_df[daily_metric_column]

    before_filter_count: int = len(evaluation_df)
    evaluation_df = evaluation_df[
        evaluation_df["upstream_area_GEB"] >= minimum_upstream_area_km2 * 1_000_000.0
    ].copy()
    logger.info(
        "Upstream-area plot filter retained %d/%d GEB stations at %.1f km2 or larger.",
        len(evaluation_df),
        before_filter_count,
        minimum_upstream_area_km2,
    )
    return evaluation_df


def get_geb_station_keys(
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
            return _station_keys_from_evaluation(evaluation_df)

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
            _format_grdc_station_key
        )
        if station_key is not None
    )
    station_keys.discard("")
    return station_keys


def prepare_skill_score_boxplot_inputs(
    evaluation_metrics_path: Path,
    snapped_locations_path: Path,
    model_folder: Path,
    output_folder: Path,
    logger: logging.Logger,
    minimum_upstream_area_km2: float,
) -> SkillScorePlotInputs:
    """Prepare the GEB-plus-external table set for the main boxplot.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`.
        snapped_locations_path: Path to discharge snapped-locations geometry.
        model_folder: Model folder containing external evaluation files.
        output_folder: Folder where matched external tables are saved.
        logger: Logger used for diagnostics.
        minimum_upstream_area_km2: Minimum modeled upstream-area threshold (km2).

    Returns:
        Prepared GEB and external plot inputs.
    """
    evaluation_df: pd.DataFrame = load_geb_skill_score_metrics(
        evaluation_metrics_path=evaluation_metrics_path,
        minimum_upstream_area_km2=minimum_upstream_area_km2,
        logger=logger,
    )
    external_models: dict[str, pd.DataFrame] = read_external_evaluation_raw(
        model_folder=model_folder,
        logger=logger,
    )
    if external_models:
        station_keys: set[str] = (
            _station_keys_from_evaluation(evaluation_df)
            if not evaluation_df.empty
            else get_geb_station_keys(evaluation_metrics_path, snapped_locations_path)
        )
        external_models = prepare_external_evaluation(
            external_models=external_models,
            station_keys=station_keys,
            output_folder=output_folder,
            logger=logger,
        )
        logger.info("External models in plot: %s", list(external_models))
    else:
        logger.info("No external models found; showing GEB only.")

    return SkillScorePlotInputs(
        evaluation_df=evaluation_df,
        external_models=external_models,
        minimum_upstream_area_km2=minimum_upstream_area_km2,
    )


def prepare_pairwise_skill_score_inputs(
    evaluation_df: pd.DataFrame,
    external_models: dict[str, pd.DataFrame],
    output_folder: Path,
    logger: logging.Logger,
    minimum_upstream_area_km2: float,
) -> dict[str, SkillScorePlotInputs]:
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
    pairwise_inputs: dict[str, SkillScorePlotInputs] = {}
    if evaluation_df.empty or not external_models:
        return pairwise_inputs

    keyed_evaluation_df: pd.DataFrame = _add_match_keys(evaluation_df)
    for model_name, external_model_df in external_models.items():
        model_minimum_area_km2: float = max(
            minimum_upstream_area_km2,
            _external_minimum_area_km2(model_name) or minimum_upstream_area_km2,
        )
        eligible_geb_df: pd.DataFrame = keyed_evaluation_df[
            keyed_evaluation_df["upstream_area_GEB"]
            >= model_minimum_area_km2 * 1_000_000.0
        ].copy()
        if eligible_geb_df.empty:
            continue

        external_station_keys: set[str] = set(external_model_df.index.str.upper())
        matched_geb_df: pd.DataFrame = eligible_geb_df[
            eligible_geb_df["station_name_key"].isin(external_station_keys)
            | eligible_geb_df["station_id_key"].isin(external_station_keys)
        ].copy()
        matched_geb_df = _drop_rows_with_missing_plotted_scores(matched_geb_df)
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
        matched_external_df = _drop_rows_with_missing_plotted_scores(
            matched_external_df
        )
        common_index: pd.Index = matched_geb_df.index.intersection(
            matched_external_df.index
        )
        matched_geb_df = matched_geb_df.loc[common_index].copy()
        matched_external_df = matched_external_df.loc[common_index].copy()
        if matched_geb_df.empty or matched_external_df.empty:
            continue

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

        pairwise_inputs[model_name] = SkillScorePlotInputs(
            evaluation_df=matched_geb_df.drop(
                columns=["station_name_key", "station_id_key"], errors="ignore"
            ),
            external_models={model_name: matched_external_df},
            minimum_upstream_area_km2=model_minimum_area_km2,
        )
    return pairwise_inputs
