"""Load external discharge scores and align them with GEB stations.

The export command selects external scores for known GEB stations.
The plotting workflow filters GEB stations by upstream area before matching, so each
GEB score is compared with an external score for the same station.
"""

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.evaluate.hydrology import Hydrology


# Download manually from https://zenodo.org/records/6390219.
EXTERNAL_EVALUATION_FOLDER_NAME: str = "external_evaluation_data"
UTRECHT_EVALUATION_FILE_NAME: str = "Utrecht_1KM_daily_discharge.csv"
# Download metrics.tgz manually from https://zenodo.org/records/10397664 and
# save it under this fixed name in the external evaluation folder.
GOOGLE_STREAMFLOW_FILE_NAME: str = "google_streamflow_metrics.tgz"
GOOGLE_MODEL_NAME: str = "Google Streamflow"
GLOFAS_MODEL_NAME: str = "GloFAS"  # part of Google streamflow paper/archive
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


@dataclass(frozen=True)
class MatchedSkillScores:
    """GEB and external scores aligned to the same gauging stations.

    Args:
        geb_scores: Filtered GEB skill-score table.
        external_scores: External skill-score table in the same station order.
    """

    geb_scores: pd.DataFrame
    external_scores: pd.DataFrame


def format_grdc_station_id(station_id: object) -> str | None:
    """Express a station ID in the ``GRDC_<ID>`` format used by Caravan.

    For example, 6340100 and 6340100.0 both become ``GRDC_6340100``.
    Existing GRDC prefixes are retained and text is stripped and uppercased.
    This formats an identifier; it does not verify membership in GRDC.

    Args:
        station_id: Station identifier from a GEB score table or GRDC metadata.
            This is an ID, not a station name or river ID.

    Returns:
        Uppercase identifier such as ``"GRDC_1234567"``, or `None` for missing values.
    """
    if pd.isna(station_id):
        return None
    station_id_text: str = str(station_id).strip().upper()
    if not station_id_text or station_id_text == "NAN":
        return None
    if station_id_text.startswith("GRDC_"):
        return station_id_text
    try:
        numeric_id: float = float(station_id_text)
        if numeric_id.is_integer():
            station_id_text = str(int(numeric_id))
    except ValueError:
        pass
    return f"GRDC_{station_id_text}"


def _add_station_matching_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add normalized station names and GRDC IDs for external-score joins.

    External datasets can identify stations by name (such as Utrecht) or
    by GRDC ID (such as Google and GloFAS). Keep both alternatives separate;
    matching prefers a GRDC ID and falls back to the name.

    Args:
        table: Skill-score table with optional ``station_name`` and
            ``station_ID`` columns.

    Returns:
        Copy of the table with ``station_name_for_matching`` and ``grdc_id_for_matching``.
    """
    station_table: pd.DataFrame = table.copy()
    if "station_name" in station_table.columns:
        station_table["station_name_for_matching"] = (
            station_table["station_name"].fillna("").astype(str).str.strip().str.upper()
        )
    else:
        station_table["station_name_for_matching"] = ""
    if "station_ID" not in station_table.columns and table.index.name == "station_ID":
        station_table["station_ID"] = table.index
    if "station_ID" in station_table.columns:
        station_table["grdc_id_for_matching"] = (
            station_table["station_ID"].map(format_grdc_station_id).fillna("")
        )
    else:
        station_table["grdc_id_for_matching"] = ""
    return station_table


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

    model_scores: pd.DataFrame = pd.concat(metric_series.values(), axis=1)
    model_scores.index = model_scores.index.map(str).str.strip().str.upper()
    model_scores["R2"] = model_scores["KGE_correlation"] ** 2
    return model_scores.dropna(how="all")


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
    model_folder: Path = input_folder.resolve().parent
    if model_folder.name == "base":
        # Merged models use <models>/<merged name>/base/input, while ordinary
        # models keep external data directly beside their input folder.
        model_folder = model_folder.parents[1]
    external_evaluation_folder: Path = model_folder / EXTERNAL_EVALUATION_FOLDER_NAME
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
        utrecht_df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=utrecht_path, index_col=0
        )
        utrecht_df.index = utrecht_df.index.map(str).str.strip().str.upper()
        external_models[UTRECHT_MODEL_NAME] = utrecht_df

    metrics_archive_path: Path = (
        external_evaluation_folder / GOOGLE_STREAMFLOW_FILE_NAME
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
            GOOGLE_STREAMFLOW_FILE_NAME,
        )
        return external_models

    for model_name, model_scores in external_models.items():
        duplicate_count: int = int(model_scores.index.duplicated(keep="first").sum())
        if duplicate_count:
            logger.info(
                "External model '%s': keeping the first row for %d duplicate "
                "station keys.",
                model_name,
                duplicate_count,
            )
            model_scores = model_scores[
                ~model_scores.index.duplicated(keep="first")
            ].copy()
            external_models[model_name] = model_scores
        logger.info(
            "Loaded external model '%s' metrics for %d stations.",
            model_name,
            len(model_scores),
        )
    return external_models


def export_external_skill_scores(
    self: Hydrology,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Export external scores for all stations present in this model.

    This optional table export does not create plots and is not required before
    plotting. Discharge score plotting loads and matches external data itself,
    using the selected evaluation period and upstream-area threshold.

    Notes:
        Station names are matched case-insensitively. Falls back to
        ``discharge_snapped_locations.geoparquet`` when
        ``evaluation_metrics.xlsx`` does not yet exist.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        **kwargs: Ignored (CLI compatibility).

    Returns:
        Mapping from model label to matched-stations DataFrame.
    """
    external_models: dict[str, pd.DataFrame] = load_external_skill_scores(
        input_folder=self.model.input_folder,
        logger=self.model.logger,
    )
    if not external_models:
        self.model.logger.info("No external evaluation data found, skipping.")
        return {}

    evaluation_metrics_path: Path = (
        self.evaluate_discharge_output_folder / "evaluation_metrics.xlsx"
    )
    geb_station_identifiers: set[str] = load_geb_station_identifiers(
        evaluation_metrics_path=evaluation_metrics_path,
        snapped_locations_path=self.model.files["geom"][
            "discharge/discharge_snapped_locations"
        ],
    )

    return filter_external_skill_scores(
        external_models=external_models,
        geb_station_identifiers=geb_station_identifiers,
        output_folder=self.evaluate_discharge_output_folder,
        logger=self.model.logger,
    )


def filter_external_skill_scores(
    external_models: dict[str, pd.DataFrame],
    geb_station_identifiers: set[str],
    output_folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Match external model tables to GEB stations.

    Args:
        external_models: External model skill-score tables keyed by model label.
        geb_station_identifiers: Normalized GEB station names and GRDC IDs,
            e.g. a station name or ``GRDC_6340100``; not river IDs.
        output_folder: Folder where matched external tables are saved.
        logger: Logger used for diagnostics.

    Returns:
        Non-empty matched external tables keyed by model label.
    """
    matched_external_models: dict[str, pd.DataFrame] = {}
    output_folder.mkdir(parents=True, exist_ok=True)
    normalized_geb_identifiers: set[str] = {
        station_key.upper() for station_key in geb_station_identifiers
    }
    for model_name, all_station_scores in external_models.items():
        matched_scores: pd.DataFrame = all_station_scores[
            all_station_scores.index.isin(normalized_geb_identifiers)
        ].copy()
        logger.info(
            "External model '%s': %d/%d external stations matched.",
            model_name,
            len(matched_scores),
            len(all_station_scores),
        )
        matched_scores.to_excel(
            output_folder / f"external_evaluation_filtered_{model_name}.xlsx"
        )
        if not matched_scores.empty:
            matched_external_models[model_name] = matched_scores
    return matched_external_models


def load_geb_station_identifiers(
    evaluation_metrics_path: Path,
    snapped_locations_path: Path,
) -> set[str]:
    """Collect GEB station names and GRDC IDs accepted by external datasets.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`.
        snapped_locations_path: Path to discharge snapped-locations geometry.

    Returns:
        Uppercase station names and GRDC-style station IDs. This is a set
        of alternative identifiers for matching, not a list of unique stations.
    """
    if evaluation_metrics_path.exists():
        station_scores: pd.DataFrame = pd.read_excel(evaluation_metrics_path)
        if not station_scores.empty:
            station_scores_with_identifiers: pd.DataFrame = (
                _add_station_matching_columns(station_scores)
            )
            geb_station_identifiers: set[str] = set(
                station_scores_with_identifiers["station_name_for_matching"]
            )
            geb_station_identifiers.update(
                station_scores_with_identifiers["grdc_id_for_matching"]
            )
            geb_station_identifiers.discard("")
            return geb_station_identifiers

    snapped_locations = read_geom(snapped_locations_path)
    geb_station_identifiers: set[str] = set(
        snapped_locations["discharge_observations_station_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    geb_station_identifiers.update(
        station_key
        for station_key in snapped_locations.index.to_series().map(
            format_grdc_station_id
        )
        if station_key is not None
    )
    geb_station_identifiers.discard("")
    return geb_station_identifiers


def match_external_skill_scores(
    station_scores: pd.DataFrame,
    external_models: dict[str, pd.DataFrame],
    output_folder: Path,
    logger: logging.Logger,
) -> dict[str, MatchedSkillScores]:
    """Pair already-selected GEB stations with each external model by ID or name.

    The caller applies the GEB upstream-area threshold once. This function
    only takes the station intersection and aligns rows; external datasets'
    own inclusion criteria do not replace GEB's station selection.

    Args:
        station_scores: Loaded and upstream-area-filtered GEB metrics.
        external_models: Loaded external metrics keyed by model name.
        output_folder: Folder where matched external tables are saved.
        logger: Logger used for diagnostics.

    Returns:
        Plot inputs keyed by external model label.
    """
    matched_scores: dict[str, MatchedSkillScores] = {}
    if station_scores.empty or not external_models:
        return matched_scores

    station_scores_with_identifiers: pd.DataFrame = _add_station_matching_columns(
        station_scores
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    for model_name, external_model_scores in external_models.items():
        external_station_identifiers: set[str] = set(
            external_model_scores.index.str.upper()
        )
        matched_geb_scores: pd.DataFrame = station_scores_with_identifiers[
            station_scores_with_identifiers["station_name_for_matching"].isin(
                external_station_identifiers
            )
            | station_scores_with_identifiers["grdc_id_for_matching"].isin(
                external_station_identifiers
            )
        ].copy()
        if matched_geb_scores.empty:
            continue

        matched_external_identifiers: pd.Series = matched_geb_scores[
            "grdc_id_for_matching"
        ].where(
            matched_geb_scores["grdc_id_for_matching"].isin(
                external_station_identifiers
            ),
            matched_geb_scores["station_name_for_matching"],
        )
        matched_external_scores: pd.DataFrame = external_model_scores.reindex(
            matched_external_identifiers
        ).copy()
        matched_external_scores.index = matched_geb_scores.index

        if (
            "KGE" in matched_geb_scores.columns
            and "KGE" in matched_external_scores.columns
        ):
            matched_geb_scores["KGE_difference"] = (
                pd.to_numeric(matched_geb_scores["KGE"], errors="coerce").to_numpy()
                - pd.to_numeric(
                    matched_external_scores["KGE"], errors="coerce"
                ).to_numpy()
            )

        matched_external_scores.to_excel(
            output_folder / f"external_evaluation_filtered_{model_name}.xlsx"
        )
        logger.info(
            "Pairwise external model '%s': %d matched stations.",
            model_name,
            len(matched_geb_scores),
        )

        matched_scores[model_name] = MatchedSkillScores(
            geb_scores=matched_geb_scores.drop(
                columns=["station_name_for_matching", "grdc_id_for_matching"],
                errors="ignore",
            ),
            external_scores=matched_external_scores,
        )
    return matched_scores
