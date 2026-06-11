"""External discharge skill-score helpers."""

from __future__ import annotations

import logging
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from geb.workflows.io import read_geom

GOOGLE_STREAMFLOW_MODEL_NAME: str = "Google Streamflow"
GLOFAS_MODEL_NAME: str = "GloFAS"
GOOGLE_STREAMFLOW_METRICS_URL: str = (
    "https://zenodo.org/records/10397664/files/metrics.tgz?download=1"
)
GOOGLE_STREAMFLOW_LEAD_TIME: str = "0"
GOOGLE_STREAMFLOW_METRIC_FILES: dict[str, str] = {
    "KGE": "KGE.csv",
    "NSE": "NSE.csv",
    "R": "Pearson-r.csv",
    "RMSE": "RMSE.csv",
}
GOOGLE_STREAMFLOW_METRIC_ROOT: Path = Path(
    "metrics/hydrograph_metrics/per_metric/google/2014/dual_lstm/hydrologically_separated"
)
GOOGLE_STREAMFLOW_ARCHIVE_NAME: str = "google_streamflow_metrics.tgz"
GOOGLE_STREAMFLOW_CSV_NAME: str = "google_streamflow.csv"
GLOFAS_METRIC_ROOT: Path = Path(
    "metrics/hydrograph_metrics/per_metric/glofas/2014/glofas_prediction"
)
GLOFAS_CSV_NAME: str = "glofas.csv"
EXTERNAL_MODEL_MINIMUM_UPSTREAM_AREA_KM2: dict[str, float] = {
    # Utrecht 1 km evaluation uses stations with upstream area >= 400 km2.
    "utrecht": 400.0,
}

_PLOTTED_SKILL_SCORE_COLUMNS: tuple[str, ...] = (
    "KGE",
    "KGE_correlation",
    "KGE_bias_ratio",
    "KGE_variability_ratio",
    "NSE",
    "R2",
    "RRMSE",
)


@dataclass(frozen=True)
class SkillScorePlotInputs:
    """Prepared data and filter description for skill-score boxplots.

    Args:
        evaluation_df: Filtered GEB skill-score table.
        external_models: Filtered external skill-score tables keyed by model label.
        filter_summary: Short human-readable summary of applied filters.
    """

    evaluation_df: pd.DataFrame
    external_models: dict[str, pd.DataFrame]
    filter_summary: str


@dataclass(frozen=True)
class GoogleMetricsArchiveModel:
    """Description of one model stored inside Google's metrics archive.

    Args:
        model_name: Human-readable plot label.
        csv_name: Normalized CSV file name saved in the external data folder.
        metric_root: Folder inside `metrics.tgz` containing per-metric CSV files.
    """

    model_name: str
    csv_name: str
    metric_root: Path


GOOGLE_METRICS_ARCHIVE_MODELS: tuple[GoogleMetricsArchiveModel, ...] = (
    GoogleMetricsArchiveModel(
        model_name=GOOGLE_STREAMFLOW_MODEL_NAME,
        csv_name=GOOGLE_STREAMFLOW_CSV_NAME,
        metric_root=GOOGLE_STREAMFLOW_METRIC_ROOT,
    ),
    GoogleMetricsArchiveModel(
        model_name=GLOFAS_MODEL_NAME,
        csv_name=GLOFAS_CSV_NAME,
        metric_root=GLOFAS_METRIC_ROOT,
    ),
)
_ARCHIVE_MODEL_BY_CSV_NAME: dict[str, GoogleMetricsArchiveModel] = {
    archive_model.csv_name: archive_model
    for archive_model in GOOGLE_METRICS_ARCHIVE_MODELS
}


def _get_external_model_minimum_upstream_area_km2(model_name: str) -> float | None:
    """Get the paper-specific upstream-area threshold for an external model.

    Args:
        model_name: External model label from a CSV file or built-in reader.

    Returns:
        Minimum upstream area threshold (km2), or `None` when no
        paper-specific threshold is configured.
    """
    model_name_normalized: str = model_name.lower()
    for (
        model_key,
        minimum_upstream_area_km2,
    ) in EXTERNAL_MODEL_MINIMUM_UPSTREAM_AREA_KM2.items():
        if model_key in model_name_normalized:
            return minimum_upstream_area_km2
    return None


def _get_external_filter_summary(external_models: dict[str, pd.DataFrame]) -> str:
    """Describe configured paper-specific external-model filters.

    Args:
        external_models: External model tables keyed by model label.

    Returns:
        Semicolon-separated summary of configured paper-specific filters.
    """
    filter_parts: list[str] = []
    for model_name in external_models:
        minimum_upstream_area_km2 = _get_external_model_minimum_upstream_area_km2(
            model_name
        )
        if minimum_upstream_area_km2 is not None:
            filter_parts.append(
                f"{model_name}: upstream area >= {minimum_upstream_area_km2:g} km2"
            )
    return "; ".join(filter_parts)


def _format_filter_summary(
    geb_minimum_upstream_area_km2: float,
    external_models: dict[str, pd.DataFrame],
    matched_only: bool,
) -> str:
    """Create a concise plot annotation for applied station filters.

    Args:
        geb_minimum_upstream_area_km2: Effective GEB upstream-area threshold (km2).
        external_models: External model tables keyed by model label.
        matched_only: Whether GEB and external stations are restricted to overlap.

    Returns:
        Multi-line filter summary for the plot.
    """
    filter_lines: list[str] = [
        f"GEB upstream area >= {geb_minimum_upstream_area_km2:g} km2"
    ]
    external_summary: str = _get_external_filter_summary(external_models)
    if external_summary:
        filter_lines.append(external_summary)
    if matched_only and external_models:
        filter_lines.append("matched stations only")
    return "\n".join(filter_lines)


def _empty_skill_score_plot_inputs(
    geb_minimum_upstream_area_km2: float,
    external_models: dict[str, pd.DataFrame],
) -> SkillScorePlotInputs:
    """Create empty plot inputs with a useful filter annotation.

    Args:
        geb_minimum_upstream_area_km2: Effective GEB upstream-area threshold (km2).
        external_models: External model tables keyed by model label.

    Returns:
        Empty plot inputs with filter summary text.
    """
    return SkillScorePlotInputs(
        evaluation_df=pd.DataFrame(),
        external_models={},
        filter_summary=_format_filter_summary(
            geb_minimum_upstream_area_km2=geb_minimum_upstream_area_km2,
            external_models=external_models,
            matched_only=True,
        ),
    )


def _standardize_external_metric_columns(skill_score_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize external metric names to the GEB evaluation column names.

    Args:
        skill_score_df: External skill-score table.

    Returns:
        Skill-score table with Pearson-r source values stored as
        `KGE_correlation` and no standalone `R` column.
    """
    standardized_df: pd.DataFrame = skill_score_df.copy()
    if "R" in standardized_df.columns:
        if "KGE_correlation" not in standardized_df.columns:
            standardized_df["KGE_correlation"] = standardized_df["R"]
        if "R2" not in standardized_df.columns:
            standardized_df["R2"] = standardized_df["R"] ** 2
        standardized_df = standardized_df.drop(columns=["R"])
    return standardized_df


def _get_external_model_name(csv_path: Path) -> str:
    """Get the plot label for an external skill-score CSV.

    Args:
        csv_path: External skill-score CSV path.

    Returns:
        Human-readable model label.
    """
    archive_model: GoogleMetricsArchiveModel | None = _ARCHIVE_MODEL_BY_CSV_NAME.get(
        csv_path.name
    )
    return archive_model.model_name if archive_model is not None else csv_path.stem


def _format_grdc_station_keys(station_ids: pd.Series) -> set[str]:
    """Format station IDs as GRDC keys used by Google streamflow metrics.

    Args:
        station_ids: Station IDs from GEB evaluation outputs.

    Returns:
        Uppercase GRDC-style station keys.
    """
    return {
        station_key
        for station_id in station_ids
        if (station_key := _format_grdc_station_key(station_id)) is not None
    }


def _format_grdc_station_key(station_id: object) -> str | None:
    """Format one station ID as a GRDC key.

    Args:
        station_id: Raw station ID value.

    Returns:
        Uppercase GRDC key, or `None` for missing values.
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


def _get_evaluation_station_keys(evaluation_df: pd.DataFrame) -> set[str]:
    """Get station-name and GRDC-style station-ID keys from GEB metrics.

    Args:
        evaluation_df: GEB skill-score table.

    Returns:
        Uppercase station keys.
    """
    station_keys: set[str] = set()
    if "station_name" in evaluation_df.columns:
        station_keys.update(evaluation_df["station_name"].dropna().str.upper())
    if "station_ID" in evaluation_df.columns:
        station_keys.update(_format_grdc_station_keys(evaluation_df["station_ID"]))
    return station_keys


def _filter_evaluation_to_station_keys(
    evaluation_df: pd.DataFrame,
    station_keys: set[str],
) -> pd.DataFrame:
    """Keep GEB metric rows whose station name or station ID is in `station_keys`.

    Args:
        evaluation_df: GEB skill-score table.
        station_keys: Uppercase station-name or GRDC-style station-ID keys to retain.

    Returns:
        Filtered GEB skill-score table.
    """
    retain_mask: pd.Series = pd.Series(False, index=evaluation_df.index)
    if "station_name" in evaluation_df.columns:
        retain_mask |= evaluation_df["station_name"].str.upper().isin(station_keys)
    if "station_ID" in evaluation_df.columns:
        grdc_keys: pd.Series = evaluation_df["station_ID"].apply(
            lambda station_id: _format_grdc_station_key(station_id) or ""
        )
        retain_mask |= grdc_keys.isin(station_keys)
    return evaluation_df[retain_mask].copy()


def _filter_to_complete_plotted_skill_scores(
    skill_score_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep rows with finite values for all available plotted skill scores.

    Args:
        skill_score_df: Skill-score table after station matching.

    Returns:
        Filtered skill-score table.
    """
    available_metric_columns: list[str] = [
        column_name
        for column_name in _PLOTTED_SKILL_SCORE_COLUMNS
        if column_name in skill_score_df.columns
    ]
    if not available_metric_columns:
        return skill_score_df

    valid_metric_values: pd.DataFrame = skill_score_df[available_metric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    return skill_score_df[valid_metric_values.notna().all(axis=1)].copy()


def _get_archive_metric_series(
    metric_df: pd.DataFrame,
    metric_source: str,
    metric_name: str,
    model_name: str,
) -> pd.Series:
    """Extract the configured lead-time series from an archive metric table.

    Args:
        metric_df: Metric table with stations as rows and lead times as columns.
        metric_source: Human-readable source path for diagnostics.
        metric_name: Output metric name.
        model_name: Human-readable model label for diagnostics.

    Returns:
        Per-station metric values for lead time 0.

    Raises:
        ValueError: If the lead-time column is missing.
    """
    if GOOGLE_STREAMFLOW_LEAD_TIME not in metric_df.columns:
        raise ValueError(
            f"{model_name} metric file {metric_source} is missing "
            f"lead-time column {GOOGLE_STREAMFLOW_LEAD_TIME!r}."
        )
    metric_series: pd.Series = pd.to_numeric(
        metric_df[GOOGLE_STREAMFLOW_LEAD_TIME], errors="coerce"
    )
    metric_series.name = metric_name
    return metric_series


def _read_external_archive_metric_file(
    metric_path: Path,
    metric_name: str,
    model_name: str,
) -> pd.Series:
    """Read one per-station metric file from an extracted metrics folder.

    Args:
        metric_path: Path to a metric CSV file.
        metric_name: Output metric name.
        model_name: Human-readable model label for diagnostics.

    Returns:
        Per-station metric values for lead time 0.
    """
    return _get_archive_metric_series(
        metric_df=pd.read_csv(metric_path, index_col=0),
        metric_source=str(metric_path),
        metric_name=metric_name,
        model_name=model_name,
    )


def _assemble_external_archive_metrics(
    metric_series: dict[str, pd.Series],
) -> pd.DataFrame:
    """Combine archive metric series into the external skill-score table shape.

    Args:
        metric_series: Metric series keyed by GEB-compatible metric name.

    Returns:
        Per-station skill-score table.
    """
    metrics_df: pd.DataFrame = pd.concat(metric_series.values(), axis=1)
    metrics_df.index = metrics_df.index.str.upper()
    return _standardize_external_metric_columns(metrics_df).dropna(how="all")


def _find_external_archive_metric_folders(
    folder: Path,
    metric_root: Path,
) -> list[Path]:
    """Find folders that may contain per-metric CSVs from Google's archive.

    Args:
        folder: External evaluation folder to inspect.
        metric_root: Expected metric folder inside Google's `metrics.tgz`.

    Returns:
        Candidate folders that should contain the per-metric CSV files.
    """
    candidates: list[Path] = [
        folder / metric_root,
        folder / metric_root.relative_to("metrics"),
        folder,
    ]
    candidates.extend(
        metrics_folder / metric_root.relative_to("metrics")
        for metrics_folder in sorted(folder.glob("**/metrics"))
    )

    metric_folders: list[Path] = []
    seen_folders: set[Path] = set()
    for candidate in candidates:
        resolved_candidate: Path = candidate.resolve()
        if resolved_candidate not in seen_folders:
            metric_folders.append(candidate)
            seen_folders.add(resolved_candidate)
    return metric_folders


def read_external_archive_skill_scores(
    folder: Path,
    archive_model: GoogleMetricsArchiveModel,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Read one model's GRDC skill scores from an extracted metrics folder.

    Args:
        folder: External evaluation folder or extracted Google metrics folder.
        archive_model: Model description inside Google's metrics archive.
        logger: Logger used for processing diagnostics.

    Returns:
        Per-station skill-score table. Empty when the expected metric files are
        not present.
    """
    for metric_root in _find_external_archive_metric_folders(
        folder,
        archive_model.metric_root,
    ):
        metric_paths: dict[str, Path] = {
            metric_name: metric_root / file_name
            for metric_name, file_name in GOOGLE_STREAMFLOW_METRIC_FILES.items()
        }
        if all(metric_path.exists() for metric_path in metric_paths.values()):
            logger.info(
                "Reading %s metrics from %s.", archive_model.model_name, metric_root
            )
            return _assemble_external_archive_metrics(
                {
                    metric_name: _read_external_archive_metric_file(
                        metric_path,
                        metric_name,
                        archive_model.model_name,
                    )
                    for metric_name, metric_path in metric_paths.items()
                }
            )
    logger.info(
        "No extracted %s metrics found below %s. Expected %s.",
        archive_model.model_name,
        folder,
        archive_model.metric_root,
    )
    return pd.DataFrame()


def read_external_archive_skill_scores_from_archive(
    metrics_archive_path: Path,
    archive_model: GoogleMetricsArchiveModel,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Read one model's GRDC skill scores from the Zenodo `metrics.tgz` archive.

    Args:
        metrics_archive_path: Path to the Zenodo `metrics.tgz` archive.
        archive_model: Model description inside Google's metrics archive.
        logger: Logger used for processing diagnostics.

    Returns:
        Per-station skill-score table. Empty when this model is absent from the
        archive.
    """
    logger.info(
        "Reading %s metrics from %s.",
        archive_model.model_name,
        metrics_archive_path,
    )
    metric_series: dict[str, pd.Series] = {}
    with tarfile.open(metrics_archive_path, "r:gz") as archive:
        for metric_name, file_name in GOOGLE_STREAMFLOW_METRIC_FILES.items():
            member_name: str = str(archive_model.metric_root / file_name)
            member_file = archive.extractfile(member_name)
            if member_file is None:
                logger.info(
                    "%s metric %s was not found in %s.",
                    archive_model.model_name,
                    member_name,
                    metrics_archive_path,
                )
                return pd.DataFrame()
            metric_series[metric_name] = _get_archive_metric_series(
                metric_df=pd.read_csv(member_file, index_col=0),
                metric_source=member_name,
                metric_name=metric_name,
                model_name=archive_model.model_name,
            )
    return _assemble_external_archive_metrics(metric_series)


def read_google_streamflow_skill_scores(
    folder: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Read Google streamflow GRDC skill scores from an extracted metrics folder.

    Args:
        folder: External evaluation folder or extracted Google metrics folder.
        logger: Logger used for processing diagnostics.

    Returns:
        Per-station Google streamflow skill-score table. Empty when the expected
        Google metrics files are not present.
    """
    return read_external_archive_skill_scores(
        folder,
        GOOGLE_METRICS_ARCHIVE_MODELS[0],
        logger,
    )


def read_google_streamflow_skill_scores_from_archive(
    metrics_archive_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Read Google streamflow GRDC skill scores from the Zenodo `metrics.tgz` archive.

    Args:
        metrics_archive_path: Path to the Zenodo `metrics.tgz` archive.
        logger: Logger used for processing diagnostics.

    Returns:
        Per-station Google streamflow skill-score table.
    """
    return read_external_archive_skill_scores_from_archive(
        metrics_archive_path,
        GOOGLE_METRICS_ARCHIVE_MODELS[0],
        logger,
    )


def _save_external_archive_model_csv(
    metrics_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    logger: logging.Logger,
) -> None:
    """Save normalized archive metrics beside other external CSVs.

    Args:
        metrics_df: Skill-score table read from Google's metrics archive.
        output_path: CSV file path receiving the normalized metrics.
        model_name: Human-readable model label.
        logger: Logger used for processing diagnostics.
    """
    metrics_df.to_csv(output_path)
    logger.info(
        "Saved normalized %s metrics for %d stations to %s.",
        model_name,
        len(metrics_df),
        output_path,
    )


def fetch_google_streamflow_metrics_archive(
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    """Download Google's original streamflow metrics archive.

    Args:
        output_path: Local path where `metrics.tgz` is saved.
        logger: Logger used for processing diagnostics.

    Returns:
        Path to the downloaded archive.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Fetching Google Streamflow metrics from %s to %s.",
        GOOGLE_STREAMFLOW_METRICS_URL,
        output_path,
    )
    urlretrieve(GOOGLE_STREAMFLOW_METRICS_URL, output_path)
    logger.info("Fetched Google Streamflow metrics archive: %s.", output_path)
    return output_path


def _get_archive_paths(folder: Path) -> list[Path]:
    """Get local Google metrics archives that can be reused.

    Args:
        folder: External evaluation folder.

    Returns:
        Existing metrics archive paths, with GEB's normalized archive name first.
    """
    archive_path: Path = folder / GOOGLE_STREAMFLOW_ARCHIVE_NAME
    archive_paths: list[Path] = [archive_path] if archive_path.exists() else []
    archive_paths.extend(sorted(folder.glob("**/metrics.tgz")))
    return archive_paths


def _load_or_fetch_external_archive_model_metrics(
    folder: Path,
    archive_model: GoogleMetricsArchiveModel,
    logger: logging.Logger,
    auto_fetch_google_streamflow: bool,
) -> pd.DataFrame:
    """Load one archive model from local files or fetch it into the external folder.

    Args:
        folder: External evaluation folder.
        archive_model: Model description inside Google's metrics archive.
        logger: Logger used for processing diagnostics.
        auto_fetch_google_streamflow: Whether to download the original Google
            metrics archive if no local Google data are found.

    Returns:
        Normalized skill-score table.
    """
    csv_path: Path = folder / archive_model.csv_name
    if csv_path.exists():
        metrics_df: pd.DataFrame = pd.read_csv(csv_path, index_col=0)
        metrics_df.index = metrics_df.index.map(str).str.upper()
        logger.info(
            "Loaded normalized %s metrics for %d stations from %s.",
            archive_model.model_name,
            len(metrics_df),
            csv_path,
        )
        return _standardize_external_metric_columns(metrics_df)

    metrics_df: pd.DataFrame = read_external_archive_skill_scores(
        folder,
        archive_model,
        logger,
    )
    if metrics_df.empty:
        archive_paths: list[Path] = _get_archive_paths(folder)
        if archive_paths:
            metrics_df = read_external_archive_skill_scores_from_archive(
                archive_paths[0],
                archive_model,
                logger,
            )

    if metrics_df.empty:
        if not auto_fetch_google_streamflow:
            return pd.DataFrame()
        archive_path: Path = folder / GOOGLE_STREAMFLOW_ARCHIVE_NAME
        if not archive_path.exists():
            fetch_google_streamflow_metrics_archive(archive_path, logger)
        metrics_df = read_external_archive_skill_scores_from_archive(
            archive_path,
            archive_model,
            logger,
        )
        if not metrics_df.empty:
            logger.info(
                "Fetched %s metrics for %d stations.",
                archive_model.model_name,
                len(metrics_df),
            )

    if not metrics_df.empty and not csv_path.exists():
        _save_external_archive_model_csv(
            metrics_df,
            csv_path,
            archive_model.model_name,
            logger,
        )
    return metrics_df


def _resolve_external_evaluation_folder(
    external_evaluation_folder: str | Path | None,
    configured_external_evaluation_folder: str | Path | None,
    model_folder: Path,
) -> Path | None:
    """Resolve the external evaluation path from an override or configuration.

    Args:
        external_evaluation_folder: Optional external score folder override.
        configured_external_evaluation_folder: Configured external score folder.
        model_folder: Model folder used to resolve relative paths.

    Returns:
        Absolute or model-relative external evaluation path, or `None` when not
        configured.
    """
    selected_folder: str | Path | None = (
        external_evaluation_folder
        if external_evaluation_folder is not None
        else configured_external_evaluation_folder
    )
    if selected_folder is None:
        return None

    folder: Path = Path(selected_folder)
    return folder if folder.is_absolute() else model_folder / folder


def _read_external_csv_models(
    folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Read normalized external model CSV files from a folder.

    Args:
        folder: External evaluation folder.
        logger: Logger used for processing diagnostics.

    Returns:
        External model tables keyed by model label.
    """
    external_models: dict[str, pd.DataFrame] = {}
    csv_paths: list[Path] = sorted(folder.glob("*.csv"))
    if csv_paths:
        logger.info("Reading external evaluation data from %s.", folder)

    for csv_path in csv_paths:
        external_evaluation_df: pd.DataFrame = pd.read_csv(csv_path, index_col=0)
        external_evaluation_df.index = external_evaluation_df.index.map(str).str.upper()
        model_name: str = _get_external_model_name(csv_path)
        external_models[model_name] = _standardize_external_metric_columns(
            external_evaluation_df
        )
        logger.info(
            "Loaded external model '%s' metrics for %d stations from %s.",
            model_name,
            len(external_models[model_name]),
            csv_path,
        )
    return external_models


def _read_archive_file_models(
    archive_path: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Read all supported models from a Google metrics archive.

    Args:
        archive_path: Path to Google's `metrics.tgz` archive.
        logger: Logger used for processing diagnostics.

    Returns:
        Archive model tables keyed by model label.
    """
    archive_models: dict[str, pd.DataFrame] = {}
    for archive_model in GOOGLE_METRICS_ARCHIVE_MODELS:
        metrics_df: pd.DataFrame = read_external_archive_skill_scores_from_archive(
            archive_path,
            archive_model,
            logger,
        )
        if not metrics_df.empty:
            archive_models[archive_model.model_name] = metrics_df
    return archive_models


def _add_archive_models(
    external_models: dict[str, pd.DataFrame],
    folder: Path,
    logger: logging.Logger,
    auto_fetch_google_streamflow: bool,
) -> None:
    """Add Google-archive models to an external-model mapping in place.

    Args:
        external_models: External model tables keyed by model label.
        folder: External evaluation folder.
        logger: Logger used for processing diagnostics.
        auto_fetch_google_streamflow: Whether to fetch Google's archive when
            local archive models are missing.
    """
    for archive_model in GOOGLE_METRICS_ARCHIVE_MODELS:
        metrics_df: pd.DataFrame | None = external_models.get(archive_model.model_name)
        if metrics_df is None:
            metrics_df = _load_or_fetch_external_archive_model_metrics(
                folder,
                archive_model,
                logger,
                auto_fetch_google_streamflow=auto_fetch_google_streamflow,
            )
        if not metrics_df.empty:
            external_models[archive_model.model_name] = metrics_df
            logger.info(
                "%s metrics available for %d stations.",
                archive_model.model_name,
                len(metrics_df),
            )


def read_external_evaluation_raw(
    external_evaluation_folder: str | Path | None,
    configured_external_evaluation_folder: str | Path | None,
    model_folder: Path,
    logger: logging.Logger,
    auto_fetch_google_streamflow: bool = True,
) -> dict[str, pd.DataFrame]:
    """Read external model skill-score CSVs and optional Google metrics.

    Args:
        external_evaluation_folder: Optional folder override containing one CSV
            per external model.
        configured_external_evaluation_folder: Configured folder used when no
            override is supplied.
        model_folder: Model folder used to resolve relative paths.
        logger: Logger used for processing diagnostics.
        auto_fetch_google_streamflow: Whether to download Google's original
            metrics archive when no local Google metrics are found.

    Returns:
        Mapping from model label to prepared external skill-score table.
    """
    folder: Path | None = _resolve_external_evaluation_folder(
        external_evaluation_folder=external_evaluation_folder,
        configured_external_evaluation_folder=configured_external_evaluation_folder,
        model_folder=model_folder,
    )
    if folder is None:
        logger.info("No external evaluation data folder configured, skipping.")
        return {}

    if folder.is_file() and folder.name == "metrics.tgz":
        return _read_archive_file_models(folder, logger)

    if not folder.exists():
        if not auto_fetch_google_streamflow:
            logger.info(
                "No external evaluation data folder found at %s, skipping.", folder
            )
            return {}
        logger.info("Creating external evaluation data folder at %s.", folder)
        folder.mkdir(parents=True, exist_ok=True)
    if not folder.is_dir():
        logger.warning(
            "External evaluation path is not a folder: %s. Skipping.", folder
        )
        return {}

    external_models: dict[str, pd.DataFrame] = _read_external_csv_models(
        folder,
        logger,
    )
    _add_archive_models(
        external_models=external_models,
        folder=folder,
        logger=logger,
        auto_fetch_google_streamflow=auto_fetch_google_streamflow,
    )

    if not external_models:
        logger.info(
            "No external evaluation CSV or Google metrics files found at %s.", folder
        )
    return external_models


def prepare_external_evaluation(
    external_models: dict[str, pd.DataFrame],
    station_keys: set[str],
    output_folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Filter external model evaluation tables to stations present in GEB.

    Args:
        external_models: External model skill-score tables keyed by model label.
        station_keys: GEB station names and GRDC IDs to retain.
        output_folder: Folder where filtered external model tables are written.
        logger: Logger used for processing diagnostics.

    Returns:
        Mapping from model label to non-empty matched-station skill-score tables.
    """
    matched_external_models: dict[str, pd.DataFrame] = {}
    station_keys_upper: set[str] = {station_key.upper() for station_key in station_keys}
    for model_name, all_stations_df in external_models.items():
        matched_stations_df: pd.DataFrame = all_stations_df[
            all_stations_df.index.isin(station_keys_upper)
        ].copy()
        matched_stations_df = matched_stations_df[
            ~matched_stations_df.index.duplicated(keep="first")
        ].copy()
        matched_stations_df = _filter_to_complete_plotted_skill_scores(
            matched_stations_df
        )
        logger.info(
            "External model '%s': %d/%d external stations matched.",
            model_name,
            len(matched_stations_df),
            len(all_stations_df),
        )
        matched_stations_df.to_excel(
            output_folder / f"external_evaluation_filtered_{model_name}.xlsx"
        )
        if not matched_stations_df.empty:
            matched_external_models[model_name] = matched_stations_df

    return matched_external_models


def get_external_model_output_suffix(model_name: str) -> str:
    """Create a stable filename suffix for a matched external-model plot.

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
    """Load and upstream-area filter GEB discharge skill scores.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`.
        minimum_upstream_area_km2: Minimum modeled upstream area threshold for
            retained GEB stations (km2).
        logger: Logger used for filtering diagnostics.

    Returns:
        GEB skill-score table, or an empty dataframe when the metrics file is missing.
    """
    if not evaluation_metrics_path.exists():
        return pd.DataFrame()

    evaluation_df: pd.DataFrame = pd.read_excel(evaluation_metrics_path)
    if evaluation_df.empty:
        return evaluation_df

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


def _filter_geb_metrics_for_external_model(
    evaluation_df: pd.DataFrame,
    model_name: str,
    minimum_upstream_area_km2: float,
) -> pd.DataFrame:
    """Filter GEB metrics using the threshold for one external comparison.

    Args:
        evaluation_df: GEB skill-score table.
        model_name: External model label.
        minimum_upstream_area_km2: User or configuration threshold (km2).

    Returns:
        GEB skill-score rows eligible for comparison with the external model.
    """
    model_minimum_upstream_area_km2: float = (
        _get_pairwise_geb_minimum_upstream_area_km2(
            model_name=model_name,
            minimum_upstream_area_km2=minimum_upstream_area_km2,
        )
    )
    if evaluation_df.empty or "upstream_area_GEB" not in evaluation_df.columns:
        return evaluation_df
    return evaluation_df[
        evaluation_df["upstream_area_GEB"]
        >= model_minimum_upstream_area_km2 * 1_000_000.0
    ].copy()


def _get_pairwise_geb_minimum_upstream_area_km2(
    model_name: str,
    minimum_upstream_area_km2: float,
) -> float:
    """Get the effective GEB upstream-area threshold for one comparison.

    Args:
        model_name: External model label.
        minimum_upstream_area_km2: User or configuration threshold (km2).

    Returns:
        Effective GEB upstream-area threshold for this comparison (km2).
    """
    return max(
        minimum_upstream_area_km2,
        _get_external_model_minimum_upstream_area_km2(model_name)
        or minimum_upstream_area_km2,
    )


def _prepare_pairwise_matched_skill_score_boxplot_input(
    evaluation_df: pd.DataFrame,
    model_name: str,
    model_df: pd.DataFrame,
    output_folder: Path,
    logger: logging.Logger,
    minimum_upstream_area_km2: float,
) -> SkillScorePlotInputs:
    """Prepare one GEB-vs-external matched skill-score comparison.

    Args:
        evaluation_df: GEB skill-score table after the user-requested base
            upstream-area filter.
        model_name: External model label.
        model_df: External model skill-score table.
        output_folder: Folder where the matched external score file is written.
        logger: Logger used for processing diagnostics.
        minimum_upstream_area_km2: User or configuration upstream-area threshold
            for GEB stations (km2).

    Returns:
        Pairwise GEB/external plot inputs for the exact overlapping station set.
    """
    effective_minimum_upstream_area_km2: float = (
        _get_pairwise_geb_minimum_upstream_area_km2(
            model_name=model_name,
            minimum_upstream_area_km2=minimum_upstream_area_km2,
        )
    )
    model_evaluation_df: pd.DataFrame = _filter_geb_metrics_for_external_model(
        evaluation_df=evaluation_df,
        model_name=model_name,
        minimum_upstream_area_km2=minimum_upstream_area_km2,
    )
    matched_external_models: dict[str, pd.DataFrame] = prepare_external_evaluation(
        external_models={model_name: model_df},
        station_keys=_get_evaluation_station_keys(model_evaluation_df),
        output_folder=output_folder,
        logger=logger,
    )
    external_model_df: pd.DataFrame = matched_external_models.get(
        model_name, pd.DataFrame()
    )
    if external_model_df.empty:
        return _empty_skill_score_plot_inputs(
            geb_minimum_upstream_area_km2=effective_minimum_upstream_area_km2,
            external_models={model_name: model_df},
        )

    external_station_keys: set[str] = set(external_model_df.index.str.upper())
    before_station_match_count: int = len(model_evaluation_df)
    matched_evaluation_df: pd.DataFrame = _filter_evaluation_to_station_keys(
        model_evaluation_df,
        external_station_keys,
    )
    logger.info(
        "Pairwise matched '%s': %d GEB stations retained from %d eligible GEB stations.",
        model_name,
        len(matched_evaluation_df),
        before_station_match_count,
    )

    before_complete_score_count: int = len(matched_evaluation_df)
    matched_evaluation_df = _filter_to_complete_plotted_skill_scores(
        matched_evaluation_df
    )
    if len(matched_evaluation_df) != before_complete_score_count:
        logger.info(
            "Pairwise matched '%s': GEB retained %d/%d stations with complete "
            "plotted skill scores.",
            model_name,
            len(matched_evaluation_df),
            before_complete_score_count,
        )

    geb_station_keys: set[str] = _get_evaluation_station_keys(matched_evaluation_df)
    external_model_df = external_model_df[
        external_model_df.index.isin(geb_station_keys)
    ].copy()
    if matched_evaluation_df.empty or external_model_df.empty:
        return _empty_skill_score_plot_inputs(
            geb_minimum_upstream_area_km2=effective_minimum_upstream_area_km2,
            external_models={model_name: model_df},
        )

    external_models: dict[str, pd.DataFrame] = {model_name: external_model_df}
    return SkillScorePlotInputs(
        evaluation_df=matched_evaluation_df,
        external_models=external_models,
        filter_summary=_format_filter_summary(
            geb_minimum_upstream_area_km2=effective_minimum_upstream_area_km2,
            external_models=external_models,
            matched_only=True,
        ),
    )


def get_geb_station_keys(
    evaluation_metrics_path: Path,
    snapped_locations_path: Path,
) -> set[str]:
    """Get GEB station keys for matching external skill-score tables.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`, used when
            discharge evaluation has already been run.
        snapped_locations_path: Path to the discharge snapped-locations geometry,
            used as a fallback before `evaluate_discharge` has produced metrics.

    Returns:
        Uppercase GEB station names and GRDC-style station ID keys.
    """
    if evaluation_metrics_path.exists():
        evaluation_df: pd.DataFrame = pd.read_excel(evaluation_metrics_path)
        if not evaluation_df.empty:
            station_keys: set[str] = set(
                evaluation_df["station_name"].dropna().str.upper()
            )
            if "station_ID" in evaluation_df.columns:
                station_keys.update(
                    _format_grdc_station_keys(evaluation_df["station_ID"])
                )
            return station_keys

    snapped_locations = read_geom(snapped_locations_path)
    station_keys = set(
        snapped_locations["discharge_observations_station_name"].dropna().str.upper()
    )
    station_keys.update(_format_grdc_station_keys(snapped_locations.index.to_series()))
    return station_keys


def prepare_skill_score_boxplot_inputs(
    evaluation_metrics_path: Path,
    snapped_locations_path: Path,
    external_evaluation_folder: str | Path | None,
    configured_external_evaluation_folder: str | Path | None,
    model_folder: Path,
    output_folder: Path,
    logger: logging.Logger,
    minimum_upstream_area_km2: float,
    include_geb: bool,
    matched_only: bool,
    include_external: bool = True,
    auto_fetch_google_streamflow: bool = True,
) -> SkillScorePlotInputs:
    """Prepare GEB and external skill-score tables for boxplot rendering.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`.
        snapped_locations_path: Path to the discharge snapped-locations geometry.
        external_evaluation_folder: Optional external score folder override.
        configured_external_evaluation_folder: Configured external score folder.
        model_folder: Model folder used to resolve relative external paths.
        output_folder: Folder where matched external score files are written.
        logger: Logger used for processing diagnostics.
        minimum_upstream_area_km2: Minimum modeled upstream area threshold for
            retained GEB stations (km2).
        include_geb: Whether GEB scores will be included in the plot.
        matched_only: Whether to restrict GEB and external scores to overlapping stations.
        include_external: Whether external model scores should be included.
        auto_fetch_google_streamflow: Whether to download Google's original
            metrics archive when no local Google metrics are found.

    Returns:
        Prepared GEB/external tables and the filter summary for plot annotation.
    """
    external_models: dict[str, pd.DataFrame] = {}
    if include_external:
        external_models = read_external_evaluation_raw(
            external_evaluation_folder=external_evaluation_folder,
            configured_external_evaluation_folder=configured_external_evaluation_folder,
            model_folder=model_folder,
            logger=logger,
            auto_fetch_google_streamflow=auto_fetch_google_streamflow,
        )
    evaluation_df: pd.DataFrame = load_geb_skill_score_metrics(
        evaluation_metrics_path=evaluation_metrics_path,
        minimum_upstream_area_km2=minimum_upstream_area_km2,
        logger=logger,
    )
    if matched_only and external_models:
        if include_geb:
            matched_external_models: dict[str, pd.DataFrame] = {}
            for model_name, model_df in external_models.items():
                model_evaluation_df: pd.DataFrame = (
                    _filter_geb_metrics_for_external_model(
                        evaluation_df=evaluation_df,
                        model_name=model_name,
                        minimum_upstream_area_km2=minimum_upstream_area_km2,
                    )
                )
                matched_external_models.update(
                    prepare_external_evaluation(
                        external_models={model_name: model_df},
                        station_keys=_get_evaluation_station_keys(model_evaluation_df),
                        output_folder=output_folder,
                        logger=logger,
                    )
                )
            external_models = matched_external_models
        else:
            geb_station_keys: set[str] = get_geb_station_keys(
                evaluation_metrics_path=evaluation_metrics_path,
                snapped_locations_path=snapped_locations_path,
            )
            external_models = prepare_external_evaluation(
                external_models=external_models,
                station_keys=geb_station_keys,
                output_folder=output_folder,
                logger=logger,
            )

    if external_models:
        logger.info("External models in plot: %s", list(external_models))
    else:
        logger.info("No external models found; showing GEB only.")

    if matched_only and include_geb and external_models and not evaluation_df.empty:
        external_station_keys: set[str] = set()
        for external_df in external_models.values():
            external_station_keys.update(external_df.index.str.upper())

        before_filter_count: int = len(evaluation_df)
        evaluation_df = _filter_evaluation_to_station_keys(
            evaluation_df, external_station_keys
        )
        logger.info(
            "matched_only=True: GEB restricted from %d to %d stations.",
            before_filter_count,
            len(evaluation_df),
        )

        before_complete_score_count: int = len(evaluation_df)
        evaluation_df = _filter_to_complete_plotted_skill_scores(evaluation_df)
        if len(evaluation_df) != before_complete_score_count:
            logger.info(
                "matched_only=True: GEB retained %d/%d matched stations with at "
                "complete plotted skill scores.",
                len(evaluation_df),
                before_complete_score_count,
            )

        geb_station_keys = _get_evaluation_station_keys(evaluation_df)
        valid_external_models: dict[str, pd.DataFrame] = {}
        for model_name, model_df in external_models.items():
            matched_model_df: pd.DataFrame = model_df[
                model_df.index.isin(geb_station_keys)
            ].copy()
            if not matched_model_df.empty:
                valid_external_models[model_name] = matched_model_df
        external_models = valid_external_models

    return SkillScorePlotInputs(
        evaluation_df=evaluation_df,
        external_models=external_models,
        filter_summary=_format_filter_summary(
            geb_minimum_upstream_area_km2=minimum_upstream_area_km2,
            external_models=external_models,
            matched_only=matched_only,
        ),
    )


def prepare_pairwise_skill_score_boxplot_inputs(
    evaluation_metrics_path: Path,
    external_evaluation_folder: str | Path | None,
    configured_external_evaluation_folder: str | Path | None,
    model_folder: Path,
    output_folder: Path,
    logger: logging.Logger,
    minimum_upstream_area_km2: float,
    auto_fetch_google_streamflow: bool = True,
) -> dict[str, SkillScorePlotInputs]:
    """Prepare separate matched GEB-vs-external skill-score plot inputs.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`.
        external_evaluation_folder: Optional external score folder override.
        configured_external_evaluation_folder: Configured external score folder.
        model_folder: Model folder used to resolve relative external paths.
        output_folder: Folder where matched external score files are written.
        logger: Logger used for processing diagnostics.
        minimum_upstream_area_km2: Minimum modeled upstream area threshold for
            retained GEB stations (km2).
        auto_fetch_google_streamflow: Whether to download Google's original
            metrics archive when no local Google metrics are found.

    Returns:
        Mapping from external model label to pairwise matched plot inputs.
    """
    external_models: dict[str, pd.DataFrame] = read_external_evaluation_raw(
        external_evaluation_folder=external_evaluation_folder,
        configured_external_evaluation_folder=configured_external_evaluation_folder,
        model_folder=model_folder,
        logger=logger,
        auto_fetch_google_streamflow=auto_fetch_google_streamflow,
    )
    if not external_models:
        return {}

    evaluation_df: pd.DataFrame = load_geb_skill_score_metrics(
        evaluation_metrics_path=evaluation_metrics_path,
        minimum_upstream_area_km2=minimum_upstream_area_km2,
        logger=logger,
    )
    if evaluation_df.empty:
        return {}

    pairwise_plot_inputs: dict[str, SkillScorePlotInputs] = {}
    for model_name, model_df in external_models.items():
        plot_inputs: SkillScorePlotInputs = (
            _prepare_pairwise_matched_skill_score_boxplot_input(
                evaluation_df=evaluation_df,
                model_name=model_name,
                model_df=model_df,
                output_folder=output_folder,
                logger=logger,
                minimum_upstream_area_km2=minimum_upstream_area_km2,
            )
        )
        if plot_inputs.external_models and not plot_inputs.evaluation_df.empty:
            pairwise_plot_inputs[model_name] = plot_inputs

    return pairwise_plot_inputs
