"""External discharge skill-score helpers."""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

import pandas as pd

from geb.workflows.io import read_geom

GOOGLE_STREAMFLOW_MODEL_NAME: str = "Google Streamflow"
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

_PLOTTED_SKILL_SCORE_COLUMNS: tuple[str, ...] = (
    "KGE",
    "NSE",
    "R2",
    "RMSE",
    "RRMSE",
)


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


def _read_google_metric_file(metric_path: Path, metric_name: str) -> pd.Series:
    """Read one Google streamflow per-station metric file.

    Args:
        metric_path: Path to a Google metric CSV file.
        metric_name: Output metric name.

    Returns:
        Per-station metric values for lead time 0.

    Raises:
        ValueError: If the lead-time column is missing.
    """
    metric_df: pd.DataFrame = pd.read_csv(metric_path, index_col=0)
    if GOOGLE_STREAMFLOW_LEAD_TIME not in metric_df.columns:
        raise ValueError(
            f"Google streamflow metric file {metric_path} is missing "
            f"lead-time column {GOOGLE_STREAMFLOW_LEAD_TIME!r}."
        )
    metric_series: pd.Series = pd.to_numeric(
        metric_df[GOOGLE_STREAMFLOW_LEAD_TIME], errors="coerce"
    )
    metric_series.name = metric_name
    return metric_series


def _assemble_google_streamflow_metrics(
    metric_series: dict[str, pd.Series],
) -> pd.DataFrame:
    """Combine Google metric series into the external skill-score table shape.

    Args:
        metric_series: Metric series keyed by GEB-compatible metric name.

    Returns:
        Per-station Google streamflow skill-score table.
    """
    google_metrics_df: pd.DataFrame = pd.concat(metric_series.values(), axis=1)
    google_metrics_df.index = google_metrics_df.index.str.upper()
    if "R" in google_metrics_df.columns:
        google_metrics_df["R2"] = google_metrics_df["R"] ** 2
    return google_metrics_df.dropna(how="all")


def _find_google_metric_folders(folder: Path) -> list[Path]:
    """Find folders that may contain Google metric CSVs.

    Args:
        folder: External evaluation folder to inspect.

    Returns:
        Candidate folders that should contain the Google per-metric CSV files.
    """
    candidates: list[Path] = [
        folder / GOOGLE_STREAMFLOW_METRIC_ROOT,
        folder / GOOGLE_STREAMFLOW_METRIC_ROOT.relative_to("metrics"),
        folder,
    ]
    candidates.extend(
        metrics_folder / GOOGLE_STREAMFLOW_METRIC_ROOT.relative_to("metrics")
        for metrics_folder in sorted(folder.glob("**/metrics"))
    )
    candidates.extend(
        hydrograph_metrics_folder
        / "per_metric"
        / "google"
        / "2014"
        / "dual_lstm"
        / "hydrologically_separated"
        for hydrograph_metrics_folder in sorted(folder.glob("**/hydrograph_metrics"))
    )

    metric_folders: list[Path] = []
    seen_folders: set[Path] = set()
    for candidate in candidates:
        resolved_candidate: Path = candidate.resolve()
        if resolved_candidate not in seen_folders:
            metric_folders.append(candidate)
            seen_folders.add(resolved_candidate)
    return metric_folders


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
    for metric_root in _find_google_metric_folders(folder):
        metric_paths: dict[str, Path] = {
            metric_name: metric_root / file_name
            for metric_name, file_name in GOOGLE_STREAMFLOW_METRIC_FILES.items()
        }
        if all(metric_path.exists() for metric_path in metric_paths.values()):
            logger.info("Reading Google streamflow metrics from %s.", metric_root)
            return _assemble_google_streamflow_metrics(
                {
                    metric_name: _read_google_metric_file(metric_path, metric_name)
                    for metric_name, metric_path in metric_paths.items()
                }
            )
    logger.info(
        "No extracted Google streamflow metrics found below %s. Expected %s.",
        folder,
        GOOGLE_STREAMFLOW_METRIC_ROOT,
    )
    return pd.DataFrame()


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

    Raises:
        FileNotFoundError: If a required metric file is missing from the archive.
        ValueError: If a metric file is missing the requested lead-time column.
    """
    logger.info("Reading Google streamflow metrics from %s.", metrics_archive_path)
    metric_series: dict[str, pd.Series] = {}
    with tarfile.open(metrics_archive_path, "r:gz") as archive:
        for metric_name, file_name in GOOGLE_STREAMFLOW_METRIC_FILES.items():
            member_name: str = str(GOOGLE_STREAMFLOW_METRIC_ROOT / file_name)
            member_file = archive.extractfile(member_name)
            if member_file is None:
                raise FileNotFoundError(
                    f"Google streamflow metric {member_name} was not found in "
                    f"{metrics_archive_path}."
                )
            metric_df: pd.DataFrame = pd.read_csv(member_file, index_col=0)
            if GOOGLE_STREAMFLOW_LEAD_TIME not in metric_df.columns:
                raise ValueError(
                    f"Google streamflow metric {member_name} is missing lead-time "
                    f"column {GOOGLE_STREAMFLOW_LEAD_TIME!r}."
                )
            metric_series[metric_name] = pd.to_numeric(
                metric_df[GOOGLE_STREAMFLOW_LEAD_TIME], errors="coerce"
            )
            metric_series[metric_name].name = metric_name
    return _assemble_google_streamflow_metrics(metric_series)


def read_external_evaluation_raw(
    external_evaluation_folder: str | Path | None,
    configured_external_evaluation_folder: str | Path | None,
    model_folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Read external model skill-score CSVs and optional Google metrics.

    Args:
        external_evaluation_folder: Optional folder override containing one CSV
            per external model.
        configured_external_evaluation_folder: Configured folder used when no
            override is supplied.
        model_folder: Model folder used to resolve relative paths.
        logger: Logger used for processing diagnostics.

    Returns:
        Mapping from model label to prepared external skill-score table.
    """
    selected_folder: str | Path | None = (
        external_evaluation_folder
        if external_evaluation_folder is not None
        else configured_external_evaluation_folder
    )
    if selected_folder is None:
        logger.info("No external evaluation data folder configured, skipping.")
        return {}

    folder: Path = Path(selected_folder)
    if not folder.is_absolute():
        folder = model_folder / folder
    if folder.is_file() and folder.name == "metrics.tgz":
        google_metrics_df: pd.DataFrame = (
            read_google_streamflow_skill_scores_from_archive(folder, logger)
        )
        return (
            {GOOGLE_STREAMFLOW_MODEL_NAME: google_metrics_df}
            if not google_metrics_df.empty
            else {}
        )
    if not folder.exists():
        logger.info("No external evaluation data folder found at %s, skipping.", folder)
        return {}
    if not folder.is_dir():
        logger.warning(
            "External evaluation path is not a folder: %s. Skipping.", folder
        )
        return {}

    external_models: dict[str, pd.DataFrame] = {}
    csv_paths: list[Path] = sorted(folder.glob("*.csv"))
    if csv_paths:
        logger.info("Reading external evaluation data from %s.", folder)
    for csv_path in csv_paths:
        external_evaluation_df: pd.DataFrame = pd.read_csv(csv_path, index_col=0)
        external_evaluation_df.index = external_evaluation_df.index.map(str).str.upper()
        external_models[csv_path.stem] = external_evaluation_df

    google_metrics_df: pd.DataFrame = read_google_streamflow_skill_scores(
        folder, logger
    )
    metrics_archive_paths: list[Path] = sorted(folder.glob("**/metrics.tgz"))
    if google_metrics_df.empty and metrics_archive_paths:
        google_metrics_df = read_google_streamflow_skill_scores_from_archive(
            metrics_archive_paths[0], logger
        )
    if not google_metrics_df.empty:
        external_models[GOOGLE_STREAMFLOW_MODEL_NAME] = google_metrics_df

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
            "External model '%s': %d/%d stations matched.",
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
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
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

    Returns:
        Tuple containing the prepared GEB skill-score table and external model tables.
    """
    evaluation_df: pd.DataFrame = load_geb_skill_score_metrics(
        evaluation_metrics_path=evaluation_metrics_path,
        minimum_upstream_area_km2=minimum_upstream_area_km2,
        logger=logger,
    )

    external_models: dict[str, pd.DataFrame] = {}
    if include_external:
        external_models = read_external_evaluation_raw(
            external_evaluation_folder=external_evaluation_folder,
            configured_external_evaluation_folder=configured_external_evaluation_folder,
            model_folder=model_folder,
            logger=logger,
        )
    if matched_only and external_models:
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

    return evaluation_df, external_models
