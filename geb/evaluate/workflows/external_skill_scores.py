"""External discharge skill-score processing helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from geb.workflows.io import read_geom

UTRECHT_MINIMUM_END_YEAR: int = 1991
UTRECHT_MINIMUM_DATA_YEARS: float = 10.0
UTRECHT_MINIMUM_CATCHMENT_AREA_KM2: float = 400.0

_UTRECHT_END_YEAR_COLUMNS: tuple[str, ...] = ("end_year", "end_date")
_UTRECHT_DATA_YEARS_COLUMNS: tuple[str, ...] = ("data_years", "n_years")
_UTRECHT_CATCHMENT_AREA_COLUMNS: tuple[str, ...] = (
    "catchment_area_km2",
    "catchment_area_m2",
)


def _first_existing_column(
    dataframe: pd.DataFrame, column_names: tuple[str, ...]
) -> str | None:
    """Find the first available column from a small ordered candidate list.

    Args:
        dataframe: Table to inspect.
        column_names: Candidate column names in priority order.

    Returns:
        Matching column name, or `None` if no candidate is present.
    """
    available_columns: set[str] = {
        str(column_name) for column_name in dataframe.columns
    }
    for column_name in column_names:
        if column_name in available_columns:
            return column_name
    return None


def filter_utrecht_skill_scores(
    external_evaluation_df: pd.DataFrame,
    logger: logging.Logger,
    minimum_end_year: int = UTRECHT_MINIMUM_END_YEAR,
    minimum_data_years: float = UTRECHT_MINIMUM_DATA_YEARS,
    minimum_catchment_area_km2: float = UTRECHT_MINIMUM_CATCHMENT_AREA_KM2,
) -> pd.DataFrame:
    """Filter Utrecht external skill scores to the published GRDC station criteria.

    Args:
        external_evaluation_df: Utrecht external skill-score table. Required
            metadata columns are the last available year, the number of years
            with data, and the GRDC catchment area (km2, or m2 when the column
            name explicitly contains `m2`).
        logger: Logger used for filtering diagnostics.
        minimum_end_year: Minimum final calendar year in the station record.
        minimum_data_years: Minimum number of years with discharge data.
        minimum_catchment_area_km2: Minimum station catchment area (km2).

    Returns:
        Filtered Utrecht skill-score table.

    Raises:
        ValueError: If required Utrecht metadata columns are missing.
    """
    end_year_column: str | None = _first_existing_column(
        external_evaluation_df, _UTRECHT_END_YEAR_COLUMNS
    )
    data_years_column: str | None = _first_existing_column(
        external_evaluation_df, _UTRECHT_DATA_YEARS_COLUMNS
    )
    catchment_area_column: str | None = _first_existing_column(
        external_evaluation_df, _UTRECHT_CATCHMENT_AREA_COLUMNS
    )
    missing_metadata: list[str] = []
    if end_year_column is None:
        missing_metadata.append(
            f"time-series end year ({', '.join(_UTRECHT_END_YEAR_COLUMNS)})"
        )
    if data_years_column is None:
        missing_metadata.append(
            f"number of years with data ({', '.join(_UTRECHT_DATA_YEARS_COLUMNS)})"
        )
    if catchment_area_column is None:
        missing_metadata.append(
            f"catchment area ({', '.join(_UTRECHT_CATCHMENT_AREA_COLUMNS)})"
        )
    if missing_metadata:
        raise ValueError(
            "Utrecht external skill-score filtering requires metadata columns for: "
            + "; ".join(missing_metadata)
            + "."
        )
    assert end_year_column is not None
    assert data_years_column is not None
    assert catchment_area_column is not None

    numeric_end_years: pd.Series = pd.to_numeric(
        external_evaluation_df[end_year_column], errors="coerce"
    )
    parsed_end_years: pd.Series = pd.to_datetime(
        external_evaluation_df[end_year_column], errors="coerce"
    ).dt.year
    end_years: pd.Series = numeric_end_years.where(
        numeric_end_years.between(1000, 3000), parsed_end_years
    )
    data_years: pd.Series = pd.to_numeric(
        external_evaluation_df[data_years_column], errors="coerce"
    )
    catchment_area_km2: pd.Series = pd.to_numeric(
        external_evaluation_df[catchment_area_column], errors="coerce"
    )
    if catchment_area_column.endswith("_m2"):
        catchment_area_km2 = catchment_area_km2 / 1_000_000.0

    valid_station_mask: pd.Series = (
        end_years.ge(minimum_end_year)
        & data_years.ge(minimum_data_years)
        & catchment_area_km2.ge(minimum_catchment_area_km2)
    )
    filtered_evaluation_df: pd.DataFrame = external_evaluation_df[
        valid_station_mask
    ].copy()
    logger.info(
        "Utrecht skill-score filter retained %d/%d stations "
        "(end year >= %d, data length >= %.1f years, catchment area >= %.1f km2).",
        len(filtered_evaluation_df),
        len(external_evaluation_df),
        minimum_end_year,
        minimum_data_years,
        minimum_catchment_area_km2,
    )
    return filtered_evaluation_df


def prepare_google_streamflow_skill_scores(
    external_evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare Google streamflow external skill scores for plotting.

    This function currently keeps the supplied table unchanged. It exists as the
    source-specific integration point for future Google streamflow score
    normalization, station metadata filtering, or metric renaming.

    Args:
        external_evaluation_df: Google streamflow external skill-score table.

    Returns:
        Prepared Google streamflow skill-score table.
    """
    return external_evaluation_df.copy()


def prepare_external_skill_scores(
    model_name: str,
    external_evaluation_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Apply source-specific preparation to one external skill-score table.

    Args:
        model_name: External model label, typically derived from the CSV file stem.
        external_evaluation_df: External skill-score table.
        logger: Logger used for processing diagnostics.

    Returns:
        Prepared external skill-score table.
    """
    normalized_model_name: str = model_name.lower()
    if "utrecht" in normalized_model_name:
        return filter_utrecht_skill_scores(external_evaluation_df, logger)
    if "google" in normalized_model_name or "streamflow" in normalized_model_name:
        logger.info(
            "Google streamflow skill-score processor has no source-specific filters yet."
        )
        return prepare_google_streamflow_skill_scores(external_evaluation_df)
    return external_evaluation_df.copy()


def read_external_evaluation_raw(
    external_evaluation_folder: str | Path | None,
    configured_external_evaluation_folder: str | Path | None,
    model_folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Read and source-filter all external model evaluation CSV files.

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
    if not folder.exists():
        logger.info("No external evaluation data folder found at %s, skipping.", folder)
        return {}
    if not folder.is_dir():
        logger.warning(
            "External evaluation path is not a folder: %s. Skipping.", folder
        )
        return {}

    csv_paths: list[Path] = sorted(folder.glob("*.csv"))
    if not csv_paths:
        logger.info("No external evaluation CSV files found at %s, skipping.", folder)
        return {}

    logger.info("Reading external evaluation data from %s.", folder)
    external_models: dict[str, pd.DataFrame] = {}
    for csv_path in csv_paths:
        model_name: str = csv_path.stem
        external_evaluation_df: pd.DataFrame = pd.read_csv(csv_path, index_col=0)
        external_evaluation_df.index = external_evaluation_df.index.str.upper()
        external_models[model_name] = prepare_external_skill_scores(
            model_name=model_name,
            external_evaluation_df=external_evaluation_df,
            logger=logger,
        )
    return external_models


def prepare_external_evaluation(
    external_models: dict[str, pd.DataFrame],
    station_names: set[str],
    output_folder: Path,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Filter external model evaluation tables to stations present in GEB.

    Args:
        external_models: External model skill-score tables keyed by model label.
        station_names: GEB station names to retain, matched case-insensitively.
        output_folder: Folder where filtered external model tables are written.
        logger: Logger used for processing diagnostics.

    Returns:
        Mapping from model label to non-empty matched-station skill-score tables.
    """
    matched_external_models: dict[str, pd.DataFrame] = {}
    station_names_upper: set[str] = {
        station_name.upper() for station_name in station_names
    }
    for model_name, all_stations_df in external_models.items():
        matched_stations_df: pd.DataFrame = all_stations_df[
            all_stations_df.index.isin(station_names_upper)
        ].copy()
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


def get_geb_station_names(
    evaluation_metrics_path: Path,
    snapped_locations_path: Path,
) -> set[str]:
    """Get GEB station names for matching external skill-score tables.

    Args:
        evaluation_metrics_path: Path to `evaluation_metrics.xlsx`, used when
            discharge evaluation has already been run.
        snapped_locations_path: Path to the discharge snapped-locations geometry,
            used as a fallback before `evaluate_discharge` has produced metrics.

    Returns:
        Uppercase GEB station names.
    """
    if evaluation_metrics_path.exists():
        evaluation_df: pd.DataFrame = pd.read_excel(evaluation_metrics_path)
        return set(evaluation_df["station_name"].dropna().str.upper())

    snapped_locations = read_geom(snapped_locations_path)
    return set(
        snapped_locations["discharge_observations_station_name"].dropna().str.upper()
    )


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

    Returns:
        Tuple containing the prepared GEB skill-score table and external model tables.
    """
    evaluation_df: pd.DataFrame = load_geb_skill_score_metrics(
        evaluation_metrics_path=evaluation_metrics_path,
        minimum_upstream_area_km2=minimum_upstream_area_km2,
        logger=logger,
    )

    external_models: dict[str, pd.DataFrame] = read_external_evaluation_raw(
        external_evaluation_folder=external_evaluation_folder,
        configured_external_evaluation_folder=configured_external_evaluation_folder,
        model_folder=model_folder,
        logger=logger,
    )
    if matched_only and external_models:
        station_names: set[str] = get_geb_station_names(
            evaluation_metrics_path=evaluation_metrics_path,
            snapped_locations_path=snapped_locations_path,
        )
        external_models = prepare_external_evaluation(
            external_models=external_models,
            station_names=station_names,
            output_folder=output_folder,
            logger=logger,
        )

    if external_models:
        logger.info("External models in plot: %s", list(external_models))
    else:
        logger.info("No external models found; showing GEB only.")

    if matched_only and include_geb and external_models and not evaluation_df.empty:
        external_station_names: set[str] = set()
        for external_df in external_models.values():
            external_station_names.update(external_df.index.str.upper())

        before_filter_count: int = len(evaluation_df)
        evaluation_df = evaluation_df[
            evaluation_df["station_name"].str.upper().isin(external_station_names)
        ].copy()
        logger.info(
            "matched_only=True: GEB restricted from %d to %d stations.",
            before_filter_count,
            len(evaluation_df),
        )

        geb_station_names: set[str] = set(
            evaluation_df["station_name"].dropna().str.upper()
        )
        external_models = {
            model_name: model_df[model_df.index.isin(geb_station_names)].copy()
            for model_name, model_df in external_models.items()
        }

    return evaluation_df, external_models
