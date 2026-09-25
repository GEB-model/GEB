"""Evaluate reported discharge against station observations and export skill scores."""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from geb.evaluate.workflows import (
    discharge_characteristics,
    discharge_helpers,
    discharge_plots,
)
from geb.evaluate.workflows.dashboard import (
    DischargeDashboardGeometries,
    StationChartBundleWriter,
    build_station_chart_data,
    determine_main_time_index,
    load_discharge_dashboard_geometries,
    serialize_main_timeline,
    write_discharge_dashboard,
)
from geb.evaluate.workflows.discharge_helpers import (
    DischargeEvaluationPaths,
    _get_discharge_evaluation_paths,
    load_station_discharge_comparison,
)
from geb.evaluate.workflows.discharge_metrics import (
    DISCHARGE_SCORE_COLUMNS,
    SEASONAL_KGE_METRICS,
    DischargeMetrics,
    calculate_discharge_metrics,
    calculate_seasonal_discharge_metrics,
    use_daily_discharge_scores,
)
from geb.evaluate.workflows.discharge_station_checks import (
    find_dashboard_excluded_stations,
    find_excluded_stations,
)
from geb.workflows.io import read_geom

if TYPE_CHECKING:
    from geb.evaluate.hydrology import Hydrology


def evaluate_discharge(
    self: Hydrology,
    run_name: str = "default",
    export_yearly_timeseries_plots: bool = True,
    correct_discharge_observations: bool = False,
    enable_plotting: bool = True,
    include_return_period_plots: bool = True,
    minimum_upstream_area_km2: float | None = None,
    minimum_timeseries_length_years: float | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    clean_output: bool = False,
    export_timeseries_plots: bool = False,
    export_return_period_plots: bool = False,
) -> dict[str, float | None]:
    """Evaluate the discharge grid from GEB against observations from the discharge observations database.

    Compares simulated discharge from the GEB model with observed discharge data from
    gauging stations. Calculates discharge skill scores and creates
    evaluation plots and interactive maps for analysis.

    Notes:
        The discharge simulation files must exist in the report directory structure.
        If no discharge stations are found in the basin, empty evaluation datasets
        are created. The evaluation can be skipped if results already exist.
        Daily KGE is also calculated for the meteorological seasons winter
        (December-February), spring (March-May), summer (June-August), and
        autumn (September-November).
        Excluded stations are saved separately in ``diagnostic_metrics`` for
        dashboard inspection and never contribute to summary scores.

    Args:
        self: Hydrology evaluator providing model settings and output paths.
        run_name: Name of the simulation run to evaluate. Must correspond to an
            existing run directory in the model output folder.
        export_yearly_timeseries_plots: Whether to save one discharge PNG per station
            and calendar year when `export_timeseries_plots` is True.
        correct_discharge_observations: Whether to multiply simulated discharge by the station upstream
            area divided by the low-resolution GEB routing upstream area.
        enable_plotting: The overall switch for the dashboard and all evaluation plots.
            Defaults to True. Station figure exports also require their respective
            export options. Set to False to save only evaluation metrics.
        include_return_period_plots: Whether to include interactive return-period
            curves in the dashboard. Defaults to True. Only applies when
            `enable_plotting` is True; independent of static figure exports.
        minimum_upstream_area_km2: Optional minimum modeled upstream area threshold for station evaluation (km2).
            If omitted, `hydrology.evaluation.discharge.minimum_upstream_area_km2` is used.
        minimum_timeseries_length_years: Optional minimum paired observation-simulation timeseries length for station evaluation (years).
            If omitted, `hydrology.evaluation.discharge.minimum_timeseries_length_years` is used.
        start_year: Optional first calendar year included in the skill-score
            calculation. If omitted, the first overlapping year is used.
        end_year: Optional last calendar year included in the skill-score
            calculation. If omitted, the last overlapping year is used.
        clean_output: Whether to remove the existing discharge evaluation
            output folder before writing new files. Defaults to `False` so
            period-specific evaluations do not delete full-period outputs.
        export_timeseries_plots: Whether to save static station time-series
            images, including yearly variants. Set to False to keep the dashboard
            and skill-score plots without writing station time-series images.
            Defaults to False. Only applies when `enable_plotting` is True.
        export_return_period_plots: Whether to save static station return-period
            PNG and SVG figures. Defaults to False. Only applies when
            `enable_plotting` is True; independent of dashboard curves.

    Returns:
        Dictionary containing median frequency-specific discharge skill
        scores (e.g., KGE_hourly, KGE_daily) and median seasonal daily KGE.
        Stations with hourly data are also evaluated on the daily resampled data, and those metrics are included in
        the returned dictionary. Stations with only daily data are not evaluated on the hourly data.

    Raises:
        FileNotFoundError: If the run folder does not exist in the report directory.
        ValueError: If a non-existing frequency label is encountered in the discharge observations data.
    """
    output_folder: Path = self.evaluate_discharge_output_folder
    evaluation_paths: DischargeEvaluationPaths = _get_discharge_evaluation_paths(
        output_folder=output_folder,
        start_year=start_year,
        end_year=end_year,
    )
    dashboard_path: Path = (
        evaluation_paths.plot_folder
        / f"discharge_evaluation_map{evaluation_paths.suffix}.html"
    )
    self.model.logger.info(
        "Evaluating discharge skill scores for %s.",
        evaluation_paths.label,
    )

    if minimum_upstream_area_km2 is None:
        minimum_upstream_area_km2 = self.model.config["hydrology"]["evaluation"][
            "discharge"
        ]["minimum_upstream_area_km2"]
    self.model.logger.info(
        "Using %.1f km2 as the minimum upstream area threshold for discharge evaluation.",
        minimum_upstream_area_km2,
    )
    if minimum_timeseries_length_years is None:
        if start_year is not None and end_year is not None:
            # A fixed evaluation window is already a time constraint; applying
            # a minimum-length filter on top would wrongly exclude stations
            # (e.g. the 8-year 2014-2021 Google/GloFAS window).
            minimum_timeseries_length_years = 0.0
            self.model.logger.info(
                "start_year and end_year both provided; disabling timeseries-length filter."
            )
        else:
            minimum_timeseries_length_years = self.model.config["hydrology"][
                "evaluation"
            ]["discharge"]["minimum_timeseries_length_years"]
            self.model.logger.info(
                "Using %.2f years as the minimum paired timeseries length for discharge evaluation.",
                minimum_timeseries_length_years,
            )

    if not np.isfinite(minimum_upstream_area_km2) or minimum_upstream_area_km2 < 0:
        raise ValueError("Minimum upstream area must be finite and non-negative.")
    if (
        not np.isfinite(minimum_timeseries_length_years)
        or minimum_timeseries_length_years < 0
    ):
        raise ValueError("Minimum record length must be finite and non-negative.")

    observations_by_frequency: dict[str, pd.DataFrame] = (
        discharge_helpers.load_discharge_observations(self.model.files["table"])
    )

    snapped_locations: gpd.GeoDataFrame = read_geom(
        self.model.files["geom"]["discharge/discharge_snapped_locations"]
    )
    if (
        "snapping_method" not in snapped_locations.columns
        or not (
            snapped_locations["snapping_method"] == "original_subgrid_pixel_v1"
        ).all()
    ):
        raise ValueError(
            "Discharge snaps use an older method. Re-run setup_hydrography and "
            "setup_discharge_observations, then rerun discharge reporting and evaluation."
        )
    # Custom stations and older inputs without an offset use UTC.
    snapped_locations["timezone_utc_offset"] = snapped_locations.get(
        "timezone_utc_offset", 0.0
    )
    snapped_locations["timezone_utc_offset"] = snapped_locations[
        "timezone_utc_offset"
    ].fillna(0.0)

    run_output_folder: Path = (
        Path(self.model.config["general"]["output_folder"]) / run_name
    )
    report_folder: Path = run_output_folder / "report"
    if not report_folder.exists():
        raise FileNotFoundError(
            f"Run folder '{run_name}' does not exist in the report directory. Did you run the model?"
        )
    self.model.logger.info(
        "Loading discharge simulation for run '%s' from %s.",
        run_name,
        report_folder,
    )

    if clean_output and output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if enable_plotting:
        evaluation_paths.plot_folder.mkdir(parents=True, exist_ok=True)

    period_start: pd.Timestamp | None = (
        cast(pd.Timestamp, pd.Timestamp(year=start_year, month=1, day=1))
        if start_year is not None
        else None
    )
    period_end: pd.Timestamp | None = (
        cast(
            pd.Timestamp,
            pd.Timestamp(year=end_year + 1, month=1, day=1) - pd.Timedelta("1ns"),
        )
        if end_year is not None
        else None
    )
    minimum_upstream_area_m2: float = minimum_upstream_area_km2 * 1_000_000.0
    excluded_stations: gpd.GeoDataFrame = find_excluded_stations(
        snapped_locations, report_folder
    )
    station_score_records: list[dict[str, Any]] = []
    chart_writer: StationChartBundleWriter | None = (
        StationChartBundleWriter(
            dashboard_path=dashboard_path,
            max_stations_per_bundle=50,
        )
        if enable_plotting
        else None
    )
    chart_timelines: dict[str, Any] = {}

    self.model.logger.info("Starting discharge evaluation...")
    for (
        frequency_label,
        observations_by_station,
    ) in observations_by_frequency.items():
        if observations_by_station.empty:
            continue
        main_time_index: pd.DatetimeIndex | None = (
            determine_main_time_index(
                observations_index=cast(
                    pd.DatetimeIndex, observations_by_station.index
                ),
                simulation_output_folder=run_output_folder,
                frequency=frequency_label,
                period_start=period_start,
                period_end=period_end,
            )
            if enable_plotting
            else None
        )
        if main_time_index is not None:
            chart_timelines[frequency_label] = serialize_main_timeline(
                main_time_index
            )
        minimum_paired_timesteps: float = (
            minimum_timeseries_length_years
            * 365
            * (24 if frequency_label == "hourly" else 1)
        )
        for station_id in tqdm(observations_by_station.columns):
            observed_discharge_series: pd.Series = observations_by_station[station_id]

            station_metadata: pd.Series = snapped_locations.loc[station_id]
            station_name: str = station_metadata.discharge_observations_station_name
            station_coordinates: tuple[float, float] = (
                station_metadata.discharge_observations_station_coords
            )
            routing_grid_coordinates: tuple[float, float] = tuple(
                station_metadata.snapped_grid_pixel_lonlat
            )
            original_subgrid_pixel_coordinates: tuple[float, float] = tuple(
                station_metadata.original_subgrid_pixel_lonlat
            )
            upstream_area_ratio: float = float(
                station_metadata.discharge_observations_to_GEB_upstream_area_ratio
            )
            geb_upstream_area_m2: float = float(
                station_metadata.GEB_upstream_area_from_grid
            )
            if geb_upstream_area_m2 < minimum_upstream_area_m2:
                # Smaller catchments tend to be dominated by local timing and snapping
                # errors, so the default benchmark excludes them from summary scores.
                continue

            timezone_utc_offset: float = float(station_metadata["timezone_utc_offset"])

            try:
                discharge_comparison: pd.DataFrame = load_station_discharge_comparison(
                    run_output_folder,
                    station_id,
                    observed_discharge_series,
                    correct_discharge_observations,
                    upstream_area_ratio,
                    timezone_utc_offset=timezone_utc_offset,
                )
            except FileNotFoundError:
                self.model.logger.warning(
                    "Skipping station %s: no simulation output found.", station_id
                )
                continue
            if period_start is not None or period_end is not None:
                # Keep complete calendar years and include the final instant.
                discharge_comparison = discharge_comparison.loc[period_start:period_end]

            if discharge_comparison.dropna().shape[0] < max(
                2, minimum_paired_timesteps
            ):
                continue

            discharge_metrics: DischargeMetrics = calculate_discharge_metrics(
                discharge_comparison
            )
            station_metrics: dict[str, float] = discharge_metrics._asdict()

            if enable_plotting:
                if export_timeseries_plots:
                    discharge_plots.save_discharge_timeseries_plots(
                        station_id=station_id,
                        discharge_comparison=discharge_comparison,
                        upstream_area_ratio=upstream_area_ratio,
                        metrics=station_metrics,
                        plot_folder=evaluation_paths.plot_folder,
                        export_yearly_timeseries_plots=export_yearly_timeseries_plots,
                    )
                if export_return_period_plots:
                    discharge_plots.save_station_return_period_plots(
                        discharge_comparison=discharge_comparison,
                        station_id=station_id,
                        eval_plot_folder=evaluation_paths.plot_folder,
                    )
                station_id_text: str = str(station_id)
                assert chart_writer is not None
                chart_writer.add_station(
                    station_id=station_id_text,
                    chart_data=build_station_chart_data(
                        discharge_comparison=discharge_comparison,
                        station_name=station_name,
                        upstream_area_ratio=upstream_area_ratio,
                        timezone_utc_offset=timezone_utc_offset,
                        metrics=station_metrics,
                        frequency=frequency_label,
                        logger=self.model.logger,
                        include_return_period_plots=include_return_period_plots,
                        main_time_index=main_time_index,
                    ),
                )

            station_score_record: dict[str, Any] = {
                "station_ID": station_id,
                "station_name": station_name,
                "station_longitude": station_coordinates[0],
                "station_latitude": station_coordinates[1],
                "routing_grid_longitude": routing_grid_coordinates[0],
                "routing_grid_latitude": routing_grid_coordinates[1],
                "original_subgrid_longitude": original_subgrid_pixel_coordinates[0],
                "original_subgrid_latitude": original_subgrid_pixel_coordinates[1],
                "upstream_area_GRDC": float(
                    station_metadata.discharge_observations_upstream_area_m2
                ),
                "upstream_area_GEB_original_subgrid": float(
                    station_metadata.GEB_upstream_area_from_original_subgrid
                ),
                "upstream_area_GEB": geb_upstream_area_m2,
                "discharge_observations_to_GEB_upstream_area_ratio": upstream_area_ratio,
                "station_to_original_subgrid_distance_m": float(
                    station_metadata.station_to_original_subgrid_distance_m
                ),
                "snapped_river_id": int(station_metadata.snapped_river_id),
                "snapping_method": station_metadata.snapping_method,
                "timezone_utc_offset": timezone_utc_offset,
                "discharge_observations_country_code": station_metadata.get(
                    "discharge_observations_country_code", ""
                ),
                **{
                    f"{metric_name}_{frequency_label}": metric_value
                    for metric_name, metric_value in station_metrics.items()
                },
            }

            if frequency_label == "hourly":
                # Incomplete days must not influence daily benchmark scores.
                daily_resampler: Any = discharge_comparison.resample("D")
                valid_hourly_counts_per_day: pd.DataFrame = daily_resampler.count()
                daily_discharge_comparison: pd.DataFrame = daily_resampler.mean()[
                    valid_hourly_counts_per_day == 24
                ].dropna()
                daily_discharge_metrics: DischargeMetrics = calculate_discharge_metrics(
                    daily_discharge_comparison
                )
                station_score_record.update(
                    {
                        f"{metric_name}_daily": metric_value
                        for metric_name, metric_value in daily_discharge_metrics._asdict().items()
                    }
                )
            elif frequency_label == "daily":
                daily_discharge_comparison = discharge_comparison
            else:
                raise ValueError(
                    f"Unexpected frequency label '{frequency_label}' in evaluation loop."
                )

            seasonal_metrics: dict[str, DischargeMetrics] = (
                calculate_seasonal_discharge_metrics(daily_discharge_comparison)
            )
            station_score_record.update(
                {
                    f"{metric_name}_daily_{season_name}": getattr(
                        season_metrics, metric_name
                    )
                    for season_name, season_metrics in seasonal_metrics.items()
                    for metric_name in SEASONAL_KGE_METRICS
                }
            )

            # Both monthly means must use the same observed time steps.
            monthly_discharge_comparison: pd.DataFrame = (
                discharge_comparison.dropna().resample("ME").mean().dropna()
            )
            monthly_discharge_metrics: DischargeMetrics = calculate_discharge_metrics(
                monthly_discharge_comparison
            )
            station_score_record.update(
                {
                    f"{metric_name}_monthly": metric_value
                    for metric_name, metric_value in monthly_discharge_metrics._asdict().items()
                }
            )

            station_score_records.append(station_score_record)

    station_dashboard_chart_files: dict[str, str] = (
        chart_writer.finish() if chart_writer is not None else {}
    )

    station_scores: pd.DataFrame
    if not station_score_records:
        station_scores = pd.DataFrame(
            columns=[
                "station_name",
                "station_longitude",
                "station_latitude",
                "routing_grid_longitude",
                "routing_grid_latitude",
                "original_subgrid_longitude",
                "original_subgrid_latitude",
                "upstream_area_GRDC",
                "upstream_area_GEB_original_subgrid",
                "upstream_area_GEB",
                "discharge_observations_to_GEB_upstream_area_ratio",
                "station_to_original_subgrid_distance_m",
                "snapped_river_id",
                "snapping_method",
                "timezone_utc_offset",
                "discharge_observations_country_code",
                *DISCHARGE_SCORE_COLUMNS,
            ],
            index=pd.Index([], name="station_ID"),
        )
    else:
        station_scores = pd.DataFrame(station_score_records).set_index("station_ID")

    excluded_station_scores: pd.DataFrame = station_scores.loc[
        station_scores.index.isin(excluded_stations.index)
    ].copy()
    excluded_station_scores["exclusion_reason"] = (
        excluded_stations["exclusion_reason"].reindex(excluded_station_scores.index)
        if not excluded_stations.empty
        else ""
    )
    excluded_scores_with_geometry: gpd.GeoDataFrame = gpd.GeoDataFrame(
        excluded_station_scores,
        geometry=gpd.points_from_xy(
            excluded_station_scores["station_longitude"],
            excluded_station_scores["station_latitude"],
        ),
        crs="EPSG:4326",
    )
    excluded_scores_with_geometry.to_parquet(
        evaluation_paths.metrics_geoparquet.with_name(
            f"diagnostic_metrics{evaluation_paths.suffix}.geoparquet"
        )
    )
    station_scores = station_scores.loc[
        ~station_scores.index.isin(excluded_stations.index)
    ]
    station_scores.to_excel(
        evaluation_paths.metrics_excel,
        index=True,
    )

    station_scores_with_geometry: gpd.GeoDataFrame = gpd.GeoDataFrame(
        station_scores,
        geometry=gpd.points_from_xy(
            station_scores["station_longitude"],
            station_scores["station_latitude"],
        ),
        crs="EPSG:4326",
    )  # create a geodataframe from the evaluation dataframe
    station_scores_with_geometry.to_parquet(
        evaluation_paths.metrics_geoparquet,
    )
    self.model.logger.info(
        "Saved discharge evaluation metrics to %s and %s.",
        evaluation_paths.metrics_excel,
        evaluation_paths.metrics_geoparquet,
    )

    dashboard_station_scores: gpd.GeoDataFrame = gpd.GeoDataFrame(
        pd.concat([station_scores_with_geometry, excluded_scores_with_geometry]),
        geometry="geometry",
        crs=station_scores_with_geometry.crs,
    )
    excluded_stations = find_dashboard_excluded_stations(
        dashboard_station_scores,
        excluded_stations,
        snapped_locations,
        Path(self.model.files["geom"]["discharge/discharge_snapped_locations"]),
        minimum_upstream_area_km2,
    )
    excluded_stations.to_parquet(
        evaluation_paths.metrics_geoparquet.with_name(
            f"excluded_stations{evaluation_paths.suffix}.geoparquet"
        )
    )
    median_skill_scores: dict[str, float | None] = dict.fromkeys(
        DISCHARGE_SCORE_COLUMNS
    )
    if enable_plotting:
        dashboard_geometries: DischargeDashboardGeometries = (
            load_discharge_dashboard_geometries(self.model.files["geom"])
        )
        use_daily_discharge_scores(dashboard_station_scores)
        dashboard_characteristics: pd.DataFrame | None = (
            discharge_characteristics.load_dashboard_catchment_characteristics(
                mapped_station_scores=dashboard_station_scores,
                logger=self.model.logger,
            )
            if not dashboard_station_scores.empty
            else None
        )
        write_discharge_dashboard(
            mapped_station_scores=dashboard_station_scores,
            output_path=dashboard_path,
            region_geom=dashboard_geometries.region,
            rivers=dashboard_geometries.rivers,
            station_chart_files=station_dashboard_chart_files,
            waterbodies=dashboard_geometries.waterbodies,
            station_characteristics=dashboard_characteristics,
            excluded_stations=excluded_stations,
            chart_timeline=chart_timelines,
        )
        self.model.logger.info(
            "Discharge dashboard created. Keep its HTML and charts folder together."
        )

    if not station_scores.empty:
        if enable_plotting:
            self.plot_discharge_skill_scores(
                export=True,
                start_year=start_year,
                end_year=end_year,
                plots=("maps", "boxplots"),
            )

        for metric_column in median_skill_scores:
            if metric_column in station_scores.columns:
                median_skill_scores[metric_column] = float(
                    station_scores[metric_column].median()
                )
    else:
        self.model.logger.warning(
            "No discharge stations found for evaluation. Returning None for all metrics."
        )

    self.model.logger.info(
        "Discharge evaluation completed. Scores: %s", median_skill_scores
    )

    return median_skill_scores
