"""Coordinates the hydrology evaluation workflow modules.

Workflow modules do the data loading, calculations, dashboards, and plots.
"""

from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
<<<<<<< HEAD
=======
import xarray as xr
from matplotlib import colormaps as mcolormaps
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# from scores.continuous import (
#     kge as calculate_kge,
#     nse as calculate_nse,
#     rmse as calculate_rmse,
# )
from tqdm import tqdm
>>>>>>> a0b65a00 (Early Warning Module (#882))

from geb.evaluate.workflows import (
    dashboard,
    discharge_evaluation,
    discharge_helpers,
    discharge_plots,
    discharge_publication,
    external_skill_scores,
    water_balance_plots,
)
from geb.evaluate.workflows.water_balance_helpers import (
    _get_datetime_index_step_label as _get_datetime_index_step_label,
)

if TYPE_CHECKING:
    from geb.evaluate import Evaluate
    from geb.model import GEBModel


class Hydrology:
    """Expose discharge, dashboard, water-balance, and storage evaluation commands."""

    def __init__(self, model: GEBModel, evaluator: Evaluate) -> None:
        """Initialize the Hydrology evaluation module."""
        self.model = model
        self.evaluator = evaluator

    # Discharge evaluation and data access
    evaluate_discharge = discharge_evaluation.evaluate_discharge
    get_discharge_per_river = discharge_helpers.get_discharge_per_river

<<<<<<< HEAD
    # Discharge plots (load and match external scores automatically)
    plot_discharge = discharge_plots.plot_discharge
    plot_discharge_skill_scores = discharge_plots.plot_discharge_skill_scores
    plot_skill_score_maps = partialmethod(plot_discharge_skill_scores, plots=("maps",))
    plot_skill_score_boxplots = partialmethod(
        plot_discharge_skill_scores, plots=("boxplots",)
    )
    plot_skill_scores_vs_upstream_area = partialmethod(
        plot_discharge_skill_scores, plots=("upstream_area",)
    )
    plot_discharge_characteristics = partialmethod(
        plot_discharge_skill_scores, plots=("characteristics",)
    )
=======
        Args:
            run_name: Name of the simulation run to evaluate. Must correspond to an existing
                run directory in the model output folder.

        Raises:
            FileNotFoundError: If the discharge file for the specified run does not exist
                in the report directory.

        Returns:
            A GeoDataFrame containing the river geometries and a DataFrame containing the discharge data for each river.
        """
        # check if discharge files exists
        discharge_folder: Path = (
            self.evaluator.output_folder_evaluate.parent
            / "report"
            / "hydrology.routing"
        )
        if not discharge_folder.exists():
            raise FileNotFoundError(
                f"Discharge files for run '{run_name}' does not exist in the report directory. Did you run the model?"
            )

        # load rivers
        all_rivers: gpd.GeoDataFrame = read_geom(
            self.model.files["geom"]["routing/rivers"]
        )
        rivers_of_interest: gpd.GeoDataFrame = all_rivers[
            ~(
                all_rivers["is_downstream_outflow"]
                | all_rivers["is_upstream_of_downstream_basin"]
                | all_rivers["is_further_downstream_outflow"]
            )
        ].copy()

        # In merged multi-cluster runs some rivers may not have output files, mostly caused by the outflow reporter to be false in the model.yml. Filter out those rivers here.
        rivers_of_interest = rivers_of_interest[
            rivers_of_interest.index.map(
                lambda rid: (
                    discharge_folder / f"river_outflow_hourly_m3_per_s_{rid}.parquet"
                ).exists()
            )
        ].copy()

        discharge: pd.DataFrame = get_discharge_per_river(
            folder=discharge_folder,
            rivers=rivers_of_interest,
            all_rivers=all_rivers,
        )
        return rivers_of_interest, discharge

    def plot_discharge(
        self,
        run_name: str = "default",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot the mean discharge map and all exported outflow time series.

        Creates a spatial visualization of mean discharge values over time from
        the GEB model simulation results. If outflow-point reporter files are
        available, the method also creates one time-series plot per outflow
        point in the hydrology evaluation output folder.

        Notes:
            The discharge data must exist in the report directory structure. If the discharge
            file is not found, a FileNotFoundError will be raised. The mean is calculated
            across the entire simulation time period.

        Args:
            run_name: Name of the simulation run to plot. Must correspond to an existing
                run directory in the model output folder.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            FileNotFoundError: If the hydrology routing report folder is missing.
        """
        if self.discharge_output_folder.exists():
            shutil.rmtree(self.discharge_output_folder)
        self.discharge_output_folder.mkdir(parents=True, exist_ok=True)

        discharge_folder: Path = (
            self.evaluator.output_folder_evaluate.parent
            / "report"
            / "hydrology.routing"
        )
        if not discharge_folder.exists():
            raise FileNotFoundError(
                f"Discharge files for run '{run_name}' does not exist in the report directory. Did you run the model?"
            )

        all_rivers: gpd.GeoDataFrame = read_geom(
            self.model.files["geom"]["routing/rivers"]
        )
        rivers_of_interest: gpd.GeoDataFrame = all_rivers[
            ~(
                all_rivers["is_downstream_outflow"]
                | all_rivers["is_upstream_of_downstream_basin"]
                | all_rivers["is_further_downstream_outflow"]
            )
        ].copy()
        rivers_of_interest = rivers_of_interest[
            rivers_of_interest.index.map(
                lambda river_id: (
                    discharge_folder
                    / f"river_outflow_hourly_m3_per_s_{river_id}.parquet"
                ).exists()
            )
        ].copy()
        discharge: pd.DataFrame = read_discharge_per_river(
            folder=discharge_folder,
            rivers=rivers_of_interest,
            all_rivers=all_rivers,
        )
        for river_id in discharge.columns:
            rivers_of_interest.loc[river_id, "discharge_m3_per_s"] = discharge[
                river_id
            ].mean()

        ax = rivers_of_interest.plot(
            column="discharge_m3_per_s",
            cmap="Blues",
            legend=True,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Mean discharge (m3/s)")

        plt.savefig(
            self.discharge_output_folder / "mean_discharge_m3_per_s_map.svg",
        )
        plt.close()

        outflow_plot_count: int = _plot_outflow_discharge_timeseries(
            model=self.model,
            output_folder=self.model.output_folder,
            run_name=run_name,
            eval_plot_folder=self.discharge_output_folder,
        )
        if outflow_plot_count > 0:
            self.model.logger.info(
                f"Created {outflow_plot_count} outflow discharge plots."
            )

    def evaluate_discharge(
        self,
        run_name: str = "default",
        include_yearly_plots: bool = True,
        correct_discharge_observations: bool = False,
        create_plots: bool = True,
        include_return_period_plots: bool = False,
        minimum_upstream_area_km2: float | None = None,
        minimum_timeseries_length_years: float | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        clean_output: bool = False,
    ) -> dict[str, float | None]:
        """Evaluate the discharge grid from GEB against observations from the discharge observations database.

        Compares simulated discharge from the GEB model with observed discharge data from
        gauging stations. Calculates discharge skill scores and creates
        evaluation plots and interactive maps for analysis.

        Notes:
            The discharge simulation files must exist in the report directory structure.
            If no discharge stations are found in the basin, empty evaluation datasets
            are created. The evaluation can be skipped if results already exist.

        Args:
            run_name: Name of the simulation run to evaluate. Must correspond to an
                existing run directory in the model output folder.
            include_yearly_plots: Whether to save one discharge PNG per station
                and calendar year.
            correct_discharge_observations: Whether to correct the discharge observations discharge timeseries for the difference
                in upstream area between the discharge observations station and the discharge from GEB.
            create_plots: Whether to create evaluation plots. Set to False to only calculate the evaluation metrics and save the results without plotting.
            include_return_period_plots: Whether to fit extreme-value models and
                create detailed station return-period plots. Defaults to `False`
                because these plots are expensive for large station collections.
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

        Returns:
            Dictionary containing median discharge skill scores. In addition, the returned dictionary contains
            frequency-specific metrics (e.g., KGE_hourly, KGE_daily).
            Stations with hourly data are also evaluated on the daily resampled data, and those metrics are included in
            the returned dictionary. Stations with only daily data are not evaluated on the hourly data.

        Raises:
            FileNotFoundError: If the run folder does not exist in the report directory.
            ValueError: If a non-existing frequency label is encountered in the discharge observations data.
        """
        output_folder = self.evaluate_discharge_output_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        if clean_output:
            shutil.rmtree(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        evaluation_paths: DischargeEvaluationPaths = _get_discharge_evaluation_paths(
            output_folder=output_folder,
            start_year=start_year,
            end_year=end_year,
        )
        if create_plots:
            evaluation_paths.plot_folder.mkdir(parents=True, exist_ok=True)
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

        # load input data files
        discharge_observations_hourly: pd.DataFrame = read_table(
            self.model.files["table"]["discharge/discharge_observations_hourly"]
        )
        discharge_observations_daily: pd.DataFrame = read_table(
            self.model.files["table"]["discharge/discharge_observations_daily"]
        )
        if not discharge_observations_hourly.empty:
            discharge_observations_hourly = regularize_discharge_timeseries(
                discharge_observations_hourly
            )
        if not discharge_observations_daily.empty:
            discharge_observations_daily = regularize_discharge_timeseries(
                discharge_observations_daily
            )

        snapped_locations = read_geom(
            self.model.files["geom"]["discharge/discharge_snapped_locations"]
        )

        self.model.logger.info(f"Loaded discharge simulation from {run_name} run.")

        report_folder: Path = self.model.output_folder / "report"
        if not report_folder.exists():
            raise FileNotFoundError(
                f"Run folder '{run_name}' does not exist in the report directory. Did you run the model?"
            )

        evaluation_per_station: list[dict[str, Any]] = []
        station_dashboard_chart_files: dict[str, str] = {}

        self.model.logger.info("Starting discharge evaluation...")
        for frequency_label, discharge_observations_df in zip(
            ["hourly", "daily"],
            [
                discharge_observations_hourly,
                discharge_observations_daily,
            ],
            strict=True,
        ):
            if discharge_observations_df.empty:
                continue
            for station_id in tqdm(discharge_observations_df.columns):
                # create a discharge timeseries dataframe
                observed_discharge_series = discharge_observations_df[station_id]
                if isinstance(observed_discharge_series, pd.DataFrame):
                    observed_discharge_series.columns = ["Q"]
                observed_discharge_series.name = "Q"

                # extract the properties from the snapping dataframe
                discharge_observations_station_name = snapped_locations.loc[
                    station_id
                ].discharge_observations_station_name
                discharge_observations_station_coords = snapped_locations.loc[
                    station_id
                ].discharge_observations_station_coords
                discharge_observations_to_GEB_upstream_area_ratio = (
                    snapped_locations.loc[
                        station_id
                    ].discharge_observations_to_GEB_upstream_area_ratio
                )
                geb_upstream_area_m2: float = float(
                    snapped_locations.at[station_id, "GEB_upstream_area_from_grid"]
                )
                if geb_upstream_area_m2 < minimum_upstream_area_km2 * 1_000_000.0:
                    # Smaller catchments tend to be dominated by local timing and snapping
                    # errors, so the default benchmark excludes them from summary scores.
                    continue

                timezone_utc_offset: float = float(
                    snapped_locations.at[station_id, "timezone_utc_offset"]
                    if "timezone_utc_offset" in snapped_locations.columns
                    else 0.0
                )

                try:
                    validation_df: pd.DataFrame = create_validation_df(
                        self.model.output_folder,
                        run_name,
                        station_id,
                        observed_discharge_series,
                        correct_discharge_observations,
                        discharge_observations_to_GEB_upstream_area_ratio,
                        timezone_utc_offset=timezone_utc_offset,
                    )
                except FileNotFoundError:
                    self.model.logger.warning(
                        "Skipping station %s: no simulation output found.", station_id
                    )
                    continue
                validation_df = _filter_validation_df_to_years(
                    validation_df=validation_df,
                    start_year=start_year,
                    end_year=end_year,
                )

                minimum_valid_steps = (
                    minimum_timeseries_length_years
                    * 365.25
                    * (24 if frequency_label == "hourly" else 1)
                )
                if validation_df.dropna().shape[0] < minimum_valid_steps:
                    continue

                discharge_metrics = _calculate_discharge_validation_metrics(
                    validation_df
                )
                discharge_metric_values: dict[str, float] = discharge_metrics._asdict()

                if create_plots:
                    save_discharge_timeseries_plots(
                        station_id=station_id,
                        validation_df=validation_df,
                        station_name=discharge_observations_station_name,
                        upstream_area_ratio=discharge_observations_to_GEB_upstream_area_ratio,
                        metrics=discharge_metric_values,
                        plot_folder=evaluation_paths.plot_folder,
                        include_yearly_plots=include_yearly_plots,
                    )
                    if include_return_period_plots:
                        _plot_validation_return_periods(
                            validation_df=validation_df,
                            station_id=station_id,
                            station_name=discharge_observations_station_name,
                            eval_plot_folder=evaluation_paths.plot_folder,
                            frequency=frequency_label,
                        )
                    station_id_text: str = str(station_id)
                    station_dashboard_chart_files[station_id_text] = (
                        write_discharge_dashboard_chart_data(
                            dashboard_path=dashboard_path,
                            station_id=station_id_text,
                            chart_data=build_discharge_dashboard_chart_data(
                                validation_df=validation_df,
                                station_name=discharge_observations_station_name,
                                upstream_area_ratio=discharge_observations_to_GEB_upstream_area_ratio,
                                metrics=discharge_metric_values,
                                frequency=frequency_label,
                            ),
                        )
                    )

                station_evaluation: dict[str, Any] = {
                    "station_ID": station_id,
                    "station_name": discharge_observations_station_name,
                    "x": discharge_observations_station_coords[0],
                    "y": discharge_observations_station_coords[1],
                    "discharge_observations_to_GEB_upstream_area_ratio": discharge_observations_to_GEB_upstream_area_ratio,
                    "upstream_area_GEB": geb_upstream_area_m2,
                    "timezone_utc_offset": timezone_utc_offset,
                    **discharge_metric_values,
                    **{
                        f"{metric_name}_{frequency_label}": metric_value
                        for metric_name, metric_value in discharge_metric_values.items()
                    },
                }

                # if the frequency is hourly, also calculate the metrics on the daily resampled data
                if frequency_label == "hourly":
                    # Resample to daily, keeping only days with 24 valid hourly observations.
                    daily_resampler: Any = validation_df.resample("D")
                    valid_hourly_counts_per_day = daily_resampler.count()
                    validation_df_daily = daily_resampler.mean()[
                        valid_hourly_counts_per_day == 24
                    ].dropna()
                    daily_discharge_metrics = _calculate_discharge_validation_metrics(
                        validation_df_daily
                    )
                    station_evaluation.update(
                        {
                            f"{metric_name}_daily": metric_value
                            for metric_name, metric_value in daily_discharge_metrics._asdict().items()
                        }
                    )
                # Daily-frequency stations have no hourly data; fill with NaN so the
                # DataFrame columns remain consistent across all stations.
                elif frequency_label == "daily":
                    station_evaluation.update(
                        {
                            f"{metric_name}_hourly": float("nan")
                            for metric_name in DischargeMetrics._fields
                        }
                    )
                else:
                    raise ValueError(
                        f"Unexpected frequency label '{frequency_label}' in evaluation loop."
                    )

                # Monthly metrics
                validation_df_monthly = validation_df.resample("ME").mean().dropna()
                monthly_discharge_metrics = _calculate_discharge_validation_metrics(
                    validation_df_monthly
                )
                station_evaluation.update(
                    {
                        f"{metric_name}_monthly": metric_value
                        for metric_name, metric_value in monthly_discharge_metrics._asdict().items()
                    }
                )

                # attach to the evaluation dataframe
                evaluation_per_station.append(station_evaluation)

        if not evaluation_per_station:
            # Create empty evaluation dataframe with proper structure
            # Column names are derived from DischargeMetrics so they stay in sync.
            freq_cols: list[str] = [
                f"{metric_name}_{frequency}"
                for frequency in ("monthly", "daily", "hourly")
                for metric_name in DischargeMetrics._fields
            ]
            evaluation_df = pd.DataFrame(
                columns=[
                    "station_name",
                    "x",
                    "y",
                    "upstream_area_GEB",
                    "discharge_observations_to_GEB_upstream_area_ratio",
                    *freq_cols,
                    *DischargeMetrics._fields,
                ],
                index=pd.Index([], name="station_ID"),
            )
        else:
            evaluation_df = pd.DataFrame(evaluation_per_station).set_index("station_ID")

        evaluation_df.to_excel(
            evaluation_paths.xlsx,
            index=True,
        )

        # Save evaluation metrics as as excel and parquet file
        evaluation_gdf = gpd.GeoDataFrame(
            evaluation_df,
            geometry=gpd.points_from_xy(evaluation_df.x, evaluation_df.y),
            crs="EPSG:4326",
        )  # create a geodataframe from the evaluation dataframe
        evaluation_gdf.to_parquet(
            evaluation_paths.geoparquet,
        )
        self.model.logger.info(
            "Saved discharge evaluation metrics to %s and %s.",
            evaluation_paths.xlsx,
            evaluation_paths.geoparquet,
        )

        # Return median metrics if available
        if not evaluation_df.empty:
            if create_plots:
                dashboard_geometries: DischargeDashboardGeometries = (
                    load_discharge_dashboard_geometries(self.model)
                )

                create_discharge_folium_map(
                    evaluation_gdf=evaluation_gdf,
                    output_path=dashboard_path,
                    region_geom=dashboard_geometries.region,
                    rivers=dashboard_geometries.rivers,
                    station_chart_files=station_dashboard_chart_files,
                    waterbodies=dashboard_geometries.waterbodies,
                )

                self.model.logger.info("Discharge evaluation dashboard created.")
                self.model.logger.info(
                    "Tip: If station charts do not appear, download the dashboard "
                    "HTML and its charts folder to the same local directory."
                )

                # GEB standalone plot — all GEB stations.
                self.plot_skill_score_boxplots(
                    export=True,
                    start_year=start_year,
                    end_year=end_year,
                )
                self.plot_skill_score_maps(
                    export=True,
                    start_year=start_year,
                    end_year=end_year,
                )

            scores: dict[str, float | None] = {
                **{
                    f"{metric_name}_{frequency}": float(
                        evaluation_df[f"{metric_name}_{frequency}"].median()
                    )
                    for frequency in ("hourly", "daily", "monthly")
                    for metric_name in DischargeMetrics._fields
                },
                **{
                    metric_name: float(evaluation_df[metric_name].median())
                    for metric_name in DischargeMetrics._fields
                },
            }
        else:
            self.model.logger.warning(
                "No discharge stations found for evaluation. Returning None for all metrics."
            )

            scores: dict[str, float | None] = {
                **{
                    f"{metric_name}_{frequency}": None
                    for frequency in ("hourly", "daily", "monthly")
                    for metric_name in DischargeMetrics._fields
                },
                **{metric_name: None for metric_name in DischargeMetrics._fields},
            }

        self.model.logger.info(f"Discharge evaluation completed. Scores: {scores}")

        return scores
>>>>>>> a0b65a00 (Early Warning Module (#882))

    def create_discharge_dashboard(
        self,
        run_name: str = "default",
        correct_discharge_observations: bool = False,
        output_filename: str = "discharge_evaluation_map.html",
        include_return_period_plots: bool = True,
    ) -> dict[str, str]:
        """Create a dashboard from saved discharge evaluation metrics.

        Args:
            run_name: Simulation run to display.
            correct_discharge_observations: Apply the station-to-model area ratio.
            output_filename: Filename relative to the evaluation folder, or an absolute path.
            include_return_period_plots: Include return-period curves; defaults to True.

        Returns:
            Path to the created dashboard, keyed by ``dashboard``.
        """
        return dashboard.create_discharge_dashboard(
            evaluation_folder=self.evaluate_discharge_output_folder,
            run_output_folder=self.evaluator.output_folder_evaluate.parent,
            geometry_files=self.model.files["geom"],
            table_files=self.model.files["table"],
            minimum_upstream_area_km2=self.model.config["hydrology"]["evaluation"][
                "discharge"
            ]["minimum_upstream_area_km2"],
            logger=self.model.logger,
            correct_discharge_observations=correct_discharge_observations,
            output_filename=output_filename,
            include_return_period_plots=include_return_period_plots,
        )

    # Optional data exports; these are not prerequisites for plotting
    export_discharge_publication_data = (
        discharge_publication.export_discharge_publication_data
    )

    def export_external_skill_scores(self, **kwargs: Any) -> dict[str, pd.DataFrame]:
        """Export external skill scores matched to this model's stations.

        Args:
            **kwargs: Evaluation CLI arguments; no export settings are required.

        Returns:
            Matched station scores keyed by external model name.
        """
        return external_skill_scores.export_external_skill_scores(
            input_folder=self.model.input_folder,
            output_folder=self.evaluate_discharge_output_folder,
            snapped_locations_path=self.model.files["geom"][
                "discharge/discharge_snapped_locations"
            ],
            logger=self.model.logger,
        )

    prepare_external_evaluation = export_external_skill_scores

    # Water-circle, water-balance, and water-storage plots
    plot_water_circle = water_balance_plots.plot_water_circle
    plot_water_balance = water_balance_plots.plot_water_balance
    plot_water_storage = water_balance_plots.plot_water_storage

    # Output folders
    @property
    def discharge_output_folder(self) -> Path:
        """Path to the folder where discharge map outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "discharge"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def evaluate_discharge_output_folder(self) -> Path:
        """Path to the folder where discharge evaluation outputs are stored."""
        folder = (
            self.evaluator.output_folder_evaluate / "hydrology" / "evaluate_discharge"
        )
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_circle_output_folder(self) -> Path:
        """Path to the folder where water circle outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_circle"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_balance_output_folder(self) -> Path:
        """Path to the folder where water balance outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_balance"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def water_storage_output_folder(self) -> Path:
        """Path to the folder where water storage outputs are stored."""
        folder = self.evaluator.output_folder_evaluate / "hydrology" / "water_storage"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
