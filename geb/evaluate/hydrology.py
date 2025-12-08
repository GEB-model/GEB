"""Module implementing hydrology evaluation functions for the GEB model."""

import base64
import os
from pathlib import Path
from typing import Any

import branca.colormap as cm
import contextily as ctx
import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from permetrics.regression import RegressionMetric
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from tqdm import tqdm

from geb.workflows.io import read_zarr, write_zarr


def calculate_hit_rate(model: xr.DataArray, observations: xr.DataArray) -> float:
    """Calculate the hit rate metric.

    Args:
        model: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        Hit rate as a float.
    """
    miss = np.sum(((model == 0) & (observations == 1)).values)
    hit = np.sum(((model == 1) & (observations == 1)).values)
    hit_rate = hit / (hit + miss)
    return float(hit_rate)


def calculate_false_alarm_ratio(
    model: xr.DataArray, observations: xr.DataArray
) -> float:
    """Calculate the false alarm ratio metric.

    Args:
        model: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        False alarm ratio as a float.
    """
    false_alarm = np.sum(((model == 1) & (observations == 0)).values)
    hit = np.sum(((model == 1) & (observations == 1)).values)
    false_alarm_ratio = false_alarm / (false_alarm + hit)
    return float(false_alarm_ratio)


def calculate_critical_success_index(
    model: xr.DataArray, observations: xr.DataArray
) -> float:
    """Calculate the critical success index (CSI) metric.

    Args:
        model: Model flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).
        observations: Observed flood extent as a binary xarray DataArray (1 for flood, 0 for no flood).

    Returns:
        Critical success index as a float.
    """
    hit = np.sum(((model == 1) & (observations == 1)).values)
    false_alarm = np.sum(((model == 1) & (observations == 0)).values)
    miss = np.sum(((model == 0) & (observations == 1)).values)
    csi = hit / (hit + false_alarm + miss)
    return float(csi)


class Hydrology:
    """Implements several functions to evaluate the hydrological module of GEB."""

    def __init__(self) -> None:
        """Initialize the Hydrology evaluation module."""
        pass

    def plot_discharge(
        self, run_name: str = "default", *args: Any, **kwargs: Any
    ) -> None:
        """Plot the mean discharge from the GEB model as a spatial map.

        Creates a spatial visualization of mean discharge values over time from the GEB model
        simulation results. The plot is saved as both a zarr file and PNG image for analysis.

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
            FileNotFoundError: If the discharge file for the specified run does not exist
                in the report directory.
        """
        # check if discharge file exists
        if not (
            self.model.output_folder
            / "report"
            / run_name
            / "hydrology.routing"
            / "discharge_daily.zarr"
        ).exists():
            raise FileNotFoundError(
                f"Discharge file for run '{run_name}' does not exist in the report directory. Did you run the model?"
            )
        # load the discharge simulation
        GEB_discharge = read_zarr(
            self.model.output_folder
            / "report"
            / run_name
            / "hydrology.routing"
            / "discharge_daily.zarr"
        )

        # calculate the mean discharge over time, and plot spatially
        mean_discharge = GEB_discharge.mean(dim="time")
        mean_discharge.attrs["_FillValue"] = np.nan

        write_zarr(
            mean_discharge,
            self.output_folder_evaluate / "mean_discharge_m3_per_s.zarr",
            crs=4326,
        )

        fig, ax = plt.subplots(figsize=(10, 10))

        mean_discharge.plot(ax=ax, cmap="Blues")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.savefig(
            self.output_folder_evaluate / "mean_discharge_m3_per_s.png", dpi=300
        )

    def evaluate_discharge(
        self,
        spinup_name: str = "spinup",
        run_name: str = "default",
        include_spinup: bool = False,
        include_yearly_plots: bool = False,
        correct_Q_obs: bool = False,
    ) -> None:
        """Evaluate the discharge grid from GEB against observations from the Q_obs database.

        Compares simulated discharge from the GEB model with observed discharge data from
        gauging stations. Calculates performance metrics (KGE, NSE, R) and creates
        evaluation plots and interactive maps for analysis.

        Notes:
            The discharge simulation files must exist in the report directory structure.
            If no discharge stations are found in the basin, empty evaluation datasets
            are created. The evaluation can be skipped if results already exist.

        Args:
            spinup_name: Name of the spinup run to include in the evaluation.
            run_name: Name of the simulation run to evaluate. Must correspond to an
                existing run directory in the model output folder.
            include_spinup: Whether to include the spinup run in the evaluation.
            include_yearly_plots: Whether to create plots for every year showing the evaluation.
            correct_Q_obs: Whether to correct the Q_obs discharge timeseries for the difference
                in upstream area between the Q_obs station and the discharge from GEB.

        Raises:
            FileNotFoundError: If the run folder does not exist in the report directory.
        """
        #  create folders
        eval_plot_folder: Path = (
            Path(self.output_folder_evaluate) / "discharge" / "plots"
        )
        eval_result_folder = (
            Path(self.output_folder_evaluate) / "discharge" / "evaluation_results"
        )

        eval_plot_folder.mkdir(parents=True, exist_ok=True)
        eval_result_folder.mkdir(parents=True, exist_ok=True)

        # load input data files
        Q_obs = pd.read_parquet(
            self.model.files["table"]["discharge/Q_obs"]
        )  # load the Q_obs discharge data
        region_shapefile = gpd.read_parquet(
            self.model.files["geom"]["mask"]
        )  # load the region shapefile
        rivers = gpd.read_parquet(
            self.model.files["geom"]["routing/rivers"]
        )  # load the rivers shapefiles
        snapped_locations = gpd.read_parquet(
            self.model.files["geom"]["discharge/discharge_snapped_locations"]
        )

        if len(snapped_locations) == 0:
            print(
                "No discharge stations found in the basin. Creating empty evaluation datasets."
            )

            # Create empty evaluation dataframe with proper structure
            empty_evaluation_df = pd.DataFrame(
                columns=[
                    "station_name",
                    "x",
                    "y",
                    "Q_obs_to_GEB_upstream_area_ratio",
                    "KGE",
                    "NSE",
                    "R",
                ]
            ).set_index(pd.Index([], name="station_ID"))

            # Save empty evaluation metrics as Excel file
            empty_evaluation_df.to_excel(
                eval_result_folder / "evaluation_metrics.xlsx",
                index=True,
            )

            # Create empty GeoDataFrame and save as parquet
            empty_evaluation_gdf = gpd.GeoDataFrame(
                empty_evaluation_df,
                geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                crs="EPSG:4326",
            )
            empty_evaluation_gdf.to_parquet(
                eval_result_folder / "evaluation_metrics.geoparquet",
            )
            return

        # check if evaluation has already been executed
        if eval_result_folder.joinpath("evaluation_metrics.xlsx").exists():
            print(
                "evaluation already executed, skipping. If you want to re-run the discharge evaluation, delete the evaluation_results folder."
            )
            evaluation_df = pd.read_excel(
                eval_result_folder.joinpath("evaluation_metrics.xlsx")
            )
            return
        GEB_discharge = read_zarr(
            self.model.output_folder
            / "report"
            / run_name
            / "hydrology.routing"
            / "discharge_daily.zarr"
        )
        print(f"Loaded discharge simulation from {run_name} run.")

        # check if run file exists, if not, raise an error
        if not (self.model.output_folder / "report" / run_name).exists():
            raise FileNotFoundError(
                f"Run folder '{run_name}' does not exist in the report directory. Did you run the model?"
            )

        evaluation_per_station: list = []

        print("Starting discharge evaluation...")

        for ID in tqdm(Q_obs.columns):
            # create a discharge timeseries dataframe
            discharge_Q_obs_df = Q_obs[ID]
            discharge_Q_obs_df.columns = ["Q"]
            discharge_Q_obs_df.name = "Q"

            # check if there is data in the model time period
            start_date = GEB_discharge.time.min().values
            end_date = GEB_discharge.time.max().values
            data_check = discharge_Q_obs_df[
                (discharge_Q_obs_df.index >= start_date)
                & (discharge_Q_obs_df.index <= end_date)
            ].dropna()  # filter the dataframe to the model time period
            if len(data_check) < 365:
                print(
                    f"Station {ID} has only {len(data_check)} days of data, less than 1 year. Skipping."
                )
                continue

            # extract the properties from the snapping dataframe
            Q_obs_station_name = snapped_locations.loc[ID].Q_obs_station_name
            snapped_xy_coords = snapped_locations.loc[ID].snapped_grid_pixel_xy
            Q_obs_station_coords = snapped_locations.loc[ID].Q_obs_station_coords
            Q_obs_to_GEB_upstream_area_ratio = snapped_locations.loc[
                ID
            ].Q_obs_to_GEB_upstream_area_ratio

            def create_validation_df() -> pd.DataFrame:
                """Create a validation dataframe with the Q_obs discharge observations and the GEB discharge simulation for the selected station.

                Returns:
                    DataFrame with the Q_obs discharge observations and the GEB discharge simulation for the selected station.

                Raises:
                    FileNotFoundError: If the hydrology routing directory does not exist.
                """
                # Check if the hydrology.routing directory exists
                routing_dir = (
                    self.model.output_folder / "report" / run_name / "hydrology.routing"
                )
                if not routing_dir.exists():
                    raise FileNotFoundError(
                        f"Hydrology routing directory does not exist: {routing_dir}"
                    )

                # Construct the path to the individual station discharge file
                station_file_name = f"discharge_hourly_m3_per_s_{ID}.csv"
                station_file_path = routing_dir / station_file_name

                # Check if the station file exists
                if not station_file_path.exists():
                    print(
                        f"WARNING: Station file {station_file_path} does not exist. Skipping station {ID}."
                    )
                    return pd.DataFrame()

                try:
                    # Load the individual station discharge timeseries
                    GEB_discharge_station = pd.read_csv(
                        station_file_path, index_col=0, parse_dates=True
                    )
                    GEB_discharge_station = (
                        GEB_discharge_station.squeeze()
                    )  # Convert to Series if only one column

                    if np.isnan(GEB_discharge_station.values).all():
                        print(
                            f"WARNING: Station {ID} has only NaN values in the GEB discharge simulation. Skipping."
                        )
                        return pd.DataFrame()

                    # Handle spinup data if needed
                    if include_spinup:
                        spinup_station_file_path = (
                            self.model.output_folder
                            / "report"
                            / spinup_name
                            / "hydrology.routing"
                            / station_file_name
                        )
                        if spinup_station_file_path.exists():
                            GEB_discharge_station_spinup = pd.read_csv(
                                spinup_station_file_path, index_col=0, parse_dates=True
                            )
                            GEB_discharge_station_spinup = (
                                GEB_discharge_station_spinup.squeeze()
                            )
                            # Concatenate the spinup and main run data
                            GEB_discharge_station = pd.concat(
                                [GEB_discharge_station_spinup, GEB_discharge_station]
                            )
                            print(f"Loaded spinup data for station {ID}")
                        else:
                            print(
                                f"WARNING: Spinup file for station {ID} not found, using only main run data"
                            )

                    # rename series to Q
                    GEB_discharge_station.name = "Q"
                    discharge_sim_station_df = GEB_discharge_station

                    # merge to one df but keep only the rows where both have data
                    validation_df = pd.merge(
                        discharge_Q_obs_df,
                        discharge_sim_station_df,
                        left_index=True,
                        right_index=True,
                        how="inner",
                        suffixes=("_obs", "_sim"),
                    )  # merge the two dataframes on the index (time)

                    validation_df.dropna(how="any", inplace=True)  # drop rows with nans

                    if correct_Q_obs:
                        """ correct the Q_obs values for the difference in upstream area between subgrid and grid """
                        validation_df["Q_obs"] = (
                            validation_df["Q_obs"] * Q_obs_to_GEB_upstream_area_ratio
                        )  # correct the Q_obs values for the difference in upstream area between subgrid and grid

                    return validation_df

                except Exception as e:
                    print(f"ERROR loading station file {station_file_path}: {e}")
                    return pd.DataFrame()

            validation_df = create_validation_df()

            # Check if validation_df is empty (station was skipped due to all NaN values)
            if validation_df.empty:
                continue

            def calculate_validation_metrics() -> tuple[float, float, float]:
                """Calculate the validation metrics for the current station.

                Returns:
                    KGE: Kling-Gupta Efficiency
                    NSE: Nash-Sutcliffe Efficiency
                    R: Pearson correlation coefficient
                """
                # calculate kupta coefficient
                y_true = validation_df["Q_obs"].values
                y_pred = validation_df["Q_sim"].values
                evaluator = RegressionMetric(y_true, y_pred)  # from permetrics package

                KGE = (
                    evaluator.kling_gupta_efficiency()
                )  # https://hess.copernicus.org/articles/23/4323/2019/

                NSE = evaluator.nash_sutcliffe_efficiency()  # https://hess.copernicus.org/articles/27/1827/2023/hess-27-1827-2023.pdf
                R = evaluator.pearson_correlation_coefficient()

                return KGE, NSE, R

            KGE, NSE, R = calculate_validation_metrics()

            def plot_validation_graphs(ID: Any) -> None:
                """Plot the validation results for the current station."""
                # scatter plot
                fig, ax = plt.subplots()
                ax.scatter(validation_df["Q_obs"], validation_df["Q_sim"])
                ax.set_xlabel(
                    "Q_obs Discharge observations [m3/s] (%s)" % Q_obs_station_name
                )
                ax.set_ylabel("GEB discharge simulation [m3/s]")
                ax.set_title("GEB vs observations (discharge)")
                m, b = np.polyfit(validation_df["Q_obs"], validation_df["Q_sim"], 1)
                ax.plot(
                    validation_df["Q_obs"],
                    m * validation_df["Q_obs"] + b,
                    color="red",
                )
                ax.text(0.02, 0.9, f"$R$ = {R:.2f}", transform=ax.transAxes)
                ax.text(0.02, 0.85, f"KGE = {KGE:.2f}", transform=ax.transAxes)
                ax.text(0.02, 0.8, f"NSE = {NSE:.2f}", transform=ax.transAxes)
                ax.text(
                    0.02,
                    0.75,
                    f"Q_obs to GEB upstream area ratio: {Q_obs_to_GEB_upstream_area_ratio:.2f}",
                    transform=ax.transAxes,
                )

                plt.savefig(
                    eval_plot_folder / f"scatter_plot_{ID}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

                plt.show()
                plt.close()

                # timeseries plot
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(
                    validation_df.index,
                    validation_df["Q_sim"],
                    label="GEB simulation",
                    linewidth=0.5,
                )
                ax.plot(
                    validation_df.index,
                    validation_df["Q_obs"],
                    label="Q_obs observations",
                    linewidth=0.5,
                )
                ax.set_ylabel("Discharge [m3/s]")
                ax.set_xlabel("Time")
                ax.set_ylim(0, None)
                ax.legend()

                ax.text(
                    0.02, 0.9, f"$R^2$={R:.2f}", transform=ax.transAxes, fontsize=12
                )
                ax.text(
                    0.02,
                    0.85,
                    f"KGE={KGE:.2f}",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.text(
                    0.02, 0.8, f"NSE={NSE:.2f}", transform=ax.transAxes, fontsize=12
                )
                ax.text(
                    0.02,
                    0.75,
                    f"Mean={validation_df['Q_sim'].mean():.2f}",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.text(
                    0.02,
                    0.70,
                    f"Q_obs to GEB upstream area ratio: {Q_obs_to_GEB_upstream_area_ratio:.2f}",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                plt.title(
                    f"GEB discharge vs observations for station {Q_obs_station_name}"
                )
                plt.savefig(
                    eval_plot_folder / f"timeseries_plot_{ID}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()
                plt.close()

                # Making yearly plots for every year in validation_df
                # Get available years from validation_df (intersection of obs & sim time range)
                if include_yearly_plots:
                    years_to_plot = sorted(validation_df.index.year.unique())

                    for year in years_to_plot:
                        # Filter data for the current year
                        one_year_df = validation_df[validation_df.index.year == year]

                        # Skip if there's no data for the year
                        if one_year_df.empty:
                            print(f"No data available for year {year}, skipping.")
                            continue

                        # Create the plot
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(
                            one_year_df.index,
                            one_year_df["Q_sim"],
                            label="GEB simulation",
                        )
                        ax.plot(
                            one_year_df.index,
                            one_year_df["Q_obs"],
                            label="Q_obs observations",
                        )
                        ax.set_ylabel("Discharge [m3/s]")
                        ax.set_xlabel("Time")
                        ax.legend()

                        ax.text(
                            0.02,
                            0.9,
                            f"$R^2$={R:.2f}",
                            transform=ax.transAxes,
                            fontsize=12,
                        )
                        ax.text(
                            0.02,
                            0.85,
                            f"KGE={KGE:.2f}",
                            transform=ax.transAxes,
                            fontsize=12,
                        )
                        ax.text(
                            0.02,
                            0.8,
                            f"NSE={NSE:.2f}",
                            transform=ax.transAxes,
                            fontsize=12,
                        )
                        ax.text(
                            0.02,
                            0.75,
                            f"Q_obs to GEB upstream area ratio: {Q_obs_to_GEB_upstream_area_ratio:.2f}",
                            transform=ax.transAxes,
                            fontsize=12,
                        )

                        plt.title(
                            f"GEB discharge vs observations for {year} at station {Q_obs_station_name}"
                        )
                        plt.savefig(
                            eval_plot_folder / f"timeseries_plot_{ID}_{year}.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.show()
                        plt.close()

            plot_validation_graphs(ID)

            # attach to the evaluation dataframe
            evaluation_per_station.append(
                {
                    "station_ID": ID,
                    "station_name": Q_obs_station_name,
                    "x": Q_obs_station_coords[0],
                    "y": Q_obs_station_coords[1],
                    "Q_obs_to_GEB_upstream_area_ratio": Q_obs_to_GEB_upstream_area_ratio,
                    "KGE": KGE,  # https://permetrics.readthedocs.io/en/latest/pages/regression/KGE.html
                    "NSE": NSE,  # https://permetrics.readthedocs.io/en/latest/pages/regression/NSE.html # ranges from -inf to 1.0, where 1.0 is a perfect fit. Values less than 0.36 are considered unsatisfactory, while values between 0.36 to 0.75 are classified as good, and values greater than 0.75 are regarded as very good.
                    "R": R,  # https://permetrics.readthedocs.io/en/latest/pages/regression/R.html
                }
            )

        evaluation_df = pd.DataFrame(evaluation_per_station).set_index("station_ID")
        evaluation_df.to_excel(
            eval_result_folder / "evaluation_metrics.xlsx",
            index=True,
        )

        # Save evaluation metrics as as excel and parquet file
        evaluation_gdf = gpd.GeoDataFrame(
            evaluation_df,
            geometry=gpd.points_from_xy(evaluation_df.x, evaluation_df.y),
            crs="EPSG:4326",
        )  # create a geodataframe from the evaluation dataframe
        evaluation_gdf.to_parquet(
            eval_result_folder / "evaluation_metrics.geoparquet",
        )

        # plot the evaluation metrics (R, KGE, NSE) on a 1x3 subplot
        def plot_validation_map() -> None:
            """Plot the validation results on a map."""
            fig, ax = plt.subplots(1, 3, figsize=(20, 10))

            # Plot evaluation metrics without default colorbars
            evaluation_gdf.plot(
                column="R",
                ax=ax[0],
                legend=False,  # Disable default colorbar
                cmap="viridis",
                markersize=50,
                zorder=3,
            )
            evaluation_gdf.plot(
                column="KGE",
                ax=ax[1],
                legend=False,  # Disable default colorbar
                cmap="viridis",
                markersize=50,
                zorder=3,
            )
            evaluation_gdf.plot(
                column="NSE",
                ax=ax[2],
                legend=False,  # Disable default colorbar
                cmap="viridis",
                markersize=50,
                zorder=3,
            )

            # Add the region shapefile and rivers to each subplot
            region_shapefile.plot(
                ax=ax[0], color="none", edgecolor="black", linewidth=1, zorder=2
            )
            region_shapefile.plot(
                ax=ax[1], color="none", edgecolor="black", linewidth=1, zorder=2
            )
            region_shapefile.plot(
                ax=ax[2], color="none", edgecolor="black", linewidth=1, zorder=2
            )

            rivers.plot(ax=ax[0], color="blue", linewidth=0.5, zorder=2)
            rivers.plot(ax=ax[1], color="blue", linewidth=0.5, zorder=2)
            rivers.plot(ax=ax[2], color="blue", linewidth=0.5, zorder=2)

            # Add satellite basemap to each subplot without attribution text
            ctx.add_basemap(
                ax[0],
                crs=evaluation_gdf.crs.to_string(),
                source=ctx.providers.Esri.WorldImagery,
                attribution=False,  # Remove attribution text
            )
            ctx.add_basemap(
                ax[1],
                crs=evaluation_gdf.crs.to_string(),
                source=ctx.providers.Esri.WorldImagery,
                attribution=False,  # Remove attribution text
            )
            ctx.add_basemap(
                ax[2],
                crs=evaluation_gdf.crs.to_string(),
                source=ctx.providers.Esri.WorldImagery,
                attribution=False,  # Remove attribution text
            )

            # Titles
            ax[0].set_title("R")
            ax[1].set_title("KGE")
            ax[2].set_title("NSE")

            # Set axis labels
            ax[0].set_xlabel("Longitude")
            ax[0].set_ylabel("Latitude")
            ax[1].set_xlabel("Longitude")
            ax[2].set_xlabel("Longitude")

            # Create custom colorbars
            R_colorbar = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=mcolors.Normalize(
                    vmin=evaluation_gdf.R.min(), vmax=evaluation_gdf.R.max()
                ),
            )

            KGE_colorbar = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=mcolors.Normalize(
                    vmin=evaluation_gdf.KGE.min(), vmax=evaluation_gdf.KGE.max()
                ),
            )

            NSE_colorbar = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=mcolors.Normalize(
                    vmin=evaluation_gdf.NSE.min(), vmax=evaluation_gdf.NSE.max()
                ),
            )

            # Add custom colorbars and move them down
            fig.colorbar(
                R_colorbar,
                ax=ax[0],
                orientation="horizontal",
                pad=0.1,  # Move colorbar down
                aspect=50,
                label="R",
            )
            fig.colorbar(
                KGE_colorbar,
                ax=ax[1],
                orientation="horizontal",
                pad=0.1,  # Move colorbar down
                aspect=50,
                label="KGE",
            )
            fig.colorbar(
                NSE_colorbar,
                ax=ax[2],
                orientation="horizontal",
                pad=0.1,  # Move colorbar down
                aspect=50,
                label="NSE",
            )

            # Set the layout of the subplots
            plt.tight_layout()

            # Save the plot
            plt.savefig(
                eval_result_folder / "discharge_evaluation_metrics.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()
            plt.close()
            # plt.close()

        plot_validation_map()

        def create_folium_map(evaluation_gdf: gpd.GeoDataFrame) -> folium.Map:
            """Create a Folium map with evaluation results and station markers.

            Returns:
                Folium Map object with evaluation results.
            """
            # Create a Folium map centered on the mean coordinates of the stations
            map_center = [
                evaluation_gdf.geometry.y.mean(),
                evaluation_gdf.geometry.x.mean(),
            ]
            m = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB positron")

            # Create colormaps for R, KGE, and NSE (Red → Orange → Yellow → Blue → Green)
            colormap_r = cm.LinearColormap(
                colors=[
                    "red",
                    "orange",
                    "yellow",
                    "blue",
                    "green",
                ],  # Updated color scheme
                vmin=evaluation_gdf["R"].min(),
                vmax=evaluation_gdf["R"].max(),
                caption="R",
            )
            colormap_kge = cm.LinearColormap(
                colors=[
                    "red",
                    "orange",
                    "yellow",
                    "blue",
                    "green",
                ],  # Updated color scheme
                vmin=evaluation_gdf["KGE"].min(),
                vmax=evaluation_gdf["KGE"].max(),
                caption="KGE",
            )
            colormap_nse = cm.LinearColormap(
                colors=[
                    "red",
                    "orange",
                    "yellow",
                    "blue",
                    "green",
                ],  # Updated color scheme
                vmin=evaluation_gdf["NSE"].min(),
                vmax=evaluation_gdf["NSE"].max(),
                caption="NSE",
            )

            # Add colormaps to the map
            colormap_r.add_to(m)
            colormap_kge.add_to(m)
            colormap_nse.add_to(m)

            if not evaluation_gdf["Q_obs_to_GEB_upstream_area_ratio"].isna().all():
                colormap_upstream = cm.LinearColormap(
                    colors=[
                        "red",
                        "orange",
                        "yellow",
                        "blue",
                        "green",
                    ],  # Updated color scheme
                    vmin=evaluation_gdf["Q_obs_to_GEB_upstream_area_ratio"].min(),
                    vmax=evaluation_gdf["Q_obs_to_GEB_upstream_area_ratio"].max(),
                    caption="Upstream Area Ratio",
                )
                colormap_upstream.add_to(m)
                layer_upstream = folium.FeatureGroup(
                    name="Upstream Area Ratio", show=False
                )

            # Create FeatureGroups for R, KGE, and NSE
            layer_r = folium.FeatureGroup(name="R", show=True)
            layer_kge = folium.FeatureGroup(name="KGE", show=False)
            layer_nse = folium.FeatureGroup(name="NSE", show=False)

            # Add markers for R, KGE, and NSE to their respective layers
            for station_ID, row in evaluation_gdf.iterrows():
                coords = [row.geometry.y, row.geometry.x]
                station_name = row["station_name"]

                # Generate scatter plot for the station
                scatter_plot_path = eval_plot_folder / f"scatter_plot_{station_ID}.png"
                time_series_plot_path = (
                    eval_plot_folder / f"timeseries_plot_{station_ID}.png"
                )
                # Encode the scatter plot image as a base64 string
                with open(scatter_plot_path, "rb") as img_file:
                    encoded_image_scatter = base64.b64encode(img_file.read()).decode(
                        "utf-8"
                    )
                with open(time_series_plot_path, "rb") as img_file:
                    encoded_image_time_series = base64.b64encode(
                        img_file.read()
                    ).decode("utf-8")

                # Create an HTML popup with the 2 plots
                popup_html = f"""
                <b>Station Name:</b> {station_name}<br>
                <b>R:</b> {row["R"]:.2f}<br>
                <b>KGE:</b> {row["KGE"]:.2f}<br>
                <b>NSE:</b> {row["NSE"]:.2f}<br>
                <b>Upstream Area Ratio:</b> {row["Q_obs_to_GEB_upstream_area_ratio"]:.2f}<br>
                <img src="data:image/png;base64,{encoded_image_scatter}" width="500">
                <img src="data:image/png;base64,{encoded_image_time_series}" width="500">
                """

                # Add R layer
                color_r = colormap_r(row["R"])
                popup_r = folium.Popup(popup_html, max_width=400)
                folium.CircleMarker(
                    location=coords,
                    radius=10,
                    color="black",
                    fill=True,
                    fill_color=color_r,  # Use R colormap for color
                    fill_opacity=0.9,
                    popup=popup_r,
                ).add_to(layer_r)

                # Add KGE layer
                color_kge = colormap_kge(row["KGE"])
                popup_kge = folium.Popup(popup_html, max_width=400)
                folium.CircleMarker(
                    location=coords,
                    radius=10,
                    color="black",
                    fill=True,
                    fill_color=color_kge,
                    fill_opacity=0.9,
                    popup=popup_kge,
                ).add_to(layer_kge)

                # Add NSE layer
                color_nse = colormap_nse(row["NSE"])
                popup_nse = folium.Popup(popup_html, max_width=400)
                folium.CircleMarker(
                    location=coords,
                    radius=10,
                    color="black",
                    fill=True,
                    fill_color=color_nse,
                    fill_opacity=0.9,
                    popup=popup_nse,
                ).add_to(layer_nse)

                if not evaluation_gdf["Q_obs_to_GEB_upstream_area_ratio"].isna().all():
                    # Add Upstream Area Ratio layer
                    color_upstream = colormap_upstream(
                        float(row["Q_obs_to_GEB_upstream_area_ratio"])
                    )
                    if not isinstance(color_upstream, (str)) or color_upstream == "nan":
                        # do not add to map if color is NaN
                        continue

                    popup_upstream = folium.Popup(popup_html, max_width=400)
                    folium.CircleMarker(
                        location=coords,
                        radius=10,
                        color="black",
                        fill=True,
                        fill_color=color_upstream,
                        fill_opacity=0.9,
                        popup=popup_upstream,
                    ).add_to(layer_upstream)

            # Add the layers to the map
            layer_r.add_to(m)
            layer_kge.add_to(m)
            layer_nse.add_to(m)

            # Add the colormaps to the map
            colormap_r.add_to(m)
            colormap_kge.add_to(m)
            colormap_nse.add_to(m)

            # add upstream area (if not ONLY nans)
            if not evaluation_gdf["Q_obs_to_GEB_upstream_area_ratio"].isna().all():
                layer_upstream.add_to(m)
                colormap_upstream.add_to(m)

            # Add the catchment shapefile as a GeoJSON layer
            folium.GeoJson(
                region_shapefile,
                name="Catchment",
                style_function=lambda x: {
                    "fillColor": "blue",
                    "color": "blue",
                    "weight": 1,
                    "fillOpacity": 0.2,
                },
            ).add_to(m)

            # Add rivers as a GeoJSON layer
            folium.GeoJson(
                rivers["geometry"],
                name="Rivers",
                style_function=lambda x: {"color": "blue", "weight": 1},
            ).add_to(m)

            # Add a layer control to toggle layers
            folium.LayerControl().add_to(m)

            # Save the map to an HTML file
            m.save(eval_result_folder / "discharge_evaluation_map.html")

            # Display the map in a Jupyter Notebook (if applicable)
            return m

        create_folium_map(evaluation_gdf)

        print("Discharge evaluation dashboard created.")

    def skill_score_graphs(
        self,
        export: bool = True,
    ) -> None:
        """Create skill score boxplot graphs for hydrological model evaluation metrics.

        Generates boxplot visualizations of discharge evaluation metrics (KGE, NSE, R)
        from previously calculated station evaluations. Creates a plot
        showing the distribution of performance metrics across all gauging stations.

        Notes:
            Requires evaluation metrics to exist from a previous `evaluate_discharge` run.
            If no discharge stations are found for evaluation, the method will skip
            graph creation and return early.

        Args:
            export: Whether to save the skill score graphs to PNG files.
        """
        eval_result_folder = (
            Path(self.output_folder_evaluate) / "discharge" / "evaluation_results"
        )
        evaluation_df = pd.read_excel(
            eval_result_folder.joinpath("evaluation_metrics.xlsx")
        )

        # Check if evaluation dataframe is empty
        if evaluation_df.empty:
            print(
                "No discharge stations found for evaluation. Skipping skill score graphs."
            )
            return

        # Create fancy boxplots for evaluation metrics
        print("Creating evaluation metrics boxplots...")

        # Prepare data for boxplots
        metrics = [
            {
                "data": evaluation_df["KGE"].dropna(),
                "label": "KGE\n(Kling-Gupta)",
                "color": "#2E86AB",
            },
            {
                "data": evaluation_df["NSE"].dropna(),
                "label": "NSE\n(Nash-Sutcliffe)",
                "color": "#A23B72",
            },
            {
                "data": evaluation_df["R"].dropna(),
                "label": "R\n(Correlation)",
                "color": "#F18F01",
            },
        ]

        # Create the figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle(
            "Hydrological Model Evaluation Metrics", fontsize=16, fontweight="bold"
        )

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Create boxplot
            bp = ax.boxplot(
                metric["data"],
                patch_artist=True,
                medianprops={"color": "white", "linewidth": 2},
                boxprops={"linewidth": 1.5},
                whiskerprops={"linewidth": 1.5},
                capprops={"linewidth": 1.5},
            )

            # Color the box
            bp["boxes"][0].set_facecolor(metric["color"])
            bp["boxes"][0].set_alpha(0.7)

            # Add title and styling
            ax.set_title(metric["label"], fontsize=12, fontweight="bold")
            ax.set_ylabel("Score", fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks([])

            # Set specific y-limits for each metric
            if i == 0:  # KGE
                ax.set_ylim(0, 1)
            elif i == 1:  # NSE
                ax.set_ylim(-1, 1)
            # R (correlation) keeps automatic limits

        plt.tight_layout()

        # Save the plot
        if export:
            boxplot_path = eval_result_folder / "evaluation_boxplots_simple.png"
            plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
            print(f"Boxplots saved to: {boxplot_path}")

        plt.show()
        plt.close()

        print("Skill score graphs created.")

    def water_circle(
        self,
        run_name: str,
        include_spinup: bool,
        spinup_name: str,
        *args: Any,
        export: bool = True,
        **kwargs: Any,
    ) -> go.Figure:
        """Create a water circle plot for the GEB model.

        Adapted from: https://github.com/mikhailsmilovic/flowplot
        Also see the paper: https://doi.org/10.1088/1748-9326/ad18de

        This method installs a headless version of Chrome if not already available,

        Args:
            run_name: Name of the run to evaluate.
            include_spinup: Whether to include the spinup run in the evaluation.
            spinup_name: Name of the spinup run to include in the evaluation.
            export: Whether to export the water circle plot to a file.
            *args: ignored.
            **kwargs: ignored.

        Returns:
            A Plotly Figure object representing the water circle.
        """
        import plotly.io as pio

        # auto install chrome if not available
        pio.get_chrome()

        folder = self.model.output_folder / "report" / run_name

        def read_csv_with_date_index(
            folder: Path, module: str, name: str, skip_first_day: bool = True
        ) -> pd.Series:
            """Read a CSV file with a date index.

            Args:
                folder: Path to the folder containing the CSV file.
                module: Name of the module (subfolder) containing the CSV file.
                name: Name of the CSV file (without extension).
                skip_first_day: Whether to skip the first day of the time series.

            Returns:
                A pandas Series with the date index and the values from the CSV file.

            """
            df = pd.read_csv(
                (folder / module / name).with_suffix(".csv"),
                index_col=0,
                parse_dates=True,
            )[name]

            if skip_first_day:
                df = df.iloc[1:]

            return df

        # because storage is the storage at the end of the timestep, we need to calculate the change
        # across the entire simulation period. For all other variables we do skip the first day.
        storage = read_csv_with_date_index(
            folder, "hydrology", "_water_circle_storage", skip_first_day=False
        )
        storage_change = storage.iloc[-1] - storage.iloc[0]

        rain = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_circle_rain"
        ).sum()
        snow = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_circle_snow"
        ).sum()

        domestic_water_loss = read_csv_with_date_index(
            folder, "hydrology.water_demand", "_water_circle_domestic_water_loss"
        ).sum()
        industry_water_loss = read_csv_with_date_index(
            folder, "hydrology.water_demand", "_water_circle_industry_water_loss"
        ).sum()
        livestock_water_loss = read_csv_with_date_index(
            folder, "hydrology.water_demand", "_water_circle_livestock_water_loss"
        ).sum()

        river_outflow = read_csv_with_date_index(
            folder, "hydrology.routing", "_water_circle_river_outflow"
        ).sum()

        transpiration = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_circle_transpiration"
        ).sum()
        bare_soil_evaporation = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_circle_bare_soil_evaporation"
        ).sum()
        open_water_evaporation = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_circle_open_water_evaporation"
        ).sum()
        interception_evaporation = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_circle_interception_evaporation"
        ).sum()
        sublimation_or_deposition = read_csv_with_date_index(
            folder, "hydrology.landsurface", "_water_circle_sublimation_or_deposition"
        ).sum()
        river_evaporation = read_csv_with_date_index(
            folder, "hydrology.routing", "_water_circle_river_evaporation"
        ).sum()
        waterbody_evaporation = read_csv_with_date_index(
            folder, "hydrology.routing", "_water_circle_waterbody_evaporation"
        ).sum()

        hierarchy: dict[str, Any] = {
            "in": {
                "rain": rain,
                "snow": snow,
            },
            "out": {
                "evapotranspiration": {
                    "transpiration": transpiration,
                    "bare soil evaporation": bare_soil_evaporation,
                    "open water evaporation": open_water_evaporation,
                    "interception evaporation": interception_evaporation,
                    "river evaporation": river_evaporation,
                    "waterbody evaporation": waterbody_evaporation,
                },
                "water demand": {
                    "domestic water loss": domestic_water_loss,
                    "industry water loss": industry_water_loss,
                    "livestock water loss": livestock_water_loss,
                },
                "river outflow": river_outflow,
            },
            "storage change": abs(storage_change),
        }

        if sublimation_or_deposition > 0:
            hierarchy["in"]["deposition"] = sublimation_or_deposition
        else:
            hierarchy["out"]["evapotranspiration"]["sublimation"] = abs(
                sublimation_or_deposition
            )

        # the size of a section is the sum of the flows in that section
        # plus the size of the section itself. So if all of the section
        # is made up of its children, the size of the section is 0.
        water_circle_list: list[
            tuple[str | None, str | None, str, float | int, str]
        ] = []
        color_map: dict[str, str] = {
            "in": "#636EFA",
            "out": "#EF5538",
            "balance": "#000000",
            "storage change": "#D2D2D3",
        }

        def add_flow(
            water_circle_list: list[
                tuple[str | None, str | None, str, float | int, str]
            ],
            color_map: dict[str, str],
            root_section: str | None,
            parent: str | None,
            flow: str | None,
            value: int | float | dict[str, Any],
        ) -> tuple[
            list[tuple[str | None, str | None, str, float | int, str]], dict[str, str]
        ]:
            """Recursive function to add flows to the water circle list.

            Args:
                water_circle_list: List of tuples containing the water circle data with parent, flow, and value.
                color_map: Dictionary mapping flow names to colors.
                root_section: Root section of the current flow hierarchy.
                parent: Parent of the current flow section.
                flow: Name of the current flow section.
                value: Value of the current flow section, can be a number or a dictionary.
                    If a number, it is a flow and added to the water circle list immediately.
                    If a dictionary, it contains sub-sections, and it is processed recursively.

                    If one of the sections is _self, it is the size of the remainder section itself.
                    This is useful when not all of the section is made up of its children.

            Raises:
                ValueError: If the value type is not int, float, or dict.

            Returns:
                Updated water circle list with the new flow added.
                Updated color map with the new flow color added.
            """
            if parent is not None and flow is None:
                raise ValueError("Flow name cannot be None if parent is not None.")
            elif flow is not None and parent is None:
                raise ValueError("Parent cannot be None if flow name is not None.")

            if isinstance(value, (int, float)):  # stopping condition
                # adopt the color of the parent if it exists
                assert flow is not None
                if parent is not None:
                    color_map[flow] = color_map[parent]
                else:  # if no parent, this is a root section
                    root_section = flow
                water_circle_list.append(
                    (root_section, parent, flow, value, color_map[flow])
                )
            elif isinstance(value, dict):
                if parent is not None:  # adopt the color of the parent
                    assert flow is not None
                    color_map[flow] = color_map[parent]
                else:  # if no parent, this is a root section
                    root_section = flow
                _self = 0
                for sub_section, sub_value in value.items():
                    if sub_section == "_self":
                        _self = sub_value
                        continue  # skip the _self section
                    water_circle_list, color_map = add_flow(
                        water_circle_list,
                        color_map,
                        root_section,
                        flow,
                        sub_section,
                        sub_value,
                    )
                if flow is not None:
                    water_circle_list.append(
                        (root_section, parent, flow, _self, color_map[flow])
                    )
            else:
                raise ValueError(
                    f"Invalid value type for section '{flow}': {value}. Expected dict, int, or float."
                )

            return water_circle_list, color_map

        water_circle_list, _ = add_flow(
            water_circle_list,
            color_map,
            root_section=None,
            parent=None,
            flow=None,
            value=hierarchy,
        )

        water_circle_df: pd.DataFrame = pd.DataFrame(
            water_circle_list,
            columns=np.array(["root_section", "parent", "flow", "value", "color"]),
        )

        if storage_change > 0:
            category_order = ["in", "out", "storage change"]
        else:
            category_order = ["storage change", "in", "out"]

        water_circle_df["root_section"] = pd.Categorical(
            water_circle_df["root_section"],
            categories=category_order,
            ordered=True,
        )
        # sort the sections with storage change first
        water_circle_df = water_circle_df.sort_values(
            by=["root_section", "value"],
            ascending=[True, False],
        )

        water_circle = go.Figure(
            go.Sunburst(
                labels=water_circle_df["flow"],
                parents=water_circle_df["parent"],
                values=water_circle_df["value"],
                sort=False,
                marker=dict(colors=water_circle_df["color"]),
            )
        )

        water_circle.update_layout(margin=dict(l=20, r=20, t=20, b=45))
        water_circle.update_layout(template="plotly_dark")
        water_circle.update_layout(
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            title=dict(
                text="water circle",
                xanchor="center",
                yanchor="bottom",
                y=0.04,
                x=0.5,
            ),
        )

        if export:
            water_circle.write_image(
                self.output_folder_evaluate / "water_circle.png", scale=5
            )

        return water_circle

    def evaluate_hydrodynamics(
        self, run_name: str = "default", *args: Any, **kwargs: Any
    ) -> None:
        """Evaluate hydrodynamic model performance against flood observations.

        Calculates performance metrics (hit rate, false alarm ratio, critical success index)
        for flood maps generated by the hydrodynamic model by comparing them against
        observed flood extent data.

        Args:
            run_name: Name of the simulation run to evaluate.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            FileNotFoundError: If the flood map folder does not exist in the output directory.
            ValueError: If the flood observation file is not in .zarr format.
        """

        def parse_flood_forecast_initialisation(
            filename: str,
        ) -> tuple[str | None, str | None, str, str, str]:
            """Parse flood map filename to extract components.

            Expected format: YYYYMMDDTHHMMSS - MEMBER - EVENT_START - EVENT_END.zarr

            Args:
                filename: Name of the flood map file.

            Returns:
                Tuple containing (forecast_init, member, event_start, event_end, event_name)

            Raises:
                ValueError: If the filename does not match the expected format.
            """
            # Remove .zarr extension
            name_without_ext = filename.replace(".zarr", "")

            # Split by ' - ' to get components
            parts = name_without_ext.split(" - ")

            if len(parts) >= 4:
                # Handle case with forecasts included
                forecast_init = parts[0]  # First 17 characters: YYYYMMDDTHHMMSS
                member = parts[1]  # Member number
                event_start = parts[2]  # Event start time
                event_end = parts[3]  # Event end time

                # Create event name from start and end times
                event_name = f"{event_start} - {event_end}"

            elif len(parts) == 2:
                # Handle case with no forecasts included
                forecast_init = None
                member = None
                event_start = parts[0]  # Event start time
                event_end = parts[1]  # Event end time

                # Create event name from start and end times
                event_name = f"{event_start} - {event_end}"

            else:
                # Raise error for files that don't match expected format
                raise ValueError(
                    f"Filename '{filename}' does not match expected format for flood map."
                )

            return forecast_init, member, event_start, event_end, event_name

        # Main function for the performance metrics
        def calculate_performance_metrics(
            observation: Path | str,
            flood_map_path: Path | str,
            output_folder: Path,
            visualization_type: str = "Hillshade",
        ) -> dict[str, float | int] | None:
            """Calculate performance metrics for flood maps against observations.

            Args:
                observation: Path to the observed flood extent data (.zarr format).
                flood_map_path: Path to the model-generated flood map data (.zarr format).
                visualization_type: Type of visualization for plotting (default is "Hillshade").
                output_folder: Path to the folder where results will be saved.

            Returns:
                Dictionary containing performance metrics:
                    - hit_rate: Percentage of correctly predicted flooded areas.
                    - false_alarm_ratio: Percentage of falsely predicted flooded areas.
                    - critical_success_index: Overall accuracy of flood predictions.
                    - flooded_area_km2: Total flooded area in square kilometers.
                or None if an error occurs.

            Raises:
                ValueError: If the observation file is not in .zarr format.
            """
            # Step 1: Open needed datasets
            flood_map = read_zarr(flood_map_path)
            obs = read_zarr(observation)
            print("obs CRS", obs.rio.crs)
            sim = flood_map.rio.reproject_match(obs)
            rivers = gpd.read_parquet(
                Path("simulation_root")
                / run_name
                / "SFINCS"
                / "run"
                / "segments.geoparquet"
            )
            region = gpd.read_file(
                Path("simulation_root")
                / run_name
                / "SFINCS"
                / "run"
                / "gis"
                / "region.geojson"
            ).to_crs(obs.rio.crs)

            # Step 2: Clip out rivers from observations and simulations
            crs_wgs84 = CRS.from_epsg(4326)
            crs_mercator = CRS.from_epsg(3857)
            rivers.set_crs(crs_wgs84, inplace=True)
            gdf_mercator = rivers.to_crs(crs_mercator)
            gdf_mercator["geometry"] = gdf_mercator.buffer(gdf_mercator["width"] / 2)

            # Create river mask for simulation data
            gdf_buffered_sim = gdf_mercator.to_crs(sim.rio.crs)
            rivers_mask_sim = ~geometry_mask(
                gdf_buffered_sim.geometry,
                out_shape=sim.rio.shape,
                transform=sim.rio.transform(),
                all_touched=True,
                invert=False,
            )
            sim_no_rivers = sim.where(~rivers_mask_sim).fillna(0)

            # Create river mask for observation data
            gdf_buffered_obs = gdf_mercator.to_crs(obs.rio.crs)
            rivers_mask_obs = ~geometry_mask(
                gdf_buffered_obs.geometry,
                out_shape=obs.rio.shape,
                transform=obs.rio.transform(),
                all_touched=True,
                invert=False,
            )
            obs_no_rivers = obs.where(~rivers_mask_obs).fillna(0)

            # Step 3: Clip out region from observations
            obs_region = obs_no_rivers.rio.clip(region.geometry.values, region.crs)

            # Step 4: Optionally clip using extra validation region from config yml
            extra_validation_path = self.config["floods"].get(
                "extra_validation_region", None
            )

            if extra_validation_path and Path(extra_validation_path).exists():
                extra_clip_region = gpd.read_file(extra_validation_path).set_crs(28992)
                extra_clip_region = extra_clip_region.to_crs(region.crs)
                extra_clip_region_buffer = extra_clip_region.buffer(160)

                sim_extra_clipped = sim_no_rivers.rio.clip(
                    extra_clip_region_buffer.geometry.values,
                    extra_clip_region_buffer.crs,
                )
                clipped_out = (sim_no_rivers > 0.15) & (sim_extra_clipped.isnull())
                clipped_out_raster = sim_no_rivers.where(clipped_out)
            else:
                # If no extra validation region, skip clipping
                sim_extra_clipped = sim_no_rivers
                clipped_out_raster = xr.full_like(sim_no_rivers, np.nan)

            # Step 5: Mask water depth values
            hmin = 0.15
            sim_extra_clipped = sim_extra_clipped.raster.reproject_like(obs_region)
            simulation_final = sim_extra_clipped > hmin
            observation_final = obs_region > 0

            xmin, ymin, xmax, ymax = region.total_bounds
            catchment_extent = [xmin, xmax, ymin, ymax]

            xmin, ymin, xmax, ymax = observation_final.rio.bounds()
            flood_extent: tuple[float, float, float, float] = (xmin, xmax, ymin, ymax)

            # Step 6: Calculate performance metrics
            # Compute the arrays first to get concrete values
            sim_final_computed = simulation_final.compute()
            obs_final_computed = observation_final.compute()

            hit_rate = calculate_hit_rate(sim_final_computed, obs_final_computed) * 100
            false_rate = (
                calculate_false_alarm_ratio(sim_final_computed, obs_final_computed)
                * 100
            )
            csi = (
                calculate_critical_success_index(sim_final_computed, obs_final_computed)
                * 100
            )

            flooded_pixels = float(sim_final_computed.sum().item())

            # Calculate resolution in meters from coordinate spacing
            x_res = float(np.abs(flood_map.x[1] - flood_map.x[0]))
            pixel_size = x_res  # meters
            flooded_area_km2 = flooded_pixels * (pixel_size * pixel_size) / 1_000_000

            # Step 7: Save results to file and plot the results
            elevation_data = read_zarr(self.model.files["other"]["DEM/fabdem"])
            elevation_data = elevation_data.rio.reproject_match(obs)

            elevation_array = (
                elevation_data.squeeze().astype("float32").compute().values
            )

            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(elevation_array, vert_exag=1, dx=1, dy=1)

            # Catchment borders
            target_crs = obs_region.rio.crs
            catchment_borders = region.boundary

            if simulation_final.sum() > 0:
                misses = (observation_final == 1) & (simulation_final == 0)
                simulation_masked = simulation_final.where(simulation_final == 1)
                hits = simulation_masked.where(observation_final)
                misses_masked = misses.where(misses == 1)

                if visualization_type == "OSM":
                    # Ensure all arrays have the same shape by squeezing extra dimensions
                    simulation_final = simulation_final.squeeze()
                    observation_final = observation_final.squeeze()
                    misses = misses.squeeze()
                    hits = hits.squeeze()
                    simulation_masked = simulation_masked.squeeze()
                    misses_masked = misses_masked.squeeze()

                    margin = 3000
                    fig, ax = plt.subplots(figsize=(10, 10))

                    # clipped_out_raster.plot(
                    #     ax=ax, cmap=blue_cmap, add_colorbar=False, add_labels=False
                    # )

                    ax.imshow(
                        simulation_masked,  # False alarms
                        extent=flood_extent,
                        # origin="lower",
                        cmap="Wistia",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=1,
                    )
                    ax.imshow(
                        misses_masked,  # Misses
                        extent=flood_extent,
                        # origin="lower",
                        cmap="autumn_r",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=2,
                    )

                    ax.imshow(
                        hits,
                        extent=flood_extent,
                        # origin="lower",
                        cmap="brg",
                        vmin=0,
                        vmax=1,
                        interpolation="none",
                        zorder=3,
                    )

                    catchment_borders.plot(
                        ax=ax,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        zorder=4,
                        alpha=0.5,
                    )
                    # Add a base map
                    ctx.add_basemap(
                        ax,
                        crs=target_crs,
                        source="OpenStreetMap.Mapnik",
                        zoom=12,
                        zorder=0,
                        alpha=0.9,
                    )

                    ax.set_title("Validation of the Predicted Flood Areas", fontsize=14)
                    ax.set_xlabel("x [m]")
                    ax.set_ylabel("y [m]")

                    # Set the extent based on the raster bounds
                    ax.set_xlim(
                        catchment_extent[0] - margin, catchment_extent[1] + margin
                    )
                    ax.set_ylim(
                        catchment_extent[2] - margin, catchment_extent[3] + margin
                    )
                    ax.set_aspect("equal", adjustable="box")

                    green_patch = mpatches.Patch(color="#94f944", label="Hits")
                    orange_patch = mpatches.Patch(color="orange", label="False Alarms")
                    red_patch = mpatches.Patch(color="red", label="Misses")

                    catchment_patch = Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        label="Catchment Border",
                    )
                    legend = ax.legend(
                        handles=[
                            green_patch,
                            orange_patch,
                            red_patch,
                            catchment_patch,
                        ]
                    )

                    # Add a comment about the metrics in the plot
                    legend_bbox = legend.get_window_extent(
                        renderer=fig.canvas.get_renderer()
                    )
                    legend_bbox_ax = legend_bbox.transformed(ax.transAxes.inverted())

                    # Add text below legend using axes coordinates
                    ax.annotate(
                        f"Validation Metrics:\n"
                        f"HR    = {hit_rate:.2f} %\n"
                        f"FAR   = {false_rate:.2f} %\n"
                        f"CSI   = {csi:.2f} %",
                        xy=(legend_bbox_ax.x0 + 0.055, legend_bbox_ax.y0 + 0.002),
                        xycoords="axes fraction",
                        fontsize=10,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="grey",
                            boxstyle="round,pad=0.2",
                            alpha=0.8,
                        ),
                        verticalalignment="top",
                        horizontalalignment="left",
                        zorder=5,
                    )

                    crs_text = f"CRS: {target_crs.to_string()}"
                    ax.annotate(
                        crs_text,
                        xy=(0.99, 0.02),  # Bottom right corner in axes coordinates
                        xycoords="axes fraction",
                        fontsize=8,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="grey",
                            boxstyle="round,pad=0.2",
                            alpha=0.8,
                        ),
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        zorder=6,
                    )

                    simulation_filename = os.path.splitext(
                        os.path.basename(flood_map_path)
                    )[0]
                    fig.savefig(
                        output_folder
                        / f"{simulation_filename}_validation_floodextent_plot.png",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    print(
                        f"Figure with {visualization_type} saved as: {output_folder / f'{simulation_filename}_validation_floodextent_plot.png'}"
                    )

                elif visualization_type == "Hillshade":
                    green_cmap = mcolors.ListedColormap(["green"])  # Hits
                    orange_cmap = mcolors.ListedColormap(["orange"])  # False alarms
                    red_cmap = mcolors.ListedColormap(["red"])  # Misses
                    blue_cmap = mcolors.ListedColormap(
                        ["#72c1db"]
                    )  # No observation data

                    fig, ax = plt.subplots(figsize=(10, 10))

                    ax.imshow(
                        hillshade,
                        cmap="gray",
                        extent=(
                            elevation_data.x.min(),
                            elevation_data.x.max(),
                            elevation_data.y.min(),
                            elevation_data.y.max(),
                        ),
                    )

                    region.boundary.plot(
                        ax=ax, edgecolor="black", linewidth=2, label="Region Boundary"
                    )

                    clipped_out_raster.plot(
                        ax=ax, cmap=blue_cmap, add_colorbar=False, add_labels=False
                    )
                    simulation_masked.plot(
                        ax=ax, cmap=orange_cmap, add_colorbar=False, add_labels=False
                    )
                    hits.plot(
                        ax=ax, cmap=green_cmap, add_colorbar=False, add_labels=False
                    )
                    misses_masked.plot(
                        ax=ax, cmap=red_cmap, add_colorbar=False, add_labels=False
                    )

                    ax.set_aspect("equal")
                    ax.axis("off")
                    handles = [
                        mpatches.Patch(color=green_cmap(0.5)),
                        mpatches.Patch(color=red_cmap(0.5)),
                        mpatches.Patch(color=orange_cmap(0.5)),
                        mpatches.Patch(color=blue_cmap(0.5)),
                        mlines.Line2D([], [], color="black", linewidth=2),
                    ]

                    labels = [
                        "Hits",
                        "Misses",
                        "False alarms",
                        "No observation data",
                        "Region",
                    ]

                    plt.legend(
                        handles=handles, labels=labels, loc="upper right", fontsize=16
                    )

                    simulation_filename = os.path.splitext(
                        os.path.basename(flood_map_path)
                    )[0]
                    plt.savefig(
                        output_folder
                        / f"{simulation_filename}_validation_floodextent_plot.png"
                    )
                    print(
                        f"Figure with {visualization_type} saved as: {output_folder / f'{simulation_filename}_validation_floodextent_plot.png'}"
                    )

                else:
                    raise ValueError(
                        f"Unknown visualization type: {visualization_type}, choose either 'OSM' or 'Hillshade'."
                    )

                performance_numbers = (
                    output_folder / f"{simulation_filename}_performance_metrics.txt"
                )

                with open(performance_numbers, "w") as f:
                    f.write(f"Hit rate (H): {hit_rate}\n")
                    f.write(f"False alarm rate (F): {false_rate}\n")
                    f.write(f"Critical Success Index (CSI) (C): {csi}\n")
                    f.write(f"Number of flooded pixels: {flooded_pixels}\n")
                    f.write(f"Flooded area (km2): {flooded_area_km2}")

                # Return metrics for further analysis
                return {
                    "hit_rate": hit_rate,
                    "false_alarm_rate": false_rate,
                    "csi": csi,
                    "flooded_area_km2": flooded_area_km2,
                }

        def create_forecast_performance_plots(
            performance_df: pd.DataFrame, event_name: str, output_folder: Path
        ) -> None:
            """Create performance metric plots showing spread and mean across forecast initializations.

            Generates line plots showing how performance metrics vary across different
            forecast initialization times for a given flood event. Shows both the spread
            of the ensemble (min-max range) with transparent fill and the ensemble mean as a black line.

            Notes:
                The spread is calculated using the minimum and maximum values across ensemble
                members for each forecast initialization. If only one member exists for a
                forecast initialization, no spread is shown for that time point.

            Args:
                performance_df: DataFrame containing performance metrics with forecast
                    initialization times and ensemble members. Must include columns:
                    'forecast_init', 'hit_rate', 'false_alarm_rate', 'csi', 'flooded_area_km2'.
                event_name: Name of the flood event being analyzed.
                output_folder: Directory to save the performance plots.

            Raises:
                ValueError: If required columns are missing from performance_df.
            """
            # Validate required columns
            required_columns = [
                "forecast_init",
                "hit_rate",
                "false_alarm_rate",
                "csi",
                "flooded_area_km2",
            ]
            missing_columns = [
                col for col in required_columns if col not in performance_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in performance_df: {missing_columns}"
                )

            # Group by forecast initialization and calculate statistics
            grouped_stats = (
                performance_df.groupby("forecast_init")[
                    ["hit_rate", "false_alarm_rate", "csi", "flooded_area_km2"]
                ]
                .agg(["mean", "min", "max", "std", "count"])
                .reset_index()
            )

            # Convert forecast_init to datetime for better plotting
            grouped_stats["forecast_init_dt"] = pd.to_datetime(
                grouped_stats["forecast_init"], format="%Y%m%dT%H%M%S"
            )

            # Sort by datetime for proper line plotting
            grouped_stats = grouped_stats.sort_values("forecast_init_dt")

            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Forecast Performance Metrics with Ensemble Spread - {event_name}",
                fontsize=16,
                fontweight="bold",
            )

            # Define colors for spread (transparent) and metric-specific colors
            spread_colors = {
                "hit_rate": "#2E86AB",
                "false_alarm_rate": "#A23B72",
                "csi": "#F18F01",
                "flooded_area_km2": "#636EFA",
            }

            # Hit Rate
            ax = axes[0, 0]
            metric = "hit_rate"

            # Plot spread (min-max range) with transparent fill
            ax.fill_between(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "min")],
                grouped_stats[(metric, "max")],
                color=spread_colors[metric],
                alpha=0.3,
                label="Ensemble member range (Min-Max)",
            )

            # Plot mean as black line
            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="o",
                markersize=6,
                label="Ensemble Mean",
            )

            ax.set_title("Hit Rate (%)", fontweight="bold")
            ax.set_ylabel("Hit Rate (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # False Alarm Rate
            ax = axes[0, 1]
            metric = "false_alarm_rate"

            ax.fill_between(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "min")],
                grouped_stats[(metric, "max")],
                color=spread_colors[metric],
                alpha=0.3,
                label="Ensemble member range (Min-Max)",
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="s",
                markersize=6,
                label="Ensemble Mean",
            )

            ax.set_title("False Alarm Rate (%)", fontweight="bold")
            ax.set_ylabel("False Alarm Rate (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # Critical Success Index
            ax = axes[1, 0]
            metric = "csi"

            ax.fill_between(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "min")],
                grouped_stats[(metric, "max")],
                color=spread_colors[metric],
                alpha=0.3,
                label="Ensemble member range (Min-Max)",
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="^",
                markersize=6,
                label="Ensemble Mean",
            )

            ax.set_title("Critical Success Index (%)", fontweight="bold")
            ax.set_ylabel("CSI (%)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()

            # Flooded Area
            ax = axes[1, 1]
            metric = "flooded_area_km2"

            ax.fill_between(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "min")],
                grouped_stats[(metric, "max")],
                color=spread_colors[metric],
                alpha=0.3,
                label="Ensemble member range (Min-Max)",
            )

            ax.plot(
                grouped_stats["forecast_init_dt"],
                grouped_stats[(metric, "mean")],
                color="black",
                linewidth=2,
                marker="d",
                markersize=6,
                label="Ensemble Mean",
            )

            ax.set_title("Flooded Area (km²)", fontweight="bold")
            ax.set_ylabel("Area (km²)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Format x-axis for all subplots
            for ax in axes.flat:
                ax.tick_params(axis="x", rotation=45)
                ax.set_xlabel("Forecast Initialization Time")

            plt.tight_layout()

            # Save the plot
            plot_filename = (
                f"{event_name.replace(':', '_')}_forecast_performance_spread.png"
            )
            fig.savefig(output_folder / plot_filename, dpi=300, bbox_inches="tight")
            print(
                f"Forecast performance spread plot saved as: {output_folder / plot_filename}"
            )

        self.config = self.model.config["hazards"]

        eval_hydrodynamics_folders = Path(self.output_folder_evaluate) / "hydrodynamics"

        eval_hydrodynamics_folders.mkdir(parents=True, exist_ok=True)

        # Calculate performance metrics for every event in config file
        for event in self.config["floods"]["events"]:
            event_name = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}"
            print(f"event: {event_name}")

            # Create event-specific folder
            if self.model.config["general"]["forecasts"]["use"]:
                event_folder = eval_hydrodynamics_folders / "forecasts" / event_name
                event_folder.mkdir(parents=True, exist_ok=True)
                flood_maps_folder = (
                    self.model.output_folder / "flood_maps" / "forecasts"
                )
            else:
                event_folder = eval_hydrodynamics_folders / event_name
                event_folder.mkdir(parents=True, exist_ok=True)
                flood_maps_folder = self.model.output_folder / "flood_maps"

            # check if run file exists, if not, raise an error

            if not flood_maps_folder.exists():
                raise FileNotFoundError(
                    "Flood map folder does not exist in the output directory. Did you run the hydrodynamic model?"
                )

            # check if observation file exists, if not, raise an error
            if not Path(self.config["floods"]["event_observation_file"]).exists():
                raise FileNotFoundError(
                    "Flood observation file is not found in the given path in the model.yml Please check the path in the config file."
                )
            if Path(self.config["floods"]["event_observation_file"]).suffix != ".zarr":
                raise ValueError(
                    "Flood observation file is not in the correct format. Please provide a .zarr file."
                )

            # Find all flood maps corresponding to the event
            all_flood_map_files = list(flood_maps_folder.glob("*.zarr"))

            # Filter flood_map_files for the current event only
            flood_map_files = []
            for flood_map_path in all_flood_map_files:
                forecast_init, member, event_start, event_end, parsed_event_name = (
                    parse_flood_forecast_initialisation(flood_map_path.name)
                )
                # Check if file matches current event
                if parsed_event_name == event_name:
                    flood_map_files.append(flood_map_path)

            print(
                f"Found {len(flood_map_files)} flood map files for event {event_name}"
            )

            if len(flood_map_files) == 1:
                print(
                    "Only one flood map found, assuming no forecasts were included in the simulation."
                )
                flood_map_name = f"{event['start_time'].strftime('%Y%m%dT%H%M%S')} - {event['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
                flood_map_path = (
                    Path(self.model.output_folder) / "flood_maps" / flood_map_name
                )
                calculate_performance_metrics(
                    observation=self.config["floods"]["event_observation_file"],
                    flood_map_path=flood_map_path,
                    output_folder=event_folder,
                    visualization_type="OSM",
                )
                print(f"Successfully evaluated: {flood_map_path.name}")

            elif len(flood_map_files) == 0:
                raise FileNotFoundError(
                    "No flood map files found for this event. Did you run the hydrodynamic model?"
                )

            else:
                print(
                    f"Multiple flood maps found ({len(flood_map_files)}), processing each."
                )
                unique_forecast_inits = set()
                performance_metrics_list = []

                # Identify unique forecast initializations
                for flood_map_name in flood_map_files:
                    # Parse the flood map filename to extract components
                    print(f"flood_map_name: {flood_map_name}")
                    forecast_init, member, event_start, event_end, parsed_event_name = (
                        parse_flood_forecast_initialisation(flood_map_name.name)
                    )
                    unique_forecast_inits.add(forecast_init)

                # Convert to sorted list for consistent processing order
                unique_forecast_inits_list = sorted(
                    [init for init in unique_forecast_inits if init is not None]
                )
                print(
                    f"Found {len(unique_forecast_inits_list)} unique forecast initializations: {unique_forecast_inits_list}"
                )

                # Process each unique forecast initialization
                for forecast_init in unique_forecast_inits_list:
                    print(f"Processing forecast initialization: {forecast_init}")

                    # Create forecast initialization folder
                    forecast_folder = event_folder / forecast_init
                    forecast_folder.mkdir(parents=True, exist_ok=True)

                    matching_flood_maps = []
                    for flood_map_path in flood_map_files:
                        file_forecast_init, _, _, _, parsed_event_name = (
                            parse_flood_forecast_initialisation(flood_map_path.name)
                        )
                        # Only include files that match current forecast init and event
                        if (
                            file_forecast_init == forecast_init
                            and parsed_event_name == event_name
                        ):
                            matching_flood_maps.append(flood_map_path)

                    print(
                        f"Found {len(matching_flood_maps)} flood maps for forecast initialization {forecast_init}"
                    )
                    # Evaluate each matching flood map
                    forecast_metrics_list = []

                    for flood_map_path in matching_flood_maps:
                        print(f"   Evaluating: {flood_map_path.name}")

                        metrics = calculate_performance_metrics(
                            observation=self.config["floods"]["event_observation_file"],
                            flood_map_path=flood_map_path,
                            visualization_type="OSM",
                            output_folder=forecast_folder,
                        )
                        print("   Flood map evaluation complete.")
                        # Add metadata to metrics
                        forecast_init_parsed, member, _, _, _ = (
                            parse_flood_forecast_initialisation(flood_map_path.name)
                        )
                        metrics_with_metadata = {
                            "forecast_init": forecast_init_parsed,
                            "member": member,
                            "filename": flood_map_path.name,
                            **metrics,
                        }

                        performance_metrics_list.append(metrics_with_metadata)
                        forecast_metrics_list.append(metrics)
                        print(f"   Successfully evaluated: {flood_map_path.name}")

                if performance_metrics_list:
                    performance_df = pd.DataFrame(performance_metrics_list)

                    # Create forecast performance plots
                    create_forecast_performance_plots(
                        performance_df, event_name, event_folder
                    )

                    # Save detailed performance metrics
                    detailed_filename = f"{event_name.replace(':', '_')}_detailed_performance_metrics.csv"
                    performance_df.to_csv(event_folder / detailed_filename, index=False)
                    print(
                        f"Detailed performance metrics saved as: {event_folder / detailed_filename}"
                    )

            print(f"Completed processing event: {event_name}\n")

        print("Flood map performance metrics calculated for all events.")
