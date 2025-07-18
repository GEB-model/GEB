import base64
from pathlib import Path
from typing import Any

import branca.colormap as cm
import contextily as ctx
import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from permetrics.regression import RegressionMetric
from tqdm import tqdm

from geb.workflows.io import open_zarr, to_zarr


class Hydrology:
    """Implements several functions to evaluate the hydrological module of GEB."""

    def __init__(self):
        pass

    def plot_discharge(self, run_name: str = "default", *args, **kwargs) -> None:
        """Method to plot the mean discharge from the GEB model.

        Args:
            run_name: Defaults to "default".
            *args: ignored.
            **kwargs: ignored.
        """
        # load the discharge simulation
        GEB_discharge = open_zarr(
            self.model.output_folder
            / "report"
            / run_name
            / "hydrology.routing"
            / "discharge_daily.zarr"
        )

        # calculate the mean discharge over time, and plot spatially
        mean_discharge = GEB_discharge.mean(dim="time")
        mean_discharge.attrs["_FillValue"] = np.nan

        to_zarr(
            mean_discharge,
            self.output_folder_evaluate / "mean_discharge_m3_per_s.zarr",
            crs=4326,
        )

        fig, ax = plt.subplots(figsize=(10, 10))

        mean_discharge.plot(ax=ax, cmap="Blues", norm=mcolors.LogNorm(vmin=1))

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
        correct_Q_obs=False,
    ) -> None:
        """Method to evaluate the discharge grid from GEB against observations from the Q_obs database.

        Args:
            spinup_name: Name of the spinup run to include in the evaluation.
            run_name: Name of the run to evaluate.
            include_spinup: Whether to include the spinup run in the evaluation.
            correct_Q_obs: Whether to correct the Q_obs discharge timeseries for the difference in upstream area
                between the Q_obs station and the discharge from GEB.
        """
        # check if setup_discharge_observations method has been executed
        if not self.model.files["geoms"].get("discharge/discharge_snapped_locations"):
            print(
                "Discharge observations not set up, probably no stations present in the basin. Skipping discharge evaluation."
            )
            return

        #  create folders
        eval_plot_folder: Path = (
            Path(self.output_folder_evaluate) / "discharge" / "plots"
        )
        eval_result_folder = (
            Path(self.output_folder_evaluate) / "discharge" / "evaluation_results"
        )

        eval_plot_folder.mkdir(parents=True, exist_ok=True)
        eval_result_folder.mkdir(parents=True, exist_ok=True)

        GEB_discharge = open_zarr(
            self.model.output_folder
            / "report"
            / run_name
            / "hydrology.routing"
            / "discharge_daily.zarr"
        )

        # check if run file exists, if not, raise an error
        if not (self.model.output_folder / "report" / run_name).exists():
            raise FileNotFoundError(
                f"Run folder '{run_name}' does not exist in the report directory. Did you run the model?"
            )

        if include_spinup:
            # load the discharge spinup simulation
            GEB_discharge_spinup = open_zarr(
                self.model.output_folder
                / "report"
                / spinup_name
                / "hydrology.routing"
                / "discharge_daily.zarr"
            )

            GEB_discharge = xr.concat([GEB_discharge_spinup, GEB_discharge], dim="time")
        # load input data files
        snapped_locations = gpd.read_parquet(
            self.model.files["geoms"]["discharge/discharge_snapped_locations"]
        )  # load the snapped locations of the Q_obs stations
        Q_obs = pd.read_parquet(
            self.model.files["table"]["discharge/Q_obs"]
        )  # load the Q_obs discharge data

        region_shapefile = gpd.read_parquet(
            self.model.files["geoms"]["mask"]
        )  # load the region shapefile
        rivers = gpd.read_parquet(
            self.model.files["geoms"]["routing/rivers"]
        )  # load the rivers shapefiles

        evaluation_per_station: list = []

        # start validation loop over Q_obs stations
        for ID in tqdm(Q_obs.columns):
            # create a discharge timeseries dataframe
            discharge_Q_obs_df = Q_obs[ID]
            discharge_Q_obs_df.columns = ["Q"]
            discharge_Q_obs_df.name = "Q"
            # extract the properties from the snapping dataframe
            Q_obs_station_name = snapped_locations.loc[ID].Q_obs_station_name
            snapped_xy_coords = snapped_locations.loc[ID].closest_tuple
            Q_obs_station_coords = snapped_locations.loc[ID].Q_obs_station_coords
            Q_obs_to_GEB_upstream_area_ratio = snapped_locations.loc[
                ID
            ].Q_obs_to_GEB_upstream_area_ratio

            def create_validation_df():
                """Create a validation dataframe with the Q_obs discharge observations and the GEB discharge simulation for the selected station."""
                # select data closest to meerssen point
                GEB_discharge_station = GEB_discharge.isel(
                    x=snapped_xy_coords[0], y=snapped_xy_coords[1]
                )  # select the pixel in the grid that corresponds to the selected hydrography_xy value
                # rename xarray dataarray to Q
                GEB_discharge_station.name = "Q"
                discharge_sim_station_df = GEB_discharge_station.to_dataframe()
                discharge_sim_station_df = discharge_sim_station_df["Q"]
                discharge_sim_station_df.index.name = "time"  # rename index to time

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

            validation_df = create_validation_df()

            def calculate_validation_metrics():
                """Calculate the validation metrics for the current station."""
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

            def plot_validation_graphs(ID):
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
                )
                ax.plot(
                    validation_df.index,
                    validation_df["Q_obs"],
                    label="Q_obs observations",
                )
                ax.set_ylabel("Discharge [m3/s]")
                ax.set_xlabel("Time")
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
            index=False,
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
        def plot_validation_map():
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

        def create_folium_map(evaluation_gdf):
            """Create a Folium map with evaluation results and station markers."""
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

    def water_circle(
        self,
        run_name: str,
        include_spinup: bool,
        spinup_name: str,
        *args,
        **kwargs,
    ) -> None:
        """Create a water circle plot for the GEB model.

        Adapted from: https://github.com/mikhailsmilovic/flowplot
        Also see the paper: https://doi.org/10.1088/1748-9326/ad18de

        Args:
            run_name: Name of the run to evaluate.
            include_spinup: Whether to include the spinup run in the evaluation.
            spinup_name: Name of the spinup run to include in the evaluation.
            *args: ignored.
            **kwargs: ignored.
        """
        folder = self.model.output_folder / "report" / run_name

        storage = pd.read_csv(folder / "hydrology" / "storage.csv")
        storage_change = storage.iloc[-1]["storage"] - storage.iloc[0]["storage"]

        rain = pd.read_csv(
            folder / "hydrology.snowfrost" / "rain.csv",
            index_col=0,
            parse_dates=True,
        )["rain"].sum()
        snow = pd.read_csv(
            folder / "hydrology.snowfrost" / "snow.csv",
            index_col=0,
            parse_dates=True,
        )["snow"].sum()

        domestic_water_loss = pd.read_csv(
            folder / "hydrology.water_demand" / "domestic water loss.csv",
            index_col=0,
            parse_dates=True,
        )["domestic water loss"].sum()
        industry_water_loss = pd.read_csv(
            folder / "hydrology.water_demand" / "industry water loss.csv",
            index_col=0,
            parse_dates=True,
        )["industry water loss"].sum()
        livestock_water_loss = pd.read_csv(
            folder / "hydrology.water_demand" / "livestock water loss.csv",
            index_col=0,
            parse_dates=True,
        )["livestock water loss"].sum()

        river_outflow = pd.read_csv(
            folder / "hydrology.routing" / "river outflow.csv",
            index_col=0,
            parse_dates=True,
        )["river outflow"].sum()

        transpiration = pd.read_csv(
            folder / "hydrology.landcover" / "transpiration.csv",
            index_col=0,
            parse_dates=True,
        )["transpiration"].sum()
        bare_soil_evaporation = pd.read_csv(
            folder / "hydrology.landcover" / "bare soil evaporation.csv",
            index_col=0,
            parse_dates=True,
        )["bare soil evaporation"].sum()
        direct_evaporation = pd.read_csv(
            folder / "hydrology.landcover" / "direct evaporation.csv",
            index_col=0,
            parse_dates=True,
        )["direct evaporation"].sum()
        interception_evaporation = pd.read_csv(
            folder / "hydrology.landcover" / "interception evaporation.csv",
            index_col=0,
            parse_dates=True,
        )["interception evaporation"].sum()
        snow_sublimation = pd.read_csv(
            folder / "hydrology.landcover" / "snow sublimation.csv",
            index_col=0,
            parse_dates=True,
        )["snow sublimation"].sum()
        river_evaporation = pd.read_csv(
            folder / "hydrology.routing" / "river evaporation.csv",
            index_col=0,
            parse_dates=True,
        )["river evaporation"].sum()
        waterbody_evaporation = pd.read_csv(
            folder / "hydrology.routing" / "waterbody evaporation.csv",
            index_col=0,
            parse_dates=True,
        )["waterbody evaporation"].sum()

        hierarchy: dict[str, Any] = {
            "in": {
                "rain": rain,
                "snow": snow,
            },
            "out": {
                "evapotranspiration": {
                    "transpiration": transpiration,
                    "bare soil evaporation": bare_soil_evaporation,
                    "direct evaporation": direct_evaporation,
                    "interception evaporation": interception_evaporation,
                    "snow sublimation": snow_sublimation,
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
            "storage change": storage_change,
        }

        # the size of a section is the sum of the flows in that section
        # plus the size of the section itself. So if all of the section
        # is made up of its children, the size of the section is 0.
        water_circle_list: list[tuple[str, str, float | int]] = []
        color_map: dict[str, str] = {
            "in": "#636EFA",
            "out": "#EF5538",
            "balance": "#000000",
            "storage change": "#D2D2D3",
        }

        def add_flow(
            water_circle_list: list[tuple[str, str, float | int]],
            color_map: dict[str, str],
            root_section: str | None,
            parent: str | None,
            flow: str | None,
            value: int | float | dict[str, Any],
        ) -> tuple[list[tuple[str, str, float | int]], dict[str, str]]:
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
            if isinstance(value, (int, float)):  # stopping condition
                # adopt the color of the parent if it exists
                if parent is not None:
                    color_map[flow] = color_map[parent]
                else:  # if no parent, this is a root section
                    root_section = flow
                water_circle_list.append(
                    (root_section, parent, flow, value, color_map[flow])
                )
            elif isinstance(value, dict):
                if parent is not None:  # adopt the color of the parent
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
            columns=["root_section", "parent", "flow", "value", "color"],
        )

        root_section_totals = water_circle_df.groupby("root_section").sum("value")

        if (
            root_section_totals.loc["out", "value"]
            > root_section_totals.loc["in", "value"]
        ):
            category_order = ["storage change", "in", "out"]
        else:
            category_order = ["in", "out", "storage change"]

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

        water_circle.write_image(
            self.output_folder_evaluate / "water_circle.png", scale=5
        )
