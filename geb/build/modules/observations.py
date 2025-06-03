import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely.ops import nearest_points

"""
This module contains the Observations class. 
"""


class Observations:
    def __init__(self):
        pass

    def setup_discharge_observations(self, custom_river_stations=None):
        """
        setup_discharge_observations is responsible for setting up discharge observations from the Q_obs dataset.
        It clips Q_obs to the basin area, and snaps the Q_obs locations to the locations of the GEB discharge simulations, using upstream area estimates recorded in Q_obs.
        It also saves necessary input data for the model in the input folder, and some additional information in the output folder (e.g snapping plots).
        Additional stations can be added as csv files in the custom_stations folder in the GEB data catalog.
        """

        # load data
        upstream_area = self.grid["routing/upstream_area"]
        upstream_area_subgrid = self.other["drainage/original_d8_upstream_area"]
        rivers = self.geoms["routing/rivers"]
        region_shapefile = self.geoms["mask"]
        Q_obs = self.data_catalog.get_geodataset("GRDC")  # load the Q_obs dataset

        # create folders
        snapping_discharge_folder = (
            Path(self.root).parent / "output" / "build" / "snapping_discharge"
        )
        snapping_discharge_folder.mkdir(parents=True, exist_ok=True)

        # add external stations to Q_obs
        def add_station_Q_obs(station_name, station_coords, station_dataframe):
            """This function adds a new station to the Q_obs dataset (in this case GRDC). It should be a dataframe with the first row (lon, lat) and data should start at index 3 (row4)"""

            # Convert the pandas DataFrame to an xarray Dataset
            new_station_ds = xr.Dataset(
                {
                    "runoff_mean": (
                        ["time"],
                        station_dataframe["Q"].values,
                    ),  # Add the 'Q' column as runoff_mean
                    "station_name": ("id", [station_name]),  # Station name
                },
                coords={
                    "time": station_dataframe.index,  # Use the index as the time dimension
                    "id": [station_id],  # Assign the station ID
                    "x": ("id", [station_coords[0]]),
                    "y": ("id", [station_coords[1]]),
                },
            )
            # Add the new station to the Q_obs dataset
            Q_obs_merged = xr.concat([Q_obs, new_station_ds], dim="id")

            return Q_obs_merged

        def process_station_data(Q_station, dt_format, startrow):
            # process data
            station_coords = Q_station.iloc[
                0
            ].tolist()  # get the coordinates from the first row

            station_coords = [float(i) for i in station_coords]  # convert to float

            Q_station = Q_station.iloc[startrow:]  # remove the first rows
            Q_station.rename(
                columns={
                    Q_station.columns[0]: "date",
                    Q_station.columns[1]: "Q",
                },
                inplace=True,
            )
            Q_station["date"] = pd.to_datetime(Q_station["date"], format=dt_format)
            Q_station.set_index("date", inplace=True)
            Q_station["Q"] = Q_station["Q"].astype(float)  # convert to float
            Q_station = Q_station.resample("D", label="left").mean()
            Q_station.index.name = "time"  # rename index to time

            # delete missing values in the dataframe
            Q_station.dropna(inplace=True)  # drop missing time steps

            # checks
            if Q_station.shape[1] != 1:
                raise ValueError(f"File {station} does not have 1 column")

            if len(station_coords) != 2:
                raise ValueError(
                    f"File {station} does not have 2 coordinates. .csv files of discharge stations should have the coordinates in the first row of the file"
                )
            return Q_station, station_coords

        if custom_river_stations is not None:
            for station in os.listdir(custom_river_stations):
                if not station.endswith(".csv"):
                    # raise error
                    raise ValueError(f"File {station} is not a csv file")
                else:
                    station_name = station[:-4]

                    if not (
                        Path(self.root).parent / Path(custom_river_stations)
                    ).is_dir():
                        raise ValueError(
                            f"Path {Path(self.root).parent / Path(custom_river_stations)} does not exist or is not a directory. Create this directory if you want to use custom discharge stations, or set custom_river_stations to None"
                        )
                    Q_station = pd.read_csv(
                        Path(self.root).parent / Path(custom_river_stations) / station,
                        header=None,
                        delimiter=",",
                    )  # read the csv file with no header and comma delimiter

                    Q_station, station_coords = process_station_data(
                        Q_station, dt_format="%Y-%m-%d %H:%M:%S", startrow=3
                    )

                    # Check for missing or invalid dates
                    if Q_station.index.isnull().any():
                        raise ValueError(
                            "Datetime parsing failed. Found Nan values in the index."
                        )

                    # add station to Q_obs if station is not already in Q_obs
                    if station_name not in Q_obs.station_name.values:
                        station_id = int(Q_obs.id.max() + 1)  # ID for the new station
                        Q_obs_merged = add_station_Q_obs(
                            station_name, station_coords, Q_station
                        )  # name, coordinates, dataframe
                    else:
                        station_id = int(
                            Q_obs.id.values[Q_obs.station_name.values == station_name][
                                0
                            ]
                        )  # get the id of the station in the Q_obs dataset
                        Q_obs_merged = Q_obs.copy()
        else:
            Q_obs_merged = Q_obs.copy()

        # Clip the Q_obs dataset to the region shapefile
        def clip_Q_obs(Q_obs_merged, region_shapefile):
            """
            Clip Q_obs stations based on a region shapefile, to keep only Q_obs stations within the catchment boundaries
            """
            # Convert Q_obs points to GeoDataFrame
            Q_obs_gdf = gpd.GeoDataFrame(
                {
                    "id": Q_obs_merged.id.values,
                    "x": Q_obs_merged.x.values,
                    "y": Q_obs_merged.y.values,
                },
                geometry=gpd.points_from_xy(
                    Q_obs_merged.x.values, Q_obs_merged.y.values
                ),
                crs="EPSG:4326",
            )

            # Filter Q_obs stations that are in the region shapefile
            Q_obs_gdf = Q_obs_gdf[
                Q_obs_gdf.geometry.within(region_shapefile.geometry.unary_union)
            ]

            # select the Q_obs stations from the Q_obs dataset that are in the region shapefile
            Q_obs_merged = Q_obs_merged.sel(id=Q_obs_gdf.id.values)

            return Q_obs_merged

        Q_obs_clipped = clip_Q_obs(
            Q_obs_merged, region_shapefile
        )  # filter Q_obs stations based on the region shapefile

        # convert all the -999 values to NaN
        Q_obs_clipped = Q_obs_clipped.where(Q_obs_clipped != -999, np.nan)

        # save Q_obs clipped data as parquet file for later use
        discharge_df = Q_obs_clipped.runoff_mean.to_dataframe().reset_index()
        discharge_df.rename(
            columns={
                "time": "time",
                "id": "station_ID",
                "runoff_mean": "discharge",
            },
            inplace=True,
        )
        discharge_df = discharge_df.pivot(
            index="time", columns="station_ID", values="discharge"
        )
        discharge_df.dropna(how="all", inplace=True)  # remove rows that are all nan
        self.set_table(
            discharge_df, name="discharge/Q_obs"
        )  # save the discharge data as a table

        # Snapping to river and validation of discharges

        # create discharge snapping df
        discharge_snapping_df = pd.DataFrame()

        # start looping over the Q_obs stations
        for id in Q_obs_clipped.id.values:
            # create Q_obs variables
            Q_obs_station = Q_obs_clipped.sel(
                id=id
            )  # select the station from the Q_obs dataset
            Q_obs_station_name = str(
                Q_obs_station.station_name.values
            )  # get the name of the station
            Q_obs_station_coords = list(
                (
                    float(Q_obs_station.x.values),
                    float(Q_obs_station.y.values),
                )
            )  # get the coordinates of the station
            Q_obs_location = gpd.GeoDataFrame(
                geometry=[shapely.geometry.Point(Q_obs_station_coords)],
                crs=rivers.crs,
            )  # create a point geometry for the station
            Q_obs_uparea = (
                Q_obs_station.area.values.item()
            )  # get the upstream area of the station
            Q_obs_uparea_m2 = Q_obs_uparea * 1e6
            Q_obs_rivername = Q_obs_station.river_name.values.item()

            # find river section closest to the Q_obs station
            def get_distance_to_stations(rivers):
                """This function returns the distance of each river section to the station"""
                return rivers.distance(Q_obs_location).values.item()

            rivers["station_distance"] = rivers.geometry.apply(
                get_distance_to_stations
            )  # distance in degrees
            rivers_sorted = rivers.sort_values(by="station_distance")

            def select_river_segment(max_uparea_diff, max_spatial_diff):
                """
                This function selects the closest river segment to the Q_obs station based on the spatial distance.
                It returns an error if the spatial distance is larger than the max_spatial_diff. If the difference between the upstream area from MERIT (from the river centerlines)
                and the Q_obs upstream area is larger than the max_uparea_diff, it will select the closest river segment within the correct upstream area range.
                """
                if np.isnan(
                    Q_obs_uparea
                ):  # if Q_obs upstream area is NaN, only just select the closest river segment
                    closest_river_segment = rivers_sorted.head(1)
                else:
                    # add upstream area criteria
                    upstream_area_diff = (
                        max_uparea_diff * Q_obs_uparea
                    )  # 30% difference
                    closest_river_segment = rivers_sorted[
                        (rivers_sorted.uparea > (Q_obs_uparea - upstream_area_diff))
                        & (rivers_sorted.uparea < (Q_obs_uparea + upstream_area_diff))
                    ].head(1)

                    if (
                        closest_river_segment.station_distance.values.item()
                        > max_spatial_diff
                    ):
                        # raise error
                        raise ValueError(
                            f"Closest river segment is too far from the Q_obs (now: GRDC) station {Q_obs_station_name}. Distance: {closest_river_segment.station_distance.values.item()} degrees while the max distance set in the model is {max_spatial_diff} degrees."
                        )
                return closest_river_segment

            closest_river_segment = select_river_segment(
                max_uparea_diff=0.3,  # 30% max difference in upstream area
                max_spatial_diff=0.1,  # 0.1 degrees spatial difference
            )
            closest_river_segment_linestring = shapely.geometry.LineString(
                closest_river_segment.geometry.iloc[0]
            )
            closest_point_on_riverline = nearest_points(
                Q_obs_location, closest_river_segment_linestring
            )[1].geometry.iloc[0]  # find closest point to this nearest river segment

            # Read the upstream area from the subgrid at this point
            selected_subgrid_pixel = upstream_area_subgrid.sel(
                x=closest_point_on_riverline.x,
                y=closest_point_on_riverline.y,
                method="nearest",
            )
            GEB_upstream_area_from_subgrid = (
                selected_subgrid_pixel.values
            )  # get the value of the selected pixel

            # Find the closest pixel in the river network in the low-res grid
            selected_grid_pixel = upstream_area.sel(
                x=selected_subgrid_pixel.x,
                y=selected_subgrid_pixel.y,
                method="nearest",
            )

            # get the x and y of the selected_grid_pixel
            id_x = upstream_area.x.values.tolist().index(selected_grid_pixel.x.values)
            id_y = upstream_area.y.values.tolist().index(selected_grid_pixel.y.values)
            array = np.array(
                [id_x, id_y]
            )  # create an array with the x and y index of the selected_grid_pixel

            # select the closest pixel in the low-res river network
            xy_tuples = closest_river_segment[
                "hydrography_xy"
            ].values  # get the xy of the river pixels of the grid (already prepared and stored as hydrography_xy)
            xy_tuples = np.asarray(xy_tuples)[0]

            closest_tuple = min(
                xy_tuples, key=lambda x: np.linalg.norm(np.array(x) - array)
            )  # find which of the xy_tuples is closest to the xy of the selected grid pixel. This ensures that the selected pixel is in the river network of the course grid.

            # select the pixels in the different grids on the chosen location
            upstream_area_grid_pixel = upstream_area.isel(
                x=closest_tuple[0], y=closest_tuple[1]
            )  # select the pixel in the grid that corresponds to the selected hydrography_xy value

            # get the upstream area from the grid pixel
            GEB_upstream_area_from_grid = (
                upstream_area_grid_pixel.values
            )  # get the value of the selected pixel

            # make variables for all the different coordinates
            closest_point_coords = list(
                (
                    float(closest_point_on_riverline.x),
                    float(closest_point_on_riverline.y),
                )
            )  # closest point coordinates
            subgrid_pixel_coords = list(
                (
                    float(selected_subgrid_pixel.x.values),
                    float(selected_subgrid_pixel.y.values),
                )
            )  ## subgrid pixel coordinates
            grid_pixel_coords = list(
                (
                    float(upstream_area_grid_pixel.x.values),
                    float(upstream_area_grid_pixel.y.values),
                )
            )  # grid pixel coordinates

            def add_row(discharge_snapping_df):
                new_row = pd.DataFrame(
                    [
                        {
                            "Q_obs_station_name": Q_obs_station_name,
                            "Q_obs_station_ID": int(id),
                            "Q_obs_river_name": Q_obs_rivername,
                            "Q_obs_upstream_area_m2": Q_obs_uparea_m2,
                            "Q_obs_station_coords": Q_obs_station_coords,
                            "closest_point_coords": closest_point_coords,
                            "subgrid_pixel_coords": subgrid_pixel_coords,
                            "grid_pixel_coords": grid_pixel_coords,
                            "closest_tuple": closest_tuple,
                            "GEB_upstream_area_from_subgrid": float(
                                GEB_upstream_area_from_subgrid
                            ),
                            "GEB_upstream_area_from_grid": float(
                                GEB_upstream_area_from_grid
                            ),
                            "Q_obs_to_GEB_upstream_area_ratio": float(
                                GEB_upstream_area_from_subgrid / Q_obs_uparea_m2
                            ),
                            "snapping_distance_degrees": float(
                                closest_river_segment.station_distance.values.item()
                            ),
                        }
                    ]
                )

                discharge_snapping_df = pd.concat(
                    [discharge_snapping_df, new_row], ignore_index=True
                )  # add the new row to the dataframe
                return discharge_snapping_df

            discharge_snapping_df = add_row(
                discharge_snapping_df
            )  # add the row to the dataframe

            # plot locations with river line and the subgrid
            def plot_snapping():
                fig, ax = plt.subplots(
                    subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(15, 10)
                )
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS)
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.OCEAN)
                ax.add_feature(cfeature.LAKES)

                # Set the extent to zoom in around the gauge location
                buffer = 0.05  # Adjust this value to control the zoom level
                ax.set_extent(
                    [
                        Q_obs_station_coords[0] - buffer,
                        Q_obs_station_coords[0] + buffer,
                        Q_obs_station_coords[1] - buffer,
                        Q_obs_station_coords[1] + buffer,
                    ],
                    crs=ccrs.PlateCarree(),
                )

                ax.scatter(
                    Q_obs_station_coords[0],
                    Q_obs_station_coords[1],
                    color="red",
                    marker="o",
                    s=30,
                    label="Original gauge",
                    zorder=3,
                )
                ax.scatter(
                    closest_point_coords[0],
                    closest_point_coords[1],
                    color="black",
                    marker="o",
                    s=30,
                    label="Closest point to river",
                    zorder=3,
                )
                ax.scatter(
                    grid_pixel_coords[0],
                    grid_pixel_coords[1],
                    color="blue",
                    marker="o",
                    s=30,
                    label="Grid pixel",
                    zorder=3,
                )

                upstream_area.plot(
                    ax=ax,
                    cmap="viridis",
                    cbar_kwargs={"label": "Upstream area [m2]"},
                    zorder=0,
                    alpha=1,
                )
                rivers.plot(ax=ax, color="blue", linewidth=1)
                closest_river_segment.plot(
                    ax=ax, color="green", linewidth=3, label="Closest river segment"
                )

                ax.set_title(
                    "Upstream area grid and gauge snapping for %s" % Q_obs_station_name
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.legend()
                plt.savefig(
                    self.report_dir
                    / "snapping_discharge"
                    / f"snapping_discharge_{Q_obs_station_name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                # plt.show()
                plt.close()

            plot_snapping()  # plot the snapping
            print("discharge snapping done for station %s" % Q_obs_station_name)

        print("Discharge snapping done for all stations")

        # save to excel and parquet files
        discharge_snapping_df.to_excel(
            self.report_dir / "snapping_discharge" / "discharge_snapping.xlsx",
            index=False,
        )  # save the dataframe to an excel file

        discharge_snapping_gdf = gpd.GeoDataFrame(
            discharge_snapping_df,
            geometry=gpd.points_from_xy(
                discharge_snapping_df["grid_pixel_coords"].apply(lambda x: x[0]),
                discharge_snapping_df["grid_pixel_coords"].apply(lambda x: x[1]),
            ),
            crs="EPSG:4326",  # Set the coordinate reference system
        )
        discharge_snapping_gdf.set_index("Q_obs_station_ID", inplace=True)

        self.set_geoms(
            discharge_snapping_gdf, name="discharge/discharge_snapped_locations"
        )

        print("Building discharge datasets done")
