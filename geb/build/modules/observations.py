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

    def setup_discharge_observations(self, custom_river_stations=False):
        """
        setup_discharge_observations is responsible for setting up discharge observations from the GRDC dataset.
        It clips GRDC to the basin area, and snaps the GRDC locations to the locations of the GEB discharge simulations, using upstream area estimates recorded in GRDC.
        It also saves necessary input data for the model in the input folder, and some additional information in the output folder (e.g snapping plots).
        Additional stations can be added as csv files in the custom_stations folder in the GEB data catalog.
        """

        # load data
        upstream_area = self.grid["routing/upstream_area"]
        upstream_area_subgrid = self.other["drainage/original_d8_upstream_area"]
        rivers = self.geoms["routing/rivers"]
        region_shapefile = self.geoms["mask"]
        GRDC = self.data_catalog.get_geodataset("GRDC")

        # create folders
        snapping_discharge_folder = (
            Path(self.root).parent / "output" / "build" / "snapping_discharge"
        )
        snapping_discharge_folder.mkdir(parents=True, exist_ok=True)

        ############################### Load and process GRDC dataset ##########################################

        # add external stations to GRDC
        def add_station_GRDC(station_name, station_coords, station_dataframe):
            """This function adds a new station to the GRDC dataset. It should be a dataframe with the first row lon, lat and data should start at index 3 (row4)"""

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
            # Add the new station to the GRDC dataset
            GRDC_merged = xr.concat([GRDC, new_station_ds], dim="id")

            return GRDC_merged

        def process_station_data(station_df, dt_format, startrow):
            # process data
            station_coords = station_df.iloc[
                0
            ].tolist()  # get the coordinates from the first row

            station_coords = [float(i) for i in station_coords]  # convert to float

            station_df = station_df.iloc[startrow:]  # remove the first rows
            station_df.rename(
                columns={
                    Q_station.columns[0]: "date",
                    Q_station.columns[1]: "Q",
                },
                inplace=True,
            )
            station_df["date"] = pd.to_datetime(station_df["date"], format=dt_format)
            station_df.set_index("date", inplace=True)
            station_df["Q"] = station_df["Q"].astype(float)  # convert to float
            station_df = station_df.resample("D", label="left").mean()
            station_df.index.name = "time"  # rename index to time

            # delete missing values in the dataframe
            station_df.dropna(inplace=True)  # drop missing time steps

            # checks
            if station_df.shape[1] != 1:
                raise ValueError(f"File {station} does not have 1 column")

            if len(station_coords) != 2:
                raise ValueError(
                    f"File {station} does not have 2 coordinates. .csv files of discharge stations should have the coordinates in the first row of the file"
                )
            return station_df, station_coords

        if custom_river_stations is True:
            for station in os.listdir(
                Path(self.data_catalog["GRDC"].path).parent.parent / "custom_stations"
            ):
                if not station.endswith(".csv"):
                    # raise error
                    raise ValueError(f"File {station} is not a csv file")
                else:
                    station_name = station[:-4]

                    Q_station = self.data_catalog.get_dataframe(
                        f"custom_discharge_stations_{station}"
                    )

                    Q_station, station_coords = process_station_data(
                        Q_station, dt_format="%Y-%m-%d %H:%M:%S", startrow=3
                    )

                    # add station to grdc if station is not already in grdc
                    if station_name not in GRDC.station_name.values:
                        station_id = int(GRDC.id.max() + 1)  # ID for the new station
                        GRDC_merged = add_station_GRDC(
                            station_name, station_coords, Q_station
                        )  # name, coordinates, dataframe
                    else:
                        station_id = int(
                            GRDC.id.values[GRDC.station_name.values == station_name][0]
                        )  # get the id of the station in the GRDC dataset
                        GRDC_merged = GRDC.copy()
        else:
            GRDC_merged = GRDC.copy()

        # Clip the GRDC dataset to the region shapefile
        def clip_GRDC(GRDC_merged, region_shapefile):
            """
            Clip GRDC stations based on a region shapefile, to keep only GRDC stations within the catchment boundaries
            """
            # Convert GRDC points to GeoDataFrame
            GRDC_gdf = gpd.GeoDataFrame(
                {
                    "id": GRDC_merged.id.values,
                    "x": GRDC_merged.x.values,
                    "y": GRDC_merged.y.values,
                },
                geometry=gpd.points_from_xy(GRDC_merged.x.values, GRDC_merged.y.values),
                crs="EPSG:4326",
            )

            # Filter GRDC stations that are in the region shapefile
            GRDC_gdf = GRDC_gdf[
                GRDC_gdf.geometry.within(region_shapefile.geometry.unary_union)
            ]

            # select the GRDC stations from the GRDC dataset that are in the region shapefile
            GRDC_merged = GRDC_merged.sel(id=GRDC_gdf.id.values)

            return GRDC_merged

        GRDC_clipped = clip_GRDC(
            GRDC_merged, region_shapefile
        )  # filter GRDC stations based on the region shapefile

        # convert all the -999 values to NaN
        GRDC_clipped = GRDC_clipped.where(GRDC_clipped != -999, np.nan)

        # save GRDC clipped data as parquet file for later use
        discharge_df = GRDC_clipped.runoff_mean.to_dataframe().reset_index()
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
            discharge_df, name="discharge/GRDC"
        )  # save the discharge data as a table

        ############################### Snapping to river and validation of discharges ############################################################################

        # create discharge snapping df
        discharge_snapping_df = pd.DataFrame()

        # start looping over the GRDC stations
        for id in GRDC_clipped.id.values:
            # create GRDC variables
            GRDC_station = GRDC_clipped.sel(
                id=id
            )  # select the station from the GRDC dataset
            GRDC_station_name = str(
                GRDC_station.station_name.values
            )  # get the name of the station
            GRDC_station_coords = list(
                (
                    float(GRDC_station.x.values),
                    float(GRDC_station.y.values),
                )
            )  # get the coordinates of the station
            GRDC_location = gpd.GeoDataFrame(
                geometry=[shapely.geometry.Point(GRDC_station_coords)],
                crs=rivers.crs,
            )  # create a point geometry for the station
            GRDC_uparea = (
                GRDC_station.area.values.item()
            )  # get the upstream area of the station
            GRDC_uparea_m2 = GRDC_uparea * 1e6
            GRDC_rivername = GRDC_station.river_name.values.item()

            # find river section closest to the GRDC station
            def get_distance_to_stations(rivers):
                """This function returns the distance of each river section to the station"""
                return rivers.distance(GRDC_location).values.item()

            rivers["station_distance"] = rivers.geometry.apply(
                get_distance_to_stations
            )  # distance in degrees
            rivers_sorted = rivers.sort_values(by="station_distance")

            # select the closest river segment based on the upstream area and distance
            def select_river_segment(max_uparea_diff, max_spatial_diff):
                if np.isnan(
                    GRDC_uparea
                ):  # if GRDC upstream area is NaN, only just select the closest river segment
                    closest_river_segment = rivers_sorted.head(1)
                else:
                    # add upstream area criteria
                    upstream_area_diff = max_uparea_diff * GRDC_uparea  # 30% difference
                    closest_river_segment = rivers_sorted[
                        (rivers_sorted.uparea > (GRDC_uparea - upstream_area_diff))
                        & (rivers_sorted.uparea < (GRDC_uparea + upstream_area_diff))
                    ].head(1)

                    if (
                        closest_river_segment.station_distance.values.item()
                        > max_spatial_diff
                    ):
                        # raise error
                        raise ValueError(
                            f"Closest river segment is too far from the GRDC station {GRDC_station_name}. Distance: {closest_river_segment.station_distance.values.item()} degrees while the max distance set in the model is {max_spatial_diff} degrees."
                        )
                    print(
                        "distance to closest river segment: %s degrees"
                        % closest_river_segment.station_distance.values.item()
                    )
                return closest_river_segment

            closest_river_segment = select_river_segment(
                max_uparea_diff=0.3,  # 30% max difference
                max_spatial_diff=0.1,  # 0.1 degrees spatial difference
            )
            closest_river_segment_linestring = shapely.geometry.LineString(
                closest_river_segment.geometry.iloc[0]
            )
            closest_point_on_riverline = nearest_points(
                GRDC_location, closest_river_segment_linestring
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
                            "GRDC_station_name": GRDC_station_name,
                            "GRDC_station_ID": int(id),
                            "GRDC_river_name": GRDC_rivername,
                            "GRDC_upstream_area_m2": GRDC_uparea_m2,
                            "GRDC_station_coords": GRDC_station_coords,
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
                            "GRDC_to_GEB_upstream_area_ratio": float(
                                GEB_upstream_area_from_subgrid / GRDC_uparea_m2
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
            print("start plotting")

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
                        GRDC_station_coords[0] - buffer,
                        GRDC_station_coords[0] + buffer,
                        GRDC_station_coords[1] - buffer,
                        GRDC_station_coords[1] + buffer,
                    ],
                    crs=ccrs.PlateCarree(),
                )

                ax.scatter(
                    GRDC_station_coords[0],
                    GRDC_station_coords[1],
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
                    "Upstream area grid and gauge snapping for %s" % GRDC_station_name
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.legend()
                plt.savefig(
                    self.report_dir
                    / "snapping_discharge"
                    / f"snapping_discharge_{GRDC_station_name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                # plt.show()
                plt.close()

            plot_snapping()  # plot the snapping
            print("discharge snapping done for station %s" % GRDC_station_name)

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
        discharge_snapping_gdf.set_index("GRDC_station_ID", inplace=True)

        self.set_geoms(
            discharge_snapping_gdf, name="discharge/discharge_snapped_locations"
        )

        print("Building discharge datasets done")
