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

# EVAL
from shapely.ops import nearest_points


class Observations:
    def __init__(self):
        pass

    def setup_discharge_observations(self, custom_river_stations=False):
        # transform = self.grid.rio.transform(recalc=True)
        # upstream_area_grid=self.grid["routing/upstream_area"]

        upstream_area = self.grid["routing/upstream_area"]
        upstream_area_subgrid = self.other["original_d8_upstream_area"]
        rivers = self.geoms["routing/rivers"]
        region_shapefile = self.geoms["mask"]

        GRDC = self.data_catalog.get_geodataset("GRDC")

        # create discharge snapping df
        discharge_snapping_df = pd.DataFrame(
            columns=[
                "station_name",
                "station_coords",
                "closest_point_coords",
                "subgrid_pixel_coords",
                "grid_pixel_coords",
                "upstream_area_from_subgrid",
                "upstream_area_from_grid",
                "upstream_area_ratio",
            ]
        )

        # keep only GRDC stations within the catchment boundaries

        ############################################ GRDC functions ###########################################

        # keep only GRDC stations within the catchment boundaries
        def clip_GRDC(GRDC, region_shapefile):
            """
            Clip GRDC stations based on the region shapefile.
            """
            # Convert GRDC points to GeoDataFrame
            GRDC_gdf = gpd.GeoDataFrame(
                {
                    "id": GRDC.id.values,
                    "x": GRDC.x.values,
                    "y": GRDC.y.values,
                },
                geometry=gpd.points_from_xy(GRDC.x.values, GRDC.y.values),
                crs="EPSG:4326",
            )

            # Filter GRDC stations that are in the region shapefile
            GRDC_gdf = GRDC_gdf[
                GRDC_gdf.geometry.within(region_shapefile.geometry.unary_union)
            ]

            # select the GRDC stations from the GRDC dataset that are in the region shapefile
            GRDC = GRDC.sel(id=GRDC_gdf.id.values)

            return GRDC

        def add_station_GRDC(station_name, station_coords, station_dataframe):
            """This function adds a new station to the GRDC dataset. It should be a dataframe"""

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
                    "x": ("id", [station_coords[0]]),  # Longitude
                    "y": ("id", [station_coords[1]]),  # Latitude
                },
            )
            # Add the new station to the GRDC dataset
            GRDC_merged = xr.concat([GRDC, new_station_ds], dim="id")

            return GRDC_merged

        ############## add "external" stations to GRDC ###############
        if custom_river_stations is True:
            for station in os.listdir(
                # data_dir = self.preprocessing_dir / "crops" / "MIRCA2000"
                "input\evaluation\discharge\seperate_stations"
            ):
                if not station.endswith(".csv"):
                    # raise error
                    raise ValueError(f"File {station} is not a csv file")
                else:
                    # load station data
                    Q_station = self.data_catalog.get_dataframe(
                        f"custom_discharge_stations_{station}"
                    )  # self.data_catalog.get_rasterdataset(f"global_irrigation_area_{fraction_gw_irrigation}",

                    # process data
                    station_coords = Q_station.iloc[
                        0
                    ].tolist()  # get the coordinates from the first row

                    station_coords = [
                        float(i) for i in station_coords
                    ]  # convert to float

                    Q_station = Q_station.iloc[3:]  # remove the first rows
                    Q_station.rename(
                        columns={
                            Q_station.columns[0]: "date",
                            Q_station.columns[1]: "Q",
                        },
                        inplace=True,
                    )
                    Q_station["date"] = pd.to_datetime(
                        Q_station["date"], format="%Y-%m-%d %H:%M:%S"
                    )
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

                    station_name = station[:-4]

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

        ############## Clip GRDC ###############
        GRDC_clipped = clip_GRDC(
            GRDC_merged, region_shapefile
        )  # filter GRDC stations based on the region shapefile

        print("GRDC dataset processed and clipped")

        ###########################################################################################################################################################
        ############################### Snapping to river and validation of discharges ############################################################################
        ###########################################################################################################################################################

        for id in GRDC_clipped.id.values:
            GRDC_station = GRDC_clipped.sel(
                id=id
            )  # select the station from the GRDC dataset
            GRDC_station_name = str(
                GRDC_station.station_name.values
            )  # get the name of the station
            GRDC_station_coords = (
                float(GRDC_station.x.values),
                float(GRDC_station.y.values),
            )  # get the coordinates of the station
            location_discharge_station = gpd.GeoDataFrame(
                geometry=[shapely.geometry.Point(GRDC_station_coords)], crs=rivers.crs
            )  # create a point geometry for the station

            ############## find point in river closest to the discharge station #############
            def get_distance_to_stations(rivers):
                """This function returns the distance of each river section to the station"""
                return rivers.distance(location_discharge_station).values.item()

            rivers["station_distance"] = rivers.geometry.apply(get_distance_to_stations)
            closest_river_segment = rivers.sort_values(by="station_distance").head(
                1
            )  # select closest river segment
            closest_river_segment_linestring = shapely.geometry.LineString(
                closest_river_segment.geometry.iloc[0]
            )

            closest_point = nearest_points(
                location_discharge_station, closest_river_segment_linestring
            )[1].geometry.iloc[0]  # find closest point to this nearest river segment

            ############### Read the upstream area from the subgrid at this point #############
            selected_subgrid_pixel = upstream_area_subgrid.sel(
                x=closest_point.x, y=closest_point.y, method="nearest"
            )  # select subgrid value at the closest point
            upstream_area_from_subgrid = (
                selected_subgrid_pixel.values
            )  # get the value of the selected pixel

            ############### Find the closest pixel in the river network in the low-res grid #############
            selected_grid_pixel = upstream_area.sel(
                x=selected_subgrid_pixel.x, y=selected_subgrid_pixel.y, method="nearest"
            )  # select the pixel in the grid that corresponds to the selected hydrography_xy value

            id_x = upstream_area.x.values.tolist().index(
                selected_grid_pixel.x.values
            )  # find the index of the selected_grid_pixel x in the upstream_area_grid
            id_y = upstream_area.y.values.tolist().index(
                selected_grid_pixel.y.values
            )  # find the index of the selected_grid_pixel y in the upstream_area_grid

            array = np.array(
                [id_x, id_y]
            )  # create an array with the x and y index of the selected_grid_pixel
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

            ############### Read the upstream area from the low-res grid and calculate upstream area ratio  #############
            upstream_area_from_grid = (
                upstream_area_grid_pixel.values
            )  # get the value of the selected pixel
            ratio_upstream_area = (
                upstream_area_from_subgrid / upstream_area_from_grid
            )  # ratio between the two grids

            ################ store the station snapping coordinates and plot them #############
            station_coords = list(
                (
                    float(GRDC_clipped.sel(id=station_id).x.values),
                    float(GRDC_clipped.sel(id=station_id).y.values),
                )
            )  # original station coordinates
            closest_point_coords = list(
                (float(closest_point.x), float(closest_point.y))
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

            ################################ add the coordinates to the dataframe ###########################
            # Create a new row as a DataFrame
            new_row = pd.DataFrame(
                [
                    {
                        "station_name": GRDC_station_name,
                        "station_coords": GRDC_station_coords,
                        "closest_point_coords": closest_point_coords,
                        "subgrid_pixel_coords": subgrid_pixel_coords,
                        "grid_pixel_coords": grid_pixel_coords,
                        "closest_tuple": closest_tuple,
                        "upstream_area_from_subgrid": upstream_area_from_subgrid,
                        "upstream_area_from_grid": upstream_area_from_grid,
                        "upstream_area_ratio": ratio_upstream_area,
                    }
                ]
            )

            discharge_snapping_df = pd.concat(
                [discharge_snapping_df, new_row], ignore_index=True
            )  # add the new row to the dataframe
            discharge_snapping_df.to_excel(
                Path(self.root).parent
                / "output"
                / "build"
                / "snapping_discharge"
                / "discharge_snapping.xlsx",
                index=False,
            )  # save the dataframe to an excel file

            # plot locations with river line and the subgrid
            fig, ax = plt.subplots(
                subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(15, 10)
            )
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.LAKES)

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
                Path(self.root).parent
                / "output"
                / "build"
                / "snapping_discharge"
                / f"snapping_discharge_{GRDC_station_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            # plt.show()
            plt.close()

            print("discharge snapping done for station %s" % GRDC_station_name)

        discharge_snapping_df = pd.DataFrame(
            columns=[
                "station_name",
                "station_coords",
                "closest_point_coords",
                "subgrid_pixel_coords",
                "grid_pixel_coords",
                "upstream_area_from_subgrid",
                "upstream_area_from_grid",
                "upstream_area_ratio",
            ]
        )

        print("discharge preprocessing done")

        # GRDC=    # later use: GRDC = data_catalog.get_source("GRDC").path

        # for i, file in enumerate(files):
        #     filename = file["filename"]
        #     longitude, latitude = file["longitude"], file["latitude"]
        #     data = pd.read_csv(filename, index_col=0, parse_dates=True)

        #     # assert data has one column
        #     assert data.shape[1] == 1

        #     px, py = ~transform * (longitude, latitude)
        #     px = math.floor(px)
        #     py = math.floor(py)

        #     discharge_data.append(
        #         xr.DataArray(
        #             np.expand_dims(data.iloc[:, 0].values, 0),
        #             dims=["pixel", "time"],
        #             coords={
        #                 "time": data.index.values,
        #                 "pixel": [i],
        #                 "px": ("pixel", [px]),
        #                 "py": ("pixel", [py]),
        #             },
        #         )
        #     )
        # discharge_data = xr.concat(discharge_data, dim="pixel")
        # self.set_other(
        #     discharge_data,
        #     name="observations/discharge",
        #     time_chunksize=1e99,  # no chunking
        # )

        ##
