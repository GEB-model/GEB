"""This module contains the classes and functions processing observational data during model building."""

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
from tqdm import tqdm

from geb.build.methods import build_method
from geb.workflows.io import get_window

"""
This module contains the Observations class. 
"""


def plot_snapping(
    station_id: int | str,
    output_folder: Path,
    rivers: gpd.GeoDataFrame,
    upstream_area: xr.DataArray,
    Q_obs_station_coords: tuple[float, float],
    closest_point_coords: tuple[float, float],
    closest_river_segment: gpd.GeoDataFrame,
    grid_pixel_coords: tuple[float, float],
) -> None:
    """Create and save a map visualizing gauge snapping results.

    This function produces a map centered on the observed gauge location that shows:
    - the original gauge location,
    - the closest point on the river centerline,
    - the selected grid pixel,
    - the upstream area raster (within the map extent), and
    - the closest river segment.

    The resulting figure is saved as a PNG file named "snapping_discharge_{station_id}.png"
    inside the provided output_folder.

    Args:
        station_id: Identifier or name of the station used in the plot title and output filename.
        output_folder: Path to the directory where the PNG file will be saved.
        rivers: GeoDataFrame containing river centerlines used for plotting.
        upstream_area: xarray DataArray with upstream area values used for background plotting.
        Q_obs_station_coords: Tuple (lon, lat) of the original observed station coordinates.
        closest_point_coords: Tuple (lon, lat) of the closest point on the river centerline to the station.
        closest_river_segment: GeoDataFrame containing the selected river segment to highlight.
        grid_pixel_coords: Tuple (lon, lat) of the snapped grid pixel coordinates.

    """
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

    xmin = Q_obs_station_coords[0] - buffer
    xmax = Q_obs_station_coords[0] + buffer
    ymin = Q_obs_station_coords[1] - buffer
    ymax = Q_obs_station_coords[1] + buffer

    ax.set_extent(
        [
            xmin,
            xmax,
            ymin,
            ymax,
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

    # Select the upstream area within the extent of the plot
    upstream_area_within_extent = upstream_area.isel(
        get_window(
            upstream_area.x,
            upstream_area.y,
            bounds=(xmin, ymin, xmax, ymax),
            buffer=1,
            raise_on_buffer_out_of_bounds=False,
            raise_on_out_of_bounds=False,
        )
    )

    upstream_area_within_extent.plot(
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

    ax.set_title("Upstream area grid and gauge snapping for %s" % station_id)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.savefig(
        output_folder / f"snapping_discharge_{station_id}.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()
    plt.close()


# add external stations to Q_obs
def add_station_Q_obs(
    station_id: int,
    station_name: str,
    Q_obs: xr.Dataset,
    station_coords: tuple[float, float],
    station_dataframe: pd.DataFrame,
) -> xr.Dataset:
    """This function adds a new station to the Q_obs dataset (in this case GRDC).

    The new station .csv file should be loaded as a dataframe with the first row being (lon, lat), and discharge data should start at index 3 (row4).

    Returns:
        The updated dataset with discharge observations with the new station added.
    """
    # Convert the pandas DataFrame to an xarray Dataset
    new_station_ds = xr.Dataset(
        {
            "runoff_mean": (
                ["time"],
                station_dataframe["Q"].values,
            ),  # Add the 'Q' column as runoff_mean
            "station_name": ("id", [station_name]),  # Station name
            "x": ("id", [station_coords[0]]),  # Longitude
            "y": ("id", [station_coords[1]]),  # Latitude
        },
        coords={
            "time": station_dataframe.index,  # Use the index as the time dimension
            "id": [station_id],  # Assign the station ID
        },
    )
    # Add the new station to the Q_obs dataset
    Q_obs_merged = xr.concat([Q_obs, new_station_ds], dim="id")

    return Q_obs_merged


def process_station_data(
    station: str, Q_station: pd.DataFrame, dt_format: str, startrow: int
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Parse and preprocess a station CSV read into a DataFrame.

    Args:
        station: Filename or identifier of the station (used in error messages).
        Q_station: Raw station DataFrame where the first row contains coordinates and data starts at row index `startrow`.
        dt_format: Datetime format string for parsing the date column.
        startrow: Row index where time series data begins (0-based).

    Returns:
        A tuple with the cleaned station DataFrame indexed by time and a tuple with station (lon, lat) as floats.

    Raises:
        ValueError: If the processed station DataFrame does not contain exactly one data column (expected 'Q'),
                    or if the first row does not contain exactly two coordinates (longitude and latitude) that can be parsed as floats.
    """
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
    return Q_station, (station_coords[0], station_coords[1])


def clip_Q_obs(Q_obs_merged: xr.Dataset, region_mask: gpd.GeoDataFrame) -> xr.Dataset:
    """
    Clip Q_obs stations based on a region shapefile, to keep only Q_obs stations within the catchment boundaries.

    Args:
        Q_obs_merged:
            Dataset with discharge observations.
        region_mask:
            Geometry of the region to clip the Q_obs stations to.

    Returns:
        The clipped discharge observations dataset with only stations within the region shapefile.
    """
    # Convert Q_obs points to GeoDataFrame
    Q_obs_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "id": Q_obs_merged.id.values,
            "x": Q_obs_merged.x.values,
            "y": Q_obs_merged.y.values,
        },
        geometry=gpd.points_from_xy(Q_obs_merged.x.values, Q_obs_merged.y.values),
        crs="EPSG:4326",
    )

    # Filter Q_obs stations that are in the region shapefile
    Q_obs_gdf: gpd.GeoDataFrame = Q_obs_gdf[
        Q_obs_gdf.geometry.within(region_mask.geometry.unary_union)
    ]

    # select the Q_obs stations from the Q_obs dataset that are in the region shapefile
    Q_obs_merged = Q_obs_merged.sel(id=Q_obs_gdf.id.values)

    return Q_obs_merged


def get_distance_to_stations(
    rivers: shapely.geometry.base.BaseGeometry, Q_obs_location: gpd.GeoDataFrame
) -> float:
    """This function returns the distance of each river section to the station.

    Args:
        rivers: A row of the rivers GeoDataFrame (a shapely geometry object).
        Q_obs_location: A GeoDataFrame containing the station location as a Point geometry.

    Returns:
        Distance in degrees between the river section and the station.
    """
    return float(rivers.distance(Q_obs_location).values.item())


def select_river_segment(
    max_uparea_difference_ratio: float,
    max_spatial_difference_degrees: float,
    Q_obs_uparea_m2: float,
    rivers_sorted: pd.DataFrame,
) -> pd.DataFrame | bool:
    """This function selects the closest river segment to the Q_obs station based on the spatial distance.

    It returns false if the spatial distance is larger than the max_spatial_difference_degrees.
    If the difference between the upstream area from MERIT (from the river centerlines)
    and the Q_obs upstream area is larger than the max_uparea_difference_ratio,
    it will select the closest river segment within the correct upstream area range.

    Args:
        max_uparea_difference_ratio: The maximum allowed difference in upstream area between the Q_obs station and the GEB river segment, as a ratio of the Q_obs upstream area.
        max_spatial_difference_degrees: The maximum allowed spatial difference in degrees between the Q_obs station and the GEB river segment.
        Q_obs_uparea_m2 : Upstream area reported for the observational station (in m2). Used to compare against river segment upstream areas to enforce the upstream area ratio criterion.
        rivers_sorted: GeoDataFrame of river segments sorted by spatial distance to the station; must contain the 'uparea_m2' attribute and will be filtered/queried to find the matching segment.

    Returns:
        The closest river segment to the Q_obs station that meets the criteria or False if no segment is found.
    """
    if np.isnan(
        Q_obs_uparea_m2
    ):  # if Q_obs upstream area is NaN, only just select the closest river segment
        closest_river_segment = rivers_sorted.head(1)
    else:
        # add upstream area criteria
        upstream_area_diff = (
            max_uparea_difference_ratio * Q_obs_uparea_m2
        )  # 30% difference
        closest_river_segment = rivers_sorted[
            (rivers_sorted["uparea_m2"] > (Q_obs_uparea_m2 - upstream_area_diff))
            & (rivers_sorted["uparea_m2"] < (Q_obs_uparea_m2 + upstream_area_diff))
        ].head(1)

        if closest_river_segment.empty:
            return False  # no river segment found within the upstream area criteria

        if (
            closest_river_segment.iloc[0].station_distance
            > max_spatial_difference_degrees
        ):
            # No river segment found within the max_spatial_difference_degrees, returning with False
            return False

    return closest_river_segment


class Observations:
    """Collects, parses and processes observational data for model evaluation."""

    def __init__(self) -> None:
        """Initialize the Observations class."""
        pass

    @build_method(depends_on=["setup_hydrography"])
    def setup_discharge_observations(
        self,
        max_uparea_difference_ratio: float = 0.3,
        max_spatial_difference_degrees: float = 0.1,
        custom_river_stations: str | None = None,
    ) -> None:
        """setup_discharge_observations is responsible for setting up discharge observations from the Q_obs dataset.

        It clips Q_obs to the basin area, and snaps the Q_obs locations to the locations of the GEB discharge simulations, using upstream area estimates recorded in Q_obs.
        It also saves necessary input data for the model in the input folder, and some additional information in the output folder (e.g snapping plots).
        Additional stations can be added as csv files in the custom_stations folder in the GEB data catalog.

        Args:
            max_uparea_difference_ratio: The maximum allowed difference in upstream area between the Q_obs station and the GEB river segment, as a ratio of the Q_obs upstream area. Default is 0.3 (30%).
            max_spatial_difference_degrees: The maximum allowed spatial difference in degrees between the Q_obs station and the GEB river segment. Default is 0.1 degrees.
            custom_river_stations: Path to a folder containing custom river stations as csv files. Each csv file should have the first row containing the coordinates (longitude, latitude) and the data starting from the fourth row. Default is None, which means no custom stations are used.

        Raises:
            ValueError: If no discharge stations are found in the region shapefile.
            ValueError: If a custom station file does not have the correct format (2 coordinates in the first row, data starting from the fourth row).
        """
        # load data
        upstream_area = self.grid[
            "routing/upstream_area"
        ].compute()  # we need to use this one many times, so we compute it once
        upstream_area_subgrid = self.other["drainage/original_d8_upstream_area"]
        rivers = self.geom["routing/rivers"]
        region_mask = self.geom["mask"]

        # Load Q_obs dataset
        Q_obs = self.new_data_catalog.fetch("GRDC").read()

        # create folders
        snapping_discharge_folder = (
            Path(self.root).parent / "output" / "build" / "snapping_discharge"
        )
        snapping_discharge_folder.mkdir(parents=True, exist_ok=True)

        Q_obs_merged = Q_obs.copy()  # Ensure Q_obs_merged is always defined

        if custom_river_stations is not None:
            for station in os.listdir(Path(self.root).parent / custom_river_stations):
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
                        station, Q_station, dt_format="%Y-%m-%d %H:%M:%S", startrow=3
                    )
                    # Check for missing or invalid dates
                    if Q_station.index.isnull().any():
                        raise ValueError(
                            "Datetime parsing failed. Found Nan values in the index."
                        )

                    # add station to Q_obs if station is not already in Q_obs
                    if station_name not in Q_obs.station_name.values:
                        station_id = int(
                            Q_obs_merged.id.max() + 1
                        )  # ID for the new station
                        Q_obs_merged = add_station_Q_obs(
                            station_id,
                            station_name,
                            Q_obs_merged,
                            station_coords,
                            Q_station,
                        )  # name, coordinates, dataframe
                    else:
                        station_id = int(
                            Q_obs.id.values[Q_obs.station_name.values == station_name][
                                0
                            ]
                        )  # get the id of the station in the Q_obs dataset

        Q_obs_clipped = clip_Q_obs(
            Q_obs_merged, region_mask
        )  # filter Q_obs stations based on the region shapefile

        if len(Q_obs_clipped.id) == 0:
            # No stations found - create empty files
            self.logger.warning(
                "No discharge stations found in the region. Creating empty files"
            )
            # Create empty snapping results Excel file with proper columns
            empty_cols = [
                "Q_obs_station_name",
                "Q_obs_station_ID",
                "Q_obs_river_name",
                "Q_obs_upstream_area_m2",
                "Q_obs_station_coords",
                "closest_point_coords",
                "subgrid_pixel_coords",
                "snapped_grid_pixel_lonlat",
                "snapped_grid_pixel_xy",
                "GEB_upstream_area_from_subgrid",
                "GEB_upstream_area_from_grid",
                "Q_obs_to_GEB_upstream_area_ratio",
                "snapping_distance_degrees",
            ]
            discharge_snapping_df = pd.DataFrame(columns=empty_cols)
            discharge_snapping_df.to_excel(
                self.report_dir / "snapping_discharge" / "discharge_snapping.xlsx",
                index=False,
            )

            # Create empty discharge table
            empty_discharge_df = pd.DataFrame()
            self.set_table(empty_discharge_df, name="discharge/Q_obs")

            # Create empty snapped locations geometry
            empty_geom = gpd.GeoDataFrame(
                discharge_snapping_df,
                geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                crs="EPSG:4326",
            ).set_index(pd.Index([], name="Q_obs_station_ID"))
            self.set_geom(empty_geom, name="discharge/discharge_snapped_locations")

            self.logger.info("Empty discharge datasets created")

            return
        # convert all the -999 values to NaN
        Q_obs_clipped = Q_obs_clipped.where(Q_obs_clipped != -999, np.nan)

        # check if there are any NaN values in the Q_obs dataset
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
        if len(Q_obs_clipped.id.values) == 0:
            self.logger.warning(
                "No discharge stations found in basin area. Skipping discharge snapping."
            )
            return  # Exit the method early

        # Snapping to river and validation of discharges
        # create list for results of snapping
        discharge_snapping_results = []

        # start looping over the Q_obs stations
        for station_id in tqdm(Q_obs_clipped.id.values):
            # create Q_obs variables
            Q_obs_station = Q_obs_clipped.sel(
                id=station_id
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
            Q_obs_uparea_m2 = (
                Q_obs_station.area.values.item()
            ) * 1e6  # get the upstream area of the station
            Q_obs_rivername = Q_obs_station.river_name.values.item()

            # find river section closest to the Q_obs station
            rivers["station_distance"] = rivers.geometry.apply(
                lambda geom: get_distance_to_stations(geom, Q_obs_location)
            )  # distance in degrees
            rivers_sorted = rivers.sort_values(by="station_distance")

            closest_river_segment = select_river_segment(
                max_uparea_difference_ratio=max_uparea_difference_ratio,
                max_spatial_difference_degrees=max_spatial_difference_degrees,
                Q_obs_uparea_m2=Q_obs_uparea_m2,
                rivers_sorted=rivers_sorted,
            )
            if closest_river_segment is False:
                self.logger.warning(
                    f"No river segment found within the max_uparea_difference_ratio ({max_uparea_difference_ratio}) and max_spatial_difference_degrees ({max_spatial_difference_degrees}) for station {Q_obs_station_name} with upstream area {Q_obs_uparea_m2} m2. Skipping this station."
                )
                continue

            closest_river_segment_linestring = shapely.geometry.LineString(
                closest_river_segment.iloc[0].geometry
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
            xy_tuples = closest_river_segment.iloc[
                0
            ][
                "hydrography_xy"
            ]  # get the xy of the river pixels of the grid (already prepared and stored as hydrography_xy)

            if xy_tuples.size == 0:
                self.logger.warning(
                    f"River not found in hydrography_xy for station {Q_obs_station_name} with river id {closest_river_segment.iloc[0].name}. Skipping this station."
                )
                continue

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

            discharge_snapping_results.append(
                {
                    "Q_obs_station_name": Q_obs_station_name,
                    "Q_obs_station_ID": int(station_id),
                    "Q_obs_river_name": Q_obs_rivername,
                    "Q_obs_upstream_area_m2": Q_obs_uparea_m2,
                    "Q_obs_station_coords": Q_obs_station_coords,
                    "closest_point_coords": closest_point_coords,
                    "subgrid_pixel_coords": subgrid_pixel_coords,
                    "snapped_grid_pixel_lonlat": grid_pixel_coords,
                    "snapped_grid_pixel_xy": closest_tuple,
                    "GEB_upstream_area_from_subgrid": float(
                        GEB_upstream_area_from_subgrid
                    ),
                    "GEB_upstream_area_from_grid": float(GEB_upstream_area_from_grid),
                    "Q_obs_to_GEB_upstream_area_ratio": float(
                        GEB_upstream_area_from_subgrid / Q_obs_uparea_m2
                    ),
                    "snapping_distance_degrees": closest_river_segment.station_distance.iloc[
                        0
                    ],
                }
            )

            plot_snapping(
                station_id,
                self.report_dir / "snapping_discharge",
                rivers,
                upstream_area,
                Q_obs_station_coords,
                closest_point_coords,
                closest_river_segment,
                grid_pixel_coords,
            )

        self.logger.info("Discharge snapping done for all stations")

        discharge_snapping_df = pd.DataFrame(discharge_snapping_results)

        # save to excel and parquet files
        discharge_snapping_df.to_excel(
            self.report_dir / "snapping_discharge" / "discharge_snapping.xlsx",
            index=False,
        )  # save the dataframe to an excel file

        discharge_snapping_gdf = gpd.GeoDataFrame(
            discharge_snapping_df,
            geometry=gpd.points_from_xy(
                discharge_snapping_df["snapped_grid_pixel_lonlat"].apply(
                    lambda coord: coord[0]
                ),
                discharge_snapping_df["snapped_grid_pixel_lonlat"].apply(
                    lambda coord: coord[1]
                ),
            ),
            crs="EPSG:4326",  # Set the coordinate reference system
        ).set_index("Q_obs_station_ID")

        # drop the columns that have not associated snapped stations
        discharge_df = discharge_df[discharge_snapping_gdf.index]

        self.set_table(
            discharge_df, name="discharge/Q_obs"
        )  # save the discharge data as a table

        self.set_geom(
            discharge_snapping_gdf, name="discharge/discharge_snapped_locations"
        )

        self.logger.info("Building discharge datasets done")
