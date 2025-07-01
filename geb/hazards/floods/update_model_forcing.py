import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from shapely.geometry import Point

from .io import import_rivers
from .sfincs_utils import configure_sfincs_model, get_logger


def to_sfincs_datetime(dt: datetime) -> str:
    """Convert a datetime object to a string in the format required by SFINCS.

    Args:
        dt: datetime object to convert.

    Returns:
        String representation of the datetime in the format "YYYYMMDD HHMMSS".
    """
    return dt.strftime("%Y%m%d %H%M%S")


def update_sfincs_model_forcing(
    model_root,
    simulation_root,
    event,
    discharge_grid,
    uparea_discharge_grid,
    forcing_method,
    precipitation_grid=None,
):
    assert os.path.isfile(os.path.join(model_root, "sfincs.inp")), (
        f"model root does not exist {model_root}"
    )
    if not isinstance(discharge_grid, str):
        assert isinstance(discharge_grid, xr.Dataset), (
            "discharge_grid should be a string or a xr.Dataset"
        )
        assert discharge_grid.raster.crs is not None, "discharge_grid should have a crs"
        assert (
            pd.to_datetime(discharge_grid.time[0].item()).to_pydatetime()
            <= event["start_time"]
        )
        assert (
            pd.to_datetime(discharge_grid.time[-1].item()).to_pydatetime()
        ) >= event["end_time"]

    # read model
    sf: SfincsModel = SfincsModel(root=model_root, mode="r+", logger=get_logger())

    # update mode time based on event tstart and tend from event dict
    sf.setup_config(
        tref=to_sfincs_datetime(event["start_time"]),
        tstart=to_sfincs_datetime(event["start_time"]),
        tstop=to_sfincs_datetime(event["end_time"]),
    )
    segments = import_rivers(model_root)

    if precipitation_grid is not None:
        if not isinstance(precipitation_grid, list):
            assert isinstance(precipitation_grid, xr.DataArray), (
                "precipitation_grid should be a list or an xr.DataArray"
            )
            precipitation_grid: list[xr.DataArray] = [precipitation_grid]

        precipitation_grid: list[xr.DataArray] = [
            pr.raster.reproject_like(sf.grid) for pr in precipitation_grid
        ]

        precipitation_grid: xr.DataArray = xr.concat(precipitation_grid, dim="time")

        assert precipitation_grid.raster.crs is not None, (
            "precipitation_grid should have a crs"
        )
        assert (
            pd.to_datetime(precipitation_grid.time[0].item()).to_pydatetime()
            <= event["start_time"]
        )
        assert (
            pd.to_datetime(precipitation_grid.time[-1].item()).to_pydatetime()
            >= event["end_time"]
        )

        precipitation_grid: xr.DataArray = precipitation_grid.sel(
            time=slice(event["start_time"], event["end_time"])
        )

        sf.set_forcing(
            (precipitation_grid * 3600).to_dataset(name="precip_2d"), name="precip_2d"
        )  # convert from kg/m2/s to mm/h

    if forcing_method == "headwater_points":
        # TODO: Cleanup and re-use nodes (or create nodes here)
        exploded = segments[segments["order"] == 1]
        exploded = exploded.explode(
            index_parts=True
        )  # To Handle any MULTILINE Geometries, we explode the MULTILINE to seperate LINES
        head_water_points = gpd.GeoDataFrame(
            columns=["uparea", "geometry"], crs=segments.crs
        )

        # Iterate through the rows of the original GeoDataFrame
        for index, row in exploded.iterrows():
            if row["order"] == 1:
                start_point = Point(
                    row["geometry"].coords[
                        0
                    ]  # TODO: Check if this is the correct point to use
                )  # Extract the starting point of the LineString
                head_water_points.loc[len(head_water_points)] = [
                    row["uparea"],
                    start_point,
                ]

        # If needed, set the geometry column
        head_water_points.set_geometry("geometry", inplace=True)
        # Align the CRS of the head_water_points with the river segments
        head_water_points = head_water_points.set_crs(crs=segments.crs)
        # update the upstream area point using merit_hydro uparea
        uparea_sfincs = sf.data_catalog.get_rasterdataset(
            "merit_hydro",
            bbox=sf.mask.raster.transform_bounds(4326),
            buffer=2,
            variables=["uparea"],
        )
        assert head_water_points.crs == uparea_sfincs.rio.crs, (
            "CRS mismatch between head_water_points and uparea_sfincs, make sure they are in the same CRS"
        )
        head_water_points["uparea"] = uparea_sfincs.raster.sample(head_water_points)
        # Combine River_inflow with Headwater points
        head_water_points = head_water_points.to_crs(sf.crs)
        if (
            "dis" in sf.forcing
        ):  # we have a basin which has both river inflow and headwater points
            river_inflow_points = sf.forcing["dis"].vector.to_gdf()
            river_inflow_points = river_inflow_points.to_crs(
                4326
            )  # <------Change crs to 4326 before sampling uparea
            uparea_sfincs = sf.data_catalog.get_rasterdataset(
                "merit_hydro",
                bbox=sf.mask.raster.transform_bounds(4326),
                buffer=2,
                variables=["uparea"],
            )
            assert river_inflow_points.crs == uparea_sfincs.rio.crs, (
                "CRS mismatch between river_inflow_points and uparea_sfincs, make sure they are in the same CRS"
            )
            river_inflow_points["uparea"] = uparea_sfincs.raster.sample(
                river_inflow_points
            )
            river_inflow_points = river_inflow_points.to_crs(
                sf.crs
            )  # <------Change crs back to sf.crs
            # Concatenate the two DataFrames
            assert river_inflow_points.crs == head_water_points.crs, (
                "CRS mismatch between river_inflow_points and head_water_points"
            )
            discharge_forcing_points = gpd.GeoDataFrame(
                pd.concat(
                    [
                        river_inflow_points[
                            ["geometry", "uparea"]
                        ],  # Select the "geometry" and "uparea "column from the dataframe
                        head_water_points[
                            ["geometry", "uparea"]
                        ],  # Select the "geometry" and "uparea" column from the dataframe
                    ],
                    ignore_index=True,
                )
            )
        else:
            discharge_forcing_points = head_water_points  # otherwise we have a basin with only headwater points

        discharge_forcing_points = discharge_forcing_points.to_crs(
            sf.crs
        )  # <--------------------------------------------------Use For setting up Discharge forcing
        discharge_forcing_points.to_file(
            model_root / "inflow_points.gpkg", driver="GPKG"
        )
        # Give discharge_forcing_points as forcing points
        sf.setup_discharge_forcing(locations=discharge_forcing_points)

    elif forcing_method == "precipitation":
        # Only set inflow points (not using headwater points)
        if (
            "dis" in sf.forcing
        ):  # if no inflow points (headwater catchment) don't set discharge forcing
            sf.setup_discharge_forcing()
        # curve number infiltration based on global CN dataset

    else:
        raise ValueError(
            "Invalid forcing method. Choose between 'headwater_points' and 'precipitation'"
        )

    if (
        precipitation_grid is None
    ):  # Case being basin contains both river_inflow & Headwater points
        # in some cases the geometry of the geojson is slightly different from the geometry of the forcing points.
        # This leads to errors. Therefore, we first read the forcing points and then update the geometry of the geojson
        # to those of the forcing points
        locations_all = gpd.read_file(Path(sf.root) / "gis" / "src.geojson")
        locations_all["index"] = locations_all["index"] + 1
        locations_all = locations_all.set_index("index")
        forcing_points = sf.forcing["dis"].vector.to_gdf()
        # Assert that the geometry columns are almost equal
        for point1, point2 in zip(locations_all.iterrows(), forcing_points.iterrows()):
            assert point1[1]["geometry"].almost_equals(point2[1]["geometry"], decimal=0)
            assert point1[0] == point2[0]
        locations_all["geometry"] = forcing_points["geometry"]

        bounds = sf.mask.raster.transform_bounds(4326)
        if isinstance(discharge_grid, (xr.Dataset, xr.DataArray)):
            discharge_grid = discharge_grid.rio.pad_box(
                minx=bounds[0],
                miny=bounds[1],
                maxx=bounds[2],
                maxy=bounds[3],
                constant_values=0,
            )
        sf.setup_discharge_forcing_from_grid(
            discharge=discharge_grid,
            locations=locations_all,  # all_points(river_inflow+Headwater Points)
            uparea=uparea_discharge_grid,
        )
        assert len(locations_all) == sf.forcing["dis"].shape[1]
    else:  # Case being basin contains only river_inflow points, and precipitation grid is provided
        if "dis" in sf.forcing:
            forcing_points = sf.forcing["dis"].vector.to_gdf()
            # change crs to 4326 if needed
            forcing_points = forcing_points.to_crs(4326)
            uparea_sfincs = sf.data_catalog.get_rasterdataset(
                "merit_hydro",
                bbox=sf.mask.raster.transform_bounds(4326),
                buffer=2,
                variables=["uparea"],
            )
            assert forcing_points.crs == uparea_sfincs.rio.crs, (
                "CRS mismatch between river_inflow_points and uparea_sfincs, make sure they are in the same CRS"
            )
            forcing_points["uparea"] = uparea_sfincs.raster.sample(forcing_points)
            forcing_points = forcing_points.to_crs(
                sf.crs
            )  # change crs back to model crs after sampling uparea
            uparea_discharge_grid = uparea_discharge_grid.compute()

            # TODO: Create a proper assertion for this
            # sf.setup_discharge_forcing_from_grid(
            #     discharge=discharge_grid,
            #     locations=forcing_points,  # Only river inflow point here
            #     uparea=uparea_discharge_grid,
            # )

    # detect whether water level forcing should be set
    if (
        sf.grid["msk"] == 2
    ).any():  # if mask is 2, the model requires water level forcing
        waterlevel = sf.data_catalog.get_dataset("waterlevel").compute()

        locations = gpd.GeoDataFrame(
            index=waterlevel.stations,
            geometry=gpd.points_from_xy(
                waterlevel.station_x_coordinate, waterlevel.station_y_coordinate
            ),
            crs=4326,
        )

        timeseries = pd.DataFrame(
            index=waterlevel.time, columns=waterlevel.stations, data=waterlevel.data
        )
        assert timeseries.columns.equals(locations.index)

        locations = locations.reset_index(names="stations")
        locations.index = (
            locations.index + 1
        )  # for hydromt/SFINCS index should start at 1
        timeseries.columns = locations.index

        sf.setup_waterlevel_forcing(timeseries=timeseries, locations=locations)

    configure_sfincs_model(sf, model_root, simulation_root)
