import os
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel

from .io import import_rivers
from .sfincs_utils import (
    configure_sfincs_model,
    get_discharge_and_river_parameters_by_river,
    get_logger,
    get_representative_river_points,
    get_start_point,
)


def to_sfincs_datetime(dt: datetime) -> str:
    """Convert a datetime object to a string in the format required by SFINCS.

    Args:
        dt: datetime object to convert.

    Returns:
        String representation of the datetime in the format "YYYYMMDD HHMMSS".
    """
    return dt.strftime("%Y%m%d %H%M%S")


def update_sfincs_model_forcing_coastal(
    model_files: dict,
    model_root: Path,
    simulation_root: Path,
    return_period: int,
) -> None:
    """Update the SFINCS model forcing for coastal flooding.

    Notes:
        This function now only updates forcing with storm surge hydrographs. Compound flooding is not yet considered

    Args:
        model_files: Dictionary containing model file paths.
        model_root: Path to the model root directory.
        simulation_root: Path to the simulation root directory.
        return_period: Return period for the simulation.

    """
    # read model
    sf: SfincsModel = SfincsModel(root=model_root, mode="r+", logger=get_logger())

    locations = (
        gpd.GeoDataFrame(
            gpd.read_parquet(model_files["geom"]["gtsm/stations_coast_rp"])
        )
        .rename(columns={"station_id": "stations"})
        .set_index("stations")
    )
    # convert index to int
    locations.index = locations.index.astype(int)
    timeseries = pd.read_csv(
        Path(
            f"output/hydrographs/gtsm_spring_tide_hydrograph_rp{return_period:04d}.csv"
        ),
        index_col=0,
    )
    timeseries.index = pd.to_datetime(timeseries.index, format="%Y-%m-%d %H:%M:%S")
    # convert columns to int
    timeseries.columns = timeseries.columns.astype(int)
    assert timeseries.columns.equals(locations.index)

    # Align timeseries columns with locations index
    timeseries.columns = locations.index
    timeseries = timeseries.iloc[300:-300]  # trim the first and last 300 rows

    sf.setup_config(
        tref=to_sfincs_datetime(timeseries.index[0]),
        tstart=to_sfincs_datetime(timeseries.index[0]),
        tstop=to_sfincs_datetime(timeseries.index[-1]),
    )
    # set forcing and configure model
    sf.setup_waterlevel_forcing(timeseries=timeseries, locations=locations)
    configure_sfincs_model(sf, model_root, simulation_root)


def setup_infiltration(
    sfincs_model: SfincsModel,
    max_water_storage: xr.Dataset,
    soil_water_capacity: xr.Dataset,
    saturated_hydraulic_conductivity: xr.Dataset,
) -> None:
    """Set up infiltration parameters in the SFINCS model.

    Uses the curve number method with recovery.

    Args:
        sfincs_model: SfincsModel object to update.
        max_water_storage: xarray Dataset containing maximum water storage (smax).
        soil_water_capacity: xarray Dataset containing soil water capacity (seff).
        saturated_hydraulic_conductivity: xarray Dataset containing saturated hydraulic conductivity (ks).
    """
    max_water_storage = max_water_storage.raster.reproject_like(
        sfincs_model.grid, method="average"
    )
    max_water_storage = max_water_storage.rename_vars({"max_water_storage": "smax"})
    max_water_storage.attrs.update(**sfincs_model._ATTRS.get("smax", {}))
    sfincs_model.set_grid(max_water_storage, name="smax")
    sfincs_model.set_config("smaxfile", "sfincs.smax")

    soil_water_capacity = soil_water_capacity.raster.reproject_like(
        sfincs_model.grid, method="average"
    )
    soil_water_capacity = soil_water_capacity.rename_vars(
        {"soil_storage_capacity": "seff"}
    )
    soil_water_capacity.attrs.update(**sfincs_model._ATTRS.get("seff", {}))
    sfincs_model.set_grid(soil_water_capacity, name="seff")
    sfincs_model.set_config("sefffile", "sfincs.seff")

    saturated_hydraulic_conductivity = (
        saturated_hydraulic_conductivity.raster.reproject_like(
            sfincs_model.grid, method="average"
        )
    )
    saturated_hydraulic_conductivity = saturated_hydraulic_conductivity.rename_vars(
        {"saturated_hydraulic_conductivity": "ks"}
    )
    saturated_hydraulic_conductivity.attrs.update(**sfincs_model._ATTRS.get("ks", {}))
    sfincs_model.set_grid(saturated_hydraulic_conductivity, name="ks")
    sfincs_model.set_config("ksfile", "sfincs.ks")


def update_sfincs_model_forcing(
    model_root: Path,
    simulation_root: Path,
    event: dict[str, Any],
    discharge_grid: str | xr.DataArray,
    waterbody_ids: npt.NDArray[np.int32],
    soil_water_capacity_grid: xr.Dataset,
    max_water_storage_grid: xr.Dataset,
    saturated_hydraulic_conductivity_grid: xr.Dataset,
    forcing_method: str,
    precipitation_grid: None | xr.DataArray | list[xr.DataArray] = None,
) -> None:
    """Sets forcing for a SFINCS model based on the provided parameters.

    Creates a new simulation directory and creteas a new sfincs model
    in that folder. Variables that do not change between simulations
    (e.g., grid, mask, etc.) are retained in the original model folder
    and inherited by the new simulation using relative paths.

    Args:
        model_root: Path to the model root directory.
        simulation_root: Path to the simulation root directory.
        event: Dictionary containing event details such as start and end times.
        discharge_grid: Discharge grid as xarray Dataset or path to netcdf file.
        waterbody_ids: Array of waterbody IDs, of identical x and y dimensions as discharge_grid.
        soil_water_capacity_grid: Dataset containing soil water capacity (seff).
        max_water_storage_grid: Dataset containing maximum water storage (smax).
        saturated_hydraulic_conductivity_grid: Dataset containing saturated hydraulic conductivity (ks).
        forcing_method: Method to set forcing, either "headwater_points" or "precipitation".
        precipitation_grid: Precipitation grid as xarray DataArray or list of DataArrays. Can also be None when
            forcing method is headwater_points. Defaults to None.

    Raises:
        ValueError: If an invalid forcing method is provided.
    """
    assert os.path.isfile(os.path.join(model_root, "sfincs.inp")), (
        f"model root does not exist {model_root}"
    )
    if not isinstance(discharge_grid, str):
        assert isinstance(discharge_grid, xr.DataArray), (
            "discharge_grid should be a string or a xr.DataArray"
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
    sfincs_model: SfincsModel = SfincsModel(
        root=model_root, mode="r+", logger=get_logger()
    )

    # update mode time based on event tstart and tend from event dict
    sfincs_model.setup_config(
        tref=to_sfincs_datetime(event["start_time"]),
        tstart=to_sfincs_datetime(event["start_time"]),
        tstop=to_sfincs_datetime(event["end_time"]),
    )

    if forcing_method == "headwater_points":
        rivers = import_rivers(model_root)
        rivers_with_forcing_point = rivers[~rivers["is_downstream_outflow_subbasin"]]
        headwater_rivers = rivers_with_forcing_point[
            rivers_with_forcing_point["maxup"] == 0
        ]

        inflow_nodes = headwater_rivers.copy()

        # Only select headwater points. Maxup is the number of upstream river segments.
        inflow_nodes["geometry"] = inflow_nodes["geometry"].apply(get_start_point)

        river_representative_points = []
        for ID in headwater_rivers.index:
            river_representative_points.append(
                get_representative_river_points(ID, headwater_rivers, waterbody_ids)
            )

        discharge_by_river, _ = get_discharge_and_river_parameters_by_river(
            headwater_rivers.index,
            river_representative_points,
            discharge=discharge_grid,
        )

        locations = inflow_nodes.to_crs(sfincs_model.crs)
        index_mapping = {
            idx: i + 1
            for i, idx in enumerate(locations.index)  # SFINCS index starts at 1
        }
        locations.index = locations.index.map(index_mapping)
        locations.index.name = "sfincs_idx"
        discharge_by_river.columns = discharge_by_river.columns.map(index_mapping)

        # Give discharge_forcing_points as forcing points
        sfincs_model.setup_discharge_forcing(
            locations=inflow_nodes.to_crs(sfincs_model.crs),
            timeseries=discharge_by_river,
        )

    elif forcing_method == "precipitation":
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

        sfincs_model.set_forcing(
            (precipitation_grid * 3600).to_dataset(name="precip_2d"), name="precip_2d"
        )  # convert from kg/m2/s to mm/h

        setup_infiltration(
            sfincs_model=sfincs_model,
            max_water_storage=max_water_storage_grid,
            soil_water_capacity=soil_water_capacity_grid,
            saturated_hydraulic_conductivity=saturated_hydraulic_conductivity_grid,
        )

    else:
        raise ValueError(
            "Invalid forcing method. Choose between 'headwater_points' and 'precipitation'"
        )

    # detect whether water level forcing should be set (use this under forcing == coastal) PLot basemap and forcing to check
    if (
        sfincs_model.grid["msk"] == 2
    ).any():  # if mask is 2, the model requires water level forcing
        waterlevel = sfincs_model.data_catalog.get_dataset(
            "waterlevel"
        ).compute()  # define water levels and stations in data_catalog.yml

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

        sfincs_model.setup_waterlevel_forcing(
            timeseries=timeseries, locations=locations
        )

    configure_sfincs_model(sfincs_model, model_root, simulation_root)
