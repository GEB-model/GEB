import json
import os
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from tqdm import tqdm

from geb.workflows.io import to_zarr

from .io import import_rivers
from .postprocess_model import read_maximum_flood_depth
from .sfincs_utils import (
    get_end_point,
    get_logger,
    get_start_point,
    make_relative_paths,
    run_sfincs_simulation,
)
from .update_model_forcing import update_sfincs_model_forcing_coastal


def get_topological_stream_order(rivers):
    """Calculate the topological stream order for each river segment.

    The topological stream order is calculated by following the river rivers upstream
    and each time finding the rivers that connect to the previous segment.
    """
    # find endpoints of each geometry
    startpoint = rivers.geometry.apply(lambda geom: geom.coords[0])
    endpoint = rivers.geometry.apply(lambda geom: geom.coords[-1])

    topological_stream_order = np.full(len(rivers), -1, dtype=np.int32)

    prev_order_idx = ~endpoint.isin(startpoint)
    topological_stream_order_idx: int = 0
    topological_stream_order[prev_order_idx] = topological_stream_order_idx

    prev_order_start_points = startpoint[prev_order_idx]

    while len(prev_order_start_points) > 0:
        topological_stream_order_idx += 1

        next_order_idx = endpoint.isin(prev_order_start_points)
        topological_stream_order[next_order_idx] = topological_stream_order_idx

        next_order_start_points = startpoint[next_order_idx]
        prev_order_start_points = next_order_start_points

    assert (topological_stream_order != -1).all()

    return topological_stream_order


def assign_calculation_group(rivers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign calculation groups to river segments based on their topological stream order and endpoints.

    We want to run every river rivers independently of the other river rivers. Here, we make
    two observations:

    1. If two river rivers have the same endpoint, they must performed in another run.
    2. If a river shares its endpoint with the startpoint of another river, they must be performed in another run.

    Thus, we first split rivers by their topological stream order. If more there is at least one stream
    order between two rivers, they can be performed in the same run. We can simplify this to running every
    even stream order in one run and every odd stream order in another run.

    Furthermore, when two (or in rare cases three) rivers share the same endpoint, we must also run them
    in different runs. Therefore, we group the rivers by their endpoint and assign them to a calculation group.

    The outcome is calculation groups 1-6, where 3 and 6 are only used when there are three rivers
    sharing the same endpoint.

    Args:
        rivers: GeoDataFrame with the river rivers

    Returns:
        GeoDataFrame with an additional column 'calculation_group' that assigns the rivers to a calculation group
    """

    def assign(group):
        # There cannot be four rivers with the same endpoint (there must be an outflow point)
        assert len(group) <= 3, "Found more than 3 nodes with the same endpoint"
        # Stream order must be the same for rivers within a group
        assert group["topological_stream_order"].nunique() == 1, (
            f"Found nodes with different topological stream order: {group['topological_stream_order'].unique()}"
        )
        # Use modulo 2 to assign even and odd stream orders to different calculation groups,
        # then multiply by 3 to get 0 and 3 as calculation groups (leaving 1, 2 and 3)
        calculation_group = (group["topological_stream_order"] % 2 * 3).values
        # for one of the branches in a group, increment the calculation group by 1
        # using the previously open 1 and 3 group
        if calculation_group.size == 2:
            calculation_group[1] += 1
        elif calculation_group.size == 3:
            calculation_group[1] += 1
            calculation_group[2] += 2
        group["calculation_group"] = calculation_group

        return group

    return (
        rivers.groupby(rivers["geometry"].apply(get_end_point))
        .apply(assign)
        .reset_index(drop=True)
    )


def run_sfincs_for_return_periods_coastal(
    model: SfincsModel,
    model_root: Path,
    gpu: bool = True,
    return_periods: list[int] = [2, 5, 10, 25, 50, 100, 250, 500, 1000],
    export_dir: Path | None = None,
    clean_working_dir: bool = True,
    export: bool = True,
) -> dict[int, xr.DataArray]:
    """Run SFINCS for coastal return periods.

    Notes:
        Updates the forcing of the build coastal sfincs model with storm surge hydrographs

    Args:
        model: The SFINCS model to run.
        model_root: The root directory of the SFINCS model.
        gpu: Whether to use GPU acceleration.
        return_periods: List of return periods to simulate.
        export_dir: Directory to export results.
        clean_working_dir: Whether to clean the working directory before simulation.
        export: Whether to export results.

    Returns:
       Dictionary containing the maximum flood depth for each return period.
    """
    if export_dir is None:
        export_dir: Path = model_root / "risk"

    export_dir.mkdir(exist_ok=True, parents=True)

    # rivers: gpd.GeoDataFrame = import_rivers(model_root, postfix="_return_periods")
    # assert (~rivers["is_downstream_outflow_subbasin"]).all()

    # rivers["topological_stream_order"] = get_topological_stream_order(rivers)
    # rivers: gpd.GeoDataFrame = assign_calculation_group(rivers)

    working_dir: Path = model_root / "working_dir"
    rp_maps = {}

    for return_period in return_periods:
        simulation_root = working_dir / f"coastal_rp_{return_period:04d}"

        shutil.rmtree(simulation_root, ignore_errors=True)  # remove old simulation root
        simulation_root.mkdir(parents=True, exist_ok=True)

        update_sfincs_model_forcing_coastal(
            model_files=model.files,
            model_root=model_root,
            simulation_root=simulation_root,
            return_period=return_period,
        )

        sf: SfincsModel = SfincsModel(
            root=simulation_root, mode="r+", logger=get_logger()
        )

        sf.read()
        # copy the model root to the simulation root
        # sf.set_root(simulation_root, mode="w+")
        # sf._write_gis = False

        sf.setup_config(
            **make_relative_paths(
                sf.config,
                model_root,
                simulation_root,
                relpath=os.path.relpath(model_root, simulation_root),
            )
        )
        sf.write_config()

        # only export if working dir is not cleaned afterwards anyway
        if not clean_working_dir:
            sf.plot_basemap(fn_out="basemap.png")

        run_sfincs_simulation(model_root, simulation_root, gpu=gpu)

        max_depth: xr.DataArray = read_maximum_flood_depth(model_root, simulation_root)
        rp_maps[return_period] = max_depth
        if export:
            max_depth: xr.DataArray = to_zarr(
                max_depth,
                export_dir / f"coastal_{return_period:04d}.zarr",
                crs=max_depth.rio.crs,
            )
    return rp_maps


def run_sfincs_for_return_periods(
    model_root,
    return_periods=[2, 5, 10, 20, 50, 100, 250, 500, 1000],
    clean_working_dir=True,
    export=True,
    export_dir=None,
    gpu=False,
):
    if export_dir is None:
        export_dir: Path = model_root / "risk"

    export_dir.mkdir(exist_ok=True, parents=True)

    rivers: gpd.GeoDataFrame = import_rivers(model_root, postfix="_return_periods")
    assert (~rivers["is_downstream_outflow_subbasin"]).all()

    rivers["topological_stream_order"] = get_topological_stream_order(rivers)
    rivers: gpd.GeoDataFrame = assign_calculation_group(rivers)

    working_dir: Path = model_root / "working_dir"

    rp_maps = {}

    for return_period in return_periods:
        print(f"Running SFINCS for return period {return_period} years")
        rp_map = []

        working_dir_return_period: Path = working_dir / f"rp_{return_period}"
        for group, group_rivers in tqdm(rivers.groupby("calculation_group")):
            simulation_root = working_dir_return_period / str(group)

            shutil.rmtree(simulation_root, ignore_errors=True)
            simulation_root.mkdir(parents=True, exist_ok=True)

            inflow_nodes = group_rivers.copy()
            inflow_nodes = inflow_nodes.reset_index(drop=True)
            inflow_nodes["geometry"] = inflow_nodes["geometry"].apply(get_start_point)

            Q: list[pd.DataFrame] = [
                pd.DataFrame.from_dict(
                    inflow_nodes[f"hydrograph_{return_period}"].iloc[idx],
                    orient="index",
                    columns=[idx],
                )
                for idx in inflow_nodes.index
            ]
            Q: pd.DataFrame = pd.concat(Q, axis=1)
            Q.index = pd.to_datetime(Q.index)

            Q: pd.DataFrame = (
                Q.fillna(method="ffill").fillna(  # pad with 0's before
                    method="bfill"
                )  # and after
            )

            sf: SfincsModel = SfincsModel(
                root=model_root, mode="r+", logger=get_logger()
            )
            sf.setup_config(
                tref=Q.index[0],
                tstart=Q.index[0],
                tstop=Q.index[-1],
            )

            sf.setup_discharge_forcing(
                locations=inflow_nodes.to_crs(sf.crs),
                timeseries=Q,
            )
            # before changing root read outflow_points from model root gis folder
            outflow = gpd.read_file(Path(sf.root) / "gis/outflow_points.gpkg")
            # only one point location is expected
            assert len(outflow) == 1, "Only one outflow point is expected"
            # before changing root read dem value from gis folder from .json file
            dem_json_path = model_root / "gis" / "outflow_elevation.json"
            with open(dem_json_path, "r") as f:
                dem_values = json.load(f)
            elevation = dem_values.get("outflow_elevation", None)

            if elevation is None or elevation == 0:
                assert False, (
                    "Elevation should have positive value to set up outflow waterlevel boundary"
                )

            # Get the model's start and stop time using the get_model_time function
            tstart, tstop = sf.get_model_time()

            # Define the time range (e.g., 1 month of hourly data)
            time_range = pd.date_range(start=tstart, end=tstop, freq="H")

            # Create DataFrame with constant elevation value
            elevation_time_series_constant = pd.DataFrame(
                data={"water_level": elevation},  # Use extracted elevation value
                index=time_range,
            )

            # Extract a unique index from the outflow point. Here, we use 1 as an example.
            outflow_index = (
                1  # This should be the index or a suitable ID of the outflow point
            )
            elevation_time_series_constant.columns = [
                outflow_index
            ]  # Use an integer as column name

            # Ensure outflow has the correct index as well
            outflow["index"] = (
                outflow_index  # Set the matching index to outflow location
            )

            # Now set the water level forcing
            sf.setup_waterlevel_forcing(
                timeseries=elevation_time_series_constant,  # Constant time series
                locations=outflow,  # Outflow point
            )
            sf.set_root(simulation_root, mode="w+")
            sf._write_gis = False

            sf.setup_config(
                **make_relative_paths(
                    sf.config,
                    model_root,
                    simulation_root,
                    relpath=os.path.relpath(model_root, simulation_root),
                )
            )

            sf.write_forcing()
            sf.write_config()
            sf.plot_forcing(fn_out="forcing.png")
            # only export if working dir is not cleaned afterwards anyway
            if not clean_working_dir:
                sf.plot_basemap(fn_out="basemap.png")

            run_sfincs_simulation(model_root, simulation_root, gpu=gpu)

            max_depth: xr.DataArray = read_maximum_flood_depth(
                model_root, simulation_root
            )

            rp_map.append(max_depth)

        rp_map: xr.DataArray = xr.concat(rp_map, dim="node")
        rp_map: xr.DataArray = rp_map.max(dim="node")
        rp_map.attrs["_FillValue"] = max_depth.attrs["_FillValue"]

        if export:
            rp_map: xr.DataArray = to_zarr(
                rp_map,
                export_dir / f"riverine_{return_period:04d}.zarr",
                crs=rp_map.rio.crs,
            )

        if clean_working_dir:
            assert working_dir_return_period.exists()
            shutil.rmtree(working_dir_return_period, ignore_errors=True)

        rp_maps[return_period] = rp_map

    return rp_maps
