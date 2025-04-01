import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel

from geb.workflows.io import to_zarr

from .io import import_rivers
from .postprocess_model import read_flood_map
from .sfincs_utils import (
    get_end_point,
    get_logger,
    get_start_point,
    make_relative_paths,
    run_sfincs_simulation,
)


def get_topological_stream_order(rivers):
    """Calculate the topological stream order for each river segment. The topological stream order is
    calculated by following the river rivers upstream and each time finding the rivers that connect
    to the previous segment."""
    # find endpoints of each geometry
    startpoint = rivers.geometry.apply(lambda geom: geom.coords[0])
    endpoint = rivers.geometry.apply(lambda geom: geom.coords[-1])

    topological_stream_order = np.full(len(rivers), -1, dtype=np.int32)

    prev_order_idx = ~endpoint.isin(startpoint)
    topological_stream_order_idx = 0
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


def assign_calculation_group(rivers):
    """We want to run every river rivers independently of the other river rivers. Here, we make
    two observations:

    1. If two river rivers have the same endpoint, they must performed in another run.
    2. If a river shares its endpoint with the startpoint of another river, they must be performed in another run.

    Thus, we first split rivers by their topological stream order. If more there is at least one stream
    order between two rivers, they can be performed in the same run. We can simplify this to running every
    even stream order in one run and every odd stream order in another run.

    Furthermore, when two rivers share the same endpoint, we must also run them in different runs. Therefore,
    we group the rivers by their endpoint and assign them to a calculation group.

    Parameters
    ----------
    rivers : GeoDataFrame
        GeoDataFrame with the river rivers

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with an additional column 'calculation_group' that assigns the rivers to a calculation group
    """

    def assign(group):
        # There cannot be three rivers with the same endpoint
        assert len(group) <= 2, "Found more than 2 nodes with the same endpoint"
        # Stream order must be the same for rivers within a group
        assert group["topological_stream_order"].nunique() == 1, (
            f"Found nodes with different topological stream order: {group['topological_stream_order'].unique()}"
        )
        # Use modulo 2 to assign even and odd stream orders to different calculation groups,
        # then multiply by 2 to get 0 and 2 as calculation groups (leaving 1 and 3)
        calculation_group = (group["topological_stream_order"] % 2 * 2).values
        # for one of the branches in a group, increment the calculation group by 1
        # using the previously open 1 and 3 group
        if calculation_group.size == 2:
            calculation_group[1] += 1
        group["calculation_group"] = calculation_group

        return group

    return (
        rivers.groupby(rivers["geometry"].apply(get_end_point))
        .apply(assign)
        .reset_index(drop=True)
    )


def run_sfincs_for_return_periods(
    model_root,
    return_periods=[2, 5, 10, 20, 50, 100, 250, 500, 1000],
    working_dir=Path(os.getenv("TMPDIR", "/tmp")),
    clean_working_dir=True,
    export=True,
    export_dir=None,
    gpu=False,
):
    if export_dir is None:
        export_dir = model_root / "risk"

    export_dir.mkdir(exist_ok=True, parents=True)

    rivers = import_rivers(model_root, postfix="_return_periods")
    assert (~rivers["is_downstream_outflow_subbasin"]).all()

    rivers["topological_stream_order"] = get_topological_stream_order(rivers)
    rivers = assign_calculation_group(rivers)

    rp_maps = {}

    for return_period in return_periods:
        rp_map = []

        working_dir_return_period = working_dir / f"rp_{return_period}"
        for group, group_rivers in rivers.groupby("calculation_group"):
            simulation_root = working_dir_return_period / str(group)

            shutil.rmtree(simulation_root, ignore_errors=True)
            simulation_root.mkdir(parents=True, exist_ok=True)

            inflow_nodes = group_rivers.copy()
            inflow_nodes = inflow_nodes.reset_index(drop=True)
            inflow_nodes["geometry"] = inflow_nodes["geometry"].apply(get_start_point)

            Q = [
                pd.DataFrame.from_dict(
                    inflow_nodes[f"hydrograph_{return_period}"].iloc[idx],
                    orient="index",
                    columns=[idx],
                )
                for idx in inflow_nodes.index
            ]
            Q = pd.concat(Q, axis=1)
            Q.index = pd.to_datetime(Q.index)

            Q = (
                Q.fillna(method="ffill").fillna(  # pad with 0's before
                    method="bfill"
                )  # and after
            )

            sf = SfincsModel(root=model_root, mode="r+", logger=get_logger())
            sf.setup_config(
                tref=Q.index[0],
                tstart=Q.index[0],
                tstop=Q.index[-1],
            )

            sf.setup_discharge_forcing(
                locations=inflow_nodes.to_crs(sf.crs),
                timeseries=Q,
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

            # only export if working dir is not cleaned afterwards anyway
            if not clean_working_dir:
                sf.plot_basemap(fn_out="basemap.png")

            run_sfincs_simulation(model_root, simulation_root, gpu=gpu)

            max_depth = read_flood_map(model_root, simulation_root)

            rp_map.append(max_depth)

        rp_map = xr.concat(rp_map, dim="node")
        rp_map = rp_map.max(dim="node")
        rp_map.attrs["_FillValue"] = max_depth.attrs["_FillValue"]

        if export:
            rp_map = to_zarr(
                rp_map,
                export_dir / f"{return_period}.zarr",
                crs=rp_map.rio.crs,
            )

        if clean_working_dir:
            assert working_dir_return_period.exists()
            shutil.rmtree(working_dir_return_period, ignore_errors=True)

        rp_maps[return_period] = rp_map

    return rp_maps


if __name__ == "__main__":
    if "snakemake" in locals():
        model_root = Path(snakemake.params.model_root)  # noqa: F821
    else:
        model_root = Path(f"models/basin{232508}")

    run_sfincs_for_return_periods(model_root, clean_working_dir=True, gpu=True)
