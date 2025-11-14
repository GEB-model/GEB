"""Tests for the SFINCS flood model and its integration in GEB."""

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geb.cli import CONFIG_DEFAULT, parse_config, run_model_with_method
from geb.hazards.floods import create_river_graph, group_subbasins
from geb.hazards.floods.sfincs import SFINCSRootModel, SFINCSSimulation
from geb.hazards.floods.workflows.utils import get_start_point
from geb.model import GEBModel
from geb.typing import TwoDArrayFloat64, TwoDArrayInt32
from geb.workflows.io import WorkingDirectory, load_dict, load_geom, open_zarr
from geb.workflows.raster import rasterize_like

from ...testconfig import IN_GITHUB_ACTIONS, tmp_folder

working_directory: Path = tmp_folder / "model"
TEST_MODEL_NAME: str = "test_model"


@pytest.fixture
def geb_model() -> GEBModel:
    """A GEB model instance with SFINCS instance.

    Returns:
        A GEB model instance with SFINCS instance.
    """
    with WorkingDirectory(working_directory):
        config: dict[str, Any] = parse_config(CONFIG_DEFAULT)
        config["hazards"]["floods"]["simulate"] = True
        model: GEBModel = run_model_with_method(
            config=config, method=None, close_after_run=False
        )
        model.run(initialize_only=True)
    return model


def create_discharge_timeseries(
    geb_model: GEBModel,
    start_time: datetime,
    end_time: datetime,
    discharge_value_m3_per_s: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Create a constant discharge timeseries for all river nodes.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
        start_time: The start time of the timeseries.
        end_time: The end time of the timeseries.
        discharge_value_m3_per_s: The constant discharge value to use for all nodes (m³/s).

    Returns:
        A tuple with the nodes and the timeseries.
    """
    nodes: gpd.GeoDataFrame = geb_model.hydrology.routing.rivers
    nodes["geometry"] = nodes["geometry"].apply(get_start_point)
    nodes.index = list(np.arange(1, len(nodes) + 1))
    timeseries: pd.DataFrame = pd.DataFrame(
        {
            "time": pd.date_range(start=start_time, end=end_time, freq="h"),
            **{node_id: discharge_value_m3_per_s for node_id in nodes.index},
        }
    ).set_index("time")
    return nodes, timeseries


def build_sfincs(
    geb_model: GEBModel,
    nr_subgrid_pixels: int | None,
    region: gpd.GeoDataFrame,
    name: str,
    rivers: gpd.GeoDataFrame,
) -> SFINCSRootModel:
    """Build a SFINCS model instance, including the static grids and configuration.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
        nr_subgrid_pixels: Number of subgrid pixels to use. If None, no subgrid pixels are used.
        region: A GeoDataFrame defining the region to build the SFINCS model for.
        name: The name of the SFINCS model. Used for the folder name.
        rivers: A GeoDataFrame with the river network.

    Returns:
        A SFINCS model instance with static grids and configuration written.
    """
    sfincs_model: SFINCSRootModel = SFINCSRootModel(geb_model, name)
    DEM_config: list[dict[str, str | Path | xr.DataArray | xr.Dataset]] = load_dict(
        geb_model.model.files["dict"]["hydrodynamics/DEM_config"]
    )
    for entry in DEM_config:
        entry["elevtn"] = open_zarr(
            geb_model.model.files["other"][entry["path"]]
        ).to_dataset(name="elevtn")

    sfincs_model.build(
        region=region,
        DEMs=DEM_config,
        rivers=rivers,
        discharge=geb_model.floods.discharge_spinup_ds,
        river_width_alpha=geb_model.model.hydrology.grid.decompress(
            geb_model.model.var.river_width_alpha
        ),
        river_width_beta=geb_model.model.hydrology.grid.decompress(
            geb_model.model.var.river_width_beta
        ),
        mannings=geb_model.floods.mannings,
        resolution=500,
        nr_subgrid_pixels=nr_subgrid_pixels,
        crs=geb_model.floods.crs,
        depth_calculation_method=geb_model.model.config["hydrology"]["routing"][
            "river_depth"
        ]["method"],
        depth_calculation_parameters=geb_model.model.config["hydrology"]["routing"][
            "river_depth"
        ]["parameters"]
        if "parameters"
        in geb_model.floods.model.config["hydrology"]["routing"]["river_depth"]
        else {},
        mask_flood_plains=False,  # setting this to True sometimes leads to errors,
        setup_outflow=False,
    )

    if nr_subgrid_pixels is None:
        assert (sfincs_model.path / "sfincs.dep").exists()
    else:
        assert (sfincs_model.path / "sfincs_subgrid.nc").exists()

    return sfincs_model


def create_sfincs_models(
    geb_model: GEBModel,
    subbasins: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    split: bool,
) -> list[SFINCSRootModel]:
    """Create a list of SFINCS models based on whether to split the region.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
        subbasins: The subbasins GeoDataFrame.
        rivers: The rivers GeoDataFrame.
        split: Whether to split the region into multiple models.

    Returns:
        A list of SFINCSRootModel instances.
    """
    if split:
        river_graph = create_river_graph(rivers, subbasins)

        # 2e8 nicely splits the test area into 2 parts. If changing the test area, this value
        # may need to be adjusted.
        grouped_subbasins = group_subbasins(river_graph, max_area_m2=2e8)
        assert len(grouped_subbasins) > 1, (
            "For testing splitting, need multiple groups."
        )

        sfincs_models: list[SFINCSRootModel] = []
        for group_id, group in grouped_subbasins.items():
            group_with_downstream_basins = set(group) | set(
                rivers.loc[rivers.index.isin(group)]["downstream_ID"]
            )
            outflow_basins = group_with_downstream_basins - set(group)
            subbasins_group = subbasins[
                subbasins.index.isin(group_with_downstream_basins)
            ]
            subbasins_group.loc[
                subbasins_group.index.isin(outflow_basins),
                "is_downstream_outflow_subbasin",
            ] = True

            sfincs_model: SFINCSRootModel = build_sfincs(
                geb_model,
                nr_subgrid_pixels=None,
                region=subbasins_group,
                name=f"test_group_{group_id}",
                rivers=rivers[rivers.index.isin(group)],
            )
            sfincs_models.append(sfincs_model)

    else:
        sfincs_models: list[SFINCSRootModel] = [
            build_sfincs(
                geb_model,
                nr_subgrid_pixels=None,
                region=subbasins,
                name=TEST_MODEL_NAME,
                rivers=rivers,
            )
        ]

    return sfincs_models


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "split",
    [False, True],
)
def test_accumulated_runoff(geb_model: GEBModel, split: bool) -> None:
    """Test SFINCS with accumulated runoff forcing.

    Args:
        geb_model: A GEB model instance with SFINCS instance.
        split: Whether to split the domain into multiple SFINCS models for testing.
    """
    with WorkingDirectory(working_directory):
        start_time: datetime = datetime(2000, 1, 1, 0)
        end_time: datetime = datetime(2000, 1, 10, 0)

        subbasins = load_geom(geb_model.model.files["geom"]["routing/subbasins"])
        rivers = geb_model.hydrology.routing.rivers
        sfincs_models = create_sfincs_models(geb_model, subbasins, rivers, split)

        runoff_rate_mm_per_hr: float = 1.0  # mm/hr
        runoff_rate_m_per_hr: float = runoff_rate_mm_per_hr / 1000.0

        runoff_m: TwoDArrayFloat64 = geb_model.hydrology.grid.decompress(
            np.full(
                geb_model.hydrology.grid.compressed_size,
                runoff_rate_m_per_hr,
                dtype=np.float64,
            ),
            fillvalue=0.0,
        )

        runoff_m: xr.DataArray = xr.DataArray(
            runoff_m,
            dims=["y", "x"],
            coords={
                "y": geb_model.hydrology.grid.lat,
                "x": geb_model.hydrology.grid.lon,
            },
        )

        # repeat for all timesteps
        runoff_m: xr.DataArray = runoff_m.expand_dims(
            time=pd.date_range(start=start_time, end=end_time, freq="h")
        )
        runoff_m: xr.DataArray = runoff_m.rio.write_crs(4326)

        river_ids: TwoDArrayInt32 = geb_model.hydrology.grid.load(
            geb_model.files["grid"]["routing/river_ids"], compress=False
        )
        upstream_area = geb_model.hydrology.grid.load(
            geb_model.files["grid"]["routing/upstream_area"], compress=False
        )

        cell_area = geb_model.hydrology.grid.decompress(
            geb_model.hydrology.grid.var.cell_area
        )

        total_flood_volume_across_models: float = 0.0
        total_discharge_volume_across_models: float = 0.0
        total_discarded_generated_discharge_across_models: float = 0.0
        for sfincs_model in sfincs_models:
            simulation: SFINCSSimulation = sfincs_model.create_simulation(
                f"accumulated_runoff_forcing_{sfincs_model.name}",
                start_time=start_time,
                end_time=end_time,
                spinup_seconds=0,
                write_figures=True,
            )

            simulation.set_accumulated_runoff_forcing(
                runoff_m=runoff_m,
                river_network=geb_model.hydrology.routing.river_network,
                mask=~geb_model.hydrology.grid.mask,
                river_ids=river_ids,
                upstream_area=upstream_area,
                cell_area=cell_area,
            )

            if simulation.sfincs_root_model.has_inflow:
                inflow_rivers: gpd.GeoDataFrame = (
                    simulation.sfincs_root_model.inflow_rivers
                )
                inflow_nodes = inflow_rivers.copy()
                inflow_nodes["geometry"] = inflow_nodes["geometry"].apply(
                    get_start_point
                )
                date_range = pd.date_range(
                    simulation.start_time,
                    simulation.end_time,
                    freq="h",
                    inclusive="both",
                )

                discharge_m3_per_s: float = 10
                timeseries = pd.DataFrame(
                    data=np.full(
                        (len(date_range), len(inflow_nodes)),
                        discharge_m3_per_s,
                        dtype=np.float32,
                    ),
                    columns=inflow_nodes.index,
                    index=date_range,
                )

                simulation.set_river_inflow(inflow_nodes, timeseries)
                discharge_m3 = (
                    (simulation.end_time - simulation.start_time).total_seconds()
                    * discharge_m3_per_s
                    * len(inflow_nodes)
                )
            else:
                discharge_m3 = 0.0

            assert (simulation.path / "sfincs.dis").exists()
            assert (simulation.path / "sfincs.src").exists()

            assert not simulation.has_outflow_boundary()

            simulation.run(gpu=False)
            flood_depth = simulation.read_final_flood_depth(minimum_flood_depth=0.00)
            total_flood_volume = simulation.get_flood_volume(flood_depth)

            region = simulation.sfincs_model.region.to_crs(runoff_m.rio.crs)
            region["value"] = 1
            region_mask = rasterize_like(
                region,
                column="value",
                raster=runoff_m.isel(time=0),
                dtype=np.int32,
                nodata=0,
                all_touched=False,
            ).astype(bool)

            total_runoff_volume: float = (
                runoff_rate_m_per_hr
                * (region_mask * cell_area).sum().item()
                * ((end_time - start_time).total_seconds() / 3600.0)
            )

            # Use tracked volumes from the simulation
            non_discharded_runoff_volume: float = simulation.total_discharge_volume_m3
            discarded_discharge: float = (
                simulation.discarded_accumulated_generated_discharge_m3
            )

            # compare to total discharge volume
            assert math.isclose(
                total_flood_volume,
                non_discharded_runoff_volume,
                abs_tol=0,
                rel_tol=0.01,
            )
            assert math.isclose(
                total_flood_volume,
                total_runoff_volume + discharge_m3 - discarded_discharge,
                abs_tol=0,
                rel_tol=0.2,
            )

            total_flood_volume_across_models += total_flood_volume
            total_discarded_generated_discharge_across_models += discarded_discharge
            total_discharge_volume_across_models += discharge_m3

            simulation.cleanup()
            sfincs_model.cleanup()

        if split:
            total_runoff_volume: float = (
                runoff_rate_m_per_hr
                * np.nansum(cell_area)
                * ((end_time - start_time).total_seconds() / 3600.0)
            )

            assert math.isclose(
                total_flood_volume_across_models
                + total_discarded_generated_discharge_across_models,
                total_runoff_volume + total_discharge_volume_across_models,
                abs_tol=0,
                rel_tol=0.05,
            )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "use_gpu",
    [False] + ([True] if os.getenv("GEB_TEST_GPU", "no") == "yes" else []),
)
def test_discharge_from_nodes(geb_model: GEBModel, use_gpu: bool) -> None:
    """Test SFINCS with discharge forcing from river nodes.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
        use_gpu: Whether to use the GPU for the simulation.
    """
    with WorkingDirectory(working_directory):
        start_time: datetime = datetime(2000, 1, 1, 0)
        end_time: datetime = datetime(2000, 1, 8, 0)

        region = load_geom(geb_model.model.files["geom"]["routing/subbasins"])
        sfincs_model = build_sfincs(
            geb_model,
            nr_subgrid_pixels=None,
            region=region,
            name=TEST_MODEL_NAME,
            rivers=geb_model.hydrology.routing.rivers,
        )

        simulation = sfincs_model.create_simulation(
            "nodes_forcing_test",
            start_time=start_time,
            end_time=end_time,
        )

        nodes, timeseries = create_discharge_timeseries(
            geb_model, start_time, end_time, discharge_value_m3_per_s=10.0
        )

        simulation.set_discharge_forcing_from_nodes(
            nodes=nodes,
            timeseries=timeseries,
        )

        assert (simulation.path / "sfincs.dis").exists()
        assert (simulation.path / "sfincs.src").exists()

        assert not simulation.has_outflow_boundary()

        simulation.run(gpu=use_gpu)
        flood_depth = simulation.read_final_flood_depth(minimum_flood_depth=0.00)
        total_flood_volume = simulation.get_flood_volume(flood_depth)

        # compare to total discharge volume
        assert math.isclose(
            total_flood_volume,
            simulation.total_discharge_volume_m3,
            abs_tol=0,
            rel_tol=0.01,
        )

        simulation.cleanup()
        sfincs_model.cleanup()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "split",
    [False, True],
)
def test_discharge_grid_forcing(geb_model: GEBModel, split: bool) -> None:
    """Test SFINCS with discharge forcing from a grid.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
        split: Whether to split the domain into multiple SFINCS models for testing.
    """
    with WorkingDirectory(working_directory):
        start_time: datetime = datetime(2000, 1, 1, 0)
        end_time: datetime = datetime(2000, 1, 8, 0)

        subbasins = load_geom(geb_model.model.files["geom"]["routing/subbasins"])
        rivers = geb_model.hydrology.routing.rivers
        sfincs_models = create_sfincs_models(geb_model, subbasins, rivers, split)

        discharge_rate_m3_per_s: float = 10.0

        total_flood_volume_across_models: float = 0.0
        total_discharge_volume_across_models: float = 0.0

        for sfincs_model in sfincs_models:
            simulation: SFINCSSimulation = sfincs_model.create_simulation(
                f"grid_forcing_test_{sfincs_model.name}",
                start_time=start_time,
                end_time=end_time,
                write_figures=True,
            )

            discharge_grid: xr.DataArray = xr.DataArray(
                discharge_rate_m3_per_s,
                dims=["y", "x"],
                coords={
                    "y": geb_model.hydrology.grid.lat,
                    "x": geb_model.hydrology.grid.lon,
                },
            )

            # Expand to include time dimension for all timesteps
            discharge_grid = discharge_grid.expand_dims(
                time=pd.date_range(start=start_time, end=end_time, freq="h")
            )

            simulation.set_headwater_forcing_from_grid(
                discharge_grid=discharge_grid,
            )

            discharge_grid = discharge_grid * 2.0

            if simulation.sfincs_root_model.has_inflow:
                simulation.set_inflow_forcing_from_grid(
                    discharge_grid=discharge_grid,
                )

            assert (simulation.path / "sfincs.dis").exists()
            assert (simulation.path / "sfincs.src").exists()

            assert not simulation.has_outflow_boundary()

            simulation.run(gpu=False)
            flood_depth: xr.DataArray = simulation.read_final_flood_depth(
                minimum_flood_depth=0.00
            )
            total_flood_volume: float = simulation.get_flood_volume(flood_depth)

            total_flood_volume_across_models += total_flood_volume
            total_discharge_volume_across_models += simulation.total_discharge_volume_m3

            simulation.cleanup()
            sfincs_model.cleanup()

        # compare total flood volume to total discharge volume across all models
        assert math.isclose(
            total_flood_volume_across_models,
            total_discharge_volume_across_models,
            abs_tol=0,
            rel_tol=0.01,
        )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_read(geb_model: GEBModel) -> None:
    """Test reading SFINCS output files.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
    """
    with WorkingDirectory(working_directory):
        region = load_geom(geb_model.model.files["geom"]["routing/subbasins"])
        sfincs_model_build = build_sfincs(
            geb_model,
            nr_subgrid_pixels=None,
            region=region,
            name=TEST_MODEL_NAME,
            rivers=geb_model.hydrology.routing.rivers,
        )

        sfincs_model_read: SFINCSRootModel = SFINCSRootModel(
            geb_model, TEST_MODEL_NAME
        ).read()

        # assert that both models have the same attributes
        assert sfincs_model_build.path == sfincs_model_read.path
        assert sfincs_model_build.name == sfincs_model_read.name
        assert sfincs_model_build.cell_area == sfincs_model_read.cell_area
        assert sfincs_model_build.area == sfincs_model_read.area
        assert sfincs_model_build.path == sfincs_model_read.path
        assert sfincs_model_build.rivers.equals(sfincs_model_read.rivers)

        for key in sfincs_model_build.sfincs_model.config:
            assert (
                sfincs_model_build.sfincs_model.config[key]
                == sfincs_model_read.sfincs_model.config[key]
            )

        sfincs_model_read.cleanup()
