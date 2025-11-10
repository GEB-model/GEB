"""Tests for the SFINCS flood model and its integration in GEB."""

import json
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
from geb.hazards.floods.sfincs import SFINCSRootModel, SFINCSSimulation
from geb.hazards.floods.workflows.utils import get_start_point
from geb.model import GEBModel
from geb.typing import (
    TwoDArrayFloat64,
    TwoDArrayInt32,
)
from geb.workflows.io import WorkingDirectory, load_geom, open_zarr

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
    discharge_value: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Create a discharge timeseries for the given nodes.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
        start_time: The start time of the timeseries.
        end_time: The end time of the timeseries.
        discharge_value: The discharge value to use for all nodes and times.

    Returns:
        A tuple with the nodes and the timeseries.
    """
    nodes: gpd.GeoDataFrame = geb_model.hydrology.routing.rivers
    nodes["geometry"] = nodes["geometry"].apply(get_start_point)
    nodes.index = list(np.arange(1, len(nodes) + 1))
    timeseries: pd.DataFrame = pd.DataFrame(
        {
            "time": pd.date_range(start=start_time, end=end_time, freq="h"),
            **{node_id: discharge_value for node_id in nodes.index},
        }
    ).set_index("time")
    return nodes, timeseries


def build_sfincs(geb_model: GEBModel, nr_subgrid_pixels: int | None) -> SFINCSRootModel:
    """Build a SFINCS model instance, including the static grids and configuration.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
        nr_subgrid_pixels: Number of subgrid pixels to use. If None, no subgrid pixels are used.

    Returns:
        A SFINCS model instance with static grids and configuration written.
    """
    sfincs_model: SFINCSRootModel = SFINCSRootModel(geb_model, TEST_MODEL_NAME)
    with open(geb_model.model.files["dict"]["hydrodynamics/DEM_config"]) as f:
        DEM_config = json.load(f)
        for entry in DEM_config:
            entry["elevtn"] = open_zarr(
                geb_model.model.files["other"][entry["path"]]
            ).to_dataset(name="elevtn")

    sfincs_model.build(
        region=load_geom(geb_model.model.files["geom"]["routing/subbasins"]),
        DEMs=DEM_config,
        rivers=geb_model.hydrology.routing.rivers,
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_SFINCS_runoff(geb_model: GEBModel) -> None:
    """Test SFINCS with runoff forcing.

    Args:
        geb_model: A GEB model instance with SFINCS instance.
    """
    with WorkingDirectory(working_directory):
        start_time: datetime = datetime(2000, 1, 1, 0)
        end_time: datetime = datetime(2000, 1, 10, 0)

        sfincs_model: SFINCSRootModel = build_sfincs(geb_model, nr_subgrid_pixels=None)

        simulation: SFINCSSimulation = sfincs_model.create_simulation(
            "runoff_forcing_test",
            start_time=start_time,
            end_time=end_time,
            spinup_seconds=0,
        )

        simulation.sfincs_model.set_config("storecumprcp", 1)

        runoff_rate_mm_per_hr: float = 1.0  # mm/hr
        runoff_rate_m_per_hr = runoff_rate_mm_per_hr / 1000.0

        precipitation: xr.DataArray = open_zarr(
            geb_model.files["other"]["climate/pr_kg_per_m2_per_s"]
        ).sel(time=slice(start_time, end_time))
        runoff_m: xr.DataArray = xr.full_like(
            precipitation, runoff_rate_m_per_hr, dtype=np.float64
        )
        mask: xr.DataArray = sfincs_model.sfincs_model.grid["msk"]

        simulation.set_runoff_forcing(
            runoff_m=runoff_m,
        )

        assert (simulation.path / "precip_2d.nc").exists()
        assert (simulation.path / "sfincs.inp").exists()

        simulation.run(gpu=False)
        flood_depth: xr.DataArray = simulation.read_final_flood_depth(
            minimum_flood_depth=0.0
        )
        # get total flood volume
        flood_volume: float = simulation.get_flood_volume(flood_depth)

        runoff_volume: float = (
            runoff_rate_m_per_hr
            * sfincs_model.area
            * ((end_time - start_time).total_seconds() / 3600.0)
        )

        cumulative_runoff = simulation.get_cumulative_precipitation().compute()
        cumulative_runoff = cumulative_runoff.mean().item() * sfincs_model.area

        assert math.isclose(
            flood_volume,
            runoff_volume,
            abs_tol=0,
            rel_tol=0.01,
        )
        assert math.isclose(
            cumulative_runoff,
            runoff_volume,
            abs_tol=0,
            rel_tol=0.01,
        )
        simulation.cleanup()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_SFINCS_accumulated_runoff(geb_model: GEBModel) -> None:
    """Test SFINCS with accumulated runoff forcing.

    Args:
        geb_model: A GEB model instance with SFINCS instance.
    """
    with WorkingDirectory(working_directory):
        start_time: datetime = datetime(2000, 1, 1, 0)
        end_time: datetime = datetime(2000, 1, 10, 0)

        sfincs_model: SFINCSRootModel = build_sfincs(geb_model, nr_subgrid_pixels=None)

        simulation: SFINCSSimulation = sfincs_model.create_simulation(
            "accumulated_runoff_forcing_test",
            start_time=start_time,
            end_time=end_time,
            spinup_seconds=0,
            write_figures=True,
        )

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

        river_ids: TwoDArrayInt32 = geb_model.hydrology.grid.load(
            geb_model.files["grid"]["routing/river_ids"], compress=False
        )
        upstream_area = geb_model.hydrology.grid.load(
            geb_model.files["grid"]["routing/upstream_area"], compress=False
        )

        discarded_generated_discharge_m3_per_s: np.float64 = (
            simulation.set_accumulated_runoff_forcing(
                runoff_m=runoff_m,
                river_network=geb_model.hydrology.routing.river_network,
                mask=~geb_model.hydrology.grid.mask,
                river_ids=river_ids,
                upstream_area=upstream_area,
                cell_area=geb_model.hydrology.grid.decompress(
                    geb_model.hydrology.grid.var.cell_area
                ),
                river_geometry=geb_model.floods.rivers,
            )
        )

        assert (simulation.path / "sfincs.dis").exists()
        assert (simulation.path / "sfincs.src").exists()

        assert not simulation.has_outflow_boundary()

        simulation.run(gpu=False)
        flood_depth = simulation.read_final_flood_depth(minimum_flood_depth=0.00)
        total_flood_volume = simulation.get_flood_volume(flood_depth)

        runoff_volume: float = (
            runoff_rate_m_per_hr
            * geb_model.hydrology.grid.var.cell_area.sum()
            * ((end_time - start_time).total_seconds() / 3600.0)
        )
        discarded_discharge = (
            discarded_generated_discharge_m3_per_s
            * (end_time - start_time).total_seconds()
        )

        # compare to total discharge volume
        assert math.isclose(
            total_flood_volume,
            runoff_volume - discarded_discharge,
            abs_tol=0,
            rel_tol=0.01,
        )

        simulation.cleanup()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "use_gpu",
    [False] + ([True] if os.getenv("GEB_TEST_GPU", "no") == "yes" else []),
)
def test_SFINCS_discharge_from_nodes(geb_model: GEBModel, use_gpu: bool) -> None:
    """Test SFINCS with discharge forcing from river nodes.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
        use_gpu: Whether to use the GPU for the simulation.
    """
    with WorkingDirectory(working_directory):
        start_time: datetime = datetime(2000, 1, 1, 0)
        end_time: datetime = datetime(2000, 1, 8, 0)

        sfincs_model = build_sfincs(geb_model, nr_subgrid_pixels=None)

        simulation = sfincs_model.create_simulation(
            "nodes_forcing_test",
            start_time=start_time,
            end_time=end_time,
        )

        nodes, timeseries = create_discharge_timeseries(
            geb_model, start_time, end_time, discharge_value=10.0
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
            (timeseries.values.sum() * 3600).item(),
            abs_tol=0,
            rel_tol=0.01,
        )

        simulation.cleanup()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_SFINCS_discharge_grid_forcing(geb_model: GEBModel) -> None:
    """Test SFINCS with discharge forcing from a grid.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
    """
    with WorkingDirectory(working_directory):
        start_time: datetime = datetime(2000, 1, 1, 0)
        end_time: datetime = datetime(2000, 1, 8, 0)

        sfincs_model: SFINCSRootModel = build_sfincs(geb_model, nr_subgrid_pixels=None)

        simulation: SFINCSSimulation = sfincs_model.create_simulation(
            "grid_forcing_test",
            start_time=start_time,
            end_time=end_time,
            write_figures=True,
        )

        discharge_rate: float = 10.0  # m3/s

        discharge_grid: xr.DataArray = xr.full_like(
            sfincs_model.sfincs_model.grid["msk"],
            discharge_rate,
            dtype=np.float64,
        )

        # repeat for all timesteps
        discharge_grid = discharge_grid.expand_dims(
            time=pd.date_range(start=start_time, end=end_time, freq="h")
        )

        simulation.set_headwater_forcing_from_grid(
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

        number_of_rivers: int = sfincs_model.sfincs_model.forcing["dis"].index.size

        # compare to total discharge volume
        assert math.isclose(
            total_flood_volume,
            (
                number_of_rivers
                * (end_time - start_time).total_seconds()
                * discharge_rate
            ),
            abs_tol=0,
            rel_tol=0.01,
        )

        simulation.cleanup()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_read(geb_model: GEBModel) -> None:
    """Test reading SFINCS output files.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
    """
    with WorkingDirectory(working_directory):
        sfincs_model_build = build_sfincs(geb_model, nr_subgrid_pixels=None)
        sfincs_model_read: SFINCSRootModel = SFINCSRootModel(
            geb_model, TEST_MODEL_NAME
        ).read()

        # assert that both models have the same attributes
        assert sfincs_model_build.path == sfincs_model_read.path
        assert sfincs_model_build.name == sfincs_model_read.name
        assert sfincs_model_build.cell_area == sfincs_model_read.cell_area
        assert sfincs_model_build.area == sfincs_model_read.area
        assert sfincs_model_build.path == sfincs_model_read.path

        for key in sfincs_model_build.sfincs_model.config:
            assert (
                sfincs_model_build.sfincs_model.config[key]
                == sfincs_model_read.sfincs_model.config[key]
            )
