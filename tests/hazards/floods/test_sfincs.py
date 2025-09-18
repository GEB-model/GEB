"""Tests for the SFINCS flood model and its integration in GEB."""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from geb.cli import CONFIG_DEFAULT, parse_config, run_model_with_method
from geb.hazards.floods.sfincs import SFINCSRootModel, SFINCSSimulation
from geb.hazards.floods.workflows.utils import get_start_point
from geb.hydrology.HRUs import load_geom
from geb.model import GEBModel
from geb.workflows.io import WorkingDirectory, open_zarr

from ...testconfig import IN_GITHUB_ACTIONS, tmp_folder

working_directory: Path = tmp_folder / "model"
TEST_MODEL_NAME = "test_model"


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
    nodes: gpd.GeoDataFrame = geb_model.sfincs.rivers
    nodes["geometry"] = nodes["geometry"].apply(get_start_point)
    nodes.index = list(np.arange(1, len(nodes) + 1))
    timeseries: pd.DataFrame = pd.DataFrame(
        {
            "time": pd.date_range(start=start_time, end=end_time, freq="H"),
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
        rivers=geb_model.sfincs.rivers,
        discharge=geb_model.sfincs.discharge_spinup_ds,
        waterbody_ids=geb_model.model.hydrology.grid.decompress(
            geb_model.model.hydrology.grid.var.waterBodyID
        ),
        river_width_alpha=geb_model.model.hydrology.grid.decompress(
            geb_model.model.var.river_width_alpha
        ),
        river_width_beta=geb_model.model.hydrology.grid.decompress(
            geb_model.model.var.river_width_beta
        ),
        mannings=geb_model.sfincs.mannings,
        resolution=500,
        nr_subgrid_pixels=nr_subgrid_pixels,
        crs=geb_model.sfincs.crs,
        depth_calculation_method=geb_model.model.config["hydrology"]["routing"][
            "river_depth"
        ]["method"],
        depth_calculation_parameters=geb_model.model.config["hydrology"]["routing"][
            "river_depth"
        ]["parameters"]
        if "parameters"
        in geb_model.sfincs.model.config["hydrology"]["routing"]["river_depth"]
        else {},
        mask_flood_plains=False,  # setting this to True sometimes leads to errors
    )

    if nr_subgrid_pixels is None:
        assert (sfincs_model.path / "sfincs.dep").exists()
    else:
        assert (sfincs_model.path / "sfincs_subgrid.nc").exists()

    return sfincs_model


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "infiltrate",
    [
        True,
        False,
    ],
)
def test_SFINCS_precipitation(geb_model: GEBModel, infiltrate: bool) -> None:
    """Test SFINCS with precipitation forcing.

    Args:
        geb_model: A GEB model instance with SFINCS instance.
        infiltrate: If True, infiltration is allowed. If False, all precipitation becomes runoff.
    """
    with WorkingDirectory(working_directory):
        start_time: datetime = datetime(2000, 1, 1, 0)
        end_time: datetime = datetime(2000, 1, 10, 0)
        no_rainfall_time = timedelta(days=1)

        sfincs_model: SFINCSRootModel = build_sfincs(geb_model, nr_subgrid_pixels=None)

        simulation: SFINCSSimulation = sfincs_model.create_simulation(
            "precipitation_forcing_test",
            start_time=start_time,
            end_time=end_time,
            spinup_seconds=0,
        )

        simulation.sfincs_model.set_config("storecumprcp", 1)

        rainfall_rate: float = 1.0 / 3600  # mm/hr -> kg/m²/s

        precipitation: xr.DataArray = open_zarr(
            geb_model.files["other"]["climate/pr_hourly"]
        ).sel(time=slice(start_time, end_time))
        precipitation_grid: xr.DataArray = xr.full_like(
            precipitation, rainfall_rate, dtype=np.float64
        )
        # set the first day to 0
        precipitation_grid.loc[
            dict(time=slice(start_time, start_time + no_rainfall_time))
        ] = 0

        mask: xr.DataArray = sfincs_model.sfincs_model.grid["msk"]

        current_water_storage_grid: xr.DataArray = xr.full_like(
            mask, 0.15, dtype=np.float32
        )

        if infiltrate:
            max_water_storage_grid: xr.DataArray = xr.full_like(
                mask, 0.30, dtype=np.float32
            )
            targeted_recovery_rate = 0.15  # m/day
        else:
            max_water_storage_grid: xr.DataArray = current_water_storage_grid
            targeted_recovery_rate = 0.0

        saturated_hydraulic_conductivity = (
            (targeted_recovery_rate * 75 / (max_water_storage_grid.mean().item() * 24))
            ** 2
            * (25.4 / 1000)
            * 24
        )
        saturated_hydraulic_conductivity_grid: xr.DataArray = xr.full_like(
            mask,
            saturated_hydraulic_conductivity / (24 * 3600),
            dtype=np.float32,
        )  # m/day -> m/s

        simulation.set_precipitation_forcing_grid(
            current_water_storage_grid=current_water_storage_grid,
            max_water_storage_grid=max_water_storage_grid,
            saturated_hydraulic_conductivity_grid=saturated_hydraulic_conductivity_grid,
            precipitation_grid=precipitation_grid,
        )

        assert (simulation.path / "sfincs.seff").exists()
        assert (simulation.path / "sfincs.smax").exists()
        assert (simulation.path / "sfincs.ks").exists()
        assert (simulation.path / "precip_2d.nc").exists()
        assert (simulation.path / "sfincs.inp").exists()

        parameterized_max_infiltration: float = (
            max_water_storage_grid - current_water_storage_grid
        ).values[
            simulation.active_cells
        ].sum() * sfincs_model.cell_area + targeted_recovery_rate / (
            24 * 3600
        ) * sfincs_model.cell_area * no_rainfall_time.total_seconds()

        simulation.run(gpu=False)
        flood_depth: xr.DataArray = simulation.read_final_flood_depth(
            minimum_flood_depth=0.0
        )
        # get total flood volume
        flood_volume: float = simulation.get_flood_volume(flood_depth)

        rainfall_volume: float = (
            precipitation_grid.mean().compute().item()
            * (end_time - start_time).total_seconds()
            * sfincs_model.area
            / 1000
        )

        cumulative_precipitation = simulation.get_cumulative_precipitation().compute()
        cumulative_precipitation = (
            cumulative_precipitation.mean().item() * sfincs_model.area
        )

        cumulative_infiltration_simulated = (
            simulation.get_cumulative_infiltration().compute()
        )
        cumulative_infiltration_simulated = (
            cumulative_infiltration_simulated.mean().item() * sfincs_model.area
        )

        if infiltrate:
            # simulated infiltration should be between 50% and 100% of parameterized infiltration
            assert parameterized_max_infiltration >= cumulative_infiltration_simulated
            assert (
                parameterized_max_infiltration * 0.5
                <= cumulative_infiltration_simulated
            )

        # print("Cumulative precipitation:", cumulative_precipitation)
        # print("rainfall volume:", rainfall_volume)
        # print("flood volume:", flood_volume)
        # print("Simulated cumulative infiltration:", cumulative_infiltration_simulated)
        # print("Parameterized max infiltration:", parameterized_max_infiltration)
        # print(
        #     "Infiltration to runoff",
        #     cumulative_infiltration_simulated / cumulative_precipitation,
        # )

        assert math.isclose(
            flood_volume + cumulative_infiltration_simulated,
            rainfall_volume,
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
            time=pd.date_range(start=start_time, end=end_time, freq="H")
        )

        waterbody_ids: npt.NDArray[np.int32] = (
            geb_model.model.hydrology.grid.decompress(
                geb_model.model.hydrology.grid.var.waterBodyID
            )
        )

        simulation.set_headwater_forcing_from_grid(
            discharge_grid=discharge_grid, waterbody_ids=waterbody_ids
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
