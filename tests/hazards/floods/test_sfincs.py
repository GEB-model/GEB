import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geb.cli import CONFIG_DEFAULT, parse_config, run_model_with_method
from geb.hazards.floods.model import SFINCSRootModel
from geb.hazards.floods.workflows.utils import get_start_point
from geb.hydrology.HRUs import load_geom
from geb.workflows.io import WorkingDirectory, open_zarr

from ...testconfig import IN_GITHUB_ACTIONS, tmp_folder

working_directory: Path = tmp_folder / "model"


@pytest.fixture
def geb_model():
    with WorkingDirectory(working_directory):
        config = parse_config(CONFIG_DEFAULT)
        config["hazards"]["floods"]["simulate"] = True
        model = run_model_with_method(config=config, method=None, close_after_run=False)
        model.run(initialize_only=True)
    return model


def build_sfincs(geb_model, nr_subgrid_pixels: int | None = 4) -> SFINCSRootModel:
    sfincs_model = SFINCSRootModel(geb_model, "test_model")
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
        resolution=geb_model.sfincs.config["resolution"],
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
    ("infiltrate", "nr_subgrid_pixels"),
    [
        (True, None),
        (False, None),
        (True, 4),
        (False, 4),
    ],
)
def test_SFINCS_precipitation(geb_model, infiltrate, nr_subgrid_pixels) -> None:
    with WorkingDirectory(working_directory):
        start_time = datetime(2000, 1, 1, 0)
        end_time = datetime(2000, 1, 8, 0)

        sfincs_model = build_sfincs(geb_model, nr_subgrid_pixels=nr_subgrid_pixels)

        simulation = sfincs_model.create_simulation(
            f"precipitation_forcing_test_{'infiltrate' if infiltrate else ''}{'_subgrid' if nr_subgrid_pixels else ''}",
            start_time=start_time,
            end_time=end_time,
            spinup_seconds=0,
        )

        rainfall_rate = 1.0 / 3600  # mm/hr -> kg/m²/s

        precipitation = open_zarr(geb_model.files["other"]["climate/pr_hourly"]).sel(
            time=slice(start_time, end_time)
        )
        precipitation_grid = xr.full_like(
            precipitation, rainfall_rate, dtype=np.float64
        )

        subgrid = open_zarr(geb_model.files["subgrid"]["mask"])

        if not infiltrate:
            max_water_storage_grid = xr.full_like(subgrid, 0.0, dtype=np.float32)
            soil_water_capacity_grid = xr.full_like(subgrid, 0.0, dtype=np.float32)
            saturated_hydraulic_conductivity_grid = xr.full_like(
                subgrid, 0.0, dtype=np.float32
            )
        else:
            max_water_storage_grid = xr.full_like(subgrid, 0.0, dtype=np.float32)
            soil_water_capacity_grid = xr.full_like(subgrid, 0.0, dtype=np.float32)
            saturated_hydraulic_conductivity_grid = xr.full_like(
                subgrid, 0.0, dtype=np.float32
            )

        simulation.set_precipitation_forcing_grid(
            soil_water_capacity_grid=soil_water_capacity_grid,
            max_water_storage_grid=max_water_storage_grid,
            saturated_hydraulic_conductivity_grid=saturated_hydraulic_conductivity_grid,
            precipitation_grid=precipitation_grid,
        )
        assert (simulation.path / "sfincs.seff").exists()
        assert (simulation.path / "sfincs.smax").exists()
        assert (simulation.path / "sfincs.ks").exists()
        assert (simulation.path / "precip_2d.nc").exists()

        assert (simulation.path / "sfincs.inp").exists()

        simulation.run(gpu=False)
        flood_depth = simulation.read_max_flood_depth(minimum_flood_depth=0.0)
        flood_depth = flood_depth.compute()
        flood_depth_ = simulation.read_final_flood_depth(minimum_flood_depth=0.0)
        flood_depth_ = flood_depth_.compute()

        # get total flood volume
        flood_volume = simulation.get_flood_volume(flood_depth)

        rainfall_volume = (
            rainfall_rate
            * (end_time - start_time).total_seconds()
            * sfincs_model.area
            / 1000
        )

        print(flood_volume, rainfall_volume)

        assert math.isclose(flood_volume, rainfall_volume, abs_tol=0, rel_tol=0.3)

        simulation.cleanup()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize("nr_subgrid_pixels", [None, 4])
def test_SFINCS_discharge_from_nodes(geb_model, nr_subgrid_pixels) -> None:
    """Test SFINCS with discharge forcing from river nodes.

    Args:
        geb_model: A GEB model instance with SFINCS configured.
    """
    geb_model.sfincs.config["nr_subgrid_pixels"] = None

    with WorkingDirectory(working_directory):
        start_time = datetime(2000, 1, 1, 0)
        end_time = datetime(2000, 1, 8, 0)

        sfincs_model = build_sfincs(geb_model, nr_subgrid_pixels=nr_subgrid_pixels)

        simulation = sfincs_model.create_simulation(
            "nodes_forcing_test",
            start_time=start_time,
            end_time=end_time,
        )

        nodes = geb_model.sfincs.rivers
        nodes["geometry"] = nodes["geometry"].apply(get_start_point)
        nodes.index = list(np.arange(1, len(nodes) + 1))

        timeseries = pd.DataFrame(
            {
                "time": pd.date_range(start=start_time, end=end_time, freq="H"),
                **{str(node_id): 10.0 for node_id in nodes.index},
            }
        ).set_index("time")

        simulation.set_discharge_forcing_from_nodes(
            nodes=nodes,
            timeseries=timeseries,
        )

        assert (simulation.path / "sfincs.dis").exists()

        simulation.run(gpu=False)
        flood_depth = simulation.read_max_flood_depth(minimum_flood_depth=0.00)
        flood_depth_ = simulation.read_final_flood_depth(minimum_flood_depth=0.00)
        flood_depth = flood_depth.compute()
        flood_depth_ = flood_depth_.compute()

        # get total flood volume
        total_flood_volume = simulation.get_flood_volume(flood_depth)

        # compare to total discharge volume
        assert math.isclose(
            total_flood_volume,
            (timeseries.values.sum() * 3600).item(),
            abs_tol=0,
            rel_tol=0.01,
        )

        simulation.cleanup()
