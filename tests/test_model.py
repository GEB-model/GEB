"""Tests for the main model functions of GEB, such as model initialization, building, and running.

Most of these tests are quite heavy and therefore skipped on GitHub Actions.
"""

import os
import shutil
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from click.testing import CliRunner
from shapely.geometry import Point

from geb.build.methods import build_method
from geb.cli import (
    BUILD_DEFAULT,
    CONFIG_DEFAULT,
    alter_fn,
    build_fn,
    cli,
    init_fn,
    run_model_with_method,
    share_fn,
    update_fn,
)
from geb.evaluate.hydrology import _get_datetime_index_step_label
from geb.hydrology.landcovers import FOREST, GRASSLAND_LIKE
from geb.hydrology.routing import get_river_width
from geb.model import GEBModel
from geb.runner import parse_config
from geb.workflows.io import (
    WorkingDirectory,
    read_zarr,
    write_zarr,
)

from .testconfig import IN_GITHUB_ACTIONS, tmp_folder

working_directory: Path = tmp_folder / "model"
working_directory_coastal: Path = tmp_folder / "model_coastal"

DEFAULT_BUILD_ARGS: dict[str, Any] = {"continue_": True}
DEFAULT_RUN_ARGS: dict[str, Any] = {}


def test_get_datetime_index_step_label_uses_day_for_daily_frequency() -> None:
    """Test that daily datetime frequencies use a readable timestep label."""
    time_index: pd.DatetimeIndex = pd.date_range("2000-01-01", periods=3, freq="D")

    assert _get_datetime_index_step_label(time_index) == "day"


@pytest.mark.parametrize(
    "clean_working_directory",
    [False, True],
)
def test_init(clean_working_directory: bool) -> None:
    """Test model initialization from example configuration.

    Creates a new model directory from the 'geul' example, verifies that
    all required configuration files are created, and tests error handling
    when attempting to initialize in an existing directory without overwrite.
    """
    if clean_working_directory and working_directory.exists():
        shutil.rmtree(working_directory, ignore_errors=True)
    working_directory.mkdir(parents=True, exist_ok=True)

    with WorkingDirectory(working_directory):
        args: dict[str, Any] = {
            "config": "model.yml",
            "build_config": "build.yml",
            "update_config": "update.yml",
            "working_directory": ".",
            "from_example": "geul",
        }
        init_fn(
            **args,
            overwrite=True,
        )

        assert Path("model.yml").exists()
        assert Path("build.yml").exists()
        assert Path("update.yml").exists()

        # should raise an error if the folder already exists
        with pytest.raises(FileExistsError):
            init_fn(
                **args,
                overwrite=False,
            )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "clean_working_directory",
    [False, True],
)
def test_init_coastal(clean_working_directory: bool) -> None:
    """Test model initialization of a coastal model from example configuration.

    Creates a new model directory from the 'geul' example, verifies that
    all required configuration files are created, and tests error handling
    when attempting to initialize in an existing directory without overwrite.

    Only the coastal-specific build steps are included in the build configuration.
    """
    if clean_working_directory and working_directory_coastal.exists():
        shutil.rmtree(working_directory_coastal, ignore_errors=True)
    working_directory_coastal.mkdir(parents=True, exist_ok=True)

    with WorkingDirectory(working_directory_coastal):
        args: dict[str, Any] = {
            "config": "model.yml",
            "build_config": "build.yml",
            "update_config": "update.yml",
            "working_directory": ".",
            "from_example": "geul",
            "basin_id": "23010758",
        }
        init_fn(
            **args,
            overwrite=True,
        )

        assert Path("model.yml").exists()
        assert Path("build.yml").exists()
        assert Path("update.yml").exists()

        # should raise an error if the folder already exists
        with pytest.raises(FileExistsError):
            init_fn(
                **args,
                overwrite=False,
            )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "working_directory",
    [working_directory, working_directory_coastal],
)
def test_build(working_directory: Path) -> None:
    """Test the model build process with default arguments.

    Runs the build function with default build arguments to ensure
    the model can be properly built from configuration files.

    Args:
        working_directory: The working directory where the model is built.
    """
    with WorkingDirectory(working_directory):
        build_fn(**DEFAULT_BUILD_ARGS)


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS or os.getenv("GEB_TEST_ALL", "no") != "yes",
    reason="Too heavy for GitHub Actions and needs GEB_TEST_ALL=yes.",
)
def test_build_dependencies() -> None:
    """Test build dependencies by running individual build methods in sequence.

    Tests that each build method can run successfully when its dependencies
    have been built first, ensuring the build dependency graph is correct.
    """
    with WorkingDirectory(working_directory):
        args = DEFAULT_BUILD_ARGS.copy()
        build_config = parse_config(args["build_config"])
        build_config = {"setup_region": build_config["setup_region"]}
        args["build_config"] = build_config
        build_fn(**args)

        shutil.copy(Path("input") / "files.json", Path("input") / "files.json.bak")

    for method in build_method.methods:
        # Skip the setup_region method as this is a special case that is handled separately.
        if method == "setup_region":
            continue

        # get all nodes which this node depends on
        dependencies = build_method.get_dependencies(method)
        with WorkingDirectory(working_directory):
            shutil.copy(Path("input") / "files.json.bak", Path("input") / "files.json")
            args: dict[str, Any] = DEFAULT_BUILD_ARGS.copy()
            build_config_original = parse_config(args["build_config"])

            if method not in build_config_original:
                continue

            build_config = {}
            for dependency in dependencies:
                build_config[dependency] = build_config_original[dependency]

            build_config[method] = build_config_original[method]

            args["build_config"] = build_config
            update_fn(**args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_update_with_file() -> None:
    """Test updating model configuration from a file.

    Verifies that model parameters can be updated by loading
    configuration from a file and that the changes are applied correctly.
    """
    with WorkingDirectory(working_directory):
        args = DEFAULT_BUILD_ARGS.copy()
        del args["continue_"]
        args["build_config"] = Path("update.yml")
        update_fn(**args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_update_with_dict() -> None:
    """Test updating model configuration from a dictionary.

    Verifies that model parameters can be updated by passing
    a dictionary of configuration values and that the changes are applied correctly.
    """
    with WorkingDirectory(working_directory):
        args = DEFAULT_BUILD_ARGS.copy()
        del args["continue_"]
        update = {"setup_land_use_parameters": {}}
        args["build_config"] = update
        update_fn(**args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "method",
    [
        "setup_hydrography",
        "setup_vegetation",
        "setup_water_demand",
        "setup_discharge_observations",
    ],
)
def test_update_with_method(method: str) -> None:
    """Test updating model configuration using different methods.

    Args:
        method: The update method to test (e.g., 'file', 'dict').
    """
    with WorkingDirectory(working_directory):
        args: dict[str, str | dict | Path | bool] = DEFAULT_BUILD_ARGS.copy()
        del args["continue_"]

        build_config: dict[str, dict] = parse_config(BUILD_DEFAULT)

        update: dict[str, dict] = {method: build_config[method]}

        args["build_config"] = update
        update_fn(**args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_profile_model_start() -> None:
    """Test model profiling at start."""
    with WorkingDirectory(working_directory):
        args: dict[str, Any] = DEFAULT_RUN_ARGS.copy()
        args["config"] = parse_config(CONFIG_DEFAULT)
        args["config"]["hazards"]["floods"]["simulate"] = True
        args["profiling"] = True
        args["method_args"] = {
            "initialize_only": True,
        }
        run_model_with_method(method="spinup", **args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_spinup() -> None:
    """Test model spinup phase.

    Verifies that the model can run its spinup phase correctly,
    initializing the system to a stable state before main simulation.
    """
    with WorkingDirectory(working_directory):
        args: dict[str, Any] = DEFAULT_RUN_ARGS.copy()
        args["config"] = parse_config(CONFIG_DEFAULT)
        args["config"]["hazards"]["floods"]["simulate"] = True
        geb: GEBModel = run_model_with_method(
            method="spinup",
            **args,
            method_args={"initialize_only": True},
            close_after_run=False,
            timing=False,
        )

        RIVER_WIDTH_OUTFLOW_RIVER = 30.0

        routing = geb.hydrology.routing
        outflow_rivers = geb.hydrology.routing.outflow_rivers
        routing.observed_average_river_width[
            routing.river_ids == geb.hydrology.routing.outflow_rivers.iloc[0].name
        ] = RIVER_WIDTH_OUTFLOW_RIVER

        geb.step_to_end()
        geb.store.save()

        geb.reporter.finalize()

        routing_report_folder: Path = (
            working_directory / "output" / "report" / "spinup" / "hydrology.routing"
        )

        hourly_discharge_data = read_zarr(
            routing_report_folder / "discharge_hourly.zarr"
        )

        daily_discharge_data = read_zarr(routing_report_folder / "discharge_daily.zarr")

        for ID, river in outflow_rivers.iterrows():
            outflow_data_csv: pd.DataFrame = pd.read_csv(
                routing_report_folder / f"river_outflow_hourly_m3_per_s_{ID}.csv",
                parse_dates=["time"],
            ).set_index("time")[f"river_outflow_hourly_m3_per_s_{ID}"]

            outflow_xy = river["hydrography_xy"][-1]
            hourly_outflow_data_zarr: pd.DataFrame = hourly_discharge_data.isel(
                y=outflow_xy[1], x=outflow_xy[0]
            ).to_dataframe()["discharge_hourly"]

            np.testing.assert_almost_equal(
                hourly_outflow_data_zarr.values, outflow_data_csv.values, decimal=4
            )

            daily_outflow_data_zarr: pd.DataFrame = daily_discharge_data.isel(
                y=outflow_xy[1], x=outflow_xy[0]
            ).to_dataframe()["discharge_daily"]

            # aggregate hourly to daily
            outflow_data_csv_daily = outflow_data_csv.resample("D").mean()
            np.testing.assert_almost_equal(
                daily_outflow_data_zarr.values, outflow_data_csv_daily.values, decimal=4
            )

            # test whether river alpha and beta are correctly calculated
            mean_discharge = np.array(outflow_data_csv.mean(), dtype=np.float32)

            linear_index = geb.hydrology.grid.linear_mapping[
                outflow_xy[1], outflow_xy[0]
            ]
            river_width_alpha = geb.hydrology.grid.var.river_width_alpha[[linear_index]]
            river_width_beta = geb.hydrology.grid.var.river_width_beta[[linear_index]]

            river_width = get_river_width(
                river_width_alpha, river_width_beta, mean_discharge
            )

            # this check only makes sense after 365 days
            if geb.n_timesteps > 365:
                assert river_width == pytest.approx(RIVER_WIDTH_OUTFLOW_RIVER, abs=0.1)

        geb.close()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_forcing() -> None:
    """Test forcing data validation and consistency.

    Verifies that forcing data is properly validated and that
    different forcing sources produce consistent data shapes.
    """
    with WorkingDirectory(working_directory):
        model: GEBModel = run_model_with_method(
            method=None,
            close_after_run=False,
            **DEFAULT_RUN_ARGS,
        )
        model.run(initialize_only=True)

        for name, loader in model.forcing._loaders.items():
            t_0: datetime = datetime(2010, 1, 1, 0, 0, 0)
            forcing_0 = loader.load(t_0)

            t_1: datetime = datetime(2020, 1, 1, 0, 0, 0)
            forcing_1 = loader.load(t_1)

            if isinstance(forcing_0, (xr.DataArray, np.ndarray)) and isinstance(
                forcing_1, (xr.DataArray, np.ndarray)
            ):
                assert forcing_0.shape == forcing_1.shape, (
                    f"Shape of forcing data for {name} does not match for times {t_0} and {t_1}."
                )
                # non-precipitation forcing basically could never be equal, so we check for inequality
                if name != "pr_kg_per_m2_per_s":
                    assert not np.array_equal(forcing_0, forcing_1)
            else:
                assert forcing_0 != forcing_1


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_run() -> None:
    """Test basic model execution.

    Verifies that the model can run a complete simulation
    from initialization through execution without errors, and that the main
    hydrology plot evaluations can be created from the resulting outputs.
    """
    args = DEFAULT_RUN_ARGS.copy()

    with WorkingDirectory(working_directory):
        args["config"] = parse_config(CONFIG_DEFAULT)
        args["config"]["report"].update(
            {
                "_water_circle": True,
                "_water_balance": True,
                "_water_storage": True,
                "_energy_balance": True,
            }
        )
        args["config"]["hazards"]["floods"]["simulate"] = True

        # run_model_with_method(method="run", **args)

        for evaluation_method in (
            "hydrology.plot_water_balance",
            "hydrology.plot_discharge",
            "hydrology.plot_water_storage",
            "hydrology.plot_water_circle",
            "energy.plot_soil_temperature",
        ):
            evaluate_args = DEFAULT_RUN_ARGS.copy()
            evaluate_args["method_args"] = {"method": evaluation_method}
            run_model_with_method(method="evaluate", **evaluate_args)

        hydrology_eval_folder: Path = Path("output") / "evaluate" / "hydrology"
        assert (hydrology_eval_folder / "water_balance_timeseries.svg").exists()
        assert (hydrology_eval_folder / "water_balance_timeseries_yearly.svg").exists()
        assert (
            hydrology_eval_folder / "water_balance_top_soil_timeseries.svg"
        ).exists()
        assert (
            hydrology_eval_folder / "water_balance_top_soil_timeseries_yearly.svg"
        ).exists()
        assert (hydrology_eval_folder / "mean_discharge_m3_per_s.png").exists()
        assert (hydrology_eval_folder / "water_storage_timeseries.svg").exists()
        assert (hydrology_eval_folder / "water_storage_timeseries_yearly.svg").exists()
        assert (hydrology_eval_folder / "outflow").exists()

        method_args = {
            "method": "hydrology.evaluate_discharge",
            "include_yearly_plots": False,
        }
        args["method_args"] = method_args
        result = run_model_with_method(method="evaluate", **args)

        # Verify that the result is a dictionary and contains expected keys
        assert isinstance(result, dict)

        for label in [
            "KGE_hourly",
            "NSE_hourly",
            "R_hourly",
            "KGE_daily",
            "NSE_daily",
            "R_daily",
            "KGE",
            "NSE",
            "R",
        ]:
            assert label in result
            assert result[label] is not None

        # Note this should be much higher.
        assert result["KGE"] > -0.05
        assert result["NSE"] > -0.49
        assert result["R"] > 0.41

        # method_args = {
        #     "method": "hydrodynamics.evaluate_hydrodynamics",
        # }
        # args["method_args"] = method_args
        # result = run_model_with_method(method="evaluate", **args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_alter() -> None:
    """Test model alteration functionality.

    Verifies that a model alternative can be made. A model alternative
    references the original model while overwriting specific settings
    or files. Verifies that the model can be run, but does not check the
    outputs of the altered model.
    """
    with WorkingDirectory(working_directory):
        args: dict[str, Any] = DEFAULT_BUILD_ARGS.copy()
        del args["continue_"]
        args["build_config"] = {
            "set_ssp": {"ssp": "ssp1"},
            "setup_CO2_concentration": {},
        }
        args["working_directory"] = Path("alter")

        args["from_model"] = ".."

        args["working_directory"].mkdir(parents=True, exist_ok=True)

        alter_fn(**args)

        run_args = DEFAULT_RUN_ARGS.copy()
        run_args["working_directory"] = args["working_directory"]
        run_args["config"] = parse_config(CONFIG_DEFAULT)
        run_args["config"]["general"]["start_time"] = run_args["config"]["general"][
            "spinup_time"
        ] + timedelta(days=370)  # run just over a year more is not needed

        run_model_with_method(method="spinup", **run_args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_land_use_change() -> None:
    """Test land use change functionality.

    Verifies that land use changes can be applied to the model
    and that the model can continue running after the land use changes.
    """
    with WorkingDirectory(working_directory):
        args = DEFAULT_RUN_ARGS.copy()
        config = parse_config(CONFIG_DEFAULT)
        config["hazards"]["floods"]["simulate"] = False  # disable flood simulation
        config["general"]["end_time"] = config["general"]["start_time"] + timedelta(
            days=370
        )
        args["config"] = config

        geb = run_model_with_method(method=None, close_after_run=False, **args)
        geb.run(initialize_only=True)

        current_grassland = geb.hydrology.HRU.var.land_use_type == GRASSLAND_LIKE

        new_forest = np.arange(current_grassland.sum()) % 2 == 0  # every second cell

        new_forest_mask = np.zeros_like(current_grassland, dtype=bool)
        new_forest_mask[current_grassland] = new_forest

        # get the farmers corresponding to the new forest HRUs
        farmers_with_land_converted_to_forest = np.unique(
            geb.hydrology.HRU.var.land_owners[new_forest_mask]
        )
        farmers_with_land_converted_to_forest = (farmers_with_land_converted_to_forest)[
            farmers_with_land_converted_to_forest != -1
        ]

        HRUs_removed_farmers = geb.agents.crop_farmers.remove_agents(
            farmers_with_land_converted_to_forest, new_land_use_type=FOREST
        )

        new_forest_mask[HRUs_removed_farmers] = True

        geb.step_to_end()
        geb.reporter.finalize()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_custom_DEM() -> None:
    """Test model build with a custom DEM."""
    with WorkingDirectory(working_directory):
        build_args = DEFAULT_BUILD_ARGS.copy()
        del build_args["continue_"]

        build_config: dict[str, dict[str, list[dict]]] = {}
        build_config["setup_elevation"] = {
            "DEMs": [{"name": "geul_dem", "nodata": np.nan}]
        }
        build_args["build_config"] = build_config

        # Path is not set, so should raise an error
        with pytest.raises(ValueError):
            update_fn(**build_args)

        # Test with zarr
        build_config["setup_elevation"]["DEMs"][0]["path"] = str(
            Path("data") / "geul_dem.zarr"
        )

        # Test setting CRS
        build_config["setup_elevation"]["DEMs"][0]["crs"] = 28992
        update_fn(**build_args)

        # Test with tif
        build_config["setup_elevation"]["DEMs"][0]["path"] = str(
            Path("data") / "Geul_Filled_DEM_EPSG28992.tif"
        )

        update_fn(**build_args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_setup_reforestation_potential() -> None:
    """Test setup of forest restoration potential.

    Verifies that the model can run the
    ``setup_forest_restoration_potential`` build step with a basic
    configuration without raising errors.
    """
    with WorkingDirectory(working_directory):
        build_args = DEFAULT_BUILD_ARGS.copy()
        del build_args["continue_"]

        build_config: dict[str, dict[str, str | bool]] = {}
        build_config["setup_forest_restoration_potential"] = {}
        build_args["build_config"] = build_config

        update_fn(**build_args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_setup_inflow() -> None:
    """Test setup of inflow hydrograph.

    Verifies that the model can set up an inflow hydrograph
    from specified locations and inflow data files.
    """
    with WorkingDirectory(working_directory):
        args = DEFAULT_RUN_ARGS.copy()
        config = parse_config(CONFIG_DEFAULT)
        config["hazards"]["floods"]["simulate"] = False  # disable flood simulation
        config["general"]["end_time"] = config["general"]["start_time"] + timedelta(
            days=10
        )
        args["config"] = config

        model: GEBModel = run_model_with_method(
            method=None,
            close_after_run=False,
            **args,
        )
        model.run(initialize_only=True)
        data_folder: Path = Path("data")
        data_folder.mkdir(parents=True, exist_ok=True)

        rivers: gpd.GeoDataFrame = model.hydrology.routing.active_rivers.copy()

        start_time = model.spinup_start
        end_time = model.run_end + model.timestep_length

        model.close()

        rivers: gpd.GeoDataFrame = rivers[["geometry"]]
        rivers.geometry = rivers.geometry.apply(lambda geom: Point(geom.coords[0]))
        rivers.index.name = "ID"
        rivers.index = rivers.index.astype(str)
        rivers.reset_index().to_file(
            data_folder / "inflow_locations.geojson", driver="GeoJSON"
        )

        data = {
            "time": pd.date_range(start=start_time, end=end_time, freq="YS"),
        }
        for ID in rivers.index:
            data[ID] = np.round(np.random.rand(len(data["time"])) * 100, 1)

        inflow = pd.DataFrame(data)
        inflow.to_csv(data_folder / "inflow_hydrograph.csv", index=False)

        try:
            build_args = DEFAULT_BUILD_ARGS.copy()
            del build_args["continue_"]

            build_config: dict[str, dict[str, str | bool]] = {}
            build_config["setup_inflow"] = {
                "locations": str(Path("data") / "inflow_locations.geojson"),
                "inflow_m3_per_s": str(Path("data") / "inflow_hydrograph.csv"),
            }
            build_args["build_config"] = build_config
            with pytest.raises(ValueError):
                update_fn(**build_args)

            build_args["build_config"]["setup_inflow"]["interpolate"] = True
            build_args["build_config"]["setup_inflow"]["extrapolate"] = True

            # copy the original input file
            shutil.copy(Path("input") / "files.yml", Path("input") / "files.yml.bak")

            update_fn(**build_args)
            run_model_with_method(method="run", **args)

        finally:
            # remove inflow data files
            if (data_folder / "inflow_hydrograph.csv").exists():
                os.remove(data_folder / "inflow_hydrograph.csv")
            if (data_folder / "inflow_locations.geojson").exists():
                os.remove(data_folder / "inflow_locations.geojson")

            # restore original input file
            if (Path("input") / "files.yml.bak").exists():
                shutil.copy(
                    Path("input") / "files.yml.bak", Path("input") / "files.yml"
                )
                os.remove(Path("input") / "files.yml.bak")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_setup_retention_basins() -> None:
    """Test setup of retention basins.

    Verifies that the model can set up retention basins
    from specified locations and retention basin data files.
    """
    with WorkingDirectory(working_directory):
        args = DEFAULT_RUN_ARGS.copy()
        config = parse_config(CONFIG_DEFAULT)
        config["hazards"]["floods"]["simulate"] = False  # disable flood simulation
        config["general"]["end_time"] = config["general"]["start_time"] + timedelta(
            days=10
        )
        args["config"] = config

        model: GEBModel = run_model_with_method(
            method=None,
            close_after_run=False,
            **args,
        )
        model.run(initialize_only=True)
        data_folder: Path = Path("data")
        data_folder.mkdir(parents=True, exist_ok=True)

        rivers: gpd.GeoDataFrame = model.hydrology.routing.active_rivers.copy()

        start_time = model.spinup_start
        end_time = model.run_end + model.timestep_length

        model.close()

        retention_basins: gpd.GeoDataFrame = rivers[["geometry"]].copy().to_crs(3857)
        retention_basins.geometry = retention_basins.geometry.centroid
        retention_basins = retention_basins.to_crs(4326)
        retention_basins["ID"] = np.arange(len(retention_basins))
        retention_basins["is_controlled"] = retention_basins.index.astype(int) % 2 == 0
        retention_basins.reset_index().to_file(
            data_folder / "retention_basins.geojson", driver="GeoJSON"
        )

        try:
            build_args = DEFAULT_BUILD_ARGS.copy()
            del build_args["continue_"]

            build_config: dict[str, dict[str, str | bool]] = {}
            build_config["setup_retention_basins"] = {
                "retention_basins": str(Path("data") / "retention_basins.geojson"),
            }
            build_args["build_config"] = build_config

            # copy the original input file
            shutil.copy(Path("input") / "files.yml", Path("input") / "files.yml.bak")

            update_fn(**build_args)
            geb_model = run_model_with_method(
                method="run", **args, close_after_run=False
            )
            assert geb_model.hydrology.routing.retention_basin_data is not None
            assert geb_model.hydrology.routing.retention_basin_ids is not None
            geb_model.close()

        finally:
            # remove retention basin data files
            if (data_folder / "retention_basins.geojson").exists():
                os.remove(data_folder / "retention_basins.geojson")

            # restore original input file
            if (Path("input") / "files.yml.bak").exists():
                shutil.copy(
                    Path("input") / "files.yml.bak", Path("input") / "files.yml"
                )
                os.remove(Path("input") / "files.yml.bak")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_run_yearly() -> None:
    """Test yearly model execution.

    Verifies that the model can run simulations on a yearly
    time step.
    """
    with WorkingDirectory(working_directory):
        args = DEFAULT_RUN_ARGS.copy()
        config = parse_config(CONFIG_DEFAULT)
        config["general"]["end_time"] = date(2049, 12, 31)
        config["hazards"]["floods"]["simulate"] = True  # enable flood simulation

        args["config"] = config
        args["config"]["report"] = {}

        with pytest.raises(
            ValueError,
            match="Yearly mode is not compatible with flood simulation. Please set 'simulate' to False in the config.",
        ):
            run_model_with_method(method="run_yearly", **args)

        config["hazards"]["floods"]["simulate"] = False  # disable flood simulation

        run_model_with_method(method="run_yearly", **args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_estimate_return_periods() -> None:
    """Test return period estimation.

    Verifies that the model can estimate return periods for
    extreme events and flood frequencies.
    """
    with WorkingDirectory(working_directory):
        run_model_with_method(method="estimate_return_periods", **DEFAULT_RUN_ARGS)

    if os.getenv("GEB_TEST_GPU", "no") == "yes":
        flood_maps_CPU: Path = working_directory / "output" / "flood_maps" / "1000.zarr"

        # move the flood maps to a separate folder
        flood_maps_folder_CPU: Path = tmp_folder / "flood_maps" / "CPU"
        flood_maps_folder_CPU.mkdir(parents=True, exist_ok=True)
        flood_maps_CPU.rename(flood_maps_folder_CPU / "1000.zarr")

        with WorkingDirectory(working_directory):
            args = DEFAULT_RUN_ARGS.copy()
            args["config"] = parse_config(CONFIG_DEFAULT)
            args["config"]["hazards"]["floods"]["SFINCS"]["gpu"] = True
            run_model_with_method(method="estimate_return_periods", **args)

        flood_maps_folder_GPU: Path = tmp_folder / "flood_maps" / "GPU"
        flood_maps_folder_GPU.mkdir(parents=True, exist_ok=True)
        flood_maps_GPU: Path = working_directory / "output" / "flood_maps" / "1000.zarr"
        flood_maps_GPU.rename(flood_maps_folder_GPU / "1000.zarr")

        # compare the flood maps
        flood_map_CPU: xr.DataArray = xr.open_dataarray(
            flood_maps_folder_CPU / "1000.zarr"
        )
        flood_map_GPU: xr.DataArray = xr.open_dataarray(
            flood_maps_folder_GPU / "1000.zarr"
        )

        flood_map_CPU = flood_map_CPU.fillna(0)
        flood_map_GPU = flood_map_GPU.fillna(0)

        np.testing.assert_almost_equal(
            flood_map_CPU.values,
            flood_map_GPU.values,
            decimal=1,
            err_msg="Flood maps for the 1000-year return period do not match between CPU and GPU runs.",
        )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_multiverse() -> None:
    """Test the multiverse functionality with flood events and forecast forcing."""
    with WorkingDirectory(working_directory):
        args = DEFAULT_RUN_ARGS.copy()

        config: dict[str, Any] = parse_config(CONFIG_DEFAULT)
        config["hazards"]["floods"]["simulate"] = True

        forecast_after_n_days: int = 3
        forecast_n_days: int = 5
        forecast_n_hours: int = 13

        forecast_issue_date: datetime = (
            datetime.combine(date=config["general"]["start_time"], time=time.min)
            + timedelta(days=forecast_after_n_days)
            + timedelta(hours=forecast_n_hours)
        )
        forecast_end_date = forecast_issue_date + timedelta(
            days=int(forecast_n_days), hours=forecast_n_hours
        )
        config["general"]["end_time"] = forecast_issue_date + timedelta(
            days=int(forecast_n_days) + 5
        )

        # inititate a forecast after three days
        config["general"]["forecasts"]["times"] = [forecast_issue_date]
        config["general"]["forecasts"]["provider"] = "test"

        # add flood event during the forecast period
        events: list = [
            {
                "start_time": forecast_issue_date + timedelta(days=1),
                "end_time": forecast_issue_date + timedelta(days=2, hours=12),
            },
            {
                "start_time": forecast_end_date - timedelta(days=1, hours=0),
                "end_time": forecast_end_date + timedelta(days=1, hours=18),
            },
        ]
        config["hazards"]["floods"]["events"] = events

        args["config"] = config

        geb: GEBModel = run_model_with_method(
            method=None, close_after_run=False, **args
        )
        geb.run(initialize_only=True)
        for i in range(forecast_after_n_days):
            geb.step()

        hashes: dict[str, dict[str, int]] = {}
        for bucket_name, bucket in geb.store.buckets.items():
            hashes[bucket_name] = {}
            for var_name, var in vars(bucket).items():
                if isinstance(var, np.ndarray):
                    hashes[bucket_name][var_name] = hash(var.tobytes())

        # setup forecast data
        for forecast_variable, loader in geb.forcing.loaders.items():
            if not loader.supports_forecast:
                continue
            da: xr.DataArray = read_zarr(
                geb.files["other"][f"climate/{forecast_variable}"]
            ).drop_encoding()
            forecast_da: xr.DataArray = da.sel(
                time=slice(
                    forecast_issue_date,
                    forecast_end_date,
                )
            ).chunk({"time": -1})

            # add member dimension
            forecast_da: xr.DataArray = forecast_da.expand_dims(
                dim={"member": [0, 1]}, axis=0
            )

            forecasts_folder: Path = (
                geb.input_folder
                / "other"
                / "forecasts"
                / config["general"]["forecasts"]["provider"]
            )
            forecasts_folder.mkdir(parents=True, exist_ok=True)

            write_zarr(
                forecast_da,
                forecasts_folder
                / (
                    forecast_variable
                    + "_"
                    + forecast_issue_date.strftime("%Y%m%dT%H%M%S.zarr")
                ),
                crs=forecast_da.rio.crs,
            )

        mean_discharge_after_forecast: dict[str | int, float] = geb.multiverse(
            return_mean_discharge=True, forecast_issue_datetime=forecast_issue_date
        )

        for bucket_name, bucket in geb.store.buckets.items():
            for var_name, var in vars(bucket).items():
                if isinstance(var, np.ndarray):
                    assert hash(var.tobytes()) == hashes[bucket_name][var_name], (
                        f"Bucket {bucket_name} variable {var_name} has changed after multiverse run."
                    )

        end_date = pd.to_datetime(forecast_da.time[-1].item()).to_pydatetime().date()
        steps_in_forecast = (end_date - geb.current_time.date()).days

        for i in range(steps_in_forecast):
            geb.step()

        mean_discharge: float = (
            geb.hydrology.routing.grid.var.discharge_m3_s.mean().item()
        )

        geb.step_to_end()

        for member, forecast_mean_discharge in mean_discharge_after_forecast.items():
            assert forecast_mean_discharge == mean_discharge

        geb.close()

        flood_map_folder: Path = Path("output") / "flood_maps"
        forecast_folder: Path = (
            flood_map_folder
            / f"forecast_{forecast_issue_date.strftime('%Y%m%dT%H%M%S')}"
            / "member_0"
        )

        # the first flood event in the multiverse is of equal length as in the main simulation
        # so the flood maps should be identical
        flood_map_first_event: xr.DataArray = read_zarr(
            flood_map_folder
            / f"{events[0]['start_time'].strftime('%Y%m%dT%H%M%S')} - {events[0]['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
        )

        flood_map_first_event_multiverse: xr.DataArray = read_zarr(
            forecast_folder
            / f"{events[0]['start_time'].strftime('%Y%m%dT%H%M%S')} - {events[0]['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
        )

        np.testing.assert_array_equal(
            actual=flood_map_first_event.values,
            desired=flood_map_first_event_multiverse.values,
            err_msg="Flood maps for the first event do not match across multiverse members.",
        )

        # the second flood event in the multiverse is shorter than in the main simulation
        # because the forecast ends before the flood event ends
        flood_map_second_event: xr.DataArray = read_zarr(
            flood_map_folder
            / f"{events[1]['start_time'].strftime('%Y%m%dT%H%M%S')} - {events[1]['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
        )

        # the name of the file is midnight before (or on) the end time of the forecast
        flood_map_second_event_multiverse: xr.DataArray = read_zarr(
            forecast_folder
            / f"{events[1]['start_time'].strftime('%Y%m%dT%H%M%S')} - {forecast_end_date.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y%m%dT%H%M%S')}.zarr"
        )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_share() -> None:
    """Test model sharing functionality.

    Verfifies that the zip file created by the share function
    exists and can be created without errors.
    """
    with WorkingDirectory(working_directory):
        share_fn(
            working_directory=Path("."),
            name="test",
            include_cache=False,
            include_output=False,
        )

        output_fn: Path = Path("test.zip")

        assert output_fn.exists()

        output_fn.unlink()


def test_set_and_clean(tmp_path: Path) -> None:
    """Test setting config values and cleaning a copied model folder with the CLI.

    Initializes a model in a temporary source directory, creates representative
    generated files, copies the full model folder to a second temporary
    directory, updates multiple config values in the copied model with
    ``geb set``, and runs ``geb clean`` on the copy. This verifies that the
    copied config is updated as expected, that only the generated files are
    removed from the copied model, and that the source model remains unchanged.

    Args:
        tmp_path: Temporary test directory provided by pytest.
    """
    source_model_directory: Path = tmp_path / "model_source"
    copied_model_directory: Path = tmp_path / "model_copy"
    updated_spinup_time: date = date(1980, 1, 1)
    updated_start_time: date = date(1981, 1, 1)
    updated_end_time: date = date(1981, 12, 31)

    source_model_directory.mkdir(parents=True, exist_ok=True)

    with WorkingDirectory(source_model_directory):
        init_fn(
            config="model.yml",
            build_config="build.yml",
            update_config="update.yml",
            working_directory=Path("."),
            from_example="geul",
            basin_id="23011134",
            overwrite=True,
        )

        generated_paths: list[Path] = [
            Path("cache"),
            Path("input"),
            Path("output"),
            Path("logs"),
        ]
        for generated_path in generated_paths:
            generated_path.mkdir(parents=True, exist_ok=True)
            (generated_path / "placeholder.txt").write_text("generated data")

    shutil.copytree(source_model_directory, copied_model_directory)

    runner = CliRunner()
    set_result = runner.invoke(
        cli,
        [
            "set",
            "--config",
            "model.yml",
            "--working-directory",
            str(copied_model_directory),
            f"general.spinup_time={updated_spinup_time.isoformat()}",
            f"general.start_time={updated_start_time.isoformat()}",
            f"general.end_time={updated_end_time.isoformat()}",
            "hazards.floods.simulate=false",
            "report=null",
            "report._discharge_stations+=true",
        ],
    )

    assert set_result.exit_code == 0, set_result.output

    copied_config: dict[str, Any] = parse_config(copied_model_directory / "model.yml")
    assert copied_config["general"]["spinup_time"] == updated_spinup_time
    assert copied_config["general"]["start_time"] == updated_start_time
    assert copied_config["general"]["end_time"] == updated_end_time
    assert copied_config["hazards"]["floods"]["simulate"] is False
    assert copied_config["report"] == {"_discharge_stations": True}

    source_config: dict[str, Any] = parse_config(source_model_directory / "model.yml")
    assert source_config["general"]["spinup_time"] != updated_spinup_time
    assert source_config["general"]["start_time"] != updated_start_time
    assert source_config["general"]["end_time"] != updated_end_time

    clean_result = runner.invoke(
        cli,
        [
            "clean",
            "--yes",
            "--working-directory",
            str(copied_model_directory),
        ],
    )

    assert clean_result.exit_code == 0, clean_result.output

    expected_remaining_files: set[str] = {"build.yml", "model.yml", "update.yml"}
    remaining_files: set[str] = {
        path.name for path in copied_model_directory.iterdir() if path.exists()
    }
    assert remaining_files == expected_remaining_files

    cleaned_config: dict[str, Any] = parse_config(copied_model_directory / "model.yml")
    assert cleaned_config["general"]["spinup_time"] == updated_spinup_time
    assert cleaned_config["general"]["start_time"] == updated_start_time
    assert cleaned_config["general"]["end_time"] == updated_end_time
    assert cleaned_config["hazards"]["floods"]["simulate"] is False
    assert cleaned_config["report"] == {"_discharge_stations": True}

    for generated_directory_name in ("cache", "input", "logs", "output"):
        assert not (copied_model_directory / generated_directory_name).exists()
        assert (source_model_directory / generated_directory_name).exists()
