import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geb.cli import (
    alter_fn,
    build_fn,
    init_fn,
    parse_config,
    run_model_with_method,
    share_fn,
    update_fn,
)
from geb.model import GEBModel
from geb.workflows.dt import round_up_to_start_of_next_day_unless_midnight
from geb.workflows.io import WorkingDirectory

from .testconfig import IN_GITHUB_ACTIONS, tmp_folder

working_directory: Path = tmp_folder / "model"

DEFAULT_BUILD_ARGS: dict[str, Any] = {
    "data_catalog": [Path(os.getenv("GEB_PACKAGE_DIR")) / "data_catalog.yml"],
    "config": "model.yml",
    "build_config": "build.yml",
    "working_directory": working_directory,
    "custom_model": None,
    "data_provider": None,
    "data_root": str(Path(os.getenv("GEB_DATA_ROOT", ""))),
}

DEFAULT_RUN_ARGS: dict[str, Any] = {
    "config": "model.yml",
    "working_directory": working_directory,
    "gui": False,
    "no_browser": True,
    "port": None,
    "profiling": False,
    "timing": False,
    "optimize": False,
}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_init():
    working_directory.mkdir(parents=True, exist_ok=True)

    args: dict[str, Any] = {
        "config": "model.yml",
        "build_config": "build.yml",
        "update_config": "update.yml",
        "working_directory": working_directory,
        "from_example": "geul",
        "basin_id": "23011134",
    }
    init_fn(
        **args,
        overwrite=True,
    )

    assert (working_directory / "model.yml").exists()
    assert (working_directory / "build.yml").exists()
    assert (working_directory / "update.yml").exists()

    assert pytest.raises(
        FileExistsError,
        init_fn,
        **args,
        overwrite=False,
    )  # should raise an error if the folder already exists


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_build():
    build_fn(**DEFAULT_BUILD_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_forcing():
    with WorkingDirectory(working_directory):
        model: GEBModel = run_model_with_method(
            method=None,
            close_after_run=False,
            **{**DEFAULT_RUN_ARGS, **{"working_directory": "."}},
        )

        for name in model.forcing.validators:
            t_0: datetime = datetime(2010, 1, 1, 0, 0, 0)
            forcing_0 = model.forcing.load(name, t_0)

            t_1: datetime = datetime(2020, 1, 1, 0, 0, 0)
            forcing_1 = model.forcing.load(name, t_1)

            if isinstance(forcing_0, (xr.DataArray, np.ndarray)):
                assert forcing_0.shape == forcing_1.shape, (
                    f"Shape of forcing data for {name} does not match for times {t_0} and {t_1}."
                )
                # non-precipitation forcing basically could never be equal, so we check for inequality
                if name not in ("pr", "pr_hourly"):
                    assert not np.array_equal(forcing_0, forcing_1)
            else:
                assert forcing_0 != forcing_1


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_alter():
    args: dict[str, Any] = DEFAULT_BUILD_ARGS.copy()
    args["build_config"] = {"set_ssp": {"ssp": "ssp1"}, "setup_CO2_concentration": {}}
    args["working_directory"] = working_directory / "alter"

    args["from_model"] = ".."

    args["working_directory"].mkdir(parents=True, exist_ok=True)

    alter_fn(**args)

    run_args = DEFAULT_RUN_ARGS.copy()
    run_args["working_directory"] = args["working_directory"]

    run_model_with_method(method="spinup", **run_args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_update_with_file():
    args = DEFAULT_BUILD_ARGS.copy()
    args["build_config"] = "update.yml"
    update_fn(**args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_update_with_dict():
    args = DEFAULT_BUILD_ARGS.copy()
    update = {"setup_land_use_parameters": {}}
    args["build_config"] = update
    update_fn(**args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.parametrize(
    "method",
    [
        "setup_crop_prices",
        "setup_discharge_observations",
        "setup_forcing_era5",
        "setup_water_demand",
        "setup_SPEI",
        "setup_CO2_concentration",
    ],
)
def test_update_with_method(method: str):
    args: dict[str, str | dict | Path | bool] = DEFAULT_BUILD_ARGS.copy()

    build_config: dict[str, dict] = parse_config(
        working_directory / args["build_config"]
    )

    update: dict[str, dict] = {method: build_config[method]}

    args["build_config"] = update
    update_fn(**args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_spinup():
    run_model_with_method(method="spinup", **DEFAULT_RUN_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_evaluate():
    run_model_with_method(method="evaluate", **DEFAULT_RUN_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_run():
    run_model_with_method(method="run", **DEFAULT_RUN_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_run_yearly():
    args = DEFAULT_RUN_ARGS.copy()
    config = parse_config(working_directory / args["config"])
    config["general"]["start_time"] = date(2000, 1, 1)
    config["general"]["end_time"] = date(2049, 12, 31)
    args["config"] = config
    args["config"]["report"] = {}
    assert pytest.raises(
        AssertionError,
        run_model_with_method,
        method="run_yearly",
        **args,
    )

    config["hazards"]["floods"]["simulate"] = False  # disable flood simulation

    run_model_with_method(method="run_yearly", **args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_estimate_return_periods():
    run_model_with_method(method="estimate_return_periods", **DEFAULT_RUN_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_multiverse():
    args = DEFAULT_RUN_ARGS.copy()

    config = parse_config(working_directory / args["config"])

    forecast_after_n_days = 3
    forecast_n_days = 5
    forecast_n_hours = 13

    forecast_date = (
        datetime.combine(config["general"]["start_time"], time.min)
        + timedelta(days=forecast_after_n_days)
        + timedelta(hours=forecast_n_hours)
    )
    forecast_end_date = forecast_date + timedelta(
        days=int(forecast_n_days), hours=forecast_n_hours
    )
    config["general"]["end_time"] = forecast_date + timedelta(
        days=int(forecast_n_days) + 5
    )

    input_folder = working_directory / config["general"]["input_folder"]

    files = input_folder / "files.json"
    files = json.loads(files.read_text())

    precipitation = xr.open_dataarray(
        input_folder / files["other"]["climate/pr_hourly"], consolidated=False
    ).drop_encoding()
    forecast = precipitation.sel(
        time=slice(
            forecast_date,
            forecast_end_date,
        )
    )

    # add member dimension
    forecast = forecast.expand_dims(dim={"member": [0]}, axis=0)

    forecasts_folder: Path = working_directory / "data" / "forecasts"
    forecasts_folder.mkdir(parents=True, exist_ok=True)

    forecast.to_zarr(
        forecasts_folder / forecast_date.strftime("%Y%m%dT%H%M%S.zarr"), mode="w"
    )

    # inititate a forecast after three days
    config["general"]["forecasts"]["times"] = [forecast_date]

    # add flood event during the forecast period
    events: list = [
        {
            "start_time": forecast_date + timedelta(days=1),
            "end_time": forecast_date + timedelta(days=2, hours=12),
        },
        {
            "start_time": forecast_end_date - timedelta(days=1, hours=0),
            "end_time": forecast_end_date + timedelta(days=1, hours=18),
        },
    ]
    config["hazards"]["floods"]["events"] = events
    config["hazards"]["floods"]["force_overwrite"] = False

    args["config"] = config

    geb: GEBModel = run_model_with_method(method=None, close_after_run=False, **args)
    with WorkingDirectory(working_directory):
        geb.run(initialize_only=True)
        for i in range(forecast_after_n_days):
            geb.step()

        mean_discharge_after_forecast: dict[Any, float] = geb.multiverse(
            return_mean_discharge=True, forecast_dt=forecast_date
        )

        end_date = round_up_to_start_of_next_day_unless_midnight(
            pd.to_datetime(forecast.time[-1].item()).to_pydatetime()
        ).date()
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

    flood_map_folder: Path = working_directory / "output" / "flood_maps"

    flood_map_first_event: xr.DataArray = xr.open_dataarray(
        flood_map_folder
        / f"{events[0]['start_time'].strftime('%Y%m%dT%H%M%S')} - {events[0]['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
    )

    flood_map_first_event_multiverse: xr.DataArray = xr.open_dataarray(
        flood_map_folder
        / f"0 - {events[0]['start_time'].strftime('%Y%m%dT%H%M%S')} - {events[0]['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
    )

    np.testing.assert_array_equal(
        flood_map_first_event.values,
        flood_map_first_event_multiverse.values,
        err_msg="Flood maps for the first event do not match across multiverse members.",
    )

    flood_map_second_event: xr.DataArray = xr.open_dataarray(
        flood_map_folder
        / f"{events[1]['start_time'].strftime('%Y%m%dT%H%M%S')} - {events[1]['end_time'].strftime('%Y%m%dT%H%M%S')}.zarr"
    )

    flood_map_second_event_multiverse: xr.DataArray = xr.open_dataarray(
        flood_map_folder
        / f"0 - {events[1]['start_time'].strftime('%Y%m%dT%H%M%S')} - {forecast_end_date.strftime('%Y%m%dT%H%M%S')}.zarr"
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_ISIMIP_forcing_low_res():
    """Test the ISIMIP forcing update function.

    This is a special case that requires a specific setup.
    """
    args: dict[str, Any] = DEFAULT_BUILD_ARGS.copy()

    build_config: dict[str, Any] = parse_config(
        working_directory / args["build_config"]
    )

    original_time_range: dict[str, date] = build_config["set_time_range"]

    args["build_config"] = {
        "set_time_range": {
            "start_date": date(2001, 1, 1),
            "end_date": date(2024, 12, 31),
        },
        "setup_forcing_ISIMIP": {
            "resolution_arcsec": 1800,
            "forcing": "gfdl-esm4",
        },
    }
    update_fn(**args)

    # Reset the time range to the original one
    args["build_config"] = {
        "set_time_range": original_time_range,
    }

    update_fn(**args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_share():
    share_fn(
        working_directory=working_directory,
        name="test",
        include_preprocessing=False,
        include_output=False,
    )

    output_fn: Path = working_directory / "test.zip"

    assert output_fn.exists()

    output_fn.unlink()
