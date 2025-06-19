import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import xarray as xr

from geb.cli import build_fn, init_fn, parse_config, run_model_with_method, update_fn
from geb.workflows.io import WorkingDirectory

from .testconfig import IN_GITHUB_ACTIONS, tmp_folder

example = Path("../../../examples/geul")


working_directory: Path = tmp_folder / "model"

DEFAULT_BUILD_ARGS = {
    "data_catalog": [Path("../../../geb/data_catalog.yml")],
    "config": "model.yml",
    "build_config": "build.yml",
    "working_directory": working_directory,
    "custom_model": None,
    "data_provider": None,
    "data_root": str(Path(os.getenv("GEB_DATA_ROOT", ""))),
}

DEFAULT_RUN_ARGS = {
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
        "working_directory": working_directory,
        "from_example": "geul",
        "basin_id": "23011134",
    }
    init_fn(
        **args,
        overwrite=True,
    )

    pytest.raises(
        FileExistsError,
        init_fn,
        **args,
        overwrite=False,
    )  # should raise an error if the folder already exists


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_build():
    build_fn(**DEFAULT_BUILD_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_update_with_file():
    args = DEFAULT_BUILD_ARGS.copy()
    args["build_config"] = example / "update.yml"
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

    forecast_date = config["general"]["start_time"] + timedelta(
        days=forecast_after_n_days
    )
    config["general"]["end_time"] = forecast_date + timedelta(days=forecast_n_days)

    input_folder = working_directory / config["general"]["input_folder"]

    files = input_folder / "files.json"
    files = json.loads(files.read_text())

    precipitation = xr.open_dataarray(
        input_folder / files["other"]["climate/pr"]
    ).drop_encoding()
    precipitation = precipitation.sel(
        time=slice(forecast_date, forecast_date + timedelta(days=5))
    )

    # add member dimension
    precipitation = precipitation.expand_dims(dim={"member": [0]}, axis=0)

    forecasts_folder = input_folder / "other" / "climate" / "forecasts"
    forecasts_folder.mkdir(parents=True, exist_ok=True)

    precipitation.to_zarr(
        forecasts_folder / forecast_date.strftime("%Y%m%d.zarr"), mode="w"
    )

    # inititate a forecast after three days
    config["general"]["forecasts"]["days"] = [forecast_date]

    args["config"] = config

    geb = run_model_with_method(method=None, close_after_run=False, gui=False, **args)
    with WorkingDirectory(working_directory):
        geb.run(initialize_only=True)
        for i in range(forecast_after_n_days):
            geb.step()
        mean_discharge_after_forecast = geb.multiverse(return_mean_discharge=True)

        for i in range(forecast_n_days):
            geb.step()

        mean_discharge = geb.hydrology.routing.grid.var.discharge_m3_s.mean().item()

    for member, forecast_mean_discharge in mean_discharge_after_forecast.items():
        assert forecast_mean_discharge == mean_discharge

    geb.close()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_ISIMIP_forcing_low_res():
    """
    Test the ISIMIP forcing update function.
    This is a special case that requires a specific setup.
    """
    args = DEFAULT_BUILD_ARGS.copy()

    build_config = parse_config(working_directory / args["build_config"])

    original_time_range = build_config["set_time_range"]

    args["build_config"] = {
        "set_time_range": {
            "start_date": date(2000, 1, 1),
            "end_date": date(2000, 12, 31),
        },
        "setup_forcing_ISIMIP": {
            "resolution_arcsec": 1800,
            "forcing": "gfdl-esm4",
            "ssp": "ssp370",
        },
    }
    update_fn(**args)

    # Reset the time range to the original one
    args["build_config"] = {
        "set_time_range": original_time_range,
    }

    update_fn(**args)
