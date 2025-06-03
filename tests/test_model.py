import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest
import xarray as xr

from geb.cli import build_fn, parse_config, run_model_with_method, update_fn

from .testconfig import IN_GITHUB_ACTIONS, tmp_folder

example = Path("../../../examples/geul")


working_directory = tmp_folder / "model"
working_directory.mkdir(parents=True, exist_ok=True)

DEFAULT_BUILD_ARGS = {
    "data_catalog": [Path("../../../geb/data_catalog.yml")],
    "config": str(example / "model.yml"),
    "build_config": str(example / "build.yml"),
    "working_directory": working_directory,
    "custom_model": None,
    "data_provider": None,
    "data_root": str(Path(os.getenv("GEB_DATA_ROOT", ""))),
}

DEFAULT_RUN_ARGS = {
    "config": str(example / "model.yml"),
    "working_directory": working_directory,
    "gui": False,
    "no_browser": True,
    "port": None,
    "profiling": False,
    "timing": False,
    "optimize": False,
}


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
@pytest.mark.dependency(name="test_build")
def test_spinup():
    run_model_with_method(method="spinup", **DEFAULT_RUN_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.dependency(depends=["test_spinup"])
def test_run():
    run_model_with_method(method="run", **DEFAULT_RUN_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.dependency(depends=["test_spinup"])
def test_run_yearly():
    args = DEFAULT_RUN_ARGS.copy()
    config = parse_config(working_directory / args["config"])
    config["general"]["start_time"] = date(2000, 1, 1)
    config["general"]["start_time"] = date(2050, 1, 1)
    args["config"] = config
    args["config"]["report"] = {}
    run_model_with_method(method="run_yearly", **args)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.dependency(depends=["test_spinup"])
def test_estimate_return_periods():
    run_model_with_method(method="estimate_return_periods", **DEFAULT_RUN_ARGS)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.dependency(depends=["test_spinup"])
def test_multiverse():
    args = DEFAULT_RUN_ARGS.copy()

    config = parse_config(working_directory / args["config"])
    config["general"]["forecasts"]["use"] = True

    forecast_date = config["general"]["start_time"] + timedelta(days=3)
    config["general"]["end_time"] = forecast_date + timedelta(days=5)

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

    geb = run_model_with_method(method=None, close_after_run=False, **args)
    geb.run()
    geb.close()
