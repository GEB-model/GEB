from pathlib import Path
from shutil import rmtree

import pytest

from geb.cli import build_fn, run_model_with_method

from ..testconfig import IN_GITHUB_ACTIONS, tmp_folder

working_directory = Path("examples/geul")
DEFAULT_VARIABLES = {
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
# @pytest.mark.dependency(name="build")
def test_build():
    working_directory = tmp_folder / "model"
    working_directory.mkdir(parents=True, exist_ok=True)

    build_fn(
        data_catalog=[Path("../../../geb/data_catalog.yml")],
        config=Path("../../../examples/geul/model.yml"),
        build_config=Path("../../../examples/geul/build.yml"),
        working_directory=working_directory,
        custom_model=None,
        data_provider=None,
    )

    rmtree(working_directory, ignore_errors=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
# @pytest.mark.dependency(name="spinup")
def test_spinup():
    run_model_with_method(method="spinup", **DEFAULT_VARIABLES)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
# @pytest.mark.dependency(depends=["spinup"])
def test_run():
    run_model_with_method(method="run", **DEFAULT_VARIABLES)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
# @pytest.mark.dependency(depends=["spinup"])
def test_run_yearly():
    run_model_with_method(method="run_yearly", **DEFAULT_VARIABLES)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
# @pytest.mark.dependency(depends=["spinup"])
def test_estimate_risk():
    run_model_with_method(method="estimate_risk", **DEFAULT_VARIABLES)
