from pathlib import Path

import pytest

from geb.cli import run_model_with_method

from ..testconfig import IN_GITHUB_ACTIONS

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
@pytest.mark.dependency(name="test_build")
def test_spinup():
    run_model_with_method(method="spinup", **DEFAULT_VARIABLES)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.dependency(depends=["test_spinup"])
def test_run():
    run_model_with_method(method="run", **DEFAULT_VARIABLES)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.dependency(depends=["test_spinup"])
def test_run_yearly():
    run_model_with_method(method="run_yearly", **DEFAULT_VARIABLES)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
@pytest.mark.dependency(depends=["test_spinup"])
def test_estimate_return_periods():
    run_model_with_method(method="estimate_return_periods", **DEFAULT_VARIABLES)
