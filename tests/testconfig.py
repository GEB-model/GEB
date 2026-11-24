"""Configuration for tests.

A temporary folder is created in the tests directory of the GEB package for the test output.

IN_GITHUB_ACTIONS is a flag that is used to turn on or off test cases in the Github Actions environment.
This is useful for tests that are too memory or data intensive to run in the limited Github Actions environment.

"""

import os
from pathlib import Path

from geb import GEB_PACKAGE_DIR

GEB_TEST_DIR: Path = GEB_PACKAGE_DIR.parent / "tests"

# Ensure the testing files end in up in the tests folder of the GEB package directory
output_folder: Path = GEB_TEST_DIR / "output"
output_folder.mkdir(exist_ok=True)

tmp_folder: Path = GEB_TEST_DIR / "tmp"
tmp_folder.mkdir(exist_ok=True)


# The shared test model, built and spun up by tests/test_model.py (test_init, test_build,
# test_spinup). Tests that need it (e.g. SFINCS) call `ensure_test_model`, which builds
# and spins it up only when it is not there yet, and otherwise reuses it.
TEST_MODEL_DIR: Path = tmp_folder / "model"
TEST_MODEL_BUILD_MARKER: Path = TEST_MODEL_DIR / "input" / "build_complete.txt"
TEST_MODEL_SPINUP_CHECKPOINTS: Path = (
    TEST_MODEL_DIR / "simulation_root" / "spinup" / "checkpoints"
)


def ensure_test_model() -> Path:
    """Build and spin up the shared test model if that has not been done yet.

    The first dependent test pays for the build and spinup (same steps as
    tests/test_model.py); later runs find the build marker and spinup checkpoint
    and reuse the model.

    Returns:
        The test model directory.
    """
    # imported here so that importing testconfig stays light
    from geb.cli import CONFIG_DEFAULT, build_fn, init_fn
    from geb.runner import parse_config, run_model_with_method
    from geb.workflows.io import WorkingDirectory

    TEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with WorkingDirectory(TEST_MODEL_DIR):
        if not TEST_MODEL_BUILD_MARKER.exists():
            if not Path("model.yml").exists():
                init_fn(
                    config="model.yml",
                    build_config="build.yml",
                    region="region.yml",
                    working_directory=".",
                    from_example="geul",
                    overwrite=True,
                )
            build_fn(continue_=True)

        if not (
            TEST_MODEL_SPINUP_CHECKPOINTS.exists()
            and any(TEST_MODEL_SPINUP_CHECKPOINTS.iterdir())
        ):
            config = parse_config(CONFIG_DEFAULT)
            config["hazards"]["floods"]["simulate"] = True
            model = run_model_with_method(
                method="spinup",
                config=config,
                method_args={"initialize_only": True},
                close_after_run=False,
                timing=False,
            )
            model.step_to_end()
            # the report (e.g. river outflow, used by the SFINCS tests) is only written
            # when the reporter is finalized, as at the end of `geb spinup`
            model.reporter.finalize()
            model.store.save()
            model.close()
    return TEST_MODEL_DIR


# This flag is used to turn on or off test cases in the Github Actions environment.
IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"
