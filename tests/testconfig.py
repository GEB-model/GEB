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


# This flag is used to turn on or off test cases in the Github Actions environment.
IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"
