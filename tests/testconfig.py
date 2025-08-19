import os
from pathlib import Path

# Get the GEB package directory from the environment variable
GEB_PACKAGE_DIR_ENV: str | None = os.environ.get("GEB_PACKAGE_DIR", None)
if GEB_PACKAGE_DIR_ENV is None:
    raise RuntimeError("GEB_PACKAGE_DIR environment variable is not set.")

GEB_PACKAGE_DIR: Path = Path(GEB_PACKAGE_DIR_ENV)
GEB_TEST_DIR: Path = GEB_PACKAGE_DIR.parent / "tests"

# Ensure the testing files end in up in the tests folder of the GEB package directory
output_folder: Path = GEB_TEST_DIR / "output"
output_folder.mkdir(exist_ok=True)

tmp_folder: Path = GEB_TEST_DIR / "tmp"
tmp_folder.mkdir(exist_ok=True)

# This flag is used to turn on or off test cases in the Github Actions environment.
IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"
