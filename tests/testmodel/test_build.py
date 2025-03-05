import pytest
from pathlib import Path
from shutil import rmtree
from geb.cli import build_fn

from ..testconfig import tmp_folder, IN_GITHUB_ACTIONS


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_build():
    working_directory = tmp_folder / "build"
    working_directory.mkdir(parents=True, exist_ok=True)

    build_fn(
        data_catalog=[Path("../../../geb/data_catalog.yml")],
        config=Path("../../../examples/model.yml"),
        build_config=Path("../../../examples/build.yml"),
        working_directory=working_directory,
        custom_model=None,
        data_provider=None,
    )

    rmtree(working_directory, ignore_errors=True)
