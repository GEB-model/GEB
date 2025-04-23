import os
from pathlib import Path

import pytest

from geb.cli import build_fn

from ..testconfig import IN_GITHUB_ACTIONS, tmp_folder

example = Path("../../../examples/geul")


working_directory = tmp_folder / "model"
working_directory.mkdir(parents=True, exist_ok=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_build():
    build_fn(
        data_catalog=[Path("../../../geb/data_catalog.yml")],
        config=example / "model.yml",
        build_config=example / "build.yml",
        working_directory=working_directory,
        custom_model=None,
        data_provider=None,
        data_root=Path(os.getenv("GEB_DATA_ROOT")),
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Too heavy for GitHub Actions.")
def test_update():
    pass
