import pytest
from click.testing import CliRunner

from geb.cli import cli, multi_level_merge

from .testconfig import IN_GITHUB_ACTIONS


def test_multi_level_merge() -> None:
    dict1 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    dict2 = {"a": 2, "c": {"d": 4, "f": 5}}

    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": 2, "b": 2, "c": {"d": 4, "e": 4, "f": 5}}


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cli() -> None:
    runner = CliRunner()
    for cmd in (
        "spinup",
        "run",
        "build",
        "alter",
        "update",
        "calibrate",
        "sensitivity",
        "multirun",
        "share",
        "evaluate",
    ):
        assert runner.invoke(cli, [cmd, "--help"]).exit_code == 0
