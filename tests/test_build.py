import pytest
from click.testing import CliRunner

from geb.cli import build

from .testconfig import IN_GITHUB_ACTIONS


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test test requires local data.")
def test_build():
    runner = CliRunner()
    result = runner.invoke(build, ["-wd", "examples"])

    if result.exit_code != 0:
        error_message = (
            result.exception.strerror
            if hasattr(result.exception, "strerror")
            else result.output
        )
        raise AssertionError(f"Build failed: {error_message}")
