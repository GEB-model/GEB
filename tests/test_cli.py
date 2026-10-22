"""Tests for the command line interface (CLI) of GEB.

These tests include checking that the CLI commands are accessible and that the
multi-level dictionary merge function works correctly.
"""

import pytest
from click.testing import CliRunner

from geb.cli import AVAILABLE_EVALUATION_METHODS, cli

from .testconfig import IN_GITHUB_ACTIONS


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in GitHub Actions.")
def test_cli() -> None:
    """Test that all CLI commands are accessible and return help information."""
    runner = CliRunner()
    for cmd in (
        "init",
        "build",
        "set",
        "exec",
        "spinup",
        "run",
        "alter",
        "update",
        "share",
        "evaluate",
        "clean",
    ):
        assert runner.invoke(cli, [cmd, "--help"]).exit_code == 0


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in GitHub Actions.")
def test_cli_evaluate_help_lists_methods() -> None:
    """Test that evaluate help includes the available evaluation methods."""
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])

    assert result.exit_code == 0
    assert "Available methods:" in result.output
    for method_name in AVAILABLE_EVALUATION_METHODS:
        assert method_name in result.output
