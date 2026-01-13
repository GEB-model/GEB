"""Tests for the command line interface (CLI) of GEB.

These tests include checking that the CLI commands are accessible and that the
multi-level dictionary merge function works correctly.
"""

import pytest
from click.testing import CliRunner

from geb.cli import cli

from .testconfig import IN_GITHUB_ACTIONS


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
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
    ):
        assert runner.invoke(cli, [cmd, "--help"]).exit_code == 0


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cli_init_help() -> None:
    """Test that the init command returns help information."""
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0
    assert "Name of the example" in result.output


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cli_exec_help() -> None:
    """Test that the exec command returns help information."""
    runner = CliRunner()
    result = runner.invoke(cli, ["exec", "--help"])
    assert result.exit_code == 0
    assert "Method to run" in result.output
