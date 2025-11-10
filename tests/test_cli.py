"""Tests for the command line interface (CLI) of GEB.

These tests include checking that the CLI commands are accessible and that the
multi-level dictionary merge function works correctly.
"""

import pytest
from click.testing import CliRunner

from geb.cli import cli, multi_level_merge

from .testconfig import IN_GITHUB_ACTIONS


def test_multi_level_merge() -> None:
    """Test multi-level dictionary merging.

    Verifies that the multi_level_merge function correctly merges
    two dictionaries, including nested dictionaries.
    """
    dict1 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    dict2 = {"a": 2, "c": {"d": 4, "f": 5}}

    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": 2, "b": 2, "c": {"d": 4, "e": 4, "f": 5}}


def test_multi_level_merge_empty_dicts() -> None:
    """Test merging two empty dictionaries."""
    dict1 = {}
    dict2 = {}
    merged = multi_level_merge(dict1, dict2)
    assert merged == {}


def test_multi_level_merge_one_empty() -> None:
    """Test merging an empty dictionary with a non-empty one."""
    dict1 = {}
    dict2 = {"a": 1, "b": {"c": 2}}
    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": 1, "b": {"c": 2}}

    # Reverse order
    merged_reverse = multi_level_merge(dict2, dict1)
    assert merged_reverse == {"a": 1, "b": {"c": 2}}


def test_multi_level_merge_deep_nesting() -> None:
    """Test merging deeply nested dictionaries."""
    dict1 = {"a": {"b": {"c": {"d": 1}}}}
    dict2 = {"a": {"b": {"c": {"e": 2}}, "f": 3}}
    merged = multi_level_merge(dict1, dict2)
    expected = {"a": {"b": {"c": {"d": 1, "e": 2}}, "f": 3}}
    assert merged == expected


def test_multi_level_merge_non_dict_values() -> None:
    """Test merging when keys have non-dictionary values."""
    dict1 = {"a": 1, "b": [1, 2], "c": "string"}
    dict2 = {"a": 2, "b": [3, 4], "d": 5}
    merged = multi_level_merge(dict1, dict2)
    # Assuming non-dict values are overwritten, not merged
    assert merged == {"a": 2, "b": [3, 4], "c": "string", "d": 5}


def test_multi_level_merge_mixed_types() -> None:
    """Test merging with mixed types, including overwriting dict with non-dict."""
    dict1 = {"a": {"nested": 1}}
    dict2 = {"a": "not_a_dict"}
    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": "not_a_dict"}


def test_multi_level_merge_no_overlap() -> None:
    """Test merging dictionaries with no overlapping keys."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    merged = multi_level_merge(dict1, dict2)
    assert merged == {"a": 1, "b": 2, "c": 3, "d": 4}


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
