"""Tests for the command line interface (CLI) of GEB.

These tests include checking that the CLI commands are accessible and that the
config override and id options work correctly.
"""

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from geb.cli import AVAILABLE_EVALUATION_METHODS, cli
from geb.runner import (
    apply_config_overrides,
    normalize_config,
    parse_override_value,
    parse_with_args,
)

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


def test_parse_override_value() -> None:
    """Test type conversion for CLI string override values."""
    assert parse_override_value("123") == 123
    assert parse_override_value("3.14") == 3.14
    assert parse_override_value("2024-05-01") == date(2024, 5, 1)
    assert parse_override_value("true") is True
    assert parse_override_value("false") is False
    assert parse_override_value("null") is None
    assert parse_override_value("my_scenario") == "my_scenario"
    assert parse_override_value("[2, 5, 10]") == [2, 5, 10]
    assert parse_override_value("['a', 'b']") == ["a", "b"]
    assert parse_override_value("{'a': 1}") == {"a": 1}


def test_parse_with_args() -> None:
    """Test parsing --with arguments passed as repeated flags or space-separated."""
    parsed = parse_with_args(("a=1", "b=true", "c=null"))
    assert parsed == {"a": 1, "b": True, "c": None}

    parsed_spaced = parse_with_args(("a=1 b=true", "c=null d=3.14"))
    assert parsed_spaced == {"a": 1, "b": True, "c": None, "d": 3.14}


def test_apply_config_overrides() -> None:
    """Test applying nested dot-notation overrides to configuration dictionary."""
    config: dict[str, Any] = {
        "id": None,
        "name": "default",
        "hazards": {"floods": {"simulate": False}},
    }
    overrides = {
        "id": "exp1",
        "hazards.floods.simulate": True,
        "new_key+": "created",
    }
    updated = apply_config_overrides(config, overrides)

    assert updated["id"] == "exp1"
    assert updated["hazards"]["floods"]["simulate"] is True
    assert updated["new_key"] == "created"


def test_normalize_config_id_derivation() -> None:
    """Test that id derives spinup_name and name unless explicitly overridden."""
    config: dict[str, Any] = {
        "id": "scenario_a",
        "spinup_name": "spinup",
        "name": "default",
    }
    normalized = normalize_config(config)
    assert normalized["spinup_name"] == "spinup_scenario_a"
    assert normalized["name"] == "run_scenario_a"

    # Explicit override takes precedence
    config_b: dict[str, Any] = {
        "id": "scenario_b",
        "spinup_name": "custom_spinup",
        "name": "default",
    }
    explicit = {"spinup_name": "custom_spinup"}
    with_override = normalize_config(config_b, explicit_overrides=explicit)
    assert with_override["spinup_name"] == "custom_spinup"
    assert with_override["name"] == "run_scenario_b"


def test_init_creates_region_file(tmp_path: Path) -> None:
    """Test that geb init creates region.yml and model.yml without region."""
    from geb.runner import init_fn, read_params

    init_fn(
        working_directory=tmp_path,
        basin_id="23011134",
    )

    assert (tmp_path / "model.yml").exists()
    assert (tmp_path / "region.yml").exists()
    assert (tmp_path / "build.yml").exists()
    assert (tmp_path / "update.yml").exists()

    region_cfg: dict[str, Any] = read_params(tmp_path / "region.yml")
    assert region_cfg["subbasin"] == 23011134

    model_cfg: dict[str, Any] = read_params(tmp_path / "model.yml")
    assert "region" not in model_cfg
