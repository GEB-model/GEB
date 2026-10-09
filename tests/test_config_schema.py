"""Tests for configuration schema and parameters validation."""

import os
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from geb import GEB_PACKAGE_DIR
from geb.config_schema import Config, KarstConfig, ParametersConfig
from geb.runner import parse_config

EXPECTED_PARAMETERS: set[str] = {
    "mannings_n_multiplier",
    "bankfull_discharge_multiplier",
    "crop_factor_multiplier",
    "saturated_hydraulic_conductivity_multiplier",
    "groundwater_hydraulic_conductivity_multiplier",
    "reservoir_release_factor",
    "lake_outflow_multiplier",
    "interflow_multiplier",
    "variable_runoff_shape_beta_multiplier",
}


def test_karst_default_matches_configuration() -> None:
    """Use full capture in both the schema and default configuration."""
    with (GEB_PACKAGE_DIR / "reasonable_default_config.yml").open() as config_file:
        defaults: dict[str, Any] = yaml.safe_load(config_file)
    assert KarstConfig().capture_fraction == 1.0
    assert defaults["hydrology"]["karst"]["capture_fraction"] == 1.0
    assert KarstConfig().enabled is False
    assert KarstConfig(capture_fraction=0.5).capture_fraction == 0.5
    assert "release_time_days" not in defaults["hydrology"]["karst"]


def test_parameters_config_requires_all_parameters() -> None:
    """Test that ParametersConfig raises ValidationError when parameters are missing."""
    with pytest.raises(ValidationError) as exc_info:
        ParametersConfig.model_validate({})

    missing_fields: set[str] = {
        err["loc"][0]
        for err in exc_info.value.errors()
        if isinstance(err["loc"][0], str)
    }
    assert missing_fields == EXPECTED_PARAMETERS


def test_parameters_config_validates_when_all_provided() -> None:
    """Test that ParametersConfig successfully initializes when all parameters are supplied."""
    params = {
        "mannings_n_multiplier": 1.0,
        "bankfull_discharge_multiplier": 1.0,
        "crop_factor_multiplier": 1.0,
        "saturated_hydraulic_conductivity_multiplier": 1.0,
        "groundwater_hydraulic_conductivity_multiplier": 1.0,
        "reservoir_release_factor": 0.1,
        "lake_outflow_multiplier": 1.0,
        "interflow_multiplier": 1.0,
        "variable_runoff_shape_beta_multiplier": 1.0,
    }
    config = ParametersConfig(**params)
    for key, value in params.items():
        assert getattr(config, key) == value


def test_reasonable_default_config_has_all_parameters() -> None:
    """Test that reasonable_default_config.yml contains all required parameters."""
    default_config_path = GEB_PACKAGE_DIR / "reasonable_default_config.yml"
    with open(default_config_path, "r") as f:
        data = yaml.safe_load(f)

    assert "parameters" in data, (
        "'parameters' section missing in reasonable_default_config.yml"
    )
    params_data = data["parameters"]
    assert set(params_data.keys()) == EXPECTED_PARAMETERS

    # Ensure ParametersConfig validates the section
    validated = ParametersConfig(**params_data)
    assert validated is not None


def test_example_model_config_has_all_parameters() -> None:
    """Test that examples/geul/model.yml explicitly defines all parameters."""
    example_model_path = GEB_PACKAGE_DIR / "examples" / "geul" / "model.yml"
    with open(example_model_path, "r") as f:
        data = yaml.safe_load(f)

    assert "parameters" in data, (
        "'parameters' section missing in examples/geul/model.yml"
    )
    params_data = data["parameters"]
    assert set(params_data.keys()) == EXPECTED_PARAMETERS

    # Ensure ParametersConfig validates the section
    validated = ParametersConfig(**params_data)
    assert validated is not None


def test_example_model_parses_with_config_schema() -> None:
    """Test that examples/geul/model.yml parses successfully with Config schema."""
    os.environ["GEB_PACKAGE_DIR"] = str(GEB_PACKAGE_DIR)
    example_model_path = GEB_PACKAGE_DIR / "examples" / "geul" / "model.yml"
    parsed = parse_config(example_model_path, schema=Config)
    assert "parameters" in parsed
    assert set(parsed["parameters"].keys()) == EXPECTED_PARAMETERS
