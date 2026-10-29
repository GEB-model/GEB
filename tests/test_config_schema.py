"""Tests for configuration schema and validation."""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geb import GEB_PACKAGE_DIR
from geb.config_schema import (
    AgentsConfig,
    Config,
    FloodsConfig,
    ForecastsConfig,
    GroundWaterConfig,
    HouseholdsConfig,
    LandSurfaceConfig,
    ReservoirOperatorsConfig,
    RiverDepthConfig,
    RiverWidthConfig,
    RoutingConfig,
    RunoffConcentratorConfig,
)
from geb.runner import parse_config, set_fn


def test_reasonable_default_config_validates() -> None:
    """Test that reasonable_default_config.yml parses and validates with Config schema."""
    default_config_path = GEB_PACKAGE_DIR / "reasonable_default_config.yml"
    with open(default_config_path, "r") as f:
        data = yaml.safe_load(f)

    validated = Config.model_validate(data)
    assert validated.forecasts is not None
    assert validated.hydrology is not None
    assert validated.agents is not None
    assert validated.evaluation is not None
    assert validated.loglevel == "DEBUG"
    assert "parameters" not in data
    assert "general" not in data
    assert "agent_settings" not in data
    assert "logging" not in data


def test_example_model_parses_with_config_schema() -> None:
    """Test that examples/geul/model.yml parses successfully with Config schema."""
    os.environ["GEB_PACKAGE_DIR"] = str(GEB_PACKAGE_DIR)
    example_model_path = GEB_PACKAGE_DIR / "examples" / "geul" / "model.yml"
    parsed = parse_config(example_model_path, schema=Config)
    assert "hydrology" in parsed
    assert "landsurface" in parsed["hydrology"]
    assert "routing" in parsed["hydrology"]


def test_all_schema_fields_have_description() -> None:
    """Test that every field in every model in config_schema has a non-empty description."""
    from pydantic import BaseModel

    import geb.config_schema as cs

    missing: list[str] = []
    for name, obj in vars(cs).items():
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            for fname, ffield in obj.model_fields.items():
                if not ffield.description or not ffield.description.strip():
                    missing.append(f"{name}.{fname}")

    assert not missing, f"The following fields are missing a description: {missing}"


def test_parameters_are_required_in_schema() -> None:
    """Test that model parameter sections require explicit parameter values."""
    with pytest.raises(ValidationError):
        Config.model_validate({})
    with pytest.raises(ValidationError):
        ForecastsConfig.model_validate({})
    with pytest.raises(ValidationError):
        FloodsConfig.model_validate({})
    with pytest.raises(ValidationError):
        HouseholdsConfig.model_validate({})
    with pytest.raises(ValidationError):
        AgentsConfig.model_validate({})
    with pytest.raises(ValidationError):
        LandSurfaceConfig.model_validate({})
    with pytest.raises(ValidationError):
        GroundWaterConfig.model_validate({})
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate({})
    with pytest.raises(ValidationError):
        RunoffConcentratorConfig.model_validate({})
    with pytest.raises(ValidationError):
        ReservoirOperatorsConfig.model_validate({})
    with pytest.raises(ValidationError):
        RiverWidthConfig.model_validate({})
    with pytest.raises(ValidationError):
        RiverDepthConfig.model_validate({})


def test_return_periods_validation() -> None:
    """Test validation of return_periods in FloodsConfig."""
    default_config_path = GEB_PACKAGE_DIR / "reasonable_default_config.yml"
    with open(default_config_path, "r") as f:
        data = yaml.safe_load(f)
    floods_data = data["hazards"]["floods"]

    # String instead of list of ints must raise ValidationError
    invalid_data = floods_data.copy()
    invalid_data["return_periods"] = "invalid_string"
    with pytest.raises(ValidationError):
        FloodsConfig.model_validate(invalid_data)

    # Empty list must raise ValidationError
    invalid_data_empty = floods_data.copy()
    invalid_data_empty["return_periods"] = []
    with pytest.raises(ValidationError):
        FloodsConfig.model_validate(invalid_data_empty)

    # Negative integer must raise ValidationError
    invalid_data_neg = floods_data.copy()
    invalid_data_neg["return_periods"] = [2, -5, 10]
    with pytest.raises(ValidationError):
        FloodsConfig.model_validate(invalid_data_neg)

    # Valid list of positive integers succeeds
    valid_data = floods_data.copy()
    valid_data["return_periods"] = [10, 25, 100]
    validated = FloodsConfig.model_validate(valid_data)
    assert validated.return_periods == [10, 25, 100]


def test_set_fn_validates_and_rejects_invalid_values(tmp_path: Path) -> None:
    """Test that set_fn raises ValidationError on invalid configuration values."""
    default_config_path = GEB_PACKAGE_DIR / "reasonable_default_config.yml"
    with open(default_config_path, "r") as f:
        data = yaml.safe_load(f)

    test_model_yml = tmp_path / "model.yml"
    with open(test_model_yml, "w") as f:
        yaml.dump(data, f)

    # Setting a string for return_periods must raise ValidationError
    with pytest.raises(ValidationError):
        set_fn(
            config=test_model_yml,
            working_directory=tmp_path,
            **{"hazards.floods.return_periods": "mystring"},
        )

    # Setting valid list succeeds and updates the file
    set_fn(
        config=test_model_yml,
        working_directory=tmp_path,
        **{"hazards.floods.return_periods": [10, 50, 100]},
    )
    with open(test_model_yml, "r") as f:
        updated = yaml.safe_load(f)
    assert updated["hazards"]["floods"]["return_periods"] == [10, 50, 100]


def test_return_periods_validation() -> None:
    """Test validation of return_periods in FloodsConfig."""
    default_config_path = GEB_PACKAGE_DIR / "reasonable_default_config.yml"
    with open(default_config_path, "r") as f:
        data = yaml.safe_load(f)
    floods_data = data["hazards"]["floods"]

    # String instead of list of ints must raise ValidationError
    invalid_data = floods_data.copy()
    invalid_data["return_periods"] = "invalid_string"
    with pytest.raises(ValidationError):
        FloodsConfig.model_validate(invalid_data)

    # Empty list must raise ValidationError
    invalid_data_empty = floods_data.copy()
    invalid_data_empty["return_periods"] = []
    with pytest.raises(ValidationError):
        FloodsConfig.model_validate(invalid_data_empty)

    # Negative integer must raise ValidationError
    invalid_data_neg = floods_data.copy()
    invalid_data_neg["return_periods"] = [2, -5, 10]
    with pytest.raises(ValidationError):
        FloodsConfig.model_validate(invalid_data_neg)

    # Valid list of positive integers succeeds
    valid_data = floods_data.copy()
    valid_data["return_periods"] = [10, 25, 100]
    validated = FloodsConfig.model_validate(valid_data)
    assert validated.return_periods == [10, 25, 100]


def test_set_fn_validates_and_rejects_invalid_values(tmp_path: Path) -> None:
    """Test that set_fn raises ValidationError on invalid configuration values."""
    default_config_path = GEB_PACKAGE_DIR / "reasonable_default_config.yml"
    with open(default_config_path, "r") as f:
        data = yaml.safe_load(f)

    test_model_yml = tmp_path / "model.yml"
    with open(test_model_yml, "w") as f:
        yaml.dump(data, f)

    # Setting a string for return_periods must raise ValidationError
    with pytest.raises(ValidationError):
        set_fn(
            config=test_model_yml,
            working_directory=tmp_path,
            **{"hazards.floods.return_periods": "mystring"},
        )

    # Setting valid list succeeds and updates the file
    set_fn(
        config=test_model_yml,
        working_directory=tmp_path,
        **{"hazards.floods.return_periods": [10, 50, 100]},
    )
    with open(test_model_yml, "r") as f:
        updated = yaml.safe_load(f)
    assert updated["hazards"]["floods"]["return_periods"] == [10, 50, 100]


def test_config_reference_docs_generator() -> None:
    """Test that the documentation generator produces all reference tables."""
    from docs.hooks.config_reference import (
        generate_reference_markdown,
        on_page_markdown,
    )

    content: str = generate_reference_markdown()
    assert "### General settings" in content
    assert "### Forecasts" in content
    assert "### Hazards" in content
    assert "### Hydrology" in content
    assert "### Evaluation" in content
    assert "### Agents" in content
    assert "### Reporting" in content
    # Assert nested submodels are unrolled
    assert "traditional_insurance.loading_rate" in content
    assert "flood_protection_standard.mode" in content
    assert "retention_basins.release_threshold_factor" in content
    assert "_config.compression_level" in content
    # Assert union and literal types use readable 'or'
    assert '`list[int] or "all" or "auto"`' in content

    # Test hook replacement
    sample_page_md: str = "# Config\n\n<!-- autogenerated:config-reference -->"
    replaced: str = on_page_markdown(
        markdown=sample_page_md, page=None, config=None, files=None
    )
    assert "### General settings" in replaced
    assert "<!-- autogenerated:config-reference -->" not in replaced
