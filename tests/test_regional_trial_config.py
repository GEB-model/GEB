# Explicit Returns/Raises sections follow project instructions.
# ruff: noqa: DOC202, DOC502
"""Check the Europe_003 overlay preserves domain, inputs and simulation dates."""

from pathlib import Path
from shutil import copy2

import yaml

from geb import GEB_PACKAGE_DIR
from geb.config_schema import Config
from geb.runner import parse_config


def test_regional_trial_preserves_base_domain(tmp_path: Path) -> None:
    """Resolve the portable overlay beside a regional model without overwriting it.

    Args:
        tmp_path: Isolated model directory.

    Returns:
        None.

    Raises:
        AssertionError: If regional settings are lost or baseline output is reused.
    """
    base: dict = {
        "inherits": str(GEB_PACKAGE_DIR / "reasonable_default_config.yml"),
        "general": {
            "region": {"subbasin": [24017156, 24017221]},
            "input_folder": "regional_input",
            "start_time": "2000-01-01",
            "end_time": "2010-12-31",
            "spinup_time": "1990-01-01",
            "output_folder": "baseline_output",
        },
    }
    baseline_path: Path = tmp_path / "model.yml"
    baseline_text: str = yaml.safe_dump(base)
    baseline_path.write_text(baseline_text)
    trial_path: Path = tmp_path / "model_lithuania_trial.yml"
    copy2(
        GEB_PACKAGE_DIR / "examples/regional_trials/model_lithuania_trial.yml",
        trial_path,
    )
    resolved: dict = parse_config(trial_path)
    Config.model_validate(resolved)
    for key in ["region", "input_folder", "start_time", "end_time", "spinup_time"]:
        assert resolved["general"][key] == base["general"][key]
    assert resolved["general"]["output_folder"] != "baseline_output"
    assert resolved["general"]["simulation_root"] == "simulation_root_lithuania_trial"
    assert resolved["parameters"]["interflow_multiplier"] == 3000.0
    assert resolved["parameters"]["crop_factor_multiplier"] == 1.25
    assert resolved["hydrology"]["routing"]["interflow_peak_hours"] == 48.0
    assert baseline_path.read_text() == baseline_text
