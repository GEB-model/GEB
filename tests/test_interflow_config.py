# Explicit Returns/Raises sections follow project instructions.
# ruff: noqa: DOC202, DOC502
"""Check optional interflow travel-time schema and shipped defaults."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geb import GEB_PACKAGE_DIR
from geb.config_schema import Config


@pytest.mark.parametrize("peak", [0.0, 48.0, -1.0, float("inf"), float("nan")])
def test_interflow_peak_schema(peak: float) -> None:
    """Validate travel-time bounds in the full shipped configuration.

    Args:
        peak: Candidate interflow peak (hours).

    Returns:
        None.

    Raises:
        AssertionError: If defaults or schema validation differ from the contract.
    """
    path: Path = GEB_PACKAGE_DIR / "reasonable_default_config.yml"
    config: dict = yaml.safe_load(path.read_text())
    assert config["hydrology"]["routing"]["interflow_peak_hours"] == 0.0
    config["hydrology"]["routing"]["interflow_peak_hours"] = peak
    if peak < 0 or not peak < float("inf"):
        with pytest.raises(ValidationError):
            Config.model_validate(config)
    else:
        result: Config = Config.model_validate(config)
        assert result.hydrology.routing.interflow_peak_hours == peak
