"""Test suite for flood damage calculations."""

from typing import Any

import numpy as np

from geb.agents.modules.flood_risk import (
    FloodRiskModule,
)


class DummyHouseholdAgents:
    """A dummy model class to simulate the household agents object."""

    def __init__(self) -> None:
        """Initialize the dummy household agents with test data."""
        self.return_periods = np.array([10, 50, 100, 200, 500])
        self.damages = np.array(
            [
                [1000, 2000, 3000],
                [1200, 2200, 4000],
                [1300, 2300, 4300],
                [1400, 2400, 4400],
                [1500, 2500, 4500],
            ]
        )  # structured as [return_period, damage]
        self.comid_of_household = np.array([1, 2, 3])


class DummyFloodRiskModule(FloodRiskModule):
    """A dummy model class to simulate the household agents object."""

    def __init__(self, model: Any) -> None:
        """Initialize the dummy flood risk module with test data."""
        self.households = DummyHouseholdAgents()
        self.flood_protection_standard_subbasins = {1: 10, 2: 200, 3: 60}


def test_adjust_damages_for_flood_protection() -> None:
    """Test that damages below COMID flood protection standards are set to zero."""
    flood_risk_module = DummyFloodRiskModule(model=None)

    input_damages: np.ndarray = flood_risk_module.households.damages.copy()
    adjusted_damages: np.ndarray = (
        flood_risk_module._adjust_damages_for_flood_protection(input_damages)
    )

    expected_damages: np.ndarray = np.array(
        [
            [1000, 0, 0],
            [1200, 0, 0],
            [1300, 0, 4300],
            [1400, 2400, 4400],
            [1500, 2500, 4500],
        ]
    )

    np.testing.assert_array_equal(adjusted_damages, expected_damages)
