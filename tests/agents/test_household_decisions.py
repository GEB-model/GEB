"""Test suite for household decision making module."""

import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import fixture

from geb.agents.decision_module import (
    DecisionModule,
)  # update to now decision model after merge
from geb.agents.households import Households


@fixture
def buildings() -> pd.DataFrame:
    """This function creates a sample of buildings for testing.

    Returns:
        A pandas DataFrame containing the building attributes for testing.
    """
    buildings = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "x": [0, 1, 2, 3, 4],
            "y": [0, 1, 2, 3, 4],
            "COST_STRUCTURAL_USD_SQM": [100, 200, 300, 400, 500],
            "COST_CONTENTS_USD_SQM": [50, 100, 150, 200, 250],
            "COMID": [1, 1, 2, 2, 3],
            "distance_to_river_m": [10, 20, 30, 40, 50],
            "distance_to_coastline_m": [5, 10, 15, 20, 25],
        }
    )
    return buildings


@fixture
def decision_template() -> dict[str, Any]:
    """This function creates a decision template for the decision module.

    Returns:
        A dictionary containing the parameters for the decision module.
    """
    geom_id = "<GeoID>"
    n_agents = 1
    return_periods = np.array([10, 50])
    income = np.random.randint(10_000, 20_000, n_agents)
    wealth = 1.5 * income
    expendature_cap = 0.1
    risk_perception = np.full(n_agents, 1)
    expected_damages_no_adapt = np.array(
        [np.full(n_agents, 5_000), np.full(n_agents, 15_000)]
    )
    damages_adapt = expected_damages_no_adapt * 0.8
    adaptation_costs = 100
    time_adapted = np.zeros(n_agents)
    loan_duration = 20
    T = 15
    r = 0.03
    sigma = 1
    adapted = np.full(n_agents, 0)
    distance_to_river = np.random.randint(0, 10e3, n_agents)
    distance_to_coastline = np.maximum([10e3], [np.random.randint(5, 50e3, n_agents)])[
        0
    ]
    GDP_i_t = 1
    max_migration_cost = 5e5
    sampled_building_ids = np.random.choice(np.arange(6), (n_agents, 3))
    distance_to_building_m = np.full_like(sampled_building_ids, 5e3)
    distance_to_river_buildings = np.random.randint(0, 10e3, (n_agents, 3))
    distance_to_coastline_buildings = np.random.randint(0, 10e3, (n_agents, 3))

    decision_template = {
        "geom_id": geom_id,
        "n_agents": n_agents,
        "wealth": wealth,
        "income": income,
        "expendature_cap": expendature_cap,
        "household_distance_to_river_m": distance_to_river,
        "household_distance_to_coastline_m": distance_to_coastline,
        "distance_to_river_m": distance_to_river_buildings,
        "distance_to_coastline_m": distance_to_coastline_buildings,
        "risk_perception": risk_perception,
        "expected_damages": expected_damages_no_adapt,
        "expected_damages_adapt": damages_adapt,
        "adaptation_costs": adaptation_costs,
        "time_adapted": time_adapted,
        "loan_duration": loan_duration,
        "adapted": adapted,
        "p_floods": 1 / return_periods,
        "T": T,
        "r": r,
        "sigma": sigma,
        "GDP_i_t": GDP_i_t,
        "sampled_building_ids": sampled_building_ids,
        "max_migration_costs": max_migration_cost,
        "distance_to_building_m": distance_to_building_m,
    }

    return decision_template


@fixture
def decision_module() -> DecisionModule:
    """Create a decision module with the household configuration it needs.

    Returns:
        Decision module configured with the household amenity scaling factor.
    """
    model = SimpleNamespace(
        config={
            "agent_settings": {
                "households": {
                    "ammenity_value_scaling_factor": 0.254820579,
                }
            }
        }
    )
    return DecisionModule(model)


def test_expenditure_cap(
    decision_template: dict,
    decision_module: DecisionModule,
) -> None:
    """This function tests the functionality of the expenditure cap.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
        decision_module: Decision module configured with the household amenity scaling factor.
    """
    # quick basic test
    EU_do_not_adapt = decision_module.calcEU_do_nothing_flood(**decision_template)
    assert all(EU_do_not_adapt > -np.inf), (
        "Expected all EU_do_not_adapt values to be greater than -inf as there are no budget constraints"
    )

    # make sure an expendature cap results in no adaptation
    decision_template["expendature_cap"] = 0
    EU_adapt = decision_module.calcEU_adapt_flood(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing_flood(**decision_template)
    assert all(EU_adapt == -np.inf), "Expected all EU_adapt values to be -inf"
    assert all(EU_do_not_adapt > EU_adapt), (
        "Expected all EU_do_not_adapt values to be greater than EU_adapt"
    )
    assert all(EU_do_not_adapt > -np.inf), (
        "Expected all EU_do_not_adapt values to be greater than -inf. Should not be affected"
    )

    # make sure setting expenditure cap to high results in all agents being able to afford adaptation
    decision_template["expendature_cap"] = 10
    EU_adapt = decision_module.calcEU_adapt_flood(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing_flood(**decision_template)
    assert all(EU_adapt > -np.inf), (
        "Expected all EU_adapt values to be greater than -inf"
    )
    assert all(EU_do_not_adapt > -np.inf), (
        "Expected all EU_do_not_adapt values to be greater than -inf. Should not be affected"
    )


def test_risk_perception(
    decision_template: dict,
    decision_module: DecisionModule,
) -> None:
    """This function tests the functionality of risk perception.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
        decision_module: Decision module configured with the household amenity scaling factor.

    """
    decision_template["expendature_cap"] = 10  # ensure all can adapt
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 0.01)
    EU_do_nothing_low_risk_perception = decision_module.calcEU_do_nothing_flood(
        **decision_template
    )
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 10)
    EU_do_nothing_high_risk_perception = decision_module.calcEU_do_nothing_flood(
        **decision_template
    )
    # make sure EU_do_nothing_high_risk_perception EU of adaptation is
    assert all(
        EU_do_nothing_low_risk_perception > EU_do_nothing_high_risk_perception
    ), "Expected EU_do_nothing_low to be greater than EU_do_nothing_high"


def test_damages(
    decision_template: dict,
    decision_module: DecisionModule,
) -> None:
    """This function tests the functionality of damages with and without adaptation.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
        decision_module: Decision module configured with the household amenity scaling factor.
    """
    # make sure all can adapt and behave rationally
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)
    # set damages under adaptation to zero
    decision_template["expected_damages_adapt"] *= 0

    # calculate EU
    EU_adapt = decision_module.calcEU_adapt_flood(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing_flood(**decision_template)
    assert all(EU_adapt > EU_do_not_adapt), (
        "Expected all EU_adapt values to be greater than EU_do_not_adapt"
    )

    # now check with no effect of adaptation on damage
    decision_template["expected_damages_adapt"] = decision_template["expected_damages"]
    EU_adapt_no_effect = decision_module.calcEU_adapt_flood(**decision_template)
    assert all(EU_adapt_no_effect < EU_do_not_adapt), (
        "Expected all EU_adapt_no_effect values to be less than EU_do_not_adapt"
    )


def test_time_adapted(
    decision_template: dict,
    decision_module: DecisionModule,
) -> None:
    """This function tests the functionality of the time since adaptation decision.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
        decision_module: Decision module configured with the household amenity scaling factor.
    """
    # make sure all can adapt and behave rationally
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)

    # set damages to equal
    decision_template["expected_damages_adapt"] = decision_template["expected_damages"]
    # calculate EU
    EU_adapt = decision_module.calcEU_adapt_flood(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing_flood(**decision_template)
    assert all(EU_adapt < EU_do_not_adapt), (
        "Expected all EU_adapt values to be less than EU_do_not_adapt, as there is no damage reduction but adaptation costs are incurred."
    )

    # set time since adaptation to be the loan duration + 1
    decision_template["time_adapted"] = np.full(
        decision_template["n_agents"], decision_template["loan_duration"] + 1
    )
    decision_template["expected_damages_adapt"] = decision_template["expected_damages"]

    # calculate EU
    EU_adapt = decision_module.calcEU_adapt_flood(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing_flood(**decision_template)
    assert all(EU_adapt == EU_do_not_adapt), (
        "Expected all EU_adapt values to be equal to EU_do_not_adapt as there are no costs incurred anymore"
    )


def test_loan_duration(
    decision_template: dict,
    decision_module: DecisionModule,
) -> None:
    """This function tests the functionality of the loan duration.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
        decision_module: Decision module configured with the household amenity scaling factor.
    """
    # make sure all can adapt and behave rationally and set time discounting to zero
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)
    decision_template["r"] = 0

    # set loan duration short
    decision_template["loan_duration"] = 5
    EU_adapt_short = decision_module.calcEU_adapt_flood(**decision_template)

    # set loan duration long
    decision_template["loan_duration"] = 10
    EU_adapt_long = decision_module.calcEU_adapt_flood(**decision_template)

    assert all(EU_adapt_short > EU_adapt_long), (
        "Expected all EU_adapt_long values to be less than EU_adapt_short due to higher interest costs incurred"
    )


def test_time_discounting(
    decision_template: dict,
    decision_module: DecisionModule,
) -> None:
    """This function tests the functionality of time discounting.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
        decision_module: Decision module configured with the household amenity scaling factor.
    """
    # make sure all can adapt and behave rationally
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)

    # set time discounting low
    decision_template["r"] = 0.01
    EU_adapt_low = decision_module.calcEU_adapt_flood(**decision_template)

    # set time discounting high
    decision_template["r"] = 0.1
    EU_adapt_high = decision_module.calcEU_adapt_flood(**decision_template)

    assert all(EU_adapt_low > EU_adapt_high), (
        "Expected all EU_adapt_high values to be less than EU_adapt_low due to higher time discounting"
    )


def test_decision_horizon(
    decision_template: dict,
    decision_module: DecisionModule,
) -> None:
    """This function tests the functionality of the decision horizon.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
        decision_module: Decision module configured with the household amenity scaling factor.
    """
    # make sure all can adapt and behave rationally
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)

    # set time discounting to zero (no adaptation costs should be incurred in year zero)
    decision_template["T"] = 0
    EU_adapt_low = decision_module.calcEU_adapt_flood(**decision_template)
    EU_do_not_adapt_low = decision_module.calcEU_do_nothing_flood(**decision_template)
    assert all(EU_adapt_low == EU_do_not_adapt_low), (
        "Expected all EU_adapt_low values to be equal to EU_do_not_adapt_low as there are no costs incurred anymore"
    )
    # set time discounting high
    decision_template["T"] = 20
    EU_adapt_high = decision_module.calcEU_adapt_flood(**decision_template)

    assert all(EU_adapt_low < EU_adapt_high), (
        "Expected all EU_adapt_high values to be greater than EU_adapt_low due to summation over longer time period"
    )


def test_amenity_values(
    decision_template: dict,
    decision_module: DecisionModule,
) -> None:
    """This function tests the functionality of amenity values.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
        decision_module: Decision module configured with the household amenity scaling factor.
    """
    # make sure all can adapt and behave rationally
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)
    decision_template["adaptation_costs"] = 0  # ensure no adaptation costs are incurred

    # set damages to zero
    decision_template["expected_damages"] = np.zeros(
        (len(decision_template["p_floods"]), decision_template["n_agents"])
    )
    decision_template["expected_damages_adapt"] = np.zeros(
        (len(decision_template["p_floods"]), decision_template["n_agents"])
    )
    # set migration costs to zero
    decision_template["max_migration_costs"] = 0

    # set amenity values to 100
    decision_template["household_distance_to_river_m"] = np.full(
        decision_template["n_agents"], 100
    )
    decision_template["household_distance_to_coastline_m"] = np.full(
        decision_template["n_agents"], 100
    )
    decision_template["distance_to_river_m"] = np.full(
        (decision_template["n_agents"], 3), 100
    )
    decision_template["distance_to_coastline_m"] = np.full(
        (decision_template["n_agents"], 3), 100
    )

    EU_adapt = decision_module.calcEU_adapt_flood(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing_flood(**decision_template)
    building_idx, EU_relocate = decision_module.calcEU_relocate(**decision_template)

    # The two calculations are mathematically equivalent here: zero damages,
    # zero migration costs, and identical amenity values across all sampled homes.
    # They are evaluated through different float32 summation paths, so tolerance-
    # based comparison is required rather than exact equality.
    assert np.allclose(EU_adapt, EU_do_not_adapt), (
        "Expected all EU_adapt values to be approximately equal to EU_do_not_adapt as there are no costs incurred and no damages expected"
    )
    assert np.allclose(EU_relocate, EU_do_not_adapt), (
        "Expected all EU_relocate values to be approximately equal to EU_do_not_adapt as there are no costs incurred and no damages expected"
    )


@pytest.mark.parametrize(
    ("project_income", "project_costs"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_update_monetary_variables_respects_projection_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_income: bool,
    project_costs: bool,
) -> None:
    """Check that income and cost projections follow their respective flags."""
    years = [2020, 2021]
    gdp_map = xr.DataArray(
        np.array(
            [
                [[100.0, 100.0], [100.0, 100.0]],
                [[110.0, 110.0], [110.0, 110.0]],
            ]
        ),
        dims=("year", "lon", "lat"),
        coords={"year": years, "lon": [0.0, 1.0], "lat": [0.0, 1.0]},
        attrs={
            "lon_origin": 0.0,
            "lon_resolution": 1.0,
            "lat_origin": 0.0,
            "lat_resolution": -1.0,
        },
    )
    population_map = xr.DataArray(
        np.full((2, 2, 2), 10.0),
        dims=("year", "lon", "lat"),
        coords={"year": years, "lon": [0.0, 1.0], "lat": [0.0, 1.0]},
    )

    opened_paths: list[str] = []

    def fake_open_dataarray(path: Path) -> xr.DataArray:
        opened_paths.append(path.name)
        return population_map if path.name == "POP.zarr" else gdp_map

    monkeypatch.setattr(
        "geb.agents.households.xr.open_dataarray",
        fake_open_dataarray,
    )

    class FakeTransform:
        def to_gdal(self) -> tuple[int, int, int, int, int, int]:
            return (0, 1, 0, 0, 0, -1)

    monkeypatch.setattr(
        xr.DataArray,
        "rio",
        property(lambda _: SimpleNamespace(transform=lambda: FakeTransform())),
        raising=False,
    )

    climrisk_config: dict[str, Any] = {
        "project_income": project_income,
        "project_costs": project_costs,
        "country_wide": True,
        "ssp": "2",
        "damage_function": "GDP",
    }

    households = Households.__new__(Households)
    households.start_adaptation_year = 2020
    households.model = SimpleNamespace(
        input_folder=tmp_path,
        current_time=datetime(2021, 1, 1),
        logger=logging.getLogger("test"),
        config={"agent_settings": {"climrisk": climrisk_config}},
        files={
            "other": {
                "ssp_maps/GDP": tmp_path / "GDP.zarr",
                "ssp_maps/POP": tmp_path / "POP.zarr",
            }
        },
    )
    households.var = SimpleNamespace(
        income=np.array([100.0]),
        wealth=np.array([200.0]),
    )

    households.update_monetary_variables_to_ssp()

    income_factor = 1.1 if project_income else 1.0
    cost_factor = 1.1 if project_costs else 1.0
    np.testing.assert_allclose(households.var.income, [100.0 * income_factor])
    np.testing.assert_allclose(households.var.wealth, [200.0 * income_factor])
    assert households.var.factor_change == pytest.approx(cost_factor)

    if project_income:
        assert "POP.zarr" in opened_paths
    else:
        assert "POP.zarr" not in opened_paths


def test_update_monetary_variables_requires_climrisk_config() -> None:
    """Raise a clear error when the climrisk configuration is absent."""
    households = Households.__new__(Households)
    households.model = SimpleNamespace(config={"agent_settings": {}})

    with pytest.raises(ValueError, match="agent_settings.climrisk"):
        households.update_monetary_variables_to_ssp()
