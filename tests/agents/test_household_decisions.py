"""Test suite for household decision making module."""

import numpy as np
from pytest import fixture

from geb.agents.decision_module_flood import (
    DecisionModule,
)  # update to now decision model after merge


@fixture
def decision_template() -> None:
    """This function creates a decision template for the decision module"""
    geom_id = "<GeoID>"
    n_agents = 100
    return_periods = np.array([10, 50])
    income = np.random.randint(10_000, 20_000, n_agents)
    wealth = 1.5 * income
    expendature_cap = 0.1
    amenity_value = np.random.random(n_agents) * wealth
    amenity_weight = 1
    risk_perception = np.full(n_agents, 1)
    expected_damages_no_adapt = np.array(
        [np.full(n_agents, 5_000), np.full(n_agents, 15_000)]
    )
    damages_adapt = expected_damages_no_adapt * 0.8
    adaptation_costs = 100
    time_adapted = np.zeros(n_agents)
    loan_duration = 20
    T = 35
    r = 0.03
    sigma = 1
    adapted = np.full(n_agents, 0)

    decision_template = {
        "geom_id": geom_id,
        "n_agents": n_agents,
        "wealth": wealth,
        "income": income,
        "expendature_cap": expendature_cap,
        "amenity_value": amenity_value,
        "amenity_weight": amenity_weight,
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
    }

    return decision_template


def test_expenditure_cap(decision_template: dict) -> None:
    """This function tests the functionality of the expenditure cap.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
    """
    decision_module = DecisionModule(model=None, agents=None)

    # quick basic test
    EU_do_not_adapt = decision_module.calcEU_do_nothing(**decision_template)
    assert all(EU_do_not_adapt > -np.inf), (
        "Expected all EU_do_not_adapt values to be greater than -inf as there are no budget constraints"
    )

    # make sure an expendature cap results in no adaptation
    decision_template["expendature_cap"] = 0
    EU_adapt = decision_module.calcEU_adapt(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing(**decision_template)
    assert all(EU_adapt == -np.inf), "Expected all EU_adapt values to be -inf"
    assert all(EU_do_not_adapt > EU_adapt), (
        "Expected all EU_do_not_adapt values to be greater than EU_adapt"
    )
    assert all(EU_do_not_adapt > -np.inf), (
        "Expected all EU_do_not_adapt values to be greater than -inf. Should not be affected"
    )

    # make sure setting expenditure cap to high results in all agents being able to afford adaptation
    decision_template["expendature_cap"] = 10
    EU_adapt = decision_module.calcEU_adapt(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing(**decision_template)
    assert all(EU_adapt > -np.inf), (
        "Expected all EU_adapt values to be greater than -inf"
    )
    assert all(EU_do_not_adapt > -np.inf), (
        "Expected all EU_do_not_adapt values to be greater than -inf. Should not be affected"
    )


def test_risk_perception(decision_template: dict) -> None:
    """This function tests the functionality of risk perception.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.

    """
    decision_module = DecisionModule(model=None, agents=None)
    decision_template["expendature_cap"] = 10  # ensure all can adapt
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 0.01)
    EU_do_nothing_low_risk_perception = decision_module.calcEU_do_nothing(
        **decision_template
    )
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 10)
    EU_do_nothing_high_risk_perception = decision_module.calcEU_do_nothing(
        **decision_template
    )
    # make sure EU_do_nothing_high_risk_perception EU of adaptation is
    assert all(
        EU_do_nothing_low_risk_perception > EU_do_nothing_high_risk_perception
    ), "Expected EU_do_nothing_low to be greater than EU_do_nothing_high"


def test_damages(decision_template: dict) -> None:
    """This function tests the functionality of damages with and without adaptation.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
    """
    decision_module = DecisionModule(model=None, agents=None)

    # make sure all can adapt and behave rationally
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)
    # set damages under adaptation to zero
    decision_template["expected_damages_adapt"] *= 0

    # calculate EU
    EU_adapt = decision_module.calcEU_adapt(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing(**decision_template)
    assert all(EU_adapt > EU_do_not_adapt), (
        "Expected all EU_adapt values to be greater than EU_do_not_adapt"
    )

    # now check with no effect of adaptation on damage
    decision_template["expected_damages_adapt"] = decision_template["expected_damages"]
    EU_adapt_no_effect = decision_module.calcEU_adapt(**decision_template)
    assert all(EU_adapt_no_effect < EU_do_not_adapt), (
        "Expected all EU_adapt_no_effect values to be less than EU_do_not_adapt"
    )


def test_time_adapted(decision_template: dict) -> None:
    """This function tests the functionality of the time since adaptation decision.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
    """
    decision_module = DecisionModule(model=None, agents=None)
    # make sure all can adapt and behave rationally
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)

    # set damages to equal
    decision_template["expected_damages_adapt"] = decision_template["expected_damages"]
    # calculate EU
    EU_adapt = decision_module.calcEU_adapt(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing(**decision_template)
    assert all(EU_adapt < EU_do_not_adapt), (
        "Expected all EU_adapt values to be less than EU_do_not_adapt, as there is no damage reduction but adaptation costs are incurred."
    )

    # set time since adaptation to be the loan duration + 1
    decision_template["time_adapted"] = np.full(
        decision_template["n_agents"], decision_template["loan_duration"] + 1
    )
    decision_template["expected_damages_adapt"] = decision_template["expected_damages"]

    # calculate EU
    EU_adapt = decision_module.calcEU_adapt(**decision_template)
    EU_do_not_adapt = decision_module.calcEU_do_nothing(**decision_template)
    assert all(EU_adapt == EU_do_not_adapt), (
        "Expected all EU_adapt values to be equal to EU_do_not_adapt as there are no costs incurred anymore"
    )


def test_loan_duration(decision_template: dict) -> None:
    """This function tests the functionality of the loan duration.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
    """
    decision_module = DecisionModule(model=None, agents=None)
    # make sure all can adapt and behave rationally and set time discounting to zero
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)
    decision_template["r"] = 0

    # set loan duration short
    decision_template["loan_duration"] = 5
    EU_adapt_short = decision_module.calcEU_adapt(**decision_template)

    # set loan duration long
    decision_template["loan_duration"] = 10
    EU_adapt_long = decision_module.calcEU_adapt(**decision_template)

    assert all(EU_adapt_short > EU_adapt_long), (
        "Expected all EU_adapt_long values to be less than EU_adapt_short due to higher interest costs incurred"
    )


def test_time_discounting(decision_template: dict) -> None:
    """This function tests the functionality of time discounting.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
    """
    decision_module = DecisionModule(model=None, agents=None)
    # make sure all can adapt and behave rationally and set time discounting to zero
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)

    # set time discounting low
    decision_template["r"] = 0.01
    EU_adapt_low = decision_module.calcEU_adapt(**decision_template)

    # set time discounting high
    decision_template["r"] = 0.1
    EU_adapt_high = decision_module.calcEU_adapt(**decision_template)

    assert all(EU_adapt_low > EU_adapt_high), (
        "Expected all EU_adapt_high values to be less than EU_adapt_low due to higher time discounting"
    )


def test_decision_horizon(decision_template: dict) -> None:
    """This function tests the functionality of the decision horizon.

    Args:
        decision_template: A dictionary containing the parameters for the decision module.
    """
    decision_module = DecisionModule(model=None, agents=None)
    # make sure all can adapt and behave rationally and set time discounting to zero
    decision_template["expendature_cap"] = 10
    decision_template["risk_perception"] = np.full(decision_template["n_agents"], 1)

    # set time discounting low (no adaptation costs should be incurred in year zero)
    decision_template["T"] = 0
    EU_adapt_low = decision_module.calcEU_adapt(**decision_template)
    EU_do_not_adapt_low = decision_module.calcEU_do_nothing(**decision_template)
    assert all(EU_adapt_low == EU_do_not_adapt_low), (
        "Expected all EU_adapt_low values to be equal to EU_do_not_adapt_low as there are no costs incurred anymore"
    )
    # set time discounting high
    decision_template["T"] = 20
    EU_adapt_high = decision_module.calcEU_adapt(**decision_template)

    assert all(EU_adapt_low < EU_adapt_high), (
        "Expected all EU_adapt_high values to be greater than EU_adapt_low due to summation over longer time period"
    )
