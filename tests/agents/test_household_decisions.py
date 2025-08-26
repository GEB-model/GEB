import numpy as np

from geb.agents.decision_module_flood import (
    DecisionModule,
)  # update to now decision model after merge


def create_decision_template():
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


def test_expenditure_cap() -> None:
    decision_module = DecisionModule(model=None, agents=None)
    decision_template = create_decision_template()

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


def test_risk_perception():
    decision_module = DecisionModule(model=None, agents=None)
    decision_template = create_decision_template()
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


def test_damages():
    decision_module = DecisionModule(model=None, agents=None)
    decision_template = create_decision_template()

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
