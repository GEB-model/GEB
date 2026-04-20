"""Tests for the insurers agent functions."""

import types
from typing import Any

import numpy as np
import pytest

from geb.agents.crop_farmers import INDEX_INSURANCE_ADAPTATION
from geb.agents.insurers import Insurers
from geb.store import DynamicArray


def make_insurers_stub() -> Insurers:
    """Create a minimal Insurers instance for unit tests.

    Returns:
        Insurers: Minimal insurers object with stubbed namespaces.
    """
    insurers = Insurers.__new__(Insurers)
    insurers.var = types.SimpleNamespace()
    insurers.agents = types.SimpleNamespace()
    insurers.agents.crop_farmers = types.SimpleNamespace()
    insurers.agents.crop_farmers.var = types.SimpleNamespace()
    return insurers


def test_premium_traditional_insurance() -> None:
    """Test the premium_traditional_insurance function."""
    insurers = make_insurers_stub()
    insurers.traditional_loading_rate = 1.0

    crop_farmers = insurers.agents.crop_farmers
    crop_farmers.field_size_per_farmer = np.array([10.0, 10.0], dtype=np.float32)
    crop_farmers.well_irrigated = np.array([0, 0], dtype=np.int32)

    def create_unique_groups(
        values: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Return a single test group for all farmers.

        Args:
            values: Grouping values, unused in this test stub.

        Returns:
            tuple[np.ndarray, int]: Group indices and number of groups.
        """
        return np.array([0, 0], dtype=np.int32), 1

    crop_farmers.create_unique_groups = create_unique_groups

    potential_insured_loss = np.array(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ],
        dtype=np.float32,
    )
    government_premium_cap = np.array([5.0, 1000.0], dtype=np.float32)
    masked_income = np.array(
        [
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )

    result = insurers.premium_traditional_insurance(
        potential_insured_loss,
        government_premium_cap,
        masked_income,
    )

    assert result.shape == (2,)
    assert result[0] == 5.0
    assert result[1] > result[0]


def test_insured_payouts_traditional() -> None:
    """Test the insured_payouts_traditional function."""
    insurers = make_insurers_stub()
    insurers.var.insured_yearly_income = np.zeros((2, 3), dtype=np.float32)

    masked_income = np.array(
        [
            [10.0, 8.0, 6.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    insured_farmers_mask = np.array([True, False])

    result = insurers.insured_payouts_traditional(
        insured_farmers_mask,
        masked_income,
    )

    cumsum = np.cumsum(masked_income, axis=1, dtype=float)
    years = np.arange(masked_income.shape[1])
    expected_threshold = np.empty_like(masked_income, dtype=float)
    first7 = years < 7
    expected_threshold[:, first7] = cumsum[:, first7] / (years[first7] + 1)
    expected_threshold = expected_threshold[:, ::-1]
    expected = np.maximum(expected_threshold - masked_income, 0.0)

    np.testing.assert_allclose(result, expected)
    assert insurers.var.insured_yearly_income[0, 0] == pytest.approx(expected[0, 0])
    assert insurers.var.insured_yearly_income[1, 0] == 0.0


def test_insured_payouts_index() -> None:
    """Test the insured_payouts_index function."""
    insurers = make_insurers_stub()
    insurers.var.insured_yearly_income = np.zeros((2, 2), dtype=np.float32)

    insurers.agents.crop_farmers.var.yearly_SPEI = types.SimpleNamespace(
        data=np.array(
            [
                [-0.5, -1.5],
                [-2.5, -1.0],
            ],
            dtype=np.float32,
        )
    )

    strike = np.array([-1.0, -1.0], dtype=np.float32)
    exit = np.array([-2.0, -2.0], dtype=np.float32)
    rate = np.array([100.0, 100.0], dtype=np.float32)
    insured_farmers_mask = np.array([True, False])
    valid_years = np.array([True, True])

    result = insurers.insured_payouts_index(
        strike,
        exit,
        rate,
        insured_farmers_mask,
        INDEX_INSURANCE_ADAPTATION,
        valid_years,
    )

    expected = np.array(
        [
            [0.0, 50.0],
            [100.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected)
    assert insurers.var.insured_yearly_income[0, 0] == pytest.approx(expected[0, 0])
    assert insurers.var.insured_yearly_income[1, 0] == 0.0


def test_insured_yields() -> None:
    """Test the insured_yields function."""
    insurers = make_insurers_stub()

    captured: dict[str, np.ndarray] = {}

    def fake_relation(
        insured_yearly_yield_ratio: np.ndarray,
        yearly_spei_probability: np.ndarray,
    ) -> np.ndarray:
        captured["yield_ratio"] = insured_yearly_yield_ratio.copy()
        captured["probability"] = yearly_spei_probability.copy()
        return insured_yearly_yield_ratio + yearly_spei_probability

    insurers.agents.crop_farmers.calculate_yield_spei_relation_group_lin = fake_relation
    insurers.agents.crop_farmers.var.yearly_SPEI_probability = types.SimpleNamespace(
        data=np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            dtype=np.float32,
        )
    )

    potential_insured_loss = np.array(
        [
            [30.0, 0.0],
            [10.0, 50.0],
        ],
        dtype=np.float32,
    )
    masked_income = np.array(
        [
            [80.0, 90.0],
            [95.0, 80.0],
        ],
        dtype=np.float32,
    )
    masked_potential_income = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
        ],
        dtype=np.float32,
    )
    valid_years = np.array([True, True])

    result = insurers.insured_yields(
        potential_insured_loss,
        valid_years,
        masked_income,
        masked_potential_income,
    )

    expected_yield_ratio = np.array(
        [
            [1.0, 0.9],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    expected_probability = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(captured["yield_ratio"], expected_yield_ratio)
    np.testing.assert_allclose(captured["probability"], expected_probability)
    np.testing.assert_allclose(result, expected_yield_ratio + expected_probability)


def test_premium_index_insurance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the premium_index_insurance function."""
    insurers = make_insurers_stub()

    def fake_compute(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return fixed contract-search outputs for testing.

        Args:
            *args: Positional arguments, unused in this stub.
            **kwargs: Keyword arguments, unused in this stub.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Strike indices, exit indices, rate indices, RMSE, and premiums.
        """
        return (
            np.array([0, 1], dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([10.0, 20.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        "geb.agents.insurers.compute_premiums_and_best_contracts_numba",
        fake_compute,
    )

    potential_insured_loss = np.ones((2, 3), dtype=np.float32)
    history = np.ones((2, 3), dtype=np.float32)
    gev_params = np.ones((2, 3), dtype=np.float32)
    strike_vals = np.array([1.0, 2.0], dtype=np.float32)
    exit_vals = np.array([0.1, 0.2], dtype=np.float32)
    rate_vals = np.array([100.0, 200.0], dtype=np.float32)
    government_premium_cap = np.array([100.0, 25.0], dtype=np.float32)
    valid_years = np.array([True, True, True])

    strike, exit, rate, premium = insurers.premium_index_insurance(
        potential_insured_loss=potential_insured_loss,
        history=history,
        gev_params=gev_params,
        strike_vals=strike_vals,
        exit_vals=exit_vals,
        rate_vals=rate_vals,
        government_premium_cap=government_premium_cap,
        loading_rate=np.float32(2.0),
        valid_years=valid_years,
    )

    np.testing.assert_allclose(strike, np.array([1.0, 2.0]))
    np.testing.assert_allclose(exit, np.array([0.2, 0.1]))
    np.testing.assert_allclose(rate, np.array([200.0, 200.0]))
    np.testing.assert_allclose(premium, np.array([20.0, 25.0]))


def test_insurance_premiums_traditional_dispatch() -> None:
    """Test that insurance_premiums dispatches to traditional_insurance."""
    insurers = make_insurers_stub()

    insurers.config = {"government_premium_cap": False}
    insurers.traditional_insurance_adaptation_active = True
    insurers.index_insurance_adaptation_active = False
    insurers.pr_insurance_adaptation_active = False

    insurers.var.insured_yearly_income = np.zeros((2, 2), dtype=np.float32)

    insurers.agents.crop_farmers.var.n = 2
    insurers.agents.crop_farmers.var.yearly_income = DynamicArray(
        n=2,
        max_n=2,
        extra_dims=(2,),
        extra_dims_names=("year",),
        dtype=np.float32,
        fill_value=0,
    )
    insurers.agents.crop_farmers.var.yearly_income[:] = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    )

    insurers.agents.crop_farmers.var.yearly_potential_income = DynamicArray(
        n=2,
        max_n=2,
        extra_dims=(2,),
        extra_dims_names=("year",),
        dtype=np.float32,
        fill_value=0,
    )

    insurers.agents.crop_farmers.var.yearly_potential_income[:] = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    )
    insurers.agents.crop_farmers.farmer_yield_probability_relation = np.zeros(
        (2, 2), dtype=np.float32
    )
    insurers.agents.crop_farmers.farmer_yield_probability_relation_budget_cap = (
        np.zeros((2, 2), dtype=np.float32)
    )

    insurers.potential_insured_loss = lambda: np.ones((2, 2), dtype=np.float32)

    expected_premium = np.array([1.0, 2.0], dtype=np.float32)
    expected_relation = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float32,
    )

    def fake_traditional_insurance(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed traditional-insurance outputs for testing.

        Args:
            *args: Positional arguments, unused in this stub.
            **kwargs: Keyword arguments, unused in this stub.

        Returns:
            tuple[np.ndarray, np.ndarray]: Premiums and insured relations.
        """
        return expected_premium, expected_relation

    insurers.traditional_insurance = fake_traditional_insurance

    premium, relation = insurers.insurance_premiums()

    np.testing.assert_allclose(premium, expected_premium)
    np.testing.assert_allclose(relation, expected_relation)


def test_insurance_premiums_index_dispatch() -> None:
    """Test that insurance_premiums dispatches to index_insurance."""
    insurers = make_insurers_stub()

    insurers.config = {"government_premium_cap": False}
    insurers.traditional_insurance_adaptation_active = False
    insurers.index_insurance_adaptation_active = True
    insurers.pr_insurance_adaptation_active = False

    insurers.var.insured_yearly_income = np.zeros((2, 2), dtype=np.float32)

    insurers.agents.crop_farmers.var.n = 2

    insurers.agents.crop_farmers.var.yearly_income = DynamicArray(
        n=2,
        max_n=2,
        extra_dims=(2,),
        extra_dims_names=("year",),
        dtype=np.float32,
        fill_value=0,
    )
    insurers.agents.crop_farmers.var.yearly_income[:] = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    )

    insurers.agents.crop_farmers.var.yearly_potential_income = DynamicArray(
        n=2,
        max_n=2,
        extra_dims=(2,),
        extra_dims_names=("year",),
        dtype=np.float32,
        fill_value=0,
    )

    insurers.agents.crop_farmers.var.yearly_potential_income[:] = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    )

    insurers.agents.crop_farmers.farmer_yield_probability_relation = np.zeros(
        (2, 2), dtype=np.float32
    )
    insurers.agents.crop_farmers.farmer_yield_probability_relation_budget_cap = (
        np.zeros((2, 2), dtype=np.float32)
    )

    insurers.potential_insured_loss = lambda: np.ones((2, 2), dtype=np.float32)

    expected_premium = np.array([3.0, 4.0], dtype=np.float32)
    expected_relation = np.array(
        [
            [0.5, 0.6],
            [0.7, 0.8],
        ],
        dtype=np.float32,
    )

    def fake_index_insurance(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed index-insurance outputs for testing.

        Args:
            *args: Positional arguments, unused in this stub.
            **kwargs: Keyword arguments, unused in this stub.

        Returns:
            tuple[np.ndarray, np.ndarray]: Premiums and insured relations.
        """
        return expected_premium, expected_relation

    insurers.index_insurance = fake_index_insurance

    premium, relation = insurers.insurance_premiums()

    np.testing.assert_allclose(premium, expected_premium)
    np.testing.assert_allclose(relation, expected_relation)


def test_insurance_premiums_pr_dispatch() -> None:
    """Test that insurance_premiums dispatches to pr_insurance."""
    insurers = make_insurers_stub()

    insurers.config = {"government_premium_cap": False}
    insurers.traditional_insurance_adaptation_active = False
    insurers.index_insurance_adaptation_active = False
    insurers.pr_insurance_adaptation_active = True

    insurers.var.insured_yearly_income = np.zeros((2, 2), dtype=np.float32)

    insurers.agents.crop_farmers.var.n = 2
    insurers.agents.crop_farmers.var.yearly_income = DynamicArray(
        n=2,
        max_n=2,
        extra_dims=(2,),
        extra_dims_names=("year",),
        dtype=np.float32,
        fill_value=0,
    )
    insurers.agents.crop_farmers.var.yearly_income[:] = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    )

    insurers.agents.crop_farmers.var.yearly_potential_income = DynamicArray(
        n=2,
        max_n=2,
        extra_dims=(2,),
        extra_dims_names=("year",),
        dtype=np.float32,
        fill_value=0,
    )

    insurers.agents.crop_farmers.var.yearly_potential_income[:] = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=np.float32,
    )
    insurers.agents.crop_farmers.farmer_yield_probability_relation = np.zeros(
        (2, 2), dtype=np.float32
    )
    insurers.agents.crop_farmers.farmer_yield_probability_relation_budget_cap = (
        np.zeros((2, 2), dtype=np.float32)
    )

    insurers.potential_insured_loss = lambda: np.ones((2, 2), dtype=np.float32)

    expected_premium = np.array([5.0, 6.0], dtype=np.float32)
    expected_relation = np.array(
        [
            [0.9, 1.0],
            [1.1, 1.2],
        ],
        dtype=np.float32,
    )

    def fake_pr_insurance(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed precipitation-insurance outputs for testing.

        Args:
            *args: Positional arguments, unused in this stub.
            **kwargs: Keyword arguments, unused in this stub.

        Returns:
            tuple[np.ndarray, np.ndarray]: Premiums and insured relations.
        """
        return expected_premium, expected_relation

    insurers.pr_insurance = fake_pr_insurance

    premium, relation = insurers.insurance_premiums()

    np.testing.assert_allclose(premium, expected_premium)
    np.testing.assert_allclose(relation, expected_relation)
