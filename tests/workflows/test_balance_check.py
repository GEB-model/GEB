"""Tests for the balance_check workflow function."""

import numpy as np
import pytest

from geb.workflows import balance_check


def test_balance_check_cellwise_balanced() -> None:
    """Test cellwise balance check when inputs and outputs are balanced."""
    influx: np.ndarray = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    outflux: np.ndarray = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    prestorage: np.ndarray = np.zeros(3, dtype=np.float32)
    poststorage: np.ndarray = np.zeros(3, dtype=np.float32)

    result = balance_check(
        name="test_balanced",
        how="cellwise",
        influxes=[influx],
        outfluxes=[outflux],
        prestorages=[prestorage],
        poststorages=[poststorage],
        tolerance=1e-5,
        raise_on_error=True,
    )
    assert result is True


def test_balance_check_cellwise_imbalance_index() -> None:
    """Test cellwise balance check reporting max imbalance index without raising."""
    influx: np.ndarray = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    outflux: np.ndarray = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    prestorage: np.ndarray = np.zeros(3, dtype=np.float32)
    poststorage: np.ndarray = np.zeros(3, dtype=np.float32)

    is_balanced, max_idx = balance_check(
        name="test_imbalance",
        how="cellwise",
        influxes=[influx],
        outfluxes=[outflux],
        prestorages=[prestorage],
        poststorages=[poststorage],
        tolerance=1e-5,
        raise_on_error=False,
        return_max_imbalance_index=True,
    )
    assert is_balanced is False
    assert max_idx == 2


def test_balance_check_cellwise_nan_returns_index_when_no_raise() -> None:
    """Test cellwise balance check with NaN values returns the NaN index when raise_on_error is False."""
    influx: np.ndarray = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    outflux: np.ndarray = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    prestorage: np.ndarray = np.zeros(3, dtype=np.float32)
    poststorage: np.ndarray = np.zeros(3, dtype=np.float32)

    is_balanced, nan_idx = balance_check(
        name="test_nan_cellwise",
        how="cellwise",
        influxes=[influx],
        outfluxes=[outflux],
        prestorages=[prestorage],
        poststorages=[poststorage],
        tolerance=1e-5,
        raise_on_error=False,
        return_max_imbalance_index=True,
    )
    assert is_balanced is False
    assert nan_idx == 1


def test_balance_check_cellwise_nan_raises_when_raise_on_error() -> None:
    """Test cellwise balance check raises ValueError on NaN when raise_on_error is True."""
    influx: np.ndarray = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    outflux: np.ndarray = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
    prestorage: np.ndarray = np.zeros(3, dtype=np.float32)
    poststorage: np.ndarray = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN values found in outflux component 1"):
        balance_check(
            name="test_nan_cellwise_raise",
            how="cellwise",
            influxes=[influx],
            outfluxes=[outflux],
            prestorages=[prestorage],
            poststorages=[poststorage],
            tolerance=1e-5,
            raise_on_error=True,
        )


def test_balance_check_sum_nan_handling() -> None:
    """Test sum balance check returns False when raise_on_error is False and raises when True."""
    influx: np.ndarray = np.array([1.0, np.nan], dtype=np.float32)
    outflux: np.ndarray = np.array([1.0, 2.0], dtype=np.float32)

    result: bool = balance_check(
        name="test_nan_sum_no_raise",
        how="sum",
        influxes=[influx],
        outfluxes=[outflux],
        tolerance=1e-5,
        raise_on_error=False,
    )
    assert result is False

    with pytest.raises(ValueError, match="Balance check failed, NaN values found"):
        balance_check(
            name="test_nan_sum_raise",
            how="sum",
            influxes=[influx],
            outfluxes=[outflux],
            tolerance=1e-5,
            raise_on_error=True,
        )
