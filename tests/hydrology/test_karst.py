"""Unit tests for karst capture, release and storage conservation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from geb.config_schema import KarstConfig
from geb.hydrology.karst import Karst
from geb.hydrology.landcovers import ALL_LAND_COVER_TYPES


@pytest.fixture
def karst() -> Karst:
    """Create a karst module with six land covers across two grid cells.

    Returns:
        A karst module with equal HRU areas and otherwise isolated dependencies.
    """
    model: MagicMock = MagicMock()
    model.config = {
        "hydrology": {
            "karst": {
                "capture_fraction": 1.0,
                "max_storage_m": None,
                "release_time_days": 10.0,
            }
        }
    }
    model.files = {"grid": {"groundwater/karst_fraction": "karst.zarr"}}
    model.in_spinup = True
    hydrology: MagicMock = MagicMock()
    hydrology.grid.var = SimpleNamespace()
    hydrology.grid.load2d.return_value = np.ones(2, dtype=np.float32)
    hydrology.HRU.var.land_use_type = np.array(ALL_LAND_COVER_TYPES)
    hydrology.to_HRU.side_effect = lambda *, data: np.repeat(data, 3)
    hydrology.to_grid.side_effect = lambda *, HRU_data: HRU_data.reshape(2, 3).mean(
        axis=1
    )
    return Karst(model, hydrology)


@pytest.mark.parametrize("coverage", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("capture_fraction", [0.0, 0.5, 1.0])
def test_karst_step(karst: Karst, coverage: float, capture_fraction: float) -> None:
    """Capture only eligible land flows and conserve water across both grids.

    Args:
        karst: Isolated karst module with all six land-cover types.
        coverage: Dimensionless karst coverage (0–1).
        capture_fraction: Dimensionless runoff and interflow capture (0–1).
    """
    # Different coverage and initial storage in each cell expose swapped indices.
    karst.karst_fraction[:] = [coverage, coverage / 2]
    karst.storage_m[:] = [0.01, 0.02]
    karst.model.config["hydrology"]["karst"]["capture_fraction"] = capture_fraction
    runoff_m: np.ndarray = np.full((24, 6), 0.0005, dtype=np.float32)
    interflow_m: np.ndarray = np.full((24, 6), 0.00025, dtype=np.float32)
    soil_m: np.ndarray = np.full(6, 0.002, dtype=np.float32)
    previous_storage_m: np.ndarray = karst.storage_m.copy()
    # Sealed and open-water HRUs (the last two) must retain all their input flows.
    karst_presence_cell_0: float = 1.0 if coverage > 0.0 else 0.0
    karst_presence_cell_1: float = 1.0 if (coverage / 2) > 0.0 else 0.0
    eligible_karst_fraction: np.ndarray = np.array(
        [
            karst_presence_cell_0,
            karst_presence_cell_0,
            karst_presence_cell_0,
            karst_presence_cell_1,
            0.0,
            0.0,
        ]
    )
    expected_capture_m: np.ndarray = 0.018 * eligible_karst_fraction * capture_fraction
    expected_soil_m: np.ndarray = 0.002 * eligible_karst_fraction
    expected_inflow_m: np.ndarray = np.array(
        [
            (0.018 * capture_fraction + 0.002) * karst_presence_cell_0,
            ((0.018 * capture_fraction + 0.002) * karst_presence_cell_1) / 3,
        ]
    )
    recharge_m: np.ndarray
    captured_m: np.ndarray
    captured_soil_m: np.ndarray
    overflow_m: np.ndarray
    recharge_m, captured_m, captured_soil_m, overflow_m = karst.step(
        runoff_m, interflow_m, soil_m
    )
    np.testing.assert_array_equal(overflow_m, np.zeros(2))
    assert overflow_m.dtype == np.float64

    np.testing.assert_allclose(captured_m, expected_capture_m, rtol=1e-6)
    np.testing.assert_allclose(captured_soil_m, expected_soil_m, rtol=1e-6)
    np.testing.assert_allclose(
        runoff_m,
        np.broadcast_to(
            0.0005 * (1 - eligible_karst_fraction * capture_fraction), (24, 6)
        ),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        interflow_m,
        np.broadcast_to(
            0.00025 * (1 - eligible_karst_fraction * capture_fraction), (24, 6)
        ),
        rtol=1e-6,
    )
    np.testing.assert_allclose(soil_m, 0.002 - expected_soil_m, rtol=1e-6)
    # Analytical solution for a one-day step with constant inflow and K = 10 days.
    expected_storage_m: np.ndarray = previous_storage_m * np.exp(-0.1) + (
        10.0 * expected_inflow_m * (1 - np.exp(-0.1))
    )
    np.testing.assert_allclose(karst.storage_m, expected_storage_m, rtol=1e-6)
    np.testing.assert_allclose(
        recharge_m,
        previous_storage_m + expected_inflow_m - expected_storage_m,
        rtol=1e-6,
    )
    assert recharge_m.dtype == np.float64
    assert karst.storage_m.dtype == np.float64
    assert captured_m.dtype == captured_soil_m.dtype == np.float32
    # Each cell represents three equal-area HRUs; all original water is accounted for.
    np.testing.assert_allclose(
        (runoff_m.sum(axis=0) + interflow_m.sum(axis=0) + soil_m)
        .reshape(2, 3)
        .mean(axis=1)
        + recharge_m
        + karst.storage_m,
        previous_storage_m + 0.020,
        rtol=1e-6,
    )


def test_karst_dry_recession(karst: Karst) -> None:
    """Release existing storage on dry days even when capture is disabled.

    Args:
        karst: Isolated karst module.
    """
    karst.model.config["hydrology"]["karst"]["capture_fraction"] = 0.0
    karst.storage_m[:] = [0.1, 0.2]
    initial_storage_m: np.ndarray = karst.storage_m.copy()
    cumulative_recharge_m: np.ndarray = np.zeros(2)
    for day in range(1, 31):
        recharge_m: np.ndarray = karst.step(
            np.zeros((24, 6), dtype=np.float32),
            np.zeros((24, 6), dtype=np.float32),
            np.zeros(6, dtype=np.float32),
        )[0]
        cumulative_recharge_m += recharge_m
        np.testing.assert_allclose(
            karst.storage_m, initial_storage_m * np.exp(-day / 10.0)
        )
        np.testing.assert_allclose(
            karst.storage_m + cumulative_recharge_m, initial_storage_m
        )


def test_karst_preserves_loaded_storage(karst: Karst) -> None:
    """Reuse checkpoint storage when initializing outside spinup.

    Args:
        karst: Isolated karst module.
    """
    karst.storage_m[:] = [0.1, 0.2]
    karst.model.in_spinup = False
    restored: Karst = Karst(karst.model, karst.hydrology)
    assert restored.storage_m is karst.storage_m
    np.testing.assert_array_equal(restored.storage_m, [0.1, 0.2])


@pytest.mark.parametrize(
    "settings",
    [None, {}, {"capture_fraction": 0.5}, {"release_time_days": 20.0}],
)
def test_karst_requires_config(karst: Karst, settings: dict[str, float] | None) -> None:
    """Require explicit karst settings when calculating recharge.

    Args:
        karst: Isolated karst module.
        settings: Incomplete settings, or None to omit the section.
    """
    karst.model.config = {"hydrology": {}}
    if settings is not None:
        karst.model.config["hydrology"]["karst"] = settings
    initialized: Karst = Karst(karst.model, karst.hydrology)
    with pytest.raises(KeyError):
        initialized.step(
            np.zeros((1, 6), dtype=np.float32),
            np.zeros((1, 6), dtype=np.float32),
            np.zeros(6, dtype=np.float32),
        )


@pytest.mark.parametrize(
    "settings", [{"capture_fraction": 1.1}, {"release_time_days": 0.0}]
)
def test_karst_rejects_invalid_config(karst: Karst, settings: dict[str, float]) -> None:
    """Reject invalid karst settings through the configuration schema.

    Args:
        karst: Isolated karst module.
        settings: Invalid capture fraction (dimensionless) or release time (days).
    """
    complete_settings: dict[str, float] = {
        "capture_fraction": 1.0,
        "release_time_days": 10.0,
        **settings,
    }
    with pytest.raises(ValueError):
        KarstConfig.model_validate(complete_settings)


def test_karst_runoff_capture_ceiling(karst: Karst) -> None:
    """Cap daily captured runoff and interflow at MAX_RUNOFF_CAPTURE_M_PER_DAY.

    Args:
        karst: Isolated karst module.
    """
    karst.karst_fraction[:] = [1.0, 1.0]
    karst.storage_m[:] = [0.0, 0.0]
    karst.model.config["hydrology"]["karst"]["capture_fraction"] = 1.0
    karst.model.config["hydrology"]["karst"]["max_storage_m"] = None

    # Total inflow 24 * 0.002 = 0.048 m (48 mm), exceeding the 0.025 m (25 mm) ceiling.
    runoff_m: np.ndarray = np.full((24, 6), 0.002, dtype=np.float32)
    interflow_m: np.ndarray = np.zeros((24, 6), dtype=np.float32)
    soil_m: np.ndarray = np.zeros(6, dtype=np.float32)

    recharge_m: np.ndarray
    captured_m: np.ndarray
    captured_soil_m: np.ndarray
    overflow_m: np.ndarray
    recharge_m, captured_m, captured_soil_m, overflow_m = karst.step(
        runoff_m, interflow_m, soil_m
    )

    # Eligible HRUs (0, 1, 2, 3) must be capped at exactly 0.025 m.
    # Ineligible HRUs (4: sealed, 5: open water) must capture 0.
    expected_capture: np.ndarray = np.array(
        [0.025, 0.025, 0.025, 0.025, 0.0, 0.0], dtype=np.float32
    )
    np.testing.assert_allclose(captured_m, expected_capture, rtol=1e-6)

    # Remaining runoff: 0.048 - 0.025 = 0.023 m per eligible HRU, evenly across 24 hours.
    np.testing.assert_allclose(
        runoff_m[:, :4],
        np.full((24, 4), 0.023 / 24.0, dtype=np.float32),
        rtol=1e-6,
    )
    # Ineligible HRUs retain all 0.048 m.
    np.testing.assert_allclose(
        runoff_m[:, 4:], np.full((24, 2), 0.002, dtype=np.float32), rtol=1e-6
    )


def test_karst_storage_capacity_overflow(karst: Karst) -> None:
    """Overflow storage exceeding max_storage_m to river discharge.

    Args:
        karst: Isolated karst module.
    """
    karst.karst_fraction[:] = [1.0, 0.0]
    karst.storage_m[:] = [0.04, 0.0]
    karst.model.config["hydrology"]["karst"]["capture_fraction"] = 1.0
    karst.model.config["hydrology"]["karst"]["max_storage_m"] = 0.05

    # Inflow of 0.02 m on cell 0.
    # Initial storage 0.04 m + inflow 0.02 m - recharge ~0.005 m > 0.05 m ceiling.
    runoff_m: np.ndarray = np.full((24, 6), 0.02 / 24.0, dtype=np.float32)
    interflow_m: np.ndarray = np.zeros((24, 6), dtype=np.float32)
    soil_m: np.ndarray = np.zeros(6, dtype=np.float32)

    recharge_m: np.ndarray
    captured_m: np.ndarray
    captured_soil_m: np.ndarray
    overflow_m: np.ndarray
    recharge_m, captured_m, captured_soil_m, overflow_m = karst.step(
        runoff_m, interflow_m, soil_m
    )

    # Cell 0 storage must be capped at exactly max_storage_m (0.05 m).
    assert karst.storage_m[0] == pytest.approx(0.05, abs=1e-7)
    assert overflow_m[0] > 0.0
    assert overflow_m[1] == 0.0

    # Total water conservation: initial_storage + inflow == storage + recharge + overflow.
    inflow_m: float = 0.02
    np.testing.assert_allclose(
        0.04 + inflow_m,
        karst.storage_m[0] + recharge_m[0] + overflow_m[0],
        rtol=1e-6,
    )
