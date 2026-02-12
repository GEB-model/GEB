"""Tests for interception functions in GEB."""

import math

import numpy as np

from geb.hydrology.interception import interception


def test_interception_no_rainfall_evaporation_only() -> None:
    """Test interception with no rainfall, only evaporation from existing storage."""
    rainfall_m = np.float32(0.0)
    storage_m = np.float32(0.001)  # 1mm storage
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.0005)  # 0.5mm potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # No throughfall when no rainfall
    assert throughfall == 0.0

    # Evaporation should be less than or equal to potential transpiration
    assert evaporation <= potential_transpiration_m

    # Evaporation should be less than or equal to storage
    assert evaporation <= storage_m

    # New storage should be reduced by evaporation
    assert new_storage == storage_m - evaporation

    # Water balance: input = output + storage change
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_rainfall_below_capacity() -> None:
    """Test interception when rainfall is less than remaining capacity."""
    rainfall_m = np.float32(0.001)  # 1mm rainfall
    storage_m = np.float32(0.0)  # No existing storage
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.0005)  # 0.5mm potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # No throughfall when rainfall < capacity
    assert throughfall == 0.0

    # All rainfall goes to storage
    assert new_storage == storage_m + rainfall_m - evaporation

    # Evaporation occurs from the intercepted water
    assert evaporation > 0.0
    assert evaporation <= potential_transpiration_m

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_rainfall_exceeds_capacity() -> None:
    """Test interception when rainfall exceeds interception capacity."""
    rainfall_m = np.float32(0.003)  # 3mm rainfall
    storage_m = np.float32(0.001)  # 1mm existing storage
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.0005)  # 0.5mm potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # Throughfall should be rainfall + storage - capacity
    expected_throughfall = rainfall_m + storage_m - capacity_m
    assert throughfall == expected_throughfall

    # Storage should be at capacity after evaporation
    assert new_storage == capacity_m - evaporation

    # Evaporation should occur
    assert evaporation > 0.0
    assert evaporation <= potential_transpiration_m

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_zero_capacity() -> None:
    """Test interception with zero interception capacity."""
    rainfall_m = np.float32(0.002)  # 2mm rainfall
    storage_m = np.float32(0.0)  # No storage
    capacity_m = np.float32(0.0)  # Zero capacity
    potential_transpiration_m = np.float32(0.001)  # 1mm potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # All rainfall becomes throughfall
    assert throughfall == rainfall_m

    # No evaporation from interception
    assert evaporation == 0.0

    # Storage remains zero
    assert new_storage == 0.0

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_zero_potential_evaporation() -> None:
    """Test interception with zero potential evaporation."""
    rainfall_m = np.float32(0.001)  # 1mm rainfall
    storage_m = np.float32(0.0)  # No storage
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.0)  # Zero potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # No throughfall
    assert throughfall == 0.0

    # No evaporation
    assert evaporation == 0.0

    # All rainfall goes to storage
    assert new_storage == rainfall_m

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_full_storage_no_rainfall() -> None:
    """Test interception when storage is already at capacity and no rainfall."""
    rainfall_m = np.float32(0.0)  # No rainfall
    storage_m = np.float32(0.002)  # Storage at capacity
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.001)  # 1mm potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # No throughfall
    assert throughfall == 0.0

    # Evaporation should be potential transpiration (since storage = capacity)
    assert evaporation == potential_transpiration_m

    # Storage reduced by evaporation
    assert new_storage == storage_m - evaporation

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_evaporation_formula() -> None:
    """Test that evaporation follows the correct formula."""
    rainfall_m = np.float32(0.0)  # No rainfall
    storage_m = np.float32(0.001)  # 1mm storage
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.002)  # 2mm potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # No throughfall
    assert throughfall == 0.0

    # Evaporation should follow the formula: min(storage, PE * (storage/capacity)^(2/3))
    expected_evaporation = min(
        storage_m,
        potential_transpiration_m * (storage_m / capacity_m) ** (2.0 / 3.0),
    )
    assert math.isclose(evaporation, expected_evaporation, abs_tol=1e-7)

    # Storage reduced by evaporation
    assert new_storage == storage_m - evaporation

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_large_rainfall() -> None:
    """Test interception with large rainfall amounts."""
    rainfall_m = np.float32(0.01)  # 10mm rainfall
    storage_m = np.float32(0.001)  # 1mm storage
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.001)  # 1mm potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # Throughfall should be large
    expected_throughfall = rainfall_m + storage_m - capacity_m
    assert throughfall == expected_throughfall

    # Storage should be at capacity after throughfall, then reduced by evaporation
    storage_after_throughfall = capacity_m  # Filled to capacity
    expected_evaporation = min(
        storage_after_throughfall,
        potential_transpiration_m
        * (storage_after_throughfall / capacity_m) ** (2.0 / 3.0),
    )
    assert math.isclose(evaporation, expected_evaporation, abs_tol=1e-7)
    assert math.isclose(
        new_storage, storage_after_throughfall - evaporation, abs_tol=1e-7
    )

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_edge_case_zero_storage_zero_rainfall() -> None:
    """Test interception with zero storage and zero rainfall."""
    rainfall_m = np.float32(0.0)
    storage_m = np.float32(0.0)
    capacity_m = np.float32(0.002)
    potential_transpiration_m = np.float32(0.001)

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # Everything should be zero
    assert throughfall == 0.0
    assert evaporation == 0.0
    assert new_storage == 0.0

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_evaporation_limited_by_storage() -> None:
    """Test that evaporation cannot exceed available storage."""
    rainfall_m = np.float32(0.0)
    storage_m = np.float32(0.0005)  # Small storage
    capacity_m = np.float32(0.002)
    potential_transpiration_m = np.float32(0.001)  # Large potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # No throughfall
    assert throughfall == 0.0

    # Evaporation should follow the formula and be limited by storage
    expected_evaporation = min(
        storage_m,
        potential_transpiration_m * (storage_m / capacity_m) ** (2.0 / 3.0),
    )
    assert math.isclose(evaporation, expected_evaporation, abs_tol=1e-7)

    # Storage becomes storage minus evaporation
    assert new_storage == storage_m - evaporation

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_storage_exceeds_capacity_initially() -> None:
    """Test interception when initial storage exceeds interception capacity."""
    rainfall_m = np.float32(0.001)  # 1mm rainfall
    storage_m = np.float32(0.003)  # 3mm storage (exceeds capacity)
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.0005)  # 0.5mm potential transpiration

    new_storage, throughfall, evaporation, remaining_potential_transpiration = (
        interception(
            rainfall_m=rainfall_m,
            storage_m=storage_m,
            capacity_m=capacity_m,
            potential_transpiration_m=potential_transpiration_m,
            potential_evaporation_m=potential_transpiration_m,
        )
    )

    # Throughfall should account for excess storage plus rainfall
    expected_throughfall = rainfall_m + storage_m - capacity_m
    assert throughfall == expected_throughfall

    # Storage after throughfall should be at capacity
    storage_after_throughfall = capacity_m
    assert math.isclose(
        new_storage, storage_after_throughfall - evaporation, abs_tol=1e-7
    )

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_leaf_area_index_to_interception_capacity_m() -> None:
    """Test the Von Hoyningen-Huene (1981) formula implementation.

    Formula: S_max (mm) = 0.935 + 0.498*LAI - 0.00575*LAI^2 for LAI > 0.1
    """
    from geb.hydrology.interception import leaf_area_index_to_interception_capacity_m

    # Test case 1: LAI <= 0.1 -> Capacity 0
    lai_low = np.array([0.05, 0.1, 0.0], dtype=np.float32)
    result_low = leaf_area_index_to_interception_capacity_m(lai_low)
    expected_low = np.zeros_like(lai_low)
    np.testing.assert_allclose(result_low, expected_low)

    # Test case 2: LAI = 5.0
    # S_max = 0.935 + 0.498*5 - 0.00575*25
    # S_max = 0.935 + 2.49 - 0.14375
    # S_max = 3.28125 mm -> 0.00328125 m
    lai_val = 5.0
    lai_arr = np.array([lai_val], dtype=np.float32)
    result = leaf_area_index_to_interception_capacity_m(lai_arr)

    expected_mm = 0.935 + 0.498 * lai_val - 0.00575 * (lai_val**2)
    expected_m = expected_mm / 1000.0

    np.testing.assert_allclose(
        result, np.array([expected_m], dtype=np.float32), rtol=1e-5
    )
