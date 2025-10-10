"""Tests for interception functions in GEB."""

import math

import numpy as np

from geb.hydrology.interception import interception


def test_interception_no_rainfall_evaporation_only() -> None:
    """Test interception with no rainfall, only evaporation from existing storage."""
    rainfall = np.float32(0.0)
    storage = np.float32(0.001)  # 1mm storage
    interception_capacity = np.float32(0.002)  # 2mm capacity
    potential_evaporation = np.float32(0.0005)  # 0.5mm potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # No throughfall when no rainfall
    assert throughfall == 0.0

    # Evaporation should be less than or equal to potential evaporation
    assert evaporation <= potential_evaporation

    # Evaporation should be less than or equal to storage
    assert evaporation <= storage

    # New storage should be reduced by evaporation
    assert new_storage == storage - evaporation

    # Water balance: input = output + storage change
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_rainfall_below_capacity() -> None:
    """Test interception when rainfall is less than remaining capacity."""
    rainfall = np.float32(0.001)  # 1mm rainfall
    storage = np.float32(0.0)  # No existing storage
    interception_capacity = np.float32(0.002)  # 2mm capacity
    potential_evaporation = np.float32(0.0005)  # 0.5mm potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # No throughfall when rainfall < capacity
    assert throughfall == 0.0

    # All rainfall goes to storage
    assert new_storage == storage + rainfall - evaporation

    # Evaporation occurs from the intercepted water
    assert evaporation > 0.0
    assert evaporation <= potential_evaporation

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_rainfall_exceeds_capacity() -> None:
    """Test interception when rainfall exceeds interception capacity."""
    rainfall = np.float32(0.003)  # 3mm rainfall
    storage = np.float32(0.001)  # 1mm existing storage
    interception_capacity = np.float32(0.002)  # 2mm capacity
    potential_evaporation = np.float32(0.0005)  # 0.5mm potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # Throughfall should be rainfall + storage - capacity
    expected_throughfall = rainfall + storage - interception_capacity
    assert throughfall == expected_throughfall

    # Storage should be at capacity after evaporation
    assert new_storage == interception_capacity - evaporation

    # Evaporation should occur
    assert evaporation > 0.0
    assert evaporation <= potential_evaporation

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_zero_capacity() -> None:
    """Test interception with zero interception capacity."""
    rainfall = np.float32(0.002)  # 2mm rainfall
    storage = np.float32(0.0)  # No storage
    interception_capacity = np.float32(0.0)  # Zero capacity
    potential_evaporation = np.float32(0.001)  # 1mm potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # All rainfall becomes throughfall
    assert throughfall == rainfall

    # No evaporation from interception
    assert evaporation == 0.0

    # Storage remains zero
    assert new_storage == 0.0

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_zero_potential_evaporation() -> None:
    """Test interception with zero potential evaporation."""
    rainfall = np.float32(0.001)  # 1mm rainfall
    storage = np.float32(0.0)  # No storage
    interception_capacity = np.float32(0.002)  # 2mm capacity
    potential_evaporation = np.float32(0.0)  # Zero potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # No throughfall
    assert throughfall == 0.0

    # No evaporation
    assert evaporation == 0.0

    # All rainfall goes to storage
    assert new_storage == rainfall

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_full_storage_no_rainfall() -> None:
    """Test interception when storage is already at capacity and no rainfall."""
    rainfall = np.float32(0.0)  # No rainfall
    storage = np.float32(0.002)  # Storage at capacity
    interception_capacity = np.float32(0.002)  # 2mm capacity
    potential_evaporation = np.float32(0.001)  # 1mm potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # No throughfall
    assert throughfall == 0.0

    # Evaporation should be potential evaporation (since storage = capacity)
    assert evaporation == potential_evaporation

    # Storage reduced by evaporation
    assert new_storage == storage - evaporation

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_evaporation_formula() -> None:
    """Test that evaporation follows the correct formula."""
    rainfall = np.float32(0.0)  # No rainfall
    storage = np.float32(0.001)  # 1mm storage
    interception_capacity = np.float32(0.002)  # 2mm capacity
    potential_evaporation = np.float32(0.002)  # 2mm potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # No throughfall
    assert throughfall == 0.0

    # Evaporation should follow the formula: min(storage, PE * (storage/capacity)^(2/3))
    expected_evaporation = min(
        storage,
        potential_evaporation * (storage / interception_capacity) ** (2.0 / 3.0),
    )
    assert math.isclose(evaporation, expected_evaporation, abs_tol=1e-7)

    # Storage reduced by evaporation
    assert new_storage == storage - evaporation

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_large_rainfall() -> None:
    """Test interception with large rainfall amounts."""
    rainfall = np.float32(0.01)  # 10mm rainfall
    storage = np.float32(0.001)  # 1mm storage
    interception_capacity = np.float32(0.002)  # 2mm capacity
    potential_evaporation = np.float32(0.001)  # 1mm potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # Throughfall should be large
    expected_throughfall = rainfall + storage - interception_capacity
    assert throughfall == expected_throughfall

    # Storage should be at capacity after throughfall, then reduced by evaporation
    storage_after_throughfall = interception_capacity  # Filled to capacity
    expected_evaporation = min(
        storage_after_throughfall,
        potential_evaporation
        * (storage_after_throughfall / interception_capacity) ** (2.0 / 3.0),
    )
    assert math.isclose(evaporation, expected_evaporation, abs_tol=1e-7)
    assert math.isclose(
        new_storage, storage_after_throughfall - evaporation, abs_tol=1e-7
    )

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_edge_case_zero_storage_zero_rainfall() -> None:
    """Test interception with zero storage and zero rainfall."""
    rainfall = np.float32(0.0)
    storage = np.float32(0.0)
    interception_capacity = np.float32(0.002)
    potential_evaporation = np.float32(0.001)

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # Everything should be zero
    assert throughfall == 0.0
    assert evaporation == 0.0
    assert new_storage == 0.0

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_evaporation_limited_by_storage() -> None:
    """Test that evaporation cannot exceed available storage."""
    rainfall = np.float32(0.0)
    storage = np.float32(0.0005)  # Small storage
    interception_capacity = np.float32(0.002)
    potential_evaporation = np.float32(0.001)  # Large potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # No throughfall
    assert throughfall == 0.0

    # Evaporation should follow the formula and be limited by storage
    expected_evaporation = min(
        storage,
        potential_evaporation * (storage / interception_capacity) ** (2.0 / 3.0),
    )
    assert math.isclose(evaporation, expected_evaporation, abs_tol=1e-7)

    # Storage becomes storage minus evaporation
    assert new_storage == storage - evaporation

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )


def test_interception_storage_exceeds_capacity_initially() -> None:
    """Test interception when initial storage exceeds interception capacity."""
    rainfall = np.float32(0.001)  # 1mm rainfall
    storage = np.float32(0.003)  # 3mm storage (exceeds capacity)
    interception_capacity = np.float32(0.002)  # 2mm capacity
    potential_evaporation = np.float32(0.0005)  # 0.5mm potential evaporation

    new_storage, throughfall, evaporation = interception(
        rainfall=rainfall,
        storage=storage,
        interception_capacity=interception_capacity,
        potential_evaporation=potential_evaporation,
    )

    # Throughfall should account for excess storage plus rainfall
    expected_throughfall = rainfall + storage - interception_capacity
    assert throughfall == expected_throughfall

    # Storage after throughfall should be at capacity
    storage_after_throughfall = interception_capacity
    assert new_storage == storage_after_throughfall - evaporation

    # Evaporation should be calculated from the storage at capacity
    expected_evaporation = min(
        storage_after_throughfall,
        potential_evaporation
        * (storage_after_throughfall / interception_capacity) ** (2.0 / 3.0),
    )
    assert math.isclose(evaporation, expected_evaporation, abs_tol=1e-7)

    # Water balance check
    assert math.isclose(
        rainfall, throughfall + evaporation + (new_storage - storage), abs_tol=1e-7
    )
