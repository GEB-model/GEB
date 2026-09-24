"""Tests for interception functions in GEB."""

import math

import numpy as np

from geb.hydrology.landsurface.interception import interception


def test_interception_no_rainfall_evaporation_only() -> None:
    """Test interception with no rainfall, only evaporation from existing storage."""
    rainfall_m = np.float32(0.0)
    storage_m = np.float32(0.001)  # 1mm storage
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.0005)  # 0.5mm potential transpiration

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
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
    """Test dynamic interception (Aston 1978 / LISFLOOD Eq. 2-6) when rainfall is below capacity."""
    rainfall_m = np.float32(0.001)  # 1mm rainfall
    storage_m = np.float32(0.0)  # No existing storage
    capacity_m = np.float32(0.002)  # 2mm capacity
    potential_transpiration_m = np.float32(0.0005)  # 0.5mm potential transpiration

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
    )

    # Under dynamic interception (Eq. 2-6), a fraction of rainfall penetrates as throughfall
    assert throughfall > 0.0
    assert throughfall < rainfall_m

    # Intercepted amount = rainfall - throughfall
    int_captured = rainfall_m - throughfall
    assert math.isclose(new_storage, int_captured - evaporation, abs_tol=1e-7)

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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
    )

    # Throughfall must account for unintercepted rainfall
    assert throughfall > 0.0
    int_captured = rainfall_m - throughfall
    # Capture cannot exceed available capacity (2mm - 1mm = 1mm)
    assert int_captured <= (capacity_m - storage_m) + 1e-7

    # Storage should be updated with captured water minus evaporation
    assert math.isclose(
        new_storage, storage_m + int_captured - evaporation, abs_tol=1e-7
    )

    # Evaporation should occur
    assert evaporation > 0.0
    assert evaporation <= potential_transpiration_m

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )

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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
    )

    # Dynamic throughfall
    assert throughfall > 0.0

    # No evaporation
    assert evaporation == 0.0

    # Captured rainfall goes to storage
    int_captured = rainfall_m - throughfall
    assert math.isclose(new_storage, int_captured, abs_tol=1e-7)

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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
    )

    # Throughfall should be large (majority of rain passes through)
    assert throughfall > 0.008

    int_captured = rainfall_m - throughfall
    storage_after_capture = storage_m + int_captured
    expected_evaporation = min(
        storage_after_capture,
        potential_transpiration_m * (storage_after_capture / capacity_m) ** (2.0 / 3.0),
    )
    assert math.isclose(evaporation, expected_evaporation, abs_tol=1e-7)
    assert math.isclose(new_storage, storage_after_capture - evaporation, abs_tol=1e-7)

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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
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

    (
        new_storage,
        throughfall,
        evaporation,
        remaining_potential_transpiration,
        remaining_potential_direct_evaporation,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_transpiration_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(2.5),
    )

    # Throughfall should account for excess storage plus rainfall
    expected_throughfall = rainfall_m + storage_m - capacity_m
    assert math.isclose(throughfall, expected_throughfall, abs_tol=1e-6)

    # Storage after throughfall should be at capacity
    storage_after_throughfall = capacity_m
    assert math.isclose(
        new_storage, storage_after_throughfall - evaporation, abs_tol=1e-7
    )

    # Water balance check
    assert math.isclose(
        rainfall_m, throughfall + evaporation + (new_storage - storage_m), abs_tol=1e-7
    )


def test_interception_budget_distribution() -> None:
    """Test that interception evaporation consumes budget from both potentials."""
    rainfall_m = np.float32(0.0)
    storage_m = np.float32(0.01)  # Large storage
    capacity_m = np.float32(0.01)
    potential_transpiration_m = np.float32(0.005)
    potential_bare_soil_m = np.float32(0.005)
    # Total potential budget is 0.01
    potential_evaporation_m = np.float32(0.01)

    (
        _,
        _,
        evaporation,
        remaining_pot_transp,
        remaining_pot_bare_soil,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_evaporation_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=potential_bare_soil_m,
        leaf_area_index=np.float32(5.0),
    )

    # Since potential_evaporation_m == total budget, and storage is full,
    # it should evaporate the full 0.01.
    assert math.isclose(evaporation, 0.01, abs_tol=1e-7)
    assert remaining_pot_transp == 0.0
    assert remaining_pot_bare_soil == 0.0


def test_interception_budget_distribution_partial() -> None:
    """Test that interception evaporation consumes budget from both potentials correctly (partial)."""
    rainfall_m = np.float32(0.0)
    storage_m = np.float32(0.01)  # Large storage
    capacity_m = np.float32(0.01)
    potential_transpiration_m = np.float32(0.005)
    potential_bare_soil_m = np.float32(0.005)
    # Total potential budget is 0.01, but we only have energy for 0.007
    potential_evaporation_m = np.float32(0.007)

    (
        _,
        _,
        evaporation,
        remaining_pot_transp,
        remaining_pot_bare_soil,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=potential_evaporation_m,
        potential_transpiration_m=potential_transpiration_m,
        potential_direct_evaporation_m=potential_bare_soil_m,
        leaf_area_index=np.float32(5.0),
    )

    # evaporation = 0.007
    # transp consumed = 0.005, remaining = 0.0
    # bare soil consumed = 0.002, remaining = 0.003
    assert math.isclose(evaporation, 0.007, abs_tol=1e-7)
    assert remaining_pot_transp == 0.0
    assert math.isclose(remaining_pot_bare_soil, 0.003, abs_tol=1e-7)


def test_lisflood_dynamic_interception_equation() -> None:
    """Test exact analytical values from LISFLOOD manual Eq. (2-6) and (2-8)."""
    lai = np.float32(4.0)
    # Eq. (2-8): k = 0.046 * LAI
    k = np.float32(0.046) * lai  # 0.184
    capacity_m = np.float32(0.002)  # 2 mm
    rainfall_m = np.float32(0.005)  # 5 mm
    storage_m = np.float32(0.0)

    # Eq. (2-6): Int = S_max * [1 - exp(-k * R * dt / S_max)]
    expected_int = capacity_m * (
        1.0 - math.exp(-float(k) * float(rainfall_m) / float(capacity_m))
    )
    expected_throughfall = float(rainfall_m) - expected_int

    (
        new_storage,
        throughfall,
        evaporation,
        _,
        _,
    ) = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=np.float32(0.0),
        potential_transpiration_m=np.float32(0.0),
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=lai,
    )

    assert math.isclose(throughfall, expected_throughfall, rel_tol=1e-5)
    assert math.isclose(new_storage, expected_int, rel_tol=1e-5)


def test_lisflood_interception_lai_density_effect() -> None:
    """Test that higher LAI (denser canopy) intercepts a higher fraction of rainfall."""
    rainfall_m = np.float32(0.005)  # 5 mm
    capacity_m = np.float32(0.003)  # 3 mm
    storage_m = np.float32(0.0)

    # Low LAI = 1.0 vs High LAI = 6.0
    _, tf_low, _, _, _ = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=np.float32(0.0),
        potential_transpiration_m=np.float32(0.0),
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(1.0),
    )

    _, tf_high, _, _, _ = interception(
        rainfall_m=rainfall_m,
        storage_m=storage_m,
        capacity_m=capacity_m,
        potential_interception_evaporation_m=np.float32(0.0),
        potential_transpiration_m=np.float32(0.0),
        potential_direct_evaporation_m=np.float32(0.0),
        leaf_area_index=np.float32(6.0),
    )

    # Higher LAI intercepts more water, leaving less throughfall
    assert tf_high < tf_low
