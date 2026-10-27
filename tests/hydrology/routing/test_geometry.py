"""Tests for hydrological routing."""

import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from geb.geb_types import ArrayFloat32
from geb.hydrology.routing import (
    RoutingVariables,
)
from geb.hydrology.routing.geometry import (
    FLOODPLAIN_PROFILE_DEPTHS,
    NUM_FLOODPLAIN_LEVELS,
    build_inertial_geometry,
    compute_cfl_area_and_top_width,
    compute_cross_section_from_depth,
    compute_hydraulic_geometry_width,
    compute_overbank_depth_and_top_width,
    compute_static_geometry,
    compute_top_width_from_depth,
)


class DummyGridVar:
    """Mock grid variables for routing tests."""

    def __init__(self, n: int = 100) -> None:
        """Initialize mock grid variables.

        Args:
            n: Number of compressed grid cells.
        """
        self.n: int = n
        self.river_mannings: ArrayFloat32 = np.full(n, 0.03, dtype=np.float32)
        self.river_length: ArrayFloat32 = np.full(n, 1000.0, dtype=np.float32)
        self.waterbody_outflow_points: np.ndarray = np.full(n, -1, dtype=np.int32)
        self.waterbody_ids: np.ndarray = np.full(n, -1, dtype=np.int32)
        self.river_slope_m_per_m: ArrayFloat32 = np.full(n, 0.001, dtype=np.float32)
        self.groundwater_hydraulic_conductivity_m_per_day: np.ndarray = np.full(
            (1, n), 10.0, dtype=np.float32
        )


class DummyGrid:
    """Mock grid for routing tests."""

    def __init__(self, n: int = 100) -> None:
        """Initialize mock grid.

        Args:
            n: Number of compressed grid cells.
        """
        self.var = DummyGridVar(n)

    def load2d(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Mock loading 2D raster data.

        Returns:
            Dummy 2D float32 array.
        """
        return np.zeros(2, dtype=np.float32)

    def load3d(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Mock loading 3D raster data.

        Returns:
            Dummy 3D float32 array.
        """
        return np.full((10, 2), 100.0, dtype=np.float32)


def test_compute_cfl_area_and_top_width_and_overbank_depth() -> None:
    """Tests geometry derivation from stored volume for in-bank and overbank regimes."""
    reach_length_m = np.float32(1000.0)
    bankfull_width_m = np.float32(20.0)
    shape_exponent = np.float32(0.5)
    bankfull_depth_m = np.float32(2.0)
    floodplain_width_m = np.float32(80.0)

    (
        inverse_reach_length,
        inverse_shape_exponent_plus_one,
        bankfull_area_arr,
        bankfull_volume_arr,
        bankfull_perimeter_arr,
        width_over_sqrt_bankfull_depth_arr,
    ) = compute_static_geometry(
        river_length=np.array([reach_length_m], dtype=np.float32),
        river_width=np.array([bankfull_width_m], dtype=np.float32),
        shape_exponent=np.array([shape_exponent], dtype=np.float32),
        bankfull_depth=np.array([bankfull_depth_m], dtype=np.float32),
    )
    bankfull_area = bankfull_area_arr[0]
    bankfull_volume = bankfull_volume_arr[0]

    # Precompute a profile for testing compute_cfl_area_and_top_width
    test_fp_widths = np.array(
        [bankfull_width_m + 2.0 * 2.0 * d for d in FLOODPLAIN_PROFILE_DEPTHS],
        dtype=np.float32,
    )
    test_fp_areas = np.empty(len(FLOODPLAIN_PROFILE_DEPTHS), dtype=np.float32)
    test_fp_areas[0] = (
        0.5 * (bankfull_width_m + test_fp_widths[0]) * FLOODPLAIN_PROFILE_DEPTHS[0]
    )
    for k in range(1, len(FLOODPLAIN_PROFILE_DEPTHS)):
        test_fp_areas[k] = test_fp_areas[k - 1] + 0.5 * (
            test_fp_widths[k - 1] + test_fp_widths[k]
        ) * (FLOODPLAIN_PROFILE_DEPTHS[k] - FLOODPLAIN_PROFILE_DEPTHS[k - 1])

    # In-bank test at half bankfull volume (V = 13,333.33 m3)
    # Expected depth: h_bf * (0.5)^(1/1.5) = 2.0 * 0.5^(2/3) ≈ 1.25992 m
    expected_h = 2.0 * (0.5 ** (2.0 / 3.0))
    expected_top_width = 20.0 * ((expected_h / 2.0) ** 0.5)
    area_in, top_width_in = compute_cfl_area_and_top_width(
        volume_m3=np.float32(bankfull_volume * 0.5),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_reach_length=inverse_reach_length[0],
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area,
        bankfull_volume_m3=bankfull_volume,
        floodplain_side_slope_m_per_m=np.float32(2.0),
        floodplain_areas=test_fp_areas,
        floodplain_widths=test_fp_widths,
    )
    assert np.isclose(area_in, bankfull_area * 0.5, atol=1e-4)
    assert np.isclose(top_width_in, expected_top_width, atol=1e-4)

    # Overbank test without flat bench: valley side slopes (z = 2.0) rise from channel banks (W_bf = 20.0 m):
    # overbank_area = 20 m2
    expected_h_ob = float((-20.0 + math.sqrt(20.0**2 + 4.0 * 2.0 * 20.0)) / (2.0 * 2.0))
    expected_w_ob = float(20.0 + 4.0 * expected_h_ob)
    h_fp_ob, _ = compute_overbank_depth_and_top_width(
        overbank_area=np.float32(20.0),
        bankfull_width_m=bankfull_width_m,
        floodplain_areas=test_fp_areas,
        floodplain_widths=test_fp_widths,
        floodplain_side_slope_m_per_m=np.float32(2.0),
    )
    assert np.isclose(h_fp_ob, expected_h_ob, atol=1e-3)

    area_ob, top_width_ob = compute_cfl_area_and_top_width(
        volume_m3=np.float32(bankfull_volume + 20000.0),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_reach_length=inverse_reach_length[0],
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area,
        bankfull_volume_m3=bankfull_volume,
        floodplain_side_slope_m_per_m=np.float32(2.0),
        floodplain_areas=test_fp_areas,
        floodplain_widths=test_fp_widths,
    )
    assert np.isclose(area_ob, bankfull_area + 20.0, atol=1e-4)
    assert np.isclose(top_width_ob, expected_w_ob, atol=1e-4)

    # Overbank test WITH flat bench (W_fp = 30.0 m): total base width W_base = 20 + 30 = 50.0 m
    # Quadratic: 2 * h^2 + 50 * h - 20 = 0 -> h = (-50 + sqrt(2500 + 160)) / 4 ≈ 0.39384 m
    # Top width: W_base + 2 * z * h = 50.0 + 4 * h ≈ 51.5754 m
    expected_h_bench = float(
        (-50.0 + math.sqrt(50.0**2 + 4.0 * 2.0 * 20.0)) / (2.0 * 2.0)
    )
    expected_w_bench = float(50.0 + 4.0 * expected_h_bench)

    bench_fp_widths = np.array(
        [bankfull_width_m + 30.0 + 2.0 * 2.0 * d for d in FLOODPLAIN_PROFILE_DEPTHS],
        dtype=np.float32,
    )
    bench_fp_areas = np.empty(len(FLOODPLAIN_PROFILE_DEPTHS), dtype=np.float32)
    bench_fp_areas[0] = (
        0.5 * (bankfull_width_m + bench_fp_widths[0]) * FLOODPLAIN_PROFILE_DEPTHS[0]
    )
    for k in range(1, len(FLOODPLAIN_PROFILE_DEPTHS)):
        bench_fp_areas[k] = bench_fp_areas[k - 1] + 0.5 * (
            bench_fp_widths[k - 1] + bench_fp_widths[k]
        ) * (FLOODPLAIN_PROFILE_DEPTHS[k] - FLOODPLAIN_PROFILE_DEPTHS[k - 1])
    area_bench, top_width_bench = compute_cfl_area_and_top_width(
        volume_m3=np.float32(bankfull_volume + 20000.0),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_reach_length=inverse_reach_length[0],
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area,
        bankfull_volume_m3=bankfull_volume,
        floodplain_side_slope_m_per_m=np.float32(2.0),
        floodplain_areas=bench_fp_areas,
        floodplain_widths=bench_fp_widths,
    )
    assert np.isclose(area_bench, bankfull_area + 20.0, atol=1e-4)
    assert np.isclose(top_width_bench, expected_w_bench, atol=0.2)

    # Zero overbank area test -> depth is 0
    h_fp_zero, _ = compute_overbank_depth_and_top_width(
        overbank_area=np.float32(0.0),
        bankfull_width_m=bankfull_width_m,
        floodplain_areas=bench_fp_areas,
        floodplain_widths=bench_fp_widths,
        floodplain_side_slope_m_per_m=np.float32(2.0),
    )
    assert h_fp_zero == 0.0

    # Custom side-slope test (z = 4.0, i.e., 4:1 slope immediately from banks)
    # 4 * h^2 + 20 * h - 20 = 0 -> h = (-20 + sqrt(400 + 320)) / 8 ≈ 0.85410 m
    expected_h_z4 = float((-20.0 + math.sqrt(20.0**2 + 4.0 * 4.0 * 20.0)) / (2.0 * 4.0))
    expected_w_z4 = float(20.0 + 2.0 * 4.0 * expected_h_z4)

    z4_fp_widths = np.array(
        [bankfull_width_m + 2.0 * 4.0 * d for d in FLOODPLAIN_PROFILE_DEPTHS],
        dtype=np.float32,
    )
    z4_fp_areas = np.empty(len(FLOODPLAIN_PROFILE_DEPTHS), dtype=np.float32)
    z4_fp_areas[0] = (
        0.5 * (bankfull_width_m + z4_fp_widths[0]) * FLOODPLAIN_PROFILE_DEPTHS[0]
    )
    for k in range(1, len(FLOODPLAIN_PROFILE_DEPTHS)):
        z4_fp_areas[k] = z4_fp_areas[k - 1] + 0.5 * (
            z4_fp_widths[k - 1] + z4_fp_widths[k]
        ) * (FLOODPLAIN_PROFILE_DEPTHS[k] - FLOODPLAIN_PROFILE_DEPTHS[k - 1])

    h_fp_z4, _ = compute_overbank_depth_and_top_width(
        overbank_area=np.float32(20.0),
        bankfull_width_m=bankfull_width_m,
        floodplain_areas=z4_fp_areas,
        floodplain_widths=z4_fp_widths,
        floodplain_side_slope_m_per_m=np.float32(4.0),
    )
    assert np.isclose(h_fp_z4, expected_h_z4, atol=1e-3)
    area_z4, top_width_z4 = compute_cfl_area_and_top_width(
        volume_m3=np.float32(bankfull_volume + 20000.0),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_reach_length=inverse_reach_length[0],
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area,
        bankfull_volume_m3=bankfull_volume,
        floodplain_side_slope_m_per_m=np.float32(4.0),
        floodplain_areas=z4_fp_areas,
        floodplain_widths=z4_fp_widths,
    )
    assert np.isclose(area_z4, bankfull_area + 20.0, atol=1e-4)
    assert np.isclose(top_width_z4, expected_w_z4, atol=1e-4)


def test_compute_cross_section_from_depth() -> None:
    """Tests cross-sectional area, width, and perimeter evaluation at interface flow depths."""
    bankfull_width_m = np.float32(20.0)
    shape_exponent = np.float32(0.5)
    bankfull_depth_m = np.float32(2.0)
    floodplain_width_m = np.float32(50.0)

    (
        inverse_reach_length,
        inverse_shape_exponent_plus_one,
        bankfull_area_arr,
        bankfull_volume_arr,
        bankfull_perimeter_arr,
        width_over_sqrt_bankfull_depth_arr,
    ) = compute_static_geometry(
        river_length=np.array([1000.0], dtype=np.float32),
        river_width=np.array([bankfull_width_m], dtype=np.float32),
        shape_exponent=np.array([shape_exponent], dtype=np.float32),
        bankfull_depth=np.array([bankfull_depth_m], dtype=np.float32),
    )

    # In-bank interface depth (1.5 m)
    # Top width = 20 * (1.5 / 2.0)^0.5 = 20 * sqrt(0.75) ≈ 17.3205 m
    # Area = top_width * h / 1.5 = 17.3205 * 1.5 / 1.5 = 17.3205 m2
    area, top_width, p_wetted = compute_cross_section_from_depth(
        effective_depth_m=np.float32(1.5),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area_arr[0],
        bankfull_perimeter_m=bankfull_perimeter_arr[0],
        width_over_sqrt_bankfull_depth=width_over_sqrt_bankfull_depth_arr[0],
    )
    expected_w = 20.0 * math.sqrt(0.75)
    assert np.isclose(top_width, expected_w, atol=1e-4)
    assert np.isclose(area, expected_w, atol=1e-4)

    # Overbank interface depth (2.5 m -> 0.5 m overbank)
    # CaMa-Flood style: floodplain acts as dead storage; flow is carried by the main channel column.
    # W_bf=20.0, h_bf=2.0, effective_depth=2.5 -> floodplain_depth = 0.5 m
    # Channel flow area = bankfull_area + W_bf * floodplain_depth = (40.0 / 1.5) + 20.0 * 0.5 = 36.6667 m2
    # Top width = W_bf = 20.0 m; Wetted perimeter = bankfull_perimeter + 2 * floodplain_depth
    area_ob, top_width_ob, p_wetted_ob = compute_cross_section_from_depth(
        effective_depth_m=np.float32(2.5),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area_arr[0],
        bankfull_perimeter_m=bankfull_perimeter_arr[0],
        width_over_sqrt_bankfull_depth=width_over_sqrt_bankfull_depth_arr[0],
    )
    expected_area_ob = (40.0 / 1.5) + (20.0 * 0.5)
    expected_bankfull_rh = bankfull_area_arr[0] / bankfull_perimeter_arr[0]
    expected_p_wetted_ob = (expected_area_ob / expected_bankfull_rh) + np.float32(
        2.0 * 0.5
    )
    assert np.isclose(area_ob, expected_area_ob, atol=1e-4)
    assert np.isclose(top_width_ob, 20.0, atol=1e-4)
    assert np.isclose(p_wetted_ob, expected_p_wetted_ob, atol=1e-4)


def test_free_overfall_slope_bounded_by_bed_slope() -> None:
    """Tests that free overfall slope across elevation steps is bounded by bed slope."""
    bed_elev_node: np.float32 = np.float32(100.0)
    bed_elev_ds: np.float32 = np.float32(80.0)
    inv_interface_len: np.float32 = np.float32(1.0 / 2000.0)
    effective_depth: np.float32 = np.float32(1.0)

    bed_slope: np.float32 = (
        max(bed_elev_node - bed_elev_ds, np.float32(0.0)) * inv_interface_len
    )
    water_slope: np.float32 = -max(effective_depth * inv_interface_len, bed_slope)

    assert np.isclose(float(bed_slope), 0.01, atol=1e-5)
    assert np.isclose(float(water_slope), -0.01, atol=1e-5)


def test_lowland_mannings_n_calculation() -> None:
    """Tests that lowland Manning roughness scales down to the 0.020 floor for flat slopes."""
    slope: np.float32 = np.float32(0.0001)
    mannings_lowland: np.float32 = np.float32(0.020) + (
        np.float32(0.040 - 0.020) * (slope / np.float32(0.002))
    )
    raw_mannings: np.float32 = mannings_lowland
    clipped_n: np.float32 = np.clip(raw_mannings, np.float32(0.020), np.float32(0.075))
    assert 0.020 <= float(clipped_n) < 0.025
    assert np.isclose(float(clipped_n), 0.021, atol=1e-3)


def test_compute_cross_section_divided_channel_conveyance_monotonicity() -> None:
    """Tests that compound cross-section conveyance increases monotonically into overbank flow."""
    bankfull_width_m: np.float32 = np.float32(300.0)
    shape_exponent: np.float32 = np.float32(0.5)
    bankfull_depth_m: np.float32 = np.float32(4.0)
    floodplain_width_m: np.float32 = np.float32(1500.0)

    (
        inverse_reach_length,
        inverse_shape_exponent_plus_one,
        bankfull_area_arr,
        bankfull_volume_arr,
        bankfull_perimeter_arr,
        width_over_sqrt_bankfull_depth_arr,
    ) = compute_static_geometry(
        river_length=np.array([5000.0], dtype=np.float32),
        river_width=np.array([bankfull_width_m], dtype=np.float32),
        shape_exponent=np.array([shape_exponent], dtype=np.float32),
        bankfull_depth=np.array([bankfull_depth_m], dtype=np.float32),
    )

    depths: list[float] = [3.0, 3.8, 4.0, 4.1, 4.5, 5.0, 6.0]
    conveyances: list[float] = []

    for d in depths:
        area, top_width, p_wetted = compute_cross_section_from_depth(
            effective_depth_m=np.float32(d),
            bankfull_width_m=bankfull_width_m,
            shape_exponent=shape_exponent,
            bankfull_depth_m=bankfull_depth_m,
            inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
            bankfull_area_m2=bankfull_area_arr[0],
            bankfull_perimeter_m=bankfull_perimeter_arr[0],
            width_over_sqrt_bankfull_depth=width_over_sqrt_bankfull_depth_arr[0],
        )
        r_eff: float = float(area / p_wetted)
        conveyance: float = float(area * (r_eff ** (2.0 / 3.0)))
        conveyances.append(conveyance)

    # Conveyance must strictly increase as water level rises through and beyond bankfull
    for i in range(len(conveyances) - 1):
        assert conveyances[i + 1] > conveyances[i], (
            f"Conveyance at depth {depths[i + 1]} ({conveyances[i + 1]}) must exceed depth {depths[i]} ({conveyances[i]})"
        )


def test_compute_static_geometry_validation() -> None:
    """Tests that compute_static_geometry raises explicit ValueErrors for invalid inputs."""
    valid_length: ArrayFloat32 = np.array([1000.0], dtype=np.float32)
    valid_width: ArrayFloat32 = np.array([20.0], dtype=np.float32)
    valid_shape: ArrayFloat32 = np.array([0.5], dtype=np.float32)
    valid_depth: ArrayFloat32 = np.array([2.0], dtype=np.float32)
    valid_fp_width: ArrayFloat32 = np.array([50.0], dtype=np.float32)

    # Happy path succeeds
    results = compute_static_geometry(
        river_length=valid_length,
        river_width=valid_width,
        shape_exponent=valid_shape,
        bankfull_depth=valid_depth,
    )
    assert len(results) == 6

    # Happy path with zero floodplain width
    zero_fp_results = compute_static_geometry(
        river_length=valid_length,
        river_width=valid_width,
        shape_exponent=valid_shape,
        bankfull_depth=valid_depth,
    )
    assert len(zero_fp_results) == 6
    assert np.isclose(zero_fp_results[0][0], np.float32(1.0 / 1000.0))

    # Mismatched shapes
    with pytest.raises(ValueError, match="All input arrays must have the same shape"):
        compute_static_geometry(
            river_length=np.array([1000.0, 500.0], dtype=np.float32),
            river_width=valid_width,
            shape_exponent=valid_shape,
            bankfull_depth=valid_depth,
        )

    # Invalid river_length (NaN, inf, <= 0)
    for bad_length in [
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-10.0], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError, match="river_length must contain positive, finite numbers"
        ):
            compute_static_geometry(
                river_length=bad_length,
                river_width=valid_width,
                shape_exponent=valid_shape,
                bankfull_depth=valid_depth,
            )

    # Invalid river_width (NaN, <= 0)
    for bad_width in [
        np.array([np.nan], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-5.0], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError, match="river_width must contain positive, finite numbers"
        ):
            compute_static_geometry(
                river_length=valid_length,
                river_width=bad_width,
                shape_exponent=valid_shape,
                bankfull_depth=valid_depth,
            )

    # Invalid shape_exponent (NaN, <= 0)
    for bad_shape in [
        np.array([np.nan], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-0.5], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError, match="shape_exponent must contain positive, finite numbers"
        ):
            compute_static_geometry(
                river_length=valid_length,
                river_width=valid_width,
                shape_exponent=bad_shape,
                bankfull_depth=valid_depth,
            )

    # Invalid bankfull_depth (NaN, <= 0)
    for bad_depth in [
        np.array([np.nan], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([-1.0], dtype=np.float32),
    ]:
        with pytest.raises(
            ValueError, match="bankfull_depth must contain positive, finite numbers"
        ):
            compute_static_geometry(
                river_length=valid_length,
                river_width=valid_width,
                shape_exponent=valid_shape,
                bankfull_depth=bad_depth,
            )


class DummyModel:
    """Minimal mock model providing parameters config for Routing tests."""

    def __init__(self, mult: float = 1.0) -> None:
        """Initialize DummyModel.

        Args:
            mult: Bankfull discharge multiplier (dimensionless).
        """
        self.in_spinup: bool = False
        self.config: dict[str, Any] = {
            "parameters": {
                "bankfull_discharge_multiplier": mult,
                "mannings_n_multiplier": 1.0,
            }
        }
        self.files: dict[str, Any] = {
            "grid": {
                "routing/floodplain_profile_widths_m": "dummy",
                "routing/floodplain_side_slope_m_per_m": "dummy",
                "routing/bankfull_river_elevation_m": "dummy",
                "routing/floodplain_profile_widths_m": "dummy",
            }
        }
        self.grid = DummyGrid()


def test_routing_calculate_bankfull_depth_integration() -> None:
    """Test Routing.calculate_bankfull_depth with simulated Q2 and cold-start fallback.

    Verifies that:
    1. When self.var.rivers has 'return_period_2_years_daily_m3_per_s', simulated Q2 is mapped to grid.
    2. When the column is absent (cold-start spinup), empirical width inversion is used as fallback.
    3. Method 'power_law' also operates as expected.
    """
    from geb.hydrology.routing import Routing

    routing = object.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(4)  # ty:ignore[invalid-assignment]
    routing.config = {
        "max_threads": None,
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            },
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            },
        },
    }
    routing.default_missing_channel_width = 10.0

    var: RoutingVariables = RoutingVariables()
    # 4 cells: cell 0 (overland, river_id -1), cell 1 (river 0), cell 2 (river 1, narrow), cell 3 (river 1, wide)
    var.river_ids = np.array([-1, 0, 1, 1], dtype=np.int32)
    var.river_width_alpha = np.full(4, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(4, 0.5, dtype=np.float32)
    widths = np.array([10.0, 50.0, 100.0, 200.0], dtype=np.float32)
    var.observed_average_river_width = widths.copy()

    # Scenario 1: rivers GeoDataFrame has return_period_2_years_daily_m3_per_s and use_simulated_bankfull_q=True
    rivers_df: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [40.0, 200.0],
            "represented_in_grid": [True, True],
        },
        index=[0, 1],
    )
    var.rivers = rivers_df
    routing.var = var

    h_depth_simulated = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )

    # Cell 2 and Cell 3 both belong to river 1 (Q2 = 200 m3/s).
    # Cell 2 is width 100 m, Cell 3 is width 200 m:
    assert h_depth_simulated[2] > h_depth_simulated[3]
    assert np.isclose(h_depth_simulated[2] / h_depth_simulated[3], 2.0, rtol=1e-3)

    # Scenario 2: Year 1 / spinup cold-start (use_simulated_bankfull_q=False)
    var.rivers = gpd.GeoDataFrame({}, index=[0, 1])
    h_depth_fallback = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=False,
    )
    assert np.all(np.isfinite(h_depth_fallback))
    assert np.all(h_depth_fallback >= 0.1)

    # Scenario 3: Verify strict error handling (never silent fail if use_simulated_bankfull_q=True but Q2 is missing or null)
    import pytest

    # 3a. Column missing entirely raises KeyError directly (columns must exist)
    with pytest.raises(KeyError):
        routing.calculate_bankfull_depth(
            bankfull_top_width_m=widths,
            use_simulated_bankfull_q=True,
        )

    # 3b. Column contains null values raises ValueError
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [np.nan, 200.0],
            "represented_in_grid": [True, True],
        },
        index=[0, 1],
    )
    with pytest.raises(ValueError, match="contains null values"):
        routing.calculate_bankfull_depth(
            bankfull_top_width_m=widths,
            use_simulated_bankfull_q=True,
        )

    # Scenario 4: Continuity method respects min_depth_m floor even on zero discharge
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [0.0, 0.0],
            "represented_in_grid": [True, True],
        },
        index=[0, 1],
    )
    h_depth_zero_q = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )
    assert np.all(h_depth_zero_q[1:] >= 0.1)

    # Scenario 5: Unrealistic river depth exceeds 250m (deepest river in the world is Congo River at ~220-250m)
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [5000.0, 5000.0],
            "represented_in_grid": [True, True],
        },
        index=[0, 1],
    )
    unrealistic_widths = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds realistic bounds"):
        routing.calculate_bankfull_depth(
            bankfull_top_width_m=unrealistic_widths,
            use_simulated_bankfull_q=True,
        )


def test_routing_updates_rivers_geometry() -> None:
    """Verify that _update_rivers_geometry syncs grid width, depth, and manning to self.var.rivers."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(5)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()
    routing.var.river_width_alpha = np.full(5, 7.2, dtype=np.float32)
    routing.var.river_width_beta = np.full(5, 0.5, dtype=np.float32)
    routing.var.observed_average_river_width = np.full(5, np.nan, dtype=np.float32)

    rivers = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan],
            "depth": [np.nan, np.nan],
            "manning": [np.nan, np.nan],
            "represented_in_grid": [True, True],
        },
        index=[10, 20],
    )
    routing.var.rivers = rivers

    # Grid of 5 cells: cell 0, 1 -> reach 10; cell 2, 3 -> reach 20; cell 4 -> overland (-1)
    routing.var.river_ids = np.array([10, 10, 20, 20, -1], dtype=np.int32)
    width_grid = np.array([20.0, 30.0, 100.0, 100.0, 0.5], dtype=np.float32)
    depth_grid = np.array([2.0, 4.0, 6.0, 8.0, 0.1], dtype=np.float32)
    manning_grid = np.array([0.03, 0.05, 0.02, 0.04, 0.05], dtype=np.float32)

    floodplain_profile_widths_grid_m = np.zeros((5, 10), dtype=np.float32)
    floodplain_profile_widths_grid_m[0:2, 0] = 50.0
    floodplain_profile_widths_grid_m[2:4, 0] = 80.0
    floodplain_side_slope_grid_m_per_m = np.array(
        [2.0, 2.0, 3.0, 3.0, 2.0], dtype=np.float32
    )

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=floodplain_profile_widths_grid_m,
        floodplain_side_slope_grid_m_per_m=floodplain_side_slope_grid_m_per_m,
    )

    # Reach 10: mean width (20+30)/2 = 25, mean depth (2+4)/2 = 3, mean manning (0.03+0.05)/2 = 0.04
    assert np.isclose(routing.var.rivers.loc[10, "width"], 25.0)
    assert np.isclose(routing.var.rivers.loc[10, "depth"], 3.0)
    assert np.isclose(routing.var.rivers.loc[10, "manning"], 0.04)
    assert np.isclose(routing.var.rivers.loc[10, "floodplain_profile_width_0_m"], 50.0)
    assert np.isclose(routing.var.rivers.loc[10, "floodplain_side_slope_m_per_m"], 2.0)

    # Reach 20: mean width 100, mean depth 7, mean manning (0.02+0.04)/2 = 0.03
    assert np.isclose(routing.var.rivers.loc[20, "width"], 100.0)
    assert np.isclose(routing.var.rivers.loc[20, "depth"], 7.0)
    assert np.isclose(routing.var.rivers.loc[20, "manning"], 0.03)
    assert np.isclose(routing.var.rivers.loc[20, "floodplain_profile_width_0_m"], 80.0)
    assert np.isclose(routing.var.rivers.loc[20, "floodplain_side_slope_m_per_m"], 3.0)


def test_routing_updates_rivers_geometry_missing_data_raises() -> None:
    """Verify that an assertion error is raised when a represented river has missing geometry data."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(3)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()
    routing.var.river_width_alpha = np.full(3, 7.2, dtype=np.float32)
    routing.var.river_width_beta = np.full(3, 0.5, dtype=np.float32)
    routing.var.observed_average_river_width = np.full(3, np.nan, dtype=np.float32)

    rivers = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan],
            "depth": [np.nan, np.nan],
            "manning": [np.nan, np.nan],
            "represented_in_grid": [True, True],
        },
        index=[10, 30],
    )
    routing.var.rivers = rivers

    # Only reach 10 is in the grid; reach 30 is missing
    routing.var.river_ids = np.array([10, 10, -1], dtype=np.int32)
    width_grid = np.array([20.0, 30.0, 0.5], dtype=np.float32)
    depth_grid = np.array([2.0, 4.0, 0.1], dtype=np.float32)
    manning_grid = np.array([0.03, 0.05, 0.05], dtype=np.float32)

    with pytest.raises(
        AssertionError,
        match="All rivers represented in the grid must have width, depth, and manning defined",
    ):
        routing._update_rivers_geometry(
            width_grid=width_grid,
            depth_grid=depth_grid,
            manning_grid=manning_grid,
            floodplain_profile_widths_grid_m=np.zeros(
                (len(width_grid), 10), dtype=np.float32
            ),
            floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
        )


def test_routing_unobserved_channel_width_depth_estimation() -> None:
    """Verify that unobserved rivers get width and depth estimated via hydraulic geometry without distortion."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(3)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.36,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            },
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            },
        },
    }
    routing.default_missing_channel_width = 0.5

    var: RoutingVariables = RoutingVariables()
    # 3 cells: cell 0 (overland, -1), cell 1 (river 10, observed), cell 2 (river 20, unobserved)
    var.river_ids = np.array([-1, 10, 20], dtype=np.int32)
    var.river_width_alpha = np.full(3, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(3, 0.5, dtype=np.float32)

    # River 10 has observed width 50.0m; River 20 has NaN (unobserved)
    var.observed_average_river_width = np.array(
        [np.nan, 50.0, np.nan], dtype=np.float32
    )

    # Q2 = 100 m3/s for both rivers
    rivers_df: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [100.0, 100.0],
            "represented_in_grid": [True, True],
        },
        index=[10, 20],
    )
    var.rivers = rivers_df
    routing.var = var

    # Expected width: 7.2 * 100^0.5 = 72.0 m
    # Expected depth: 0.27 * 100^0.36 = 1.414 m
    q2_val: float = 100.0
    expected_w: float = 7.2 * (q2_val**0.5)
    expected_h: float = 0.27 * (q2_val**0.36)

    # For river 10 (observed width 50m < 72m), depth is scaled up: (72 / 50) * expected_h
    # For river 20 (unobserved), width is estimated as expected_w (72m), depth is exactly expected_h
    input_widths: ArrayFloat32 = np.array([0.5, 50.0, expected_w], dtype=np.float32)

    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=input_widths,
        use_simulated_bankfull_q=True,
    )

    observed_depth: float = float(depths[1])
    unobserved_depth: float = float(depths[2])

    # Parabolic centerline depth factor = (shape_exponent + 1) = 1.5 * mean_depth
    expected_centerline_h: float = 1.5 * expected_h
    assert np.isclose(
        observed_depth, (expected_w / 50.0) * expected_centerline_h, rtol=1e-3
    )
    assert np.isclose(unobserved_depth, expected_centerline_h, rtol=1e-3)


def test_routing_set_router_estimates_width_from_simulated_q2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that set_router estimates width from Q2 for unobserved channels when Q2 is present."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing = Routing.__new__(Routing)
    routing.config = {
        "max_threads": None,
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.36,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            },
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            },
        },
        "retention_basins": {"release_threshold_factor": 1.0},
        "riverbed_hydraulic_conductivity_multiplier": 0.1,
    }
    routing.default_missing_channel_width = 0.5
    routing.retention_max_storage_m3 = np.zeros(0, dtype=np.float32)
    routing.retention_basin_ids = np.full(2, -1, dtype=np.int32)
    routing.controlled_retention = np.zeros(0, dtype=bool)
    routing.river_network = None  # ty:ignore[invalid-assignment]
    routing._riverbed_hydraulic_conductivity_m_per_s = np.zeros(2, dtype=np.float32)

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([10, 20], dtype=np.int32)
    var.river_width_alpha = np.full(2, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(2, 0.5, dtype=np.float32)
    var.river_storage_alpha = np.ones(2, dtype=np.float32)
    var.river_storage_beta = np.full(2, 0.6, dtype=np.float32)
    var.water_stage_m = np.zeros(2, dtype=np.float32)
    var.river_storage_m3 = np.zeros(2, dtype=np.float64)
    # Cell 0 has observed width 30.0m, cell 1 is unobserved
    var.observed_average_river_width = np.array([30.0, np.nan], dtype=np.float32)

    # Both rivers have Q2 = 64 m3/s. Expected width: 7.2 * 64^0.5 = 7.2 * 8 = 57.6 m
    rivers_df: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [64.0, 64.0],
            "represented_in_grid": [True, True],
            "width": [np.nan, np.nan],
            "depth": [np.nan, np.nan],
            "manning": [np.nan, np.nan],
        },
        index=[10, 20],
    )
    var.rivers = rivers_df
    routing.var = var

    # Mock grid objects required by set_router
    class DummyGridVar:
        waterbody_outflow_points: np.ndarray = np.array([-1, -1], dtype=np.int32)
        river_mannings: ArrayFloat32 = np.array([0.03, 0.03], dtype=np.float32)
        river_slope_m_per_m: ArrayFloat32 = np.array([0.001, 0.001], dtype=np.float32)
        river_length: ArrayFloat32 = np.array([1000.0, 1000.0], dtype=np.float32)
        waterbody_ids: np.ndarray = np.array([-1, -1], dtype=np.int32)
        groundwater_hydraulic_conductivity_m_per_day: np.ndarray = np.array(
            [[10.0, 10.0]], dtype=np.float32
        )

    class DummyGrid:
        var = DummyGridVar()

        def load2d(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return np.zeros(2, dtype=np.float32)

        def load3d(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return np.full((10, 2), 100.0, dtype=np.float32)

    class DummyWaterbodiesVar:
        waterbody_outflow_linear_mapping = np.zeros(0, dtype=np.int32)
        lake_area = np.zeros(0, dtype=np.float32)
        lake_factor = np.zeros(0, dtype=np.float32)
        outflow_height = np.zeros(0, dtype=np.float32)
        storage = np.zeros(0, dtype=np.float32)

    class DummyWaterbodies:
        n = 0
        var = DummyWaterbodiesVar()

    class DummyHydrology:
        waterbodies = DummyWaterbodies()

    routing.grid = DummyGrid()  # ty:ignore[invalid-assignment]
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.hydrology = DummyHydrology()  # ty:ignore[invalid-assignment]

    class MockRouter:
        def initialize_stage(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "geb.hydrology.routing.LocalInertial",
        lambda **kwargs: MockRouter(),
    )

    # Call set_router
    routing.set_router()

    # Unobserved cell (index 1) should have estimated width 57.6m, observed cell should keep 30.0m
    expected_unobserved_w: float = 7.2 * (64.0**0.5)
    assert np.isclose(routing.var.river_width[0], 30.0)
    assert np.isclose(routing.var.river_width[1], expected_unobserved_w, rtol=1e-3)
    assert np.isclose(routing.var.rivers.loc[10, "width"], 30.0)
    assert np.isclose(
        routing.var.rivers.loc[20, "width"], expected_unobserved_w, rtol=1e-3
    )


def test_unrepresented_river_geometry_confluence() -> None:
    """Verify that unrepresented rivers at confluences combine bankfull Q and parameters."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(4)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 10 and 20 are represented; Reach 30 is unrepresented and receives both
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan, np.nan],
            "depth": [np.nan, np.nan, np.nan],
            "manning": [np.nan, np.nan, np.nan],
            "represented_in_grid": [True, True, False],
            "downstream_ID": [30, 30, -1],
            "shreve_stream_order": [1, 1, 2],
            "return_period_2_years_daily_m3_per_s": [50.0, 100.0, np.nan],
            "width_is_observed": [False, False, False],
        },
        index=[10, 20, 30],
    )
    routing.var.rivers = rivers

    # 4 cells: cell 0, 1 -> reach 10; cell 2, 3 -> reach 20
    routing.var.river_ids = np.array([10, 10, 20, 20], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([50.0, 50.0, 74.0, 74.0], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.0, 1.0, 1.5, 1.5], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03, 0.03, 0.04, 0.04], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.0, 7.0, 7.4, 7.4], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.full(4, np.nan, dtype=np.float32)

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # Reach 30 must have combined bankfull discharge Q = 50 + 100 = 150 m3/s
    q30: float = float(
        routing.var.rivers.loc[30, "return_period_2_years_daily_m3_per_s"]
    )
    assert np.isclose(q30, 150.0)

    # Discharge-weighted alpha: (50 * 7.0 + 100 * 7.4) / 150 = 7.2667
    alpha30: float = float(routing.var.rivers.loc[30, "river_width_alpha"])
    assert np.isclose(alpha30, 7.26667, rtol=1e-3)

    # Discharge-weighted Manning: (50 * 0.03 + 100 * 0.04) / 150 = 0.03667
    manning30: float = float(routing.var.rivers.loc[30, "manning"])
    assert np.isclose(manning30, 0.03667, rtol=1e-3)

    # Estimated width: alpha30 * 150^0.5
    w30: float = float(routing.var.rivers.loc[30, "width"])
    assert np.isclose(w30, alpha30 * (150.0**0.5), rtol=1e-3)

    # Estimated depth: 1.5 * 0.27 * 150^0.30 (parabolic centerline depth factor = 1.5)
    h30: float = float(routing.var.rivers.loc[30, "depth"])
    assert np.isclose(h30, 1.5 * 0.27 * (150.0**0.30), rtol=1e-3)


def test_unrepresented_river_geometry_iterative_cascade() -> None:
    """Verify that multi-step chains of unrepresented rivers are resolved iteratively."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(2)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 1 is represented, Reach 2 is unrepresented, Reach 3 is unrepresented (1 -> 2 -> 3)
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan, np.nan],
            "depth": [np.nan, np.nan, np.nan],
            "manning": [np.nan, np.nan, np.nan],
            "represented_in_grid": [True, False, False],
            "downstream_ID": [2, 3, -1],
            "shreve_stream_order": [1, 2, 3],
            "return_period_2_years_daily_m3_per_s": [64.0, np.nan, np.nan],
            "width_is_observed": [False, False, False],
        },
        index=[1, 2, 3],
    )
    routing.var.rivers = rivers

    routing.var.river_ids = np.array([1, 1], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([57.6, 57.6], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.2, 1.2], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03, 0.03], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.2, 7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5, 0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.full(2, np.nan, dtype=np.float32)

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # Both Reach 2 and Reach 3 should be resolved
    assert np.isclose(
        float(routing.var.rivers.loc[2, "return_period_2_years_daily_m3_per_s"]), 64.0
    )
    assert np.isclose(
        float(routing.var.rivers.loc[3, "return_period_2_years_daily_m3_per_s"]), 64.0
    )
    assert np.isclose(float(routing.var.rivers.loc[2, "width"]), 57.6)
    assert np.isclose(float(routing.var.rivers.loc[3, "width"]), 57.6)
    assert np.isclose(
        float(routing.var.rivers.loc[2, "depth"]), 1.5 * 0.27 * (64.0**0.30), rtol=1e-3
    )
    assert np.isclose(
        float(routing.var.rivers.loc[3, "depth"]), 1.5 * 0.27 * (64.0**0.30), rtol=1e-3
    )


def test_unrepresented_river_cascade_with_bankfull_multiplier() -> None:
    """Verify that unrepresented cascades do not compound bankfull multiplier across hops or distort weights."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    # Configure calibration multiplier to 2.0 to verify non-compounding
    routing.model = DummyModel(2.0)  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(2)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 1 is represented, Reach 2 is unrepresented, Reach 3 is unrepresented (1 -> 2 -> 3)
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan, np.nan],
            "depth": [np.nan, np.nan, np.nan],
            "manning": [np.nan, np.nan, np.nan],
            "represented_in_grid": [True, False, False],
            "downstream_ID": [2, 3, -1],
            "shreve_stream_order": [1, 2, 3],
            "return_period_2_years_daily_m3_per_s": [64.0, np.nan, np.nan],
            "width_is_observed": [False, False, False],
        },
        index=[1, 2, 3],
    )
    routing.var.rivers = rivers

    routing.var.river_ids = np.array([1, 1], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([57.6, 57.6], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.2, 1.2], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03, 0.03], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.2, 7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5, 0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.full(2, np.nan, dtype=np.float32)

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # 1. Raw summed Q (64.0 m3/s) must be propagated to both reach 2 and 3 without compounding to 128 or 256
    assert np.isclose(
        float(routing.var.rivers.loc[2, "return_period_2_years_daily_m3_per_s"]), 64.0
    )
    assert np.isclose(
        float(routing.var.rivers.loc[3, "return_period_2_years_daily_m3_per_s"]), 64.0
    )

    # 2. Weights must sum to 1.0 (not 0.5), so Manning roughness and alpha/beta must NOT be scaled down
    assert np.isclose(float(routing.var.rivers.loc[2, "manning"]), 0.03)
    assert np.isclose(float(routing.var.rivers.loc[3, "manning"]), 0.03)
    assert np.isclose(float(routing.var.rivers.loc[2, "river_width_alpha"]), 7.2)
    assert np.isclose(float(routing.var.rivers.loc[3, "river_width_alpha"]), 7.2)

    # 3. Geometry must apply the multiplier (2.0) exactly once: Q_bf = 64.0 * 2.0 = 128.0 m3/s
    expected_w: float = 7.2 * (128.0**0.5)
    expected_h: float = 1.5 * 0.27 * (128.0**0.30)
    assert np.isclose(float(routing.var.rivers.loc[2, "width"]), expected_w)
    assert np.isclose(float(routing.var.rivers.loc[3, "width"]), expected_w)
    assert np.isclose(float(routing.var.rivers.loc[2, "depth"]), expected_h, rtol=1e-3)
    assert np.isclose(float(routing.var.rivers.loc[3, "depth"]), expected_h, rtol=1e-3)


def test_unrepresented_river_isolated_no_upstream_retains_nan() -> None:
    """Verify that an unrepresented river without upstream segments retains NaN."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 10 is represented; Reach 99 is unrepresented and has NO upstream rivers
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan],
            "depth": [np.nan, np.nan],
            "manning": [np.nan, np.nan],
            "represented_in_grid": [True, False],
            "downstream_ID": [-1, -1],
            "shreve_stream_order": [1, 1],
            "return_period_2_years_daily_m3_per_s": [40.0, np.nan],
            "width_is_observed": [False, False],
        },
        index=[10, 99],
    )
    routing.var.rivers = rivers

    routing.var.river_ids = np.array([10], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([20.0], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.5], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.full(1, np.nan, dtype=np.float32)

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # Reach 10 is updated; Reach 99 has no upstream segments and must remain NaN
    assert not np.isnan(float(routing.var.rivers.loc[10, "width"]))
    assert np.isnan(float(routing.var.rivers.loc[99, "width"]))
    assert np.isnan(float(routing.var.rivers.loc[99, "depth"]))
    assert np.isnan(
        float(routing.var.rivers.loc[99, "return_period_2_years_daily_m3_per_s"])
    )


def test_unrepresented_river_confluence_with_isolated_headwater() -> None:
    """Verify that an unrepresented reach is resolved even if one upstream branch is an unrepresented headwater."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 10 is represented -> flows into 30
    # Reach 20 is unrepresented headwater (no upstream) -> flows into 30
    # Reach 30 is unrepresented confluence
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan, np.nan],
            "depth": [np.nan, np.nan, np.nan],
            "manning": [np.nan, np.nan, np.nan],
            "represented_in_grid": [True, False, False],
            "downstream_ID": [30, 30, -1],
            "shreve_stream_order": [1, 1, 2],
            "return_period_2_years_daily_m3_per_s": [49.0, np.nan, np.nan],
            "width_is_observed": [False, False, False],
        },
        index=[10, 20, 30],
    )
    routing.var.rivers = rivers

    routing.var.river_ids = np.array([10], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([50.4], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.5], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.full(1, np.nan, dtype=np.float32)

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # Reach 20 retains NaN (isolated headwater)
    assert np.isnan(float(routing.var.rivers.loc[20, "width"]))
    # Reach 30 is not blocked by 20 and gets resolved from 10
    assert not np.isnan(float(routing.var.rivers.loc[30, "width"]))
    assert np.isclose(float(routing.var.rivers.loc[30, "width"]), 50.4)
    assert np.isclose(
        float(routing.var.rivers.loc[30, "return_period_2_years_daily_m3_per_s"]), 49.0
    )


def test_unrepresented_river_inherits_single_upstream_observed_width() -> None:
    """Verify that an unrepresented river inherits its upstream observed width rather than calculating from Q."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 10 is represented with observed width; Reach 20 is unrepresented
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan],
            "depth": [np.nan, np.nan],
            "manning": [np.nan, np.nan],
            "represented_in_grid": [True, False],
            "downstream_ID": [20, -1],
            "shreve_stream_order": [1, 2],
            "return_period_2_years_daily_m3_per_s": [36.0, np.nan],
            "width_is_observed": [True, False],
        },
        index=[10, 20],
    )
    routing.var.rivers = rivers

    routing.var.river_ids = np.array([10], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([80.0], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.2], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.array([80.0], dtype=np.float32)

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # Reach 20 must inherit the exact observed width of 80.0m (instead of 7.2 * 36^0.5 = 43.2m)
    w20: float = float(routing.var.rivers.loc[20, "width"])
    assert np.isclose(w20, 80.0)
    assert bool(routing.var.rivers.loc[20, "width_is_observed"])

    # Depth should use continuity scaling: 1.5 * (43.2 / 80.0) * (0.27 * 36^0.30)
    w_expected: float = 7.2 * (36.0**0.5)
    h_expected: float = 0.27 * (36.0**0.30)
    expected_depth: float = 1.5 * (w_expected / 80.0) * h_expected
    h20: float = float(routing.var.rivers.loc[20, "depth"])
    assert np.isclose(h20, expected_depth, rtol=1e-3)


def test_unrepresented_river_confluence_both_observed_widths() -> None:
    """Verify that unrepresented confluence combines multiple upstream observed widths via power law."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(2)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 10 and 20 are represented with observed widths; Reach 30 is unrepresented confluence
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan, np.nan],
            "depth": [np.nan, np.nan, np.nan],
            "manning": [np.nan, np.nan, np.nan],
            "represented_in_grid": [True, True, False],
            "downstream_ID": [30, 30, -1],
            "shreve_stream_order": [1, 1, 2],
            "return_period_2_years_daily_m3_per_s": [50.0, 50.0, np.nan],
            "width_is_observed": [True, True, False],
        },
        index=[10, 20, 30],
    )
    routing.var.rivers = rivers

    routing.var.river_ids = np.array([10, 20], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([40.0, 30.0], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.0, 1.0], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03, 0.03], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.2, 7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5, 0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.array([40.0, 30.0], dtype=np.float32)

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # Power law combination: sqrt(40^2 + 30^2) = sqrt(1600 + 900) = sqrt(2500) = 50.0m
    w30: float = float(routing.var.rivers.loc[30, "width"])
    assert np.isclose(w30, 50.0)
    assert bool(routing.var.rivers.loc[30, "width_is_observed"])


def test_unrepresented_river_confluence_one_observed_one_unobserved() -> None:
    """Verify that an unrepresented confluence with one observed and one unobserved river combines widths and sets width_is_observed."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(2)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 10 has observed width 40m, Reach 20 is unobserved with estimated width 30m
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan, np.nan],
            "depth": [np.nan, np.nan, np.nan],
            "manning": [np.nan, np.nan, np.nan],
            "represented_in_grid": [True, True, False],
            "downstream_ID": [30, 30, -1],
            "shreve_stream_order": [1, 1, 2],
            "return_period_2_years_daily_m3_per_s": [50.0, 50.0, np.nan],
            "width_is_observed": [True, False, False],
        },
        index=[10, 20, 30],
    )
    routing.var.rivers = rivers

    routing.var.river_ids = np.array([10, 20], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([40.0, 30.0], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.0, 1.0], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03, 0.03], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.2, 7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5, 0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.array(
        [40.0, np.nan], dtype=np.float32
    )

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # Combined width: sqrt(40^2 + 30^2) = 50.0m
    w30: float = float(routing.var.rivers.loc[30, "width"])
    assert np.isclose(w30, 50.0)
    assert bool(routing.var.rivers.loc[30, "width_is_observed"])


def test_unrepresented_river_cascade_propagates_observed_width() -> None:
    """Verify that multi-step cascading unrepresented reaches propagate the observed width."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 1 is represented (observed 60m) -> Reach 2 (unrepresented) -> Reach 3 (unrepresented)
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan, np.nan],
            "depth": [np.nan, np.nan, np.nan],
            "manning": [np.nan, np.nan, np.nan],
            "represented_in_grid": [True, False, False],
            "downstream_ID": [2, 3, -1],
            "shreve_stream_order": [1, 2, 3],
            "return_period_2_years_daily_m3_per_s": [64.0, np.nan, np.nan],
            "width_is_observed": [True, False, False],
        },
        index=[1, 2, 3],
    )
    routing.var.rivers = rivers

    routing.var.river_ids = np.array([1], dtype=np.int32)
    width_grid: ArrayFloat32 = np.array([60.0], dtype=np.float32)
    depth_grid: ArrayFloat32 = np.array([1.2], dtype=np.float32)
    manning_grid: ArrayFloat32 = np.array([0.03], dtype=np.float32)
    routing.var.river_width_alpha = np.array([7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.array([60.0], dtype=np.float32)

    routing._update_rivers_geometry(
        width_grid=width_grid,
        depth_grid=depth_grid,
        manning_grid=manning_grid,
        floodplain_profile_widths_grid_m=np.zeros(
            (len(width_grid), 10), dtype=np.float32
        ),
        floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
    )

    # Both Reach 2 and Reach 3 should inherit the 60.0m observed width and be marked observed
    assert np.isclose(float(routing.var.rivers.loc[2, "width"]), 60.0)
    assert np.isclose(float(routing.var.rivers.loc[3, "width"]), 60.0)
    assert bool(routing.var.rivers.loc[2, "width_is_observed"])
    assert bool(routing.var.rivers.loc[3, "width_is_observed"])


def test_unrepresented_river_non_positive_width_raises_assertion() -> None:
    """Verify that non-positive upstream channel width raises an AssertionError."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    routing.var = RoutingVariables()

    # Reach 1 is represented with invalid width <= 0; Reach 2 is unrepresented
    rivers: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "width": [np.nan, np.nan],
            "depth": [np.nan, np.nan],
            "manning": [np.nan, np.nan],
            "represented_in_grid": [True, False],
            "downstream_ID": [2, -1],
            "shreve_stream_order": [1, 2],
            "return_period_2_years_daily_m3_per_s": [36.0, np.nan],
            "width_is_observed": [False, False],
        },
        index=[1, 2],
    )
    routing.var.rivers = rivers
    routing.var.river_ids = np.array([1], dtype=np.int32)
    routing.var.river_width_alpha = np.array([7.2], dtype=np.float32)
    routing.var.river_width_beta = np.array([0.5], dtype=np.float32)
    routing.var.observed_average_river_width = np.full(1, np.nan, dtype=np.float32)

    with pytest.raises(AssertionError, match="must be strictly positive"):
        routing._update_rivers_geometry(
            width_grid=np.array([0.0], dtype=np.float32),
            depth_grid=np.array([1.0], dtype=np.float32),
            manning_grid=np.array([0.03], dtype=np.float32),
            floodplain_profile_widths_grid_m=np.zeros((1, 10), dtype=np.float32),
            floodplain_side_slope_grid_m_per_m=np.array([2.0], dtype=np.float32),
        )


def test_parabolic_channel_geometry_and_mean_depth_conversion() -> None:
    """Verify that calculate_bankfull_depth scales mean channel depth to centerline depth.

    For a parabolic channel (shape_exponent r = 0.5), cross-sectional area is
    A = (2/3) * W * h_centerline. Bankfull mean depth is h_mean = A / W = (2/3) * h_centerline.
    Therefore, the maximum centerline depth must equal 1.5 * h_mean so that bankfull
    area exactly reproduces hydraulic geometry expectations.
    """
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 10.0

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([0], dtype=np.int32)
    var.river_width_alpha = np.array([7.2], dtype=np.float32)
    var.river_width_beta = np.array([0.5], dtype=np.float32)
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [100.0],
            "represented_in_grid": [True],
        },
        index=[0],
    )
    routing.var = var

    # Bankfull Q = 100 m3/s
    q_bf: float = 100.0
    w_expected: float = 7.2 * (q_bf**0.5)
    mean_depth_expected: float = 0.27 * (q_bf**0.30)

    # 1. Unobserved / matching width reach: W = w_expected
    widths: ArrayFloat32 = np.array([w_expected], dtype=np.float32)
    routing.var.observed_average_river_width = widths.copy()
    var.observed_average_river_width = widths.copy()
    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )

    # Centerline depth should be 1.5 * mean_depth for r = 0.5
    expected_centerline_depth: float = 1.5 * mean_depth_expected
    assert np.isclose(float(depths[0]), expected_centerline_depth, rtol=1e-3)

    # Parabolic cross-sectional area: A_bf = (2/3) * W * h_centerline
    # This must equal W * h_mean
    parabolic_area: float = (2.0 / 3.0) * float(widths[0]) * float(depths[0])
    expected_area: float = float(widths[0]) * mean_depth_expected
    assert np.isclose(parabolic_area, expected_area, rtol=1e-3)

    # 2. Narrow constriction reach (half width): mean depth doubles, centerline depth also doubles
    narrow_widths: ArrayFloat32 = np.array([0.5 * w_expected], dtype=np.float32)
    var.observed_average_river_width = narrow_widths.copy()
    narrow_depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=narrow_widths,
        use_simulated_bankfull_q=True,
    )
    assert np.isclose(
        float(narrow_depths[0]), 2.0 * expected_centerline_depth, rtol=1e-3
    )


def test_routing_calculate_bankfull_depth_continuity_scaling_with_observed_width() -> (
    None
):
    """Verify that calculate_bankfull_depth scales depth inversely with observed width.

    Tests that:
    1. Baseline expected width W_expected is computed using default hydraulic geometry parameters
       (default_alpha = 7.2, beta = 0.5), preventing double-counting or cancellation when
       reach-specific river_width_alpha has been scaled from low mean annual discharge.
    2. Wide reaches scale depth down proportionally (W_obs = 2 * W_expected -> depth = 0.5 * baseline).
    3. Narrow reaches scale depth up proportionally (W_obs = 0.5 * W_expected -> depth = 2.0 * baseline).
    4. Constrictions scale depth up proportionally without exploding to unphysical values.
    5. Fallback width inversion operates correctly during cold start when simulated Q2 is unavailable.
    """
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(4)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 10.0

    q2_flood: float = 100.0  # m3/s
    w_expected: float = 7.2 * (q2_flood**0.5)  # 72.0 m
    expected_mean_depth: float = 0.27 * (q2_flood**0.30)  # ~1.075 m
    expected_centerline_depth: float = 1.5 * expected_mean_depth  # ~1.612 m

    # 4 reaches:
    # Reach 0: matching expected width (72 m)
    # Reach 1: wide reach (2 * 72 = 144 m)
    # Reach 2: narrow reach (0.5 * 72 = 36 m)
    # Reach 3: constriction (0.25 * 72 = 18 m)
    widths: ArrayFloat32 = np.array(
        [w_expected, 2.0 * w_expected, 0.5 * w_expected, 0.25 * w_expected],
        dtype=np.float32,
    )

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    var.observed_average_river_width = widths.copy()
    # Reach-specific river_width_alpha values are inflated (e.g. scaled from low baseflow Q_mean during spinup)
    var.river_width_alpha = np.array([120.0, 180.0, 95.0, 150.0], dtype=np.float32)
    var.river_width_beta = np.full(4, 0.5, dtype=np.float32)
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [q2_flood] * 4,
            "represented_in_grid": [True] * 4,
        },
        index=[0, 1, 2, 3],
    )
    routing.var = var

    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )

    # 1. Matching width reach: depth equals baseline expected centerline depth
    assert np.isclose(float(depths[0]), expected_centerline_depth, rtol=1e-3)

    # 2. Wide reach (2x width): depth is halved (0.5x)
    assert np.isclose(float(depths[1]), 0.5 * expected_centerline_depth, rtol=1e-3)

    # 3. Narrow reach (0.5x width): depth is doubled (2.0x)
    assert np.isclose(float(depths[2]), 2.0 * expected_centerline_depth, rtol=1e-3)

    # 4. Constriction (0.25x width): depth is quadrupled (4.0x)
    assert np.isclose(float(depths[3]), 4.0 * expected_centerline_depth, rtol=1e-3)

    # 5. Verify flow area conservation: cross-sectional area A = (2/3) * W * h_centerline
    # must equal expected flow area W_expected * expected_mean_depth for all reaches
    expected_area: float = w_expected * expected_mean_depth
    for i in range(4):
        actual_area: float = (2.0 / 3.0) * float(widths[i]) * float(depths[i])
        assert np.isclose(actual_area, expected_area, rtol=1e-3)

    # 6. Cold-start fallback (use_simulated_bankfull_q = False): operates without error and produces positive finite depths
    fallback_depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=False,
    )
    assert np.all(np.isfinite(fallback_depths))
    assert np.all(fallback_depths >= 0.1)


def test_routing_calculate_bankfull_depth_flag_revert_machinery() -> None:
    """Verify that use_observed_width_as_bankfull flag toggles between scaling modes.

    Tests that:
    1. When use_observed_width_as_bankfull is True (default), expected top width W_expected
       is calculated using default_alpha (7.2), scaling depth inversely with observed width.
    2. When use_observed_width_as_bankfull is False, expected top width W_expected reverts
       to using reach-specific river_width_alpha, allowing easy rollback to previous machinery.
    3. Cold-start fallback width inversion proxy q_proxy similarly respects the flag.
    """
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 10.0

    q2_flood: float = 100.0  # m3/s
    w_obs: float = 72.0  # m
    widths: ArrayFloat32 = np.array([w_obs], dtype=np.float32)

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([0], dtype=np.int32)
    var.observed_average_river_width = widths.copy()
    # Inflated reach-specific river_width_alpha (e.g. 14.4, twice default_alpha)
    var.river_width_alpha = np.array([14.4], dtype=np.float32)
    var.river_width_beta = np.array([0.5], dtype=np.float32)
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [q2_flood],
            "represented_in_grid": [True],
        },
        index=[0],
    )
    routing.var = var

    # 1. Flag True (default): baseline expected width is default_alpha * Q2^beta = 7.2 * 10 = 72.0 m
    # continuity_ratio = 72.0 / 72.0 = 1.0
    depth_new: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )
    expected_depth_new: float = 1.5 * (0.27 * (q2_flood**0.30))
    assert np.isclose(float(depth_new[0]), expected_depth_new, rtol=1e-3)

    # 2. Flag False (revert machinery): expected width is reach alpha * Q2^beta = 14.4 * 10 = 144.0 m
    # continuity_ratio = 144.0 / 72.0 = 2.0 -> depth is doubled
    routing.config["river_depth"]["parameters"]["use_observed_width_as_bankfull"] = (
        False
    )
    depth_revert: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )
    assert np.isclose(float(depth_revert[0]), 2.0 * expected_depth_new, rtol=1e-3)

    # 3. Test cold-start fallback width inversion proxy q_proxy toggles properly
    fallback_revert: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=False,
    )
    assert np.all(np.isfinite(fallback_revert))
    assert np.all(fallback_revert >= 0.1)


def test_routing_missing_config_keys_raise_key_error() -> None:
    """Verify that calculate_bankfull_depth raises KeyError if required config keys are missing."""
    from geb.hydrology.routing import Routing

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    widths: ArrayFloat32 = np.array([50.0], dtype=np.float32)

    # Missing river_width entirely raises KeyError
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
    }
    with pytest.raises(KeyError):
        routing.calculate_bankfull_depth(widths, use_simulated_bankfull_q=False)

    # Missing use_observed_width_as_bankfull raises KeyError
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    with pytest.raises(KeyError):
        routing.calculate_bankfull_depth(widths, use_simulated_bankfull_q=False)


def test_routing_calculate_bankfull_width() -> None:
    """Test Routing.calculate_bankfull_width with simulated Q2 and cold start."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(3)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 10.0
    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([-1, 0, 1], dtype=np.int32)
    var.river_width_alpha = np.full(3, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(3, 0.5, dtype=np.float32)
    var.observed_average_river_width = np.array(
        [np.nan, 50.0, np.nan], dtype=np.float32
    )
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [100.0, 100.0],
            "represented_in_grid": [True, True],
        },
        index=[0, 1],
    )
    import types

    grid_var = types.SimpleNamespace(upstream_area=np.zeros(3, dtype=np.float32))
    routing.grid = types.SimpleNamespace(var=grid_var)  # ty:ignore[invalid-assignment]
    routing.var = var
    routing.floodplain_profile_widths_m = np.full((3, 10), 1000.0, dtype=np.float32)

    # Cold start (use_simulated_bankfull_q=False): observed width preserved, unobserved gets default_missing
    widths_cold: ArrayFloat32 = routing.calculate_bankfull_width(
        use_simulated_bankfull_q=False
    )
    assert widths_cold[0] == 10.0
    assert widths_cold[1] == 50.0
    assert widths_cold[2] == 10.0

    # Simulated Q2 (use_simulated_bankfull_q=True): cell 2 estimated as 7.2 * 100^0.5 = 72.0
    widths_q2: ArrayFloat32 = routing.calculate_bankfull_width(
        use_simulated_bankfull_q=True
    )
    assert widths_q2[0] == 10.0
    assert widths_q2[1] == 50.0
    assert np.isclose(float(widths_q2[2]), 72.0)


def test_routing_unobserved_river_width_and_depth_uses_parameters() -> None:
    """Verify that unobserved rivers always use alpha/beta for width and c/d for depth without continuity distortion."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 0.5
    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([42], dtype=np.int32)
    var.river_width_alpha = np.array([7.2], dtype=np.float32)
    var.river_width_beta = np.array([0.5], dtype=np.float32)
    # River 42 has NO observed width
    var.observed_average_river_width = np.array([np.nan], dtype=np.float32)
    q2_val: float = 110.71
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [q2_val],
            "represented_in_grid": [True],
        },
        index=[42],
    )
    routing.var = var
    routing.floodplain_profile_widths_m = np.full((1, 10), 1000.0, dtype=np.float32)

    # 1. Width calculation uses alpha * Q^beta = 7.2 * 110.71^0.5 = ~75.76 m
    widths: ArrayFloat32 = routing.calculate_bankfull_width(
        use_simulated_bankfull_q=True
    )
    expected_w: float = 7.2 * (q2_val**0.5)
    assert np.isclose(float(widths[0]), expected_w, rtol=1e-3)

    # For an unconstrained channel (bankfull_top_width_m == expected_w), depth is unscaled:
    depths_unconstrained: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )
    expected_mean_depth: float = 0.27 * (q2_val**0.30)
    expected_centerline_depth: float = 1.5 * expected_mean_depth
    assert np.isclose(
        float(depths_unconstrained[0]), expected_centerline_depth, rtol=1e-3
    )

    # 2. When bankfull width is constrained below expected width (e.g. by narrow floodplain),
    # depth scales inversely with width via hydraulic continuity to conserve bankfull conveyance area:
    constrained_widths: ArrayFloat32 = np.array([10.0], dtype=np.float32)
    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=constrained_widths,
        use_simulated_bankfull_q=True,
    )
    continuity_ratio: float = expected_w / 10.0
    assert np.isclose(
        float(depths[0]), expected_centerline_depth * continuity_ratio, rtol=1e-3
    )


def test_bankfull_discharge_multiplier_scaling() -> None:
    """Verifies that bankfull_discharge_multiplier scales bankfull discharge, width, and depth."""
    from geb.hydrology.routing import Routing, RoutingVariables

    def make_routing(mult: float) -> Routing:
        r: Routing = Routing.__new__(Routing)
        r.model = DummyModel(mult)  # ty:ignore[invalid-assignment]
        r.grid = DummyGrid(2)  # ty:ignore[invalid-assignment]
        r.config = {
            "river_width": {
                "parameters": {
                    "default_alpha": 7.2,
                    "beta": 0.5,
                },
            },
            "river_depth": {
                "parameters": {
                    "c": 0.27,
                    "d": 0.30,
                    "e": 0.0,
                    "min_depth_m": 0.1,
                    "shape_exponent": 0.5,
                    "use_observed_width_as_bankfull": False,
                },
            },
        }
        r.default_missing_channel_width = 0.5
        var = RoutingVariables()
        var.river_ids = np.array([10, 20], dtype=np.int32)
        var.river_width_alpha = np.array([7.2, 7.2], dtype=np.float32)
        var.river_width_beta = np.array([0.5, 0.5], dtype=np.float32)
        # Reach 10 has observed width 50m; Reach 20 is unobserved (NaN)
        var.observed_average_river_width = np.array([50.0, np.nan], dtype=np.float32)
        var.rivers = gpd.GeoDataFrame(
            {
                "return_period_2_years_daily_m3_per_s": [100.0, 100.0],
                "represented_in_grid": [True, True],
            },
            index=[10, 20],
        )
        r.var = var
        r.floodplain_profile_widths_m = np.full((2, 10), 1000.0, dtype=np.float32)
        return r

    r_base: Routing = make_routing(1.0)
    w_base: ArrayFloat32 = r_base.calculate_bankfull_width(
        use_simulated_bankfull_q=True
    )
    d_base: ArrayFloat32 = r_base.calculate_bankfull_depth(
        bankfull_top_width_m=w_base, use_simulated_bankfull_q=True
    )

    # Multiplier = 2.0
    r_high: Routing = make_routing(2.0)
    w_high: ArrayFloat32 = r_high.calculate_bankfull_width(
        use_simulated_bankfull_q=True
    )
    d_high: ArrayFloat32 = r_high.calculate_bankfull_depth(
        bankfull_top_width_m=w_high, use_simulated_bankfull_q=True
    )

    # Multiplier = 0.5
    r_low: Routing = make_routing(0.5)
    w_low: ArrayFloat32 = r_low.calculate_bankfull_width(use_simulated_bankfull_q=True)
    d_low: ArrayFloat32 = r_low.calculate_bankfull_depth(
        bankfull_top_width_m=w_low, use_simulated_bankfull_q=True
    )

    # 1. Observed width reach (index 0): width remains strictly observed (50m) across all multipliers
    assert np.isclose(w_base[0], 50.0)
    assert np.isclose(w_high[0], 50.0)
    assert np.isclose(w_low[0], 50.0)

    # For observed reach, depth scales with (Q_mult)^(beta + d) = (Q_mult)^0.80
    assert np.isclose(d_high[0] / d_base[0], 2.0 ** (0.5 + 0.30), rtol=1e-3)
    assert np.isclose(d_low[0] / d_base[0], 0.5 ** (0.5 + 0.30), rtol=1e-3)

    # 2. Unobserved width reach (index 1): width scales with Q_mult^beta = Q_mult^0.5
    assert np.isclose(w_high[1] / w_base[1], 2.0**0.5, rtol=1e-3)
    assert np.isclose(w_low[1] / w_base[1], 0.5**0.5, rtol=1e-3)

    # For unobserved reach, depth scales with Q_mult^d = Q_mult^0.30
    assert np.isclose(d_high[1] / d_base[1], 2.0**0.30, rtol=1e-3)
    assert np.isclose(d_low[1] / d_base[1], 0.5**0.30, rtol=1e-3)


def test_load_rivers_prunes_starved_channels_and_updates_represented_in_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that load_rivers prunes starved channels downstream of non-outflow waterbody cells."""
    import pyflwdir
    from shapely.geometry import Point

    from geb.hydrology.routing import Routing
    from geb.workflows.io import read_geom, write_geom

    routing = Routing.__new__(Routing)

    # 1x4 grid:
    # (0, 0): Lake 5 (non-outflow), drains East (6)
    # (0, 1): River reach 20, drains East (6)
    # (0, 2): River reach 20 (outlet pit 5)
    # (0, 3): River reach 30 (outlet pit 5, not starved)
    wb_id_2d = np.array([[5, -1, -1, -1]], dtype=np.int32)
    wb_outflows_2d = np.array([[-1, -1, -1, -1]], dtype=np.int32)
    river_ids_2d = np.array([[-1, 20, 20, 30]], dtype=np.int32)
    ldd_2d = np.array([[6, 6, 5, 5]], dtype=np.uint8)

    routing.river_network = pyflwdir.from_array(ldd_2d, ftype="ldd")

    class MockGrid:
        def load2d(self, file_path: Any, compress: bool = False) -> np.ndarray:
            if "waterbody_id" in str(file_path):
                return wb_id_2d
            elif "waterbody_outflow_points" in str(file_path):
                return wb_outflows_2d
            elif "river_ids" in str(file_path):
                return river_ids_2d.copy()
            raise ValueError(f"Unknown file {file_path}")

        def compress(self, arr: np.ndarray) -> np.ndarray:
            return arr.ravel()

    routing.grid = MockGrid()  # ty:ignore[invalid-assignment]

    class MockModel:
        import logging

        logger = logging.getLogger("test")
        files = {
            "grid": {
                "waterbodies/waterbody_id": "waterbody_id",
                "waterbodies/waterbody_outflow_points": "waterbody_outflow_points",
                "routing/river_ids": "river_ids",
            },
            "geom": {
                "waterbodies/waterbody_data": "waterbody_data",
                "routing/rivers": "rivers",
            },
        }

    routing.model = MockModel()  # ty:ignore[invalid-assignment]

    wb_data = gpd.GeoDataFrame(
        {"waterbody_id": [5], "waterbody_type": [1]},
        geometry=[Point(0, 0)],
    )
    rivers_data = gpd.GeoDataFrame(
        {
            "width": [10.0, 15.0],
            "hydrography_xy": [[(1, 0), (2, 0)], [(3, 0)]],
            "hydrography_upstream_area_m2": [[1000.0, 2000.0], [5000.0]],
        },
        index=[20, 30],
        geometry=[Point(1, 0), Point(3, 0)],
    )

    def mock_read_geom(path: Any) -> gpd.GeoDataFrame:
        if "waterbody_data" in str(path):
            return wb_data.copy()
        elif "rivers" in str(path):
            return rivers_data.copy()
        raise ValueError(f"Unknown geom path: {path}")

    monkeypatch.setattr("geb.hydrology.routing.read_geom", mock_read_geom)

    grid_linear_mapping = np.array([[0, 1, 2, 3]], dtype=np.int32)
    rivers, river_ids_comp, river_ids_orig = routing.load_rivers(grid_linear_mapping)

    # Reach 20 is starved, so it should not be represented in the grid
    assert not rivers.loc[20, "represented_in_grid"]
    assert len(rivers.loc[20, "hydrography_xy"]) == 0
    assert len(rivers.loc[20, "hydrography_upstream_area_m2"]) == 0
    # Reach 30 is active, so it should be represented in the grid
    assert rivers.loc[30, "represented_in_grid"]
    assert rivers.loc[30, "hydrography_xy"] == [(3, 0)]
    assert (rivers.loc[30, "hydrography_upstream_area_m2"] == [5000.0]).all()
    # The river IDs grid must have had reach 20 pruned to -1, but reach 30 retained
    assert (river_ids_comp[:3] == -1).all()
    assert river_ids_comp[3] == 30

    # Ensure rivers can be saved to geoparquet and restored without PyArrow serialization errors
    saved_parquet: Path = tmp_path / "rivers.geoparquet"
    write_geom(rivers, saved_parquet)
    restored_rivers: gpd.GeoDataFrame = read_geom(saved_parquet)
    assert len(restored_rivers.loc[20, "hydrography_xy"]) == 0
    assert len(restored_rivers.loc[30, "hydrography_xy"]) == 1


def test_compute_overbank_depth_and_top_width_10_point_profile() -> None:
    """Tests 10-point floodplain profile piecewise linear lookup and quadratic extrapolation."""
    bankfull_width_m = np.float32(20.0)
    fp_widths = np.array(
        [22.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0],
        dtype=np.float32,
    )
    side_slope_z = np.float32(2.0)

    # Compute expected cumulative areas at each of the 10 depth levels
    areas = np.zeros(10, dtype=np.float32)
    depths = np.array(FLOODPLAIN_PROFILE_DEPTHS, dtype=np.float32)
    areas[0] = np.float32(0.5) * (bankfull_width_m + fp_widths[0]) * depths[0]
    for k in range(1, 10):
        areas[k] = areas[k - 1] + np.float32(0.5) * (
            fp_widths[k - 1] + fp_widths[k]
        ) * (depths[k] - depths[k - 1])

    # Zero or negative area -> inbank
    h_zero, w_zero = compute_overbank_depth_and_top_width(
        np.float32(0.0), bankfull_width_m, areas, fp_widths, side_slope_z
    )
    assert h_zero == 0.0
    assert w_zero == bankfull_width_m

    # Exactly at each depth level k, depth must be depths[k] and top width fp_widths[k]
    for k in range(10):
        h_k, w_k = compute_overbank_depth_and_top_width(
            areas[k], bankfull_width_m, areas, fp_widths, side_slope_z
        )
        assert np.isclose(h_k, depths[k], rtol=1e-5)
        assert np.isclose(w_k, fp_widths[k], rtol=1e-5)

    # Shallow overbank flow between 0 and level 0 (e.g. h in (0, 0.1))
    mid_area_0 = areas[0] * np.float32(0.5)
    h_mid_0, w_mid_0 = compute_overbank_depth_and_top_width(
        mid_area_0, bankfull_width_m, areas, fp_widths, side_slope_z
    )
    assert 0.0 < h_mid_0 < depths[0]
    assert bankfull_width_m < w_mid_0 < fp_widths[0]

    # Extreme flood beyond 5.0m
    extreme_area = areas[9] + np.float32(100.0)
    h_ext, w_ext = compute_overbank_depth_and_top_width(
        extreme_area, bankfull_width_m, areas, fp_widths, side_slope_z
    )
    assert h_ext > 5.0
    assert w_ext > fp_widths[9]

    # Test build_inertial_geometry with 10-point profile
    n_reaches = 2
    fp_prof = np.zeros((n_reaches, 10), dtype=np.float32)
    for k, d in enumerate(FLOODPLAIN_PROFILE_DEPTHS):
        fp_prof[0, k] = 20.0 + 50.0 + 2.0 * 2.0 * d
        fp_prof[1, k] = 30.0 + 80.0 + 2.0 * 3.0 * d
    geom = build_inertial_geometry(
        river_length=np.array([1000.0, 2000.0], dtype=np.float32),
        river_width=np.array([20.0, 30.0], dtype=np.float32),
        shape_exponent=np.array([0.5, 0.5], dtype=np.float32),
        bankfull_depth=np.array([2.0, 3.0], dtype=np.float32),
        bed_elevation=np.array([10.0, 5.0], dtype=np.float32),
        ds_bed_elevation=np.array([5.0, 0.0], dtype=np.float32),
        interface_bed_elevation_max=np.array([10.0, 5.0], dtype=np.float32),
        manning_n_sq=np.array([0.0009, 0.0009], dtype=np.float32),
        inv_dx_interface=np.array([0.001, 0.0005], dtype=np.float32),
        pit_slope=np.array([0.001, 0.001], dtype=np.float32),
        floodplain_side_slope_m_per_m=np.array([2.0, 3.0], dtype=np.float32),
        floodplain_profile_widths=fp_prof,
    )
    assert geom.geom_floodplain_areas.shape == (n_reaches, 10)
    assert geom.geom_floodplain_widths.shape == (n_reaches, 10)
    for k in range(1, 10):
        assert (
            geom.geom_floodplain_areas[:, k] > geom.geom_floodplain_areas[:, k - 1]
        ).all()
        assert (
            geom.geom_floodplain_widths[:, k] >= geom.geom_floodplain_widths[:, k - 1]
        ).all()


def test_routing_calculate_bankfull_width_and_depth_limited_by_floodplain() -> None:
    """Verify that calculate_bankfull_width limits width to floodplain width and depth scales by continuity."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(4)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": 0.27,
                "d": 0.30,
                "e": 0.0,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 1.0
    var: RoutingVariables = RoutingVariables()
    # 4 cells:
    # Cell 0: overland (river_id -1)
    # Cell 1: unobserved reach where W_expected = 72.0 m, floodplain width = 40.0 m (limited)
    # Cell 2: unobserved reach where W_expected = 72.0 m, floodplain width = 100.0 m (unconstrained)
    # Cell 3: observed reach where W_obs = 60.0 m, floodplain width = 35.0 m (limited)
    var.river_ids = np.array([-1, 10, 20, 30], dtype=np.int32)
    var.river_width_alpha = np.full(4, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(4, 0.5, dtype=np.float32)
    var.observed_average_river_width = np.array(
        [np.nan, np.nan, np.nan, 60.0], dtype=np.float32
    )
    q2_val: float = 100.0  # W_expected = 7.2 * 10 = 72.0 m
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [q2_val, q2_val, q2_val],
            "represented_in_grid": [True, True, True],
        },
        index=[10, 20, 30],
    )
    routing.var = var

    # Floodplain profile width at level 0:
    routing.floodplain_profile_widths_m = np.zeros((4, 10), dtype=np.float32)
    routing.floodplain_profile_widths_m[0, 0] = 50.0  # overland
    routing.floodplain_profile_widths_m[1, 0] = 40.0  # Cell 1: limited to 40m (< 72m)
    routing.floodplain_profile_widths_m[2, 0] = 100.0  # Cell 2: unconstrained (> 72m)
    routing.floodplain_profile_widths_m[3, 0] = 35.0  # Cell 3: limited to 35m (< 60m)

    # 1. Test width calculation:
    widths: ArrayFloat32 = routing.calculate_bankfull_width(
        use_simulated_bankfull_q=True
    )
    assert widths[0] == 1.0  # Overland cell retains default_missing_channel_width
    assert np.isclose(widths[1], 40.0)  # Clamped from 72.0 to 40.0 m
    assert np.isclose(widths[2], 72.0)  # Unconstrained at 72.0 m
    assert np.isclose(widths[3], 35.0)  # Clamped from 60.0 to 35.0 m

    # 2. Test depth calculation and continuity scaling:
    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )
    expected_mean_depth: float = 0.27 * (q2_val**0.30)
    expected_centerline_unconstrained: float = 1.5 * expected_mean_depth

    # Cell 2 is unconstrained: centerline depth is 1.5 * h_mean_expected
    assert np.isclose(depths[2], expected_centerline_unconstrained, rtol=1e-3)

    # Cell 1 is clamped to 40m: depth scales by 72.0 / 40.0
    continuity_ratio_1: float = 72.0 / 40.0
    assert np.isclose(
        depths[1], expected_centerline_unconstrained * continuity_ratio_1, rtol=1e-3
    )

    # Cross-sectional flow area (2/3 * W * h_centerline) must be identical between Cell 1 and Cell 2
    area_1: float = (2.0 / 3.0) * float(widths[1]) * float(depths[1])
    area_2: float = (2.0 / 3.0) * float(widths[2]) * float(depths[2])
    assert np.isclose(area_1, area_2, rtol=1e-3)

    # Cell 3 is clamped observed width: depth scales by W_expected / 35.0 = 72.0 / 35.0
    continuity_ratio_3: float = 72.0 / 35.0
    assert np.isclose(
        depths[3], expected_centerline_unconstrained * continuity_ratio_3, rtol=1e-3
    )
    area_3: float = (2.0 / 3.0) * float(widths[3]) * float(depths[3])
    assert np.isclose(area_3, area_2, rtol=1e-3)


def test_routing_calculate_bankfull_depth_slope_scaling_swot() -> None:
    """Verify that calculate_bankfull_depth scales depth with reach slope following hydraulic geometry.

    Tests that:
    1. Bankfull depth follows h_mean = c * Q2^d * max(S, S_min)^e with hydraulic geometry parameters
       c = 0.24, d = 0.34, e = -0.13.
    2. Flatter reach (S = 0.0001) has greater depth and higher conveyance than steep reach (S = 0.01).
    3. The ratio between flat and steep reach depths matches (S_flat / S_steep)^e = (1e-4 / 1e-2)^(-0.13) = 100^0.13 ≈ 1.820.
    4. Reaches with zero or sub-floor slope are floored at 1e-4 m/m and do not blow up or produce NaNs.
    5. When observed width is available, depth scales inversely via continuity to preserve bankfull area.
    """
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    c_val: float = 0.24
    d_val: float = 0.34
    e_val: float = -0.13
    routing.config = {
        "river_depth": {
            "parameters": {
                "c": c_val,
                "d": d_val,
                "e": e_val,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": 7.2,
                "beta": 0.5,
            }
        },
    }
    routing.default_missing_channel_width = 10.0

    q2_val: float = 100.0
    w_expected: float = 7.2 * (q2_val**0.5)  # 72.0 m

    # 4 reaches:
    # Reach 0: steep (S = 0.01)
    # Reach 1: flat lowland (S = 0.0001)
    # Reach 2: dead flat / lake (S = 0.0, should be floored at 1e-4)
    # Reach 3: narrow observed width (W = 36.0m = 0.5 * W_expected, S = 0.001)
    slopes: ArrayFloat32 = np.array([0.01, 0.0001, 0.0, 0.001], dtype=np.float32)
    widths: ArrayFloat32 = np.array(
        [w_expected, w_expected, w_expected, 0.5 * w_expected], dtype=np.float32
    )

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    var.river_width_alpha = np.full(4, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(4, 0.5, dtype=np.float32)
    var.observed_average_river_width = np.array(
        [np.nan, np.nan, np.nan, 0.5 * w_expected], dtype=np.float32
    )
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [q2_val] * 4,
            "represented_in_grid": [True] * 4,
        },
        index=[0, 1, 2, 3],
    )
    routing.var = var

    # Attach grid with specific slopes
    class CustomGridVar(DummyGridVar):
        def __init__(self) -> None:
            super().__init__(4)
            self.river_slope_m_per_m = slopes

    class CustomGrid(DummyGrid):
        def __init__(self) -> None:
            self.var = CustomGridVar()

    routing.grid = CustomGrid()  # ty:ignore[invalid-assignment]

    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )

    # 1. Expected depths:
    h_mean_steep: float = c_val * (q2_val**d_val) * (0.01**e_val)
    h_bf_steep: float = 1.5 * h_mean_steep
    assert np.isclose(float(depths[0]), h_bf_steep, rtol=1e-3)

    h_mean_flat: float = c_val * (q2_val**d_val) * (0.0001**e_val)
    h_bf_flat: float = 1.5 * h_mean_flat
    assert np.isclose(float(depths[1]), h_bf_flat, rtol=1e-3)

    # 2. Ratio matches theoretical power law: (0.0001 / 0.01)^(-0.13) = 100^0.13 ≈ 1.8197
    expected_ratio: float = (0.0001 / 0.01) ** e_val
    assert np.isclose(float(depths[1] / depths[0]), expected_ratio, rtol=1e-3)

    # 3. Dead flat reach (slope 0.0) is floored at 1e-4:
    h_mean_floor: float = c_val * (q2_val**d_val) * (1e-4**e_val)
    h_bf_floor: float = 1.5 * h_mean_floor
    assert np.isclose(float(depths[2]), h_bf_floor, rtol=1e-3)
    assert np.all(np.isfinite(depths))

    # 4. Observed width reach (0.5x width): depth is doubled via continuity
    h_mean_ref: float = c_val * (q2_val**d_val) * (0.001**e_val)
    expected_narrow_depth: float = 2.0 * (1.5 * h_mean_ref)
    assert np.isclose(float(depths[3]), expected_narrow_depth, rtol=1e-3)

    # Cross-sectional area A = 2/3 * W * h_bf is conserved
    area_narrow: float = (2.0 / 3.0) * float(widths[3]) * float(depths[3])
    expected_area: float = w_expected * h_mean_ref
    assert np.isclose(area_narrow, expected_area, rtol=1e-3)


def test_routing_calculate_bankfull_bjerklie_coupled_system() -> None:
    """Verify that calculate_bankfull_depth and width operate consistently under the Bjerklie (2003) system.

    Tests that:
    1. Default parameters follow Bjerklie et al. (2003): default_alpha = 3.91, beta = 0.62,
       c = 0.15, d = 0.21, e = -0.20.
    2. Bankfull width scales as W = 3.91 * Q2^0.62.
    3. Centerline depth scales as h = 1.5 * 0.15 * Q2^0.21 * S^(-0.20).
    4. For a typical river reach (Q2 = 50 m3/s, S = 0.002), conveyance and bankfull velocity
       remain realistic and prevent cross-sectional over-inflation.
    """
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    alpha_val: float = 3.91
    beta_val: float = 0.62
    c_val: float = 0.15
    d_val: float = 0.21
    e_val: float = -0.20

    routing.config = {
        "river_depth": {
            "parameters": {
                "c": c_val,
                "d": d_val,
                "e": e_val,
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": True,
            }
        },
        "river_width": {
            "parameters": {
                "default_alpha": alpha_val,
                "beta": beta_val,
            }
        },
    }
    routing.default_missing_channel_width = 10.0

    q2_val: float = 50.0
    w_expected: float = alpha_val * (q2_val**beta_val)

    slopes: ArrayFloat32 = np.array([0.002, 0.0001], dtype=np.float32)
    widths: ArrayFloat32 = np.array([w_expected, w_expected], dtype=np.float32)

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([0, 1], dtype=np.int32)
    var.river_width_alpha = np.full(2, alpha_val, dtype=np.float32)
    var.river_width_beta = np.full(2, beta_val, dtype=np.float32)
    var.observed_average_river_width = np.array([np.nan, np.nan], dtype=np.float32)
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [q2_val, q2_val],
            "represented_in_grid": [True, True],
            "river_depth_c": [c_val, c_val],
            "river_depth_d": [d_val, d_val],
            "river_depth_e": [e_val, e_val],
            "slope": slopes,
        },
        index=[0, 1],
    )
    routing.var = var

    class CustomGridVar(DummyGridVar):
        def __init__(self) -> None:
            self.river_slope_m_per_m = slopes
            self.river_id = np.array([0, 1], dtype=np.int32)
            self.is_river = np.array([True, True], dtype=bool)

    class CustomGrid(DummyGrid):
        def __init__(self) -> None:
            self.var = CustomGridVar()

    routing.grid = CustomGrid()  # ty:ignore[invalid-assignment]

    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )

    h_mean_steep: float = c_val * (q2_val**d_val) * (0.002**e_val)
    h_bf_steep: float = 1.5 * h_mean_steep
    assert np.isclose(float(depths[0]), h_bf_steep, rtol=1e-3)

    h_mean_flat: float = c_val * (q2_val**d_val) * (0.0001**e_val)
    h_bf_flat: float = 1.5 * h_mean_flat
    assert np.isclose(float(depths[1]), h_bf_flat, rtol=1e-3)

    # In flatter reach, depth is greater:
    assert depths[1] > depths[0]


def test_compute_hydraulic_geometry_width() -> None:
    """Verifies hydraulic geometry power law w = alpha * |Q|^beta and zero flow handling."""
    alpha = np.float32(3.5)
    beta = np.float32(0.5)

    # Zero and near-zero flow
    assert compute_hydraulic_geometry_width(np.float32(0.0), alpha, beta) == np.float32(
        0.0
    )
    assert compute_hydraulic_geometry_width(
        np.float32(1e-15), alpha, beta
    ) == np.float32(0.0)

    # Positive flow
    q_pos = np.float32(16.0)
    w_pos: np.float32 = compute_hydraulic_geometry_width(q_pos, alpha, beta)
    assert np.isclose(float(w_pos), 3.5 * (16.0**0.5), rtol=1e-5)

    # Negative flow (uses absolute magnitude)
    q_neg = np.float32(-16.0)
    w_neg: np.float32 = compute_hydraulic_geometry_width(q_neg, alpha, beta)
    assert np.isclose(float(w_neg), float(w_pos), rtol=1e-5)


def test_compute_top_width_from_depth() -> None:
    """Verifies unified top width calculation across in-bank and overbank regimes."""
    bankfull_depth = np.float32(2.0)
    bankfull_width = np.float32(20.0)
    shape_exponent = np.float32(0.5)
    fp_widths = np.full(NUM_FLOODPLAIN_LEVELS, 40.0, dtype=np.float32)
    fp_widths[0] = 30.0  # at 0.1 m overbank
    fp_widths[1] = 40.0  # at 0.25 m overbank
    side_slope_z = np.float32(2.0)

    # Dry / negative water depth
    assert compute_top_width_from_depth(
        water_depth_m=np.float32(0.0),
        bankfull_depth_m=bankfull_depth,
        bankfull_width_m=bankfull_width,
        shape_exponent=shape_exponent,
        floodplain_widths=fp_widths,
        side_slope_z=side_slope_z,
    ) == np.float32(0.0)
    assert compute_top_width_from_depth(
        water_depth_m=np.float32(-1.0),
        bankfull_depth_m=bankfull_depth,
        bankfull_width_m=bankfull_width,
        shape_exponent=shape_exponent,
        floodplain_widths=fp_widths,
        side_slope_z=side_slope_z,
    ) == np.float32(0.0)

    # In-bank flow at half bankfull depth (depth = 0.5 m, bankfull = 2.0 m -> depth_ratio = 0.25)
    w_inbank: np.float32 = compute_top_width_from_depth(
        water_depth_m=np.float32(0.5),
        bankfull_depth_m=bankfull_depth,
        bankfull_width_m=bankfull_width,
        shape_exponent=shape_exponent,
        floodplain_widths=fp_widths,
        side_slope_z=side_slope_z,
    )
    # 20.0 * (0.25 ** 0.5) = 10.0
    assert np.isclose(float(w_inbank), 10.0, rtol=1e-5)

    # Overbank flow at depth = 2.1 m (overbank depth = 0.1 m, matching profile level 0)
    w_overbank: np.float32 = compute_top_width_from_depth(
        water_depth_m=np.float32(2.1),
        bankfull_depth_m=bankfull_depth,
        bankfull_width_m=bankfull_width,
        shape_exponent=shape_exponent,
        floodplain_widths=fp_widths,
        side_slope_z=side_slope_z,
    )
    assert np.isclose(float(w_overbank), 30.0, rtol=1e-5)
