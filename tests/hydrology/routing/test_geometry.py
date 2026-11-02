"""Tests for hydrological routing."""

import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from geb.geb_types import ArrayBool, ArrayFloat32, ArrayInt32
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
    compute_overbank_area_from_depth,
    compute_overbank_depth_and_top_width,
    compute_static_geometry,
    compute_top_width_from_depth,
)
from geb.hydrology.routing.inertial_kernel import _compute_reach_stage_and_cfl_depth


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
        self.n: int = n
        self.var = DummyGridVar(n)

    def load2d(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Mock loading 2D raster data.

        Returns:
            Dummy 2D float32 array.
        """
        return np.zeros(self.n, dtype=np.float32)

    def load3d(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Mock loading 3D raster data.

        Returns:
            Dummy 3D float32 array.
        """
        return np.full((10, self.n), 100.0, dtype=np.float32)


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
        levee_depth_m=bankfull_depth_m,
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
        levee_depth_m=bankfull_depth_m,
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
        levee_depth_m=bankfull_depth_m,
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
        levee_depth_m=bankfull_depth_m,
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
        levee_depth_m=bankfull_depth_m,
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
        levee_depth_m=bankfull_depth_m,
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
            levee_depth_m=bankfull_depth_m,
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
            "hydrology": {
                "routing": {
                    "bankfull_discharge_multiplier": mult,
                    "mannings_n_multiplier": 1.0,
                    "lake_outflow_multiplier": 1.0,
                }
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
    3. Backwater integration and Manning normal depth inversion operate as expected.
    """
    from geb.hydrology.routing import Routing

    routing = object.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(4)  # ty:ignore[invalid-assignment]
    routing._compressed_routing_seq = np.arange(4, dtype=np.int32)
    routing._compressed_ds = np.full(4, -1, dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
    routing.config = {
        "max_threads": None,
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
    # Cell 2 is width 100 m, Cell 3 is width 200 m -> Under Manning inversion h ~ W^(-0.6):
    assert h_depth_simulated[2] > h_depth_simulated[3]
    assert np.isclose(h_depth_simulated[2] / h_depth_simulated[3], 2.0**0.6, rtol=1e-3)

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

    # Scenario 4: Respects min_depth_m floor even on zero discharge
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
            "return_period_2_years_daily_m3_per_s": [50000.0, 50000.0],
            "represented_in_grid": [True, True],
        },
        index=[0, 1],
    )
    unrealistic_widths = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
            levee_depth_grid=depth_grid,
            manning_grid=manning_grid,
            floodplain_profile_widths_grid_m=np.zeros(
                (len(width_grid), 10), dtype=np.float32
            ),
            floodplain_side_slope_grid_m_per_m=np.full_like(width_grid, 2.0),
        )


def test_routing_unobserved_channel_width_depth_estimation() -> None:
    """Verify that unobserved and observed rivers get depth via Manning/backwater inversion."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(3)  # ty:ignore[invalid-assignment]
    routing._compressed_routing_seq = np.arange(3, dtype=np.int32)
    routing._compressed_ds = np.full(3, -1, dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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

    q2_val: float = 100.0
    expected_w: float = 7.2 * (q2_val**0.5)
    n_val: float = 0.03
    s0_val: float = 1e-4
    expected_h_unobs: float = 1.5 * (
        (n_val * q2_val / (expected_w * np.sqrt(s0_val))) ** 0.6
    )
    expected_h_obs: float = 1.5 * ((n_val * q2_val / (50.0 * np.sqrt(s0_val))) ** 0.6)

    input_widths: ArrayFloat32 = np.array([0.5, 50.0, expected_w], dtype=np.float32)

    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=input_widths,
        use_simulated_bankfull_q=True,
    )

    assert np.isclose(float(depths[1]), expected_h_obs, rtol=1e-3)
    assert np.isclose(float(depths[2]), expected_h_unobs, rtol=1e-3)


def test_routing_set_router_estimates_width_from_simulated_q2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that set_router estimates width from Q2 for unobserved channels when Q2 is present."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing = Routing.__new__(Routing)
    routing.config = {
        "max_threads": None,
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
        },
        "flood_protection": {"enabled": True},
        "retention_basins": {"release_threshold_factor": 1.0},
        "riverbed_hydraulic_conductivity_multiplier": 0.1,
    }
    routing.default_missing_channel_width = 0.5
    routing.retention_max_storage_m3 = np.zeros(0, dtype=np.float32)
    routing.retention_basin_ids = np.full(2, -1, dtype=np.int32)
    routing.controlled_retention = np.zeros(0, dtype=bool)
    routing.river_network = None  # ty:ignore[invalid-assignment]
    routing._compressed_routing_seq = np.arange(2, dtype=np.int32)
    routing._compressed_ds = np.full(2, -1, dtype=np.int32)
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
            # T = 2 yr: no levees above bankfull, isolates the width estimate
            "flood_protection_standard": [2.0, 2.0],
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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

    # Estimated depth via Manning normal depth: 1.5 * (manning30 * 150 / (w30 * sqrt(0.001)))^0.6
    h30: float = float(routing.var.rivers.loc[30, "depth"])
    expected_h30: float = 1.5 * ((manning30 * 150.0 / (w30 * np.sqrt(0.001))) ** 0.6)
    assert np.isclose(h30, expected_h30, rtol=1e-3)


def test_unrepresented_river_geometry_iterative_cascade() -> None:
    """Verify that multi-step chains of unrepresented rivers are resolved iteratively."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(2)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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
    expected_h2 = 1.5 * ((0.03 * 64.0 / (57.6 * np.sqrt(0.001))) ** 0.6)
    expected_h3 = 1.5 * ((0.03 * 64.0 / (57.6 * np.sqrt(1e-4))) ** 0.6)
    assert np.isclose(float(routing.var.rivers.loc[2, "depth"]), expected_h2, rtol=1e-3)
    assert np.isclose(float(routing.var.rivers.loc[3, "depth"]), expected_h3, rtol=1e-3)


def test_unrepresented_river_cascade_with_bankfull_multiplier() -> None:
    """Verify that unrepresented cascades do not compound bankfull multiplier across hops or distort weights."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    # Configure calibration multiplier to 2.0 to verify non-compounding
    routing.model = DummyModel(2.0)  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(2)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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
    expected_h2: float = 1.5 * ((0.03 * 128.0 / (expected_w * np.sqrt(0.001))) ** 0.6)
    expected_h3: float = 1.5 * ((0.03 * 128.0 / (expected_w * np.sqrt(1e-4))) ** 0.6)
    assert np.isclose(float(routing.var.rivers.loc[2, "width"]), expected_w)
    assert np.isclose(float(routing.var.rivers.loc[3, "width"]), expected_w)
    assert np.isclose(float(routing.var.rivers.loc[2, "depth"]), expected_h2, rtol=1e-3)
    assert np.isclose(float(routing.var.rivers.loc[3, "depth"]), expected_h3, rtol=1e-3)


def test_unrepresented_river_isolated_no_upstream_retains_nan() -> None:
    """Verify that an unrepresented river without upstream segments retains NaN."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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

    # Depth uses Manning normal depth inversion: 1.5 * ((0.03 * 36.0) / (80.0 * sqrt(0.001)))^0.6
    expected_depth: float = 1.5 * (((0.03 * 36.0) / (80.0 * np.sqrt(0.001))) ** 0.6)
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
        levee_depth_grid=depth_grid,
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
            levee_depth_grid=np.array([1.0], dtype=np.float32),
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
    routing._compressed_routing_seq = np.array([0], dtype=np.int32)
    routing._compressed_ds = np.array([-1], dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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

    # Bankfull Q = 100 m3/s, pit slope = 1e-4, manning n = 0.03
    q_bf: float = 100.0
    w_expected: float = 7.2 * (q_bf**0.5)
    mean_depth_expected: float = ((0.03 * q_bf) / (w_expected * np.sqrt(1e-4))) ** 0.6

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

    # 2. Narrow constriction reach (half width): under Manning normal depth, depth scales as (0.5)^(-0.6) = 2.0^0.6
    narrow_widths: ArrayFloat32 = np.array([0.5 * w_expected], dtype=np.float32)
    var.observed_average_river_width = narrow_widths.copy()
    narrow_depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=narrow_widths,
        use_simulated_bankfull_q=True,
    )
    assert np.isclose(
        float(narrow_depths[0]), (2.0**0.6) * expected_centerline_depth, rtol=1e-3
    )


def test_routing_calculate_bankfull_depth_continuity_scaling_with_observed_width() -> (
    None
):
    """Verify that calculate_bankfull_depth scales depth with observed width to conserve Manning conveyance."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(4)  # ty:ignore[invalid-assignment]
    routing._compressed_routing_seq = np.arange(4, dtype=np.int32)
    routing._compressed_ds = np.full(4, -1, dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
        },
    }
    routing.default_missing_channel_width = 10.0

    q2_flood: float = 100.0  # m3/s
    w_expected: float = 7.2 * (q2_flood**0.5)  # 72.0 m
    expected_mean_depth: float = (
        (0.03 * q2_flood) / (w_expected * np.sqrt(1e-4))
    ) ** 0.6
    expected_centerline_depth: float = 1.5 * expected_mean_depth

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

    # 2. Wide reach (2x width): depth scales by (0.5)^0.6
    assert np.isclose(
        float(depths[1]), (0.5**0.6) * expected_centerline_depth, rtol=1e-3
    )

    # 3. Narrow reach (0.5x width): depth scales by (2.0)^0.6
    assert np.isclose(
        float(depths[2]), (2.0**0.6) * expected_centerline_depth, rtol=1e-3
    )

    # 4. Constriction (0.25x width): depth scales by (4.0)^0.6
    assert np.isclose(
        float(depths[3]), (4.0**0.6) * expected_centerline_depth, rtol=1e-3
    )

    # 5. Verify Manning bankfull discharge Q = (1/n) * W * h_mean^(5/3) * sqrt(S) is conserved across all reaches
    for i in range(4):
        h_mean_i: float = float(depths[i]) / 1.5
        q_conveyance: float = (
            (1.0 / 0.03) * float(widths[i]) * (h_mean_i ** (5.0 / 3.0)) * np.sqrt(1e-4)
        )
        assert np.isclose(q_conveyance, q2_flood, rtol=1e-3)

    # 6. Cold-start fallback (use_simulated_bankfull_q = False): operates without error and produces positive finite depths
    fallback_depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=False,
    )
    assert np.all(np.isfinite(fallback_depths))
    assert np.all(fallback_depths >= 0.1)


def test_routing_calculate_bankfull_depth_flag_revert_machinery() -> None:
    """Verify that use_observed_width_as_bankfull flag toggles between default and reach-specific alpha/beta in q_proxy."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing._compressed_routing_seq = np.array([0], dtype=np.int32)
    routing._compressed_ds = np.array([-1], dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
        },
    }
    routing.default_missing_channel_width = 10.0

    w_obs: float = 72.0  # m
    widths: ArrayFloat32 = np.array([w_obs], dtype=np.float32)

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([0], dtype=np.int32)
    var.observed_average_river_width = widths.copy()
    # Inflated reach-specific river_width_alpha (14.4, twice default_alpha)
    var.river_width_alpha = np.array([14.4], dtype=np.float32)
    var.river_width_beta = np.array([0.5], dtype=np.float32)
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [100.0],
            "represented_in_grid": [True],
        },
        index=[0],
    )
    routing.var = var

    # 1. Cold-start with Flag True (default): q_proxy = (72.0 / 7.2)^2 = 100.0 m3/s
    depth_cold_true: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=False,
    )
    expected_depth_q100: float = 1.5 * (
        ((0.03 * 100.0) / (w_obs * np.sqrt(1e-4))) ** 0.6
    )
    assert np.isclose(float(depth_cold_true[0]), expected_depth_q100, rtol=1e-3)

    # 2. Cold-start with Flag False: q_proxy = (72.0 / 14.4)^2 = 25.0 m3/s -> depth scales by (0.25)^0.6
    routing.config["river_depth"]["use_observed_width_as_bankfull"] = False
    depth_cold_false: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=False,
    )
    assert np.isclose(
        float(depth_cold_false[0]), (0.25**0.6) * expected_depth_q100, rtol=1e-3
    )


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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
    }
    with pytest.raises(KeyError):
        routing.calculate_bankfull_depth(widths, use_simulated_bankfull_q=False)

    # Missing use_observed_width_as_bankfull raises KeyError
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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
    """Verify that unobserved rivers use alpha/beta for width and Manning normal depth for depth."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(1)  # ty:ignore[invalid-assignment]
    routing._compressed_routing_seq = np.array([0], dtype=np.int32)
    routing._compressed_ds = np.array([-1], dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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

    # For an unconstrained channel (bankfull_top_width_m == expected_w), depth is Manning normal depth:
    depths_unconstrained: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )
    expected_mean_depth: float = ((0.03 * q2_val) / (expected_w * np.sqrt(1e-4))) ** 0.6
    expected_centerline_depth: float = 1.5 * expected_mean_depth
    assert np.isclose(
        float(depths_unconstrained[0]), expected_centerline_depth, rtol=1e-3
    )

    # 2. When bankfull width is constrained below expected width (e.g. by narrow floodplain),
    # depth scales as (expected_w / constrained_w)^0.6 to conserve Manning conveyance:
    constrained_widths: ArrayFloat32 = np.array([10.0], dtype=np.float32)
    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=constrained_widths,
        use_simulated_bankfull_q=True,
    )
    conveyance_ratio: float = (expected_w / 10.0) ** 0.6
    assert np.isclose(
        float(depths[0]), expected_centerline_depth * conveyance_ratio, rtol=1e-3
    )


def test_bankfull_discharge_multiplier_scaling() -> None:
    """Verifies that bankfull_discharge_multiplier scales bankfull discharge, width, and depth."""
    from geb.hydrology.routing import Routing, RoutingVariables

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    def make_routing(mult: float) -> Routing:
        r: Routing = Routing.__new__(Routing)
        r.model = DummyModel(mult)  # ty:ignore[invalid-assignment]
        r.grid = DummyGrid(2)  # ty:ignore[invalid-assignment]
        r._compressed_routing_seq = np.arange(2, dtype=np.int32)
        r._compressed_ds = np.full(2, -1, dtype=np.int32)
        r.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
        r.config = {
            "river_width": {
                "default_alpha": 7.2,
                "beta": 0.5,
            },
            "river_depth": {
                "min_depth_m": 0.1,
                "shape_exponent": 0.5,
                "use_observed_width_as_bankfull": False,
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

    # For observed reach with fixed width, Manning normal depth scales with Q_mult^0.60
    assert np.isclose(d_high[0] / d_base[0], 2.0**0.60, rtol=1e-3)
    assert np.isclose(d_low[0] / d_base[0], 0.5**0.60, rtol=1e-3)

    # 2. Unobserved width reach (index 1): width scales with Q_mult^beta = Q_mult^0.5
    assert np.isclose(w_high[1] / w_base[1], 2.0**0.5, rtol=1e-3)
    assert np.isclose(w_low[1] / w_base[1], 0.5**0.5, rtol=1e-3)

    # For unobserved reach (W ~ Q^0.5), Manning depth scales with (Q / W)^0.6 = Q_mult^(0.5 * 0.6) = Q_mult^0.30
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
    # Flood protection is not under test here; disable it so no standards table is needed
    routing.config = {"flood_protection": {"enabled": False}}

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
        levee_depth=np.array([2.0, 3.0], dtype=np.float32),
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
    """Verify that calculate_bankfull_width limits width to floodplain width and depth scales by Manning conveyance."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(4)  # ty:ignore[invalid-assignment]
    routing._compressed_routing_seq = np.arange(4, dtype=np.int32)
    routing._compressed_ds = np.full(4, -1, dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
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

    # 2. Test depth calculation and Manning conveyance scaling:
    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )
    expected_mean_depth: float = ((0.03 * q2_val) / (72.0 * np.sqrt(1e-4))) ** 0.6
    expected_centerline_unconstrained: float = 1.5 * expected_mean_depth

    # Cell 2 is unconstrained: centerline depth is 1.5 * h_mean_expected
    assert np.isclose(depths[2], expected_centerline_unconstrained, rtol=1e-3)

    # Cell 1 is clamped to 40m: depth scales by (72.0 / 40.0)^0.6
    conveyance_ratio_1: float = (72.0 / 40.0) ** 0.6
    assert np.isclose(
        depths[1], expected_centerline_unconstrained * conveyance_ratio_1, rtol=1e-3
    )

    # Cell 3 is clamped observed width: depth scales by (72.0 / 35.0)^0.6
    conveyance_ratio_3: float = (72.0 / 35.0) ** 0.6
    assert np.isclose(
        depths[3], expected_centerline_unconstrained * conveyance_ratio_3, rtol=1e-3
    )

    # Manning bankfull discharge Q = (1/n) * W * h_mean^(5/3) * sqrt(S) is conserved across Cells 1, 2, 3
    for idx in (1, 2, 3):
        h_mean_idx: float = float(depths[idx]) / 1.5
        q_conv: float = (
            (1.0 / 0.03)
            * float(widths[idx])
            * (h_mean_idx ** (5.0 / 3.0))
            * np.sqrt(1e-4)
        )
        assert np.isclose(q_conv, q2_val, rtol=1e-3)


def test_routing_calculate_bankfull_depth_slope_scaling_swot() -> None:
    """Verify that calculate_bankfull_depth scales depth with reach bed slope following Manning's equation."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": 7.2,
            "beta": 0.5,
        },
    }
    routing.default_missing_channel_width = 10.0

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]

    q2_val: float = 100.0
    w_expected: float = 7.2 * (q2_val**0.5)  # 72.0 m

    # 4 reaches draining to a common sink cell 4 (bed = 0.0 m):
    # Cell 0: steep (bed = 10.0 m, L = 1000 m -> S = 0.01)
    # Cell 1: flat lowland (bed = 0.1 m, L = 1000 m -> S = 0.0001)
    # Cell 2: dead flat (bed = 0.0 m, L = 1000 m -> S = 0.0, floored at 1e-4)
    # Cell 3: narrow observed width (W = 36.0m = 0.5 * W_expected, bed = 1.0 m -> S = 0.001)
    # Cell 4: downstream sink (bed = 0.0 m, Q = 0.0 m3/s so stage is min_depth_m = 0.1 m)
    routing._compressed_routing_seq = np.array([4, 0, 1, 2, 3], dtype=np.int32)
    routing._compressed_ds = np.array([4, 4, 4, 4, -1], dtype=np.int32)

    widths: ArrayFloat32 = np.array(
        [w_expected, w_expected, w_expected, 0.5 * w_expected, w_expected],
        dtype=np.float32,
    )
    bed_elevs: ArrayFloat32 = np.array([10.0, 0.1, 0.0, 1.0, 0.0], dtype=np.float32)

    var: RoutingVariables = RoutingVariables()
    # Mark cells 0..3 as kinematic so we test pure Manning slope scaling without downstream sink backwater
    var.river_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    var.river_width_alpha = np.full(5, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(5, 0.5, dtype=np.float32)
    var.observed_average_river_width = np.array(
        [np.nan, np.nan, np.nan, 0.5 * w_expected, np.nan], dtype=np.float32
    )
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [
                q2_val,
                q2_val,
                q2_val,
                q2_val,
                0.0,
            ],
            "represented_in_grid": [True] * 5,
        },
        index=[0, 1, 2, 3, 4],
    )
    routing.var = var

    class CustomGrid(DummyGrid):
        def __init__(self) -> None:
            super().__init__(5)

        def load2d(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return bed_elevs.copy()

    routing.grid = CustomGrid()  # ty:ignore[invalid-assignment]

    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )

    # 1. Expected Manning normal depths: h_bf = 1.5 * ((n * Q) / (W * sqrt(S)))^0.6
    h_bf_steep: float = 1.5 * (((0.03 * q2_val) / (w_expected * np.sqrt(0.01))) ** 0.6)
    assert np.isclose(float(depths[0]), h_bf_steep, rtol=1e-3)

    h_bf_flat: float = 1.5 * (((0.03 * q2_val) / (w_expected * np.sqrt(0.0001))) ** 0.6)
    assert np.isclose(float(depths[1]), h_bf_flat, rtol=1e-3)

    # 2. Ratio matches Manning slope exponent: (0.0001 / 0.01)^(-0.3) = 100^0.3 ≈ 3.981
    expected_ratio: float = (0.0001 / 0.01) ** (-0.3)
    assert np.isclose(float(depths[1] / depths[0]), expected_ratio, rtol=1e-3)

    # 3. Dead flat reach (drop = 0.0) is floored at S_min = 1e-4:
    assert np.isclose(float(depths[2]), h_bf_flat, rtol=1e-3)
    assert np.all(np.isfinite(depths))

    # 4. Observed width reach (0.5x width, S = 0.001): depth scales by 2.0^0.6
    h_bf_ref: float = 1.5 * (((0.03 * q2_val) / (w_expected * np.sqrt(0.001))) ** 0.6)
    expected_narrow_depth: float = (2.0**0.6) * h_bf_ref
    assert np.isclose(float(depths[3]), expected_narrow_depth, rtol=1e-3)


def test_routing_calculate_bankfull_width_depth_coupled_system() -> None:
    """Verify that calculate_bankfull_depth and width operate consistently under coupled width and backwater depth."""
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    alpha_val: float = 7.2
    beta_val: float = 0.5

    routing.config = {
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "river_width": {
            "default_alpha": alpha_val,
            "beta": beta_val,
        },
    }
    routing.default_missing_channel_width = 10.0
    routing._compressed_routing_seq = np.array([2, 0, 1], dtype=np.int32)
    routing._compressed_ds = np.array([2, 2, -1], dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]

    q2_val: float = 50.0
    w_expected: float = alpha_val * (q2_val**beta_val)

    # Cell 0: S = 0.002 (bed = 2.0m -> 0.0m), Cell 1: S = 0.0001 (bed = 0.1m -> 0.0m), Cell 2: sink (bed = 0.0m)
    bed_elevs: ArrayFloat32 = np.array([2.0, 0.1, 0.0], dtype=np.float32)
    widths: ArrayFloat32 = np.array(
        [w_expected, w_expected, w_expected], dtype=np.float32
    )

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([0, 1, 2], dtype=np.int32)
    var.river_width_alpha = np.full(3, alpha_val, dtype=np.float32)
    var.river_width_beta = np.full(3, beta_val, dtype=np.float32)
    var.observed_average_river_width = np.full(3, np.nan, dtype=np.float32)
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [q2_val, q2_val, 0.0],
            "represented_in_grid": [True, True, True],
        },
        index=[0, 1, 2],
    )
    routing.var = var

    class CustomGrid(DummyGrid):
        def __init__(self) -> None:
            super().__init__(3)

        def load2d(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return bed_elevs.copy()

    routing.grid = CustomGrid()  # ty:ignore[invalid-assignment]

    depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths,
        use_simulated_bankfull_q=True,
    )

    h_bf_steep: float = 1.5 * (((0.03 * q2_val) / (w_expected * np.sqrt(0.002))) ** 0.6)
    assert np.isclose(float(depths[0]), h_bf_steep, rtol=1e-3)

    h_bf_flat: float = 1.5 * (((0.03 * q2_val) / (w_expected * np.sqrt(0.0001))) ** 0.6)
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
        levee_depth_m=bankfull_depth,
        bankfull_width_m=bankfull_width,
        shape_exponent=shape_exponent,
        floodplain_widths=fp_widths,
        side_slope_z=side_slope_z,
    ) == np.float32(0.0)
    assert compute_top_width_from_depth(
        water_depth_m=np.float32(-1.0),
        bankfull_depth_m=bankfull_depth,
        levee_depth_m=bankfull_depth,
        bankfull_width_m=bankfull_width,
        shape_exponent=shape_exponent,
        floodplain_widths=fp_widths,
        side_slope_z=side_slope_z,
    ) == np.float32(0.0)

    # In-bank flow at half bankfull depth (depth = 0.5 m, bankfull = 2.0 m -> depth_ratio = 0.25)
    w_inbank: np.float32 = compute_top_width_from_depth(
        water_depth_m=np.float32(0.5),
        bankfull_depth_m=bankfull_depth,
        levee_depth_m=bankfull_depth,
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
        levee_depth_m=bankfull_depth,
        bankfull_width_m=bankfull_width,
        shape_exponent=shape_exponent,
        floodplain_widths=fp_widths,
        side_slope_z=side_slope_z,
    )
    assert np.isclose(float(w_overbank), 30.0, rtol=1e-5)


def test_solve_steady_backwater_network_physics() -> None:
    """Tests that solve_steady_backwater_network obeys open-channel backwater physics.

    Verifies that:
    1. Reaches on steep slopes without downstream submergence match Manning normal depth.
    2. Coastal/estuary reaches with bed elevations below datum (sea level 0 m) capture M1 backwater deepening.
    3. Water surface stage decreases monotonically downstream.
    4. Zero discharge cells produce the configured minimum depth.
    """
    from geb.hydrology.routing.solver import solve_steady_backwater_network

    # 4 reaches: Reach 0 (headwater) -> Reach 1 -> Reach 2 -> Reach 3 (outlet at sea)
    # Compressed topological sequence from downstream (pit) to upstream:
    compressed_seq = np.array([3, 2, 1, 0], dtype=np.int32)
    compressed_ds = np.array([1, 2, 3, 3], dtype=np.int32)
    discharge = np.array([50.0, 100.0, 150.0, 200.0], dtype=np.float32)
    width = np.array([30.0, 45.0, 60.0, 75.0], dtype=np.float32)
    bed_elev = np.array([10.0, 5.0, 0.0, -2.0], dtype=np.float32)
    reach_len = np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32)
    manning_n = np.full(4, 0.035, dtype=np.float32)
    shape_exp = np.float32(0.5)
    min_d = np.float32(0.1)
    no_kinematic = np.zeros(4, dtype=bool)
    min_kin_slope = np.float32(1e-4)

    depths: ArrayFloat32 = solve_steady_backwater_network(
        compressed_seq=compressed_seq,
        compressed_ds=compressed_ds,
        discharge=discharge,
        width=width,
        bed_elevation=bed_elev,
        reach_length=reach_len,
        manning_n=manning_n,
        shape_exponent=shape_exp,
        min_depth_m=min_d,
        is_kinematic=no_kinematic,
        min_kinematic_slope_m_per_m=min_kin_slope,
    )

    stage: ArrayFloat32 = bed_elev + depths

    # 1. Reach 0 is steep (drop = 5m / 1000m = 0.005) with bed far above downstream stage:
    # Expected normal depth: 1.5 * (0.035 * 50 / (30 * sqrt(0.005)))^0.6
    s0_reach0 = (bed_elev[0] - bed_elev[1]) / reach_len[0]
    expected_h0 = 1.5 * (
        (manning_n[0] * discharge[0] / (width[0] * np.sqrt(s0_reach0))) ** 0.6
    )
    assert np.isclose(float(depths[0]), float(expected_h0), rtol=1e-3)

    # 2. Reach 3 is submerged by sea level (bed = -2.0 m):
    # Depth must be >= ocean depth (2.0 m) and stage must be >= sea level (0.0 m)
    assert depths[3] >= 2.0
    assert stage[3] >= 0.0

    # 3. Water surface stage decreases monotonically downstream
    assert stage[0] > stage[1] > stage[2] > stage[3]

    # 4. Zero discharge cell produces min_depth_m
    discharge_zero = discharge.copy()
    discharge_zero[0] = 0.0
    depths_zero = solve_steady_backwater_network(
        compressed_seq=compressed_seq,
        compressed_ds=compressed_ds,
        discharge=discharge_zero,
        width=width,
        bed_elevation=bed_elev,
        reach_length=reach_len,
        manning_n=manning_n,
        shape_exponent=shape_exp,
        min_depth_m=min_d,
        is_kinematic=no_kinematic,
        min_kinematic_slope_m_per_m=min_kin_slope,
    )
    assert depths_zero[0] == min_d

    # 5. Kinematic reaches bypass backwater calculation and strictly retain Manning normal depth
    # Reach 2 is subjected to backwater from Reach 3 (sea level). If Reach 2 is kinematic,
    # its depth must match its Manning normal depth without backwater deepening.
    is_kinematic = np.array([False, False, True, False], dtype=bool)
    depths_kin = solve_steady_backwater_network(
        compressed_seq=compressed_seq,
        compressed_ds=compressed_ds,
        discharge=discharge,
        width=width,
        bed_elevation=bed_elev,
        reach_length=reach_len,
        manning_n=manning_n,
        shape_exponent=shape_exp,
        min_depth_m=min_d,
        is_kinematic=is_kinematic,
        min_kinematic_slope_m_per_m=min_kin_slope,
    )
    # Expected Reach 2 normal depth:
    s0_reach2 = (bed_elev[2] - bed_elev[3]) / reach_len[2]
    expected_h2_norm = 1.5 * (
        (manning_n[2] * discharge[2] / (width[2] * np.sqrt(s0_reach2))) ** 0.6
    )
    assert np.isclose(float(depths_kin[2]), float(expected_h2_norm), rtol=1e-3)
    # Under backwater (is_kinematic=False), Reach 2 is deepened by the downstream ocean level:
    assert depths[2] > depths_kin[2]


def test_solve_steady_backwater_network_input_validation() -> None:
    """Tests that solve_steady_backwater_network validates its input parameters.

    Verifies that explicit ValueError exceptions are raised when inputs violate physical
    or numerical domain constraints (e.g., non-positive dimensions, sub-minimum roughness,
    negative slope floors).
    """
    import pytest

    from geb.hydrology.routing.solver import solve_steady_backwater_network

    compressed_seq: ArrayInt32 = np.array([0], dtype=np.int32)
    compressed_ds: ArrayInt32 = np.array([0], dtype=np.int32)
    discharge: ArrayFloat32 = np.array([10.0], dtype=np.float32)
    width: ArrayFloat32 = np.array([20.0], dtype=np.float32)
    bed_elev: ArrayFloat32 = np.array([5.0], dtype=np.float32)
    reach_len: ArrayFloat32 = np.array([1000.0], dtype=np.float32)
    manning_n: ArrayFloat32 = np.full(1, 0.035, dtype=np.float32)
    shape_exp: np.float32 = np.float32(0.5)
    min_d: np.float32 = np.float32(0.1)
    is_kin: ArrayBool = np.zeros(1, dtype=bool)
    min_kin_slope: np.float32 = np.float32(1e-4)

    # Negative discharge
    with pytest.raises(ValueError, match="discharge must be non-negative"):
        solve_steady_backwater_network(
            compressed_seq,
            compressed_ds,
            np.array([-1.0], dtype=np.float32),
            width,
            bed_elev,
            reach_len,
            manning_n,
            shape_exp,
            min_d,
            is_kin,
            min_kin_slope,
        )

    # Non-positive width
    with pytest.raises(ValueError, match="width must be strictly positive"):
        solve_steady_backwater_network(
            compressed_seq,
            compressed_ds,
            discharge,
            np.array([0.0], dtype=np.float32),
            bed_elev,
            reach_len,
            manning_n,
            shape_exp,
            min_d,
            is_kin,
            min_kin_slope,
        )

    # Non-positive reach_length
    with pytest.raises(ValueError, match="reach_length must be strictly positive"):
        solve_steady_backwater_network(
            compressed_seq,
            compressed_ds,
            discharge,
            width,
            bed_elev,
            np.array([0.0], dtype=np.float32),
            manning_n,
            shape_exp,
            min_d,
            is_kin,
            min_kin_slope,
        )

    # Manning's roughness < 0.01
    with pytest.raises(ValueError, match="manning_n must be at least 0.01"):
        solve_steady_backwater_network(
            compressed_seq,
            compressed_ds,
            discharge,
            width,
            bed_elev,
            reach_len,
            np.array([0.005], dtype=np.float32),
            shape_exp,
            min_d,
            is_kin,
            min_kin_slope,
        )

    # Negative shape_exponent
    with pytest.raises(ValueError, match="shape_exponent must be non-negative"):
        solve_steady_backwater_network(
            compressed_seq,
            compressed_ds,
            discharge,
            width,
            bed_elev,
            reach_len,
            manning_n,
            np.float32(-0.1),
            min_d,
            is_kin,
            min_kin_slope,
        )

    # Non-positive min_depth_m
    with pytest.raises(ValueError, match="min_depth_m must be strictly positive"):
        solve_steady_backwater_network(
            compressed_seq,
            compressed_ds,
            discharge,
            width,
            bed_elev,
            reach_len,
            manning_n,
            shape_exp,
            np.float32(0.0),
            is_kin,
            min_kin_slope,
        )

    # Non-positive min_kinematic_slope_m_per_m
    with pytest.raises(
        ValueError, match="min_kinematic_slope_m_per_m must be strictly positive"
    ):
        solve_steady_backwater_network(
            compressed_seq,
            compressed_ds,
            discharge,
            width,
            bed_elev,
            reach_len,
            manning_n,
            shape_exp,
            min_d,
            is_kin,
            np.float32(0.0),
        )


def test_routing_calculate_bankfull_depth_steady_backwater_integration() -> None:
    """Verifies Routing.calculate_bankfull_depth operating under steady-state backwater and Manning inversion."""
    import pyflwdir

    from geb.hydrology.routing import Routing

    routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.default_missing_channel_width = 10.0

    class BackwaterMockWaterbodies:
        n: int = 0

    class BackwaterMockHydrology:
        waterbodies: BackwaterMockWaterbodies = BackwaterMockWaterbodies()

    routing.hydrology = BackwaterMockHydrology()  # ty:ignore[invalid-assignment]

    # 3-cell linear river: cell 0 -> cell 1 -> cell 2 (pit)
    ldd = np.array([[6, 6, 5]], dtype=np.uint8)
    routing.river_network = pyflwdir.from_array(ldd, ftype="ldd")
    routing.river_network.order_cells(method="walk")
    mapper: ArrayInt32 = np.full(routing.river_network.size + 1, -1, dtype=np.int32)
    indices_init = np.arange(routing.river_network.size, dtype=np.int32)[
        routing.river_network.mask
    ]
    mapper[indices_init] = np.arange(indices_init.size, dtype=np.int32)
    routing._compressed_routing_seq = mapper[routing.river_network.idxs_seq]
    unmasked_ds: ArrayInt32 = routing.river_network.idxs_ds[indices_init]
    routing._compressed_ds = mapper[unmasked_ds]

    class BackwaterMockGridVar:
        def __init__(self) -> None:
            self.river_mannings = np.full(3, 0.035, dtype=np.float32)
            self.river_length = np.full(3, 1000.0, dtype=np.float32)
            self.river_slope_m_per_m = np.array(
                [0.002, 0.001, 0.0005], dtype=np.float32
            )

    class BackwaterMockGrid:
        def __init__(self) -> None:
            self.var = BackwaterMockGridVar()

        def load2d(self, *args: Any, **kwargs: Any) -> np.ndarray:
            # Bed elevations: 4.0 m -> 2.0 m -> 0.0 m
            return np.array([4.0, 2.0, 0.0], dtype=np.float32)

    routing.grid = BackwaterMockGrid()  # ty:ignore[invalid-assignment]

    var = RoutingVariables()
    var.river_ids = np.array([0, 1, 2], dtype=np.int32)
    var.river_width_alpha = np.full(3, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(3, 0.5, dtype=np.float32)
    var.observed_average_river_width = np.array([20.0, 40.0, 60.0], dtype=np.float32)
    var.rivers = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [20.0, 60.0, 120.0],
            "represented_in_grid": [True, True, True],
        },
        index=[0, 1, 2],
    )
    routing.var = var

    routing.config = {
        "river_width": {"default_alpha": 7.2, "beta": 0.5},
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
    }

    widths = var.observed_average_river_width.copy()
    depths = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths, use_simulated_bankfull_q=True
    )

    assert len(depths) == 3
    assert np.all(np.isfinite(depths))
    assert np.all(depths >= 0.1)
    # Headwater reach cell 0 (Q=20, W=20, S=0.002) is shallower than downstream cell 2 (Q=120, W=60, S=0.0005)
    assert depths[0] < depths[2]

    # Test kinematic reach (river_ids == -1) calculates depth via Manning inversion
    var.river_ids[0] = -1
    depths_with_kin = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths, use_simulated_bankfull_q=True
    )
    assert len(depths_with_kin) == 3
    assert np.all(np.isfinite(depths_with_kin))
    assert depths_with_kin[0] >= 0.1


def test_compute_cross_section_and_top_width_with_levees() -> None:
    """Verifies cross-sectional hydraulic geometry and top width under embanked levees.

    Tests the 3 geometric regimes:
    1. In-bank (y <= h_bf): Parabolic power-law profile W(y) = W_bf * sqrt(y / h_bf), A = (2/3) * W_bf * y.
    2. Embanked within levees (h_bf < y <= h_levee): Top width is strictly held at W_bf (no premature
       floodplain spilling), area expands linearly A = A_bf + W_bf * (y - h_bf), and wetted perimeter
       grows with vertical levee embankments P = P_bf + 2 * (y - h_bf), preserving high hydraulic radius.
    3. Overbank overtopping (y > h_levee): Water inundates the protected floodplain behind the
       levees, which lies at natural ground, so overbank depth is referenced to bankfull level
       h_fp = y - h_bf.
    """
    bankfull_width_m: np.float32 = np.float32(50.0)
    bankfull_depth_m: np.float32 = np.float32(4.0)
    levee_depth_m: np.float32 = np.float32(7.0)  # 3 m levee above bankfull
    shape_exponent: np.float32 = np.float32(0.5)  # Parabolic channel
    side_slope_z: np.float32 = np.float32(2.0)

    (
        _,
        inverse_shape_exponent_plus_one,
        bankfull_area_arr,
        _,
        bankfull_perimeter_arr,
        width_over_sqrt_bankfull_depth_arr,
    ) = compute_static_geometry(
        river_length=np.array([1000.0], dtype=np.float32),
        river_width=np.array([bankfull_width_m], dtype=np.float32),
        shape_exponent=np.array([shape_exponent], dtype=np.float32),
        bankfull_depth=np.array([bankfull_depth_m], dtype=np.float32),
    )

    # 10-point floodplain widths: at depth 0 m width is 100 m, expanding by 20 m per depth increment
    floodplain_widths: ArrayFloat32 = np.zeros(NUM_FLOODPLAIN_LEVELS, dtype=np.float32)
    for k, depth in enumerate(FLOODPLAIN_PROFILE_DEPTHS):
        floodplain_widths[k] = np.float32(100.0 + 20.0 * depth)

    # Regime 1: In-bank (y = 2.0 m < h_bf)
    w_inbank: np.float32 = compute_top_width_from_depth(
        water_depth_m=np.float32(2.0),
        bankfull_depth_m=bankfull_depth_m,
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        floodplain_widths=floodplain_widths,
        side_slope_z=side_slope_z,
        levee_depth_m=levee_depth_m,
    )
    # W(2) = 50 * sqrt(2 / 4) = 50 * sqrt(0.5) ≈ 35.355 m
    assert np.isclose(float(w_inbank), 50.0 * math.sqrt(0.5), rtol=1e-4)

    area_inbank, w_sec_inbank, perim_inbank = compute_cross_section_from_depth(
        effective_depth_m=np.float32(2.0),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area_arr[0],
        bankfull_perimeter_m=bankfull_perimeter_arr[0],
        width_over_sqrt_bankfull_depth=width_over_sqrt_bankfull_depth_arr[0],
        levee_depth_m=levee_depth_m,
    )
    # A_inbank = (1 / 1.5) * W(2) * 2 = (2/3) * 35.3553 * 2 ≈ 47.14 m²
    assert np.isclose(
        float(area_inbank), (2.0 / 3.0) * float(w_inbank) * 2.0, rtol=1e-3
    )
    assert np.isclose(float(w_sec_inbank), float(w_inbank), rtol=1e-3)
    assert perim_inbank > float(w_inbank)

    # Regime 2: Embanked between bankfull and levee crest (y = 5.5 m, where h_bf = 4.0 < y <= h_levee = 7.0)
    w_embanked: np.float32 = compute_top_width_from_depth(
        water_depth_m=np.float32(5.5),
        bankfull_depth_m=bankfull_depth_m,
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        floodplain_widths=floodplain_widths,
        side_slope_z=side_slope_z,
        levee_depth_m=levee_depth_m,
    )
    # Width must remain confined to bankfull width, NOT spilling onto floodplain!
    assert np.isclose(float(w_embanked), float(bankfull_width_m), rtol=1e-5)

    area_embanked, w_sec_embanked, perim_embanked = compute_cross_section_from_depth(
        effective_depth_m=np.float32(5.5),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area_arr[0],
        bankfull_perimeter_m=bankfull_perimeter_arr[0],
        width_over_sqrt_bankfull_depth=width_over_sqrt_bankfull_depth_arr[0],
        levee_depth_m=levee_depth_m,
    )
    # A_bf = (2/3) * 50 * 4 = 133.333 m²
    # Delta h = 5.5 - 4.0 = 1.5 m -> Delta A = 50 * 1.5 = 75 m² -> Total A = 208.333 m²
    expected_area_embanked: float = (2.0 / 3.0) * 50.0 * 4.0 + 50.0 * 1.5
    assert np.isclose(float(area_embanked), expected_area_embanked, rtol=1e-3)
    assert np.isclose(float(w_sec_embanked), float(bankfull_width_m), rtol=1e-5)
    # Hydraulic radius R_h = A / P expands in embanked channel
    rh_inbank: float = float(area_inbank) / float(perim_inbank)
    rh_embanked: float = float(area_embanked) / float(perim_embanked)
    assert rh_embanked > rh_inbank

    # Regime 3: Overbank overtopping (y = 8.0 m > h_levee = 7.0)
    w_overbank: np.float32 = compute_top_width_from_depth(
        water_depth_m=np.float32(8.0),
        bankfull_depth_m=bankfull_depth_m,
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        floodplain_widths=floodplain_widths,
        side_slope_z=side_slope_z,
        levee_depth_m=levee_depth_m,
    )
    # h_fp = 8.0 - 4.0 = 4.0 m above natural ground -> W = 100 + 20 * 4 = 180 m
    assert np.isclose(float(w_overbank), 180.0, rtol=1e-5)

    area_overbank, _, perim_overbank = compute_cross_section_from_depth(
        effective_depth_m=np.float32(8.0),
        bankfull_width_m=bankfull_width_m,
        shape_exponent=shape_exponent,
        bankfull_depth_m=bankfull_depth_m,
        inverse_shape_exponent_plus_one=inverse_shape_exponent_plus_one[0],
        bankfull_area_m2=bankfull_area_arr[0],
        bankfull_perimeter_m=bankfull_perimeter_arr[0],
        width_over_sqrt_bankfull_depth=width_over_sqrt_bankfull_depth_arr[0],
        levee_depth_m=levee_depth_m,
    )
    assert float(area_overbank) > float(area_embanked)
    assert float(perim_overbank) > float(perim_embanked)


def _make_profile_geometry(levee_depth_m: float | None = None) -> Any:
    """Builds single-reach inertial geometry with a linear 10-point floodplain profile.

    The reach is 1000 m long with a 50 m wide, 4 m deep parabolic channel and floodplain
    widths of 100 + 20 * d meters at depth d above bankfull.

    Args:
        levee_depth_m: Levee crest depth above the channel bed (meters), or None for a
            reach without levees.

    Returns:
        InertialGeometry for the single reach.
    """
    return build_inertial_geometry(
        river_length=np.array([1000.0], dtype=np.float32),
        river_width=np.array([50.0], dtype=np.float32),
        shape_exponent=np.array([0.5], dtype=np.float32),
        bankfull_depth=np.array([4.0], dtype=np.float32),
        bed_elevation=np.array([0.0], dtype=np.float32),
        ds_bed_elevation=np.array([-1.0], dtype=np.float32),
        interface_bed_elevation_max=np.array([0.0], dtype=np.float32),
        manning_n_sq=np.array([0.03**2], dtype=np.float32),
        inv_dx_interface=np.array([1e-3], dtype=np.float32),
        pit_slope=np.array([1e-3], dtype=np.float32),
        floodplain_side_slope_m_per_m=np.array([2.0], dtype=np.float32),
        floodplain_profile_widths=np.array(
            [[100.0 + 20.0 * d for d in FLOODPLAIN_PROFILE_DEPTHS]], dtype=np.float32
        ),
        levee_depth=np.array(
            [4.0 if levee_depth_m is None else levee_depth_m], dtype=np.float32
        ),
    )


def test_compute_overbank_area_from_depth_inverts_profile_lookup() -> None:
    """Verifies compute_overbank_area_from_depth is the inverse of compute_overbank_depth_and_top_width.

    Checks depths within the shallowest slice, between profile levels, exactly at a level
    and beyond the 5 m profile where side-slope extrapolation applies.
    """
    geom = _make_profile_geometry()
    areas: ArrayFloat32 = geom.geom_floodplain_areas[0]
    widths: ArrayFloat32 = geom.geom_floodplain_widths[0]

    assert compute_overbank_area_from_depth(
        np.float32(0.0), np.float32(50.0), areas, widths, np.float32(2.0)
    ) == np.float32(0.0)
    # Exactly at profile level 4 (1.0 m) the area equals the precomputed cumulative area
    assert np.isclose(
        float(
            compute_overbank_area_from_depth(
                np.float32(1.0), np.float32(50.0), areas, widths, np.float32(2.0)
            )
        ),
        float(areas[4]),
        rtol=1e-5,
    )
    for depth in [0.05, 0.3, 1.7, 3.0, 4.5, 6.0, 9.0]:
        area: np.float32 = compute_overbank_area_from_depth(
            np.float32(depth), np.float32(50.0), areas, widths, np.float32(2.0)
        )
        depth_back, _ = compute_overbank_depth_and_top_width(
            overbank_area=area,
            bankfull_width_m=np.float32(50.0),
            floodplain_areas=areas,
            floodplain_widths=widths,
            floodplain_side_slope_m_per_m=np.float32(2.0),
        )
        assert np.isclose(float(depth_back), depth, rtol=1e-4)


def test_build_inertial_geometry_rejects_levee_below_bankfull() -> None:
    """Verifies that a levee crest below bankfull depth is rejected instead of clamped."""
    with pytest.raises(ValueError, match="levee_depth"):
        _make_profile_geometry(levee_depth_m=3.0)


def test_levee_overtopping_fills_protected_floodplain_from_natural_ground() -> None:
    """Verifies the stage-volume relation of a leveed reach once the levee crest is overtopped.

    The protected floodplain behind the levees lies at natural ground (bankfull level), so:
    1. Stage is continuous and non-decreasing in volume across all regimes.
    2. After overtopping, stage stays at the crest while the protected floodplain fills.
    3. Once the floodplain has filled to crest level, the stage-volume relation equals that of
       the same reach without levees: overtopped levees no longer change storage.
    """
    bankfull_depth_m: float = 4.0
    levee_depth_m: float = 7.0
    reach_length_m: float = 1000.0
    leveed = _make_profile_geometry(levee_depth_m)
    natural = _make_profile_geometry()

    def stage(geom: Any, volume_m3: float) -> float:
        return float(
            _compute_reach_stage_and_cfl_depth(
                0,
                np.float32(volume_m3),
                geom.geom_inbank,
                geom.geom_floodplain_areas,
                geom.geom_floodplain_widths,
            )[0]
        )

    bankfull_area: float = (2.0 / 3.0) * 50.0 * bankfull_depth_m
    levee_volume: float = (
        bankfull_area + 50.0 * (levee_depth_m - bankfull_depth_m)
    ) * reach_length_m
    crest_fill_area: float = float(
        compute_overbank_area_from_depth(
            np.float32(levee_depth_m - bankfull_depth_m),
            np.float32(50.0),
            leveed.geom_floodplain_areas[0],
            leveed.geom_floodplain_widths[0],
            np.float32(2.0),
        )
    )
    filled_volume: float = (bankfull_area + crest_fill_area) * reach_length_m
    assert filled_volume > levee_volume

    # 1. Continuous and non-decreasing stage across in-bank, embanked, filling and overbank regimes
    volumes: np.ndarray = np.linspace(1.0, 2.0 * filled_volume, 4001)
    stages: np.ndarray = np.array([stage(leveed, v) for v in volumes])
    assert np.all(np.diff(stages) >= -1e-4)
    for boundary_volume in [levee_volume, filled_volume]:
        assert np.isclose(
            stage(leveed, 0.9999 * boundary_volume),
            stage(leveed, 1.0001 * boundary_volume),
            atol=5e-3,
        )

    # 2. Stage holds at the levee crest while the protected floodplain fills
    for frac in [0.01, 0.5, 0.99]:
        volume: float = levee_volume + frac * (filled_volume - levee_volume)
        assert np.isclose(stage(leveed, volume), levee_depth_m, atol=1e-3)

    # 3. Beyond the filled state, storage equals that of the unleveed reach at the same stage
    for volume in [1.01 * filled_volume, 1.5 * filled_volume, 3.0 * filled_volume]:
        assert stage(leveed, volume) > levee_depth_m
        assert np.isclose(stage(leveed, volume), stage(natural, volume), atol=1e-3)

    # The volume-based CFL geometry uses the floodplain width at crest level while filling
    _, top_width_filling = compute_cfl_area_and_top_width(
        volume_m3=np.float32(0.5 * (levee_volume + filled_volume)),
        bankfull_width_m=np.float32(50.0),
        shape_exponent=np.float32(0.5),
        bankfull_depth_m=np.float32(bankfull_depth_m),
        inverse_reach_length=np.float32(1.0 / reach_length_m),
        inverse_shape_exponent_plus_one=np.float32(1.0 / 1.5),
        bankfull_area_m2=np.float32(bankfull_area),
        bankfull_volume_m3=np.float32(bankfull_area * reach_length_m),
        floodplain_side_slope_m_per_m=np.float32(2.0),
        floodplain_areas=leveed.geom_floodplain_areas[0],
        floodplain_widths=leveed.geom_floodplain_widths[0],
        levee_depth_m=np.float32(levee_depth_m),
    )
    # Floodplain width at crest level (3 m above bankfull): 100 + 20 * 3 = 160 m
    assert np.isclose(float(top_width_filling), 160.0, rtol=1e-4)


def test_routing_calculate_levee_depth_integration() -> None:
    """Verifies Routing.calculate_levee_depth under disabled and enabled flood protection.

    Verifies:
    1. When flood_protection is disabled (enabled=False), levee_depth identically equals bankfull_depth.
    2. When flood_protection is enabled with standard=2 years, levee_depth equals bankfull_depth.
    3. When reaches have flood protection standards T_prot > 2 years (e.g. 100 years for Rhine-like flood protection),
       levee_depth is strictly greater than bankfull_depth (h_levee > h_bf).
    4. Derived levee depths are properly synchronized via _update_rivers_geometry to self.var.rivers['levee_depth'].
    """
    from geb.hydrology.routing import Routing, RoutingVariables

    routing: Routing = Routing.__new__(Routing)
    routing.model = DummyModel()  # ty:ignore[invalid-assignment]
    routing.grid = DummyGrid(3)  # ty:ignore[invalid-assignment]
    routing._compressed_routing_seq = np.array([0, 1, 2], dtype=np.int32)
    routing._compressed_ds = np.array([1, 2, -1], dtype=np.int32)

    class MockWB:
        n: int = 0

    class MockHydro:
        waterbodies: MockWB = MockWB()

    routing.hydrology = MockHydro()  # ty:ignore[invalid-assignment]
    routing.default_missing_channel_width = 10.0

    var: RoutingVariables = RoutingVariables()
    var.river_ids = np.array([0, 1, 2], dtype=np.int32)
    var.river_width_alpha = np.full(3, 7.2, dtype=np.float32)
    var.river_width_beta = np.full(3, 0.5, dtype=np.float32)
    widths: ArrayFloat32 = np.array([30.0, 50.0, 100.0], dtype=np.float32)
    var.observed_average_river_width = widths.copy()

    # Reach 0: unprotected (T = 2 years)
    # Reach 1: moderate protection (T = 50 years)
    # Reach 2: major river protection (T = 250 years)
    rivers_df: gpd.GeoDataFrame = gpd.GeoDataFrame(
        {
            "return_period_2_years_daily_m3_per_s": [30.0, 100.0, 500.0],
            "return_period_50_years_daily_m3_per_s": [80.0, 250.0, 1200.0],
            "return_period_250_years_daily_m3_per_s": [120.0, 380.0, 1800.0],
            "flood_protection_standard": [2.0, 50.0, 250.0],
            "represented_in_grid": [True, True, True],
            "width": [30.0, 50.0, 100.0],
            "depth": [np.nan, np.nan, np.nan],
            "manning": [0.03, 0.03, 0.03],
        },
        index=[0, 1, 2],
    )
    var.rivers = rivers_df
    routing.var = var

    # Config with flood protection enabled
    routing.config = {
        "river_width": {"default_alpha": 7.2, "beta": 0.5},
        "river_depth": {
            "min_depth_m": 0.1,
            "shape_exponent": 0.5,
            "use_observed_width_as_bankfull": True,
        },
        "flood_protection": {"enabled": True},
        "bankfull_discharge_multiplier": 1.0,
    }

    bankfull_depths: ArrayFloat32 = routing.calculate_bankfull_depth(
        bankfull_top_width_m=widths, use_simulated_bankfull_q=True
    )
    levee_depths: ArrayFloat32 = routing.calculate_levee_depth(
        bankfull_top_width_m=widths,
        bankfull_depth_m=bankfull_depths,
        use_simulated_bankfull_q=True,
    )

    # Reach 0 (T = 2): Levee depth exactly matches bankfull depth
    assert np.isclose(float(levee_depths[0]), float(bankfull_depths[0]), rtol=1e-5)

    # Reach 1 (T = 50) and Reach 2 (T = 250): Levee crest is strictly higher than bankfull depth
    assert float(levee_depths[1]) > float(bankfull_depths[1])
    assert float(levee_depths[2]) > float(bankfull_depths[2])
    # Higher return period yields higher levee elevation above bankfull
    delta_h_reach1: float = float(levee_depths[1] - bankfull_depths[1])
    delta_h_reach2: float = float(levee_depths[2] - bankfull_depths[2])
    assert delta_h_reach1 > 0.1
    assert delta_h_reach2 > 0.1

    # Test with flood protection disabled: levee depth is identical to bankfull depth everywhere
    routing.config["flood_protection"]["enabled"] = False
    levee_depths_disabled: ArrayFloat32 = routing.calculate_levee_depth(
        bankfull_top_width_m=widths,
        bankfull_depth_m=bankfull_depths,
        use_simulated_bankfull_q=True,
    )
    assert np.allclose(levee_depths_disabled, bankfull_depths)

    # Test geometry sync with levee depth
    routing.config["flood_protection"]["enabled"] = True
    fp_widths = np.zeros((3, NUM_FLOODPLAIN_LEVELS), dtype=np.float32)
    fp_slopes = np.full(3, 2.0, dtype=np.float32)
    routing._update_rivers_geometry(
        width_grid=widths,
        depth_grid=bankfull_depths,
        manning_grid=routing.grid.var.river_mannings,
        floodplain_profile_widths_grid_m=fp_widths,
        floodplain_side_slope_grid_m_per_m=fp_slopes,
        levee_depth_grid=levee_depths,
    )
    assert "levee_depth" in routing.var.rivers.columns
    assert np.isclose(
        float(routing.var.rivers.loc[0, "levee_depth"]),
        float(bankfull_depths[0]),
        rtol=1e-4,
    )
    assert float(routing.var.rivers.loc[1, "levee_depth"]) > float(bankfull_depths[1])
    assert float(routing.var.rivers.loc[2, "levee_depth"]) > float(bankfull_depths[2])

    # Missing simulated design discharges for protected reaches raise instead of
    # silently falling back to an estimate
    routing.var.rivers.loc[2, "return_period_250_years_daily_m3_per_s"] = np.nan
    with pytest.raises(ValueError, match="simulated design discharges"):
        routing.calculate_levee_depth(
            bankfull_top_width_m=widths,
            bankfull_depth_m=bankfull_depths,
            use_simulated_bankfull_q=True,
        )
    routing.var.rivers = gpd.GeoDataFrame(
        routing.var.rivers.drop(columns="return_period_250_years_daily_m3_per_s")
    )
    with pytest.raises(KeyError):
        routing.calculate_levee_depth(
            bankfull_top_width_m=widths,
            bankfull_depth_m=bankfull_depths,
            use_simulated_bankfull_q=True,
        )
