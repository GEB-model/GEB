"""Unit tests for river-to-groundwater transmission loss and wetted channel area calculation."""

import numpy as np
import pytest

from geb.hydrology.routing import (
    calculate_potential_river_transmission_loss,
    calculate_wetted_channel_area,
)
from geb.hydrology.routing.geometry import (
    NUM_FLOODPLAIN_LEVELS,
)


def test_calculate_wetted_channel_area_kinematic() -> None:
    """Verify kinematic reaches use hydraulic geometry power-law w = alpha * |Q|^beta."""
    n: int = 1
    use_kinematic: np.ndarray = np.array([True], dtype=bool)
    water_stage_m: np.ndarray = np.array([0.0], dtype=np.float32)
    bed_elevation_m: np.ndarray = np.array([0.0], dtype=np.float32)
    bankfull_width_m: np.ndarray = np.array([20.0], dtype=np.float32)
    bankfull_depth_m: np.ndarray = np.array([2.0], dtype=np.float32)
    shape_exponent: np.ndarray = np.array([0.5], dtype=np.float32)
    floodplain_widths: np.ndarray = np.full(
        (1, NUM_FLOODPLAIN_LEVELS), 50.0, dtype=np.float32
    )
    floodplain_side_slope: np.ndarray = np.array([2.0], dtype=np.float32)
    alpha: np.ndarray = np.array([3.91], dtype=np.float32)
    beta: np.ndarray = np.array([0.62], dtype=np.float32)
    discharge: np.ndarray = np.array([10.0], dtype=np.float32)
    river_length: np.ndarray = np.array([1000.0], dtype=np.float32)
    cell_area: np.ndarray = np.array([1e6], dtype=np.float32)
    is_wb: np.ndarray = np.array([False], dtype=bool)
    wetted_area: np.ndarray = np.zeros(n, dtype=np.float32)

    calculate_wetted_channel_area(
        use_kinematic=use_kinematic,
        water_stage_m=water_stage_m,
        bed_elevation_m=bed_elevation_m,
        bankfull_width_m=bankfull_width_m,
        bankfull_depth_m=bankfull_depth_m,
        shape_exponent=shape_exponent,
        floodplain_profile_widths_m=floodplain_widths,
        floodplain_side_slope_m_per_m=floodplain_side_slope,
        river_width_alpha=alpha,
        river_width_beta=beta,
        discharge_m3_s=discharge,
        river_length_m=river_length,
        cell_area_m2=cell_area,
        is_waterbody=is_wb,
        wetted_channel_area_m2=wetted_area,
    )

    expected_width: float = float(3.91 * (10.0**0.62))
    expected_area: float = min(1e6, expected_width * 1000.0)
    assert np.isclose(wetted_area[0], expected_area, rtol=1e-4)


def test_calculate_wetted_channel_area_local_inertial_inbank() -> None:
    """Verify local inertial reaches use parabolic/power-law geometry for in-bank flow."""
    n: int = 1
    use_kinematic: np.ndarray = np.array([False], dtype=bool)
    bed_elev: np.ndarray = np.array([100.0], dtype=np.float32)
    stage: np.ndarray = np.array([100.5], dtype=np.float32)  # depth = 0.5 m (in-bank)
    bankfull_width: np.ndarray = np.array([20.0], dtype=np.float32)
    bankfull_depth: np.ndarray = np.array([2.0], dtype=np.float32)
    shape_exponent: np.ndarray = np.array([0.5], dtype=np.float32)  # parabolic
    floodplain_widths: np.ndarray = np.full(
        (1, NUM_FLOODPLAIN_LEVELS), 100.0, dtype=np.float32
    )
    floodplain_side_slope: np.ndarray = np.array([2.0], dtype=np.float32)
    alpha: np.ndarray = np.array([0.0], dtype=np.float32)
    beta: np.ndarray = np.array([0.0], dtype=np.float32)
    discharge: np.ndarray = np.array(
        [0.0], dtype=np.float32
    )  # zero flow, standing water
    river_length: np.ndarray = np.array([500.0], dtype=np.float32)
    cell_area: np.ndarray = np.array([1e6], dtype=np.float32)
    is_wb: np.ndarray = np.array([False], dtype=bool)
    wetted_area: np.ndarray = np.zeros(n, dtype=np.float32)

    calculate_wetted_channel_area(
        use_kinematic=use_kinematic,
        water_stage_m=stage,
        bed_elevation_m=bed_elev,
        bankfull_width_m=bankfull_width,
        bankfull_depth_m=bankfull_depth,
        shape_exponent=shape_exponent,
        floodplain_profile_widths_m=floodplain_widths,
        floodplain_side_slope_m_per_m=floodplain_side_slope,
        river_width_alpha=alpha,
        river_width_beta=beta,
        discharge_m3_s=discharge,
        river_length_m=river_length,
        cell_area_m2=cell_area,
        is_waterbody=is_wb,
        wetted_channel_area_m2=wetted_area,
    )

    # depth = 0.5 m, bankfull depth = 2.0 m -> depth ratio = 0.25 -> sqrt(0.25) = 0.5
    # top width = 20.0 * 0.5 = 10.0 m
    expected_width: float = 10.0
    expected_area: float = expected_width * 500.0
    assert np.isclose(wetted_area[0], expected_area, rtol=1e-4)


def test_calculate_wetted_channel_area_local_inertial_overbank() -> None:
    """Verify local inertial reaches use multi-level floodplain profiles during overbank flooding."""
    n: int = 1
    use_kinematic: np.ndarray = np.array([False], dtype=bool)
    bed_elev: np.ndarray = np.array([100.0], dtype=np.float32)
    stage: np.ndarray = np.array(
        [102.5], dtype=np.float32
    )  # depth = 2.5 m (0.5 m overbank)
    bankfull_width: np.ndarray = np.array([20.0], dtype=np.float32)
    bankfull_depth: np.ndarray = np.array([2.0], dtype=np.float32)
    shape_exponent: np.ndarray = np.array([0.5], dtype=np.float32)
    # Profile depths: 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0
    # Overbank depth = 0.5 matches profile level index 2 (FLOODPLAIN_PROFILE_DEPTHS[2] == 0.5)
    floodplain_widths: np.ndarray = np.zeros(
        (1, NUM_FLOODPLAIN_LEVELS), dtype=np.float32
    )
    floodplain_widths[0, 2] = 80.0  # 80 m at 0.5 m overbank
    floodplain_side_slope: np.ndarray = np.array([2.0], dtype=np.float32)
    alpha: np.ndarray = np.array([0.0], dtype=np.float32)
    beta: np.ndarray = np.array([0.0], dtype=np.float32)
    discharge: np.ndarray = np.array([100.0], dtype=np.float32)
    river_length: np.ndarray = np.array([500.0], dtype=np.float32)
    cell_area: np.ndarray = np.array([1e6], dtype=np.float32)
    is_wb: np.ndarray = np.array([False], dtype=bool)
    wetted_area: np.ndarray = np.zeros(n, dtype=np.float32)

    calculate_wetted_channel_area(
        use_kinematic=use_kinematic,
        water_stage_m=stage,
        bed_elevation_m=bed_elev,
        bankfull_width_m=bankfull_width,
        bankfull_depth_m=bankfull_depth,
        shape_exponent=shape_exponent,
        floodplain_profile_widths_m=floodplain_widths,
        floodplain_side_slope_m_per_m=floodplain_side_slope,
        river_width_alpha=alpha,
        river_width_beta=beta,
        discharge_m3_s=discharge,
        river_length_m=river_length,
        cell_area_m2=cell_area,
        is_waterbody=is_wb,
        wetted_channel_area_m2=wetted_area,
    )

    expected_width: float = 80.0
    expected_area: float = expected_width * 500.0
    assert np.isclose(wetted_area[0], expected_area, rtol=1e-4)


def test_potential_transmission_loss_disconnected_unit_gradient() -> None:
    """Verify Darcy bed infiltration when groundwater head is below riverbed (unit gradient)."""
    n: int = 1
    wetted_area: np.ndarray = np.array([10000.0], dtype=np.float32)
    riverbed_k: np.ndarray = np.array([1e-5], dtype=np.float32)
    bed_elev: np.ndarray = np.array([100.0], dtype=np.float32)
    stage: np.ndarray = np.array([101.5], dtype=np.float32)  # depth = 1.5 m
    gw_head: np.ndarray = np.array([95.0], dtype=np.float32)  # deep table
    dt: np.float32 = np.float32(3600.0)
    out: np.ndarray = np.zeros(n, dtype=np.float32)

    calculate_potential_river_transmission_loss(
        wetted_channel_area_m2=wetted_area,
        riverbed_hydraulic_conductivity_m_s=riverbed_k,
        bed_elevation_m=bed_elev,
        water_stage_m=stage,
        groundwater_head_m=gw_head,
        timestep_s=dt,
        potential_transmission_loss_m3=out,
    )

    expected_vol: float = 1e-5 * 1.0 * 10000.0 * 3600.0
    assert np.isclose(out[0], expected_vol, rtol=1e-4)
    assert out[0] > 0.0


def test_potential_transmission_loss_gaining_stream_zero_loss() -> None:
    """Verify transmission loss shuts down when groundwater head exceeds stream stage."""
    n: int = 1
    wetted_area: np.ndarray = np.array([10000.0], dtype=np.float32)
    riverbed_k: np.ndarray = np.array([1e-5], dtype=np.float32)
    bed_elev: np.ndarray = np.array([100.0], dtype=np.float32)
    stage: np.ndarray = np.array([101.5], dtype=np.float32)
    gw_head: np.ndarray = np.array(
        [102.0], dtype=np.float32
    )  # water table above river stage
    dt: np.float32 = np.float32(3600.0)
    out: np.ndarray = np.zeros(n, dtype=np.float32)

    calculate_potential_river_transmission_loss(
        wetted_channel_area_m2=wetted_area,
        riverbed_hydraulic_conductivity_m_s=riverbed_k,
        bed_elevation_m=bed_elev,
        water_stage_m=stage,
        groundwater_head_m=gw_head,
        timestep_s=dt,
        potential_transmission_loss_m3=out,
    )

    assert out[0] == 0.0


def test_potential_transmission_loss_partially_connected_gradient() -> None:
    """Verify reduced gradient when groundwater head is between bed and stage."""
    n: int = 1
    wetted_area: np.ndarray = np.array([10000.0], dtype=np.float32)
    riverbed_k: np.ndarray = np.array([1e-5], dtype=np.float32)
    bed_elev: np.ndarray = np.array([100.0], dtype=np.float32)
    stage: np.ndarray = np.array([102.0], dtype=np.float32)  # depth = 2.0 m
    gw_head: np.ndarray = np.array(
        [101.0], dtype=np.float32
    )  # head is halfway up depth
    dt: np.float32 = np.float32(3600.0)
    out: np.ndarray = np.zeros(n, dtype=np.float32)

    calculate_potential_river_transmission_loss(
        wetted_channel_area_m2=wetted_area,
        riverbed_hydraulic_conductivity_m_s=riverbed_k,
        bed_elevation_m=bed_elev,
        water_stage_m=stage,
        groundwater_head_m=gw_head,
        timestep_s=dt,
        potential_transmission_loss_m3=out,
    )

    # Expected gradient: (stage - gw_head) / depth = (102.0 - 101.0) / 2.0 = 0.5
    expected_vol: float = 1e-5 * 0.5 * 10000.0 * 3600.0
    assert np.isclose(out[0], expected_vol, rtol=1e-4)


@pytest.mark.parametrize(
    ("area", "k_bed"),
    [
        (0.0, 1e-5),  # dry reach (zero wetted area)
        (10000.0, 0.0),  # impermeable bed (k_bed = 0)
    ],
)
def test_potential_transmission_loss_edge_cases(area: float, k_bed: float) -> None:
    """Verify potential loss is zero for boundary edge cases."""
    out: np.ndarray = np.zeros(1, dtype=np.float32)
    calculate_potential_river_transmission_loss(
        wetted_channel_area_m2=np.array([area], dtype=np.float32),
        riverbed_hydraulic_conductivity_m_s=np.array([k_bed], dtype=np.float32),
        bed_elevation_m=np.array([100.0], dtype=np.float32),
        water_stage_m=np.array([101.0], dtype=np.float32),
        groundwater_head_m=np.array([90.0], dtype=np.float32),
        timestep_s=np.float32(3600.0),
        potential_transmission_loss_m3=out,
    )
    assert out[0] == 0.0
