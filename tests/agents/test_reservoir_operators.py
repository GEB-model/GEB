"""Unit tests for reservoir operator functions in the GEB agents module."""

import matplotlib.pyplot as plt
import numpy as np

from geb.agents.reservoir_operators import get_irrigation_reservoir_release

from ..testconfig import output_folder


def _run_release_scenario(
    current_demands_m3: np.ndarray,
    *,
    long_term_demand_m3: float | np.ndarray,
    inflow_m3: float = 100.0,
    storage_year_start_m3: float = 800.0,
    capacity_m3: float = 1000.0,
    alpha: float = 0.85,
    reservoir_m_factor: float = 0.7,
    n_monthly_substeps: int = 30,
) -> np.ndarray:
    """Run the irrigation release function for a single scenario.

    Args:
        current_demands_m3: Current irrigation demand over the tested range (m3).
        long_term_demand_m3: Long-term monthly irrigation demand (m3).
        inflow_m3: Long-term monthly inflow and current-month inflow (m3).
        storage_year_start_m3: Storage at the start of the hydrological year (m3).
        capacity_m3: Reservoir capacity (m3).
        alpha: Reservoir capacity reduction factor (-).
        reservoir_m_factor: Reservoir M factor (-).
        n_monthly_substeps: Number of substeps in the month (-).

    Returns:
        Irrigation release per substep for the scenario (m3).
    """
    capacity_array = np.full_like(current_demands_m3, capacity_m3, dtype=np.float64)
    storage_year_start_array = np.full_like(
        current_demands_m3, storage_year_start_m3, dtype=np.float64
    )
    long_term_inflow_array = np.full_like(
        current_demands_m3, inflow_m3, dtype=np.float32
    )
    long_term_inflow_this_month_array = np.full_like(
        current_demands_m3, inflow_m3, dtype=np.float32
    )
    alpha_array = np.full_like(current_demands_m3, alpha, dtype=np.float32)
    reservoir_m_factor_array = np.full_like(
        current_demands_m3, reservoir_m_factor, dtype=np.float32
    )
    if np.isscalar(long_term_demand_m3):
        long_term_demand_array = np.full_like(
            current_demands_m3, long_term_demand_m3, dtype=np.float32
        )
    else:
        long_term_demand_array = np.asarray(long_term_demand_m3, dtype=np.float32)

    return get_irrigation_reservoir_release(
        capacity_array,
        storage_year_start_array,
        long_term_inflow_array,
        long_term_inflow_this_month_array,
        current_demands_m3,
        long_term_demand_array,
        alpha_array,
        reservoir_m_factor_array,
        n_monthly_substeps,
    )


def test_get_irrigation_reservoir_release_basic() -> None:
    """Test basic functionality of irrigation reservoir release."""
    capacity = np.array([1000.0], dtype=np.float64)
    storage_year_start = np.array([800.0], dtype=np.float64)
    long_term_monthly_inflow_m3 = np.array([100.0], dtype=np.float32)
    long_term_monthly_inflow_this_month_m3 = np.array([100.0], dtype=np.float32)
    current_irrigation_demand_m3 = np.array([50.0], dtype=np.float32)
    long_term_monthly_irrigation_demand_m3 = np.array([50.0], dtype=np.float32)
    alpha = np.array([0.85], dtype=np.float32)
    reservoir_M_factor = np.array([0.7], dtype=np.float32)
    n_monthly_substeps = 30

    release = get_irrigation_reservoir_release(
        capacity,
        storage_year_start,
        long_term_monthly_inflow_m3,
        long_term_monthly_inflow_this_month_m3,
        current_irrigation_demand_m3,
        long_term_monthly_irrigation_demand_m3,
        alpha,
        reservoir_M_factor,
        n_monthly_substeps,
    )

    assert release.shape == (1,)
    assert release[0] >= 0


def test_get_irrigation_reservoir_release_zero_inflow() -> None:
    """Test that zero inflow leads to np.inf ratio but doesn't crash."""
    capacity = np.array([1000.0], dtype=np.float64)
    storage_year_start = np.array([800.0], dtype=np.float64)
    long_term_monthly_inflow_m3 = np.array([0.0], dtype=np.float32)
    long_term_monthly_inflow_this_month_m3 = np.array([0.0], dtype=np.float32)
    current_irrigation_demand_m3 = np.array([50.0], dtype=np.float32)
    long_term_monthly_irrigation_demand_m3 = np.array([50.0], dtype=np.float32)
    alpha = np.array([0.85], dtype=np.float32)
    reservoir_M_factor = np.array([0.7], dtype=np.float32)
    n_monthly_substeps = 30

    # This should not raise DivisionByZero and should handle the inf ratio correctly
    release = get_irrigation_reservoir_release(
        capacity,
        storage_year_start,
        long_term_monthly_inflow_m3,
        long_term_monthly_inflow_this_month_m3,
        current_irrigation_demand_m3,
        long_term_monthly_irrigation_demand_m3,
        alpha,
        reservoir_M_factor,
        n_monthly_substeps,
    )

    assert not np.isnan(release).any()
    assert release[0] >= 0


def test_get_irrigation_reservoir_release_dominant_demand() -> None:
    """Test when irrigation demand is dominant (ratio > 1 - M)."""
    capacity = np.array([1000.0], dtype=np.float64)
    storage_year_start = np.array([800.0], dtype=np.float64)
    long_term_monthly_inflow_m3 = np.array([100.0], dtype=np.float32)
    long_term_monthly_inflow_this_month_m3 = np.array([100.0], dtype=np.float32)
    # Demand 80, Inflow 100 -> Ratio 0.8. 0.8 > 1 - 0.7 (0.3) -> Dominant
    current_irrigation_demand_m3 = np.array([80.0], dtype=np.float32)
    long_term_monthly_irrigation_demand_m3 = np.array([80.0], dtype=np.float32)
    alpha = np.array([0.85], dtype=np.float32)
    reservoir_M_factor = np.array([0.7], dtype=np.float32)
    n_monthly_substeps = 30

    release = get_irrigation_reservoir_release(
        capacity,
        storage_year_start,
        long_term_monthly_inflow_m3,
        long_term_monthly_inflow_this_month_m3,
        current_irrigation_demand_m3,
        long_term_monthly_irrigation_demand_m3,
        alpha,
        reservoir_M_factor,
        n_monthly_substeps,
    )

    assert release[0] >= 0


def test_get_irrigation_reservoir_release_non_dominant_demand() -> None:
    """Test when irrigation demand is not dominant."""
    capacity = np.array([1000.0], dtype=np.float64)
    storage_year_start = np.array([800.0], dtype=np.float64)
    long_term_monthly_inflow_m3 = np.array([100.0], dtype=np.float32)
    long_term_monthly_inflow_this_month_m3 = np.array([100.0], dtype=np.float32)
    # Demand 10, Inflow 100 -> Ratio 0.1. 0.1 < 1 - 0.7 (0.3) -> Non-dominant
    current_irrigation_demand_m3 = np.array([10.0], dtype=np.float32)
    long_term_monthly_irrigation_demand_m3 = np.array([10.0], dtype=np.float32)
    alpha = np.array([0.85], dtype=np.float32)
    reservoir_M_factor = np.array([0.7], dtype=np.float32)
    n_monthly_substeps = 30

    release = get_irrigation_reservoir_release(
        capacity,
        storage_year_start,
        long_term_monthly_inflow_m3,
        long_term_monthly_inflow_this_month_m3,
        current_irrigation_demand_m3,
        long_term_monthly_irrigation_demand_m3,
        alpha,
        reservoir_M_factor,
        n_monthly_substeps,
    )

    assert release[0] >= 0


def test_plot_irrigation_reservoir_release_sensitivity() -> None:
    """Visualize release behaviour across representative operating conditions."""
    inflow_m3 = 100.0
    reservoir_m_factor = 0.7
    threshold_m3 = inflow_m3 * (1.0 - reservoir_m_factor)
    current_demands_m3 = np.linspace(0.001, 250.0, 200, dtype=np.float32)

    releases_non_dominant_m3 = _run_release_scenario(
        current_demands_m3,
        long_term_demand_m3=10.0,
        inflow_m3=inflow_m3,
        reservoir_m_factor=reservoir_m_factor,
    )
    releases_dominant_transition_m3 = _run_release_scenario(
        current_demands_m3,
        long_term_demand_m3=80.0,
        inflow_m3=inflow_m3,
        reservoir_m_factor=reservoir_m_factor,
    )
    releases_equal_demands_m3 = _run_release_scenario(
        current_demands_m3,
        long_term_demand_m3=current_demands_m3,
        inflow_m3=inflow_m3,
        reservoir_m_factor=reservoir_m_factor,
    )
    releases_low_storage_m3 = _run_release_scenario(
        current_demands_m3,
        long_term_demand_m3=80.0,
        inflow_m3=inflow_m3,
        storage_year_start_m3=200.0,
        reservoir_m_factor=reservoir_m_factor,
    )
    releases_high_storage_m3 = _run_release_scenario(
        current_demands_m3,
        long_term_demand_m3=80.0,
        inflow_m3=inflow_m3,
        storage_year_start_m3=800.0,
        reservoir_m_factor=reservoir_m_factor,
    )

    for releases_m3 in (
        releases_non_dominant_m3,
        releases_dominant_transition_m3,
        releases_equal_demands_m3,
        releases_low_storage_m3,
        releases_high_storage_m3,
    ):
        assert np.isfinite(releases_m3).all()
        assert (releases_m3 >= 0).all()

    assert np.all(np.diff(releases_non_dominant_m3) >= -1e-6)
    assert releases_dominant_transition_m3[-1] > releases_dominant_transition_m3[0]
    dominant_mask = current_demands_m3 > threshold_m3
    assert np.ptp(releases_equal_demands_m3[dominant_mask]) < 1e-5
    assert releases_high_storage_m3.mean() > releases_low_storage_m3.mean()

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    axis_top_left, axis_top_right = axes[0]
    axis_bottom_left, axis_bottom_right = axes[1]

    axis_top_left.plot(
        current_demands_m3,
        releases_non_dominant_m3,
        color="tab:blue",
        linewidth=2.2,
    )
    axis_top_left.axvline(threshold_m3, color="gray", linestyle="--", alpha=0.35)
    axis_top_left.set_title("Low long-term demand: non-dominant behaviour")
    axis_top_left.set_ylabel("Release per substep (m3)")
    axis_top_left.grid(True, alpha=0.3)

    axis_top_right.plot(
        current_demands_m3,
        releases_dominant_transition_m3,
        color="tab:green",
        linewidth=2.2,
    )
    axis_top_right.axvline(threshold_m3, color="gray", linestyle="--", alpha=0.35)
    axis_top_right.set_title("Fixed high long-term demand: branch transition visible")
    axis_top_right.grid(True, alpha=0.3)

    axis_bottom_left.plot(
        current_demands_m3,
        releases_equal_demands_m3,
        color="black",
        linewidth=2.2,
    )
    axis_bottom_left.axvline(threshold_m3, color="gray", linestyle="--", alpha=0.35)
    axis_bottom_left.set_title(
        "Current demand equals long-term demand: dominant branch flattens"
    )
    axis_bottom_left.set_xlabel("Current irrigation demand (m3)")
    axis_bottom_left.set_ylabel("Release per substep (m3)")
    axis_bottom_left.grid(True, alpha=0.3)

    axis_bottom_right.plot(
        current_demands_m3,
        releases_low_storage_m3,
        color="tab:red",
        linewidth=2.0,
        label="Low storage at year start",
    )
    axis_bottom_right.plot(
        current_demands_m3,
        releases_high_storage_m3,
        color="tab:purple",
        linewidth=2.0,
        label="High storage at year start",
    )
    axis_bottom_right.axvline(threshold_m3, color="gray", linestyle="--", alpha=0.35)
    axis_bottom_right.set_title("Storage sensitivity under dominant demand")
    axis_bottom_right.set_xlabel("Current irrigation demand (m3)")
    axis_bottom_right.grid(True, alpha=0.3)
    axis_bottom_right.legend(loc="upper left")

    figure.suptitle(
        "Irrigation reservoir release behaviour across scenarios", fontsize=14
    )
    figure.tight_layout()
    figure.savefig(output_folder / "test_reservoir_operators_sensitivity.png")
    plt.close(figure)
