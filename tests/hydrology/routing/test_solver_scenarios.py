"""Scenario-based tests and visual outputs for the 1D local inertial momentum solver.

Simulates dynamic 1D open channel hydraulics under diverse boundary and initial conditions,
producing visual profiles of river bed elevation, initial water levels, and transient water
surface evolution over time colored using viridis.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from geb.geb_types import ArrayFloat32
from geb.hydrology.routing.geometry import (
    GEOM_IN_BANKFULL_VOLUME,
    build_inertial_geometry,
)
from geb.hydrology.routing.inertial_kernel import _compute_reach_stage_from_vol
from geb.hydrology.routing.solver import solve_inertial_momentum
from tests.testconfig import output_folder

SCENARIOS_OUTPUT_DIR: Path = output_folder / "solver_scenarios"
SCENARIOS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ChannelSimulationResults(NamedTuple):
    """Container for reach routing simulation time series and spatial profiles."""

    time_s: list[float]
    reach_centers_km: np.ndarray
    bed_elevation_m: np.ndarray
    initial_stage_m: np.ndarray
    sampled_stages_m: list[np.ndarray]
    sampled_times_s: list[float]
    inflow_m3_s: list[float]
    outflow_m3_s: list[float]
    total_volume_m3: list[float]


def run_channel_routing_simulation(
    reach_length_m: float,
    bed_elevation_m: ArrayFloat32,
    initial_depth_m: float | ArrayFloat32,
    inflow_fn: Callable[[float], float],
    total_time_s: float,
    title: str,
    dt_s: float = 5.0,
    bankfull_width_m: float = 20.0,
    bankfull_depth_m: float = 3.0,
    floodplain_width_m: float = 40.0,
    manning_n: float = 0.035,
    downstream_boundary_type: str = "free",
    downstream_stage_fn: Callable[[float], float] | None = None,
    n_plot_snapshots: int = 12,
) -> ChannelSimulationResults:
    """Simulates 1D channel reach routing using the local inertial momentum solver.

    Args:
        reach_length_m: Length of each discrete reach interface (meters).
        bed_elevation_m: 1D array of reach bed elevations (meters).
        initial_depth_m: Initial water depth per reach, scalar or array (meters).
        inflow_fn: Callable returning upstream inflow discharge Q_in(t) given time (m³/s).
        total_time_s: Total simulation duration (seconds).
        title: Descriptive title for plot labeling and output filename.
        dt_s: Substep integration timestep duration (seconds).
        bankfull_width_m: Bankfull channel width (meters).
        bankfull_depth_m: Bankfull channel depth before floodplain spilling (meters).
        floodplain_width_m: Additional floodplain storage width (meters).
        manning_n: Manning roughness coefficient (s / m^(1/3)).
        downstream_boundary_type: 'free' for gravity drawdown, 'stage' for prescribed stage,
            or 'closed' / 'no_outflow' for a terminal wall with zero discharge.
        downstream_stage_fn: Optional callable Q_ds_stage(t) for prescribed downstream stage (meters).
        n_plot_snapshots: Number of temporal snapshots to display in the viridis profile plot.

    Returns:
        Aggregated time series and spatial profiles for the simulated channel scenario.
    """
    n_reaches: int = len(bed_elevation_m)
    lengths: ArrayFloat32 = np.full(n_reaches, reach_length_m, dtype=np.float32)
    widths: ArrayFloat32 = np.full(n_reaches, bankfull_width_m, dtype=np.float32)
    shapes: ArrayFloat32 = np.full(
        n_reaches, 0.5, dtype=np.float32
    )  # Parabolic channel
    bankfull_depths: ArrayFloat32 = np.full(
        n_reaches, bankfull_depth_m, dtype=np.float32
    )
    fp_widths: ArrayFloat32 = np.full(n_reaches, floodplain_width_m, dtype=np.float32)

    # Downstream bed drop for reach slope calculation
    ds_bed_elevation: ArrayFloat32 = np.empty(n_reaches, dtype=np.float32)
    ds_bed_elevation[:-1] = bed_elevation_m[1:]
    ds_bed_elevation[-1] = (
        bed_elevation_m[-1] - (bed_elevation_m[0] - bed_elevation_m[1])
        if n_reaches > 1
        else bed_elevation_m[-1] - np.float32(0.5)
    )
    interface_bed_elevation_max: ArrayFloat32 = np.maximum(
        bed_elevation_m, ds_bed_elevation
    )

    manning_n_sq: ArrayFloat32 = np.full(
        n_reaches, np.float32(manning_n**2), dtype=np.float32
    )
    inv_dx_interface: ArrayFloat32 = np.float32(1.0) / lengths
    raw_bed_drop: float = float(bed_elevation_m[0] - bed_elevation_m[-1])
    bed_slope: float = (
        raw_bed_drop / max(n_reaches * reach_length_m, 1.0)
        if raw_bed_drop > 0.0
        else 0.0001
    )
    pit_slope: ArrayFloat32 = np.full(
        n_reaches, np.float32(bed_slope), dtype=np.float32
    )

    geom = build_inertial_geometry(
        river_length=lengths,
        river_width=widths,
        shape_exponent=shapes,
        bankfull_depth=bankfull_depths,
        floodplain_width=fp_widths,
        bed_elevation=bed_elevation_m,
        ds_bed_elevation=ds_bed_elevation,
        interface_bed_elevation_max=interface_bed_elevation_max,
        manning_n_sq=manning_n_sq,
        inv_dx_interface=inv_dx_interface,
        pit_slope=pit_slope,
    )

    # Initialize reach storage volumes from initial depths
    depth_arr: np.ndarray = (
        np.full(n_reaches, initial_depth_m, dtype=np.float32)
        if np.isscalar(initial_depth_m)
        else np.asarray(initial_depth_m, dtype=np.float32)
    )
    bankfull_volumes: np.ndarray = geom.geom_inbank[:, GEOM_IN_BANKFULL_VOLUME]
    # In parabolic channel: V = V_bf * (h / h_bf)^(1.5)
    normalized_depth: np.ndarray = np.clip(depth_arr / bankfull_depth_m, 0.0, 1.0)
    initial_storage_m3: np.ndarray = bankfull_volumes * (normalized_depth**1.5)
    storage_m3: np.ndarray = initial_storage_m3.astype(np.float64)

    curr_discharge_m3_s: np.ndarray = np.zeros(n_reaches, dtype=np.float32)

    g_dt_substep: np.float32 = np.float32(9.80665 * dt_s)
    sqrt_gravity: np.float32 = np.float32(np.sqrt(9.80665))
    inv_dt_substep: np.float32 = np.float32(1.0 / dt_s)
    min_wet_depth_m: np.float32 = np.float32(0.005)

    n_steps: int = int(np.ceil(total_time_s / dt_s))
    snapshot_interval: int = max(n_steps // n_plot_snapshots, 1)

    time_series_s: list[float] = []
    inflow_series_m3_s: list[float] = []
    outflow_series_m3_s: list[float] = []
    total_volume_series_m3: list[float] = []

    sampled_stages: list[np.ndarray] = []
    sampled_times: list[float] = []

    # Record initial stage at t=0
    initial_stages = np.empty(n_reaches, dtype=np.float32)
    for k in range(n_reaches):
        initial_stages[k] = _compute_reach_stage_from_vol(
            k,
            np.float32(storage_m3[k]),
            geom.geom_inbank,
        )
    sampled_stages.append(initial_stages.copy())
    sampled_times.append(0.0)

    for step in range(n_steps):
        t_sec: float = step * dt_s
        q_in: float = float(inflow_fn(t_sec))

        # 1. Compute reach stages from current storage
        stage_buf = np.empty(n_reaches, dtype=np.float32)
        for k in range(n_reaches):
            stage_buf[k] = _compute_reach_stage_from_vol(
                k,
                np.float32(storage_m3[k]),
                geom.geom_inbank,
            )

        # 2. Solve interface discharges via solve_inertial_momentum
        new_discharge_m3_s = np.empty(n_reaches, dtype=np.float32)
        for k in range(n_reaches):
            if k < n_reaches - 1:
                z_int: np.float32 = max(bed_elevation_m[k], bed_elevation_m[k + 1])
                eff_depth: np.float32 = max(stage_buf[k], stage_buf[k + 1]) - z_int
                w_slope: np.float32 = (stage_buf[k + 1] - stage_buf[k]) / np.float32(
                    reach_length_m
                )
            else:
                if downstream_boundary_type in ("closed", "no_outflow"):
                    new_discharge_m3_s[k] = np.float32(0.0)
                    continue
                if (
                    downstream_boundary_type == "stage"
                    and downstream_stage_fn is not None
                ):
                    ds_stage: np.float32 = np.float32(downstream_stage_fn(t_sec))
                    z_int = bed_elevation_m[k]
                    eff_depth = max(stage_buf[k], ds_stage) - z_int
                    w_slope = (ds_stage - stage_buf[k]) / np.float32(reach_length_m)
                else:
                    eff_depth = max(stage_buf[k] - bed_elevation_m[k], np.float32(0.0))
                    drawdown_slope: np.float32 = eff_depth / np.float32(reach_length_m)
                    w_slope = -max(drawdown_slope, np.float32(bed_slope))

            new_discharge_m3_s[k] = solve_inertial_momentum(
                reach_idx=k,
                effective_depth=eff_depth,
                water_slope=w_slope,
                curr_discharge=curr_discharge_m3_s[k],
                min_wet_depth_m=min_wet_depth_m,
                geom_inbank=geom.geom_inbank,
                g_dt_substep=g_dt_substep,
                sqrt_gravity=sqrt_gravity,
                can_reverse=True,
                river_storage_m3_inertial=storage_m3,
                inv_dt_substep=inv_dt_substep,
            )

        # 3. Mass continuity update: dV/dt = Q_in - Q_out
        storage_m3[0] += dt_s * (q_in - new_discharge_m3_s[0])
        for k in range(1, n_reaches):
            storage_m3[k] += dt_s * (new_discharge_m3_s[k - 1] - new_discharge_m3_s[k])
        np.maximum(storage_m3, 0.0, out=storage_m3)
        curr_discharge_m3_s[:] = new_discharge_m3_s

        time_series_s.append(t_sec)
        inflow_series_m3_s.append(q_in)
        outflow_series_m3_s.append(float(new_discharge_m3_s[-1]))
        total_volume_series_m3.append(float(np.sum(storage_m3)))

        if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
            sampled_stages.append(stage_buf.copy())
            sampled_times.append(t_sec + dt_s)

    reach_centers_km: np.ndarray = (
        (np.arange(n_reaches) + 0.5) * reach_length_m / 1000.0
    )

    results = ChannelSimulationResults(
        time_s=time_series_s,
        reach_centers_km=reach_centers_km,
        bed_elevation_m=np.asarray(bed_elevation_m, dtype=float),
        initial_stage_m=initial_stages.astype(float),
        sampled_stages_m=sampled_stages,
        sampled_times_s=sampled_times,
        inflow_m3_s=inflow_series_m3_s,
        outflow_m3_s=outflow_series_m3_s,
        total_volume_m3=total_volume_series_m3,
    )

    # Visualization
    fig, (ax_profile, ax_hydro) = plt.subplots(
        2,
        1,
        figsize=(11, 8.5),
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )

    # Panel 1: Longitudinal channel water profile with viridis progression
    min_bed: float = float(np.min(bed_elevation_m))
    y_floor: float = min_bed - 0.8
    ax_profile.fill_between(
        reach_centers_km,
        y_floor,
        bed_elevation_m,
        color="#c2b280",
        alpha=0.45,
        label="River Bed",
    )
    ax_profile.plot(
        reach_centers_km,
        bed_elevation_m,
        color="#4a3c28",
        linewidth=2.2,
        zorder=3,
    )

    # Initial water level
    ax_profile.plot(
        reach_centers_km,
        results.initial_stage_m,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="Initial Water Level (t = 0)",
        zorder=4,
    )

    # Water levels evolving over time (viridis)
    norm = Normalize(vmin=0.0, vmax=total_time_s / 60.0)
    cmap = plt.cm.viridis
    for t_val_s, stage_profile in zip(
        results.sampled_times_s[1:], results.sampled_stages_m[1:]
    ):
        t_min: float = t_val_s / 60.0
        color = cmap(norm(t_min))
        ax_profile.plot(
            reach_centers_km,
            stage_profile,
            color=color,
            alpha=0.85,
            linewidth=1.6,
            zorder=2,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax_profile, pad=0.02)
    cbar.set_label("Simulation Time (minutes)", fontsize=10)

    ax_profile.set_xlabel("Longitudinal Distance (km)", fontsize=10)
    ax_profile.set_ylabel("Water Elevation (meters)", fontsize=10)
    ax_profile.set_title(
        f"{title}: River Bed and Transient Water Levels",
        fontsize=12,
        fontweight="bold",
    )
    ax_profile.grid(True, linestyle=":", alpha=0.6)
    ax_profile.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Hydrograph (Inflow vs Outflow)
    time_min: np.ndarray = np.array(results.time_s) / 60.0
    ax_hydro.plot(
        time_min,
        results.inflow_m3_s,
        color="#1f77b4",
        linewidth=1.8,
        label="Upstream Inflow (m³/s)",
    )
    ax_hydro.plot(
        time_min,
        results.outflow_m3_s,
        color="#2ca02c",
        linewidth=1.8,
        linestyle="-",
        label="Downstream Outflow (m³/s)",
    )
    ax_hydro.set_xlabel("Time (minutes)", fontsize=10)
    ax_hydro.set_ylabel("Discharge (m³/s)", fontsize=10)
    ax_hydro.set_title("Channel Hydrograph (Inflow and Outflow)", fontsize=11)
    ax_hydro.grid(True, linestyle=":", alpha=0.6)
    ax_hydro.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    slug: str = title.lower().replace(" ", "_").replace("-", "_")
    plot_path = SCENARIOS_OUTPUT_DIR / f"solver_scenario_{slug}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return results


def test_scenario_flood_pulse_propagation() -> None:
    """Tests a flood hydrograph wave propagating down a sloping river channel."""
    n_reaches = 12
    reach_length_m = 250.0  # 3 km total length
    bed_elevation: ArrayFloat32 = np.linspace(9.0, 3.0, n_reaches, dtype=np.float32)
    initial_depth: float = 1.0  # Steady initial baseflow depth

    # Inflow hydrograph: baseflow 8 m3/s + flood pulse peaking at 48 m3/s at t = 900s (15 min)
    def flood_pulse(t_s: float) -> float:
        base: float = 8.0
        peak: float = 40.0 * np.exp(-(((t_s - 900.0) / 250.0) ** 2))
        return base + peak

    results = run_channel_routing_simulation(
        reach_length_m=reach_length_m,
        bed_elevation_m=bed_elevation,
        initial_depth_m=initial_depth,
        inflow_fn=flood_pulse,
        total_time_s=3600.0,  # 1 hour
        title="Flood Pulse Propagation",
    )

    # Peak outflow must occur after peak inflow (wave celerity translation lag)
    peak_inflow_time_idx = int(np.argmax(results.inflow_m3_s))
    peak_outflow_time_idx = int(np.argmax(results.outflow_m3_s))
    assert peak_outflow_time_idx > peak_inflow_time_idx

    # Peak outflow should be attenuated due to reach storage diffusion
    assert max(results.outflow_m3_s) < max(results.inflow_m3_s)
    assert max(results.outflow_m3_s) > 15.0


def test_scenario_dam_break_flash_wave() -> None:
    """Tests a sudden release of water from an upstream reservoir into a shallow reach."""
    n_reaches = 24
    reach_length_m = 400.0
    bed_elevation: ArrayFloat32 = np.linspace(15.0, 8.0, n_reaches, dtype=np.float32)

    # Initial condition: first 6 reaches are a deep reservoir pool (3.0 m), downstream is shallow (0.2 m)
    initial_depths: ArrayFloat32 = np.full(n_reaches, 0.2, dtype=np.float32)
    initial_depths[:6] = 3.0

    def base_inflow(_t: float) -> float:
        return 2.0

    results = run_channel_routing_simulation(
        reach_length_m=reach_length_m,
        bed_elevation_m=bed_elevation,
        initial_depth_m=initial_depths,
        inflow_fn=base_inflow,
        total_time_s=2400.0,  # 40 min
        title="Dam Break Flash Wave",
    )

    # Downstream reaches must experience depth increase as surge arrives
    final_stage = results.sampled_stages_m[-1]
    initial_stage = results.initial_stage_m
    assert final_stage[-1] > initial_stage[-1]


def test_scenario_downstream_backwater_estuary() -> None:
    """Tests adverse slope deceleration and upstream backwater wave from a rising boundary."""
    n_reaches = 20
    reach_length_m = 500.0
    bed_elevation: ArrayFloat32 = np.linspace(8.0, 4.0, n_reaches, dtype=np.float32)
    initial_depth: float = 1.2

    def steady_inflow(_t: float) -> float:
        return 15.0

    # Downstream water level rises by 2.5 meters between t=600s and t=1800s
    def rising_downstream_stage(t_s: float) -> float:
        base_stage: float = float(bed_elevation[-1] + 1.2)
        if t_s < 600.0:
            return base_stage
        surge: float = 2.5 * min((t_s - 600.0) / 1200.0, 1.0)
        return base_stage + surge

    results = run_channel_routing_simulation(
        reach_length_m=reach_length_m,
        bed_elevation_m=bed_elevation,
        initial_depth_m=initial_depth,
        inflow_fn=steady_inflow,
        total_time_s=3600.0,
        downstream_boundary_type="stage",
        downstream_stage_fn=rising_downstream_stage,
        title="Downstream Backwater Intrusion",
    )

    # Water levels in the lower reaches must rise above initial stage
    final_stage = results.sampled_stages_m[-1]
    initial_stage = results.initial_stage_m
    assert final_stage[-2] > initial_stage[-2] + 1.0


def test_scenario_channel_drawdown() -> None:
    """Tests channel drainage when upstream inflow is abruptly shut off."""
    n_reaches = 20
    reach_length_m = 500.0
    bed_elevation: ArrayFloat32 = np.linspace(15.0, 9.0, n_reaches, dtype=np.float32)
    initial_depth: float = 2.5

    def zero_inflow(_t: float) -> float:
        return 0.0

    results = run_channel_routing_simulation(
        reach_length_m=reach_length_m,
        bed_elevation_m=bed_elevation,
        initial_depth_m=initial_depth,
        inflow_fn=zero_inflow,
        total_time_s=3000.0,
        title="Channel Drainage Drawdown",
    )

    # Total channel volume must decrease as water drains out
    v_start = results.total_volume_m3[0]
    v_end = results.total_volume_m3[-1]
    assert v_end < v_start * 0.7


def test_scenario_flow_over_bed_sill() -> None:
    """Tests pool formation and acceleration over an obstacle/bed sill."""
    n_reaches = 15
    reach_length_m = 200.0  # 3 km total length
    base_bed = np.linspace(10.0, 4.0, n_reaches, dtype=np.float32)
    sill_shape = np.zeros(n_reaches, dtype=np.float32)
    for i in range(n_reaches):
        dist = abs(i - 7)
        if dist <= 2:
            sill_shape[i] = np.float32(0.8 * (1.0 - (dist / 2.0) ** 2))
    bed_elevation: ArrayFloat32 = base_bed + sill_shape

    # Initial condition: pool upstream of sill, shallow downstream
    initial_depths: ArrayFloat32 = np.full(n_reaches, 0.4, dtype=np.float32)
    initial_depths[:7] = 1.2

    def pulsed_inflow(t_s: float) -> float:
        return 8.0 + 25.0 * np.exp(-(((t_s - 600.0) / 200.0) ** 2))

    results = run_channel_routing_simulation(
        reach_length_m=reach_length_m,
        bed_elevation_m=bed_elevation,
        initial_depth_m=initial_depths,
        inflow_fn=pulsed_inflow,
        total_time_s=2400.0,
        title="Flow over Bed Sill",
    )

    # Water must overflow the sill crest and register outflow at downstream end
    assert max(results.outflow_m3_s) > 10.0


def test_scenario_flat_bed_drainage() -> None:
    """Tests pure surface-slope driven flow in a completely horizontal (flat) river bed."""
    n_reaches = 16
    reach_length_m = 250.0  # 4 km total length
    # Flat horizontal bed elevation: 5.0 m everywhere (slope = 0.0)
    bed_elevation: ArrayFloat32 = np.full(n_reaches, 5.0, dtype=np.float32)
    initial_depth: float = 2.0  # Uniform initial depth

    # No upstream inflow: drainage is driven purely by water surface gradient towards the outfall
    def zero_inflow(_t: float) -> float:
        return 0.0

    results = run_channel_routing_simulation(
        reach_length_m=reach_length_m,
        bed_elevation_m=bed_elevation,
        initial_depth_m=initial_depth,
        inflow_fn=zero_inflow,
        total_time_s=3600.0,
        downstream_boundary_type="free",
        title="Flat Bed Drainage",
    )

    # In a flat channel with free outfall, water surface slopes downward towards the outlet
    final_stage = results.sampled_stages_m[-1]
    assert final_stage[0] > final_stage[-1]  # Upstream stage higher than outlet stage
    # Outflow occurs even with zero bed slope due to water surface slope dz/dx
    assert max(results.outflow_m3_s) > 5.0


def test_scenario_flat_bed_filling_no_outflow() -> None:
    """Tests water filling a horizontal basin with zero outflow (closed terminal wall)."""
    n_reaches = 15
    reach_length_m = 200.0  # 3 km total length
    # Completely flat bed: 4.0 m
    bed_elevation: ArrayFloat32 = np.full(n_reaches, 4.0, dtype=np.float32)
    initial_depth: float = 0.5

    # Inflow pulse: 20 m3/s for first 600s, then 0
    inflow_rate = 20.0
    pulse_duration_s = 600.0

    def pulse_inflow(t_s: float) -> float:
        return inflow_rate if t_s < pulse_duration_s else 0.0

    results = run_channel_routing_simulation(
        reach_length_m=reach_length_m,
        bed_elevation_m=bed_elevation,
        initial_depth_m=initial_depth,
        inflow_fn=pulse_inflow,
        total_time_s=2400.0,
        downstream_boundary_type="closed",
        title="Flat Bed Filling No Outflow",
    )

    # 1. Outflow must be strictly zero at all times
    assert all(q == 0.0 for q in results.outflow_m3_s)

    # 2. Perfect mass conservation: Final volume = Initial volume + integrated inflow
    dt_s = 5.0
    added_volume = sum(q * dt_s for q in results.inflow_m3_s)
    final_vol = results.total_volume_m3[-1]
    initial_vol = results.total_volume_m3[0] - results.inflow_m3_s[0] * dt_s
    assert abs(final_vol - (initial_vol + added_volume)) < 0.1

    # 3. Water surface should level out horizontally at the end across all reaches
    final_stage = results.sampled_stages_m[-1]
    stage_difference = abs(float(final_stage[0] - final_stage[-1]))
    assert (
        stage_difference < 0.30
    )  # Levels towards horizontal equilibrium across 3 km basin


def test_scenario_sloping_bed_reservoir_filling_no_outflow() -> None:
    """Tests reservoir pool impoundment behind a closed dam wall on a sloping bed."""
    n_reaches = 16
    reach_length_m = 250.0  # 4 km total length
    bed_elevation: ArrayFloat32 = np.linspace(12.0, 6.0, n_reaches, dtype=np.float32)
    initial_depth: float = 0.5

    # Steady continuous inflow filling the reservoir
    inflow_rate = 15.0

    def steady_inflow(_t: float) -> float:
        return inflow_rate

    total_time_s = 2400.0
    results = run_channel_routing_simulation(
        reach_length_m=reach_length_m,
        bed_elevation_m=bed_elevation,
        initial_depth_m=initial_depth,
        inflow_fn=steady_inflow,
        total_time_s=total_time_s,
        downstream_boundary_type="no_outflow",
        title="Sloping Bed Dam Impoundment No Outflow",
    )

    # Outflow must remain strictly zero
    assert all(q == 0.0 for q in results.outflow_m3_s)

    # Reservoir depth against the dam wall (reach -1) must increase substantially
    final_stage = results.sampled_stages_m[-1]
    initial_stage = results.initial_stage_m
    dam_depth_increase = float(final_stage[-1] - initial_stage[-1])
    assert dam_depth_increase > 0.8

    # Strict volume conservation
    dt_s = 5.0
    added_volume = sum(q * dt_s for q in results.inflow_m3_s)
    initial_vol = results.total_volume_m3[0] - results.inflow_m3_s[0] * dt_s
    final_vol = results.total_volume_m3[-1]
    assert abs(final_vol - (initial_vol + added_volume)) < 0.1
