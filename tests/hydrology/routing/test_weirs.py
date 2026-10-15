"""Rectangular weir ratings and conservative local inertial routing tests."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyflwdir
import pytest
from affine import Affine

from geb.hydrology.routing import create_river_network
from geb.hydrology.routing.local_inertial import LocalInertial
from geb.hydrology.routing.solver import rectangular_weir_discharge
from tests.hydrology.routing.test_local_inertial import _make_local_inertial
from tests.testconfig import output_folder


@pytest.mark.parametrize("head_m", [0.01, 0.5, 1.0, 2.0])
def test_rectangular_free_overflow(head_m: float) -> None:
    """Match Table 2 for a rectangular crest with free downstream overflow.

    Args:
        head_m: Upstream water depth above the crest (meters).
    """
    expected: float = 2 / 3 * 0.577 * np.sqrt(2 * 9.80665) * 10 * head_m**1.5
    assert rectangular_weir_discharge(
        np.float32(5 + head_m), np.float32(4), np.float32(5), np.float32(10)
    ) == pytest.approx(expected, rel=5e-5)


@pytest.mark.parametrize("tailwater_m", [0.0, 0.67, 0.8, 0.95, 1.0])
def test_submerged_and_reverse_overflow(tailwater_m: float) -> None:
    """Drown the crest smoothly and reverse flow when water levels reverse.

    Args:
        tailwater_m: Downstream depth above the crest (meters).
    """
    free: float = 2 / 3 * 0.577 * np.sqrt(2 * 9.80665) * 10
    reduction: float = 1 - (max(tailwater_m - 0.67, 0) / 0.33) ** 3
    args: tuple[np.float32, ...] = tuple(map(np.float32, (1, tailwater_m, 0, 10)))
    actual: np.float32 = rectangular_weir_discharge(*args)
    assert actual == pytest.approx(free * reduction, abs=1e-5)
    assert rectangular_weir_discharge(args[1], args[0], args[2], args[3]) == -actual


@pytest.mark.parametrize("stages", [(0.0, -1.0), (-1.0, -2.0), (1.0, 1.0)])
def test_no_flow_below_crest_or_equal_levels(stages: tuple[float, float]) -> None:
    """Keep a dry crest closed and give no exchange at equal water levels.

    Args:
        stages: Upstream and downstream elevations relative to the crest (meters).
    """
    assert (
        rectangular_weir_discharge(
            np.float32(stages[0]), np.float32(stages[1]), np.float32(0), np.float32(10)
        )
        == 0
    )


def make_weir_channel(enabled: bool, dt_s: int = 60) -> LocalInertial:
    """Create a 12-reach test channel with one half-bankfull-depth weir.

    Args:
        enabled: Whether the weir controls discharge.
        dt_s: Macro routing timestep (seconds).

    Returns:
        Router with 200 m reaches, 10 m width and 2 m bankfull depth.
    """
    count: int = 12
    ldd: np.ndarray = np.full((count, 1), 2, dtype=np.uint8)
    ldd[-1, 0] = 5
    network: pyflwdir.FlwdirRaster = create_river_network(
        ldd, np.ones_like(ldd, dtype=bool), transform=Affine.identity()
    )
    fractions: np.ndarray = np.zeros(count, dtype=np.float32)
    fractions[6] = 0.5
    return _make_local_inertial(
        dt=dt_s,
        river_network=network,
        river_length=np.full(count, 200, dtype=np.float32),
        river_width=np.full(count, 10, dtype=np.float32),
        bankfull_depth_m=2.0,
        bankfull_river_elevation_m=np.linspace(1.1, 0, count, dtype=np.float32),
        weir_bankfull_depth_fraction=fractions,
        weirs_enabled=enabled,
    )


def run_channel(
    router: LocalInertial,
    duration_s: int = 21600,
    inflow_m3_s: float = 1.0,
    initial_depth_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route constant inflow and verify conservation at every macro timestep.

    Args:
        router: Configured test channel.
        duration_s: Simulation length (seconds).
        inflow_m3_s: Input at the head of the channel (m³/s).
        initial_depth_m: Initial depth everywhere in the channel (meters).

    Returns:
        Final water elevations (meters), stored volumes (m³), and outlet
        discharge series (m³/s).
    """
    count: int = router.river_length.size
    bankfull_volume_m3: float = 200 * 10 * 2 / 1.5
    initial_volume_m3: float = (
        bankfull_volume_m3 * (initial_depth_m / 2) ** 1.5
        if initial_depth_m <= 2
        else bankfull_volume_m3 + 200 * 2 * (initial_depth_m - 2) ** 2
    )
    storage: np.ndarray = np.full(count, initial_volume_m3, dtype=np.float64)
    discharge: np.ndarray = np.zeros(count, dtype=np.float32)
    sideflow: np.ndarray = np.zeros(count, dtype=np.float32)
    sideflow[0] = inflow_m3_s * router.dt
    router.initialize_stage(
        water_stage_m=router.bed_elevation + np.float32(initial_depth_m)
    )
    hydrograph: list[float] = []
    empty: np.ndarray = np.zeros(0, dtype=np.float32)
    step: int
    for step in range(int(duration_s / router.dt)):
        before: float = storage.sum()
        result: tuple = router.step(
            discharge,
            storage,
            sideflow,
            np.zeros(count, dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            empty,
            empty,
            empty,
        )
        discharge, storage = result[:2]
        assert storage.min() >= -1e-7
        assert storage.sum() + result[6] == pytest.approx(
            before + sideflow.sum(), abs=0.02
        )
        hydrograph.append(float(result[6] / router.dt))
    return router.get_water_stage(), storage, np.asarray(hydrograph)


def test_weir_backwater_and_plot() -> None:
    """Demonstrate upstream backwater, downstream release and mass conservation."""
    with_weir: LocalInertial = make_weir_channel(True)
    without_weir: LocalInertial = make_weir_channel(False)
    stage_weir, _, flow_weir = run_channel(with_weir)
    stage_control, _, flow_control = run_channel(without_weir)
    assert stage_weir[6] > stage_control[6] + 0.3
    assert stage_weir[6] > with_weir.bed_elevation[6] + 1
    assert np.mean(flow_weir[-30:]) == pytest.approx(1, abs=0.03)
    folder: Path = output_folder / "solver_scenarios"
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    distance: np.ndarray = np.arange(12) * 200
    axes[0].plot(
        distance, with_weir.bed_elevation, color="saddlebrown", label="River bed"
    )
    edges: np.ndarray = np.arange(13) * 200 - 100
    axes[0].stairs(
        stage_control, edges, baseline=None, color="tab:blue", label="Without weir"
    )
    axes[0].stairs(
        stage_weir, edges, baseline=None, color="tab:orange", label="With weir"
    )
    axes[0].vlines(
        1300,
        with_weir.bed_elevation[6],
        with_weir.bed_elevation[6] + 1,
        color="black",
        linewidth=3,
        label="Weir crest",
    )
    axes[0].set(
        xlabel="Distance downstream (m)",
        ylabel="Elevation (m)",
        title="Rectangular weir: same inflow, same channel",
    )
    axes[1].plot(
        np.arange(flow_weir.size) / 60, flow_weir, color="tab:orange", label="With weir"
    )
    axes[1].plot(
        np.arange(flow_control.size) / 60,
        flow_control,
        color="tab:blue",
        label="Without weir",
    )
    axes[1].set(xlabel="Time (hours)", ylabel="Outlet discharge (m³/s)")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(folder / "solver_scenario_rectangular_weir.png", dpi=150)
    plt.close(figure)


def test_pool_stays_below_crest() -> None:
    """Do not drain the upstream pool through a dry crest."""
    router: LocalInertial = make_weir_channel(True)
    _, storage, _ = run_channel(router, duration_s=1800, inflow_m3_s=0)
    crest_storage: float = 200 * 10 * 2 / 1.5 * (1 / 2) ** 1.5
    assert storage[6] <= crest_storage
    assert storage[:7].sum() > 7 * (200 * 10 * 2 / 1.5 * (0.5 / 2) ** 1.5) - 0.02


def test_weir_timestep_convergence() -> None:
    """Keep water levels consistent when the outer routing timestep changes."""
    fine, _, _ = run_channel(make_weir_channel(True, 30), duration_s=3600)
    coarse, _, _ = run_channel(make_weir_channel(True, 300), duration_s=3600)
    np.testing.assert_allclose(coarse, fine, atol=0.03)


@pytest.mark.parametrize("enabled", [True, False])
def test_real_model_profile_exports(tmp_path: Path, enabled: bool) -> None:
    """Export elevations and profiles for the same cells in both comparison runs.

    Args:
        tmp_path: Temporary output directory.
        enabled: Whether weir flow is enabled.
    """
    from datetime import date

    import pandas as pd

    from geb.hydrology.routing.plots import plot_weir_water_levels

    router: LocalInertial = make_weir_channel(enabled)
    run_channel(router, duration_s=300)
    paths: tuple[Path, Path] | None = plot_weir_water_levels(
        router, np.arange(12, dtype=np.int32), tmp_path, date(2000, 1, 1)
    )
    assert paths is not None
    assert all(path.is_file() and path.stat().st_size > 100 for path in paths)
    table: pd.DataFrame = pd.read_csv(paths[1])
    assert table.grid_cell.tolist() == [6]
    assert table.weir_height_m.tolist() == [1.0]
    assert table.weirs_enabled.tolist() == [enabled]
    assert table.upstream_head_m.iloc[0] == pytest.approx(
        max(table.upstream_water_level_m.iloc[0] - table.crest_elevation_m.iloc[0], 0),
        abs=5e-7,
    )
    assert table.water_level_drop_m.iloc[0] == pytest.approx(
        table.upstream_water_level_m.iloc[0] - table.downstream_water_level_m.iloc[0],
        abs=5e-7,
    )
    assert table.crest_elevation_m.iloc[0] == pytest.approx(router.bed_elevation[6] + 1)
    profiles: pd.DataFrame = pd.read_csv(
        tmp_path / "water_levels_2000-01-01_profiles.csv"
    )
    assert profiles.grid_cell.tolist() == [4, 5, 6, 7, 8]
    np.testing.assert_allclose(
        profiles.water_depth_m,
        profiles.water_level_m - profiles.bed_elevation_m,
        atol=5e-7,
    )
    assert profiles.focal_grid_cell.eq(6).all()
    assert profiles.water_depth_m.ge(0).all()
    router.weir_bankfull_depth_fraction[:] = 0
    assert (
        plot_weir_water_levels(
            router, np.arange(12, dtype=np.int32), tmp_path, date(2000, 1, 2)
        )
        is None
    )


def test_disabled_weirs_match_unmodified_channel() -> None:
    """Disabling weirs gives identical flow to a channel without weir inputs."""
    disabled: LocalInertial = make_weir_channel(False)
    no_weirs: LocalInertial = make_weir_channel(True)
    no_weirs.weir_bankfull_depth_fraction[:] = 0
    no_weirs.update_channel_geometry(no_weirs.river_width, no_weirs.bankfull_depth)
    first: tuple = run_channel(disabled, duration_s=600)
    second: tuple = run_channel(no_weirs, duration_s=600)
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)


def test_serial_parallel_weirs_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the same conservative weir flux in both kernel execution paths.

    Args:
        monkeypatch: Fixture forcing serial and parallel kernel dispatch.
    """
    import geb.hydrology.routing.local_inertial as local_inertial

    monkeypatch.setattr(local_inertial, "INERTIAL_PARALLEL_THRESHOLD", 1000)
    serial: tuple = run_channel(make_weir_channel(True), duration_s=600)
    monkeypatch.setattr(local_inertial, "INERTIAL_PARALLEL_THRESHOLD", 0)
    parallel: tuple = run_channel(make_weir_channel(True), duration_s=600)
    for actual, expected in zip(parallel, serial):
        np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("can_reverse", [True, False])
def test_momentum_weir_reverse_policy(can_reverse: bool) -> None:
    """Respect routing's reversal policy at submerged weir interfaces.

    Args:
        can_reverse: Whether upstream directed flow is permitted.
    """
    from geb.hydrology.routing.geometry import GEOM_IN_WEIR_HEIGHT
    from geb.hydrology.routing.solver import solve_inertial_momentum
    from tests.hydrology.routing.test_solver import _create_single_reach_geometry

    inbank, overbank = _create_single_reach_geometry()
    inbank[0, GEOM_IN_WEIR_HEIGHT] = 1
    flow: np.float32 = solve_inertial_momentum(
        reach_idx=0,
        effective_depth=np.float32(1),
        water_slope=np.float32(0.001),
        curr_discharge=np.float32(0),
        min_wet_depth_m=np.float32(1e-4),
        geom_inbank=inbank,
        geom_overbank=overbank,
        g_dt_substep=np.float32(9.80665),
        sqrt_gravity=np.float32(np.sqrt(9.80665)),
        can_reverse=can_reverse,
        river_storage_m3_inertial=np.asarray([10000.0]),
        inv_dt_substep=np.float32(1),
        upstream_stage_m=np.float32(1.5),
        downstream_stage_m=np.float32(2),
    )
    assert (flow < 0) if can_reverse else flow == 0


@pytest.mark.parametrize(
    "initial_depth_m,inflow_m3_s", [(0.0, 5.0), (1.9, 10.0), (2.05, 10.0)]
)
def test_dry_and_drowned_weir_routing(
    initial_depth_m: float, inflow_m3_s: float
) -> None:
    """Conserve water through dry-start filling and submerged high flow.

    Args:
        initial_depth_m: Initial channel water depth (meters).
        inflow_m3_s: Constant upstream inflow (m³/s).
    """
    router: LocalInertial = make_weir_channel(True, 300)
    stages, storage, _ = run_channel(
        router,
        duration_s=3600,
        inflow_m3_s=inflow_m3_s,
        initial_depth_m=initial_depth_m,
    )
    assert np.isfinite(stages).all()
    assert np.isfinite(storage).all()
    assert stages[6] > router.bed_elevation[6] + 1


def test_unsupported_outlet_weir_is_ignored() -> None:
    """Do not apply an internal exchange law at an open domain boundary."""
    router: LocalInertial = make_weir_channel(True)
    fractions: np.ndarray = router.weir_bankfull_depth_fraction.copy()
    fractions[-1] = 0.5
    drainage: np.ndarray = np.full((12, 1), 2, dtype=np.uint8)
    drainage[-1, 0] = 5
    network: pyflwdir.FlwdirRaster = create_river_network(
        drainage, np.ones_like(drainage, dtype=bool), transform=Affine.identity()
    )
    rebuilt: LocalInertial = _make_local_inertial(
        dt=60,
        river_network=network,
        river_length=router.river_length,
        river_width=router.river_width,
        bankfull_depth_m=router.bankfull_depth,
        weir_bankfull_depth_fraction=fractions,
    )
    assert rebuilt.weir_height_m[-1] == 0
    assert rebuilt._geometry.weir_indices.size == 1


@pytest.mark.parametrize("head_m", [1.0, 100.0, 1000.0])
def test_deeply_submerged_rating_keeps_precision(head_m: float) -> None:
    """Retain a small level drop when both sides are far above the crest.

    Args:
        head_m: Higher water elevation above the crest (meters).
    """
    upstream: np.float32 = np.float32(head_m)
    downstream: np.float32 = upstream - np.float32(0.001)
    # Evaluate the original submergence formula in double precision, using
    # the represented inputs: float32 already rounds the 1 mm drop at high heads.
    submergence: float = float(downstream) / float(upstream)
    free_flow: float = 2 / 3 * 0.577 * np.sqrt(2 * 9.80665) * 10 * head_m**1.5
    expected: float = free_flow * (1 - ((submergence - 0.67) / 0.33) ** 3)
    actual: np.float32 = rectangular_weir_discharge(
        upstream, downstream, np.float32(0), np.float32(10)
    )
    assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("depth_m", [1.5, 2.05])
def test_weir_timestep_matches_storage_response(depth_m: float) -> None:
    """Check exchange stability against the actual stage-volume derivative.

    Args:
        depth_m: Initial water depth, inside or above the river banks (meters).
    """
    from geb.hydrology.routing.geometry import GEOM_IN_BANKFULL_VOLUME
    from geb.hydrology.routing.inertial_kernel import (
        _compute_reach_stage_from_vol,
        _limit_weir_timestep,
    )

    router: LocalInertial = make_weir_channel(True)
    router.initialize_stage(water_stage_m=router.bed_elevation + np.float32(depth_m))
    upstream: int = int(router._geometry.weir_indices[0])
    downstream: int = int(router._topology.ds_inertial_k[upstream])
    stages: np.ndarray = router._substep_buffers.stage_buf
    inbank: np.ndarray = router._geometry.geom_inbank
    overbank: np.ndarray = router._geometry.geom_overbank
    inverse_area: float = 0
    cell: int
    for cell in (upstream, downstream):
        bankfull_volume: float = float(inbank[cell, GEOM_IN_BANKFULL_VOLUME])
        volume: float = (
            bankfull_volume * (depth_m / 2) ** 1.5
            if depth_m <= 2
            else bankfull_volume + 200 * 2 * (depth_m - 2) ** 2
        )
        perturbation_m3: float = 0.02
        stage_plus: float = float(
            _compute_reach_stage_from_vol(
                cell, np.float32(volume + perturbation_m3), inbank, overbank
            )
        )
        stage_minus: float = float(
            _compute_reach_stage_from_vol(
                cell, np.float32(volume - perturbation_m3), inbank, overbank
            )
        )
        inverse_area += (stage_plus - stage_minus) / (2 * perturbation_m3)
    flow: float = float(
        rectangular_weir_discharge(
            stages[upstream], stages[downstream], np.float32(1.5), np.float32(10)
        )
    )
    expected_dt: float = min(
        3600, 0.2 * float(stages[upstream] - stages[downstream]) / (flow * inverse_area)
    )
    actual_dt: np.float32 = _limit_weir_timestep(
        np.float32(3600), router._geometry, router._topology, router._substep_buffers
    )
    assert actual_dt == pytest.approx(expected_dt, rel=0.02)


@pytest.mark.parametrize("height_m", [-0.1, np.nan, np.inf, 2.1])
def test_invalid_crest_geometry(height_m: float) -> None:
    """Reject heights for which the in-bank crest-storage relation is invalid.

    Args:
        height_m: Invalid crest height above a bed with 2 m bankfull depth (meters).
    """
    from geb.hydrology.routing.geometry import build_inertial_geometry

    with pytest.raises(ValueError, match="Weir heights"):
        build_inertial_geometry(
            river_length=np.array([100], dtype=np.float32),
            river_width=np.array([10], dtype=np.float32),
            shape_exponent=np.array([0.5], dtype=np.float32),
            bankfull_depth=np.array([2], dtype=np.float32),
            floodplain_width=np.array([0], dtype=np.float32),
            bed_elevation=np.array([0], dtype=np.float32),
            ds_bed_elevation=np.array([0], dtype=np.float32),
            interface_bed_elevation_max=np.array([0], dtype=np.float32),
            manning_n_sq=np.array([0.03**2], dtype=np.float32),
            inv_dx_interface=np.array([0.01], dtype=np.float32),
            up_count=np.array([0], dtype=np.int32),
            pit_slope=np.array([0.001], dtype=np.float32),
            weir_height_m=np.array([height_m], dtype=np.float32),
        )
