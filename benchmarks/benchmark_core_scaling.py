"""Benchmark tool to evaluate the influence of thread/core counts on local inertial routing.

Measures multi-threaded scaling, parallel efficiency, and identifies potential
cache contention, memory bandwidth saturation, and cross-CCX penalties.
"""

import argparse
import time

import geopandas as gpd
import numba
import numpy as np
from affine import Affine
from numba import njit

from geb.hydrology.routing import create_river_network
from geb.hydrology.routing.inertial_kernel import (
    ArrayFloat32,
    ArrayFloat64,
    InertialGeometry,
    InertialTopology,
    RetentionCoupling,
    SubstepBuffers,
    WaterbodyCoupling,
    _apply_reverse_flow_limits,
    _run_inertial_substeps_parallel,
    _solve_momentum_loop,
    _update_continuity_and_cfl,
)
from geb.hydrology.routing.local_inertial import LocalInertial
from tests.hydrology.routing.test_local_inertial import _make_local_inertial


@njit(parallel=True, fastmath=True)
def _bench_momentum_parallel(
    n_inertial: int,
    substep_discharge_m3_s: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    geom: InertialGeometry,
    topo: InertialTopology,
    wb: WaterbodyCoupling,
    buffers: SubstepBuffers,
    g_dt_substep: np.float32,
    inv_dt_substep: np.float32,
    sqrt_gravity: np.float32,
    min_wet_depth_m: np.float32,
) -> None:
    _solve_momentum_loop(
        n_inertial,
        substep_discharge_m3_s,
        river_storage_m3_inertial,
        geom,
        topo,
        wb,
        buffers,
        g_dt_substep,
        inv_dt_substep,
        sqrt_gravity,
        min_wet_depth_m,
    )


@njit(parallel=True, fastmath=True)
def _bench_continuity_parallel(
    n_inertial: int,
    substep_discharge_m3_s: ArrayFloat32,
    kinematic_inflow_rate: ArrayFloat32,
    sideflow_m3_inertial: ArrayFloat32,
    evaporation_m3_inertial: ArrayFloat32,
    river_storage_m3_inertial: ArrayFloat64,
    over_abstraction_m3_inertial: ArrayFloat32,
    actual_evaporation_m3_inertial: ArrayFloat32,
    geom: InertialGeometry,
    topo: InertialTopology,
    buffers: SubstepBuffers,
    dt_substep: np.float32,
    dt_f32: np.float32,
    inv_dt_substep: np.float32,
    gravity_acceleration: np.float32,
    block_min_buf: ArrayFloat32,
) -> np.float32:
    return _update_continuity_and_cfl(
        n_inertial,
        substep_discharge_m3_s,
        kinematic_inflow_rate,
        sideflow_m3_inertial,
        evaporation_m3_inertial,
        river_storage_m3_inertial,
        over_abstraction_m3_inertial,
        actual_evaporation_m3_inertial,
        geom,
        topo,
        buffers,
        dt_substep,
        dt_f32,
        inv_dt_substep,
        gravity_acceleration,
        block_min_buf,
    )


def create_benchmark_router(n_reaches: int = 35_447) -> LocalInertial:
    """Creates a realistic tree-like river network matching Rhine basin dimensions.

    Args:
        n_reaches: Number of reaches in the synthetic river domain.

    Returns:
        Configured LocalInertial router instance.
    """
    np.random.seed(42)
    ldd = np.full((1, n_reaches), 6, dtype=np.uint8)
    ldd[0, -1] = 5  # terminal pit
    mask = np.ones((1, n_reaches), dtype=bool)

    network = create_river_network(ldd, mask, transform=Affine.identity())

    bed_elevation = np.linspace(500.0, 5.0, n_reaches, dtype=np.float32)
    river_length = np.random.uniform(500.0, 2000.0, n_reaches).astype(np.float32)
    river_width = np.linspace(5.0, 250.0, n_reaches, dtype=np.float32)
    river_ids = np.arange(n_reaches, dtype=np.int32)
    use_kinematic = np.zeros(n_reaches, dtype=bool)

    downstream_ids = np.append(np.arange(1, n_reaches, dtype=np.int32), -1)
    rivers_gdf = gpd.GeoDataFrame(
        {
            "downstream_ID": downstream_ids,
            "slope": np.full(n_reaches, 0.001, dtype=np.float32),
        },
        index=np.arange(n_reaches, dtype=np.int32),
    )

    router = _make_local_inertial(
        dt=3600,
        river_network=network,
        river_length=river_length,
        river_width=river_width,
        river_ids=river_ids,
        bankfull_river_elevation_m=bed_elevation,
        use_kinematic=use_kinematic,
        rivers_gdf=rivers_gdf,
    )
    return router


def run_scaling_benchmark(
    thread_counts: list[int],
    n_reaches: int = 35_447,
    n_substeps: int = 20,
) -> list[dict[str, float]]:
    """Runs scaling benchmark across specified thread counts.

    Args:
        thread_counts: Sequence of thread counts to benchmark.
        n_reaches: Number of river reaches in the benchmark domain.
        n_substeps: Number of local inertial substeps executed per run.

    Returns:
        List of timing and efficiency records per thread count.
    """
    print("=" * 80)
    print(
        f"ROUTING MULTI-THREAD SCALING BENCHMARK (N={n_reaches:,} reaches, {n_substeps} substeps)"
    )
    print("=" * 80)

    print("Initializing benchmark router...")
    t0 = time.time()
    router = create_benchmark_router(n_reaches)
    print(f"Router created in {time.time() - t0:.2f}s")

    storage = np.random.uniform(5000.0, 50000.0, n_reaches).astype(np.float64)
    discharge = np.random.uniform(10.0, 100.0, n_reaches).astype(np.float32)
    kinematic_inflow = np.random.uniform(5.0, 20.0, n_reaches).astype(np.float32)
    sideflow = np.zeros(n_reaches, dtype=np.float32)
    evap = np.zeros(n_reaches, dtype=np.float32)
    over_abs = np.zeros(n_reaches, dtype=np.float32)
    actual_evap = np.zeros(n_reaches, dtype=np.float32)
    updated_q = np.empty(n_reaches, dtype=np.float32)

    topo = router._topology
    geom = router._geometry
    buffers = router._substep_buffers

    wb_storage = np.zeros(0, dtype=np.float64)
    wb_outflow = np.zeros(0, dtype=np.float32)
    ret_storage = np.zeros(0, dtype=np.float32)

    wb = WaterbodyCoupling(
        storage_m3=wb_storage,
        prescribed_outflow_m3=wb_outflow,
        inflow_m3=router._waterbody_inflow_perm,
        lake_area=router._wb_lake_area,
        lake_factor=router._wb_lake_factor,
        outflow_height=router._wb_outflow_height,
        outflow_bed_elev=router._wb_outflow_bed_elev,
        inertial_wb_ds_k=router._wb_links.inertial_wb_ds_k,
        inertial_wb_ds_id=router._wb_links.inertial_wb_ds_id,
        lake_outflow_target_k=router._wb_links.lake_outflow_target_k,
        lake_outflow_wb_id=router._wb_links.lake_outflow_wb_id,
        wb_to_wb_src_wb=router._wb_links.wb_to_wb_src_wb,
        wb_to_wb_tgt_wb=router._wb_links.wb_to_wb_tgt_wb,
        wb_terminal_wb_ids=router._wb_links.wb_terminal_wb_ids,
    )

    ret = RetentionCoupling(
        storage_m3=ret_storage,
        inflow_m3=router._retention_inflow_perm,
        outflow_m3=router._retention_outflow_perm,
        max_storage_m3=router.retention_max_storage_m3,
        inertial_retention_k=router._ret_links.inertial_retention_k,
        inertial_retention_id=router._ret_links.inertial_retention_id,
        controlled_retention=router.controlled_retention,
        activation_threshold_m3_s=np.zeros(0, dtype=np.float32),
        release_threshold_factor=router.retention_basin_release_threshold_factor,
    )

    # Warmup Numba JIT at 1 thread
    numba.set_num_threads(1)
    print("Warming up JIT compiler...")
    _ = _run_inertial_substeps_parallel(
        num_inertial_substeps=2,
        dt_f32=np.float32(3600.0),
        n_inertial=n_reaches,
        substep_discharge_m3_s=discharge.copy(),
        river_storage_m3_inertial=storage.copy(),
        sideflow_m3_inertial=sideflow,
        evaporation_m3_inertial=evap,
        over_abstraction_m3_inertial=over_abs,
        actual_evaporation_m3_inertial=actual_evap,
        updated_discharge_m3_s_inertial=updated_q,
        kinematic_inflow_rate=kinematic_inflow,
        geom=geom,
        topo=topo,
        wb=wb,
        ret=ret,
        buffers=buffers,
        initialize_stages=True,
    )
    print("JIT compilation complete.\n")

    results: list[dict[str, float]] = []
    baseline_time: float | None = None

    print(
        f"{'Threads':>7} | {'Total Time (ms)':>15} | {'Per Substep (us)':>17} | {'Speedup':>9} | {'Efficiency':>10}"
    )
    print("-" * 70)

    for n_threads in thread_counts:
        numba.set_num_threads(n_threads)

        repeats = 5
        timings: list[float] = []

        for _ in range(repeats):
            st_copy = storage.copy()
            q_copy = discharge.copy()
            over_abs_copy = np.zeros(n_reaches, dtype=np.float32)
            actual_evap_copy = np.zeros(n_reaches, dtype=np.float32)
            updated_q_copy = np.empty(n_reaches, dtype=np.float32)

            t_start = time.perf_counter()
            _run_inertial_substeps_parallel(
                num_inertial_substeps=n_substeps,
                dt_f32=np.float32(3600.0),
                n_inertial=n_reaches,
                substep_discharge_m3_s=q_copy,
                river_storage_m3_inertial=st_copy,
                sideflow_m3_inertial=sideflow,
                evaporation_m3_inertial=evap,
                over_abstraction_m3_inertial=over_abs_copy,
                actual_evaporation_m3_inertial=actual_evap_copy,
                updated_discharge_m3_s_inertial=updated_q_copy,
                kinematic_inflow_rate=kinematic_inflow,
                geom=geom,
                topo=topo,
                wb=wb,
                ret=ret,
                buffers=buffers,
                initialize_stages=False,
            )
            t_elapsed = (time.perf_counter() - t_start) * 1000.0
            timings.append(t_elapsed)

        best_time = min(timings)
        time_per_substep_us = (best_time / n_substeps) * 1000.0

        if baseline_time is None:
            baseline_time = best_time

        speedup = baseline_time / best_time
        efficiency = (speedup / n_threads) * 100.0

        results.append(
            {
                "threads": float(n_threads),
                "total_ms": best_time,
                "substep_us": time_per_substep_us,
                "speedup": speedup,
                "efficiency": efficiency,
            }
        )

        print(
            f"{n_threads:>7} | {best_time:>15.3f} | {time_per_substep_us:>17.1f} | {speedup:>8.2f}x | {efficiency:>9.1f}%"
        )

    print("\n" + "=" * 80)
    print("DETAILED KERNEL SCALING BREAKDOWN")
    print("=" * 80)
    print(
        f"{'Threads':>7} | {'Momentum (us)':>15} | {'Continuity (us)':>16} | {'ReverseLim (us)':>16}"
    )
    print("-" * 65)

    g_dt = np.float32(9.80665 * 30.0)
    inv_dt = np.float32(1.0 / 30.0)
    sqrt_g = np.float32(np.sqrt(9.80665))
    min_wet = np.float32(1e-4)

    for n_threads in thread_counts:
        numba.set_num_threads(n_threads)
        runs = 25

        # 1. Momentum loop
        t_mom_list: list[float] = []
        for _ in range(runs):
            t_s = time.perf_counter()
            _bench_momentum_parallel(
                n_reaches,
                discharge,
                storage,
                geom,
                topo,
                wb,
                buffers,
                g_dt,
                inv_dt,
                sqrt_g,
                min_wet,
            )
            t_mom_list.append((time.perf_counter() - t_s) * 1e6)
        t_mom = min(t_mom_list)

        # 2. Continuity + CFL loop
        t_cont_list: list[float] = []
        for _ in range(runs):
            t_s = time.perf_counter()
            _bench_continuity_parallel(
                n_reaches,
                discharge,
                kinematic_inflow,
                sideflow,
                evap,
                storage,
                over_abs,
                actual_evap,
                geom,
                topo,
                buffers,
                np.float32(30.0),
                np.float32(3600.0),
                inv_dt,
                np.float32(9.80665),
                buffers.block_min_buf,
            )
            t_cont_list.append((time.perf_counter() - t_s) * 1e6)
        t_cont = min(t_cont_list)

        # 3. Reverse limit loop
        t_rev_list: list[float] = []
        for _ in range(runs):
            t_s = time.perf_counter()
            _apply_reverse_flow_limits(
                n_inertial=n_reaches,
                substep_discharge_m3_s=discharge,
                river_storage_m3_inertial=storage,
                topo=topo,
                buffers=buffers,
                dt_substep_f64=30.0,
                inv_dt_substep=inv_dt,
            )
            t_rev_list.append((time.perf_counter() - t_s) * 1e6)
        t_rev = min(t_rev_list)

        print(f"{n_threads:>7} | {t_mom:>15.1f} | {t_cont:>16.1f} | {t_rev:>16.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate multi-threaded core scaling and cache contention in local inertial routing."
    )
    parser.add_argument(
        "--reaches",
        type=int,
        default=35_447,
        help="Number of river reaches (default: 35447)",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=20,
        help="Number of substeps to benchmark (default: 20)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32, 64],
        help="Thread counts to evaluate",
    )
    args = parser.parse_args()

    run_scaling_benchmark(
        thread_counts=args.threads,
        n_reaches=args.reaches,
        n_substeps=args.substeps,
    )
