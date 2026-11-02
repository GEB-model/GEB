"""Benchmark script for GEB Land Surface Model (LSM).

Measures runtime and memory usage of `land_surface_model` and verifies correctness
against pre-recorded baseline outputs from Rhine simulation.
"""

import gc
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from geb.hydrology.landsurface.constants import (
    N_STATIC_SOIL_PROPS,
    STATIC_SOIL_PROPERTY_NAMES,
)
from geb.hydrology.landsurface.landsurface_model import land_surface_model

DEFAULT_DATA_DIR = Path("/scistor/ivm/jbn271/GEB/models/rhine/benchmark_data")
OUTPUT_NAMES = (
    "rain_m",
    "snow_m",
    "topwater_m",
    "reference_evapotranspiration_grass_m",
    "reference_evapotranspiration_water_m",
    "snow_water_equivalent_m",
    "liquid_water_in_snow_m",
    "sublimation_or_deposition_m",
    "snow_enthalpy_J_per_m2",
    "snow_density_kg_per_m3",
    "interception_storage_m",
    "interception_evaporation_m",
    "open_water_evaporation_m",
    "runoff_m",
    "groundwater_recharge_m",
    "capillar_rise_m",
    "interflow_m",
    "bare_soil_evaporation_m",
    "transpiration_m",
    "potential_transpiration_m",
    "potential_evapotranspiration_m",
    "soil_boundary_enthalpy_flux_J_per_m2",
    "rain_advection_enthalpy_flux_J_per_m2",
    "evaporative_cooling_enthalpy_loss_J_per_m2",
    "interflow_enthalpy_loss_J_per_m2",
    "groundwater_recharge_enthalpy_loss_J_per_m2",
    "transpiration_enthalpy_loss_J_per_m2",
    "top_soil_infiltration_m",
    "top_soil_rise_from_layer_2_m",
    "top_soil_percolation_to_layer_2_m",
    "top_soil_transpiration_m",
    "evapotranspiration_m",
)


def get_peak_rss_mb() -> float:
    """Return max RSS in megabytes from getrusage."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in kilobytes
    return rusage.ru_maxrss / 1024.0


def load_benchmark_data(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load inputs and expected outputs from npz archives.

    Args:
        data_dir: Directory containing rhine_lsm_input.npz and rhine_lsm_output.npz.

    Returns:
        Tuple of (inputs_dict, expected_outputs_dict).

    Raises:
        FileNotFoundError: If input or output benchmark files are missing.
    """
    input_path = data_dir / "rhine_lsm_input.npz"
    output_path = data_dir / "rhine_lsm_output.npz"

    if not input_path.exists() or not output_path.exists():
        raise FileNotFoundError(f"Benchmark data files not found in {data_dir}")

    print(f"Loading input data from {input_path}...")
    with np.load(input_path) as npz:
        inputs = {k: npz[k] for k in npz.files}

    print(f"Loading expected outputs from {output_path}...")
    with np.load(output_path) as npz:
        expected = {k: npz[k] for k in npz.files}

    if "packed_soil_properties" not in inputs:
        num_cells = inputs["slope_m_per_m"].shape[0]
        n_layers = inputs["water_content_residual_m"].shape[1]
        packed_soil = np.empty(
            (num_cells, N_STATIC_SOIL_PROPS, n_layers), dtype=np.float32
        )
        for idx, prop in enumerate(STATIC_SOIL_PROPERTY_NAMES):
            packed_soil[:, idx, :] = inputs[prop]
        for prop in STATIC_SOIL_PROPERTY_NAMES:
            del inputs[prop]
        inputs["packed_soil_properties"] = packed_soil

    return inputs, expected


def copy_mutable_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Create fresh copies of mutable in-out arrays for land_surface_model.

    Args:
        inputs: Dictionary of input arrays and parameters.

    Returns:
        Deep-copied dictionary with mutable NumPy arrays copied.
    """
    # Several arrays are modified in-place by land_surface_model:
    # water_content_m, soil_enthalpy_J_per_m2, wetting_front_*, snow_water_equivalent_m, etc.
    copied = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            copied[k] = v.copy()
        else:
            copied[k] = v
    return copied


def verify_correctness(
    actual_outputs: tuple[Any, ...], expected_dict: dict[str, Any]
) -> bool:
    """Verify that actual outputs match expected baseline outputs.

    Args:
        actual_outputs: Tuple of 32 output arrays returned by land_surface_model.
        expected_dict: Baseline outputs loaded from reference npz archive.

    Returns:
        True if all output arrays match within specified tolerance, False otherwise.
    """
    all_ok = True
    print("\n--- Correctness Check ---")
    for name, actual in zip(OUTPUT_NAMES, actual_outputs):
        if name not in expected_dict:
            print(f"Warning: {name} not in expected outputs")
            continue
        expected = expected_dict[name]
        abs_diff = float(np.max(np.abs(actual - expected)))
        max_exp = float(np.max(np.abs(expected)))
        rel_diff = abs_diff / (max_exp + 1e-12)

        # Enthalpy variables are in Joules/m2 (orders of magnitude up to 1e7)
        if "enthalpy" in name:
            atol = 500.0
            rtol = 1e-4
        else:
            atol = 2e-4
            rtol = 1e-4

        match = np.allclose(actual, expected, rtol=rtol, atol=atol)
        if not match:
            print(
                f"FAIL: {name} max abs diff: {abs_diff:.3e}, max rel diff: {rel_diff:.3e}"
            )
            all_ok = False
        else:
            if abs_diff > 1e-7:
                print(
                    f"  OK (within tolerance): {name} max abs diff: {abs_diff:.3e}, rel diff: {rel_diff:.3e}"
                )
    if all_ok:
        print(">>> All 32 output arrays MATCH baseline within tolerance! <<<")
    else:
        print(">>> CORRECTNESS FAILURE DETECTED! <<<")
    return all_ok


def run_benchmark(num_runs: int = 5) -> None:
    """Run warm-up, correctness verification, and timed benchmark iterations."""
    inputs, expected = load_benchmark_data()
    num_cells = inputs["slope_m_per_m"].shape[0]
    print(
        f"Benchmark problem size: {num_cells:,} HRU cells, 6 soil layers, 24 hourly steps"
    )

    # Warm-up run (compiles Numba kernel on first invocation)
    print("\nExecuting warm-up run (including JIT compilation)...")
    warmup_inputs = copy_mutable_inputs(inputs)
    t0 = time.perf_counter()
    warmup_outputs = land_surface_model(**warmup_inputs)
    t_warmup = time.perf_counter() - t0
    print(f"Warm-up finished in {t_warmup:.3f} s")

    # Correctness verification
    verify_correctness(warmup_outputs, expected)

    # Benchmark runs
    print(f"\nRunning {num_runs} benchmark iterations...")
    runtimes: list[float] = []

    for run_idx in range(num_runs):
        gc.collect()
        rss_before = get_peak_rss_mb()
        run_inputs = copy_mutable_inputs(inputs)

        t_start = time.perf_counter()
        outputs = land_surface_model(**run_inputs)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        runtimes.append(elapsed)
        rss_after = get_peak_rss_mb()
        print(
            f"  Run {run_idx + 1}/{num_runs}: {elapsed:.4f} s (Peak RSS: {rss_after:.1f} MB)"
        )

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Cells:    {num_cells:,}")
    print(f"Runs:     {num_runs}")
    print(f"Min:      {np.min(runtimes):.4f} s")
    print(f"Median:   {np.median(runtimes):.4f} s")
    print(f"Mean:     {np.mean(runtimes):.4f} s +/- {np.std(runtimes):.4f} s")
    print(f"Max:      {np.max(runtimes):.4f} s")
    print(f"Peak RSS: {get_peak_rss_mb():.1f} MB")
    print("====================================================\n")


if __name__ == "__main__":
    run_benchmark()
