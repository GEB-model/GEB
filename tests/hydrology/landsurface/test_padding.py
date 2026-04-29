"""Tests for HRU array padding in the land surface model.

Two checks are performed:
1. Identity: outputs with SIMD padding are bit-for-bit identical to the
   unpadded baseline for all original cells.
2. Speed: wall-clock time for the padded run vs the unpadded run is printed
   (float32 only; no threshold assertion, purely diagnostic).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from geb.hydrology import landsurface as landsurface_module
from geb.hydrology.landcovers import FOREST, GRASSLAND_LIKE
from geb.hydrology.landsurface.landsurface_model import (
    LandSurfaceInputs,
    _pad_hru_arrays,
    land_surface_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ERROR_CASES_DIR = Path(__file__).parent / "landsurface_error_cases"


def _get_any_error_case() -> Path | None:
    """Return the first available error-case file, or None."""
    if not _ERROR_CASES_DIR.exists():
        return None
    files = sorted(_ERROR_CASES_DIR.glob("*.npz"))
    return files[0] if files else None


def _load_and_tile(npz_path: Path, num_cells: int) -> dict:
    """Load a single-cell error case and tile it to ``num_cells``.

    Handles both the old npz format (with ``crop_group_forest``,
    ``crop_group_grassland_like``, and ``crop_group_number_per_group`` as
    separate fields) and the current format (with a per-HRU
    ``crop_group_number``).

    Args:
        npz_path: Path to the ``.npz`` file produced by error diagnostics.
        num_cells: Target number of HRU cells to produce.

    Returns:
        Dictionary of tiled arrays ready to be unpacked into
        `land_surface_model`.
    """
    with np.load(npz_path) as data:
        raw: dict = {k: data[k] for k in data.files}

    # Old error-case files used layer-major layout (N_LAYERS, num_cells).
    # Detect by shape (N, 1) with N > 1 and transpose to (1, N).
    for key, val in raw.items():
        if (
            isinstance(val, np.ndarray)
            and val.ndim == 2
            and val.shape[1] == 1
            and val.shape[0] > 1
        ):
            raw[key] = np.ascontiguousarray(val.T)

    # snow arrays must always be float64 for the kernel
    raw["snow_water_equivalent_m"] = raw["snow_water_equivalent_m"].astype(np.float64)
    raw["liquid_water_in_snow_m"] = raw["liquid_water_in_snow_m"].astype(np.float64)

    # Old format: three separate crop-group fields; convert to per-HRU array so
    # all inputs are either scalars or HRU-dimensioned (matching current API).
    if "crop_group_number_per_group" in raw:
        crop_group_number_per_group = raw.pop("crop_group_number_per_group")
        crop_group_forest = raw.pop("crop_group_forest")
        crop_group_grassland_like = raw.pop("crop_group_grassland_like")
        land_use_type = raw["land_use_type"]
        # When all cells are natural land cover the lookup table may be empty;
        # use a zeros placeholder since the else-branch of np.where is masked out.
        if len(crop_group_number_per_group) == 0:
            crop_from_table = np.zeros_like(crop_group_forest)
        else:
            crop_map = np.clip(raw["crop_map"], 0, len(crop_group_number_per_group) - 1)
            crop_from_table = crop_group_number_per_group[crop_map]
        raw["crop_group_number"] = np.where(
            land_use_type == FOREST,
            crop_group_forest,
            np.where(
                land_use_type == GRASSLAND_LIKE,
                crop_group_grassland_like,
                crop_from_table,
            ),
        ).astype(np.float32)

    # Ensure all float arrays (except snow) are float32 to match kernel dtype
    # requirements. Also convert 0-D arrays to numpy scalars so Numba sees the
    # correct type rather than a 0-dimensional array.
    _FLOAT64_KEYS = {"snow_water_equivalent_m", "liquid_water_in_snow_m"}
    for key, val in raw.items():
        if key in _FLOAT64_KEYS:
            continue
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating):
            if val.ndim == 0:
                raw[key] = np.float32(val)
            else:
                raw[key] = val.astype(np.float32)

    single_cell_count: int = raw["slope_m_per_m"].shape[0]

    tiled: dict = {}
    for key, val in raw.items():
        if (
            isinstance(val, np.ndarray)
            and val.ndim >= 1
            and val.shape[0] == single_cell_count
        ):
            # Tile along the cell axis (axis 0) to reach num_cells.
            if val.ndim == 1:
                tiled[key] = np.ascontiguousarray(np.tile(val, num_cells))
            else:
                tiled[key] = np.ascontiguousarray(np.tile(val, (num_cells, 1)))
        else:
            tiled[key] = val

    return tiled


def _deep_copy_inputs(inputs: dict) -> dict:
    """Return a dict with independent copies of all mutable arrays.

    Args:
        inputs: Input dictionary as returned by `_load_and_tile`.

    Returns:
        Deep copy with every ndarray replaced by a contiguous copy.
    """
    return {
        k: (np.ascontiguousarray(v.copy()) if isinstance(v, np.ndarray) else v)
        for k, v in inputs.items()
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def error_case_path() -> Path:
    """Pytest fixture providing an error-case path, or skipping if none.

    Returns:
        Path to the first available error-case ``.npz`` file.
    """
    path = _get_any_error_case()
    if path is None:
        pytest.skip("No landsurface error-case files found; skipping padding tests.")
    return path


def test_padding_identity(error_case_path: Path) -> None:
    """Outputs with SIMD padding are bit-identical to unpadded results.

    Uses float32 throughout (the common production dtype).  Constructs a
    non-multiple-of-16 cell count to guarantee that padding actually fires.
    """
    landsurface_module.N_SOIL_LAYERS = 6  # type: ignore[attr-defined]

    # 5 cells is deliberately not a multiple of 16 so _pad_hru_arrays pads.
    num_cells = 5
    inputs_unpadded = _load_and_tile(error_case_path, num_cells)
    inputs_padded_base = _deep_copy_inputs(inputs_unpadded)

    # Build a LandSurfaceInputs so we can call _pad_hru_arrays.
    padded_nt = _pad_hru_arrays(LandSurfaceInputs(**inputs_padded_base))
    assert padded_nt.slope_m_per_m.shape[0] % 16 == 0, (
        "_pad_hru_arrays did not pad to a multiple of 16"
    )
    assert padded_nt.slope_m_per_m.shape[0] > num_cells, (
        "_pad_hru_arrays should have increased the cell count"
    )

    # Run unpadded
    results_unpadded = land_surface_model(**inputs_unpadded)

    # Run padded (a fresh independent copy so in-place mutations don't alias)
    padded_inputs_for_run = _deep_copy_inputs(inputs_padded_base)
    padded_nt_run = _pad_hru_arrays(LandSurfaceInputs(**padded_inputs_for_run))
    results_padded_raw = land_surface_model(**padded_nt_run._asdict())

    # Trim padded outputs to the original cell count for comparison.
    results_padded = [
        r[:num_cells]
        if isinstance(r, np.ndarray) and r.ndim == 1
        else r[:num_cells, :]
        if isinstance(r, np.ndarray) and r.ndim == 2
        else r
        for r in results_padded_raw
    ]

    for idx, (unpad, pad) in enumerate(zip(results_unpadded, results_padded)):
        if not isinstance(unpad, np.ndarray):
            continue
        assert np.array_equal(unpad, pad), (
            f"Output {idx} differs between padded and unpadded runs.\n"
            f"  max abs diff = {np.abs(unpad.astype(np.float64) - pad.astype(np.float64)).max()}"
        )

    # Also verify that the in-place-modified input arrays are identical.
    for field in LandSurfaceInputs._fields:
        val_unpad = inputs_unpadded[field]
        val_pad_trimmed = getattr(padded_nt_run, field)
        if not isinstance(val_unpad, np.ndarray):
            continue
        if val_unpad.shape[0] != num_cells:
            # Not a per-cell array (e.g. crop_group_number_per_group).
            continue
        val_pad_trimmed = (
            val_pad_trimmed[:num_cells]
            if val_pad_trimmed.ndim == 1
            else val_pad_trimmed[:num_cells, :]
        )
        assert np.array_equal(val_unpad, val_pad_trimmed), (
            f"In-place input '{field}' differs between padded and unpadded runs."
        )


def test_padding_speed(error_case_path: Path) -> None:
    """Print wall-clock timing for padded vs unpadded runs (float32 only).

    No timing threshold is asserted; this test always passes.  It is intended
    to be run manually or in CI to detect regressions in throughput.

    The function is JIT-compiled by Numba, so we warm up the cache with a
    small call before benchmarking.
    """
    landsurface_module.N_SOIL_LAYERS = 6  # type: ignore[attr-defined]

    # Warm up Numba JIT cache with a single cell before timing.
    warmup_inputs = _load_and_tile(error_case_path, 1)
    land_surface_model(**warmup_inputs)

    # Use 1009 cells (prime, not a multiple of 16) for a realistic benchmark.
    num_cells = 1009
    inputs_unpadded = _load_and_tile(error_case_path, num_cells)

    # --- Unpadded run ---
    inputs_run = _deep_copy_inputs(inputs_unpadded)
    t_start = time.perf_counter()
    land_surface_model(**inputs_run)
    t_unpadded = time.perf_counter() - t_start

    # --- Padded run ---
    inputs_run_padded_base = _deep_copy_inputs(inputs_unpadded)
    padded_nt = _pad_hru_arrays(LandSurfaceInputs(**inputs_run_padded_base))
    # Make an independent copy so the JIT sees the same memory layout.
    padded_inputs_run = _deep_copy_inputs(dict(padded_nt._asdict()))
    t_start = time.perf_counter()
    land_surface_model(**padded_inputs_run)
    t_padded = time.perf_counter() - t_start

    num_padded_cells = padded_nt.slope_m_per_m.shape[0]
    print(
        f"\n[padding speed] num_cells={num_cells}, padded_to={num_padded_cells}\n"
        f"  unpadded : {t_unpadded * 1000:.1f} ms\n"
        f"  padded   : {t_padded * 1000:.1f} ms\n"
        f"  overhead : {(t_padded - t_unpadded) / t_unpadded * 100:+.1f}%"
    )
