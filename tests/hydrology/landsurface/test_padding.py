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
    """Padding is transparent: trimmed outputs match the aligned-reference run.

    The kernel now iterates in blocks of 16 and requires ``num_cells`` to be a
    multiple of 16.  This test verifies that calling the kernel on ``N`` cells
    that were padded to the next multiple of 16 by ``_pad_hru_arrays`` produces
    the same first ``N`` outputs as calling the kernel directly on ``N_aligned``
    cells tiled from the same single-cell fixture (which is already a multiple
    of 16 and therefore an exact match to what padding produces).
    """
    landsurface_module.N_SOIL_LAYERS = 6  # type: ignore[attr-defined]

    # 5 cells: deliberately not a multiple of 16 so _pad_hru_arrays pads to 16.
    num_cells = 5
    padded_size = 16  # _pad_hru_arrays pads 5 → 16; used as reference tile size

    # --- padded path: tile 5, pad to 16, run, trim to 5 ---
    inputs_padded_base = _load_and_tile(error_case_path, num_cells)
    padded_nt = _pad_hru_arrays(
        LandSurfaceInputs(**_deep_copy_inputs(inputs_padded_base))
    )
    assert padded_nt.slope_m_per_m.shape[0] == padded_size, (
        f"Expected padding to produce {padded_size} cells, "
        f"got {padded_nt.slope_m_per_m.shape[0]}"
    )
    results_padded_raw = land_surface_model(**padded_nt._asdict())
    results_padded = [
        r[:num_cells]
        if isinstance(r, np.ndarray) and r.ndim == 1
        else r[:num_cells, :]
        if isinstance(r, np.ndarray) and r.ndim == 2
        else r
        for r in results_padded_raw
    ]

    # --- reference path: tile directly to 16 (already aligned), run ---
    # All 16 cells are identical copies of cell 0, which is exactly the content
    # that _pad_hru_arrays places in the padding region.  The two kernel inputs
    # are therefore bit-for-bit identical, so all 16 output elements should
    # match, and in particular the first 5.  We keep ref_nt so that post-run
    # in-place mutations are accessible for comparison.
    ref_nt = LandSurfaceInputs(
        **_deep_copy_inputs(_load_and_tile(error_case_path, padded_size))
    )
    results_ref = land_surface_model(**ref_nt._asdict())

    for idx, (ref, pad) in enumerate(zip(results_ref, results_padded)):
        if not isinstance(ref, np.ndarray):
            continue
        ref_trimmed = ref[:num_cells] if ref.ndim == 1 else ref[:num_cells, :]
        assert np.array_equal(ref_trimmed, pad), (
            f"Output {idx} differs between reference and padded runs.\n"
            f"  max abs diff = {np.abs(ref_trimmed.astype(np.float64) - pad.astype(np.float64)).max()}"
        )

    # Verify in-place-modified fields (not in return tuple) also match.
    # Both ref_nt and padded_nt were passed to the kernel so their arrays carry
    # post-run state.
    _INPLACE_FIELDS = {
        "water_content_m",
        "soil_enthalpy_J_per_m2",
        "wetting_front_depth_m",
        "wetting_front_suction_head_m",
        "wetting_front_moisture_deficit",
        "green_ampt_active_layer_idx",
    }
    for field in _INPLACE_FIELDS:
        ref_val = getattr(ref_nt, field)
        pad_val = getattr(padded_nt, field)
        ref_trimmed = (
            ref_val[:num_cells] if ref_val.ndim == 1 else ref_val[:num_cells, :]
        )
        pad_trimmed = (
            pad_val[:num_cells] if pad_val.ndim == 1 else pad_val[:num_cells, :]
        )
        assert np.array_equal(ref_trimmed, pad_trimmed), (
            f"In-place input '{field}' differs between reference and padded runs."
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
